"""R9 sample-exposure-scaled FP32 EMA readout. Never participates in the online optimizer."""
import copy
import hashlib
import torch
from .finite import all_floating_finite

RULE={'version':'ema-warmcopy474-decay0997002999-v1','warmcopy_updates':474,'decay':.999**3,
      'dtype':'torch.float32','floating_buffers':'same_as_parameters','integer_buffers':'copy'}
R8_RULE={**RULE,'version':'ema-warmcopy1422-decay0999-v1','warmcopy_updates':1422,'decay':.999}
R11_RULE={**R8_RULE,'version':'ema-warmcopy1066-decay0999-v1','warmcopy_updates':1066}
FROZEN_BN_RULE={'version':'ema-warmcopy1422-decay0999-frozenbn-v1','warmcopy_updates':1422,'decay':.999,
                'dtype':'torch.float32','floating_buffers':'same_as_parameters_except_frozen_bn',
                'integer_buffers':'copy','frozen_bn_buffers':'copy'}

def validate_rule(rule):
 if rule not in (RULE,R8_RULE,R11_RULE,FROZEN_BN_RULE):
  from .experiment91 import ema_rule
  if rule!=ema_rule():raise ValueError('EMA rule is not an explicitly supported frozen rule')
 return rule

def tensor_digest(state):
 digest=hashlib.sha256()
 for name,tensor in sorted(state.items()):
  value=tensor.detach().cpu().contiguous()
  digest.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode()+value.numpy().tobytes())
 return digest.hexdigest()

def validate_state(state,reference,successful,rule=None):
 rule=validate_rule(rule if rule is not None else state.get('rule'))
 warm=rule['warmcopy_updates']
 if (not isinstance(state,dict) or set(state)!={'rule','model','update_count','copy_count','blend_count','last_update_success'}
     or state['rule']!=rule or type(successful)!=int or successful<0
     or any(type(state[k])!=int for k in ('update_count','copy_count','blend_count','last_update_success'))
     or (state['update_count'],state['copy_count'],state['blend_count'],state['last_update_success'])!=
       (successful,min(successful,warm),max(0,successful-warm),successful)):
  raise ValueError('EMA rule / successful count missing or incompatible')
 tensors=state['model']
 if not isinstance(tensors,dict) or tensors.keys()!=reference.keys():raise ValueError('EMA tensor keys differ')
 if rule.get('batch_norm_buffers')==0 and any(n.endswith(('.running_mean','.running_var','.num_batches_tracked')) for n in reference):raise ValueError('Tiny noBN EMA state contains BN buffers')
 fixed={name for name in reference if name.endswith(('.running_mean','.running_var','.num_batches_tracked'))} if rule.get('frozen_bn_buffers')=='copy' else set()
 if rule.get('frozen_bn_buffers')=='copy' and len(fixed)!=159:raise ValueError('EMA fixed BN buffers missing')
 if rule.get('static_integer_buffers')=='relative_position_index':
  from .swin_model import static_index_buffers
  indices=static_index_buffers(reference)
  if len(indices)!=rule['static_integer_buffer_count'] or any(
      not torch.equal(tensors[name], value) for name,value in indices.items()):
   raise ValueError('EMA static Swin relative-position index differs')
 if rule.get('static_swin_buffers'):
  from .swin_model import static_swin_buffers
  fixed_swin=static_swin_buffers(reference,rule['static_swin_buffers'])
  if len(fixed_swin)!=rule['static_swin_buffer_count'] or any(not torch.equal(tensors[n],t) for n,t in fixed_swin.items()):
   raise ValueError('EMA static Swin coordinates/index differs')
 for name,value in tensors.items():
  ref=reference[name]
  if (not isinstance(value,torch.Tensor) or value.shape!=ref.shape or value.dtype!=ref.dtype
      or (value.is_floating_point() and value.dtype!=torch.float32)):
   raise ValueError('EMA shape / FP32 dtype differs '+name)
  if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise FloatingPointError('nonfinite EMA '+name)
  if name in fixed and not torch.equal(value,ref):raise ValueError('EMA frozen BN buffer differs '+name)
 return state

class EMA:
 def __init__(self,online,rule,*,state=None,successful=0):
  self.frozen_parameter_names=tuple(sorted(name for name,parameter in online.named_parameters()
                                          if not parameter.requires_grad))
  self.rule=copy.deepcopy(validate_rule(rule))
  from .persistence import canonical_digest
  self.age_metadata_sha256=canonical_digest(getattr(online,'age_metadata',None))
  self.model=copy.deepcopy(online).float().eval().requires_grad_(False)
  self.successful=0
  if state is not None:
   validate_state(state,self.model.state_dict(),successful,self.rule)
   self.model.load_state_dict(state['model'],strict=True);self.successful=successful
  elif successful:raise ValueError('EMA resume cannot rebuild history from online')
  self.assert_finite()
 @torch.no_grad()
 def update(self,online,successful,*,skipped=False):
  from .persistence import canonical_digest
  if (canonical_digest(getattr(online,'age_metadata',None))!=self.age_metadata_sha256
      or canonical_digest(getattr(self.model,'age_metadata',None))!=self.age_metadata_sha256):
   raise ValueError('EMA immutable age metadata changed')
  if skipped:
   if successful!=self.successful:raise ValueError('EMA skip changed successful cursor')
   return
  if successful!=self.successful+1:raise ValueError('EMA updates must match each successful online step')
  source=online.state_dict();target=self.model.state_dict()
  if source.keys()!=target.keys():raise ValueError('EMA live keys differ')
  fixed=set(self.frozen_parameter_names)
  if self.rule.get('static_swin_buffers'):
   from .swin_model import static_swin_buffers
   source_fixed=static_swin_buffers(online,self.rule['static_swin_buffers']);target_fixed=static_swin_buffers(self.model,self.rule['static_swin_buffers'])
   if (set(source_fixed)!=set(dict(online.named_buffers())) or set(target_fixed)!=set(dict(self.model.named_buffers()))
       or len(source_fixed)!=self.rule['static_swin_buffer_count']
       or any(not torch.equal(t,target_fixed[n]) for n,t in source_fixed.items())):
    raise ValueError('Swin static EMA buffers changed')
   fixed.update(source_fixed)
  elif self.rule.get('static_integer_buffers')=='relative_position_index':
   from .swin_model import static_index_buffers
   source_indices=static_index_buffers(online);target_indices=static_index_buffers(self.model)
   if (set(source_indices)!=set(dict(online.named_buffers()))
       or set(target_indices)!=set(dict(self.model.named_buffers()))
       or any(not torch.equal(value,target_indices[name]) for name,value in source_indices.items())):
    raise ValueError('Swin static EMA indices changed')
   fixed.update(source_indices)
  elif self.rule.get('batch_norm_buffers')==0 and (dict(online.named_buffers()) or dict(self.model.named_buffers())):raise ValueError('Tiny noBN EMA live buffers differ')
  if self.rule.get('frozen_bn_buffers')=='copy':
   from .imagenet_resnet import ImageNetResnetBinary
   if isinstance(online,ImageNetResnetBinary):
    from .imagenet_resnet import frozen_bn_names
   else:
    from .flair_model import frozen_bn_names
   fixed.update(frozen_bn_names(online))
  for name,value in target.items():
   current=source[name]
   if current.shape!=value.shape or current.dtype!=value.dtype:raise ValueError('EMA live dtype / shape differs')
   if name in fixed or successful<=self.rule['warmcopy_updates'] or not value.is_floating_point():value.copy_(current)
   else:value.mul_(self.rule['decay']).add_(current,alpha=1-self.rule['decay'])
  self.successful=successful;self.assert_finite()
 def assert_finite(self):
  if not all_floating_finite(self.model.state_dict().values()):raise FloatingPointError('nonfinite EMA parameters or buffers')
 def state_dict(self):
  s=self.successful
  return {'rule':copy.deepcopy(self.rule),'model':self.model.state_dict(),'update_count':s,'copy_count':min(s,self.rule['warmcopy_updates']),
          'blend_count':max(0,s-self.rule['warmcopy_updates']),'last_update_success':s}
 def rank_proof(self,ctx):
  record={k:v for k,v in self.state_dict().items() if k!='model'}
  record['tensor_sha256']=tensor_digest(self.model.state_dict())
  ranks=ctx.values({'rank':ctx.rank,**record})
  if any({k:v for k,v in r.items() if k!='rank'}!=record for r in ranks):raise ValueError('EMA rank tensors / counters differ')
  return ranks

def select_variant(payload,variant,*,for_training=False):
 if variant not in ('online','ema'):raise ValueError('explicit online or ema variant required')
 if variant=='ema':
  if for_training:raise ValueError('EMA readout cannot pair with online Adam for training')
  if not payload.get('ema') or 'successful_updates' not in payload:raise ValueError('EMA payload absent')
  state=validate_state(payload['ema'],payload['model'],payload['successful_updates'])
  return state['model']
 return payload['model']

def evaluate_pair(ctx,evaluate,*,all_ranks=False):
 result={}
 for variant in ('online','ema'):
  metrics=evaluate(variant) if (all_ranks or ctx.primary) else None
  result[variant]=ctx.broadcast(metrics) # one collective between full Dev evaluations
  if result[variant].get('auroc') is None:raise ValueError('paired Dev AUROC undefined')
 return result

def commit_pair(state,pair,cycle,successful,job):
 if cycle!=state['evaluated_cycle']+1:raise ValueError('paired evaluation commit skipped or repeated a cycle')
 out=copy.deepcopy(state);improved={}
 for variant,prefix in (('online',''),('ema','ema_')):
  metric=pair[variant];key=prefix+'best_metrics';old=state[key]
  if job.get('experiment91') is not None or job.get('model') in ('flair_resnet50','resnet50_imagenet'):
   from .n05_metrics import improved as exact_improved
   improved[variant]=exact_improved({**metric,'successful_updates':successful},old)
  else:
   improved[variant]=(old is None or metric['auroc']>old['auroc'] or (metric['auroc']==old['auroc'] and metric['ap']>old['ap']))
  out[prefix+'last_metrics']=copy.deepcopy(metric)
  if improved[variant]:out[key]={**metric,'cycle':cycle,'successful_updates':successful,'variant':variant}
 early=out['early_stop'];reference=early['reference_auroc']
 if job.get('experiment91'):
  from .experiment91 import early_stopping
  policy=early_stopping()
  score=max(pair[variant]['auroc'] for variant in policy['variants'])
  delta=policy['min_delta'];patience=policy['patience']
  early.update(metric=policy['metric'],variants=list(policy['variants']))
 else:score=pair['online']['auroc'];delta=job['min_delta'];patience=job['patience']
 if reference is None or score>reference+delta:early.update(reference_auroc=score,bad_cycles=0)
 else:early['bad_cycles']+=1
 if cycle>=job['min_cycles'] and early['bad_cycles']>=patience:early['stopped']=True
 out['evaluated_cycle']=cycle
 return out,improved

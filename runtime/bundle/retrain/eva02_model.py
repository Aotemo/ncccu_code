"""Native public EVA02 Base, fixed full 512 input and original mean readout."""
import importlib,inspect,json
from pathlib import Path
import torch
from torch import nn

MODEL='eva02_base'
VARIANT='eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
FORMAT='timm-eva02-base-native448-full1000-pad512-state-dict-v1'
REVISION='81063ecfe9c381a16a19d06f396d6c7011aa426a'
WEIGHT_SHA='dbbfef826976527b9060875233f4cbeb61f9b14b8352e4116e2769c38593f054'
LIBRARY={'version':'1.0.27','source_sha256':{
    'models/eva.py':'23314ef536d7ce9cc3737e54f424841b46f426e93af651c357b4a75a8b00a12d',
    'layers/patch_embed.py':'7b7e670f8e6a2828752a65297025fb903fc2fbe0298617a3cd29d804be1455aa',
    'layers/mlp.py':'c024128a1c7e2a55ebc5e33a7c94c585ae1e960cd032b39b08445ea30f05c5a5',
    'layers/pos_embed_sincos.py':'8d1751ada7f98bebb81f973567f350b85a110b42c548e036ace40d05f32cb914'}}

def library_identity():
    import timm
    from .models import _hash
    value={'version':timm.__version__,'source_sha256':{name:_hash(inspect.getfile(importlib.import_module(
        'timm.'+name[:-3].replace('/','.'))))for name in LIBRARY['source_sha256']}}
    if value!=LIBRARY:raise ValueError('native EVA02 implementation differs')
    return value

def validate_initialization(initialization):
    from .models import _hash
    from .experiment91 import public_source
    source=public_source();path=Path(initialization['path']);meta_path=Path(initialization['provenance'])
    if (initialization.get('kind')!='imagenet' or initialization.get('format')!=FORMAT
            or (source.get('model'),source.get('format'),source.get('revision'),source.get('sha256'),source.get('original_sha256'))
                !=(VARIANT,FORMAT,REVISION,WEIGHT_SHA,WEIGHT_SHA)
            or source.get('native_library')!=LIBRARY or initialization.get('sha256')!=WEIGHT_SHA
            or any(not p.is_absolute()or p.is_symlink()for p in(path,meta_path))
            or _hash(path)!=WEIGHT_SHA or _hash(meta_path)!=source['provenance_sha256']):
        raise ValueError('fixed natural-image EVA02 public identity differs')
    meta=json.loads(meta_path.read_text())
    if (any(meta.get(k)!=source[k]for k in('model','format','revision','source_url','training_data','sha256','original_sha256','native_library'))
            or meta.get('redistribution_confirmed')is not True or meta.get('declared_license')!='MIT'
            or meta.get('license_sha256')!=source['license_sha256']
            or _hash(meta_path.parent/meta['license_file'])!=source['license_sha256']):
        raise ValueError('native EVA02 provenance/license differs')
    return {'kind':'imagenet','model':MODEL,'format':FORMAT,'weights_sha256':WEIGHT_SHA,
        'provenance_sha256':source['provenance_sha256'],'final_norm_eps':1e-6,'internal_norm_eps':1e-6,
        'source_url':source['source_url'],'training_data':source['training_data'],'source_commit':REVISION,
        'original_sha256':WEIGHT_SHA,'license_sha256':source['license_sha256'],'native_library':LIBRARY}

class EVA02Binary(nn.Module):
    def __init__(self,network,age_spec):
        super().__init__();self.network=network
        self.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(768,1))
        from .age import attach_age
        attach_age(self,age_spec)

    def forward(self,images,age_features=None):
        if images.ndim!=4 or tuple(images.shape[1:])!=(3,512,512)or images.dtype!=torch.float32:
            raise ValueError('native EVA02 requires FP32 Bx3x512x512')
        from .age import residual
        return residual(self,self.classifier(self.network(images)).flatten(),age_features)

def describe(model):
    network=model.network;patch=network.patch_embed;rope=network.rope
    norms=[{'name':n,'eps':m.eps,'shape':list(m.normalized_shape)}for n,m in network.named_modules()if isinstance(m,nn.LayerNorm)]
    rates=[float(x)for x in torch.linspace(0,.1,12)]
    actual=[float(getattr(getattr(b,f'drop_path{j}'),'drop_prob',0.))for b in network.blocks for j in(1,2)]
    buffers=dict(model.named_buffers())
    if (not isinstance(model,EVA02Binary)or network.global_pool!='avg' or network.num_prefix_tokens!=1
            or network.no_embed_class or network.dynamic_img_size or network.grad_checkpointing
            or tuple(network.pos_embed.shape)!=(1,1370,768)or tuple(network.cls_token.shape)!=(1,1,768)
            or tuple(patch.img_size)!=(518,518)or tuple(patch.grid_size)!=(37,37)or tuple(patch.patch_size)!=(14,14)
            or patch.strict_img_size or not patch.dynamic_img_pad or not patch.flatten
            or not isinstance(network.head,nn.Identity)or not isinstance(network.norm,nn.Identity)
            or not isinstance(network.fc_norm,nn.LayerNorm)or network.head_drop.p!=0.
            or tuple(rope.pos_embed.shape)!=(1369,128)or rope.pos_embed.dtype!=torch.float32
            or tuple(rope.ref_feat_shape)!=(16,16)or set(buffers)!={'network.rope.pos_embed'}
            or 'network.rope.pos_embed'in model.state_dict()or not bool(torch.isfinite(rope.pos_embed).all())
            or len(network.blocks)!=12 or len(norms)!=37 or any(n['eps']!=1e-6 for n in norms)
            or any(type(b.mlp).__name__!='SwiGLU'or not isinstance(b.mlp.norm,nn.LayerNorm)for b in network.blocks)
            or actual!=[r for r in rates for _ in(1,2)]
            or len(dict(model.named_parameters()))!=237 or sum(p.numel()for p in model.parameters())!=86614275
            or model.classifier[0].p!=.2 or (model.classifier[-1].in_features,model.classifier[-1].out_features)!=(768,1)):
        raise ValueError('native EVA02 padding/position/RoPE/mean/SwiGLU/binary graph differs')
    from .ema import tensor_digest
    return {'version':'eva02-base-native-512-pad6-grid37-mean-v1','native_variant':VARIANT,'native_library':LIBRARY,
        'image_size':[512,512],'internal_padded_size':[518,518],'right_bottom_zero_padding':[6,6],
        'patch_size':[14,14],'patch_grid':[37,37],'patch_tokens':1369,'prefix_tokens':1,'position_shape':[1,1370,768],
        'public_pool':'avg','final_norm':'Identity','fc_norm':'LayerNorm','MLP':'SwiGLU_with_LayerNorm',
        'normalization_catalog':norms,'native_head_dropout':0.,'binary_dropout':.2,
        'live_buffer_count':1,'persistent_buffer_count':0,'rope_buffer_sha256':tensor_digest(buffers),
        'parameter_tensor_count':237,'parameter_count':86614275,'age_readouts':1}

def create_model(job,*,initialize=True):
    fixed={'model':MODEL,'image_size':512,'precision':'fp32','normalization':'clip','dropout':.2,
        'stochastic_depth':.1,'head_lr_scope':'last_linear','final_norm_eps':1e-6}
    if any(job.get(k)!=v for k,v in fixed.items())or not job.get('age',{}).get('enabled')or job.get('frozen_bn'):
        raise ValueError('native EVA02 fixed model contract differs')
    import timm
    library_identity();network=timm.create_model(VARIANT,pretrained=False,drop_rate=0.,drop_path_rate=.1)
    public=None
    if initialize:
        proof=validate_initialization(job['initialization']);state=torch.load(job['initialization']['path'],map_location='cpu',weights_only=True)
        expected=network.state_dict()
        if (not isinstance(state,dict)or len(state)!=236 or state.keys()!=expected.keys()
                or any(not torch.is_tensor(t)or t.shape!=expected[n].shape or t.dtype!=torch.float32
                    or not bool(torch.isfinite(t).all())for n,t in state.items())):
            raise ValueError('complete public EVA02 full1000 state differs')
        network.load_state_dict(state,strict=True)
        if any(not torch.equal(t,network.state_dict()[n])for n,t in state.items()):raise ValueError('public EVA02 strict load is not exact')
        from .ema import tensor_digest
        public={'initialization':proof,'strict_missing_keys':[],'strict_unexpected_keys':[],
            'public_tensor_count':236,'public_tensor_sha256':tensor_digest(state),'full_head_shape':[1000,768],
            'backbone_only':False,'all_public_tensors_exact':True,'all_public_tensors_FP32_finite':True}
        del state
    network.set_input_size(518)
    network.patch_embed.strict_img_size=False;network.patch_embed.dynamic_img_pad=True
    network.head=nn.Identity()
    result=EVA02Binary(network,job['age']);describe(result);result.public_initialization_receipt=public
    return result

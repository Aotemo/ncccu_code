"""Audited continuation and rank-coordinated training; legacy engine stays unchanged."""
import contextlib
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import ImageDataset, flatten_training_views
from .classification_loss import PAIRWISE_LOSS_KINDS, classification_loss, gather_pairwise_inputs
from .parallel_evaluation import evaluate as parallel_evaluate, recovery_checkpoint_due
from .distributed import RankBatchSampler,ReferenceBatchNorm
from . import engine as single
from .persistence import (CheckpointStore,RunLock,atomic_json,canonical_digest,
                          capture_rng,restore_rng,load_file,sha256_file,CheckpointError)
from .config import job_contract
from .finite import all_floating_finite


def validate_parent(parent,spec,identity,job,plan):
    if (parent.get('identity')!=spec.get('identity') or parent.get('mode')!='train'
            or identity.get('phase_parent_sha256')!=spec.get('sha256')
            or parent.get('manifest_digest')!=identity.get('manifest_digest')
            or parent['identity'].get('manifest_digest')!=identity.get('manifest_digest')
            or parent.get('successful_updates',0)<=0 or not parent.get('optimizer',{}).get('state')
            or not parent.get('model') or not {'python','numpy','torch','cuda'}<=set(parent.get('rng',{}))
            or job_contract(parent['job'])!=parent['identity'].get('job')):
        raise ValueError('parent continuation identity / mode / checkpoint state rejected')
    old=copy.deepcopy(parent['identity']['job']);new=copy.deepcopy(identity['job'])
    if old.get('physical_batch')!=plan['reference_physical_batch'] or new.get('physical_batch')!=plan['per_rank_batch']:
        raise ValueError('parent/local physical batch differs from derived topology')
    for key in ('physical_batch','stop_after_cycles'):
        old.pop(key,None);new.pop(key,None)
    if old!=new or job_contract(job)!=identity['job']:
        raise ValueError('parent continuation changes scientific configuration')
    schedule=parent.get('schedule',{});steps=schedule.get('steps_per_cycle',0)
    if (steps<=0 or schedule.get('total_updates')!=job['cycles']*steps
            or schedule.get('warmup_updates')!=job['warmup_cycles']*steps
            or (job['precision']=='fp16' and not parent.get('scaler'))):
        raise ValueError('parent continuation changes learning-rate budget or AMP state')
    return True


def load_parent(spec,identity,job,plan):
    if sha256_file(spec['checkpoint'])!=spec['sha256']:raise ValueError('parent SHA mismatch')
    payload=load_file(spec['checkpoint'],map_location='cpu')
    validate_parent(payload,spec,identity,job,plan)
    return payload


def restore_rank_rng(payload,ctx,parent_sha=None):
    if parent_sha is None:
        states=payload.get('distributed_rng')
        if not isinstance(states,list) or len(states)!=ctx.world_size:
            raise ValueError('checkpoint rank RNG count differs from requested topology')
        restore_rng(states[ctx.rank])
    else:
        restore_rng(payload['rng'])
        if ctx.rank:
            seed=int.from_bytes(hashlib.sha256((parent_sha+':rank:'+str(ctx.rank)).encode()).digest()[:8],'big')%(2**63)
            single._seed(seed,ctx.device)


def selected_payload(selection,store):
    if selection.get('kind')=='parent':
        path=selection['checkpoint']
        if sha256_file(path)!=selection['sha256']:raise ValueError('parent best SHA mismatch')
        return load_file(path,map_location='cpu')
    return store.load('best',map_location='cpu')


def initialize_store(folder):
    """Commit empty initialization state before publishing a runnable contract."""
    store=CheckpointStore(Path(folder)/'checkpoints')
    index=store._index(allow_missing=True)
    if not (store.directory/'index.json').exists():atomic_json(store.directory/'index.json',index)
    return store


def prime_reducer(model,base,job,ctx):
    """Rebuild DDP buckets before real updates, without changing model or RNG."""
    if ctx.world_size==1:return
    rng=capture_rng();state={k:v.detach().clone() for k,v in base.state_dict().items()}
    training=base.training
    try:
        base.train()
        images=torch.zeros(job['physical_batch'],3,job['image_size'],job['image_size'],device=ctx.device)
        from .age import enabled,forward
        age_features=torch.zeros(job['physical_batch'],2,dtype=torch.float32,device=ctx.device) if enabled(job) else None
        for _ in range(2):
            base.zero_grad(set_to_none=True)
            with single._autocast(ctx.device,job['precision']):logits=forward(model,images,age_features,job)
            (logits.float().sum()*0).backward()
    finally:
        base.load_state_dict(state);base.zero_grad(set_to_none=True)
        base.train(training);restore_rng(rng)


def validate_explicit_microbatch_normalization(model, plan):
    if plan.get('explicit_per_rank_microbatch') and any(
            isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        from .imagenet_resnet import ImageNetResnetBinary
        if not isinstance(model, ImageNetResnetBinary):
            raise ValueError('explicit microbatch requires no BN or the fixed-BN ImageNet R50')
        model.assert_bn_mode()


def run_distributed(job,manifest,data_root,run_dir,identity,ctx,*,resume=False,mode='train',
                    parent_spec=None,model_factory=None,stop_event=None,limit_updates=None,stage_spec=None,fresh_spec=None,deadline_at=None,
                    derived_receipt=None):
    """Every rank calls this; only rank0 owns the writer and published artifacts."""
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    lock=RunLock(run_dir/'.writer.lock') if ctx.primary else contextlib.nullcontext()
    try:
        with lock:
            result=_run(job,manifest,data_root,run_dir,identity,ctx,resume,mode,parent_spec,
                        model_factory,stop_event,limit_updates,stage_spec,fresh_spec,deadline_at,derived_receipt)
            if fresh_spec:
                atomic_json(run_dir/f'rank-{ctx.rank}-terminal-{result["attempted_updates"]:06d}.json',{
                    'rank':ctx.rank,'at':time.time(),'status':result['status'],'mode':mode,
                    'successful_updates':result['successful_updates'],'attempted_updates':result['attempted_updates'],
                    'skipped_updates':result['skipped_updates'],'sampler_attempted_updates':result['sampler_attempted_updates'],
                    'fresh_guard':result['fresh_guard'],'ema':result.get('ema'),'ema_tensor_sha256':result.get('ema_tensor_sha256'),'stop_reason':result.get('stop_reason'),
                    'absolute_deadline_at':deadline_at})
            return result
    except BaseException as error:
        if fresh_spec and fresh_spec.get('ema'):
            import re
            fatal=(isinstance(error,(ValueError,FloatingPointError,CheckpointError,torch.OutOfMemoryError)) or
                   bool(re.search(r'non.?finite|\b(?:nan|inf|oom)\b|out.?of.?memory|EMA|size mismatch|missing key|unexpected key',str(error),re.I)))
            atomic_json(run_dir/f'rank-{ctx.rank}-failure-{time.time_ns()}.json',{
                'rank':ctx.rank,'at':time.time(),'error_type':type(error).__name__,'error':str(error),
                'non_retryable':fatal,'classification':'numeric_or_scientific_integrity' if fatal else 'engineering_unclassified',
                'identity':identity,'fresh':fresh_spec,'absolute_deadline_at':deadline_at})
        if ctx.primary:atomic_json(run_dir/'status.json',{'status':'failed','mode':mode,
            'error':type(error).__name__+': '+str(error),'distributed':identity['distributed']})
        raise


def _run(job,manifest,data_root,folder,identity,ctx,resume,mode,parent_spec,model_factory,stop_event,limit_updates,stage_spec,fresh_spec,deadline_at,derived_receipt):
    if bool(derived_receipt) != bool(fresh_spec and fresh_spec.get('ssl_recipe')):
        raise ValueError('R115 derived asset requires the frozen joint SSL recipe')
    if derived_receipt and (mode!='train' or resume or derived_receipt.get('recipe')!=fresh_spec['ssl_recipe']
            or derived_receipt.get('successful_updates')!=1600):
        raise ValueError('R115 derived asset/phase identity differs')
    if job.get('experiment91') is not None:
        from .experiment91 import validate_runtime
        validate_runtime(identity,job,manifest,fresh_spec)
    if mode not in ('train','benchmark','stage-probe','fresh-probe'):raise ValueError('invalid distributed mode')
    if (mode=='stage-probe' and not stage_spec) or (stage_spec and (parent_spec or mode=='benchmark' or limit_updates is not None)):
        raise ValueError('stage initialization requires its explicit purpose / stopping contract')
    if stage_spec:
        from .stage import load_stage,initialize_stage,sampler_start,StageOverflowGuard
        if (stage_spec['purpose']=='benchmark')!=(mode=='stage-probe'):
            raise ValueError('stage probe / formal purpose mismatch')
    if ((mode=='fresh-probe' and not fresh_spec) or
            (fresh_spec and (parent_spec or stage_spec or limit_updates is not None))):
        raise ValueError('fresh initialization forbids parent / stage / implicit limits')
    if deadline_at is not None and (not fresh_spec or not math.isfinite(deadline_at)):
        raise ValueError('absolute runtime deadline requires a finite fresh contract')
    plan=identity['distributed'];device=ctx.device
    pairwise=(job.get('loss') or {}).get('kind') in PAIRWISE_LOSS_KINDS
    if pairwise and (mode!='train' or not fresh_spec or ctx.world_size!=8 or plan['world_size']!=8
            or plan['accumulation']!=1 or job.get('mixup') is not None or job.get('training_views',1)!=1):
        raise ValueError('global pairwise loss requires formal eight-rank single-view accumulation-one training without mixup')
    precision_runtime=None
    if fresh_spec and fresh_spec.get('precision_protocol') is not None:
        from .precision import configure
        precision_runtime=configure(fresh_spec['precision_protocol'])
    if job.get('mixup') is not None:
        from .mixup import validate_execution
        validate_execution(job,plan)
        if mode!='train':raise ValueError('registered Mixup is only for formal training')
    evaluation_job={**job,'physical_batch':fresh_spec.get('execution_policy',{}).get('evaluation_physical_batch',job['physical_batch'])} if fresh_spec else job
    if fresh_spec:
        evaluation_job['runtime_deadline_at']=deadline_at
    if plan['world_size']!=ctx.world_size or job['physical_batch']!=plan['per_rank_batch']:
        raise ValueError('worker topology differs from sealed identity')
    if canonical_digest(manifest)!=identity['manifest_digest']:raise ValueError('manifest identity differs')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    negatives=sum(not manifest['rows'][i]['label'] for i in manifest['fit_indices'])
    from .fresh import resolve_steps_per_cycle
    steps=resolve_steps_per_cycle(job,negatives,fresh_spec=fresh_spec)
    total=250 if mode=='benchmark' else job['cycles']*steps
    warmup=job['warmup_cycles']*steps
    target=min(total,job.get('stop_after_cycles',0)*steps) if mode=='train' and job.get('stop_after_cycles') else total
    # A validation window has the same full LR budget; its stopping target is separately sealed.
    if limit_updates is not None:
        if not isinstance(limit_updates,int) or not 0<limit_updates<=total:raise ValueError('invalid stopping target')
        target=min(target,limit_updates)
    if stage_spec:
        schedule=stage_spec['schedule']
        if (schedule['total_updates']!=total or schedule['warmup_updates']!=warmup
                or schedule['steps_per_cycle']!=steps):raise ValueError('stage derived schedule differs')
        target=schedule['stop_after_updates']
    if fresh_spec:
        from .fresh import validate_fresh,FreshOverflowGuard
        validate_fresh(fresh_spec,identity,job,plan,steps,mode=mode)
        target=fresh_spec['schedule']['stop_after_updates']
    ema_rule=fresh_spec.get('ema') if fresh_spec else None
    if ema_rule:
        from .ema import EMA,evaluate_pair,commit_pair
    store=CheckpointStore(folder/'checkpoints')
    previous=None;from_parent=False;from_stage=False;stage_reference=None
    if not resume:
        if ctx.primary:
            if (folder/'complete.json').exists() or ((store.directory/'index.json').exists() and not store.is_empty()):
                raise FileExistsError('distributed output exists; explicit resume required')
            initialize_store(folder)
        ctx.barrier()
    if resume:
        if not store.is_empty():previous=store.load('last',map_location='cpu')
        if previous is not None and (previous.get('identity')!=identity or previous.get('mode')!=mode
                or previous.get('manifest_digest')!=identity['manifest_digest']
                or previous.get('schedule')!={**previous['schedule'],
                    'total_updates':total,'warmup_updates':warmup,'stop_after_updates':target,'steps_per_cycle':steps}):
            raise ValueError('DDP resume contract or schedule changed')
        if previous is not None and stage_spec and previous.get('stage')!=stage_spec:
            raise ValueError('stage resume provenance differs')
    if fresh_spec and previous is not None:
        if (previous.get('fresh')!=fresh_spec or not previous.get('fresh_guard')
                or previous.get('absolute_deadline_at')!=deadline_at):
            raise ValueError('fresh resume provenance / AMP window absent or changed')
        if CheckpointStore(folder/'initial-checkpoint').is_empty():
            raise ValueError('fresh resume missing immutable initialization proof')
    from .age import validate_checkpoint,validate_model
    if previous is not None:validate_checkpoint(previous,job)
    if previous is None and parent_spec is not None:
        if mode!='train':raise ValueError('benchmarks cannot continue formal checkpoints')
        previous=load_parent(parent_spec,identity,job,plan);from_parent=True
        if previous['schedule']['steps_per_cycle']!=steps:raise ValueError('parent cycle size differs')
    if previous is None and stage_spec:
        original=load_stage(stage_spec,identity,job,plan)
        if job['image_size']==original['job']['image_size']:stage_reference=copy.deepcopy(original.get('last_metrics'))
        previous=initialize_stage(original,stage_spec,identity,job,plan);from_stage=True
        del original
    single._seed(int(job['seed'])+ctx.rank,device)
    from .attention_pool import validate_resume as validate_attention_resume
    if previous is not None:validate_attention_resume(previous,job,identity)
    if model_factory is None:
        from .models import create_model
        model_factory=create_model
    base=model_factory(job,initialize=previous is None).to(device)
    factory_rng=capture_rng() if derived_receipt is not None else None
    factory_non_encoder_sha=None
    if derived_receipt is not None:
        from .ssl_stage import _non_encoder_state
        from .ema import tensor_digest
        factory_non_encoder_sha=tensor_digest(_non_encoder_state(base))
    validate_model(base,job)
    validate_explicit_microbatch_normalization(base, plan)
    base=ctx.prepare_model(base)
    optimizer=single._optimizer(base,job)
    scaler=torch.amp.GradScaler(device.type,enabled=job['precision']=='fp16')
    if previous is not None:
        base.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        if precision_runtime is not None:
            from .precision import validate_scaler_state
            validate_scaler_state(previous['scaler'], job['precision'])
        scaler.load_state_dict(previous['scaler'])
        if from_stage:
            for group in optimizer.param_groups:
                group['initial_lr']=float(job['head_lr' if group['role']=='head' else 'backbone_lr'])
    if precision_runtime is not None:
        from .precision import runtime_proof
        precision_runtime=runtime_proof(fresh_spec['precision_protocol'],scaler=scaler,device=device)
    # The no-skip path is admitted from the post-load scaler proof, not the job label.
    fixed_fp32_scaler=(mode=='train' and precision_runtime is not None
                       and precision_runtime.get('precision')=='fp32'
                       and precision_runtime.get('scaler_enabled') is False
                       and precision_runtime.get('effective_scale')==1.0
                       and precision_runtime.get('scaler_state_dict')=={})
    model=ctx.wrap(base)
    prime_reducer(model,base,job,ctx)
    if factory_rng is not None:
        restore_rng(factory_rng)
    if previous is not None:restore_rank_rng(previous,ctx,parent_spec['sha256'] if from_parent else None)
    mirror=ReferenceBatchNorm(base,ctx)
    successful=int(previous['successful_updates']) if previous else 0
    attempted=int(previous['attempted_updates']) if previous else 0
    evaluated=int(previous['evaluated_cycle']) if previous else 0
    early=copy.deepcopy(previous['early_stop']) if previous else {'reference_auroc':None,'bad_cycles':0,'stopped':False}
    smoke_interruption_completed=bool(previous.get('smoke_interruption_completed',False)) if previous else False
    best=copy.deepcopy(previous.get('best_metrics')) if previous else None
    last=copy.deepcopy(previous.get('last_metrics')) if previous else None
    ema=EMA(base,ema_rule,state=previous.get('ema') if previous else None,successful=successful) if ema_rule else None
    n05_bn_initial=None
    sampled_loss=None
    if job.get('model') in ('flair_resnet50','resnet50_imagenet'):
        if job.get('model')=='resnet50_imagenet':
            from .imagenet_resnet import frozen_bn_digest
        else:
            from .flair_model import frozen_bn_digest
        if not ema:raise ValueError('N05 requires frozen-BN EMA')
        n05_bn_initial=frozen_bn_digest(base)
        initial_bn=ctx.values({'rank':ctx.rank,'online':n05_bn_initial,'ema':frozen_bn_digest(ema.model)})
        if any(x['online']!=n05_bn_initial or x['ema']!=n05_bn_initial for x in initial_bn):
            raise ValueError('N05 initial BN buffers differ across ranks or EMA')
        if ctx.primary:atomic_json(folder/'bn-initial-ranks.json',initial_bn)
        sampled_loss=torch.zeros(4,device=device,dtype=torch.float64)
    if ema and previous is not None and any(k not in previous for k in ('ema','ema_best_metrics','ema_last_metrics','ema_rank_proof')):
        raise ValueError('coupled EMA resume history absent')
    ema_best=copy.deepcopy(previous.get('ema_best_metrics')) if previous else None
    ema_last=copy.deepcopy(previous.get('ema_last_metrics')) if previous else None
    ema_rank_proof=None
    coverage=set(previous.get('coverage',{}).get('indices',[])) if previous else set()
    overflows=int(previous.get('consecutive_overflows',0)) if previous else 0
    overflow_guard=StageOverflowGuard(previous.get('overflow_window',[])) if stage_spec else None
    if fresh_spec:
        overflow_guard=FreshOverflowGuard(previous['fresh_guard'] if previous else None)
        guard=overflow_guard.state()
        if guard['successful']!=successful or guard['total_skips']!=attempted-successful:
            raise ValueError('fresh AMP counter / committed checkpoint mismatch')
    timing={k:0.0 for k in ('training_seconds','data_wait_seconds','evaluation_seconds','checkpoint_seconds',
                            'warmup_seconds','measured_seconds')}
    if ema:timing['checkpoint_count']=0
    if previous:timing.update(previous.get('timing',{}))
    selection=copy.deepcopy(previous.get('lineage_best',{})) if previous and not from_parent else {}
    if from_parent and parent_spec.get('best'):
        selection={'kind':'parent',**parent_spec['best']}
    provenance=({'kind':'audited-single-to-ddp-v1','parent_checkpoint':parent_spec['checkpoint'],
                 'parent_sha256':parent_spec['sha256'],'parent_identity':parent_spec['identity'],
                 'parent_updates':successful} if from_parent else previous.get('continuation') if previous else None)
    if stage_spec:
        provenance={'kind':'sealed-stage-init-v1','parent_checkpoint':stage_spec['checkpoint'],
                    'parent_sha256':stage_spec['sha256'],'parent_identity':stage_spec['identity'],
                    'parent_updates':stage_spec['parent_successful_updates'],
                    'sampler_attempt_offset':stage_spec['sampler_attempt_offset'],
                    'purpose':stage_spec['purpose'],'plan_sha256':stage_spec['plan_sha256']}
        if from_stage:selection={}
    if fresh_spec:
        provenance={'kind':'sealed-public-fresh-init-v1','initialization':identity['initialization'],
                    'purpose':fresh_spec['purpose'],'plan_sha256':fresh_spec['plan_sha256']}
        if derived_receipt is not None:
            provenance={'kind':'same-owner-fit-only-vicreg-derived-init-v1',
                        'public_initialization':identity['initialization'],
                        'ssl_recipe':fresh_spec['ssl_recipe'],
                        'derived_tensor_sha256':derived_receipt['derived_tensor_sha256'],
                        'derived_asset_sha256':derived_receipt['asset_sha256'],
                        'purpose':fresh_spec['purpose'],'plan_sha256':fresh_spec['plan_sha256']}
    start=time.monotonic();last_save=start;last_saved_attempt=attempted
    complete=folder/'complete.json'
    if ctx.primary:complete.unlink(missing_ok=True)

    stop_reason=None
    probe_timed_seconds=[]
    probe_gradient_proofs=None
    probe_cgroup_peak=0
    def stopping():
        nonlocal stop_reason,probe_cgroup_peak
        reason=None
        if fresh_spec:
            from .fresh import runtime_stop_reason
            reason=runtime_stop_reason(deadline_at,device)
            if mode=='fresh-probe':
                try:probe_cgroup_peak=max(probe_cgroup_peak,int(Path('/sys/fs/cgroup/memory.current').read_text().strip()))
                except (OSError,ValueError):reason=reason or 'probe_cgroup_memory_unavailable'
        if reason is None and stop_event is not None and stop_event.is_set():reason='owner_stop_requested'
        stopped=ctx.any(reason is not None)
        if stopped:stop_reason=next((v for v in ctx.values(reason) if v),None)
        return stopped

    def log(event,**extra):
        if ctx.primary:
            data={'event':event,'successful_updates':successful,'attempted_updates':attempted,'cycle':successful//steps,**extra}
            if stage_spec:data.update(at=time.time(),sampler_attempted_updates=sampler_start(stage_spec,attempted),purpose=stage_spec['purpose'])
            if fresh_spec:data.update(at=time.time(),sampler_attempted_updates=attempted,purpose=fresh_spec['purpose'])
            with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')
            print(json.dumps(data,allow_nan=False),flush=True)

    def memory():
        devices=ctx.values(single._memory(device))
        value=dict(devices[0]);value['devices']=devices
        return value

    def status(state,**extra):
        mem=memory()
        result={'status':state,'mode':mode,'result_kind':'benchmark' if mode=='benchmark' else 'training',
            'successful_updates':successful,'attempted_updates':attempted,'skipped_updates':attempted-successful,
            'cycle':successful//steps,'evaluated_cycle':evaluated,'steps_per_cycle':steps,
            'best_metrics':best,'last_metrics':last,'audit_metrics':None,'early_stop':early,
            'termination_reason':('early_stopping' if early['stopped'] else 'maximum_updates') if state=='completed' and mode=='train' else None,
            'coverage_unique':len(coverage),'timing':dict(timing),'memory':mem,
            'wall_seconds_this_process':time.monotonic()-start,'distributed':plan,'lineage_best':selection,**extra}
        if stage_spec:
            result.update(purpose=stage_spec['purpose'],stage=stage_spec,
                          sampler_attempted_updates=sampler_start(stage_spec,attempted),
                          optimizer_expected_step=stage_spec['parent_successful_updates']+successful,
                          stage0_metrics=best if best and best.get('cycle')==0 else None,
                          result_kind='stage-probe' if mode=='stage-probe' else 'training')
        if fresh_spec:
            result.update(purpose=fresh_spec['purpose'],fresh=fresh_spec,fresh_guard=overflow_guard.state(),
                          sampler_attempted_updates=attempted,optimizer_expected_step=successful,
                          absolute_deadline_at=deadline_at,stop_reason=stop_reason,
                          result_kind='fresh-probe' if mode=='fresh-probe' else 'training')
        if derived_receipt is not None:result['ssl_derived']=derived_receipt
        if ema:
            result.update(ema={k:v for k,v in ema.state_dict().items() if k!='model'},
                          ema_best_metrics=ema_best,ema_last_metrics=ema_last,
                          ema_rank_proof=ema_rank_proof,
                          ema_tensor_sha256=ema_rank_proof[ctx.rank]['tensor_sha256'] if ema_rank_proof else None)
        if mode in ('benchmark','stage-probe','fresh-probe'):
            count=max(0,successful-50)
            result['benchmark']={'warmup_updates':min(50,successful),'measured_updates':count,
                'warmup_seconds':timing['warmup_seconds'],'measured_seconds':timing['measured_seconds'],
                'seconds_per_update':timing['measured_seconds']/count if count else None}
        if precision_runtime is not None:result['precision_runtime']=precision_runtime
        if state == 'completed' and job.get('attention_pool') is not None:
            from .attention_pool import describe
            result['attention_pool'] = {'online': describe(base,optimizer),
                'ema': describe(ema.model) if ema else None}
        if ctx.primary:atomic_json(folder/'status.json',result)
        return result

    def save(improved=False,ema_improved=False):
        nonlocal last_save,last_saved_attempt,selection,ema_rank_proof
        save_started=time.monotonic()
        validate_model(base,job)
        candidate_normalization=None
        if job.get('experiment91') is not None:
            from .experiment91 import describe_model
            candidate_normalization=describe_model(base)
            if ema and describe_model(ema.model)!=candidate_normalization:raise ValueError('Tiny online/EMA actual structure differs')
        from .models import feature_freeze_proof
        parameter_trainability=feature_freeze_proof(base,job,optimizer)
        n05_bn_proofs=None
        if n05_bn_initial is not None:
            if job.get('model')=='resnet50_imagenet':
                from .imagenet_resnet import frozen_bn_digest
            else:
                from .flair_model import frozen_bn_digest
            base.assert_bn_mode();ema.model.assert_bn_mode()
            n05_bn_proofs=ctx.values({'rank':ctx.rank,'online':frozen_bn_digest(base),
                                     'ema':frozen_bn_digest(ema.model),'expected':n05_bn_initial})
            if any(v['online']!=n05_bn_initial or v['ema']!=n05_bn_initial for v in n05_bn_proofs):
                raise ValueError('N05 fixed BN buffer drift detected at save')
        rng=ctx.values(capture_rng())
        if ema:ema_rank_proof=ema.rank_proof(ctx)
        initial_ranks=None
        if fresh_spec and successful==0 and attempted==0:
            digest=hashlib.sha256()
            for name,tensor in sorted(base.state_dict().items()):
                value=tensor.detach().cpu().contiguous()
                digest.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode()+value.numpy().tobytes())
            attention_initial={}
            if candidate_normalization is not None:attention_initial['candidate_normalization']=candidate_normalization
            if parameter_trainability is not None:attention_initial['parameter_trainability']=parameter_trainability
            if job.get('attention_pool') is not None:
                from .attention_pool import describe
                attention_initial['attention_pool']=describe(base,optimizer)
            initial_ranks=ctx.values({'rank':ctx.rank,'model_sha256':digest.hexdigest(),
                'optimizer_state_entries':len(optimizer.state),'scaler':scaler.state_dict(),
                'precision_runtime':precision_runtime,
                'bn_modules':sum(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in base.modules()),
                **({'supervised_factory_non_encoder_sha256':factory_non_encoder_sha,
                    'supervised_factory_rng_restored':True} if derived_receipt is not None else {}),
                **attention_initial})
            if len({v['model_sha256'] for v in initial_ranks})!=1:
                raise ValueError('fresh ranks initial model tensors differ')
        before=save_started if ema else time.monotonic()
        if ctx.primary:
            payload={'format_version':1,'identity':identity,'job':dict(job),'mode':mode,
                'age':copy.deepcopy(job.get('age')),
                'smoke_interruption_completed':smoke_interruption_completed,
                'manifest_digest':identity['manifest_digest'],'model':base.state_dict(),'optimizer':optimizer.state_dict(),
                'scaler':scaler.state_dict(),'rng':rng[0],'distributed_rng':rng,
                'successful_updates':successful,'attempted_updates':attempted,'cycle':successful//steps,
                'evaluated_cycle':evaluated,'early_stop':copy.deepcopy(early),'best_metrics':best,
                'last_metrics':last,'audit_metrics':None,'consecutive_overflows':overflows,
                'schedule':{'total_updates':total,'warmup_updates':warmup,'stop_after_updates':target,
                            'steps_per_cycle':steps,'successful_updates':successful},
                'coverage':{'indices':sorted(coverage),'unique':len(coverage),
                    'positive':sum(bool(manifest['rows'][i]['label']) for i in coverage),
                    'negative':sum(not manifest['rows'][i]['label'] for i in coverage)},'timing':dict(timing),
                'continuation':provenance,'lineage_best':({'kind':'child'} if improved else selection)}
            if ema:
                payload.update(ema=ema.state_dict(),ema_best_metrics=ema_best,ema_last_metrics=ema_last,
                               ema_rank_proof=ema_rank_proof,evaluation_transaction='both_variants_committed' if evaluated==successful//steps else 'pending_pair')
            if candidate_normalization is not None:payload['candidate_normalization']=candidate_normalization
            if parameter_trainability is not None:payload['parameter_trainability']=parameter_trainability
            if n05_bn_proofs is not None:payload['frozen_bn_proofs']=n05_bn_proofs
            if stage_spec:payload.update(stage=copy.deepcopy(stage_spec),overflow_window=overflow_guard.state())
            if fresh_spec:
                payload.update(fresh=copy.deepcopy(fresh_spec),fresh_guard=overflow_guard.state(),absolute_deadline_at=deadline_at)
                if derived_receipt is not None:payload['ssl_derived']=copy.deepcopy(derived_receipt)
                if precision_runtime is not None:payload['precision_runtime']=precision_runtime
                if successful==0 and attempted==0:
                    initial=CheckpointStore(folder/'initial-checkpoint')
                    if not (initial.directory/'index.json').exists():
                        atomic_json(initial.directory/'index.json',initial._index(allow_missing=True))
                    if initial.is_empty():
                        initial.save(payload,best=True)
                        atomic_json(folder/'initial-ranks.json',initial_ranks)
                        names={id(p):name for name,p in base.named_parameters()}
                        groups=[{'role':g['role'],'initial_lr':g['initial_lr'],'lr':g['lr'],
                                 'weight_decay':g['weight_decay'],
                                 'parameters':[{'name':names[id(p)],'shape':list(p.shape),'numel':p.numel()} for p in g['params']]}
                                for g in optimizer.param_groups]
                        atomic_json(folder/'parameter-groups.json',groups)
            protect=[]
            if ema:
                if improved:protect.append('best_online')
                if ema_improved:protect.append('best_ema')
                protected_updates=({steps,2*steps,3*steps,ema_rule['warmcopy_updates'],ema_rule['warmcopy_updates']+1} if fresh_spec.get('execution_policy') else {237,474,475,711})
                if successful in protected_updates:protect.append('snapshot-'+str(successful))
            store.save(payload,best=improved,protect=protect)
            if improved:selection={'kind':'child'}
            atomic_json(folder/'selection.json',{'best_metrics':best,'ema_best_metrics':ema_best,'selection':selection,'continuation':provenance})
        selection=ctx.broadcast(selection)
        ctx.barrier()
        timing['checkpoint_seconds']+=ctx.maximum(time.monotonic()-before)
        if ema:timing['checkpoint_count']+=1
        last_save=time.monotonic();last_saved_attempt=attempted

    def interrupt():
        save();log('interrupted');return status('interrupted')

    def validate_pending():
        nonlocal evaluated,best,last,early,ema_best,ema_last
        while evaluated<successful//steps:
            cycle=successful//steps
            if cycle-evaluated!=1:raise ValueError('DDP checkpoint skipped an evaluation boundary')
            save();started=time.monotonic()
            metrics=None
            if ema:
                pair=evaluate_pair(ctx,lambda variant:parallel_evaluate(base if variant=='online' else ema.model,
                    manifest['rows'],manifest['dev_indices'],data_root,evaluation_job,device,ctx,
                    folder/'predictions'/variant/f'dev-cycle-{cycle:04d}.csv',variant=variant,
                    state_identity={'successful_updates':successful,'attempted_updates':attempted,
                                    'ema_updates':ema.successful if variant=='ema' else None}),all_ranks=True)
                timing['evaluation_seconds']+=ctx.maximum(time.monotonic()-started)
                state,flags=commit_pair({'evaluated_cycle':evaluated,'early_stop':early,'best_metrics':best,
                    'last_metrics':last,'ema_best_metrics':ema_best,'ema_last_metrics':ema_last},pair,cycle,successful,job)
                evaluated=state['evaluated_cycle'];early=state['early_stop'];best=state['best_metrics'];last=state['last_metrics']
                ema_best=state['ema_best_metrics'];ema_last=state['ema_last_metrics']
                log('evaluation',metrics=pair['online'],ema_metrics=pair['ema'],best=flags['online'],ema_best=flags['ema'],early_stop=dict(early))
                save(flags['online'],flags['ema']);status('running');continue
            if ctx.primary:
                metrics=single.evaluate(base,manifest['rows'],manifest['dev_indices'],data_root,job,device,
                    folder/'predictions'/f'dev-cycle-{cycle:04d}.csv')
            metrics=ctx.broadcast(metrics)
            timing['evaluation_seconds']+=ctx.maximum(time.monotonic()-started)
            if metrics['auroc'] is None:raise ValueError('Dev AUROC is undefined')
            last=metrics
            improved=best is None or metrics['auroc']>best['auroc'] or (metrics['auroc']==best['auroc'] and metrics['ap']>best['ap'])
            if improved:best={**metrics,'cycle':cycle,'successful_updates':successful}
            reference=early['reference_auroc']
            if reference is None or metrics['auroc']>reference+job['min_delta']:
                early.update(reference_auroc=metrics['auroc'],bad_cycles=0)
            else:early['bad_cycles']+=1
            if not stage_spec and cycle>=job['min_cycles'] and early['bad_cycles']>=job['patience']:early['stopped']=True
            evaluated=cycle;log('evaluation',metrics=metrics,best=improved,early_stop=dict(early))
            save(improved);status('running')

    single._apply_lr(optimizer,successful,total,warmup)
    if previous is None or from_parent or from_stage:save()
    if fresh_spec and fresh_spec.get('startup_barrier'):
        from .fresh import wait_start_barrier
        log('waiting_start_barrier');status('waiting_start_barrier')
        if not wait_start_barrier(fresh_spec,ctx,stopping):return interrupt()
    status('running')
    if from_parent:log('continued_from_parent',parent_sha256=parent_spec['sha256'],parent_source=parent_spec['identity']['source_digest'])
    stage0_pending=bool(stage_spec and successful==0 and best is None)
    if stage0_pending and not from_stage:
        original=load_stage(stage_spec,identity,job,plan)
        if job['image_size']==original['job']['image_size']:stage_reference=copy.deepcopy(original.get('last_metrics'))
        del original
    if stage0_pending:
        started=time.monotonic();metrics=None
        if ctx.primary:
            metrics=single.evaluate(base,manifest['rows'],manifest['dev_indices'],data_root,job,device,
                folder/'predictions/dev-cycle-0000.csv')
        metrics=ctx.broadcast(metrics)
        if metrics['auroc'] is None:raise ValueError('stage0 AUROC undefined')
        if stage_reference and any(abs(metrics[k]-stage_reference[k])>1e-6 for k in ('auroc','ap')):
            raise ValueError('same-input stage0 does not reproduce parent AUROC/AP within 1e-6')
        timing['evaluation_seconds']+=ctx.maximum(time.monotonic()-started)
        best={**metrics,'cycle':0,'successful_updates':0};last=metrics
        early.update(reference_auroc=metrics['auroc'],bad_cycles=0,stopped=False)
        log('stage0_evaluation',metrics=metrics);save(True);status('running')
    if mode in ('train','stage-probe','fresh-probe'):validate_pending()
    if stopping():return interrupt()
    iterator=None;loader=None
    try:
        if successful<target and not early['stopped']:
            dataset=ImageDataset(data_root,manifest['rows'],job['image_size'],training=True,
                normalization=job.get('normalization','unit'),prepared_size=job.get('prepared_size',0),age=job.get('age'),
                training_scale=job.get('training_scale'),training_views=job.get('training_views'),
                training_photometric=job.get('training_photometric'))
            sampler=RankBatchSampler(manifest['rows'],manifest['fit_indices'],plan['reference_physical_batch'],
                job['effective_batch'],job['positive_per_batch'],job['seed'],sampler_start(stage_spec,attempted) if stage_spec else attempted,
                (target-successful+1)*101,rank=ctx.rank,world_size=ctx.world_size,per_rank_microbatch=plan.get('explicit_per_rank_microbatch'))
            loader=DataLoader(dataset,batch_sampler=sampler,**single._loader_options(job['workers'],job,
                persistent=True,seed=job['seed']+ctx.rank))
            iterator=iter(loader)
        while successful<target and not early['stopped']:
            if stopping():return interrupt()
            model.train()
            if n05_bn_initial is not None:base.assert_bn_mode()
            optimizer.zero_grad(set_to_none=True);single._synchronize(device)
            if mode=='fresh-probe' and successful>=50 and 'measurement_started_at' not in timing:
                timing['measurement_started_at']=ctx.broadcast(time.time() if ctx.primary else None)
            started=time.monotonic();local_indices=[];loss_sum=0.0;waiting=0.0
            primary_inputs=view_inputs=0
            mirror.capture()
            for micro in range(plan['accumulation']):
                from .age import unpack_batch,forward
                before=time.monotonic();batch=next(iterator);waiting+=time.monotonic()-before
                images,labels,indices,age_features=unpack_batch(batch,job,device)
                if attempted==0 and micro==0:
                    log('first_batch_dataset_policy',training_photometric=dataset.training_photometric,
                        primary_samples=int(images.shape[0]), batch_arity=len(batch),
                        age_enabled=bool(job['age']['enabled']),
                        age_features_present=age_features is not None)
                local_indices.extend(indices.tolist())
                images=images.to(device,non_blocking=True);labels=labels.to(device,dtype=torch.float32,non_blocking=True).reshape(-1)
                primary_inputs+=images.shape[0]
                images,labels,age_features=flatten_training_views(images,labels,age_features,job.get('training_views'))
                view_inputs+=images.shape[0]
                if job.get('mixup') is not None:
                    from .mixup import apply as apply_mixup
                    first_mass=float(labels.sum()) if attempted==0 else None
                    images,labels,age_features,mixup_proof=apply_mixup(job['mixup'],images,labels,age_features,
                        job_seed=job['seed'],attempted=attempted,rank=ctx.rank,micro=micro,
                        world_size=ctx.world_size,training=True)
                    if attempted==0:
                        mixup_proof.update(primary_indices=indices.tolist(),batch_shape=list(images.shape),
                            original_positive_mass=first_mass,mixed_positive_mass=float(labels.sum()))
                        proofs=ctx.values(mixup_proof)
                        if ctx.primary:atomic_json(folder/'mixup-initial-ranks.json',proofs)
                with single._autocast(device,job['precision']):
                    logits=forward(model,images,age_features,job).reshape(-1)
                    if pairwise:
                        global_logits,global_targets=gather_pairwise_inputs(logits,labels)
                        loss=classification_loss(global_logits,global_targets,job['loss'])
                    else:
                        loss=classification_loss(logits,labels,job.get('loss'))
                if sampled_loss is not None:
                    with torch.no_grad():
                        individual=nn.functional.binary_cross_entropy_with_logits(logits.detach(),labels,reduction='none')
                        positive=labels==1;negative=labels==0
                        if not bool((positive|negative).all()):raise ValueError('N05 sampled label differs')
                        sampled_loss[0]+=individual[positive].sum().double();sampled_loss[1]+=positive.sum().double()
                        sampled_loss[2]+=individual[negative].sum().double();sampled_loss[3]+=negative.sum().double()
                if ctx.any(not bool(torch.isfinite(loss))):raise FloatingPointError('nonfinite loss on a DDP rank')
                loss_sum+=float(loss.detach())/plan['accumulation']
                scaler.scale(loss/plan['accumulation']).backward()
            mirror.merge()
            attempted+=1
            for indices in ctx.gather_indices(local_indices):coverage.update(indices)
            scaler.unscale_(optimizer)
            if mode=='fresh-probe' and attempted==1:
                from .ema import tensor_digest
                named={name:param for name,param in base.named_parameters() if param.requires_grad}
                owned=[param for group in optimizer.param_groups for param in group['params']]
                missing=[name for name,param in named.items() if param.grad is None]
                bad=[name for name,param in named.items() if param.grad is not None and not bool(torch.isfinite(param.grad).all())]
                unique=(len(owned)==len({id(p) for p in owned}) and {id(p) for p in owned}=={id(p) for p in named.values()})
                digest=tensor_digest({name:param.grad for name,param in named.items()}) if not missing and not bad else None
                probe_gradient_proofs=ctx.values({'rank':ctx.rank,'forward_backward':True,'trainable_count':len(named),
                    'optimizer_owns_each_once':unique,'missing_gradients':missing,'nonfinite_gradients':bad,'gradient_sha256':digest})
                if (any(not r['optimizer_owns_each_once'] or r['missing_gradients'] or r['nonfinite_gradients'] for r in probe_gradient_proofs)
                        or len({r['gradient_sha256'] for r in probe_gradient_proofs})!=1):
                    raise ValueError('probe rank gradient ownership / synchronization proof failed')
            norm=torch.nn.utils.clip_grad_norm_(base.parameters(),1.0)
            finite=not ctx.any(not bool(torch.isfinite(norm)))
            if not finite and not scaler.is_enabled():raise FloatingPointError('nonfinite DDP gradients without AMP')
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if fixed_fp32_scaler:
                skipped=False
            else:
                skipped=ctx.agreed(scaler.is_enabled() and scaler.get_scale()<old_scale)
                if precision_runtime is not None and skipped:raise FloatingPointError('FP32 protocol forbids any skipped optimizer attempt')
            if not finite and not skipped:raise FloatingPointError('AMP did not reject nonfinite gradients')
            if skipped:overflows+=1
            else:
                successful+=1;overflows=0
                bad_parameters=not all_floating_finite(base.parameters())
                # FP32 Adam moments are scientific state, checked without touching RNG.
                bad_adam=not all_floating_finite(value for state in optimizer.state.values() for value in state.values())
                bad_parameters,bad_adam=ctx.any_flags([bad_parameters,bad_adam])
                if bad_parameters:raise FloatingPointError('nonfinite DDP parameters')
                if bad_adam:raise FloatingPointError('nonfinite Adam state on a DDP rank')
                single._apply_lr(optimizer,successful,total,warmup)
                if ema:ema.update(base,successful)
            single._synchronize(device)
            local_elapsed=time.monotonic()-started
            if mode=='fresh-probe' and 50<successful<=100 and not skipped:
                probe_timed_seconds.append(local_elapsed)
            elapsed,waiting=ctx.maximum_values([local_elapsed,waiting])
            if mode=='fresh-probe' and successful>=target:
                timing['measurement_ended_at']=ctx.broadcast(time.time() if ctx.primary else None)
            timing['training_seconds']+=elapsed;timing['data_wait_seconds']+=waiting
            if mode in ('benchmark','stage-probe','fresh-probe'):timing['warmup_seconds' if successful-int(not skipped)<50 else 'measured_seconds']+=elapsed
            if attempted%job['log_every']==0 or skipped:
                update={'rank':ctx.rank,'loss':loss_sum,'primary_inputs':primary_inputs,'view_inputs':view_inputs}
                if pairwise:
                    update['positive_inputs']=int((labels==1).sum().item())
                    positives=int((global_targets==1).sum().item())
                    negatives=int((global_targets==0).sum().item())
                    update['pairwise']={'global_inputs':global_targets.numel(),'positive_inputs':positives,
                        'negative_inputs':negatives,'pair_count':positives*negatives,
                        'world_size':torch.distributed.get_world_size()}
                if job.get('mixup') is not None:
                    ids=list(local_indices)
                    trace={**mixup_proof,'primary_indices':ids,
                        'partner_indices':[ids[i] for i in mixup_proof['permutation']],
                        'original_positive_mass':sum(manifest['rows'][i]['label'] for i in ids),
                        'mixed_positive_mass':float(labels.sum())}
                    update['mixup']=trace
                updates=ctx.values(update)
                losses=[item['loss'] for item in updates]
                mixup_traces=[item['mixup'] for item in updates] if job.get('mixup') is not None else None
                log('update',loss=sum(losses)/len(losses),skipped=skipped,scale=scaler.get_scale(),
                    learning_rates=[g['lr'] for g in optimizer.param_groups],unique_samples=len(coverage),
                    mixup=mixup_traces,primary_inputs=sum(item['primary_inputs'] for item in updates),
                    view_inputs=sum(item['view_inputs'] for item in updates),
                    pairwise_global=updates[0]['pairwise'] if pairwise else None,
                    global_loss=sum(losses)/len(losses) if pairwise else None,
                    rank_inputs=[{key:item[key] for key in ('rank','primary_inputs','view_inputs')+(('positive_inputs',) if pairwise else ())} for item in updates])
                status('running')
            if overflows>=100:save();raise FloatingPointError('100 consecutive DDP AMP overflows')
            if overflow_guard:
                try:overflow_guard.record(skipped)
                except FloatingPointError:
                    save();raise
            if (fresh_spec and fresh_spec.get('interruption_test') and not smoke_interruption_completed
                    and successful==fresh_spec['interruption_test']['successful_update'] and not skipped):
                smoke_interruption_completed=True
                stop_reason='planned_smoke_recovery'
                save();log('planned_smoke_recovery')
                return status('interrupted')
            if ema and not skipped and successful==ema_rule['warmcopy_updates']+1:save()
            if stopping():return interrupt()
            if mode in ('train','stage-probe','fresh-probe') and not skipped and successful%steps==0:
                if sampled_loss is not None:
                    parts=ctx.values({'rank':ctx.rank,'values':sampled_loss.cpu().tolist()})
                    sums=[math.fsum(row['values'][i] for row in parts) for i in range(4)]
                    if ctx.primary:atomic_json(folder/('sampled-loss-cycle-%04d.json'%(successful//steps)),
                        {'scope':'sampled training batches, not full fit','cycle':successful//steps,
                         'positive_bce_sum':sums[0],'positive_count':int(sums[1]),
                         'negative_bce_sum':sums[2],'negative_count':int(sums[3]),'ranks':parts})
                    sampled_loss.zero_()
                validate_pending()
            want_save=recovery_checkpoint_due(job,attempted,last_saved_attempt,last_save,time.monotonic()) if ctx.primary else None
            if ctx.broadcast_flag(want_save):save()
            if stopping():return interrupt()
    finally:
        if iterator is not None and hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        del iterator,loader
    if manifest.get('audit_indices'):raise ValueError('distributed entry currently requires holdout without audit split')
    if mode=='fresh-probe' and job.get('model') in ('dinov2_vits14','flair_resnet50','resnet50_imagenet'):
        from .n03_probe import slowest_mean,project_seconds,equal_state_tree
        if successful!=100 or attempted!=100 or probe_gradient_proofs is None:
            raise ValueError('N03 requires exactly 100 successful fit-only probe updates')
        rank_times=ctx.values({'rank':ctx.rank,'timed_seconds':probe_timed_seconds})
        t_step=slowest_mean(rank_times)
        fit_prefix=list(manifest['fit_indices'][:1024])
        if len(fit_prefix)!=1024 or len(set(fit_prefix))!=1024:
            raise ValueError('N03 fit-only evaluation prefix differs')
        telemetry={};started=time.monotonic()
        fit_result=parallel_evaluate(base,manifest['rows'],fit_prefix,data_root,evaluation_job,device,ctx,
            None,variant='online',state_identity={'successful_updates':successful,'purpose':'fit-only-throughput'},
            telemetry=telemetry,metrics_enabled=False)
        t_eval1024=ctx.maximum(time.monotonic()-started)
        if fit_result!={'count':1024} or any(x['sample_count']!=256 for x in telemetry['rank_statistics']):
            raise ValueError('N03 fit-only evaluation coverage or four-rank participation differs')
        checkpoint_started=time.monotonic();save();load_proof=None
        if ctx.primary:
            try:
                restored=store.load('last',map_location='cpu')
                if (restored['successful_updates']!=100 or restored['attempted_updates']!=100
                        or not equal_state_tree(restored['model'],base.state_dict())
                        or not equal_state_tree(restored['optimizer'],optimizer.state_dict())
                        or not equal_state_tree(restored['ema'],ema.state_dict())):
                    raise ValueError('probe model/optimizer/EMA checkpoint round trip differs')
                load_proof={'passed':True,'successful_updates':100}
            except Exception as error:
                load_proof={'passed':False,'error':repr(error)}
        load_proof=ctx.broadcast(load_proof)
        if not load_proof['passed']:raise ValueError('probe checkpoint reload failed: '+str(load_proof))
        t_checkpoint=ctx.maximum(time.monotonic()-checkpoint_started)
        model_sha=tensor_digest(base.state_dict())
        rank_model_sha=ctx.values({'rank':ctx.rank,'model_tensor_sha256':model_sha})
        if len({x['model_tensor_sha256'] for x in rank_model_sha})!=1:
            raise ValueError('probe final model tensors differ across ranks')
        memory_records=ctx.values(single._memory(device))
        peak_cgroup=ctx.maximum(probe_cgroup_peak)
        try:cgroup_limit=int(Path('/sys/fs/cgroup/memory.max').read_text().strip())
        except (OSError,ValueError):raise ValueError('probe cgroup memory limit unavailable')
        resources_ok=(peak_cgroup<=.9*cgroup_limit and all(
            m['peak_reserved_bytes']<=.85*m['total_bytes'] for m in memory_records))
        projected=project_seconds(t_step,t_eval1024,t_checkpoint)
        projection_limit=18000
        if job.get('model') in ('flair_resnet50','resnet50_imagenet'):
            original_deadline=fresh_spec.get('round_deadline_at')
            if not isinstance(original_deadline,(int,float)) or not math.isfinite(original_deadline):
                raise ValueError('N05 original round deadline absent from fit-only probe')
            if job.get('model')=='resnet50_imagenet':
                projection_limit=min(3600,original_deadline-time.time()-900)
            else:
                projection_limit=min(9000,original_deadline-time.time()-1800)
        proof={'successful_updates':successful,'attempted_updates':attempted,'skipped_updates':0,
            'fit_prefix_count':1024,'fit_only':True,'fit_metrics_computed':False,'rank_timing':rank_times,
            't_step':t_step,'t_eval1024':t_eval1024,'t_checkpoint':t_checkpoint,
            'projected_formal_seconds':projected,'projection_limit_seconds':projection_limit,
            'gradient_proofs':probe_gradient_proofs,'rank_model_sha256':rank_model_sha,
            'fit_telemetry':telemetry,'checkpoint_reload':load_proof,'rank_memory':memory_records,
            'sampled_cgroup_peak_bytes':peak_cgroup,'cgroup_limit_bytes':cgroup_limit,
            'resources_ok':resources_ok,'capacity_pass':resources_ok and projection_limit>0 and projected<=projection_limit}
        if ctx.primary:atomic_json(folder/'probe-evidence.json',proof)
        last={'count':1024,'purpose':'fit-only-throughput'}
        result=status('completed',final_checkpoint='last',probe=proof)
        if ctx.primary:atomic_json(complete,result)
        log('probe_completed',capacity_pass=proof['capacity_pass'],projected_formal_seconds=projected)
        return result
    if mode=='fresh-probe' and job.get('model') in ('flair_resnet50','resnet50_imagenet'):
        raise RuntimeError('N05 fit-only probe must not fall through to dev evaluation')
    if mode in ('stage-probe','fresh-probe'):
        started=time.monotonic();metrics=None
        if ema:
            pair=evaluate_pair(ctx,lambda variant:parallel_evaluate(base if variant=='online' else ema.model,
                manifest['rows'],manifest['dev_indices'],data_root,evaluation_job,device,ctx,
                folder/'predictions'/variant/'probe-final.csv',variant=variant,
                state_identity={'successful_updates':successful,'attempted_updates':attempted,
                                'ema_updates':ema.successful if variant=='ema' else None}),all_ranks=True)
            metrics=pair['online'];ema_last=pair['ema']
        else:
            if ctx.primary:
                metrics=single.evaluate(base,manifest['rows'],manifest['dev_indices'],data_root,job,device,
                    folder/'predictions/probe-final.csv')
            metrics=ctx.broadcast(metrics)
        last=metrics
        timing['evaluation_seconds']+=ctx.maximum(time.monotonic()-started)
        log('probe_evaluation',metrics=metrics,ema_metrics=ema_last if ema else None)
    save();result=status('completed',final_checkpoint='lineage_best' if best is not None else 'last')
    if ctx.primary:atomic_json(complete,result)
    log('completed',mode=mode,early_stopped=early['stopped'])
    return result

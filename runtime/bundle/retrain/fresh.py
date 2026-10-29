"""Sealed public-source initialization with a persistent, bounded AMP grace window."""
import math

from .config import job_contract


def resolve_steps_per_cycle(job, negative_count, *, fresh_spec=None) -> int:
    """Registered fresh uses its pinned clock; legacy retains the class sweep.

    The caller still validates the complete fresh schedule and source identity
    with validate_fresh; the sampler's attempted-update stream is unchanged.
    """
    if job.get('experiment91') is not None and fresh_spec is not None:
        from .experiment91 import schedule
        return schedule()['steps_per_cycle']
    return math.ceil(negative_count / (job['effective_batch'] - job['positive_per_batch']))


def validate_fresh(spec, identity, job, topology, steps, *, mode):
    if job.get('experiment91') is not None:
        from .experiment91 import validate_runtime
        validate_runtime(identity,job,fresh=spec)
    expected_scale=65536
    if job['precision']=='fp32':
        from .precision import validate_protocol
        validate_protocol(spec.get('precision_protocol'), job['precision'])
        expected_scale=1.0
    elif spec.get('precision_protocol') is not None:
        raise ValueError('FP32 protocol cannot be attached to a mixed-precision job')
    barrier=spec.get('startup_barrier')
    if spec.get('execution_policy'):
        from .execution_policy import validate_general_fresh
        validate_general_fresh(spec,identity,job,topology)
    if spec.get('cache_readonly_required') and not spec.get('execution_policy'):
        if (topology.get('world_size')!=6 or topology.get('global_effective_batch')!=192
                or topology.get('global_positive_quota')!=24 or topology.get('reference_physical_batch')!=96
                or topology.get('per_rank_batch')!=32 or topology.get('accumulation')!=1
                or steps!=237 or job['cycles']!=24 or job['warmup_cycles']!=2
                or job['stop_after_cycles']!=0 or job['seed']!=20261025 or job['model']!='convnext_tiny'
                or not job['age']['enabled'] or len(spec.get('source_digest',''))!=64
                or spec.get('plan_version')!='9.0-world6-b192'
                or spec.get('plan_sha256')!='a5cd94bd5fb1523e60da303a68de33ee6614b87ba3d72039162802e654978b37'
                or spec.get('schedule',{}).get('stop_after_updates')!=(150 if spec.get('purpose')=='probe' else 5688)):
            raise ValueError('R9 registered six-rank batch/schedule/source contract differs')
        if (not isinstance(barrier,dict) or set(barrier)!={'path','round_id','run_ids'}
                or barrier['round_id']!='r9-hz6-b192-20261025' or len(barrier['run_ids'])!=1
                or barrier['run_ids']!=['r9-6c-a25-b192-ema'+('-smoke' if spec['purpose']=='probe' else '')]
                or not barrier['path'].startswith('/public/home/acfwvv9ejn/SothisAI/exp-v2/')
                or not barrier['path'].endswith('/'+spec['purpose']+'-startup-barrier.json')):
            raise ValueError('R9 named single-model six-rank startup barrier contract absent or incompatible')
    technical=spec.get('execution_policy',{}).get('technical_smoke',False)
    if technical and (mode!='fresh-probe' or spec.get('purpose')!='probe' or not 0<spec.get('schedule',{}).get('stop_after_updates',0)<=30):raise ValueError('engineering probe limited to30 total successful updates')
    interruption=spec.get('interruption_test')
    if interruption is not None and (mode!='fresh-probe' or set(interruption)!={'successful_update'}
            or type(interruption['successful_update']) is not int
            or not (1<=interruption['successful_update']<spec['schedule']['stop_after_updates'] if technical else 30<=interruption['successful_update']<=50)):
        raise ValueError('smoke recovery must be a fixed 30..50 successful boundary')
    if 'ema' in spec:
        from .ema import RULE,validate_rule
        if spec.get('execution_policy'):validate_rule(spec['ema'])
        elif spec['ema']!=RULE:raise ValueError('fresh EMA rule differs')
    schedule=spec.get('schedule',{})
    expected={'steps_per_cycle':steps,'total_updates':job['cycles']*steps,
              'warmup_updates':job['warmup_cycles']*steps,
              'stop_after_updates':topology.get('stop_after_updates',job['cycles']*steps)}
    initialization=identity.get('initialization',{})
    if (spec.get('format')!='glaucoma-fresh-init-v1' or spec.get('purpose') not in ('probe','train')
            or (mode=='fresh-probe')!=(spec.get('purpose')=='probe') or mode not in ('fresh-probe','train')
            or not spec.get('plan_version') or len(spec.get('plan_sha256',''))!=64
            or identity.get('fresh')!=spec or identity.get('phase_parent_sha256') is not None
            or identity.get('stage') or identity.get('continuation')
            or identity.get('source_digest')!=spec.get('source_digest')
            or job_contract(job)!=identity.get('job') or job_contract(job)!=spec.get('job')
            or identity.get('distributed')!=topology or initialization!=spec.get('initialization')
            or initialization.get('kind')!=({'dinov2_vits14':'self_supervised','flair_resnet50':'retinal_vision_language'}.get(job.get('model'),'imagenet'))
            or initialization.get('weights_sha256')!=job.get('initialization',{}).get('sha256')
            or spec.get('amp_initial_scale')!=expected_scale or schedule!=expected
            or not 0<schedule.get('stop_after_updates',0)<=expected['total_updates']
            or (spec['purpose']=='train' and schedule['stop_after_updates']!=expected['total_updates'])):
        raise ValueError('fresh source / purpose / scientific schedule identity rejected')


def wait_start_barrier(spec,ctx,stopping):
    """No sample/optimizer step until every admitted arm's initial proof passed."""
    import json,time
    from pathlib import Path
    barrier=spec.get('startup_barrier')
    if barrier is None:return True  # historical/CPU contracts have no R8 barrier
    expected={'round_id':barrier['round_id'],'plan_sha256':spec['plan_sha256'],
              'source_digest':spec['source_digest'],'purpose':spec['purpose'],'run_ids':barrier['run_ids']}
    world=spec.get('execution_policy',{}).get('world_size',6)
    if spec.get('cache_readonly_required'):expected['world_size']=world
    while True:
        record=None
        if ctx.primary:
            try:record={'value':json.loads(Path(barrier['path']).read_text())}
            except FileNotFoundError:record={'value':None}
            except (OSError,ValueError) as exc:record={'error':str(exc)}
        record=ctx.broadcast(record)
        if record.get('error'):raise ValueError('startup barrier invalid: '+record['error'])
        value=record['value']
        if value is not None:
            if spec.get('cache_readonly_required') and (value.get('initial_rank_count')!=world
                    or len(value.get('rank_initial_proof_sha256',''))!=64):
                raise ValueError('six-rank initial proof absent from startup barrier')
            if (not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected.items())
                    or value.get('all_initials_passed') is not True
                    or set(value.get('initials_sha256',{}))!=set(barrier['run_ids'])
                    or any(len(v)!=64 for v in value['initials_sha256'].values())):
                raise ValueError('startup barrier identity / all-arm proof differs')
            return True
        if stopping():return False
        time.sleep(.1)


class FreshOverflowGuard:
    """The first 50 committed updates get one bounded initial-scale adjustment period."""
    def __init__(self,state=None):
        self.value={'version':1,'successful':0,'initial_skips':0,'total_skips':0,
                    'post_warmup':False,'post_skips':0,'window':[],'consecutive':0}
        if state is not None:
            if set(state)!=set(self.value) or state['version']!=1:
                raise ValueError('fresh AMP state missing or incompatible')
            self.value={**state,'window':list(state['window'])}
            s=self.value
            if (s['post_warmup']!=(s['successful']>=50) or len(s['window'])>200
                    or s['total_skips']!=s['initial_skips']+s['post_skips']
                    or any(s[k]<0 for k in ('successful','initial_skips','total_skips','post_skips','consecutive'))):
                raise ValueError('fresh AMP state counters inconsistent')
    def state(self):return {**self.value,'window':list(self.value['window'])}
    def record(self,skipped):
        s=self.value;skipped=bool(skipped)
        s['consecutive']=s['consecutive']+1 if skipped else 0
        s['total_skips']+=int(skipped)
        if s['post_warmup']:
            s['post_skips']+=int(skipped);s['window']=(s['window']+[skipped])[-200:]
        else:s['initial_skips']+=int(skipped)
        if not skipped:s['successful']+=1
        if s['consecutive']>=3:raise FloatingPointError('fresh AMP: 3 consecutive skipped attempts')
        if s['initial_skips']>8:raise FloatingPointError('fresh AMP: initial 50-success grace exceeded 8 skips')
        if s['post_warmup'] and sum(s['window'])>2:
            raise FloatingPointError('fresh AMP: >2 skips within 200 post-warmup attempts')
        if not s['post_warmup'] and s['successful']==50:
            s['post_warmup']=True;s['window']=[]


def runtime_stop_reason(deadline_at,device=None):
    """Independent of controller liveness; safe boundaries enforce the same persisted deadline."""
    import time
    from pathlib import Path
    if deadline_at is not None and time.time()>=deadline_at:return 'absolute_deadline_reached'
    if deadline_at is None:return None  # legacy/CPU fixtures do not invent a runtime contract
    current=Path('/sys/fs/cgroup/memory.current');maximum=Path('/sys/fs/cgroup/memory.max')
    if current.exists() and maximum.exists() and maximum.read_text().strip().isdigit():
        if int(current.read_text())>.9*int(maximum.read_text()):return 'pod_memory_limit_exceeded'
    if device is not None and device.type=='cuda':
        import torch
        if torch.cuda.memory_reserved(device)>.85*torch.cuda.get_device_properties(device).total_memory:
            return 'rank_vram_limit_exceeded'
    return None


def start_runtime_guard(deadline_at,stop_event,on_stop):
    import threading
    def monitor():
        while not stop_event.wait(.5):
            reason=runtime_stop_reason(deadline_at)
            if reason:
                on_stop(reason);stop_event.set();return
    thread=threading.Thread(target=monitor,name='frozen-runtime-budget',daemon=True);thread.start();return thread

"""Explicit, sealed phase initialization; ordinary resume remains unchanged."""
import copy
from .config import job_contract
from .persistence import load_file,sha256_file

ALLOWED_PHASE_FIELDS={'image_size','cycles','warmup_cycles','min_cycles','patience',
                      'min_delta','backbone_lr','head_lr','stop_after_cycles'}


def validate_stage(parent,spec,identity,job,topology):
    if (spec.get('format')!='glaucoma-stage-init-v1' or spec.get('purpose') not in ('train','benchmark')
            or identity.get('stage')!=spec or identity.get('phase_parent_sha256')!=spec.get('sha256')
            or not spec.get('plan_sha256') or not spec.get('plan_version')):
        raise ValueError('stage specification / identity binding rejected')
    old=parent.get('identity',{})
    if (old!=spec.get('identity') or old.get('source_digest')!=spec.get('parent_source_digest')
            or parent.get('mode')!='train' or parent.get('format_version')!=1
            or old.get('stage',{}).get('purpose')=='benchmark'
            or parent.get('manifest_digest')!=identity.get('manifest_digest')
            or old.get('manifest_digest')!=identity.get('manifest_digest')
            or old.get('initialization')!=identity.get('initialization')
            or job_contract(parent['job'])!=old.get('job')
            or job_contract(job)!=identity.get('job') or job_contract(job)!=spec.get('job')):
        raise ValueError('stage parent data / source / scientific identity rejected')
    for field in set(old['job'])|set(identity['job']):
        if field not in ALLOWED_PHASE_FIELDS and old['job'].get(field)!=identity['job'].get(field):
            raise ValueError('stage changes protected field '+field)
    normalize=lambda p:{k:v for k,v in p.items() if k!='stop_after_updates'}
    if normalize(old.get('distributed',{}))!=normalize(topology) or identity['distributed']!=topology:
        raise ValueError('stage changes DDP topology / BN / sampling policy')
    successful=parent.get('successful_updates',0);attempted=parent.get('attempted_updates',0)
    if (successful<=0 or attempted<successful or successful!=spec.get('parent_successful_updates')
            or attempted!=spec.get('sampler_attempt_offset')
            or not parent.get('model') or not parent.get('optimizer',{}).get('state')
            or len(parent.get('distributed_rng',[]))!=topology['world_size']
            or any(not state for state in parent['distributed_rng'])
            or (job['precision']=='fp16' and not parent.get('scaler'))
            or any(int(v.get('step',-1))!=successful for v in parent['optimizer']['state'].values())):
        raise ValueError('stage parent optimizer / counters / rank RNG / AMP incomplete')
    schedule=spec['schedule'];steps=parent['schedule']['steps_per_cycle']
    if (schedule['steps_per_cycle']!=steps or schedule['total_updates']!=job['cycles']*steps
            or schedule['warmup_updates']!=0 or job['warmup_cycles']!=0
            or not 0<schedule['stop_after_updates']<=schedule['total_updates']
            or (spec['purpose']=='train' and schedule['stop_after_updates']!=schedule['total_updates'])):
        raise ValueError('stage horizon / stopping target rejected')


def load_stage(spec,identity,job,topology):
    if sha256_file(spec['checkpoint'])!=spec['sha256']:raise ValueError('stage parent SHA mismatch')
    payload=load_file(spec['checkpoint'],map_location='cpu')
    validate_stage(payload,spec,identity,job,topology)
    return payload


def initialize_stage(parent,spec,identity,job,topology):
    validate_stage(parent,spec,identity,job,topology)
    result=copy.deepcopy(parent)
    result.update(identity=copy.deepcopy(identity),job=copy.deepcopy(job),successful_updates=0,
        attempted_updates=0,evaluated_cycle=0,cycle=0,coverage={'indices':[]},timing={},
        best_metrics=None,last_metrics=None,audit_metrics=None,consecutive_overflows=0,
        early_stop={'reference_auroc':None,'bad_cycles':0,'stopped':False},
        schedule={**spec['schedule'],'successful_updates':0},stage=copy.deepcopy(spec),overflow_window=[])
    return result


def sampler_start(spec,attempted):
    return spec['sampler_attempt_offset']+attempted


class StageOverflowGuard:
    def __init__(self,state=None):self.history=list(state or [])[-200:]
    def state(self):return list(self.history)
    def record(self,skipped):
        self.history.append(bool(skipped));self.history=self.history[-200:]
        if (len(self.history)>=3 and all(self.history[-3:])) or sum(self.history)>2:
            raise FloatingPointError('stage AMP guard: 3 consecutive or >2 skips within 200 attempts')

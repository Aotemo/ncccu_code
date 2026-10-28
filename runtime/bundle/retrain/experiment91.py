"""Pinned 9:1 metadata contract; never discovers data or regenerates a split."""
import copy
from functools import lru_cache
import json
from pathlib import Path

from .persistence import canonical_digest, sha256_file

ROOT = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/technical-prep/resume9010-A-stage1-20261028-attempt1/result')
FREEZE_SHA = '90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6'
WEIGHT_SHA = '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'


@lru_cache(maxsize=1)
def metadata():
    if sha256_file(ROOT / 'FREEZE.json') != FREEZE_SHA:
        raise ValueError('9:1 FREEZE identity differs')
    freeze = json.loads((ROOT / 'FREEZE.json').read_text())
    values = {}
    for name in ('CONTRACT.json', 'manifest.json', 'AGE.json', 'SCHEDULE.json', 'EMA-R50.json'):
        path = ROOT / name
        if path.is_symlink() or sha256_file(path) != freeze['files'][name]['sha256']:
            raise ValueError('9:1 frozen artifact differs: ' + name)
        values[name] = json.loads(path.read_text())
    c, manifest = values['CONTRACT.json'], values['manifest.json']
    if canonical_digest(manifest) != c['new_manifest_canonical_sha256']:
        raise ValueError('9:1 native manifest digest differs')
    values['binding'] = {'version': 'single-r50-public-v2-fixed-9010-v1', 'freeze_sha256': FREEZE_SHA,
        'manifest_sha256': c['new_manifest']['sha256'], 'manifest_digest': c['new_manifest_canonical_sha256'],
        'fit_indices_sha256': c['fit_indices_canonical_sha256'], 'dev_indices_sha256': c['dev_indices_canonical_sha256'],
        'age_contract_sha256': c['age_contract_sha256'], 'schedule_sha256': c['schedule']['sha256'],
        'ema_sha256': c['ema_r50']['sha256'], 'source_identity': c['source_identity']}
    return values


def binding():
    return copy.deepcopy(metadata()['binding'])


def schedule():
    raw = metadata()['SCHEDULE.json']
    return {k: raw[k] for k in ('steps_per_cycle', 'total_updates', 'warmup_updates', 'stop_after_updates')}


def ema_rule():
    return copy.deepcopy(metadata()['EMA-R50.json'])


def validate_job(job):
    expected = {'model': 'resnet50_imagenet', 'image_size': 512, 'prepared_size': 512,
        'precision': 'fp32', 'normalization': 'imagenet', 'seed': 20261027, 'effective_batch': 64,
        'positive_per_batch': 8, 'cycles': 8, 'warmup_cycles': 2, 'min_cycles': 8, 'patience': 8,
        'stop_after_cycles': 0, 'backbone_lr': 2e-5, 'head_lr': 1e-3, 'weight_decay': 1e-4,
        'dropout': .2, 'stochastic_depth': 0., 'head_lr_scope': 'last_linear', 'mixup': None,
        'attention_pool': None, 'initial_checkpoint': None, 'frozen_bn': True}
    if (any(job.get(k) != v for k, v in expected.items()) or job.get('physical_batch') not in (16, 32)
            or job.get('experiment91') != binding() or job.get('age') != metadata()['AGE.json']
            or job.get('initialization', {}).get('kind') != 'imagenet'
            or job['initialization'].get('sha256') != WEIGHT_SHA):
        raise ValueError('single R50 9:1 scientific/age/source contract differs')
    return job


def accepted_manifest(path, digest):
    expected = metadata()['CONTRACT.json']['new_manifest']
    if (not path or Path(path).resolve() != Path(expected['path']).resolve()
            or digest != expected['sha256'] or sha256_file(path) != digest):
        raise ValueError('exact accepted 9:1 manifest required; no allocation fallback')
    return copy.deepcopy(metadata()['manifest.json'])


def validate_runtime(identity, job, manifest=None, fresh=None):
    from .config import job_contract
    validate_job(job)
    if (identity.get('experiment91') != binding() or identity.get('manifest_digest') != binding()['manifest_digest']
            or identity.get('job') != job_contract(job) or len(identity.get('source_digest', '')) != 64
            or identity.get('initialization', {}).get('weights_sha256') != WEIGHT_SHA
            or identity.get('phase_parent_sha256') is not None):
        raise ValueError('9:1 runtime source/job/manifest identity differs')
    if manifest is not None and canonical_digest(manifest) != binding()['manifest_digest']:
        raise ValueError('9:1 runtime manifest changed')
    spec = fresh if fresh is not None else identity.get('fresh')
    if (not spec or spec.get('experiment91') != binding() or spec.get('schedule') != schedule()
            or spec.get('ema') != ema_rule()):
        raise ValueError('9:1 fresh schedule/EMA/binding differs')
    return True


def validate_checkpoint(payload):
    from .age import validate_checkpoint as validate_age
    job = payload['job']
    validate_runtime(payload['identity'], job, fresh=payload.get('fresh'))
    validate_age(payload, job)
    if (payload.get('manifest_digest') != binding()['manifest_digest'] or payload.get('mode') != 'train'
            or any(payload.get('schedule', {}).get(k) != v for k, v in schedule().items())
            or payload.get('ema', {}).get('rule') != ema_rule()):
        raise ValueError('9:1 checkpoint schedule/variant data contract differs')
    return True

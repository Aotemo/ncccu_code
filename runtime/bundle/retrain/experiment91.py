"""Explicit Tiny22K-to-1K source over the unchanged fixed 9:1 metadata."""
import copy
from functools import lru_cache
import json
from pathlib import Path

from .persistence import canonical_digest, sha256_file
from .site_binding import load_site

FREEZE_SHA = '90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6'
WEIGHT_SHA = '1ec1dd0b7045237eafe1c8f1becd58456cc811326c97ec416ae8b2e5ede860ed'
PROVENANCE_SHA = '40c14c8bd780d4feb88ae96d7e9def1ef78499daf93ae4412eff78502780233f'
BINDING_VERSION = 'single-tiny22k-bc48a87f-fixed-9010-v1'
CANDIDATE = {'id': 'r93-convnext-tiny-imagenet22k-to-1k-9010', 'experiment_order': 3,
    'model': 'convnext_tiny', 'public_revision': 'bc48a87f119bae1b06418f5340bfa97ef75d4609',
    'readout': 'native_avgpool_final_LN_dropout_linear_plus_age', 'features': 768,
    'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6, 'batch_norm_modules': 0,
    'batch_norm_buffers': 0, 'attention_pool': None, 'weight_sha256': WEIGHT_SHA,
    'provenance_sha256': PROVENANCE_SHA, 'stochastic_depth': .1,
    'head_initialization': 'torch.nn.Linear_native', 'head_lr_scope': 'last_linear'}


def candidate():
    site = load_site()
    if (site['science'].get('candidate') != CANDIDATE
            or site['science'].get('binding_version') != BINDING_VERSION):
        raise ValueError('explicit candidate/source/LN scientific identity differs')
    return copy.deepcopy(CANDIDATE)


def metadata():
    candidate()
    return _metadata(load_site()['metadata_root'])


@lru_cache(maxsize=4)
def _metadata(root):
    root = Path(root)
    if (root / 'FREEZE.json').is_symlink() or sha256_file(root / 'FREEZE.json') != FREEZE_SHA:
        raise ValueError('9:1 FREEZE identity differs')
    freeze = json.loads((root / 'FREEZE.json').read_text())
    values = {}
    for name in ('CONTRACT.json', 'manifest.json', 'AGE.json', 'SCHEDULE.json'):
        path = root / name
        if path.is_symlink() or sha256_file(path) != freeze['files'][name]['sha256']:
            raise ValueError('9:1 frozen artifact differs: ' + name)
        values[name] = json.loads(path.read_text())
    c = values['CONTRACT.json']
    if canonical_digest(values['manifest.json']) != c['new_manifest_canonical_sha256']:
        raise ValueError('9:1 native manifest digest differs')
    warm = values['SCHEDULE.json']['warmup_updates']
    values['EMA-TINY.json'] = {'version': f'ema-warmcopy{warm}-decay0999-nobn-v1',
        'warmcopy_updates': warm, 'decay': .999, 'dtype': 'torch.float32',
        'floating_buffers': 'same_as_parameters', 'integer_buffers': 'copy', 'batch_norm_buffers': 0}
    values['binding'] = {'version': BINDING_VERSION, 'candidate': candidate(),
        'freeze_sha256': FREEZE_SHA, 'manifest_sha256': c['new_manifest']['sha256'],
        'manifest_digest': c['new_manifest_canonical_sha256'], 'fit_indices_sha256': c['fit_indices_canonical_sha256'],
        'dev_indices_sha256': c['dev_indices_canonical_sha256'], 'age_contract_sha256': c['age_contract_sha256'],
        'schedule_sha256': c['schedule']['sha256'], 'ema_sha256': canonical_digest(values['EMA-TINY.json']),
        'source_identity': c['source_identity']}
    return values


def binding():
    return copy.deepcopy(metadata()['binding'])


def schedule():
    raw = metadata()['SCHEDULE.json']
    return {k: raw[k] for k in ('steps_per_cycle', 'total_updates', 'warmup_updates', 'stop_after_updates')}


def ema_rule():
    return copy.deepcopy(metadata()['EMA-TINY.json'])


def validate_job(job):
    expected = {'model': 'convnext_tiny', 'image_size': 512, 'prepared_size': 512,
        'precision': 'fp32', 'normalization': 'imagenet', 'seed': 20261027, 'effective_batch': 64,
        'positive_per_batch': 8, 'cycles': 8, 'warmup_cycles': 2, 'min_cycles': 8, 'patience': 8,
        'stop_after_cycles': 0, 'backbone_lr': 2e-4, 'head_lr': 1e-3, 'weight_decay': 1e-4,
        'dropout': .2, 'stochastic_depth': .1, 'head_lr_scope': 'last_linear', 'mixup': None,
        'attention_pool': None, 'initial_checkpoint': None, 'frozen_bn': False, 'final_norm_eps': candidate()['final_norm_eps']}
    init = job.get('initialization', {})
    site = load_site()
    if (any(job.get(k) != v for k, v in expected.items()) or job.get('physical_batch') not in (64 // len(site['gpus']), 32)
            or job.get('experiment91') != binding() or job.get('age') != metadata()['AGE.json']
            or init.get('kind') != 'imagenet' or init.get('sha256') != WEIGHT_SHA
            or init.get('format') != 'torchvision-1000-state-dict-v1'
            or Path(init.get('path', '')).resolve() != Path(site['public_weight']).resolve()
            or Path(init.get('provenance', '')).resolve() != Path(site['public_provenance']).resolve()):
        raise ValueError('Tiny22K-to-1K scientific/age/source/LN contract differs')
    return job


def describe_model(model):
    import torch
    from .ema import tensor_digest
    nn = torch.nn
    network = model.network
    final = network.classifier[-1]
    norms = [{'name': n, 'eps': float(m.eps), 'normalized_shape': list(m.normalized_shape)}
             for n, m in model.named_modules() if isinstance(m, nn.LayerNorm)]
    buffers = dict(model.named_buffers())
    from torchvision.ops import StochasticDepth
    depth = [{'name': n, 'p': m.p, 'mode': m.mode} for n, m in model.named_modules() if isinstance(m, StochasticDepth)]
    count_bn = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if (count_bn != 0 or buffers or not norms
            or not isinstance(network.avgpool, nn.AdaptiveAvgPool2d)
            or network.avgpool.output_size != 1
            or not isinstance(final, nn.Linear) or (final.in_features, final.out_features) != (768, 1)
            or not isinstance(network.classifier[-2], nn.Dropout) or network.classifier[-2].p != .2
            or any(n == 'attention_pool_weight' or n == 'q' for n, _ in model.named_parameters())
            or not any(n['name'] == 'network.classifier.0' for n in norms)
            or any(n['eps'] != (candidate()['final_norm_eps'] if n['name'] == 'network.classifier.0' else 1e-6) for n in norms)):
        raise ValueError('actual Tiny native avgpool/768/noq/noBN/LN/dropout differs')
    result = {'candidate': candidate(), 'batch_norm_modules': count_bn, 'buffer_count': len(buffers),
            'buffer_sha256': tensor_digest(buffers), 'layer_norms': norms, 'head_in_features': final.in_features,
            'head_out_features': final.out_features, 'dropout': network.classifier[-2].p, 'stochastic_depth': depth}
    validate_normalization(result)
    return result


def validate_normalization(value):
    import hashlib
    norms, depth = value.get('layer_norms', []), value.get('stochastic_depth', [])
    if (value.get('candidate') != candidate() or value.get('batch_norm_modules') != 0
            or value.get('buffer_count') != 0 or value.get('buffer_sha256') != hashlib.sha256(b'').hexdigest()
            or value.get('head_in_features') != 768 or value.get('head_out_features') != 1 or value.get('dropout') != .2
            or not norms or len({n['name'] for n in norms}) != len(norms)
            or not any(n['name'] == 'network.classifier.0' for n in norms)
            or any(n['eps'] != (candidate()['final_norm_eps'] if n['name'] == 'network.classifier.0' else 1e-6) for n in norms)
            or len(depth) < 2 or any(d['mode'] != 'row' or d['p'] != .1 * i / (len(depth)-1) for i, d in enumerate(depth))):
        raise ValueError('Tiny actual LN/noBN/stochastic-depth contract differs')
    return True


def accepted_manifest(path, digest):
    expected = metadata()['CONTRACT.json']['new_manifest']
    if (not path or Path(path).resolve() != (Path(load_site()['metadata_root']) / 'manifest.json').resolve()
            or digest != expected['sha256'] or sha256_file(path) != digest):
        raise ValueError('exact accepted 9:1 manifest required; no allocation fallback')
    return copy.deepcopy(metadata()['manifest.json'])


def validate_runtime(identity, job, manifest=None, fresh=None):
    from .config import job_contract
    validate_job(job)
    init = identity.get('initialization', {})
    if (identity.get('experiment91') != binding() or identity.get('manifest_digest') != binding()['manifest_digest']
            or identity.get('job') != job_contract(job) or len(identity.get('source_digest', '')) != 64
            or init.get('weights_sha256') != WEIGHT_SHA or init.get('provenance_sha256') != PROVENANCE_SHA
            or init.get('final_norm_eps') != candidate()['final_norm_eps'] or identity.get('phase_parent_sha256') is not None):
        raise ValueError('Tiny22K-to-1K runtime source/job/manifest/LN differs')
    if manifest is not None and canonical_digest(manifest) != binding()['manifest_digest']:
        raise ValueError('9:1 runtime manifest changed')
    spec = fresh if fresh is not None else identity.get('fresh')
    if (not spec or spec.get('experiment91') != binding() or spec.get('schedule') != schedule()
            or spec.get('ema') != ema_rule()):
        raise ValueError('Tiny fresh schedule/noBN EMA/binding differs')
    return True


def validate_checkpoint(payload):
    from .age import validate_checkpoint as validate_age
    job = payload['job']
    validate_runtime(payload['identity'], job, fresh=payload.get('fresh'))
    validate_age(payload, job)
    validate_normalization(payload.get('candidate_normalization', {}))
    if (payload.get('manifest_digest') != binding()['manifest_digest'] or payload.get('mode') != 'train'
            or any(payload.get('schedule', {}).get(k) != v for k, v in schedule().items())
            or payload.get('ema', {}).get('rule') != ema_rule()):
        raise ValueError('Tiny checkpoint schedule/candidate/noBN contract differs')
    return True

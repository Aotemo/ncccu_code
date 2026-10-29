"""Explicit ConvNeXt22K-to-1K source over unchanged fixed 9:1 metadata."""
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from .persistence import canonical_digest, sha256_file
from .site_binding import load_site

FREEZE_SHA = '90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6'
BINDING_VERSION = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-v1'
CANDIDATE = {'id': 'r103-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-9010', 'experiment_order': 13,
    'model': 'convnext_small', 'public_revision': 'fe6b358aa409414cdb522ae3d43a2c9922f7cc1b',
    'readout': 'native_avgpool_final_LN_dropout_linear_plus_age', 'features': 768,
    'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6, 'batch_norm_modules': 0,
    'batch_norm_buffers': 0, 'attention_pool': None, 'stochastic_depth': .1,
    'head_initialization': 'torch.nn.Linear_native', 'head_lr_scope': 'last_linear'}

def science_configuration():
    """Read the exact versioned science file pinned by this package's site."""
    site = load_site()
    try:
        reference = site['science']['configuration']
        path = Path(reference['path'])
        if (not path.is_absolute() or path.is_symlink()
                or not path.resolve().is_relative_to(Path(site['work_root']).resolve())):
            raise ValueError('pinned science configuration path/SHA differs')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('pinned science configuration path/SHA differs')
        value = json.loads(raw)
        if (type(value.get('schema_version')) is not int or value['schema_version'] != 1
                or value.get('candidate_profile') != BINDING_VERSION
                or any(type(value['job'].get(key)) not in (int, float)
                       or not math.isfinite(value['job'][key]) or value['job'][key] <= 0
                       for key in ('backbone_lr', 'head_lr'))):
            raise ValueError('science configuration profile or configured LR differs')
        return value
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('complete pinned science configuration required') from error


def candidate_from_configuration(value):
    actual = value['candidate']
    source = value['public_source']
    if (any(actual.get(key) != expected for key, expected in CANDIDATE.items())
            or actual['model'] != value['job']['model']
            or actual['weight_sha256'] != source['sha256']
            or actual['provenance_sha256'] != source['provenance_sha256']
            or actual['public_revision'] != source['revision']
            or actual['features'] != value['structure']['features']):
        raise ValueError('candidate, public source and measured structure differ')
    return copy.deepcopy(actual)


def public_source():
    return copy.deepcopy(science_configuration()['public_source'])


def structure_contract():
    return copy.deepcopy(science_configuration()['structure'])


def verify_public_asset(science, site):
    """Consume the new public source handoff without altering historical migration."""
    reference = science.get('public_asset_receipt')
    if reference is None:
        if science['job']['model'] == 'convnext_small':
            raise ValueError('Small requires its independently verified public asset')
        return None
    relative = Path(reference['path'])
    root = Path(site['work_root']).resolve()
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('public handoff must be relative to work_root')
    path = root / relative
    if path.is_symlink() or sha256_file(path) != reference['sha256']:
        raise ValueError('public handoff bytes differ')
    receipt = json.loads(path.read_text())
    source = science['public_source']
    if (receipt.get('passed') is not True or receipt.get('schema_version') != 1
            or receipt['execution']['uid'] != site['pod']['uid']
            or receipt['model'] != science['job']['model']
            or receipt['revision'] != source['revision']
            or receipt['converted'] != {'path': site['public_weight'], 'sha256': source['sha256']}
            or receipt['provenance'] != {'path': site['public_provenance'], 'sha256': source['provenance_sha256']}
            or receipt['original']['sha256'] != source['original_sha256']
            or receipt['structure'] != science['structure']):
        raise ValueError('public source, paths or measured structure differ from handoff')
    return receipt


def candidate():
    site = load_site()
    expected = candidate_from_configuration(science_configuration())
    if (site['science'].get('candidate') != expected
            or site['science'].get('binding_version') != BINDING_VERSION):
        raise ValueError('explicit candidate/source/LN scientific identity differs')
    return expected


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
    warm = schedule()['warmup_updates']
    values['EMA-TINY.json'] = {'version': f'ema-warmcopy{warm}-decay0999-nobn-v1',
        'warmcopy_updates': warm, 'decay': .999, 'dtype': 'torch.float32',
        'floating_buffers': 'same_as_parameters', 'integer_buffers': 'copy', 'batch_norm_buffers': 0}
    values['binding'] = {'version': BINDING_VERSION, 'candidate': candidate(),
        'science_config_sha256': load_site()['science']['configuration']['sha256'],
        'freeze_sha256': FREEZE_SHA, 'manifest_sha256': c['new_manifest']['sha256'],
        'manifest_digest': c['new_manifest_canonical_sha256'], 'fit_indices_sha256': c['fit_indices_canonical_sha256'],
        'dev_indices_sha256': c['dev_indices_canonical_sha256'], 'age_contract_sha256': c['age_contract_sha256'],
        'schedule_sha256': c['schedule']['sha256'], 'ema_sha256': canonical_digest(values['EMA-TINY.json']),
        'source_identity': c['source_identity']}
    return values


def binding():
    return copy.deepcopy(metadata()['binding'])


def schedule():
    raw = science_configuration()['schedule']
    return {k: raw[k] for k in ('steps_per_cycle', 'total_updates', 'warmup_updates', 'stop_after_updates')}


def early_stopping():
    return copy.deepcopy(science_configuration()['early_stopping'])


def validate_training_completion(record):
    """A full update budget or a completed patience boundary may freeze selection."""
    configured = science_configuration()
    planned = schedule()
    updates = record['successful_updates']
    cycle = record['evaluated_cycle']
    early = record.get('early_stop', {})
    if (record.get('status') != 'completed' or record.get('skipped_updates') != 0
            or record.get('attempted_updates') != updates
            or record.get('cycle') != cycle or updates != cycle * planned['steps_per_cycle']
            or not 0 < updates <= planned['total_updates']):
        raise ValueError('completed unskipped internal-dev boundary required before selection')
    if updates < planned['total_updates'] and not (
            early.get('stopped') is True and cycle >= configured['job']['min_cycles']
            and early.get('bad_cycles', 0) >= configured['job']['patience']):
        raise ValueError('training ended before its budget without completed early stopping')
    return cycle


def ema_rule():
    return copy.deepcopy(metadata()['EMA-TINY.json'])


def validate_job(job):
    configured = science_configuration()['job']
    expected = {'model': candidate()['model'], 'image_size': 512, 'prepared_size': 512,
        'precision': 'fp32', 'normalization': 'imagenet', 'seed': configured['seed'], 'effective_batch': 64,
        'positive_per_batch': 8, **{key: configured[key] for key in (
            'cycles', 'warmup_cycles', 'min_cycles', 'patience', 'min_delta', 'stop_after_cycles')},
        'backbone_lr': configured['backbone_lr'],
        'head_lr': configured['head_lr'], 'weight_decay': 1e-4, 'loss': configured.get('loss'),
        'dropout': configured['dropout'], 'stochastic_depth': .1, 'head_lr_scope': 'last_linear', 'mixup': None,
        'attention_pool': None, 'initial_checkpoint': None, 'frozen_bn': False, 'final_norm_eps': candidate()['final_norm_eps']}
    init = job.get('initialization', {})
    site = load_site()
    if (any(job.get(k) != v for k, v in expected.items()) or job.get('physical_batch') not in (64 // len(site['gpus']), 32)
            or job.get('experiment91') != binding() or job.get('age') != metadata()['AGE.json']
            or init.get('kind') != 'imagenet' or init.get('sha256') != public_source()['sha256']
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
            or not isinstance(final, nn.Linear) or (final.in_features, final.out_features) != (candidate()['features'], 1)
            or not isinstance(network.classifier[-2], nn.Dropout)
            or network.classifier[-2].p != science_configuration()['job']['dropout']
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
            or value.get('head_in_features') != candidate()['features'] or value.get('head_out_features') != 1
            or value.get('dropout') != science_configuration()['job']['dropout']
            or len(norms) != structure_contract()['layer_norms'] or len({n['name'] for n in norms}) != len(norms)
            or not any(n['name'] == 'network.classifier.0' for n in norms)
            or any(n['eps'] != (candidate()['final_norm_eps'] if n['name'] == 'network.classifier.0' else 1e-6) for n in norms)
            or len(depth) != structure_contract()['stochastic_depth_modules']
            or any(d['mode'] != 'row' or d['p'] != science_configuration()['job']['stochastic_depth'] * i / (len(depth)-1) for i, d in enumerate(depth))):
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
            or init.get('weights_sha256') != public_source()['sha256']
            or init.get('provenance_sha256') != public_source()['provenance_sha256']
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

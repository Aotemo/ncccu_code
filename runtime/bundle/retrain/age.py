"""R8 age-linear-v1: immutable fit-only metadata and two FP32 residual weights."""
import copy
import math
import numpy as np
from .persistence import canonical_digest

VERSION = 'age-linear-v1'


def parse_age(raw):
    if raw is None or raw == '':
        return {'value': None, 'missing': True, 'reason': 'empty'}
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return {'value': None, 'missing': True, 'reason': 'unparseable'}
    if isinstance(raw, bool) or not math.isfinite(value):
        return {'value': None, 'missing': True, 'reason': 'nonfinite_or_boolean'}
    if not 0 < value <= 120:
        return {'value': None, 'missing': True, 'reason': 'outside_0_120'}
    return {'value': value, 'missing': False, 'reason': None}


def age_bin(raw):
    value = parse_age(raw)
    if value['missing']:
        return 'missing'
    age = value['value']
    return 'under40' if age < 40 else '40to59' if age < 60 else '60to79' if age < 80 else '80plus'


def make_spec(manifest):
    rows, indices = manifest['rows'], manifest['fit_indices']
    if not indices or len(indices) != len(set(indices)):
        raise ValueError('age statistics require nonempty unique fit indices')
    if any('age' not in row for row in rows):
        raise ValueError('age field missing: manifest schema error')
    values = [parse_age(rows[i]['age']) for i in indices]
    valid = np.asarray([v['value'] for v in values if not v['missing']], dtype=np.float64)
    if not len(valid) or not float(valid.std(ddof=0)) > 0:
        raise ValueError('age fit needs finite nonzero population variance')
    contract = {'version': VERSION, 'field': 'age', 'unit_convention': 'years_not_independently_documented',
                'valid_min_exclusive': 0, 'valid_max_inclusive': 120,
                'features': ['standardized_age', 'missing_indicator'],
                'fit_count': len(indices), 'valid_fit_count': len(valid),
                'missing_fit_count': len(indices)-len(valid), 'median': float(np.median(valid)),
                'mean': float(valid.mean()), 'population_std': float(valid.std(ddof=0)),
                'fit_indices_sha256': canonical_digest(indices),
                'fit_rows_identity_sha256': canonical_digest([
                    {'name':rows[i]['name'], 'sha256':rows[i]['sha256'], 'age':rows[i]['age']} for i in indices]),
                'source_identity': copy.deepcopy(manifest.get('source_identity', {})),
                'lr': 1e-3, 'weight_decay': 1e-4, 'parameter_count': 2, 'bias': False}
    return {'enabled': True, 'contract': contract, 'contract_sha256': canonical_digest(contract)}


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {'enabled','contract','contract_sha256'} or type(spec['enabled']) is not bool:
        raise ValueError('age enabled/contract schema differs')
    c = spec['contract']
    if (canonical_digest(c) != spec['contract_sha256'] or c.get('version') != VERSION or c.get('field') != 'age'
            or c.get('valid_min_exclusive') != 0 or c.get('valid_max_inclusive') != 120
            or c.get('features') != ['standardized_age','missing_indicator']
            or c.get('parameter_count') != 2 or c.get('bias') is not False
            or c.get('lr') != 1e-3 or c.get('weight_decay') != 1e-4
            or c.get('valid_fit_count',0)+c.get('missing_fit_count',0) != c.get('fit_count')
            or any(not isinstance(c.get(k),(int,float)) or not math.isfinite(c[k]) for k in ('median','mean','population_std'))
            or c['population_std'] <= 0):
        raise ValueError('age rule/statistics/hash incompatible')
    return spec


def enabled(job):
    return bool(job.get('age') and job['age']['enabled'])


def features(row, spec):
    if 'age' not in row:
        raise ValueError('age field missing: sample schema error')
    parsed = parse_age(row['age']); c = spec['contract']
    value = c['median'] if parsed['missing'] else parsed['value']
    return [(value-c['mean'])/c['population_std'], float(parsed['missing'])]


def attach_age(model, spec):
    import torch
    validate_spec(spec)
    model.age_metadata = copy.deepcopy(spec)
    if spec['enabled']:
        # nn.Linear would consume RNG even when subsequently zeroed.
        model.age_weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    return model


def residual(model, image_logit, age_features):
    import torch
    if age_features is None or age_features.ndim != 2 or age_features.shape != (image_logit.numel(),2):
        raise ValueError('age features missing or not aligned with image batch')
    if model.age_weight.dtype != torch.float32 or not bool(torch.isfinite(age_features).all()):
        raise ValueError('age features/weights dtype or finite contract failed')
    with torch.autocast(device_type=image_logit.device.type, enabled=False):
        age_features = age_features.to(image_logit.device, dtype=torch.float32)
        return image_logit.reshape(-1).float() + (age_features * model.age_weight).sum(dim=1)


def forward(model, images, age_features, job, *, mixstyle_context=None):
    if job.get('mixstyle_policy') is not None:
        return (model(images, age_features, mixstyle_context=mixstyle_context) if enabled(job)
                else model(images, mixstyle_context=mixstyle_context))
    return model(images, age_features) if enabled(job) else model(images)


def unpack_batch(batch, job, device):
    import torch
    expected = 4 if enabled(job) else 3
    if len(batch) != expected:
        raise ValueError('age/image batch schema does not match scientific job')
    images, labels, indices = batch[:3]
    age_features = batch[3].to(device, dtype=torch.float32, non_blocking=True) if expected == 4 else None
    return images.to(device, non_blocking=True), labels, indices, age_features


def validate_checkpoint(payload, job):
    spec = job.get('age')
    if spec is not None:
        validate_spec(spec)
        if payload.get('age') != spec:
            raise ValueError('checkpoint age metadata absent or differs from resolved job')
    elif payload.get('age') is not None:
        raise ValueError('age checkpoint cannot resume into an image-only legacy job')


def validate_model(model, job):
    spec = job.get('age')
    if spec is not None:
        validate_spec(spec)
        if getattr(model,'age_metadata',None) != spec or hasattr(model,'age_weight') != spec['enabled']:
            raise ValueError('model age metadata/parameter registration differs')

"""Content attention pooling with an exact zero-initialized average-pool residual."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

ATTENTION_POOL_RULE = {
    'version': 'content-attention-pool-v1', 'kind': 'centered_softmax_weighted_mean',
    'channels': 768, 'score_normalization': 'per_location_nonaffine_layer_norm',
    'norm_eps': 1e-6, 'score_scale': 'inverse_sqrt_channels', 'temperature': 1.0,
    'parameter_name': 'attention_pool_weight', 'parameter_shape': [1, 768],
    'weight_init': 'zeros', 'bias': False, 'lr': .001, 'weight_decay': .0001,
    'dropout': 0.0, 'dtype': 'float32',
    'pooling_implementation': 'original_avg_plus_centered_weighted_sum',
    'position_encoding': False}


def validate_spec(spec):
    if spec is None:
        return None
    if not isinstance(spec, dict) or spec != ATTENTION_POOL_RULE:
        raise ValueError('attention pooling differs from the frozen content pooling protocol')
    return spec


def attach(model, job):
    spec = validate_spec(job.get('attention_pool'))
    if spec is None:
        return
    if job['model'] not in ('convnext_tiny', 'convnext_small') or job['precision'] != 'fp32':
        raise ValueError('content attention pooling requires a registered FP32 768-wide model')
    reference = next(model.network.parameters())
    # Allocation consumes no RNG; all original parameters retain their initialization.
    model.attention_pool_weight = nn.Parameter(torch.zeros(1, 768, device=reference.device, dtype=torch.float32))
    model.attention_pool_spec = copy.deepcopy(spec)
    validate_model(model, job)


def validate_model(model, job):
    spec = validate_spec(job.get('attention_pool'))
    weight = getattr(model, 'attention_pool_weight', None)
    if spec is None:
        if weight is not None:
            raise ValueError('attention pooling parameter present without its protocol')
        return
    if (not isinstance(weight, nn.Parameter) or tuple(weight.shape) != (1, 768)
            or weight.dtype != torch.float32 or model.attention_pool_spec != spec
            or job['head_lr'] != spec['lr'] or job['weight_decay'] != spec['weight_decay']):
        raise ValueError('attention pooling parameter shape/dtype/optimizer protocol differs')


def pool_features(model, features):
    if features.ndim != 4 or features.shape[1] != 768 or min(features.shape[2:]) < 1:
        raise ValueError('attention pooling expects nonempty Bx768xHxW features')
    locations = features.flatten(2).transpose(1, 2)
    normalized = F.layer_norm(locations, (768,), eps=1e-6)
    scores = (normalized * model.attention_pool_weight).sum(-1) / (768 ** .5)
    weights = torch.softmax(scores, dim=1)
    centered = weights - (1.0 / locations.shape[1])
    correction = (features.flatten(2) * centered.unsqueeze(1)).sum(2)[:, :, None, None]
    # Do not bypass the q=0 branch: its value is zero but its q gradient is not.
    return model.network.avgpool(features) + correction


def image_logits(model, images):
    features = model.network.features(images)
    return model.network.classifier(pool_features(model, features)).flatten()


def _validate_state(state, spec):
    weight = state.get('attention_pool_weight')
    if spec is None:
        if weight is not None:
            raise ValueError('checkpoint attention pooling parameter has no protocol')
    elif (not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (1, 768)
          or weight.dtype != torch.float32 or not bool(torch.isfinite(weight).all())):
        raise ValueError('checkpoint attention pooling parameter missing or shape/dtype/finite mismatch')


def validate_resume(previous, job, identity):
    spec = validate_spec(job.get('attention_pool'))
    contexts = [identity, previous.get('identity', {})]
    jobs = [previous.get('job', {})] + [context.get('job', {}) for context in contexts]
    jobs += [context['fresh'].get('job', {}) for context in contexts if context.get('fresh') is not None]
    if previous.get('fresh') is not None:
        jobs.append(previous['fresh'].get('job', {}))
    if any(validate_spec(item.get('attention_pool')) != spec for item in jobs):
        raise ValueError('resume attention pooling protocol missing or changed')
    _validate_state(previous.get('model', {}), spec)
    if previous.get('ema') is not None:
        _validate_state(previous['ema']['model'], spec)


def describe(model, optimizer=None):
    weight = model.attention_pool_weight
    record = {'protocol': copy.deepcopy(model.attention_pool_spec), 'shape': list(weight.shape),
              'numel': weight.numel(), 'dtype': str(weight.dtype),
              'l2_norm': float(torch.linalg.vector_norm(weight.detach())),
              'nonzero': int(torch.count_nonzero(weight.detach()))}
    if optimizer is not None:
        state = optimizer.state.get(weight, {})
        record['Adam_step'] = int(state['step']) if 'step' in state else 0
        record['Adam_state_entries'] = sorted(state)
        for name in ('exp_avg', 'exp_avg_sq'):
            record[name + '_norm'] = float(torch.linalg.vector_norm(state[name])) if name in state else None
    return record

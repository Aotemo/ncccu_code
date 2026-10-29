"""Explicit binary losses and WORLD pairwise gather; validation imports no Torch."""
import math


def validate_loss(specification):
    if specification is None:
        return None
    if isinstance(specification, dict) and specification.get('kind') == 'binary_pairwise_logistic':
        fields = {'version', 'kind', 'temperature', 'reduction', 'dtype', 'scope'}
        if (set(specification) != fields
                or specification['version'] != 'binary-pairwise-logistic-v1'
                or specification['reduction'] != 'mean' or specification['dtype'] != 'float32'
                or specification['scope'] != 'global_batch'):
            raise ValueError('pairwise loss requires the explicit binary-pairwise-logistic-v1 contract')
        temperature = specification['temperature']
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature != 1.0:
            raise ValueError('binary pairwise logistic temperature must be 1.0')
        return specification
    fields = {'version', 'kind', 'gamma', 'alpha', 'reduction', 'dtype',
              'modulation_detached'}
    if (not isinstance(specification, dict) or set(specification) != fields
            or specification['version'] != 'binary-focal-v1'
            or specification['kind'] != 'binary_focal'
            or specification['alpha'] is not None
            or specification['reduction'] != 'mean'
            or specification['dtype'] != 'float32'
            or specification['modulation_detached'] is not False):
        raise ValueError('loss must be None or the explicit binary-focal-v1 contract')
    gamma = specification['gamma']
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError('binary focal gamma must be finite and nonnegative')
    return specification


def classification_loss(logits, targets, specification=None):
    specification = validate_loss(specification)
    import torch
    from torch.nn import functional as F

    if specification is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    if specification['kind'] == 'binary_pairwise_logistic':
        logits, targets = logits.float().reshape(-1), targets.reshape(-1)
        positives, negatives = logits[targets == 1], logits[targets == 0]
        if (not positives.numel() or not negatives.numel()
                or positives.numel() + negatives.numel() != logits.numel()):
            raise ValueError('pairwise loss needs binary targets with both classes in the complete batch')
        differences = negatives[None, :] - positives[:, None]
        return F.softplus(differences / specification['temperature']).mean()
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    gamma = float(specification['gamma'])
    if gamma == 0:
        return bce.mean()
    modulation = (1. - torch.exp(-bce)).pow(gamma)
    return (bce * modulation).mean()


def gather_pairwise_inputs(logits, targets):
    """Gather a complete batch on WORLD, preserving logits' autograd edges.

    Every rank evaluates the same global mean. Differentiable gather's backward
    SUM and default WORLD DDP's gradient average cancel; no world-size factor is
    applied here or in the loss. Labels never carry gradients.
    """
    import torch
    import torch.distributed as dist
    from torch.distributed.nn.functional import all_gather

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError('pairwise gather requires an initialized default WORLD process group')
    logits = logits.float().reshape(-1).contiguous()
    targets = targets.detach().reshape(-1).contiguous()
    global_logits = torch.cat(all_gather(logits), dim=0)
    target_parts = [torch.empty_like(targets) for _ in range(dist.get_world_size())]
    dist.all_gather(target_parts, targets)
    return global_logits, torch.cat(target_parts, dim=0)

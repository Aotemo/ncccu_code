"""Default BCE and the explicit binary focal contract; validation imports no Torch."""
import math


def validate_loss(specification):
    if specification is None:
        return None
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
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    gamma = float(specification['gamma'])
    if gamma == 0:
        return bce.mean()
    modulation = (1. - torch.exp(-bce)).pow(gamma)
    return (bce * modulation).mean()

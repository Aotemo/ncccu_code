"""Explicit binary losses and WORLD pairwise gather; validation imports no Torch."""
import math
import numbers

OUTPUT_REGULARIZATION_RULE = {
    'version': 'global-raw-logit-square01-v1', 'coefficient': .1,
    'dtype': 'float32', 'scope': 'global_batch', 'reduction': 'mean',
    'logits': 'final_image_plus_age_raw', 'schedule': 'constant_from_step1',
}


PAIR_HARD_POSITIVE_LOSS = {
    'version': 'binary-pairwise-logistic-positive-top2-mix05-v1',
    'kind': 'binary_pairwise_logistic_hard_positive', 'temperature': 1.0,
    'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
    'top_positive_count': 2, 'hard_positive_weight': .5,
    'tie_break': 'WORLD_positive_order_stable',
}
PAIR_AUC_MARGIN_LOSS = {
    'version': 'binary-pairwise-batch-auc-margin1-v1', 'kind': 'binary_pairwise_auc_margin',
    'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
}
PAIR_BCE_MIX_LOSS = {
    'version': 'binary-pairwise-logistic-bce-mix05-v1', 'kind': 'binary_pairwise_logistic_bce',
    'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
    'pointwise_weight': .5,
}
PAIR_WIDE_SIGMOID_LOSS = {
    'version': 'binary-pairwise-sigmoid-wide2-peakgrad-v1', 'kind': 'binary_pairwise_sigmoid_wide',
    'temperature': 2.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
}
PAIR_SIGMOID_OHEM_LOSS = {
    'version': 'binary-pairwise-sigmoid-ohem-top25-v1', 'kind': 'binary_pairwise_sigmoid_ohem',
    'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
    'hard_fraction': 0.25, 'tie_break': 'WORLD_pair_row_major_stable',
}
PAIR_HARD_NEGATIVE_LOSS = {
    'version': 'binary-pairwise-logistic-negative-top14-mix05-v1',
    'kind': 'binary_pairwise_logistic_hard_negative', 'temperature': 1.0,
    'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
    'top_negative_count': 14, 'hard_negative_weight': .5,
    'tie_break': 'WORLD_negative_order_stable',
}
PAIR_SQUARED_HINGE_LOSS = {
    'version': 'binary-pairwise-squared-hinge-margin1-quarter-v1',
    'kind': 'binary_pairwise_squared_hinge', 'temperature': 1.0,
    'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
}
PAIRWISE_LOSS_KINDS = ('binary_pairwise_logistic', 'binary_pairwise_sigmoid',
                       PAIR_HARD_POSITIVE_LOSS['kind'], PAIR_AUC_MARGIN_LOSS['kind'], PAIR_BCE_MIX_LOSS['kind'],
                       PAIR_WIDE_SIGMOID_LOSS['kind'], PAIR_HARD_NEGATIVE_LOSS['kind'], PAIR_SQUARED_HINGE_LOSS['kind'],
                       PAIR_SIGMOID_OHEM_LOSS['kind'])
_PAIRWISE_LOSS_VERSIONS = {'binary_pairwise_logistic': 'binary-pairwise-logistic-v1',
                          'binary_pairwise_sigmoid': 'binary-pairwise-sigmoid-v1',
                          PAIR_HARD_POSITIVE_LOSS['kind']: PAIR_HARD_POSITIVE_LOSS['version'],
                          PAIR_AUC_MARGIN_LOSS['kind']: PAIR_AUC_MARGIN_LOSS['version'],
                          PAIR_BCE_MIX_LOSS['kind']: PAIR_BCE_MIX_LOSS['version'],
                          PAIR_WIDE_SIGMOID_LOSS['kind']: PAIR_WIDE_SIGMOID_LOSS['version'],
                          PAIR_HARD_NEGATIVE_LOSS['kind']: PAIR_HARD_NEGATIVE_LOSS['version'],
                          PAIR_SQUARED_HINGE_LOSS['kind']: PAIR_SQUARED_HINGE_LOSS['version'],
                          PAIR_SIGMOID_OHEM_LOSS['kind']: PAIR_SIGMOID_OHEM_LOSS['version']}


def validate_loss(specification):
    if specification is None:
        return None
    if isinstance(specification, dict) and specification.get('kind') in PAIRWISE_LOSS_KINDS:
        fields = {'version', 'kind', 'temperature', 'reduction', 'dtype', 'scope'}
        kind = specification['kind']
        if kind == PAIR_HARD_POSITIVE_LOSS['kind']:
            if specification != PAIR_HARD_POSITIVE_LOSS or type(specification.get('top_positive_count')) is not int:
                raise ValueError('hard-positive loss requires the fixed top2/mix05/stable-WORLD contract')
            fields = set(PAIR_HARD_POSITIVE_LOSS)
        if kind == PAIR_BCE_MIX_LOSS['kind']:
            if specification != PAIR_BCE_MIX_LOSS:
                raise ValueError('joint pairwise/BCE loss requires the fixed mix05 contract')
            fields = set(PAIR_BCE_MIX_LOSS)
        if kind == PAIR_HARD_NEGATIVE_LOSS['kind']:
            if specification != PAIR_HARD_NEGATIVE_LOSS or type(specification.get('top_negative_count')) is not int:
                raise ValueError('hard-negative loss requires the fixed top14/mix05/stable-WORLD contract')
            fields = set(PAIR_HARD_NEGATIVE_LOSS)
        if kind == PAIR_SIGMOID_OHEM_LOSS['kind']:
            if specification != PAIR_SIGMOID_OHEM_LOSS:
                raise ValueError('OHEM pairwise sigmoid requires the fixed top25 row-major contract')
            fields = set(PAIR_SIGMOID_OHEM_LOSS)
        if (set(specification) != fields
                or specification['version'] != _PAIRWISE_LOSS_VERSIONS[kind]
                or specification['reduction'] != 'mean' or specification['dtype'] != 'float32'
                or specification['scope'] != 'global_batch'):
            raise ValueError('pairwise loss requires its explicit versioned contract')
        temperature = specification['temperature']
        expected_temperature = 2.0 if kind == PAIR_WIDE_SIGMOID_LOSS['kind'] else 1.0
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature != expected_temperature:
            raise ValueError('binary pairwise loss temperature differs from its fixed version')
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


def validate_output_regularization(specification):
    if specification is not None and specification != OUTPUT_REGULARIZATION_RULE:
        raise ValueError('output regularization requires the fixed raw-logit square01 contract')
    return specification


def validate_output_regularization_job(job):
    policy = validate_output_regularization(job.get('output_regularization'))
    if policy is None:
        return
    loss = validate_loss(job.get('loss'))
    if (not loss or loss['kind'] != 'binary_pairwise_logistic'
            or job.get('model') != 'convnext_small' or job.get('precision') != 'fp32'
            or not (job.get('age') or {}).get('enabled')
            or job.get('effective_batch') != 64 or job.get('positive_per_batch') != 8
            or job.get('training_views', 1) != 1
            or any(job.get(key) is not None for key in
                   ('low_rank_policy', 'layerwise_lr_policy', 'head_warmup_policy',
                    'gradient_clip_policy', 'mixup', 'attention_pool', 'stage3_residual',
                    'freeze_features_before', 'final_downsample_stride',
                    'training_rotation', 'training_photometric'))):
        raise ValueError('raw-logit square01 requires the full FP32 Small age pairlog G64P8 path')


def classification_loss(logits, targets, specification=None):
    specification = validate_loss(specification)
    import torch
    from torch.nn import functional as F

    if specification is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    if specification['kind'] in PAIRWISE_LOSS_KINDS:
        logits, targets = logits.float().reshape(-1), targets.reshape(-1)
        positives, negatives = logits[targets == 1], logits[targets == 0]
        if (not positives.numel() or not negatives.numel()
                or positives.numel() + negatives.numel() != logits.numel()):
            raise ValueError('pairwise loss needs binary targets with both classes in the complete batch')
        differences = negatives[None, :] - positives[:, None]
        scaled = differences / specification['temperature']
        if specification['kind'] == PAIR_SQUARED_HINGE_LOSS['kind']:
            return .25 * F.relu(1. + scaled).square().mean()
        if specification['kind'] == PAIR_AUC_MARGIN_LOSS['kind']:
            # Empirical class variances plus the fixed raw mean-gap margin.
            mean = scaled.mean()
            return (scaled - mean).square().mean() + F.relu(1. + mean).square()
        if specification['kind'] == 'binary_pairwise_logistic':
            return F.softplus(scaled).mean()
        if specification['kind'] == PAIR_BCE_MIX_LOSS['kind']:
            weight = specification['pointwise_weight']
            return (1. - weight) * F.softplus(scaled).mean() + weight * F.binary_cross_entropy_with_logits(logits, targets.float())
        if specification['kind'] == PAIR_HARD_POSITIVE_LOSS['kind']:
            if (positives.numel(), negatives.numel()) != (8, 56):
                raise ValueError('fixed hard-positive loss requires complete WORLD G64P8 inputs')
            per_positive = F.softplus(scaled).mean(dim=1)
            # WORLD gather preserves positive order; stable sort fixes tie subgradients.
            hardest = torch.sort(per_positive, descending=True, stable=True).values[:specification['top_positive_count']]
            weight = specification['hard_positive_weight']
            return (1. - weight) * per_positive.mean() + weight * hardest.mean()
        if specification['kind'] == PAIR_HARD_NEGATIVE_LOSS['kind']:
            if (positives.numel(), negatives.numel()) != (8, 56):
                raise ValueError('fixed hard-negative loss requires complete WORLD G64P8 inputs')
            per_negative = F.softplus(scaled).mean(dim=0)
            hardest = torch.sort(per_negative, descending=True, stable=True).values[:specification['top_negative_count']]
            weight = specification['hard_negative_weight']
            return (1. - weight) * per_negative.mean() + weight * hardest.mean()
        if specification['kind'] == PAIR_WIDE_SIGMOID_LOSS['kind']:
            return 2. * specification['temperature'] * torch.sigmoid(scaled).mean()
        if specification['kind'] == PAIR_SIGMOID_OHEM_LOSS['kind']:
            return ohem_pairwise_values(differences, specification['temperature'],
                                        specification['hard_fraction']).mean()
        return 2. * torch.sigmoid(scaled).mean()
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    gamma = float(specification['gamma'])
    if gamma == 0:
        return bce.mean()
    modulation = (1. - torch.exp(-bce)).pow(gamma)
    return (bce * modulation).mean()


def regularized_pairwise_loss(logits, targets, specification, output_regularization=None):
    """Consume the already gathered final raw logits; add no communication.

    The None path returns the original objective unchanged. The fixed nonempty
    policy uses each image-plus-age logit once, including its shared bias edge.
    """
    policy = validate_output_regularization(output_regularization)
    if policy is None:
        return classification_loss(logits, targets, specification)
    loss = validate_loss(specification)
    if not loss or loss['kind'] != 'binary_pairwise_logistic':
        raise ValueError('raw-logit square01 requires the original pairwise logistic objective')
    import torch
    with torch.autocast(device_type=logits.device.type, enabled=False):
        raw = logits.float().reshape(-1)
        return classification_loss(raw, targets, specification) + policy['coefficient'] * raw.square().mean()


def ohem_pairwise_values(differences, temperature, hard_fraction):
    """WORLD pair values sorted hardest-first, truncated to the registered top share.

    ``hard_fraction`` 1.0 returns every pair in stable row-major order, making the
    mean identical to the original pairwise sigmoid objective.
    """
    import torch
    values = 2. * torch.sigmoid(differences / temperature).reshape(-1)
    selected = max(1, math.floor(values.numel() * hard_fraction))
    return torch.sort(values, descending=True, stable=True).values[:selected]


def f05_confident_joint(labels, calibrated_p1):
    """Fixed fit-only OOF candidate rule; returns no medical or label verdict."""
    labels, probabilities = list(labels), list(calibrated_p1)
    if (not labels or len(labels) != len(probabilities)
            or any(type(label) is not int or label not in (0, 1) for label in labels)
            or set(labels) != {0, 1}
            or any(isinstance(p, bool) or not isinstance(p, numbers.Real)
                   or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities)):
        raise ValueError('finite calibrated fit-only probabilities and both classes required')
    probabilities = [float(p) for p in probabilities]
    by_class = [[(1 - p if label == 0 else p) for label, p in zip(labels, probabilities) if label == cls]
                for cls in (0, 1)]
    thresholds = [max(.95, sum(values) / len(values)) for values in by_class]
    joint = [[0, 0], [0, 0]]
    suspicious, unknown = [], []
    for index, (label, p1) in enumerate(zip(labels, probabilities)):
        p0 = 1 - p1
        eligible = [cls for cls, value in enumerate((p0, p1)) if value >= thresholds[cls]]
        if len(eligible) == 2:
            if p0 == p1:
                unknown.append(index)
                continue
            predicted = int(p1 > p0)
        elif len(eligible) == 1:
            predicted = eligible[0]
        else:
            continue
        joint[label][predicted] += 1
        if predicted != label:
            suspicious.append(index)
    return {'thresholds': thresholds, 'confident_joint': joint,
            'suspicious_positions': suspicious, 'unknown_positions': unknown}


def f05_pairwise_objective(logits, targets, auxiliary, specification, *, mode):
    """Apply the preregistered objective to already aligned WORLD tensors."""
    import torch
    from torch.nn import functional as F

    spec = validate_loss(specification)
    if not spec or spec.get('kind') != 'binary_pairwise_sigmoid' or mode not in ('clean', 'distill', 'hardpos'):
        raise ValueError('F05 requires the fixed sigmoid pairwise loss and explicit mode')
    raw = logits.float().reshape(-1)
    labels = targets.detach().reshape(-1)
    values = auxiliary.detach().to(device=raw.device, dtype=torch.float32).reshape(-1)
    if (raw.numel() == 0 or labels.shape != raw.shape or values.shape != raw.shape
            or not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(values).all())):
        raise ValueError('finite, aligned WORLD logits, labels and auxiliary values required')
    positive, negative = labels == 1, labels == 0
    if not bool(positive.any()) or not bool(negative.any()) or not bool((positive | negative).all()):
        raise ValueError('binary targets and both WORLD classes required')
    if mode == 'distill':
        if not bool(((values >= 0) & (values <= 1)).all()):
            raise ValueError('calibrated OOF soft targets must be in [0, 1]')
        return classification_loss(raw, labels, spec) + .25 * F.binary_cross_entropy_with_logits(raw, values)
    if mode == 'clean' and not bool(((values == .25) | (values == 1)).all()):
        raise ValueError('cleaning weights must be exactly 0.25 or 1')
    if mode == 'hardpos' and not bool(((values == 1) | (values == 3)).all()):
        raise ValueError('hard-positive weights must be exactly 1 or 3')
    pair_weights = values[positive, None] * values[negative][None, :]
    differences = raw[negative][None, :] - raw[positive][:, None]
    return 2. * (torch.sigmoid(differences) * pair_weights).sum() / pair_weights.sum()


def gather_f05_auxiliary(values):
    """Gather detached auxiliary values in the same WORLD rank order as logits."""
    import torch
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError('F05 auxiliary gather requires the initialized default WORLD group')
    values = values.detach().reshape(-1).contiguous()
    parts = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, values)
    return torch.cat(parts, dim=0)


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

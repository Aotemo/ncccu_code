"""Explicit binary losses and WORLD pairwise gather; validation imports no Torch."""
import math

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
PAIRWISE_LOSS_KINDS = ('binary_pairwise_logistic', 'binary_pairwise_sigmoid',
                       PAIR_HARD_POSITIVE_LOSS['kind'], PAIR_AUC_MARGIN_LOSS['kind'], PAIR_BCE_MIX_LOSS['kind'],
                       PAIR_WIDE_SIGMOID_LOSS['kind'])
_PAIRWISE_LOSS_VERSIONS = {'binary_pairwise_logistic': 'binary-pairwise-logistic-v1',
                          'binary_pairwise_sigmoid': 'binary-pairwise-sigmoid-v1',
                          PAIR_HARD_POSITIVE_LOSS['kind']: PAIR_HARD_POSITIVE_LOSS['version'],
                          PAIR_AUC_MARGIN_LOSS['kind']: PAIR_AUC_MARGIN_LOSS['version'],
                          PAIR_BCE_MIX_LOSS['kind']: PAIR_BCE_MIX_LOSS['version'],
                          PAIR_WIDE_SIGMOID_LOSS['kind']: PAIR_WIDE_SIGMOID_LOSS['version']}


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
        if specification['kind'] == PAIR_WIDE_SIGMOID_LOSS['kind']:
            return 2. * specification['temperature'] * torch.sigmoid(scaled).mean()
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

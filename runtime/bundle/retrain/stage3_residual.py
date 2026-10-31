"""R125 stage3 GAP residual readout beside the unchanged final branch."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

STAGE3_RESIDUAL_RULE = {
    'version': 'stage3-gap-residual-v1', 'kind': 'intermediate_gap_linear_residual',
    'feature_index': 5, 'channels': 384, 'normalization': 'nonaffine_layer_norm_after_gap',
    'norm_eps': 1e-6, 'parameter_name': 'stage3_residual_weight', 'parameter_shape': [1, 384],
    'weight_init': 'zeros', 'bias': False, 'lr': .001, 'weight_decay': .0001, 'dropout': 0.0,
    'dtype': 'float32', 'detach': False, 'backbone_passes': 1}


def _validate_spec(spec):
    if spec is None:
        return None
    if not isinstance(spec, dict) or spec != STAGE3_RESIDUAL_RULE:
        raise ValueError('stage3 residual differs from the frozen protocol')
    return spec


def attach(model, job):
    spec = _validate_spec(job.get('stage3_residual'))
    if spec is None:
        return
    if job['model'] != 'convnext_small' or job['precision'] != 'fp32':
        raise ValueError('stage3 residual requires the registered FP32 Small model')
    reference = next(model.network.parameters())
    # Allocation consumes no RNG; all original parameters retain their initialization.
    model.stage3_residual_weight = nn.Parameter(
        torch.zeros(1, 384, device=reference.device, dtype=torch.float32))
    model.stage3_residual_spec = copy.deepcopy(spec)
    validate_model(model, job)


def validate_model(model, job):
    spec = _validate_spec(job.get('stage3_residual'))
    weight = getattr(model, 'stage3_residual_weight', None)
    if spec is None:
        if weight is not None:
            raise ValueError('stage3 residual parameter present without its protocol')
        return
    if (not isinstance(weight, nn.Parameter) or tuple(weight.shape) != (1, 384)
            or weight.dtype != torch.float32 or model.stage3_residual_spec != spec
            or job['head_lr'] != spec['lr'] or job['weight_decay'] != spec['weight_decay']):
        raise ValueError('stage3 residual parameter shape/dtype/optimizer protocol differs')


def image_logits(model, images):
    # One forward pass through the original feature stack, retaining features[5].
    stage3 = None
    hidden = images
    for index, layer in enumerate(model.network.features):
        hidden = layer(hidden)
        if index == STAGE3_RESIDUAL_RULE['feature_index']:
            stage3 = hidden
    if stage3 is None:
        raise ValueError('stage3 feature index missing from the feature stack')
    logit_image = model.network.classifier(model.network.avgpool(hidden)).flatten()
    residual = F.layer_norm(F.adaptive_avg_pool2d(stage3, 1).flatten(1), (384,),
                            weight=None, bias=None, eps=1e-6)
    return logit_image + (residual * model.stage3_residual_weight).sum(dim=1)

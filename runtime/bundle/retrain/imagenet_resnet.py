"""ImageNet-1K TorchVision ResNet-50 binary head with immutable public BN buffers.

Only the public ImageNet-1K V2 tensors enter the backbone.  The 1000-class head
is loaded strictly and then discarded; a fit-only 2048->1 head and the fit-only
age residual are trained.  No other pretrained visual asset is read.
"""
import torch
from torch import nn

from .models import IMAGENET_RESNET, validate_imagenet_provenance

BATCH_NORM = nn.modules.batchnorm._BatchNorm
HEAD_SEED = 20261027
INPUT_SIZE = 512


class ImageNetResnetBinary(nn.Module):
    def __init__(self, backbone, age_spec):
        super().__init__()
        self.backbone = backbone
        with torch.random.fork_rng(devices=[]):
            self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(2048, 1))
        generator = torch.Generator(device='cpu').manual_seed(HEAD_SEED)
        with torch.no_grad():
            self.classifier[-1].weight.copy_(
                torch.randn((1, 2048), generator=generator, dtype=torch.float32) * .01)
            self.classifier[-1].bias.zero_()
        from .age import attach_age
        attach_age(self, age_spec)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, BATCH_NORM):
                module.eval()
        return self

    def assert_bn_mode(self):
        layers = [m for m in self.modules() if isinstance(m, BATCH_NORM)]
        if len(layers) != IMAGENET_RESNET['bn_modules']:
            raise ValueError('ImageNet ResNet requires exactly 53 frozen standard BatchNorm layers')
        if any(m.training or isinstance(m, nn.SyncBatchNorm) or not m.track_running_stats for m in layers):
            raise ValueError('ImageNet ResNet requires frozen standard BatchNorm in eval mode')
        if self.training and any(not m.weight.requires_grad or not m.bias.requires_grad for m in layers):
            raise ValueError('ImageNet ResNet BN affine parameters must remain trainable')
        return layers

    def forward(self, images, age_features=None):
        if (images.ndim != 4 or tuple(images.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE)
                or images.dtype != torch.float32):
            raise ValueError('ImageNet ResNet requires FP32 Bx3x512x512 input')
        self.assert_bn_mode()
        features = self.backbone(images)
        if features.shape != (images.shape[0], 2048):
            raise ValueError('ImageNet ResNet feature dimensions differ')
        from .age import residual
        return residual(self, self.classifier(features).flatten(), age_features)


def frozen_bn_names(model):
    if not isinstance(model, ImageNetResnetBinary):
        raise ValueError('ImageNet ResNet frozen-BN model required')
    names = []
    for prefix, module in model.named_modules():
        if isinstance(module, BATCH_NORM):
            if not module.track_running_stats:
                raise ValueError('ImageNet ResNet BN running buffers required')
            names.extend(prefix + '.' + suffix for suffix in
                         ('running_mean', 'running_var', 'num_batches_tracked'))
    if len(names) != IMAGENET_RESNET['bn_buffers'] or len(set(names)) != len(names):
        raise ValueError('ImageNet ResNet requires 159 BN buffers')
    return frozenset(names)


def frozen_bn_digest(model):
    from .ema import tensor_digest
    state = model.state_dict()
    names = frozen_bn_names(model)
    return tensor_digest({name: state[name] for name in names})


def load_public_state(model, state):
    """Strict 320-key TorchVision load into the complete 1000-class network."""
    expected = model.state_dict()
    if len(expected) != IMAGENET_RESNET['full_state_keys']:
        raise ValueError('ImageNet ResNet full network key count differs')
    if not isinstance(state, dict) or len(state) != IMAGENET_RESNET['full_state_keys']:
        raise ValueError('ImageNet ResNet state key count differs')
    if set(state) != set(expected):
        raise ValueError('ImageNet ResNet state key set differs')
    if not {'fc.weight', 'fc.bias'} <= set(state):
        raise ValueError('ImageNet ResNet state lacks the public 1000-class head')
    for name, reference in expected.items():
        value = state[name]
        if not torch.is_tensor(value):
            raise ValueError('ImageNet ResNet state entry is not a tensor: ' + name)
        if tuple(value.shape) != tuple(reference.shape) or value.dtype != reference.dtype:
            raise ValueError('ImageNet ResNet tensor shape/dtype differs: ' + name)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('ImageNet ResNet tensor is not finite: ' + name)
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    if any(not torch.equal(loaded[name], state[name]) for name in loaded):
        raise ValueError('ImageNet ResNet tensors differ after strict load')
    return loaded


def create_imagenet_binary(job, *, initialize=True):
    if (job.get('model') != 'resnet50_imagenet' or job.get('image_size') != INPUT_SIZE
            or job.get('precision') != 'fp32' or job.get('normalization') != 'imagenet'
            or job.get('dropout') != .2 or job.get('frozen_bn') is not True
            or float(job.get('stochastic_depth', 0) or 0) != 0
            or not job.get('age', {}).get('enabled')):
        raise ValueError('N08 ImageNet ResNet model protocol differs')
    from torchvision.models import resnet50
    with torch.random.fork_rng(devices=[]):
        backbone = resnet50(weights=None)
    if initialize:
        initialization = job['initialization']
        if initialization.get('kind') != 'imagenet':
            raise ValueError('N08 requires explicit ImageNet-1K initialization')
        proof = validate_imagenet_provenance(initialization)
        if proof['model'] != IMAGENET_RESNET['provenance_model']:
            raise ValueError('N08 provenance model differs from TorchVision ResNet-50')
        state = torch.load(initialization['path'], map_location='cpu', weights_only=True)
        load_public_state(backbone, state)
        del state
    backbone.fc = nn.Identity()
    if len(backbone.state_dict()) != IMAGENET_RESNET['backbone_keys']:
        raise ValueError('ImageNet ResNet backbone key count differs after dropping fc')
    result = ImageNetResnetBinary(backbone, job['age'])
    result.assert_bn_mode()
    frozen_bn_names(result)
    return result

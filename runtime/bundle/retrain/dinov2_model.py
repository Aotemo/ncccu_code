"""N03 original DINOv2-S/14, fixed 504px CLS/mean-patch binary readout."""
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


def _official_factory():
    if os.environ.get('XFORMERS_DISABLED') != '1':
        raise ValueError('N03 requires XFORMERS_DISABLED=1')
    root = Path(__file__).resolve().parents[2]
    vendor = (root / 'assets/dinov2-source').resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name == 'dinov2' or name.startswith('dinov2.'):
            location = getattr(module, '__file__', None)
            if not location or not Path(location).resolve().is_relative_to(vendor):
                raise ValueError('DINOv2 imported from outside the frozen N03 vendor tree: ' + name)
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from dinov2.hub.backbones import dinov2_vits14
    for name, module in tuple(sys.modules.items()):
        if name == 'dinov2' or name.startswith('dinov2.'):
            location = getattr(module, '__file__', None)
            if not location or not Path(location).resolve().is_relative_to(vendor):
                raise ValueError('DINOv2 import escaped the frozen N03 vendor tree: ' + name)
    return dinov2_vits14


class DINOv2Binary(nn.Module):
    def __init__(self, backbone, age_spec):
        super().__init__()
        self.backbone = backbone
        self.backbone.mask_token.requires_grad_(False)
        # The head uses its own CPU generator; constructing Linear must not
        # perturb the worker's training RNG.
        with torch.random.fork_rng(devices=[]):
            self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(768, 1))
        generator = torch.Generator(device='cpu').manual_seed(20261027)
        with torch.no_grad():
            self.classifier[-1].weight.copy_(torch.randn((1, 768), generator=generator, dtype=torch.float32) * .01)
            self.classifier[-1].bias.zero_()
        from .age import attach_age
        attach_age(self, age_spec)

    def forward(self, images, age_features=None):
        if (images.ndim != 4 or tuple(images.shape[1:]) != (3, 504, 504)
                or images.dtype != torch.float32):
            raise ValueError('N03 DINOv2 requires FP32 Bx3x504x504 input')
        with sdpa_kernel(SDPBackend.MATH):
            features = self.backbone.forward_features(images)
            cls = features['x_norm_clstoken']
            patches = features['x_norm_patchtokens']
            if cls.shape != (images.shape[0], 384) or patches.shape != (images.shape[0], 1296, 384):
                raise ValueError('N03 DINOv2 CLS/patch feature geometry differs')
            logits = self.classifier(torch.cat((cls, patches.mean(dim=1)), dim=1)).flatten()
        from .age import residual
        return residual(self, logits, age_features)


def create_dinov2_binary(job, *, initialize=True):
    if (job.get('model') != 'dinov2_vits14' or job.get('image_size') != 504
            or job.get('precision') != 'fp32' or job.get('stochastic_depth') != .1
            or job.get('dropout') != .2 or not job.get('age', {}).get('enabled')):
        raise ValueError('N03 DINOv2 model protocol differs')
    factory = _official_factory()
    backbone = factory(pretrained=False, drop_path_rate=.1)
    if initialize:
        from .models import validate_initialization
        proof = validate_initialization(job['initialization'])
        if proof['kind'] != 'self_supervised' or proof['model'] != 'dinov2_vits14':
            raise ValueError('N03 initialization proof is not the frozen DINOv2 weight')
        state = torch.load(job['initialization']['path'], map_location='cpu', weights_only=True)
        expected = backbone.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError('DINOv2 public state keys differ from the official backbone')
        for name, tensor in state.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tensor.shape != expected[name].shape or not bool(torch.isfinite(tensor).all())):
                raise ValueError('DINOv2 public state shape/dtype/finite differs: ' + name)
        backbone.load_state_dict(state, strict=True)
    return DINOv2Binary(backbone, job['age'])

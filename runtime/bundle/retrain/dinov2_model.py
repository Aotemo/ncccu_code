"""N03 original DINOv2-S/14, fixed 504px CLS/mean-patch binary readout."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BASE_FORMAT = 'dinov2-official-vitb14-512-pad6-state-dict-v1'
BASE_WEIGHT_SHA = '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
BASE_PROVENANCE_SHA = 'de08471a17c713fc8a91c6e95eca38f3bf010ddcf58f4d049c37a57ad0bfc82f'


def base_provenance(initialization, *, verify_weight):
    """Bind the original natural-image Base asset, separately from old Small."""
    path, meta_path = Path(initialization['path']), Path(initialization['provenance'])
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if (initialization.get('kind') != 'self_supervised'
            or initialization.get('format') != BASE_FORMAT
            or initialization.get('sha256') != BASE_WEIGHT_SHA
            or not path.is_absolute() or path.is_symlink()
            or not meta_path.is_absolute() or meta_path.is_symlink()
            or digest(meta_path) != BASE_PROVENANCE_SHA
            or (verify_weight and digest(path) != BASE_WEIGHT_SHA)):
        raise ValueError('original DINOv2 Base asset identity differs')
    meta = json.loads(meta_path.read_text())
    expected = {'model':'dinov2_vitb14', 'format':BASE_FORMAT,
        'revision':'7764ea0f912e53c92e82eb78a2a1631e92725fc8',
        'training_data':'LVD-142M', 'sha256':BASE_WEIGHT_SHA,
        'declared_license':'Apache-2.0', 'redistribution_confirmed':True,
        'source_url':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'}
    license_path = meta_path.parent / meta['license_file']
    if (any(meta.get(k) != v for k,v in expected.items())
            or license_path.is_symlink() or digest(license_path) !=
               '600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7'):
        raise ValueError('original DINOv2 Base source/license differs')
    return meta


def public_code_root(job):
    """Verify the original public code beside the explicit provenance asset."""
    if job.get('model') == 'dinov2_vitb14':
        meta = base_provenance(job['initialization'], verify_weight=False)
        code = meta['code_asset_provenance']
        vendor = public_code_root({'initialization':{'provenance':code['path']}})
        if str(vendor) != code['code_root']:
            raise ValueError('original DINOv2 Base code root differs')
        return vendor
    path=Path(job['initialization']['provenance'])
    if (not path.is_absolute() or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest()!=
               'e76029ce54fcf519d4006d72ee23d8b35af52dde01d8afa3c21a9ed17beab9dc'):
        raise ValueError('original DINOv2 public code provenance differs')
    meta=json.loads(path.read_text());vendor=(path.parent/'dinov2-source').resolve(strict=True)
    for name,digest in meta['source_files'].items():
        relative=Path(name);source=vendor/relative
        if (relative.is_absolute() or '..' in relative.parts or source.is_symlink()
                or not source.resolve().is_relative_to(vendor)
                or hashlib.sha256(source.read_bytes()).hexdigest()!=digest):
            raise ValueError('pinned original DINOv2 source differs: '+name)
    return vendor


def _official_factory(job=None):
    if os.environ.get('XFORMERS_DISABLED') != '1':
        raise ValueError('N03 requires XFORMERS_DISABLED=1')
    root = Path(__file__).resolve().parents[2]
    vendor = public_code_root(job) if job is not None else (root / 'assets/dinov2-source').resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name == 'dinov2' or name.startswith('dinov2.'):
            location = getattr(module, '__file__', None)
            if not location or not Path(location).resolve().is_relative_to(vendor):
                raise ValueError('DINOv2 imported from outside the frozen N03 vendor tree: ' + name)
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14
    for name, module in tuple(sys.modules.items()):
        if name == 'dinov2' or name.startswith('dinov2.'):
            location = getattr(module, '__file__', None)
            if not location or not Path(location).resolve().is_relative_to(vendor):
                raise ValueError('DINOv2 import escaped the frozen N03 vendor tree: ' + name)
    return dinov2_vitb14 if job is not None and job.get('model') == 'dinov2_vitb14' else dinov2_vits14


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
    factory = _official_factory(job)
    backbone = factory(pretrained=False, drop_path_rate=.1)
    public=None
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
        loaded=backbone.load_state_dict(state, strict=True)
        from .ema import tensor_digest
        exact=all(torch.equal(backbone.state_dict()[name],tensor) for name,tensor in state.items())
        if not exact:raise ValueError('DINOv2 public tensors differ after strict load')
        public={'public_tensor_count':len(state),'full_head_shape':None,'backbone_only':True,
            'all_public_tensors_exact':exact,'all_public_tensors_FP32_finite':True,
            'strict_missing_keys':list(loaded.missing_keys),'strict_unexpected_keys':list(loaded.unexpected_keys),
            'public_tensor_sha256':tensor_digest(state),'initialization':proof,
            'source_files_sha256':'ed5db7b07fdcd8d3ce92b1ce81bc2f509e6e7472756a348d0859351543acfed5'}
    result=DINOv2Binary(backbone, job['age'])
    if public is not None:result.public_initialization_receipt=public
    if job.get('experiment91') is not None:
        from .experiment91 import describe_model
        result.candidate_normalization=describe_model(result)
    return result


class DINOv2BaseBinary(nn.Module):
    def __init__(self, backbone, age_spec):
        super().__init__()
        self.backbone = backbone
        self.backbone.mask_token.requires_grad_(False)
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(1536, 1))
        from .age import attach_age
        attach_age(self, age_spec)

    def forward(self, images, age_features=None):
        if (images.ndim != 4 or tuple(images.shape[1:]) != (3,512,512)
                or images.dtype != torch.float32):
            raise ValueError('DINOv2 Base requires FP32 Bx3x512x512 input')
        with sdpa_kernel(SDPBackend.MATH):
            features = self.backbone.forward_features(F.pad(images,(0,6,0,6)))
            cls, patches = features['x_norm_clstoken'], features['x_norm_patchtokens']
            if cls.shape != (images.shape[0],768) or patches.shape != (images.shape[0],1369,768):
                raise ValueError('DINOv2 Base native CLS/patch geometry differs')
            logits = self.classifier(torch.cat((cls,patches.mean(1)),dim=1)).flatten()
        from .age import residual
        return residual(self,logits,age_features)


def create_dinov2_base_binary(job, *, initialize=True):
    fixed = {'model':'dinov2_vitb14','image_size':512,'prepared_size':512,
        'normalization':'imagenet','precision':'fp32','stochastic_depth':.1,
        'dropout':.2,'head_lr_scope':'last_linear'}
    if any(job.get(k) != v for k,v in fixed.items()) or not job.get('age',{}).get('enabled'):
        raise ValueError('DINOv2 Base model protocol differs')
    backbone = _official_factory(job)(pretrained=False,drop_path_rate=.1)
    public = None
    if initialize:
        from .models import validate_initialization
        proof = validate_initialization(job['initialization'])
        state = torch.load(job['initialization']['path'],map_location='cpu',weights_only=True)
        expected = backbone.state_dict()
        if not isinstance(state,dict) or set(state) != set(expected) or len(state) != 175:
            raise ValueError('original DINOv2 Base public state keys differ')
        for name,tensor in state.items():
            if (not isinstance(tensor,torch.Tensor) or tensor.dtype != torch.float32
                    or tensor.shape != expected[name].shape or not bool(torch.isfinite(tensor).all())):
                raise ValueError('DINOv2 Base public shape/dtype/finite differs: '+name)
        loaded = backbone.load_state_dict(state,strict=True)
        if not all(torch.equal(backbone.state_dict()[n],t) for n,t in state.items()):
            raise ValueError('DINOv2 Base strict-loaded public tensors differ')
        from .ema import tensor_digest
        public = {'public_tensor_count':len(state),'full_head_shape':None,'backbone_only':True,
            'all_public_tensors_exact':True,'all_public_tensors_FP32_finite':True,
            'strict_missing_keys':list(loaded.missing_keys),'strict_unexpected_keys':list(loaded.unexpected_keys),
            'public_tensor_sha256':tensor_digest(state),'initialization':proof,
            'source_files_sha256':'ed5db7b07fdcd8d3ce92b1ce81bc2f509e6e7472756a348d0859351543acfed5'}
    result = DINOv2BaseBinary(backbone,job['age'])
    if public is not None: result.public_initialization_receipt = public
    if job.get('experiment91') is not None:
        from .experiment91 import describe_model
        result.candidate_normalization = describe_model(result)
    return result

"""ImageNet-1K TorchVision ResNet-50 binary head with immutable public BN buffers.

Only the public ImageNet-1K V2 tensors enter the backbone.  The 1000-class head
is loaded strictly and then discarded; a fit-only 2048->1 head and the fit-only
age residual are trained.  No other pretrained visual asset is read.
"""
import torch
from torch import nn

from .models import (IMAGENET_RESNET, DINO_RESNET,
                     validate_imagenet_provenance, validate_dino_resnet_provenance)

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


def load_dino_backbone_state(model, state):
    """The official DINO asset has 318 backbone tensors and no classifier."""
    expected = model.state_dict()
    if (not isinstance(model.fc, nn.Identity) or len(expected) != IMAGENET_RESNET['backbone_keys']
            or not isinstance(state, dict) or set(state) != set(expected)):
        raise ValueError('DINO R50 requires its exact 318-key backbone-only state')
    for name, reference in expected.items():
        value = state[name]
        if (not torch.is_tensor(value) or value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.is_floating_point() and (value.dtype != torch.float32
                    or not bool(torch.isfinite(value).all()))):
            raise ValueError('DINO R50 backbone tensor differs: ' + name)
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys or any(
            not torch.equal(model.state_dict()[name], state[name]) for name in state):
        raise ValueError('DINO R50 backbone did not load exactly')
    return model.state_dict()


def create_imagenet_binary(job, *, initialize=True):
    if (job.get('model') != 'resnet50_imagenet' or job.get('image_size') != INPUT_SIZE
            or job.get('precision') != 'fp32' or job.get('normalization') != 'imagenet'
            or job.get('dropout') != .2 or job.get('frozen_bn') is not True
            or float(job.get('stochastic_depth', 0) or 0) != 0
            or not job.get('age', {}).get('enabled') or job.get('head_seed', HEAD_SEED) != HEAD_SEED):
        raise ValueError('N08 ImageNet ResNet model protocol differs')
    from torchvision.models import resnet50
    with torch.random.fork_rng(devices=[]):
        backbone = resnet50(weights=None)
    dino = job.get('initialization', {}).get('format') == DINO_RESNET['format']
    public_receipt = None
    if initialize:
        initialization = job['initialization']
        if initialization.get('kind') != 'imagenet':
            raise ValueError('N08 requires explicit ImageNet-1K initialization')
        proof = (validate_dino_resnet_provenance(initialization) if dino
                 else validate_imagenet_provenance(initialization))
        if proof['model'] != IMAGENET_RESNET['provenance_model']:
            raise ValueError('N08 provenance model differs from TorchVision ResNet-50')
        state = torch.load(initialization['path'], map_location='cpu', weights_only=True)
        if dino:
            backbone.fc = nn.Identity()
            load_dino_backbone_state(backbone, state)
        else:
            load_public_state(backbone, state)
        from .ema import tensor_digest
        public_receipt = {'initialization': proof, 'strict_missing_keys': [], 'strict_unexpected_keys': [],
            'public_tensor_count': len(state), 'public_tensor_sha256': tensor_digest(state),
            'full_head_shape': None if dino else list(backbone.fc.weight.shape),
            'backbone_only': dino, 'all_public_tensors_exact': True,
            'all_public_floating_FP32_finite': all(not t.is_floating_point() or t.dtype == torch.float32
                                                  and bool(torch.isfinite(t).all()) for t in state.values()),
            'public_integer_counter_tensors': sum(not t.is_floating_point() for t in state.values())}
        del state
    backbone.fc = nn.Identity()
    if len(backbone.state_dict()) != IMAGENET_RESNET['backbone_keys']:
        raise ValueError('ImageNet ResNet backbone key count differs after dropping fc')
    result = ImageNetResnetBinary(backbone, job['age'])
    result.assert_bn_mode()
    frozen_bn_names(result)
    if initialize:
        apply_initial_bn_statistics(result, job)
    if job.get('experiment91') is not None:
        if initialize:
            from .experiment91 import describe_model
            result.candidate_normalization = describe_model(result)
        result.public_initialization_receipt = public_receipt
    return result


def apply_initial_bn_statistics(model, job):
    """Only the registered fit-derived floating BN asset may replace public buffers."""
    specification = job.get('bn_statistics')
    if specification is None or specification == {'kind': 'public'}:
        return None
    if specification.get('kind') != 'fit_fixed_reference' or set(specification) != {'kind', 'artifact', 'receipt', 'spec'}:
        raise ValueError('explicit R50 BN statistics identity required')
    from .bn_calibration import validate_calibration_spec
    validate_calibration_spec(specification['spec'])
    import json
    from pathlib import Path
    from .site_binding import load_site, evidence_path, sha256
    from .ema import tensor_digest
    site = load_site();paths = {}
    for key in ('artifact', 'receipt'):
        reference = specification[key];relative = Path(reference['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('derived R50 BN asset must stay inside work_root')
        path = evidence_path(str(Path(site['work_root']) / relative), site['work_root'])
        if sha256(path) != reference['sha256']:
            raise ValueError('derived R50 BN content pin differs')
        paths[key] = path
    receipt = json.loads(paths['receipt'].read_text());calibration = receipt['calibration']
    parameters_before = tensor_digest(dict(model.named_parameters()))
    public_bn = frozen_bn_digest(model)
    contract = job['age']['contract']
    if (receipt.get('schema_version') != 1 or receipt.get('scope') != 'public_r50_fit_derived_bn'
            or receipt.get('passed') is not True
            or receipt['source']['weight_sha256'] != IMAGENET_RESNET['sha256']
            or receipt['source']['parameters_sha256'] != parameters_before
            or receipt['source']['public_bn_sha256'] != public_bn
            or receipt['fit']['fit_indices_sha256'] != contract['fit_indices_sha256']
            or receipt['fit']['fit_rows_identity_sha256'] != contract['fit_rows_identity_sha256']
            or receipt['artifact'] != {'path': str(paths['artifact']), 'sha256': specification['artifact']['sha256']}
            or calibration.get('passed') is not True or calibration['spec'] != specification['spec']
            or (calibration['layer_count'], calibration['floating_buffer_count'], calibration['counter_buffer_count']) != (53, 106, 53)
            or calibration['reference_bn_digest_before'] != public_bn
            or calibration['global_images'] != contract['fit_count']
            or not all(calibration[key] for key in ('parameters_unchanged', 'counters_unchanged', 'mode_rng_restored'))):
        raise ValueError('fit-derived R50 BN source/fit/parameter proof differs')
    derived = torch.load(paths['artifact'], map_location='cpu', weights_only=True)
    state = model.state_dict();names = {n for n in frozen_bn_names(model) if not n.endswith('.num_batches_tracked')}
    counters = {n: state[n].clone() for n in frozen_bn_names(model) if n.endswith('.num_batches_tracked')}
    if set(derived) != names:
        raise ValueError('derived R50 BN asset requires exactly106 floating buffers')
    with torch.no_grad():
        for name, value in derived.items():
            if (value.shape != state[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all()
                    or name.endswith('.running_var') and bool((value < 0).any())):
                raise ValueError('invalid derived R50 BN tensor '+name)
            state[name].copy_(value)
    if (tensor_digest(dict(model.named_parameters())) != parameters_before
            or any(not torch.equal(state[n], value) for n, value in counters.items())
            or frozen_bn_digest(model) != calibration['derived_bn_digest_after']):
        raise ValueError('derived R50 BN changed parameters/counters or has wrong digest')
    model.assert_bn_mode()
    return receipt

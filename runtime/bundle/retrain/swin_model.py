"""The single registered Swin V1 Base binary readout and its static indices."""
from functools import partial

import torch
from torch import nn
from torchvision.models.swin_transformer import SwinTransformer


class SwinBinaryNetwork(SwinTransformer):
    def __init__(self, job):
        super().__init__(patch_size=[4, 4], embed_dim=128, depths=[2, 2, 18, 2],
                         num_heads=[4, 8, 16, 32], window_size=[7, 7],
                         mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0,
                         stochastic_depth_prob=job['stochastic_depth'], num_classes=1000,
                         norm_layer=partial(nn.LayerNorm, eps=1e-5))

    def forward(self, images):
        x = self.features(images)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.classifier(x)


class SwinBinary(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, images, age_features=None):
        logits = self.network(images).flatten()
        if hasattr(self, 'age_weight'):
            from .age import residual
            return residual(self, logits, age_features)
        return logits


def static_index_buffers(value):
    """Return only the known Swin V1 persistent integer position indices."""
    state = value.state_dict() if isinstance(value, nn.Module) else value
    result = {name: tensor for name, tensor in state.items()
              if name.endswith('.relative_position_index')}
    if (len(result) != 24 or any(tensor.dtype != torch.int64 or tensor.shape != (2401,)
                                 or not bool(((0 <= tensor) & (tensor < 169)).all())
                                 for tensor in result.values())):
        raise ValueError('Swin V1 static relative-position indices differ')
    return result


def create_swin_binary(job, initialize=True):
    from .ema import tensor_digest
    from .models import validate_initialization
    network = SwinBinaryNetwork(job)
    generated_indices = {name: tensor.clone() for name, tensor in static_index_buffers(network).items()}
    public_receipt = None
    if initialize:
        proof = validate_initialization(job['initialization'])
        from .experiment91 import public_source
        if proof['model'] != public_source()['model'] or proof['final_norm_eps'] != 1e-5:
            raise ValueError('Swin public provenance or native final norm differs')
        state = torch.load(job['initialization']['path'], map_location='cpu', weights_only=True)
        loaded = network.load_state_dict(state, strict=True)
        actual = network.state_dict()
        if (loaded.missing_keys or loaded.unexpected_keys or actual.keys() != state.keys()
                or network.head.out_features != 1000
                or any(not torch.equal(actual[name], tensor)
                       or (tensor.is_floating_point() and (tensor.dtype != torch.float32
                           or not bool(torch.isfinite(tensor).all()))) for name, tensor in state.items())
                or any(not torch.equal(actual[name], tensor)
                       for name, tensor in generated_indices.items())):
            raise ValueError('Swin full public1000 strict/exact/FP32/static-index proof differs')
        public_receipt = {'initialization': proof, 'strict_missing_keys': loaded.missing_keys,
                          'strict_unexpected_keys': loaded.unexpected_keys,
                          'public_tensor_count': len(state), 'public_tensor_sha256': tensor_digest(state),
                          'full_head_shape': list(network.head.weight.shape),
                          'all_public_tensors_exact': True, 'all_public_tensors_FP32_finite': True,
                          'static_index_count': len(generated_indices)}
    network.head = nn.Identity()
    network.classifier = nn.Sequential(nn.Dropout(job['dropout']), nn.Linear(1024, 1))
    result = SwinBinary(network)
    if job.get('age') is not None:
        from .age import attach_age
        attach_age(result, job['age'])
    if job.get('experiment91') is not None:
        from .experiment91 import describe_model
        result.candidate_normalization = describe_model(result)
        result.public_initialization_receipt = public_receipt
    return result

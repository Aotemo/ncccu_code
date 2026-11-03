"""One fit-only spatial RGB reconstruction signal, without labels or a teacher."""
import torch
from torch import nn
from torch.nn import functional as F

MASKED_RECIPE = {'version': 'fit-masked-rgb32-1600-v1', 'updates': 1600,
    'global_batch': 64, 'per_rank_batch': 8, 'training_views': 1,
    'image_size': 512, 'training_scale': [0.85, 1.0],
    'projector': [768, 3072, 32], 'backbone_lr': 1e-4,
    'projector_lr': 1e-3, 'weight_decay': 1e-4,
    'warmup_updates': 160, 'final_lr_factor': 0.01,
    'gradient_clip': 1.0, 'precision': 'fp32',
    'mask_patch_size': 32, 'mask_count': 154, 'mask_grid': [16, 16],
    'mask_fill': 'normalized_zero', 'target': 'same_augmented_normalized_RGB',
    'loss': 'masked_L1_mean_RGB', 'sampling_rng_domain': 'r115-fit-vicreg-1600-v1'}


def spatial_mask(seeds, *, device=None):
    """Exactly 154/256 disjoint 32px blocks per image; sample-key replay on CPU."""
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2 ** 64 for seed in seeds):
        raise ValueError('explicit nonnegative per-image mask seeds required')
    masks = []
    for seed in seeds:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        selected = torch.randperm(256, generator=generator)[:154]
        grid = torch.zeros(256, dtype=torch.bool)
        grid[selected] = True
        masks.append(grid.reshape(1, 16, 16).repeat_interleave(32, 1).repeat_interleave(32, 2))
    return torch.stack(masks).to(device=device)


def masked_l1(prediction, target, mask):
    """Only the hidden RGB pixels contribute, with one global RGB-pixel mean."""
    if (prediction.ndim != 4 or prediction.shape != target.shape or prediction.shape[1] != 3
            or mask.shape != (prediction.shape[0], 1, *prediction.shape[2:])
            or mask.dtype != torch.bool):
        raise ValueError('aligned RGB predictions/targets and binary spatial mask required')
    count = mask.sum()
    if not bool(count):
        raise ValueError('masked reconstruction requires hidden target pixels')
    return ((prediction - target.detach()).abs() * mask).sum() / (3 * count)


class MaskedReconstructionModel(nn.Module):
    """Shared native Small features/final-LN; the spatial decoder is disposable."""
    def __init__(self, binary, *, projector_seed):
        super().__init__()
        self.features = binary.network.features
        self.final_norm = binary.network.classifier[0]
        if self.final_norm.weight.numel() != 768 or self.final_norm.bias.numel() != 768:
            raise ValueError('masked reconstruction requires the native Small final LN')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(projector_seed)
            self.projector = nn.Sequential(nn.Conv2d(768, 3 * 32 ** 2, 1), nn.PixelShuffle(32))

    def forward(self, images, mask):
        if (images.ndim != 4 or images.shape[1:] != (3, 512, 512)
                or mask.shape != (images.shape[0], 1, 512, 512) or mask.dtype != torch.bool):
            raise ValueError('native512 RGB and aligned boolean mask required')
        value = self.features(images.masked_fill(mask, 0.))
        if value.shape[1:] != (768, 16, 16):
            raise ValueError('native Small spatial stride32/features differ')
        # Native classifier LN operates on [768,1,1] after GAP. Reuse its
        # exact gain/bias/epsilon independently at each spatial position.
        value = F.layer_norm(value.permute(0, 2, 3, 1), (768,),
            self.final_norm.weight.reshape(768), self.final_norm.bias.reshape(768), self.final_norm.eps)
        return self.projector(value.permute(0, 3, 1, 2).contiguous())

"""Differentiable global gather and registered fit-only embedding losses."""
import math
import torch
import torch.distributed as dist


class _GatherEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, rank, world_size):
        ctx.rank = int(rank)
        ctx.world_size = int(world_size)
        ctx.rows = local.shape[0]
        source = local.contiguous()
        parts = [torch.empty_like(source) for _ in range(ctx.world_size)]
        dist.all_gather(parts, source)
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Every rank holds the same global gradient; SUM then slice so DDP's mean cancels exactly.
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        start = ctx.rank * ctx.rows
        return grad[start:start + ctx.rows].contiguous(), None, None


def gather_embeddings(local, rank=None, world_size=None):
    """Concatenate local NxD FP32 rows in world rank order; world 1 is a no-op."""
    if local.dim() != 2:
        raise ValueError('gather_embeddings expects a 2D local embedding')
    initialized = dist.is_available() and dist.is_initialized()
    if world_size is None:
        world_size = dist.get_world_size() if initialized else 1
    if rank is None:
        rank = dist.get_rank() if initialized else 0
    if world_size == 1:
        return local
    if not initialized:
        raise RuntimeError('multi-rank gather requires an initialized process group')
    return _GatherEmbeddings.apply(local, rank, world_size)


def vicreg_loss(z1, z2):
    """VICReg on two already-global NxD FP32 views; every term stays differentiable."""
    if z1.dim() != 2 or z2.dim() != 2 or z1.shape != z2.shape:
        raise ValueError('vicreg_loss expects two equal 2D global embeddings')
    n, d = z1.shape
    if n < 2 or d < 1:
        raise ValueError('vicreg_loss needs N>=2 and D>=1')
    invariance = ((z1 - z2) ** 2).mean()

    def variance_term(value):
        var = torch.var(value, dim=0, unbiased=True)
        return torch.relu(1.0 - torch.sqrt(var + 1e-4)).mean()

    variance = 0.5 * (variance_term(z1) + variance_term(z2))

    def covariance_term(value):
        centered = value - value.mean(dim=0, keepdim=True)
        cov = centered.transpose(0, 1) @ centered / (n - 1)
        off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
        return (off_diagonal ** 2).sum() / d

    covariance = covariance_term(z1) + covariance_term(z2)
    return 25.0 * invariance + 25.0 * variance + covariance


def nt_xent_loss(z1, z2, *, temperature):
    """Symmetric instance prediction; equal row indices identify two image views."""
    if (z1.dim() != 2 or z1.shape != z2.shape or z1.shape[0] < 2 or z1.shape[1] < 1
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('NT-Xent requires equal NxD views, N>=2 and positive temperature')
    n = z1.shape[0]
    values = torch.nn.functional.normalize(torch.cat((z1, z2)), dim=1, eps=1e-12)
    logits = (values @ values.transpose(0, 1)) / temperature
    anchors = torch.arange(2 * n, device=logits.device)
    logits = logits.masked_fill(torch.eye(2 * n, device=logits.device, dtype=torch.bool), -torch.inf)
    targets = (anchors + n) % (2 * n)
    loss = (torch.logsumexp(logits, dim=1) - logits[anchors, targets]).mean()
    return loss, {'contrastive_logits_shape': list(logits.shape),
                  'contrastive_anchor_count': logits.shape[0],
                  'contrastive_negative_views_per_anchor': logits.shape[1] - 2,
                  'contrastive_positive_targets_per_anchor': 1,
                  'contrastive_temperature': float(temperature),
                  'contrastive_self_excluded': True}


def supervised_contrastive_loss(z1, z2, labels, *, temperature):
    """Average each anchor's log probabilities over its same-label other views."""
    if (z1.dim() != 2 or z1.shape != z2.shape or z1.shape[0] < 2 or z1.shape[1] < 1
            or not isinstance(labels, torch.Tensor) or labels.dim() != 1
            or labels.shape[0] != z1.shape[0] or labels.dtype not in
                (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or bool((labels < 0).any()) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('supervised contrastive loss requires equal NxD views and N nonnegative integer labels')
    n = z1.shape[0]
    values = torch.nn.functional.normalize(torch.cat((z1, z2)), dim=1, eps=1e-12)
    scores = (values @ values.transpose(0, 1)) / temperature
    tag = labels.to(device=scores.device).repeat(2)
    diagonal = torch.eye(2*n, device=scores.device, dtype=torch.bool)
    positive = (tag[:, None] == tag[None, :]) & ~diagonal
    counts = positive.sum(dim=1)
    normalizer = torch.logsumexp(scores.masked_fill(diagonal, -torch.inf), dim=1)
    positive_mean = scores.masked_fill(~positive, 0).sum(dim=1) / counts
    loss = (normalizer - positive_mean).mean()
    negatives = 2*n - 1 - counts
    return loss, {'contrastive_logits_shape': list(scores.shape),
        'contrastive_anchor_count': 2*n, 'contrastive_temperature': float(temperature),
        'contrastive_self_excluded': True, 'contrastive_target': 'fit_csv_same_label',
        'contrastive_positive_targets_per_anchor_min': int(counts.min()),
        'contrastive_positive_targets_per_anchor_max': int(counts.max()),
        'contrastive_negative_views_per_anchor_min': int(negatives.min()),
        'contrastive_negative_views_per_anchor_max': int(negatives.max()),
        'contrastive_fit_positive_images': int((labels == 1).sum())}

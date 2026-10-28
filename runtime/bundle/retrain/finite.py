"""Exact floating-point finiteness predicate with grouped device dispatch."""
import torch


@torch.no_grad()
def all_floating_finite(tensors):
    groups = {}
    for value in tensors:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            groups.setdefault((value.device, value.dtype), []).append(value.detach().reshape(-1))
    for values in groups.values():
        flat = values[0] if len(values) == 1 else torch.cat(values)
        if not bool(torch.isfinite(flat).all()):
            return False
    return True

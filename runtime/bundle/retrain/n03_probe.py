"""Frozen fit-only probe timing and capacity admission mathematics."""
import math


def project_seconds(t_step, t_eval1024, t_checkpoint):
    values = (t_step, t_eval1024, t_checkpoint)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('probe times must be finite nonnegative numbers')
    return 1.25 * (5688 * t_step + 16 * (10277 / 1024) * t_eval1024 + 12 * t_checkpoint)


def admitted(t_step, t_eval1024, t_checkpoint):
    return project_seconds(t_step, t_eval1024, t_checkpoint) <= 18000


def slowest_mean(records):
    if not isinstance(records, list) or len(records) != 4 or {r.get('rank') for r in records} != set(range(4)):
        raise ValueError('exactly four distinct rank timing records required')
    means = []
    for record in records:
        samples = record.get('timed_seconds')
        if (not isinstance(samples, list) or len(samples) != 50
                or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in samples)):
            raise ValueError('each rank requires exactly 50 finite positive post-warmup times')
        means.append(math.fsum(samples) / len(samples))
    return max(means)


def equal_state_tree(left, right):
    """Exact tensor/value comparison after loading a self-produced checkpoint."""
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(equal_state_tree(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(equal_state_tree(a, b) for a, b in zip(left, right)))
    return left == right

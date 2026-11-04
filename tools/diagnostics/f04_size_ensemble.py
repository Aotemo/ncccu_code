"""Fixed F04 weight export, paired dev comparison and size-limited equal fusion.

All calls belong in the verified CPU preparation channel. Exported weights and
sample-level predictions remain outside Git; no official data or TTA is used.
"""
import hashlib
import math
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rank_and_cap(rows, precision, byte_limit=500_000_000):
    """Fixed AUROC order; stop at the first file that exceeds the budget."""
    if precision not in ('fp16', 'fp32') or type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError('explicit precision and positive byte limit required')
    rows = list(rows)
    ids, orders = set(), set()
    for row in rows:
        value, order, name = row.get('auroc'), row.get('order'), row.get('id')
        if (not isinstance(name, str) or not name or name in ids or type(order) is not int or order < 0
                or order in orders or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not .988 <= value <= 1):
            raise ValueError('distinct eligible members and fixed discovery order required')
        for key in ('fp16_bytes', 'fp32_bytes'):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise ValueError('actual positive exported byte counts required')
        ids.add(name); orders.add(order)
    chosen, total, stopped = [], 0, None
    for row in sorted(rows, key=lambda item: (-item['auroc'], item['order'])):
        size = row[precision + '_bytes']
        if total + size > byte_limit:
            stopped = row['id']; break
        chosen.append(row['id']); total += size
    return {'members': chosen, 'bytes': total, 'stopped_before': stopped}


def export_state_files(model, directory):
    """Export and strictly reload all inference state, preserving integer buffers."""
    import torch
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError('new explicit output directory required')
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('tensor-only model state required')
    for value in state.values():
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('finite fp32 source state required')
    half = {name: value.to(torch.float16) if value.is_floating_point() else value.clone()
            for name, value in state.items()}
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in half.values()):
        raise ValueError('fp16 conversion produced nonfinite state')
    files = {}
    try:
        for precision, values in (('fp32', state), ('fp16', half)):
            path = directory / ('weights-' + precision + '.pt')
            with path.open('xb') as stream:
                torch.save(values, stream)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            if (set(loaded) != set(values) or any(not torch.equal(loaded[key], values[key])
                    or loaded[key].dtype != values[key].dtype for key in values)):
                raise ValueError('serialized state differs after reload')
            model.load_state_dict(loaded, strict=True)
            actual = model.state_dict()
            if any(not torch.equal(actual[key].cpu(), values[key].to(dtype=actual[key].dtype)) for key in values):
                raise ValueError('strict model reload differs from serialized state')
            files[precision] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    finally:
        model.load_state_dict(state, strict=True)
    return {'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
            'state_tensor_count': len(state), 'fp32': files['fp32'], 'fp16': files['fp16'],
            'strict_reload': True, 'nonfloating_buffers_preserved': True,
            'fp16_storage_in_fp32_model': True}


def _metrics(labels, values):
    from retrain.n05_metrics import metrics
    return metrics([{'label': int(label), 'logit': float(score)} for label, score in zip(labels, values)])


def compare_predictions(labels, fp32, fp16):
    labels, fp32, fp16 = np.asarray(labels), np.asarray(fp32, dtype=np.float64), np.asarray(fp16, dtype=np.float64)
    if (labels.ndim != 1 or fp32.shape != labels.shape or fp16.shape != labels.shape
            or set(labels.tolist()) != {0, 1} or not np.isfinite(fp32).all() or not np.isfinite(fp16).all()):
        raise ValueError('aligned finite binary dev predictions required')
    return {'fp32': _metrics(labels, fp32), 'fp16': _metrics(labels, fp16),
            'max_abs_logit_difference': float(np.max(np.abs(fp32 - fp16)))}


def equal_zscore(labels, vectors, member_order):
    labels, order = np.asarray(labels), list(member_order)
    if (labels.ndim != 1 or set(labels.tolist()) != {0, 1} or not order
            or len(set(order)) != len(order) or set(order) != set(vectors)):
        raise ValueError('aligned binary dev labels and fixed distinct members required')
    normalized, moments = [], {}
    for name in order:
        values = np.asarray(vectors[name], dtype=np.float64)
        if values.shape != labels.shape or not np.isfinite(values).all():
            raise ValueError('finite aligned dev scores required')
        mean, std = float(values.mean(dtype=np.float64)), float(values.std(dtype=np.float64, ddof=0))
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError('positive finite population standard deviation required')
        normalized.append((values - mean) / std); moments[name] = {'mean': mean, 'std': std}
    scores = np.stack(normalized).mean(axis=0, dtype=np.float64)
    return {'members': order, 'metrics': _metrics(labels, scores), 'normalization': moments, 'scores': scores}

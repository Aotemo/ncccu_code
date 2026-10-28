"""Intact-batch inference with canonical full-split metrics and coherent errors."""
import math
import time

import torch
from torch.utils.data import DataLoader

from . import engine
from .data import ImageDataset
from .persistence import canonical_digest, capture_rng, restore_rng


def batch_assignments(indices, batch_size, rank, world_size):
    indices = list(indices)
    if (type(batch_size) is not int or batch_size <= 0 or type(world_size) is not int or world_size <= 0
            or type(rank) is not int or not 0 <= rank < world_size
            or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices)):
        raise ValueError('invalid evaluation index/batch/rank assignment')
    return [(ordinal, indices[start:start + batch_size]) for ordinal, start in enumerate(range(0, len(indices), batch_size))
            if ordinal % world_size == rank]


def canonical_predictions(rows, indices, packets, batch_size, world_size):
    indices = list(indices)
    work = [batch_assignments(indices, batch_size, rank, world_size) for rank in range(world_size)]
    if (len(packets) != world_size or {p['rank'] for p in packets} != set(range(world_size))
            or any(p.get('error') for p in packets) or any(i >= len(rows) for i in indices)
            or len({rows[i]['name'] for i in indices}) != len(indices)):
        raise ValueError('global evaluation ranks, errors or source identity differ')
    wanted = {index: (rank, ordinal, offset) for rank, assignments in enumerate(work)
              for ordinal, batch in assignments for offset, index in enumerate(batch)}
    observed = {}
    for packet in packets:
        rank = packet['rank']
        for record in packet['records']:
            index = record['index']
            if (type(index) is not int or index in observed or index not in wanted
                    or (record['rank'], record['batch'], record['offset']) != wanted[index]
                    or record['rank'] != rank or record['name'] != rows[index]['name']
                    or record['label'] != int(rows[index]['label'])
                    or not math.isfinite(float(record['logit']))
                    or not 0 <= float(record['probability']) <= 1):
                raise ValueError('evaluation duplicate/extra/identity/batch/label/nonfinite prediction')
            observed[index] = {key: record[key] for key in ('name', 'label', 'logit', 'probability')}
    if set(observed) != set(indices):
        raise ValueError('evaluation full coverage incomplete')
    return [observed[index] for index in indices]


def recovery_checkpoint_due(job, attempted, last_saved_attempt, last_save, now):
    return bool((job.get('checkpoint_updates', 0) and attempted - last_saved_attempt >= job['checkpoint_updates'])
                or now - last_save >= job['checkpoint_seconds'])


def _error(error):
    return {'type': type(error).__name__, 'message': str(error)}


def apply_view(images, view):
    if view == 'identity':
        return images
    if view == 'hflip':
        return images.flip(-1).contiguous()
    raise ValueError('unknown evaluation view')


def evaluate(model, rows, indices, data_root, job, device, ctx, output_csv=None, *, variant, state_identity=None, telemetry=None, view='identity', metrics_enabled=True):
    """All ranks call in the same variant order, including those with no batches.

    Unequal local loops contain no collectives. A local ordinary exception is
    collected before anyone computes/writes metrics. Primary aggregation/write
    errors are broadcast before any rank can return success. Process death uses
    the existing finite process-group timeout and external owner supervision.
    """
    started = time.monotonic()
    setup_error = None
    precision_runtime = None
    try:
        if job.get('precision')=='fp32':
            from .precision import RULE, configure, runtime_proof
            configure(RULE); precision_runtime=runtime_proof(RULE,device=device)
        indices = list(indices)
        if not isinstance(metrics_enabled, bool) or (not metrics_enabled and output_csv is not None):
            raise ValueError('fit-only throughput probe cannot publish predictive metrics or CSV')
        width = int(job.get('physical_batch', 32))
        assignments = batch_assignments(indices, width, ctx.rank, ctx.world_size)
        if variant not in ('online', 'ema') or view not in ('identity', 'hflip') or any(index >= len(rows) for index in indices):
            raise ValueError('evaluation variant or source indices invalid')
        while isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        from .ema import tensor_digest
        if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()
               if t.is_floating_point() or t.is_complex()):
            raise FloatingPointError('nonfinite model tensor')
        actual_tensor_sha256 = tensor_digest(model.state_dict())
        identity = canonical_digest({'variant': variant, 'view': view, 'indices': indices,
            'rows': [(rows[i]['name'], rows[i]['label']) for i in indices],
            'batch': width, 'image_size': job['image_size'], 'precision': job.get('precision', 'fp32'),
            'normalization': job.get('normalization', 'imagenet'), 'prepared_size': job.get('prepared_size', 0),
            'age': job.get('age'), 'actual_model_age_metadata': getattr(model, 'age_metadata', None),
            'actual_tensor_sha256': actual_tensor_sha256, 'state_identity': state_identity, 'precision_runtime': precision_runtime})
    except Exception as error:
        setup_error = _error(error)
        identity = None
    headers = ctx.values({'rank': ctx.rank, 'identity': identity, 'error': setup_error, 'precision_runtime': precision_runtime})
    if any(v['error'] for v in headers) or len({v['identity'] for v in headers}) != 1:
        raise RuntimeError('coordinated evaluation setup/variant/coverage/model-state mismatch: ' + str(headers))

    # Do not invoke DDP forward on uneven loops: buffer synchronizations could
    # otherwise require every rank to run an equal number of forwards.
    while isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    local, local_error, rng, modes = [], None, None, None
    inference_started = time.monotonic()
    try:
        rng = capture_rng()
        modes = [(module, module.training) for module in model.modules()]
        dataset = ImageDataset(data_root, rows, int(job['image_size']), training=False,
            normalization=job.get('normalization', 'imagenet'), prepared_size=int(job.get('prepared_size', 0)), age=job.get('age'))
        loader = DataLoader(dataset, batch_sampler=[batch for _, batch in assignments],
            **engine._loader_options(int(job.get('eval_workers', 0)), job))
        model.eval()
        with torch.inference_mode():
            for (ordinal, expected), batch in zip(assignments, loader):
                from .fresh import runtime_stop_reason
                reason = runtime_stop_reason(job['runtime_deadline_at'], device) if 'runtime_deadline_at' in job else None
                if reason:
                    raise RuntimeError('N03 runtime guard: ' + reason)
                from .age import unpack_batch, forward
                images, labels, row_indices, age_features = unpack_batch(batch, job, device)
                actual_indices = row_indices.tolist()
                if actual_indices != expected:
                    raise ValueError('local dataset changed the original intact batch order')
                images = apply_view(images, view)
                with engine._autocast(device, job.get('precision', 'fp32')):
                    logits = forward(model, images, age_features, job).reshape(-1).float()
                reason = runtime_stop_reason(job['runtime_deadline_at'], device) if 'runtime_deadline_at' in job else None
                if reason:
                    raise RuntimeError('N03 runtime guard: ' + reason)
                if logits.numel() != len(expected) or labels.numel() != len(expected) or not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('invalid or nonfinite local evaluation logits')
                probabilities = logits.sigmoid()
                for offset, (index, label, logit, probability) in enumerate(zip(actual_indices, labels.tolist(), logits.cpu().tolist(), probabilities.cpu().tolist())):
                    local.append({'index': index, 'rank': ctx.rank, 'batch': ordinal, 'offset': offset,
                                  'name': rows[index]['name'], 'label': int(label), 'logit': logit, 'probability': probability})
    except Exception as error:
        local_error = _error(error)
    finally:
        try:
            if modes is not None:
                for module, training in modes:
                    module.training = training
            if rng is not None:
                restore_rng(rng)
        except Exception as error:
            local_error = _error(error)
    local_seconds = time.monotonic() - inference_started
    packets = ctx.values({'rank': ctx.rank, 'error': local_error, 'records': local if local_error is None else [],
                          'local_inference_seconds': local_seconds, 'batch_count': len(assignments), 'sample_count': len(local)})
    failures = [{'rank': p['rank'], 'error': p['error']} for p in packets if p['error']]
    if failures:
        raise RuntimeError('coordinated evaluation local failure: ' + str(failures))
    outcome = None
    if ctx.primary:
        before = time.monotonic()
        try:
            predictions = canonical_predictions(rows, indices, packets, width, ctx.world_size)
            metrics = engine.prediction_metrics(predictions) if metrics_enabled else {'count': len(predictions)}
            if metrics_enabled and job.get('model') in ('flair_resnet50','resnet50_imagenet'):
                from .n05_metrics import metrics as exact_metrics
                metrics.update(exact_metrics(predictions))
            if output_csv is not None:
                engine._write_predictions(output_csv, predictions)
                if (job.get('model') in ('flair_resnet50','resnet50_imagenet') and isinstance(state_identity,dict)
                        and 'successful_updates' in state_identity and 'attempted_updates' in state_identity):
                    from .persistence import sha256_file
                    metrics.update(prediction_sha256=sha256_file(output_csv),
                                   actual_tensor_sha256=actual_tensor_sha256)
                if precision_runtime is not None:
                    from pathlib import Path
                    from .persistence import atomic_json
                    atomic_json(Path(str(output_csv)+'.precision.json'),{'variant':variant,'view':view,'rank_proofs':headers,'state_identity':state_identity,'actual_tensor_sha256':actual_tensor_sha256})
            outcome = {'metrics': metrics, 'error': None}
        except Exception as error:
            outcome = {'metrics': None, 'error': _error(error)}
        outcome['global_metrics_write_seconds'] = time.monotonic() - before
    outcome = ctx.broadcast(outcome)
    if outcome['error']:
        raise RuntimeError('coordinated evaluation primary failure: ' + str(outcome['error']))
    if telemetry is not None:
        telemetry.update(end_to_end_seconds=time.monotonic() - started,
            rank_statistics=[{k: p[k] for k in ('rank', 'batch_count', 'sample_count', 'local_inference_seconds')} for p in packets],
            global_metrics_write_seconds=outcome['global_metrics_write_seconds'], actual_physical_batch=width, variant=variant, view=view,
            actual_tensor_sha256=actual_tensor_sha256, state_identity=state_identity, all_rank_model_identity_matched=True,
            precision_runtime_ranks=[h['precision_runtime'] for h in headers])
    return outcome['metrics']

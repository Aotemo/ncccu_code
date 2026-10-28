"""Configured-rank standalone evaluation using the original frozen-source checks."""
import json
from pathlib import Path
import time

from . import inference
from .parallel_evaluation import evaluate
from .persistence import atomic_json, sha256_file


def run_evaluation(source, data_root, output, *, context, device='cpu', checkpoint='best',
                   expected_count=5849, precision=None, model_factory=None, variant=None,
                   eval_manifest=None, eval_manifest_sha256=None, telemetry=None, expected_source_sha256=None, deadline_at=None):
    output = Path(output)
    sidecar = output.with_suffix('.metrics.json')
    started = time.monotonic()
    error = None
    try:
        if context.primary:
            inference._reserve_outputs(output, sidecar)
        records, csv_sha = inference._read_csv(data_root, 'valid')
        if expected_count is not None and len(records) != expected_count:
            raise ValueError('official evaluation expected count differs')
        names = inference._read_unlabeled_rows(data_root, 'valid', expected_count)
        rows = [dict(name, label=record['label'], age=record['age']) for name, record in zip(names, records)]
        if eval_manifest is not None:
            if not eval_manifest_sha256 or sha256_file(eval_manifest) != eval_manifest_sha256:
                raise ValueError('evaluation manifest SHA mismatch')
            sealed = json.loads(Path(eval_manifest).read_text())
            if (sealed.get('training_eligible') is not False or 'fit_indices' in sealed
                    or sealed.get('source_identity', {}).get('valid_csv_sha256') != csv_sha):
                raise ValueError('official monitor identity or fit isolation differs')
            frozen = sealed['rows']
            if len(frozen) != len(rows) or len({row['name'].casefold() for row in frozen}) != len(rows):
                raise ValueError('official monitor exact coverage differs')
            for actual, frozen_row in zip(rows, frozen):
                if (any(actual[key] != frozen_row[key] for key in ('name', 'label', 'age', 'relative_path'))
                        or not isinstance(frozen_row.get('sha256'), str) or len(frozen_row['sha256']) != 64):
                    raise ValueError('official monitor row/age/label identity differs')
            rows = frozen
        load_started = time.monotonic()
        model, job, metadata, origin, selected_device = inference._load_model(source, device, checkpoint, model_factory, variant)
        if expected_source_sha256 is not None and origin.get('sha256')!=expected_source_sha256:
            raise ValueError('loaded source SHA differs from the admitted frozen checkpoint')
        if job.get('age') is not None and eval_manifest is None:
            raise ValueError('age evaluation requires immutable manifest rows')
        runtime = inference._runtime(job, metadata, precision=precision)
        inference._check_precision(selected_device, runtime['precision'])
        policy = origin.get('identity', {}).get('fresh', {}).get('execution_policy', {})
        job.update(device=str(selected_device), precision=runtime['precision'],
                   physical_batch=policy.get('evaluation_physical_batch', job.get('physical_batch', 32)))
        if deadline_at is not None:job['runtime_deadline_at']=deadline_at
        if selected_device.type == 'cuda':
            import torch
            torch.cuda.synchronize(selected_device)
        load_seconds = time.monotonic() - load_started
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    errors = context.values({'rank': context.rank, 'error': error})
    if any(value['error'] for value in errors):
        raise RuntimeError('coordinated standalone evaluation setup failure: ' + str(errors))
    state_identity = {'source': origin, 'checkpoint_kind': checkpoint, 'variant': variant or 'online'}
    details = {} if telemetry is None else telemetry
    metrics = evaluate(model, rows, list(range(len(rows))), data_root, job, selected_device, context, output,
                       variant=variant or 'online', state_identity=state_identity, telemetry=details)
    outcome = None
    if context.primary:
        try:
            report = {'metrics': metrics, 'inference': runtime, 'source': origin, 'split': 'valid',
                'input_csv_sha256': csv_sha, 'eval_manifest_sha256': eval_manifest_sha256,
                'output_sha256': sha256_file(output), 'world_size': context.world_size,
                'actual_evaluation_physical_batch': job['physical_batch'], 'model_load_seconds_primary': load_seconds,
                'end_to_end_seconds_primary': time.monotonic() - started, 'parallel_evaluation': details}
            atomic_json(sidecar, report)
            outcome = {'report': report, 'error': None}
        except Exception as exc:
            outcome = {'report': None, 'error': {'type': type(exc).__name__, 'message': str(exc)}}
    outcome = context.broadcast(outcome)
    if outcome['error']:
        raise RuntimeError('coordinated standalone evaluation report failure: ' + str(outcome['error']))
    return outcome['report']

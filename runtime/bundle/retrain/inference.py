"""Frozen local model export, ordered prediction, explicit valid evaluation and timing."""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
import time

from .data import ImageDataset, _image_mapping, _read_csv, _safe_name, _safe_path
from .models import create_model
from .persistence import CheckpointStore, atomic_json, load_file, sha256_file

_EXPORT_FORMAT = 'glaucoma-frozen-inference-v1'


def _threshold(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('threshold must be finite and between zero and one')
    return value


def _precision(value):
    if value not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32, fp16 or bf16')
    return value


def _device(value):
    import torch
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('inference supports cpu or visible cuda device')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('requested CUDA-compatible device is unavailable')
        torch.cuda.set_device(device)
    return device


def _load_model(source, device='cpu', checkpoint='best', model_factory=None, variant=None):
    import torch
    source = Path(source)
    exported = source.is_dir() and (source / 'model.json').is_file()
    if exported:
        metadata = json.loads((source / 'model.json').read_text(encoding='utf-8'))
        if metadata.get('format') != _EXPORT_FORMAT:
            raise ValueError('unknown frozen model format')
        model_path = source / 'model.pt'
        if model_path.is_symlink() or sha256_file(model_path) != metadata['model_sha256']:
            raise ValueError('frozen model SHA256 hash mismatch')
        selected_variant=(metadata.get('checkpoint') or {}).get('variant')
        if variant=='ema' and selected_variant!='ema':raise ValueError('requested EMA variant identity absent from export')
        if selected_variant is not None and variant!=selected_variant:raise ValueError('explicit frozen variant differs or absent')
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        job = metadata['job']
        if job.get('age') is not None:
            from .age import validate_spec
            validate_spec(job['age'])
            if metadata.get('age') != job['age']:
                raise ValueError('export age metadata absent or differs')
        origin = {'type': 'export', 'source': str(source.resolve()), 'model_sha256': metadata['model_sha256'],
                  'checkpoint': metadata.get('checkpoint')}
        if selected_variant is not None:
            from .ema import tensor_digest
            expected=(metadata.get('checkpoint') or {}).get('tensor_sha256')
            if not expected or tensor_digest(state)!=expected:raise ValueError('exported variant tensor identity absent or differs')
            origin.update(variant=selected_variant,tensor_sha256=expected,ema=metadata['checkpoint'].get('ema'))
    else:
        if checkpoint not in ('best', 'last'):
            raise ValueError('checkpoint selection must be best or last')
        if source.is_dir():
            directory = source / 'checkpoints' if (source / 'checkpoints').is_dir() else source
            store = CheckpointStore(directory)
            latest = store.load('last', map_location='cpu') if checkpoint == 'best' else None
            selection = (latest or {}).get('lineage_best', {})
            if checkpoint == 'best' and selection.get('kind') == 'parent':
                parent_path = selection['checkpoint']
                if sha256_file(parent_path) != selection['sha256']:
                    raise ValueError('lineage parent best SHA256 mismatch')
                payload = load_file(parent_path, map_location='cpu')
                if (payload.get('identity') != latest.get('continuation', {}).get('parent_identity')
                        or payload.get('manifest_digest') != latest.get('manifest_digest')):
                    raise ValueError('lineage parent best identity mismatch')
            else:
                if latest and latest.get('ema') and variant not in ('online','ema'):raise ValueError('explicit online or ema variant required')
                kind='best_'+variant if checkpoint=='best' and latest and latest.get('ema') else checkpoint
                payload = store.load(kind, map_location='cpu')
            origin = {'type': 'checkpoint-directory', 'source': str(directory.resolve()), 'selection': checkpoint}
        else:
            payload = load_file(source, map_location='cpu')
            origin = {'type': 'checkpoint-file', 'source': str(source.resolve()), 'sha256': sha256_file(source)}
        if not {'job', 'model', 'identity', 'manifest_digest'}.issubset(payload):
            raise ValueError('checkpoint does not contain this training package\'s model/data identity')
        if payload.get('mode') in ('benchmark','fresh-probe','stage-probe'):
            raise ValueError('benchmark/probe checkpoints cannot become a final inference artifact')
        job, state = payload['job'], payload['model']
        from .age import validate_checkpoint
        validate_checkpoint(payload,job)
        if job.get('experiment91') is not None:
            from .experiment91 import validate_checkpoint as validate91
            validate91(payload)
        if payload.get('ema') is not None or variant=='ema':
            from .ema import select_variant,tensor_digest
            state=select_variant(payload,variant)
            origin.update(variant=variant,tensor_sha256=tensor_digest(state),
                          ema={k:v for k,v in payload['ema'].items() if k!='model'})
        elif variant not in (None,'online'):raise ValueError('invalid explicit variant')
        origin.update(identity=payload['identity'], manifest_digest=payload['manifest_digest'],
                      successful_updates=payload.get('successful_updates'))
        metadata = {'job': job, 'inference': {'threshold': .5, 'precision': job.get('precision', 'fp32')}}
    origin['age_contract_sha256'] = (job.get('age') or {}).get('contract_sha256')
    if job.get('model') not in ('efficientnet_v2_s', 'convnext_tiny', 'convnext_small', 'convnext_base', 'swin_base', 'swin_v2_small', 'dinov2_vits14', 'flair_resnet50', 'resnet50_imagenet'):
        raise ValueError('unsupported inference model architecture')
    selected_device = _device(device)
    protocol = origin.get('identity', {}).get('fresh', {}).get('precision_protocol')
    if protocol is not None:
        from .precision import configure, validate_protocol
        validate_protocol(protocol, job['precision']); configure(protocol)
        metadata['precision_protocol'] = protocol
    model = (model_factory or create_model)(job, initialize=False)
    from .age import validate_model
    validate_model(model,job)
    model.load_state_dict(state, strict=True)
    if job.get('mixstyle_policy') is not None:
        from .mixstyle import describe_mixstyle
        origin['mixstyle'] = describe_mixstyle(model, job)
    if job.get('tri3_downsample_policy') is not None:
        from .tri3_downsample import describe_tri3
        origin['tri3_downsample'] = describe_tri3(model, job)
    if job.get('low_rank_policy') is not None:
        from .low_rank import frozen_digest, restore_base_reference
        if exported:
            reference = metadata.get('low_rank_reference')
            if reference is None:
                raise ValueError('exported low-rank base reference missing')
            if reference != frozen_digest(model):
                raise ValueError('exported low-rank base reference differs')
            model.low_rank_initial_frozen_sha256 = reference
        else:
            restore_base_reference(model, job, payload)
    model.to(selected_device).eval()
    return model, dict(job), metadata, origin, selected_device


def _read_unlabeled_rows(data_root, split, expected_count=None):
    """Read only Image_name values, including when a CSV has extraneous labels."""
    if split not in ('train', 'valid', 'test'):
        raise ValueError('unknown image split')
    root = Path(data_root)
    csv_path = root / f'{split}.csv'
    if csv_path.is_symlink() or not csv_path.resolve().is_relative_to(root.resolve()):
        raise ValueError('CSV symlink/path may not alias another split')
    reader = csv.DictReader(io.StringIO(csv_path.read_bytes().decode('utf-8-sig')))
    if not reader.fieldnames or reader.fieldnames.count('Image_name') != 1:
        raise ValueError(f'{split}.csv requires exactly one Image_name column')
    mapping, seen, rows = _image_mapping(root, split), set(), []
    for record in reader:
        name = _safe_name(record['Image_name'])
        key = name.casefold()
        if key in seen:
            raise ValueError(f'duplicate casefold CSV name: {name}')
        seen.add(key)
        if key not in mapping:
            raise ValueError(f'missing {split} image: {name}')
        relative = mapping[key]
        _safe_path(root, relative, split)
        row={'name': name, 'relative_path': relative, 'label': 0}
        if 'Age' in reader.fieldnames:row['age']=record['Age']
        rows.append(row)
    if not rows or (expected_count is not None and len(rows) != expected_count):
        raise ValueError(f'{split} count mismatch: expected {expected_count}, found {len(rows)}')
    return rows


def _runtime(job, metadata, threshold=None, precision=None):
    fixed = metadata.get('inference', {})
    if metadata.get('precision_protocol') is not None:
        from .precision import validate_protocol
        validate_protocol(metadata['precision_protocol'], precision or fixed.get('precision',job.get('precision')))
    return dict(threshold=_threshold(fixed.get('threshold', .5) if threshold is None else threshold),
                precision=_precision(precision or fixed.get('precision', job.get('precision', 'fp32'))),
                image_size=int(job['image_size']), normalization=job.get('normalization', 'imagenet'),
                prepared_size=int(job.get('prepared_size', 0)),
                preprocessing=('rgb-conservative-border-prepared-letterbox-v2' if job.get('prepared_size')
                               else 'rgb-conservative-border-letterbox-v1'))


def _check_precision(device, precision):
    import torch
    if precision == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('bf16 inference requires native hardware support')


def _autocast(device, precision):
    import torch
    _check_precision(device, precision)
    if precision == 'fp32':
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def _forward(model, images, device, precision, age_features=None):
    import torch
    with torch.inference_mode(), _autocast(device, precision):
        images=images.to(device)
        logits = (model(images,age_features.to(device,dtype=torch.float32)) if age_features is not None else model(images)).reshape(-1).float()
    if logits.numel() != images.shape[0] or not torch.isfinite(logits).all():
        raise FloatingPointError('model produced invalid/nonfinite logits')
    probabilities = logits.sigmoid().cpu().tolist()
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise FloatingPointError('model produced invalid probabilities')
    return probabilities


def _reserve_outputs(*paths):
    for path in paths:
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'refusing to overwrite existing output: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _csv_output(destination, fields):
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    try:
        os.fchmod(descriptor,0o644)
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            yield writer, handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_prediction(source, data_root, output, *, device='cpu', checkpoint='best',
                   expected_count=5855, threshold=None, batch_size=32, precision=None, model_factory=None, variant=None):
    import torch
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    output = Path(output)
    sidecar = output.with_suffix('.metadata.json')
    _reserve_outputs(output, sidecar)
    rows = _read_unlabeled_rows(data_root, 'test', expected_count)
    model, job, metadata, origin, selected_device = _load_model(source, device, checkpoint, model_factory, variant)
    runtime = _runtime(job, metadata, threshold, precision)
    dataset = ImageDataset(data_root, rows, runtime['image_size'], False, runtime['normalization'],
                           prepared_size=runtime['prepared_size'],age=job.get('age'),
                           input_preprocessing=job.get('input_preprocessing'))
    with _csv_output(output, ['Image_name', 'Probability', 'Pred_Label']) as (writer, _):
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            batch=[dataset[index] for index in range(start,stop)]
            images=torch.stack([item[0] for item in batch])
            from .age import enabled
            ages=torch.stack([item[3] for item in batch]) if enabled(job) else None
            probabilities = _forward(model, images, selected_device, runtime['precision'],ages)
            for row, probability in zip(rows[start:stop], probabilities):
                writer.writerow({'Image_name': row['name'], 'Probability': probability,
                                 'Pred_Label': int(probability >= runtime['threshold'])})
    report = dict(count=len(rows), threshold=runtime['threshold'], inference=runtime, source=origin,
                  output_sha256=sha256_file(output), input_csv_sha256=sha256_file(Path(data_root) / 'test.csv'),
                  test_label_values_read=False, device=str(selected_device))
    atomic_json(sidecar, report)
    return report


def run_evaluation(source, data_root, output, *, device='cpu', checkpoint='best',
                   expected_count=5849, precision=None, model_factory=None, variant=None,
                   eval_manifest=None,eval_manifest_sha256=None,context=None,telemetry=None,expected_source_sha256=None,deadline_at=None):
    if context is not None:
        from .parallel_inference import run_evaluation as parallel_run
        return parallel_run(source,data_root,output,context=context,device=device,checkpoint=checkpoint,
            expected_count=expected_count,precision=precision,model_factory=model_factory,variant=variant,
            eval_manifest=eval_manifest,eval_manifest_sha256=eval_manifest_sha256,telemetry=telemetry,expected_source_sha256=expected_source_sha256,deadline_at=deadline_at)
    from .engine import evaluate
    output = Path(output)
    sidecar = output.with_suffix('.metrics.json')
    _reserve_outputs(output, sidecar)
    records, csv_sha = _read_csv(data_root, 'valid')
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f'valid count mismatch: expected {expected_count}, found {len(records)}')
    names = _read_unlabeled_rows(data_root, 'valid', expected_count)
    rows = [dict(n, label=r['label'], age=r['age']) for n, r in zip(names, records)]
    if eval_manifest is not None:
        if not eval_manifest_sha256 or sha256_file(eval_manifest)!=eval_manifest_sha256:
            raise ValueError('evaluation manifest SHA mismatch')
        sealed=json.loads(Path(eval_manifest).read_text())
        if (sealed.get('training_eligible') is not False or 'fit_indices' in sealed
                or sealed.get('source_identity',{}).get('valid_csv_sha256')!=csv_sha):
            raise ValueError('official monitoring manifest identity / fit isolation differs')
        frozen=sealed['rows']
        if len(frozen)!=len(rows) or len({r['name'].casefold() for r in frozen})!=len(rows):
            raise ValueError('official monitoring manifest coverage differs')
        for actual,expected in zip(rows,frozen):
            if (any(actual[k]!=expected[k] for k in ('name','label','age','relative_path'))
                    or not isinstance(expected.get('sha256'),str) or len(expected['sha256'])!=64):
                raise ValueError('official monitoring row/age/label identity differs')
        rows=frozen
    model, job, metadata, origin, selected_device = _load_model(source, device, checkpoint, model_factory, variant)
    if job.get('age') is not None and eval_manifest is None:
        raise ValueError('R8 evaluation requires sealed manifest rows including age and SHA')
    runtime = _runtime(job, metadata, precision=precision)
    _check_precision(selected_device, runtime['precision'])
    job.update(device=str(selected_device), precision=runtime['precision'])
    metrics = evaluate(model, rows, list(range(len(rows))), data_root, job, selected_device, output)
    report = dict(metrics=metrics, inference=runtime, source=origin, split='valid', input_csv_sha256=csv_sha,
                  eval_manifest_sha256=eval_manifest_sha256,
                  output_sha256=sha256_file(output))
    atomic_json(sidecar, report)
    return report


def _copy_materials(source, job, metadata, origin, destination, provenance=None):
    if job.get('initialization', {}).get('kind') != 'imagenet':
        return None
    if job.get('model') == 'resnet50_imagenet':
        from .models import copy_imagenet_materials
        return copy_imagenet_materials(source, job, metadata, origin, destination, provenance)
    existing = metadata.get('materials')
    initial_identity = origin.get('identity', {}).get('initialization', {})
    if provenance is not None:
        proof_path = Path(provenance)
    elif existing:
        proof_path = Path(source) / existing['provenance']
        if not proof_path.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError('exported material path escapes its artifact')
    else:
        proof_path = Path(job['initialization']['provenance'])
    if existing and provenance is None:
        expected_proof = existing.get('provenance_sha256')
    else:
        expected_proof = initial_identity.get('provenance_sha256') or (existing or {}).get('original_provenance_sha256')
    expected_license = initial_identity.get('license_sha256') or (existing or {}).get('license_sha256')
    if not expected_proof or not expected_license:
        raise ValueError('checkpoint/export lacks frozen provenance and license identity hashes')
    if sha256_file(proof_path) != expected_proof:
        raise ValueError('provenance hash differs from frozen training/material identity')
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    license_path = proof_path.parent / proof['license_file']
    if existing and provenance is None and not license_path.resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('exported license path escapes its artifact')
    if not license_path.is_file() or not license_path.read_text(encoding='utf-8').strip():
        raise ValueError('public initialization requires a full nonempty applicable license')
    if sha256_file(license_path) != expected_license:
        raise ValueError('license hash differs from frozen training/material identity')
    if proof.get('redistribution_confirmed') is not True or proof.get('sha256') != job['initialization']['sha256']:
        raise ValueError('public initialization provenance does not confirm matching licensed weights')
    if (proof.get('model') != job['model'] or proof.get('training_data') != 'ImageNet-1K'
            or not str(proof.get('source_url', '')).startswith('https://')):
        raise ValueError('public initialization provenance model/source does not match this training job')
    folder = destination / 'materials'
    folder.mkdir()
    shutil.copy2(license_path, folder / 'LICENSE.txt')
    original_hash = sha256_file(proof_path)
    proof['license_file'] = 'LICENSE.txt'
    atomic_json(folder / 'provenance.json', proof)
    return {'provenance': 'materials/provenance.json', 'license': 'materials/LICENSE.txt',
            'original_provenance_sha256': (existing or {}).get('original_provenance_sha256', original_hash),
            'provenance_sha256': sha256_file(folder / 'provenance.json'),
            'license_sha256': sha256_file(folder / 'LICENSE.txt')}


def export_model(source, output, *, checkpoint='best', threshold=.5, precision=None,
                 provenance=None, model_factory=None, variant=None):
    import torch
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'export directory already exists: {output}')
    model, job, metadata, origin, _ = _load_model(source, 'cpu', checkpoint, model_factory, variant)
    runtime = _runtime(job, metadata, threshold, precision)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    try:
        os.chmod(temporary,0o755)  # new export must be traversable by the storage account
        with (temporary / 'model.pt').open('wb') as handle:
            os.fchmod(handle.fileno(),0o644)
            torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        package = Path(__file__).resolve().parent
        (temporary / 'retrain').mkdir()
        files = ['__init__.py', 'data.py', 'verification.py', 'preprocessing.py',
                 'engine.py', 'inference.py', 'models.py', 'persistence.py', 'ema.py','finite.py','age.py','attention_pool.py','stage3_residual.py','imagenet_resnet.py']
        if job.get('low_rank_policy') is not None:
            files += ['low_rank.py', 'config.py', 'spatial_stride.py']
        for name in files:
            shutil.copy2(package / name, temporary / 'retrain' / name)
        for entry in ('predict.py', 'cost.py'):
            shutil.copy2(package.parent / entry, temporary / entry)
        requirements = package.parent / 'requirements.txt'
        if requirements.is_file():
            shutil.copy2(requirements, temporary / 'requirements.txt')
        materials = _copy_materials(source, job, metadata, origin, temporary, provenance)
        code_hashes = {str(path.relative_to(temporary)): sha256_file(path)
                       for path in sorted(temporary.rglob('*.py'))}
        versions = {}
        for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn'):
            versions[name] = importlib.metadata.version(name)
        result = dict(format=_EXPORT_FORMAT, model_sha256=sha256_file(temporary / 'model.pt'),
                      job=job,age=job.get('age'), inference=runtime, checkpoint=origin, materials=materials,
                      code_sha256=code_hashes, dependencies=versions,
                      model_bytes=(temporary / 'model.pt').stat().st_size)
        if job.get('low_rank_policy') is not None:
            result['low_rank_reference'] = model.low_rank_initial_frozen_sha256
        for member in temporary.rglob('*'):
            os.chmod(member,0o755 if member.is_dir() else 0o644)
        result['artifact_bytes'], result['within_500_mb_weight_limit'] = 0, True
        # Metadata itself contributes bytes. Iterate until the serialized count
        # includes its own digits, so threshold reports use the delivered size.
        while True:
            atomic_json(temporary / 'model.json', result)
            actual_bytes = sum(p.stat().st_size for p in temporary.rglob('*') if p.is_file())
            if actual_bytes == result['artifact_bytes']:
                break
            result['artifact_bytes'] = actual_bytes
            result['within_500_mb_weight_limit'] = actual_bytes <= 500_000_000
        os.rename(temporary, output)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def measure_latency(source, data_root, output, *, device='cpu', checkpoint='best', split='valid',
                    samples=100, rounds=3, warmup=5, precision=None, model_factory=None, variant=None):
    import numpy as np
    import torch
    if any(not isinstance(v, int) or v <= 0 for v in (samples, rounds)) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('samples and rounds must be positive, warmup nonnegative')
    output = Path(output)
    predictions_path = output.with_suffix('.predictions.csv')
    _reserve_outputs(output, predictions_path)
    rows = _read_unlabeled_rows(data_root, split)
    if samples > len(rows) or warmup > len(rows):
        raise ValueError('not enough images for requested latency protocol')
    model_started = time.perf_counter()
    model, job, metadata, origin, selected_device = _load_model(source, device, checkpoint, model_factory, variant)
    model_load_seconds = time.perf_counter() - model_started
    runtime = _runtime(job, metadata, precision=precision)
    dataset = ImageDataset(data_root, rows, runtime['image_size'], False, runtime['normalization'],
                           prepared_size=runtime['prepared_size'],age=job.get('age'),
                           input_preprocessing=job.get('input_preprocessing'))
    from .age import enabled
    def predict_one(index):
        item=dataset[index]
        ages=item[3].unsqueeze(0) if enabled(job) else None
        return _forward(model,item[0].unsqueeze(0),selected_device,runtime['precision'],ages)
    sync = lambda: torch.cuda.synchronize(selected_device) if selected_device.type == 'cuda' else None
    cold_started = time.perf_counter()
    first_prediction_seconds = None
    for index in range(warmup):
        sync()
        prediction_started = time.perf_counter()
        predict_one(index)
        sync()
        if index == 0:
            first_prediction_seconds = time.perf_counter() - prediction_started
    sync()
    warmup_seconds = time.perf_counter() - cold_started
    durations = [[] for _ in range(samples)]
    with _csv_output(predictions_path, ['round', 'Image_name', 'Probability', 'Pred_Label']) as (writer, handle):
        for repetition in range(rounds):
            for index in range(samples):
                sync()
                started = time.perf_counter()
                # Timed path includes open/decode/transform, H2D, forward,
                # probability transfer and output write/flush, at batch one.
                probability = predict_one(index)[0]
                writer.writerow({'round': repetition + 1, 'Image_name': rows[index]['name'],
                                 'Probability': probability, 'Pred_Label': int(probability >= runtime['threshold'])})
                handle.flush()
                sync()
                durations[index].append((time.perf_counter() - started) * 1000)
    per_image_means = np.mean(durations, axis=1)
    device_name = torch.cuda.get_device_name(selected_device) if selected_device.type == 'cuda' else platform.processor() or platform.machine()
    eligible = selected_device.type == 'cuda' and re.search(r'\bT4\b', device_name, re.IGNORECASE) is not None
    standard = samples == 100 and rounds == 3 and warmup == 5
    mean = float(np.mean(per_image_means))
    report = dict(samples=samples, rounds=rounds, batch_size=1, warmup=warmup,
                  model_load_seconds=model_load_seconds, first_prediction_seconds=first_prediction_seconds,
                  warmup_seconds=warmup_seconds, warmup_excluded=True, inference=runtime,
                  source=origin, split=split, input_csv_sha256=sha256_file(Path(data_root) / f'{split}.csv'),
                  names=[r['name'] for r in rows[:samples]], per_image_round_ms=durations,
                  per_image_mean_ms=per_image_means.tolist(), mean_ms=mean,
                  p95_ms=float(np.percentile(per_image_means, 95)),
                  cuda_synchronized=selected_device.type == 'cuda',
                  timing_scope='read/decode/preprocess/H2D/forward/probability-to-host/output-write-flush',
                  hardware={'device': str(selected_device), 'name': device_name, 'torch': torch.__version__,
                            'cuda': torch.version.cuda, 'hip': torch.version.hip, 'platform': platform.platform()},
                  official_rule={'eligible_t4_hardware': eligible, 'standard_protocol': standard,
                                 'threshold_ms': 100, 'pass': mean <= 100 if eligible and standard else None})
    atomic_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('predict', 'evaluate', 'export', 'latency'):
        sub = commands.add_parser(name)
        sub.add_argument('--source', required=True, help='job/checkpoint directory, checkpoint blob with sidecar, or exported model directory')
        sub.add_argument('--checkpoint', choices=('best', 'last'), default='best')
        sub.add_argument('--variant',choices=('online','ema'),help='required for coupled EMA checkpoints and their exports')
        sub.add_argument('--output', required=True)
        sub.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'))
        if name != 'export':
            sub.add_argument('--data-root', required=True)
            sub.add_argument('--device', default='cpu')
        if name == 'predict':
            sub.add_argument('--expected-count', type=int, default=5855)
            sub.add_argument('--threshold', type=float)
            sub.add_argument('--batch-size', type=int, default=32)
        elif name == 'evaluate':
            sub.add_argument('--expected-count', type=int, default=5849)
            sub.add_argument('--eval-manifest')
            sub.add_argument('--eval-manifest-sha256')
        elif name == 'export':
            sub.add_argument('--threshold', type=float, default=.5)
            sub.add_argument('--provenance', help='relocated original public-weight provenance JSON, if needed')
        else:
            sub.add_argument('--split', choices=('train', 'valid', 'test'), default='valid')
            sub.add_argument('--samples', type=int, default=100)
            sub.add_argument('--rounds', type=int, default=3)
            sub.add_argument('--warmup', type=int, default=5)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    function = {'predict': run_prediction, 'evaluate': run_evaluation, 'export': export_model,
                'latency': measure_latency}[command]
    result = function(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0

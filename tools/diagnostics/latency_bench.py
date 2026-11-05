"""End-to-end latency bench for the F06 member set (CPU decode, shared forwards).

Reuses the submission chain rather than a parallel one: models are built by the
frozen ``retrain.inference._load_model`` and each JPEG is decoded, cropped,
resized, letterboxed and normalised by ``retrain.data.ImageDataset`` exactly as
``run_prediction`` does at submission time (prepared cache absent, original
bytes read at call time). Every image is decoded and preprocessed once per
batch and shared by all members; each member then runs its own forward, and the
member logits are averaged before the final probability.

Timing protocol: batch size 10, fp16 (autocast) or fp32, warmup, then at least
five repetitions; per-stage and total per-image medians are reported. GPU
intervals are bracketed by ``torch.cuda.synchronize``. FLOPs come from
``torch.utils.flop_counter.FlopCounterMode``; when it is unavailable the value
is reported as unknown rather than estimated silently.

A T4 deployment estimate is derived two ways: FLOPs divided by an assumed
effective T4 throughput (fp16 12 TFLOPS / fp32 4.5 TFLOPS by default) plus the
measured CPU preprocessing, and measured DCU GPU time scaled by an explicit
ratio. Both are assumptions, recorded in the report, and the pass rule is
``estimate x 1.5 < 100 ms`` for every known estimate.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')
MANIFEST_FORMAT = 'f06-latency-members-v1'
REPORT_FORMAT = 'f06-latency-bench-v1'
DEFAULT_T4_TFLOPS = {'fp16': 12.0, 'fp32': 4.5}
BUDGET_MS = 100.0
SAFETY_FACTOR = 1.5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bundle_root():
    return Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'


def _bundle_imports():
    root = str(bundle_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from retrain import inference
    from retrain.age import forward as age_forward
    return inference, age_forward


def default_loader(model_path, variant):
    """Build one member through the same frozen loader the submission chain uses."""
    inference, _ = _bundle_imports()
    model, job, metadata, origin, _ = inference._load_model(model_path, 'cpu', variant=variant)
    return {'model': model.eval(), 'job': dict(job), 'metadata': metadata, 'origin': origin}


def weights_file(path):
    path = Path(path)
    return path / 'model.pt' if path.is_dir() else path


def load_members(manifest_path, loader=None):
    """Verify each member's weight bytes and build it with the frozen loader."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != MANIFEST_FORMAT:
        raise ValueError(MANIFEST_FORMAT + ' manifest required')
    rows = manifest.get('members')
    if not isinstance(rows, list) or not rows:
        raise ValueError('at least one member required')
    loader = loader or default_loader
    members, identifiers = [], []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('non-empty string member id required')
        if row['id'] in identifiers:
            raise ValueError('duplicate member id: ' + row['id'])
        identifiers.append(row['id'])
        reference = row.get('model')
        if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
                or not isinstance(reference['path'], str) or not isinstance(reference['sha256'], str)
                or len(reference['sha256']) != 64):
            raise ValueError('exact {path, sha256} model reference required for ' + row['id'])
        variant = row.get('variant')
        if variant not in (None, 'online', 'ema'):
            raise ValueError('variant must be online or ema when given')
        file = weights_file(reference['path'])
        if not file.is_file() or file.is_symlink():
            raise ValueError('member weights file absent: ' + str(file))
        if sha(file) != reference['sha256']:
            raise ValueError('member weights differ from the pinned SHA: ' + str(file))
        built = loader(reference['path'], variant)
        model = built['model']
        if not isinstance(built.get('job'), dict):
            raise ValueError('member job metadata required: ' + row['id'])
        members.append({'id': row['id'], 'path': str(reference['path']),
                        'sha256': reference['sha256'], 'weights': str(file),
                        'variant': variant, 'model': model, 'job': built['job'],
                        'origin': built.get('origin')})
    _check_shared_preprocessing(members)
    return {'path': str(manifest_path), 'sha256': sha(manifest_path), 'members': members}


def _preprocessing_identity(job):
    return {'image_size': int(job['image_size']),
            'normalization': job.get('normalization', 'imagenet'),
            'prepared_size': int(job.get('prepared_size', 0)),
            'input_preprocessing': job.get('input_preprocessing'),
            'age_contract_sha256': (job.get('age') or {}).get('contract_sha256')}


def _check_shared_preprocessing(members):
    reference = _preprocessing_identity(members[0]['job'])
    for member in members[1:]:
        if _preprocessing_identity(member['job']) != reference:
            raise ValueError('members must share image preprocessing and age contract: ' + member['id'])


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('image directory required: ' + str(directory))
    files, seen = [], set()
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name.casefold() in seen:
            raise ValueError('ambiguous casefold image name: ' + path.name)
        seen.add(path.name.casefold())
        files.append(path)
    if not files:
        raise ValueError('no supported images found in ' + str(directory))
    return files


def build_dataset(images, job, work_root):
    """Expose the flat image directory as a read-only test split for ImageDataset."""
    _, _ = _bundle_imports()
    from retrain.data import ImageDataset
    # The platform predictor has no prepared cache: decode the shipped JPEGs.
    os.environ.pop('TRAIN_IMAGE_CACHE', None)
    os.environ.pop('TRAIN_GLOBAL_IMAGE_CACHE', None)
    root = Path(work_root) / 'data'
    (root / 'test').mkdir(parents=True)
    rows = []
    for path in images:
        link = root / 'test' / path.name
        try:
            os.link(path, link)
        except OSError:
            shutil.copy2(path, link)
        rows.append({'name': path.name, 'relative_path': 'test/' + path.name,
                     'label': 0, 'age': ''})
    dataset = ImageDataset(root, rows, int(job['image_size']), False,
                           job.get('normalization', 'imagenet'),
                           prepared_size=int(job.get('prepared_size', 0)), age=job.get('age'),
                           input_preprocessing=job.get('input_preprocessing'))
    return dataset, rows, root


def _load_batch(dataset, indices, *, decode_threads=1, pool=None):
    if decode_threads > 1:
        if pool is None:
            raise ValueError('thread pool required for parallel decode')
        items = list(pool.map(dataset.__getitem__, indices))
    else:
        items = [dataset[index] for index in indices]
    images = torch.stack([item[0] for item in items])
    ages = torch.stack([item[3] for item in items]) if len(items[0]) == 4 else None
    return images, ages


def assert_parallel_decode_equivalence(dataset, batches, *, decode_threads):
    """Prove the parallel decode/preprocess output is bit-identical to serial."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=decode_threads) as pool:
        for batch in batches:
            serial = _load_batch(dataset, batch)
            parallel = _load_batch(dataset, batch, decode_threads=decode_threads, pool=pool)
            if not torch.equal(serial[0], parallel[0]):
                raise ValueError('parallel decode images differ from serial output')
            if (serial[1] is None) != (parallel[1] is None):
                raise ValueError('parallel decode age features differ')
            if serial[1] is not None and not torch.equal(serial[1], parallel[1]):
                raise ValueError('parallel decode age features differ')
    return {'checked_batches': len(batches), 'decode_threads': decode_threads,
            'images_bit_identical': True, 'age_bit_identical': True}


def _forward_logits(model, job, images, ages):
    _, age_forward = _bundle_imports()
    return age_forward(model, images, ages, job).reshape(-1).float()


def _autocast(device, precision):
    inference, _ = _bundle_imports()
    return inference._autocast(device, precision)


def _median(values):
    return float(statistics.median(values)) if values else None


def summarize(records, member_ids):
    """Median per-image milliseconds from raw per-batch records."""
    def per_image(record, key):
        return record[key] * 1000.0 / record['batch']
    return {
        'cpu_preprocess_ms_per_image': _median([per_image(r, 'cpu') for r in records]),
        'h2d_ms_per_image': _median([per_image(r, 'h2d') for r in records]),
        'fuse_ms_per_image': _median([per_image(r, 'fuse') for r in records]),
        'total_ms_per_image': _median([per_image(r, 'total') for r in records]),
        'members': {member_id: _median([per_image(r, ('member', member_id)) for r in records])
                    for member_id in member_ids},
        'batches': len(records),
    }


def measure(members, dataset, indices, *, device, precision, batch_size,
            warmup, repetitions, clock=time.perf_counter, synchronize=None,
            decode_threads=1):
    """Run warmed batches, returning raw per-batch timings in seconds."""
    if repetitions < 5:
        raise ValueError('at least five timed repetitions required')
    if batch_size <= 0 or not indices:
        raise ValueError('positive batch size and at least one image required')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('requested CUDA device is unavailable')
        torch.cuda.set_device(device)
        synchronize = synchronize or (lambda: torch.cuda.synchronize(device))
    else:
        synchronize = synchronize or (lambda: None)
    batches = [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]
    models = [member['model'].to(device).eval() for member in members]
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=decode_threads) if decode_threads > 1 else None

    def one_batch(indices_batch, record=None):
        total_started = clock() if record is not None else None
        started = clock() if record is not None else None
        images, ages = _load_batch(dataset, indices_batch, decode_threads=decode_threads, pool=pool)
        if record is not None:
            record['cpu'] = clock() - started
            started = clock()
        synchronize()
        images = images.to(device, non_blocking=False)
        ages = ages.to(device, dtype=torch.float32) if ages is not None else None
        synchronize()
        if record is not None:
            record['h2d'] = clock() - started
        logits = []
        for member, model in zip(members, models):
            if record is not None:
                synchronize()
                started = clock()
            with torch.inference_mode(), _autocast(device, precision):
                value = _forward_logits(model, member['job'], images, ages)
            if record is not None:
                synchronize()
                record[('member', member['id'])] = clock() - started
            logits.append(value)
        if record is not None:
            started = clock()
        probabilities = torch.stack(logits).mean(dim=0).sigmoid().cpu().tolist()
        if record is not None:
            record['fuse'] = clock() - started
            record['total'] = clock() - total_started
            record['batch'] = len(indices_batch)
        if len(probabilities) != len(indices_batch) or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise FloatingPointError('invalid fused probabilities')
        return probabilities

    for batch in batches:
        for _ in range(warmup):
            one_batch(batch)
    records = []
    for _ in range(repetitions):
        for batch in batches:
            record = {}
            one_batch(batch, record)
            records.append(record)
    if pool is not None:
        pool.shutdown()
    return records


def count_flops(member, images, ages, device, precision, batch_size):
    """Per-image multiply-adds from FlopCounterMode, or unknown."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None, 'unknown'
    try:
        model = member['model'].to(device).eval()
        with FlopCounterMode(display=False) as counter:
            with torch.inference_mode(), _autocast(device, precision):
                _forward_logits(model, member['job'], images.to(device),
                                ages.to(device, dtype=torch.float32) if ages is not None else None)
        total = int(counter.get_total_flops())
    except Exception:
        return None, 'unknown'
    if total <= 0:
        return None, 'unknown'
    return total / float(batch_size), 'torch_flop_counter'


def _estimate_side(gpu_ms, cpu_ms, safety, budget):
    if gpu_ms is None:
        return {'gpu_ms_per_image': None, 'total_ms_per_image': None,
                'with_safety_ms': None, 'within_budget': None}
    total = gpu_ms + cpu_ms
    with_safety = total * safety
    return {'gpu_ms_per_image': gpu_ms, 'total_ms_per_image': total,
            'with_safety_ms': with_safety, 'within_budget': with_safety < budget}


def _combinations(items, size):
    if size == 0:
        return [()]
    result = []
    for index, item in enumerate(items):
        for rest in _combinations(items[index + 1:], size - 1):
            result.append((item,) + rest)
    return result


def estimate(members, cpu_ms, *, t4_tflops, dcu_t4_ratio, safety=SAFETY_FACTOR, budget=BUDGET_MS):
    """T4 estimates per subset size (1..n), best (cheapest) subset of each size."""
    if not members:
        raise ValueError('at least one member required')
    if not isinstance(t4_tflops, (int, float)) or not math.isfinite(t4_tflops) or t4_tflops <= 0:
        raise ValueError('positive finite T4 effective TFLOPS required')
    if not isinstance(dcu_t4_ratio, (int, float)) or not math.isfinite(dcu_t4_ratio) or dcu_t4_ratio <= 0:
        raise ValueError('positive finite DCU-to-T4 ratio required')
    if not isinstance(cpu_ms, (int, float)) or not math.isfinite(cpu_ms) or cpu_ms < 0:
        raise ValueError('nonnegative finite CPU preprocessing time required')
    prepared = []
    for member in members:
        flops = member.get('flops_per_image')
        gpu = member.get('gpu_ms_per_image')
        if gpu is None or not math.isfinite(gpu) or gpu < 0:
            raise ValueError('finite measured GPU time required: ' + str(member.get('id')))
        flops_ms = (flops / (t4_tflops * 1e12) * 1000.0
                    if isinstance(flops, (int, float)) and math.isfinite(flops) and flops > 0 else None)
        prepared.append({'id': member['id'], 'flops_ms': flops_ms,
                         'dcu_ms': float(gpu) * float(dcu_t4_ratio)})
    subsets = {}
    for size in range(1, len(prepared) + 1):
        candidates = []
        for combo in _combinations(prepared, size):
            flops_sum = sum(item['flops_ms'] for item in combo) if all(
                item['flops_ms'] is not None for item in combo) else None
            dcu_sum = sum(item['dcu_ms'] for item in combo)
            by_flops = _estimate_side(flops_sum, cpu_ms, safety, budget)
            by_dcu = _estimate_side(dcu_sum, cpu_ms, safety, budget)
            conservative = max(value for value in (by_flops['total_ms_per_image'],
                                                   by_dcu['total_ms_per_image']) if value is not None)
            known = by_flops['within_budget']
            passed = by_dcu['within_budget'] if known is None else (known and by_dcu['within_budget'])
            candidates.append({'members': [item['id'] for item in combo],
                               'by_flops': by_flops, 'by_dcu_scale': by_dcu,
                               'conservative_total_ms': conservative, 'pass': passed})
        candidates.sort(key=lambda value: (value['conservative_total_ms'], value['members']))
        best = dict(candidates[0])
        best['candidate_count'] = len(candidates)
        best['all_candidates'] = candidates
        subsets[str(size)] = best
    full = subsets[str(len(prepared))]
    return {'assumptions': {'t4_effective_tflops': float(t4_tflops),
                            'dcu_to_t4_ratio': float(dcu_t4_ratio),
                            'safety_factor': float(safety), 'budget_ms': float(budget),
                            'cpu_note': 'measured on this host; T4 host CPU may differ'},
            'subset_estimates': subsets, 'pass': bool(full['pass'])}


def run_bench(*, members_path, images_path, device='cpu', precision='fp16', batch_size=10,
              warmup=2, repetitions=5, t4_tflops=None, dcu_t4_ratio=1.0,
              max_images=0, loader=None, decode_threads=1):
    if precision not in ('fp16', 'fp32'):
        raise ValueError('precision must be fp16 or fp32')
    if repetitions < 5:
        raise ValueError('at least five timed repetitions required')
    t4_tflops = DEFAULT_T4_TFLOPS[precision] if t4_tflops is None else float(t4_tflops)
    loaded = load_members(members_path, loader=loader)
    members = loaded['members']
    job = members[0]['job']
    files = list_images(images_path)
    if max_images:
        files = files[:max_images]
    target = torch.device(device)
    with tempfile.TemporaryDirectory(prefix='f06-latency-') as work:
        dataset, rows, _ = build_dataset(files, job, work)
        indices = list(range(len(rows)))
        batches = [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]
        equivalence = (assert_parallel_decode_equivalence(dataset, batches, decode_threads=decode_threads)
                       if decode_threads > 1 else None)
        records = measure(members, dataset, indices, device=target, precision=precision,
                          batch_size=batch_size, warmup=warmup, repetitions=repetitions,
                          decode_threads=decode_threads)
        summary = summarize(records, [member['id'] for member in members])
        example_images, example_ages = _load_batch(dataset, indices[:min(batch_size, len(indices))])
    flops = {}
    for member in members:
        value, method = count_flops(member, example_images, example_ages, target, precision,
                                    min(batch_size, len(indices)))
        flops[member['id']] = {'value': value, 'method': method}
    measured = []
    for member in members:
        measured.append({'id': member['id'], 'path': member['path'], 'sha256': member['sha256'],
                         'variant': member['variant'],
                         'gpu_ms_per_image': summary['members'][member['id']],
                         'flops_per_image': flops[member['id']]['value'],
                         'flops_method': flops[member['id']]['method']})
    estimates = estimate(measured, summary['cpu_preprocess_ms_per_image'],
                         t4_tflops=t4_tflops, dcu_t4_ratio=dcu_t4_ratio)
    device_name = torch.cuda.get_device_name(target) if target.type == 'cuda' else 'cpu'
    return {
        'format': REPORT_FORMAT, 'status': 'measured', 'at': time.time(),
        'manifest': {'path': loaded['path'], 'sha256': loaded['sha256']},
        'images': {'path': str(Path(images_path)), 'count': len(rows), 'batch_size': batch_size},
        'protocol': {'precision': precision, 'warmup': warmup, 'repetitions': repetitions,
                     'timed_batches': summary['batches'],
                     'timing_scope': 'decode+crop+resize+letterbox+normalise once per batch, shared; '
                                     'H2D once; each member forward; equal-weight fuse; sigmoid',
                     'preprocessing': 'original JPEG decode with prepared cache disabled, '
                                      'same as run_prediction on the platform',
                     'gpu_timing': 'torch.cuda.synchronize brackets',
                     'batch_size_allowed_by_platform': 10,
                     'decode_threads': decode_threads,
                     'decode_equivalence': equivalence},
        'hardware': {'device': str(target), 'name': device_name, 'torch': torch.__version__,
                     'cuda': torch.version.cuda, 'hip': torch.version.hip,
                     'gpu_count': torch.cuda.device_count() if target.type == 'cuda' else 0,
                     'cpu': platform.processor() or platform.machine(),
                     'cpu_count': os.cpu_count()},
        'measurements': {'cpu_preprocess_ms_per_image': summary['cpu_preprocess_ms_per_image'],
                         'h2d_ms_per_image': summary['h2d_ms_per_image'],
                         'fuse_ms_per_image': summary['fuse_ms_per_image'],
                         'total_ms_per_image': summary['total_ms_per_image'],
                         'members': measured},
        'estimates': estimates,
        'subset_estimates': estimates['subset_estimates'],
        'pass': estimates['pass'],
        'official_read': False, 'test_label_values_read': False, 'TTA': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True,
                        help='directory holding only the benchmark JPG/PNG files')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--t4-tflops', type=float)
    parser.add_argument('--dcu-t4-ratio', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--decode-threads', type=int, default=1,
                        help='parallel per-image decode/preprocess threads; 1 keeps serial')
    args = parser.parse_args()
    if args.decode_threads < 1:
        raise SystemExit('--decode-threads must be >= 1')
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS'):
            os.environ[key] = str(args.threads)
        try:
            from threadpoolctl import threadpool_limits
            limit = threadpool_limits(args.threads)
        except ImportError:
            limit = None
    else:
        limit = None
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit('fresh output path required: ' + str(args.output))
    report = run_bench(members_path=args.members, images_path=args.images, device=args.device,
                       precision=args.precision, batch_size=args.batch_size, warmup=args.warmup,
                       repetitions=args.repetitions, t4_tflops=args.t4_tflops,
                       dcu_t4_ratio=args.dcu_t4_ratio, max_images=args.max_images,
                       decode_threads=args.decode_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write('\n')
    if limit is not None:
        limit.unregister()
    print(json.dumps({'pass': report['pass'], 'total_ms_per_image':
                      report['measurements']['total_ms_per_image'],
                      'cpu_preprocess_ms_per_image':
                      report['measurements']['cpu_preprocess_ms_per_image'],
                      'subset_estimates': {key: {'members': value['members'],
                                                 'conservative_total_ms': value['conservative_total_ms'],
                                                 'pass': value['pass']}
                                           for key, value in report['subset_estimates'].items()},
                      'report_sha256': sha(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()

"""One bounded, fit-only CPU ranking diagnosis for the frozen R120 EMA selection.

The sample roster and individual logits stay in the private Pod preparation
directory. Only AGGREGATE.json is suitable for transfer or repository records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROUND = 'r120-small22k-seed20261029-scale085-pairsigmoid-24k-p4-9010-20261031'
PROFILE = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairsigmoid-v1'
COMMIT = '7922dd788f982bcb72d30a1a8e6d6e7761e42e88'
FREEZE_SHA = '5f808304a0e4ae84501525a1634f0ca55ec80e16bcba67342a66e7e211f5bcf2'
BLOB_SHA = '25229efdea3a91baeb0b5e5142a470c2a68a15bab9470f92e9de76b1b616cac6'
TENSOR_SHA = 'f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85'
SITE_SHA = '26cae5aefe899351d153dff9c1dd6d685c129dcc0cb578bc73a2f03af409ce43'
MAX_SECONDS = 600
BATCH = 4
INTRA_THREADS = 8


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path: Path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def chosen_indices(manifest):
    rows = manifest['rows']
    fit = manifest['fit_indices']
    dev = manifest['dev_indices']
    require(len(fit) == 46256 and len(dev) == 5140, 'sealed split count differs')
    require(len(set(fit)) == len(fit) and not set(fit).intersection(dev),
            'sealed fit/dev index identity differs')
    require(len({row['name'] for row in rows}) == len(rows), 'sealed original names not unique')
    groups = {0: [], 1: []}
    for index in fit:
        require(type(index) is int and 0 <= index < len(rows), 'invalid sealed fit index')
        row = rows[index]
        require(type(row['name']) is str and row['name'] and row['label'] in (0, 1),
                'invalid sealed fit metadata')
        require(row.get('sha256') and row.get('relative_path') and 'group' in row,
                'fit prepared-cache metadata incomplete')
        groups[int(row['label'])].append(index)
    require(all(len(groups[label]) >= 128 for label in (0, 1)),
            'either fit label has fewer than 128 rows')

    def key(index):
        name = rows[index]['name']
        return hashlib.sha256(name.encode('utf-8')).digest(), name

    positive = sorted(groups[1], key=key)[:128]
    negative = sorted(groups[0], key=key)[:128]
    order = [index for pair in zip(positive, negative) for index in pair]
    require(len(order) == 256 and len(set(order)) == 256, 'fixed roster cardinality differs')
    return rows, order


class NoOriginalFallback:
    """Mixin applied to the frozen ImageDataset to prohibit original-byte reads."""

    def _read_image(self, row):
        raise ValueError('prepared-cache-only contract would read an original image')


def worker(round_root: Path, private: Path):
    stage = 'frozen_identity'
    try:
        require(round_root.name == ROUND, 'round root identity differs')
        freeze_path = round_root / 'SINGLE-MODEL-FREEZE.json'
        site_path = round_root / 'SITE_BINDING.json'
        assets_path = round_root / 'ASSETS.json'
        require(all(path.is_file() and not path.is_symlink()
                    for path in (freeze_path, site_path, assets_path)),
                'frozen package identity file missing or symlinked')
        require(digest(freeze_path) == FREEZE_SHA and digest(site_path) == SITE_SHA,
                'frozen freeze/site SHA differs')
        freeze, site, assets = map(read_json, (freeze_path, site_path, assets_path))
        require(site['pod']['uid'] == os.environ['NCCCU_POD_UID']
                and site['git_commit'] == COMMIT
                and site['round_id'] == ROUND
                and site['round_root'] == str(round_root),
                'frozen Pod/Git/round binding differs')
        require(freeze['experiment91']['version'] == PROFILE
                and freeze['selected']['variant'] == freeze['variant'] == 'ema'
                and freeze['selected']['cycle'] == 6
                and freeze['selected']['successful_updates'] == 4800
                and freeze['selected']['tensor_sha256'] == freeze['tensor_sha256'] == TENSOR_SHA
                and freeze['blob_sha256'] == BLOB_SHA,
                'selected EMA checkpoint pin differs')
        checkpoint = Path(freeze['checkpoint'])
        require(checkpoint.is_file() and not checkpoint.is_symlink()
                and checkpoint.parent == round_root / 'outputs' / freeze['round_id'] / 'convnext-small' / 'checkpoints',
                'selected checkpoint path missing or outside frozen run')
        require(freeze['precision'] == 'fp32' and freeze['view'] == 'identity'
                and freeze['geometry'] == {'image_size': 512, 'prepared_size': 512},
                'frozen evaluation geometry/precision differs')
        require(assets['prepared512_cache']['read_only'] is True
                and assets['prepared512_cache']['prepared_size'] == 512
                and assets['prepared512_cache']['remote_root'] == site['prepared_cache'],
                'frozen prepared-cache binding differs')
        sealed = assets['manifests']['fit_dev']
        manifest_path = Path(sealed['remote_path'])
        require(manifest_path == Path(site['metadata_root']) / 'manifest.json'
                and manifest_path.is_file() and not manifest_path.is_symlink()
                and sealed['sha256'] == freeze['experiment91']['manifest_sha256']
                and digest(manifest_path) == sealed['sha256'],
                'sealed fit/dev manifest path or SHA differs')
        os.environ['NCCCU_SITE_BINDING'] = str(site_path)
        os.environ['NCCCU_SITE_BINDING_SHA256'] = SITE_SHA
        os.environ['TRAIN_IMAGE_CACHE'] = site['prepared_cache']
        os.environ['TRAIN_IMAGE_CACHE_READONLY'] = '1'
        os.environ['OMP_NUM_THREADS'] = str(INTRA_THREADS)
        os.environ['MKL_NUM_THREADS'] = str(INTRA_THREADS)
        os.environ['OPENBLAS_NUM_THREADS'] = str(INTRA_THREADS)
        sys.path.insert(0, str(round_root / 'bundle'))
        from retrain.persistence import canonical_digest

        manifest = read_json(manifest_path)
        require(canonical_digest(manifest) == freeze['experiment91']['manifest_digest']
                and canonical_digest(manifest['fit_indices']) == freeze['experiment91']['fit_indices_sha256']
                and canonical_digest(manifest['dev_indices']) == freeze['experiment91']['dev_indices_sha256'],
                'sealed manifest canonical identity differs')

        stage = 'fixed_roster'
        rows, order = chosen_indices(manifest)
        roster_path = private / 'ROSTER.json'
        write_new(roster_path, {'method': 'sha256(original_name_utf8), original_name; first128 per label',
                                'split': 'fit_only', 'order': [
                                    {'index': index, 'name': rows[index]['name'],
                                     'label': int(rows[index]['label']), 'group': rows[index]['group']}
                                    for index in order]})
        roster_sha = digest(roster_path)

        stage = 'checkpoint_strict_load'
        import torch
        from torch.utils.data import DataLoader
        torch.set_num_threads(INTRA_THREADS)
        torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
                and os.environ.get('HIP_VISIBLE_DEVICES') == ''
                and os.environ.get('ROCR_VISIBLE_DEVICES') == '',
                'CPU-only hidden-device contract differs')
        from retrain.inference import _load_model
        load_start = time.monotonic()
        model, job, _, origin, device = _load_model(checkpoint, device='cpu', variant='ema')
        load_seconds = time.monotonic() - load_start
        require(origin['type'] == 'checkpoint-file'
                and origin['sha256'] == BLOB_SHA
                and origin['tensor_sha256'] == TENSOR_SHA
                and origin['manifest_digest'] == freeze['experiment91']['manifest_digest']
                and origin['variant'] == 'ema'
                and job == freeze['job']
                and job['precision'] == 'fp32'
                and job['model'] == 'convnext_small'
                and job['image_size'] == job['prepared_size'] == 512
                and device.type == 'cpu',
                'strict selected EMA/model/job identity differs')
        require(all(parameter.dtype == torch.float32 for parameter in model.parameters()),
                'loaded parameter dtype differs from FP32')

        stage = 'readonly_prepared_forward'
        from retrain.age import forward, unpack_batch
        from retrain.data import ImageDataset
        from retrain.n05_metrics import metrics as exact_metrics

        class StrictDataset(NoOriginalFallback, ImageDataset):
            pass

        dataset = StrictDataset(site['data_root'], rows, 512, training=False,
                                normalization=job['normalization'], prepared_size=512,
                                age=job['age'])
        require(dataset.prepared_cache is not None and dataset.prepared_cache.readonly
                and dataset.training is False and dataset.training_views == 1,
                'strict readonly single-view dataset binding differs')
        batches = [order[start:start + BATCH] for start in range(0, len(order), BATCH)]
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=False)
        records = []
        forward_start = time.monotonic()
        model.eval()
        with torch.inference_mode():
            for expected, batch in zip(batches, loader):
                images, labels, indices, age_features = unpack_batch(batch, job, device)
                require(indices.tolist() == expected and images.shape == (BATCH, 3, 512, 512),
                        'readonly prepared batch order or geometry differs')
                logits = forward(model, images, age_features, job).reshape(-1).float()
                require(logits.numel() == BATCH and bool(torch.isfinite(logits).all()),
                        'nonfinite or incomplete batch logits')
                for index, label, logit in zip(expected, labels.tolist(), logits.tolist()):
                    require(int(label) == int(rows[index]['label']), 'sealed batch label differs')
                    records.append({'index': index, 'name': rows[index]['name'],
                                    'label': int(label), 'logit': float(logit)})
                write_new(private / f'COUNT-{len(records):03d}.json', {'completed': len(records)})
        forward_seconds = time.monotonic() - forward_start
        require(len(records) == 256, 'fixed fit256 evaluation incomplete')
        write_new(private / 'SCORES.json', records)
        exact = exact_metrics(records)
        positives = [record['logit'] for record in records if record['label'] == 1]
        negatives = [record['logit'] for record in records if record['label'] == 0]
        require(len(positives) == len(negatives) == 128
                and all(math.isfinite(value) for value in positives + negatives),
                'fit label count or finite logits differ')
        strict_wrong = sum(positive < negative for positive in positives for negative in negatives)
        ties = sum(positive == negative for positive in positives for negative in negatives)
        require(exact['error_twice'] == 2 * strict_wrong + ties,
                'exact ranking and strict/tied pair accounting differ')
        write_new(private / 'WORKER-SUMMARY.json', {
            'schema': 'r120-fit256-aggregate-v1', 'status': 'complete',
            'scope': 'fixed_seen_fit_metadata_only_selection',
            'round': ROUND, 'source_git_commit': COMMIT,
            'freeze_sha256': FREEZE_SHA, 'selected_checkpoint_blob_sha256': BLOB_SHA,
            'selected_ema_tensor_sha256': TENSOR_SHA,
            'sealed_manifest_sha256': sealed['sha256'],
            'roster_sha256_private': roster_sha,
            'fit_count': 256, 'positive_count': 128, 'negative_count': 128,
            'holdout_images_read': 0,
            'unique_content_group_count': len({rows[index]['group'] for index in order}),
            'raw_logit_auroc': exact['auroc'], 'error_twice': exact['error_twice'],
            'strict_wrong_pair_count': strict_wrong,
            'strict_wrong_positive_count': sum(any(p < n for n in negatives) for p in positives),
            'strict_wrong_negative_count': sum(any(p < n for p in positives) for n in negatives),
            'tied_pair_count': ties, 'tied_pair_ratio': ties / (128 * 128),
            'finite_all_logits': True,
            'load_seconds': load_seconds, 'forward_seconds': forward_seconds,
            'device': 'cpu', 'precision': 'fp32', 'batch_size': BATCH,
            'loader_workers': 0, 'torch_intraop_threads': INTRA_THREADS,
            'torch_interop_threads': 1, 'prepared_cache_readonly': True,
            'augmentation': False, 'view': 'identity',
            'limitations': ['seen_fit_subset_only', 'not_full_fit', 'not_holdout',
                            'not_label_validation', 'not_cache_detail_proof'],
        })
    except BaseException as error:
        write_new(private / 'WORKER-ERROR.json', {
            'stage': stage, 'type': type(error).__name__, 'detail_private': str(error)})
        raise


def parent(round_root: Path, private: Path):
    started = time.monotonic()
    private.mkdir(mode=0o700, parents=True, exist_ok=False)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
               str(round_root), str(private)]
    timed_out = False
    with (private / 'WORKER.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=os.environ.copy(), cwd=private)
        try:
            process.wait(timeout=max(0.0, MAX_SECONDS - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
    elapsed = time.monotonic() - started
    if (process.returncode == 0 and (private / 'WORKER-SUMMARY.json').is_file()
            and not timed_out and elapsed <= MAX_SECONDS):
        result = read_json(private / 'WORKER-SUMMARY.json')
        result['total_wall_seconds'] = elapsed
    else:
        error_path = private / 'WORKER-ERROR.json'
        failure = read_json(error_path) if error_path.is_file() else {}
        completed = sorted(private.glob('COUNT-*.json'))
        result = {
            'schema': 'r120-fit256-aggregate-v1', 'status': 'unknown',
            'round': ROUND, 'source_git_commit': COMMIT,
            'freeze_sha256': FREEZE_SHA, 'selected_checkpoint_blob_sha256': BLOB_SHA,
            'selected_ema_tensor_sha256': TENSOR_SHA,
            'completed_count_partial': read_json(completed[-1])['completed'] if completed else 0,
            'failure_stage': failure.get('stage', 'timeout_or_process_exit'),
            'failure_type': failure.get('type', 'Timeout' if timed_out or elapsed > MAX_SECONDS
                                         else 'ProcessExit'),
            'timed_out': timed_out, 'total_wall_seconds': elapsed,
            'no_partial_scientific_conclusion': True,
        }
    result['driver_sha256'] = digest(Path(__file__))
    result['pod_uid'] = os.environ['NCCCU_POD_UID']
    result['cpu_mask'] = sorted(os.sched_getaffinity(0))
    output = private / 'AGGREGATE.json'
    write_new(output, result)
    print(json.dumps({'aggregate_path': str(output), 'aggregate_sha256': digest(output),
                      'aggregate': result}, sort_keys=True, allow_nan=False), flush=True)


def main():
    os.umask(0o077)
    if sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        if len(sys.argv) != 3:
            raise SystemExit('usage: r120_fit256.py ROUND_ROOT PRIVATE_OUTPUT_DIR')
        parent(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == '__main__':
    main()

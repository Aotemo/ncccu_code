"""F07: sharded fp16 forward of members over internal dev and/or official valid, orig + hflip.

Rows are assigned by index modulo the shard count. Each member is loaded in a
subprocess through its own historical runtime bundle with its own site binding,
matching the environment its standalone export was validated in. The
preprocessing pipeline is the member's frozen job; hflip flips the preprocessed
tensor on the device (preprocessing pipeline unchanged).
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_rows(root, split, dev_indices_path, map_cache=None):
    from retrain.data import _image_mapping, _read_csv
    if split == 'dev':
        rows, _ = _read_csv(root, 'train')
        indices = json.loads(Path(dev_indices_path).read_text())
        selected = [rows[i] for i in indices]
    elif split == 'official':
        rows, _ = _read_csv(root, 'valid')
        selected = rows
    else:
        raise ValueError('split must be dev or official')
    split_dir = 'train' if split == 'dev' else 'valid'
    mapping = None
    if map_cache is not None and Path(map_cache).exists():
        mapping = json.loads(Path(map_cache).read_text()).get(split_dir)
        if not isinstance(mapping, dict):
            mapping = None
    if mapping is None:
        mapping = _image_mapping(root, split_dir)
        if map_cache is not None:
            path = Path(map_cache)
            merged = json.loads(path.read_text()) if path.exists() else {}
            if not isinstance(merged, dict):
                merged = {}
            merged[split_dir] = mapping
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(merged))
            os.replace(temporary, path)
    result = []
    for row in selected:
        key = row['name'].casefold()
        if key not in mapping:
            raise ValueError('image missing from split directory: ' + row['name'])
        result.append({'name': row['name'], 'label': row['label'], 'age': row.get('age', ''),
                       'relative_path': mapping[key]})
    return result


def forward_worker(args, member_id, member):
    """Load one member through its own bundle and process both splits for this shard."""
    os.environ['NCCCU_SITE_BINDING'] = member['site']
    os.environ['NCCCU_SITE_BINDING_SHA256'] = member['site_sha256']
    sys.path.insert(0, str(member['bundle']))
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('GPU must be initialized only inside the forward loop setup')
    from retrain import inference
    from retrain.age import enabled
    from retrain.data import ImageDataset
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    preprocess_batch = getattr(inference, 'preprocess_batch', None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('dev', 'official'):
        output = out_dir / f'{split}.{member_id}.shard{args.shard}of{args.shards}.csv'
        if output.exists():
            print(json.dumps({'status': 'f07_shard_skipped_existing', 'member': member_id,
                              'split': split, 'shard': args.shard}), flush=True)
            continue
        try:
            sidecar = run_split(args, member_id, member, split, device, torch, inference,
                                preprocess_batch, enabled, ImageDataset, out_dir)
            print(json.dumps({'status': 'f07_shard_done', 'member': member_id, 'split': split,
                              'shard': args.shard, 'rows': sidecar['rows_assigned'],
                              'seconds': sidecar['seconds']}), flush=True)
        except Exception as error:
            failure = {'status': 'f07_shard_failed', 'member': member_id, 'split': split,
                       'shard': args.shard, 'error_type': type(error).__name__,
                       'error': str(error)[:400]}
            (out_dir / f'{split}.{member_id}.shard{args.shard}of{args.shards}.FAILED.json').write_text(
                json.dumps(failure, indent=1) + '\n')
            print(json.dumps(failure), flush=True)


def run_split(args, member_id, member, split, device, torch, inference, preprocess_batch,
              enabled, ImageDataset, out_dir):
    rows = build_rows(args.data_root, split, args.dev_indices, args.image_map_cache)
    positives = sum(row['label'] for row in rows)
    variant = member.get('variant')
    if variant is None:
        variant = (json.loads((Path(member['export']) / 'model.json').read_text())
                   .get('checkpoint', {}).get('variant'))
    model, job, metadata, origin, selected_device = inference._load_model(member['export'], device=device,
                                                                          variant=variant)
    if selected_device.type != 'cuda':
        raise ValueError('model was not placed on the pinned device')
    try:
        runtime = inference._runtime(job, metadata, precision='fp16')
    except TypeError:
        runtime = {'image_size': int(job['image_size']), 'normalization': job.get('normalization', 'imagenet'),
                   'prepared_size': int(job.get('prepared_size', 0))}
    parameters = inspect.signature(ImageDataset.__init__).parameters
    kwargs = {'prepared_size': runtime['prepared_size'], 'age': job.get('age')}
    if 'input_preprocessing' in parameters:
        kwargs['input_preprocessing'] = job.get('input_preprocessing')
    elif job.get('input_preprocessing') is not None:
        raise ValueError('historical dataset cannot honor configured preprocessing')
    dataset = ImageDataset(args.data_root, rows, runtime['image_size'], False,
                           runtime['normalization'], **kwargs)
    age_on = enabled(job)
    assigned = [index for index in range(len(rows)) if index % args.shards == args.shard]
    model.eval()
    logits = {}
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(assigned), args.batch_size):
            chunk = assigned[start:start + args.batch_size]
            if preprocess_batch is not None and args.decode_workers > 1:
                batch = preprocess_batch(dataset, chunk, workers=args.decode_workers)
            else:
                batch = [dataset[index] for index in chunk]
            images = torch.stack([item[0] for item in batch]).to(device)
            ages = (torch.stack([item[3] for item in batch]).to(device, dtype=torch.float32)
                    if age_on else None)
            def forward(tensors):
                with torch.autocast(device.type, dtype=torch.float16):
                    out = (model(tensors, ages) if ages is not None else model(tensors))
                values = out.reshape(-1).float()
                if values.numel() != tensors.shape[0] or not torch.isfinite(values).all():
                    raise FloatingPointError('invalid logits')
                return values
            original = forward(images)
            flipped = forward(images.flip(-1))
            for position, index in enumerate(chunk):
                logits[index] = (float(original[position]), float(flipped[position]))
    del model
    name = f'{split}.{member_id}.shard{args.shard}of{args.shards}'
    output = out_dir / (name + '.csv')
    if output.exists():
        raise FileExistsError(output)
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['name', 'label', 'logit_orig', 'logit_hflip'])
        for index in sorted(logits):
            row = rows[index]
            writer.writerow([row['name'], row['label'], repr(logits[index][0]), repr(logits[index][1])])
        stream.flush()
        os.fsync(stream.fileno())
    sidecar = {
        'member': member_id, 'export': member['export'], 'export_sha256': sha(Path(member['export']) / 'model.pt'),
        'bundle': member['bundle'], 'split': split, 'rows_total': len(rows), 'rows_assigned': len(assigned),
        'positive_total': positives, 'shard': args.shard, 'shards': args.shards,
        'precision': 'fp16', 'batch_size': args.batch_size, 'decode_workers': args.decode_workers,
        'variant': variant, 'seconds': round(time.time() - started, 3),
        'csv_sha256': sha(output), 'test_label_values_read': False, 'TTA': 'orig+hflip',
    }
    (out_dir / (name + '.json')).write_text(json.dumps(sidecar, indent=1) + '\n')
    return sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, required=True, help='JSON: id -> {export, bundle, site, site_sha256, cost?}')
    parser.add_argument('--member', required=True, help="member id or 'ALL'")
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--dev-indices', type=Path, required=True)
    parser.add_argument('--split', choices=('BOTH',), default='BOTH')
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--decode-workers', type=int, default=4)
    parser.add_argument('--image-map-cache', type=Path, default=None)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.shards < 1:
        raise ValueError('invalid shard identity')
    members = json.loads(args.members.read_text())
    if args.member != 'ALL':
        if args.member not in members:
            raise ValueError('unknown member: ' + args.member)
        forward_worker(args, args.member, members[args.member])
        return
    # driver: one subprocess per member (its own historical bundle and site)
    summary = []
    for member_id in sorted(members):
        member = members[member_id]
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--members', str(args.members),
                   '--member', member_id, '--data-root', str(args.data_root),
                   '--dev-indices', str(args.dev_indices), '--split', 'BOTH',
                   '--shard', str(args.shard), '--shards', str(args.shards),
                   '--out-dir', str(args.out_dir), '--batch-size', str(args.batch_size),
                   '--decode-workers', str(args.decode_workers),
                   '--image-map-cache', str(args.image_map_cache or '')]
        if args.image_map_cache is None:
            command.remove('--image-map-cache')
            command.remove('')
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = [line for line in process.stdout.splitlines() if line.startswith('{')]
        parsed = []
        for line in lines:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        summary.extend(parsed)
        if process.returncode != 0 and not any(item.get('member') == member_id for item in parsed):
            summary.append({'status': 'f07_member_driver_failed', 'member': member_id,
                            'returncode': process.returncode, 'tail': process.stdout[-400:]})
        print(json.dumps({'status': 'f07_member_complete', 'member': member_id,
                          'shard': args.shard, 'events': len(parsed)}), flush=True)
    print(json.dumps({'status': 'f07_rank_done', 'shard': args.shard,
                      'failed': [item for item in summary if item.get('status') == 'f07_shard_failed']}), flush=True)


if __name__ == '__main__':
    main()

"""F09: sharded fp16 forward with draft (JPEG half-scale decoder) preprocessing.

One subprocess per member (own historical bundle + site binding). Single orig
view; the only difference from the frozen pipeline is the decoder draft hint.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f09_draft import make_draft_dataset  # noqa: E402


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_rows(root, split, dev_indices_path, map_cache):
    from retrain.data import _read_csv
    if split == 'dev':
        rows, _ = _read_csv(root, 'train')
        indices = json.loads(Path(dev_indices_path).read_text())
        selected = [rows[i] for i in indices]
    else:
        rows, _ = _read_csv(root, 'valid')
        selected = rows
    mapping = json.loads(Path(map_cache).read_text())['train' if split == 'dev' else 'valid']
    result = []
    for row in selected:
        key = row['name'].casefold()
        if key not in mapping:
            raise ValueError('image missing from mapping: ' + row['name'])
        result.append({'name': row['name'], 'label': row['label'], 'age': row.get('age', ''),
                       'relative_path': mapping[key]})
    return result


def forward_worker(args, member_id, member):
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('dev', 'official'):
        output = out_dir / f'{split}.{member_id}.draft.shard{args.shard}of{args.shards}.csv'
        if output.exists():
            print(json.dumps({'status': 'f09_shard_skipped_existing', 'member': member_id,
                              'split': split, 'shard': args.shard}), flush=True)
            continue
        try:
            sidecar = run_split(args, member_id, member, split, device, torch, inference,
                                enabled, ImageDataset, out_dir, output)
            print(json.dumps({'status': 'f09_shard_done', 'member': member_id, 'split': split,
                              'shard': args.shard, 'rows': sidecar['rows_assigned'],
                              'seconds': sidecar['seconds']}), flush=True)
        except Exception as error:
            failure = {'status': 'f09_shard_failed', 'member': member_id, 'split': split,
                       'shard': args.shard, 'error_type': type(error).__name__,
                       'error': str(error)[:400]}
            (out_dir / f'{split}.{member_id}.draft.shard{args.shard}of{args.shards}.FAILED.json').write_text(
                json.dumps(failure, indent=1) + '\n')
            print(json.dumps(failure), flush=True)


def run_split(args, member_id, member, split, device, torch, inference, enabled,
              ImageDataset, out_dir, output):
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
    parameters = inspect.signature(ImageDataset.__init__).parameters
    kwargs = {'age': job.get('age')}
    if 'prepared_size' in parameters:
        kwargs['prepared_size'] = int(job.get('prepared_size', 0))
    if 'input_preprocessing' in parameters:
        kwargs['input_preprocessing'] = job.get('input_preprocessing')
    elif job.get('input_preprocessing') is not None:
        raise ValueError('historical dataset cannot honor configured preprocessing')
    dataset = make_draft_dataset(ImageDataset, args.data_root, rows, int(job['image_size']),
                                 job.get('normalization', 'imagenet'), **kwargs)
    age_on = enabled(job)
    assigned = [index for index in range(len(rows)) if index % args.shards == args.shard]
    model.eval()
    logits = {}
    started = time.time()
    with torch.inference_mode():
        for index in assigned:
            item = dataset[index]
            images = item[0].unsqueeze(0).to(device)
            ages = (item[3].unsqueeze(0).to(device, dtype=torch.float32) if age_on else None)
            with torch.autocast(device.type, dtype=torch.float16):
                out = (model(images, ages) if ages is not None else model(images))
            values = out.reshape(-1).float()
            if values.numel() != 1 or not torch.isfinite(values).all():
                raise FloatingPointError('invalid logits')
            logits[index] = float(values[0])
    del model
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['name', 'label', 'logit'])
        for index in sorted(logits):
            row = rows[index]
            writer.writerow([row['name'], row['label'], repr(logits[index])])
        stream.flush()
        os.fsync(stream.fileno())
    sidecar = {
        'member': member_id, 'export': member['export'], 'bundle': member['bundle'],
        'export_sha256': sha(Path(member['export']) / 'model.pt'),
        'split': split, 'rows_total': len(rows), 'rows_assigned': len(assigned),
        'positive_total': positives, 'shard': args.shard, 'shards': args.shards,
        'precision': 'fp16', 'decode': 'draft-half', 'variant': variant,
        'seconds': round(time.time() - started, 3), 'csv_sha256': sha(output),
        'test_label_values_read': False,
    }
    (out_dir / (output.stem + '.json')).write_text(json.dumps(sidecar, indent=1) + '\n')
    return sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, required=True)
    parser.add_argument('--member', required=True, help="member id or 'ALL'")
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--dev-indices', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--image-map-cache', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.shards < 1:
        raise ValueError('invalid shard identity')
    members = json.loads(args.members.read_text())
    if args.member != 'ALL':
        if args.member not in members:
            raise ValueError('unknown member: ' + args.member)
        forward_worker(args, args.member, members[args.member])
        return
    for member_id in sorted(members):
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--members', str(args.members),
                   '--member', member_id, '--data-root', str(args.data_root),
                   '--dev-indices', str(args.dev_indices), '--shard', str(args.shard),
                   '--shards', str(args.shards), '--out-dir', str(args.out_dir),
                   '--image-map-cache', str(args.image_map_cache)]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        failures = []
        for line in process.stdout.splitlines():
            if not line.startswith('{'):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get('status') == 'f09_shard_failed':
                failures.append(item)
        if process.returncode != 0 and not failures:
            failures = [{'status': 'f09_member_driver_failed', 'member': member_id,
                         'returncode': process.returncode, 'tail': process.stdout[-400:]}]
        print(json.dumps({'status': 'f09_member_complete', 'member': member_id,
                          'shard': args.shard, 'failures': failures}), flush=True)


if __name__ == '__main__':
    main()

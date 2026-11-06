"""F07: sharded fp16 forward of one member over internal dev and/or official valid, orig + hflip.

One process per GPU rank; rows are assigned by index modulo shard count. The
preprocessing pipeline is the member's own frozen job; hflip flips the
preprocessed tensor on the device (preprocessing pipeline unchanged).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_rows(root, split, dev_indices_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'))
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
    mapping = _image_mapping(root, 'train' if split == 'dev' else 'valid')
    result = []
    for row in selected:
        key = row['name'].casefold()
        if key not in mapping:
            raise ValueError('image missing from split directory: ' + row['name'])
        result.append({'name': row['name'], 'label': row['label'], 'age': row.get('age', ''),
                       'relative_path': mapping[key]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, required=True, help='JSON: id -> {export, variant}')
    parser.add_argument('--member', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--dev-indices', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'official'), required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--decode-workers', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.shards < 1:
        raise ValueError('invalid shard identity')
    members = json.loads(args.members.read_text())
    if args.member not in members:
        raise ValueError('unknown member: ' + args.member)
    member = members[args.member]
    rows = build_rows(args.data_root, args.split, args.dev_indices)
    positives = sum(row['label'] for row in rows)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'))
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('GPU must be initialized only inside the forward loop setup')
    from retrain.inference import _load_model, _runtime, preprocess_batch
    from retrain.age import enabled
    device = torch.device('cuda')
    torch.cuda.set_device(device)
    model, job, metadata, origin, selected_device = _load_model(member['export'], device=device,
                                                                variant=member.get('variant'))
    if str(selected_device) != 'cuda':
        raise ValueError('model was not placed on the pinned device')
    runtime = _runtime(job, metadata, precision='fp16')
    dataset_args = (args.data_root, rows, runtime['image_size'], False, runtime['normalization'])
    dataset_kwargs = {'prepared_size': runtime['prepared_size'], 'age': job.get('age'),
                      'input_preprocessing': job.get('input_preprocessing')}
    from retrain.data import ImageDataset
    dataset = ImageDataset(*dataset_args, **dataset_kwargs)
    age_on = enabled(job)
    assigned = [index for index in range(len(rows)) if index % args.shards == args.shard]
    model.eval()
    logits = {}
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(assigned), args.batch_size):
            chunk = assigned[start:start + args.batch_size]
            batch = preprocess_batch(dataset, chunk, workers=args.decode_workers)
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f'{args.split}.{args.member}.shard{args.shard}of{args.shards}'
    output = args.out_dir / (name + '.csv')
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
        'member': args.member, 'export': member['export'], 'export_sha256': sha(Path(member['export']) / 'model.pt'),
        'split': args.split, 'rows_total': len(rows), 'rows_assigned': len(assigned),
        'positive_total': positives, 'shard': args.shard, 'shards': args.shards,
        'precision': 'fp16', 'batch_size': args.batch_size, 'decode_workers': args.decode_workers,
        'variant': member.get('variant'), 'seconds': round(time.time() - started, 3),
        'csv_sha256': sha(output), 'test_label_values_read': False, 'TTA': 'orig+hflip',
    }
    (args.out_dir / (name + '.json')).write_text(json.dumps(sidecar, indent=1) + '\n')
    print(json.dumps({'status': 'f07_shard_done', 'member': args.member, 'split': args.split,
                      'shard': args.shard, 'rows': len(assigned), 'seconds': sidecar['seconds']}), flush=True)


if __name__ == '__main__':
    main()

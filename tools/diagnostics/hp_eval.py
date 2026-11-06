"""HP member evaluation: selection, strict fp16 export, dev/official forward, combos.

The forward populations are the internal dev 5140 rows and the official 5849
(valid) images; official labels are used only for authorized validation
AUROC reporting. Test images and labels are never touched.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False).encode() + b'\n'
    with path.open('xb') as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    ranks = {}
    index = 0
    while index < len(pairs):
        stop = index
        while stop < len(pairs) and pairs[stop][0] == pairs[index][0]:
            stop += 1
        average = (index + 1 + stop) / 2.0
        for position in range(index, stop):
            ranks[position] = average
        index = stop
    positive = sum(rank for position, rank in ranks.items() if pairs[position][1] == 1)
    negatives = sum(1 for _, label in pairs if label == 0)
    positives = sum(1 for _, label in pairs if label == 1)
    if not positives or not negatives:
        raise ValueError('both classes required for AUROC')
    return (positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def ap(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    hits = 0
    total = 0.0
    for position, index in enumerate(order):
        if labels[index] == 1:
            hits += 1
            total += hits / (position + 1.0)
    positives = sum(labels)
    if not positives:
        raise ValueError('positives required for AP')
    return total / positives


def find_selected(run_dir):
    """Locate the run's selected checkpoint identity from its records."""
    run_dir = Path(run_dir)
    record = json.loads((run_dir.parent / 'complete.json').read_text())
    if record.get('status') != 'completed':
        raise SystemExit('round is not completed: ' + str(record.get('status')))
    index = json.loads((run_dir / 'checkpoints' / 'index.json').read_text())
    best = index['best']
    best_path = run_dir / 'checkpoints' / best['file']
    if sha(best_path) != best['sha256']:
        raise SystemExit('best checkpoint bytes differ from the index')
    return best_path, best


def cmd_export(args):
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'))
    os.environ['NCCCU_SITE_BINDING'] = args.site
    os.environ['NCCCU_SITE_BINDING_SHA256'] = sha(args.site)
    import torch
    from retrain.inference import _load_model
    from retrain.ema import tensor_digest
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    best_path, best = find_selected(args.run_dir)
    model, job, metadata, origin, device = _load_model(
        Path(args.run_dir) / 'checkpoints', device='cpu', variant=args.variant)
    if origin.get('type') == 'checkpoint':
        payload_sha = origin.get('sha256')
    state = model.state_dict()
    values = {key: (value.to(torch.float16) if value.is_floating_point() else value.clone())
              for key, value in state.items()}
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in values.values()):
        raise SystemExit('fp16 export contains nonfinite tensors')
    output = Path(args.output)
    if output.exists():
        raise SystemExit('fresh output required: ' + str(output))
    (output / 'fp16').mkdir(parents=True)
    model_path = output / 'fp16' / 'model.pt'
    with model_path.open('xb') as stream:
        torch.save(values, stream)
        stream.flush()
    tensor_sha = tensor_digest(values)
    metadata_json = {'format': 'glaucoma-frozen-inference-v1', 'model_sha256': sha(model_path),
                     'job': job, 'age': job.get('age'),
                     'checkpoint': {'variant': args.variant, 'tensor_sha256': tensor_sha,
                                    'updates': best.get('successful_updates')},
                     'inference': {'threshold': .5, 'precision': 'fp16'}}
    with (output / 'fp16' / 'model.json').open('x') as stream:
        json.dump(metadata_json, stream, indent=2, allow_nan=False)
        stream.write('\n')
    model.load_state_dict(values, strict=True)
    model.eval()
    loaded, loaded_job, _, loaded_origin, loaded_device = _load_model(output / 'fp16', device='cpu',
                                                                      variant=args.variant)
    canonical = lambda value: json.dumps(value, sort_keys=True, default=list)
    details = {'job_equal': canonical(loaded_job) == canonical(job), 'device': str(loaded_device),
               'model_sha': loaded_origin['model_sha256'] == sha(model_path),
               'tensor_sha': loaded_origin['tensor_sha256'] == tensor_sha,
               'state_digest': tensor_digest(loaded.state_dict()) == tensor_digest(model.state_dict())}
    if (not details['job_equal'] or details['device'] != 'cpu' or not details['model_sha']
            or not details['tensor_sha'] or not details['state_digest']):
        raise SystemExit('strict fp16 standalone reload differs: ' + json.dumps(details))
    size = int(job['image_size'])
    images = torch.zeros((1, 3, size, size), dtype=torch.float32)
    age = torch.zeros((1, 2), dtype=torch.float32) if (job.get('age') or {}).get('enabled') else None
    loaded.eval()
    model.eval()
    with torch.inference_mode():
        from retrain.age import forward as age_forward
        expected = age_forward(model, images, age, job).reshape(-1).float()
        actual = age_forward(loaded, images, age, job).reshape(-1).float()
    if expected.shape != (1,) or not bool(torch.isfinite(expected).all()) or not torch.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        raise SystemExit('synthetic fp16 inference differs')
    receipt = {'status': 'hp_standalone_fp16_strict_synthetic_passed', 'run_dir': str(args.run_dir),
               'checkpoint': {'file': best['file'], 'sha256': best['sha256']},
               'variant': args.variant, 'updates': best.get('successful_updates'),
               'fp16_tensor_sha256': tensor_sha,
               'model': {'path': str(model_path), 'sha256': sha(model_path),
                         'bytes': model_path.stat().st_size},
               'metadata': {'path': str(output / 'fp16' / 'model.json'), 'sha256': sha(output / 'fp16' / 'model.json')},
               'GPU_initialized': False, 'official_read': False, 'test_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(output / 'STANDALONE.json', receipt)}, indent=1))


def build_rows(root, split, dev_indices_path, map_cache):
    from retrain.data import _image_mapping, _read_csv
    if split == 'dev':
        rows, _ = _read_csv(root, 'train')
        indices = json.loads(Path(dev_indices_path).read_text())
        selected = [rows[i] for i in indices]
        split_dir = 'train'
    elif split == 'official':
        selected, _ = _read_csv(root, 'valid')
        split_dir = 'valid'
    else:
        raise ValueError('split must be dev or official')
    mapping = None
    if map_cache is not None and Path(map_cache).exists():
        mapping = json.loads(Path(map_cache).read_text()).get(split_dir)
    if mapping is None:
        mapping = _image_mapping(root, split_dir)
    return selected, mapping


def cmd_forward(args):
    import inspect
    import os
    import sys
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'))
    os.environ['NCCCU_SITE_BINDING'] = args.site
    os.environ['NCCCU_SITE_BINDING_SHA256'] = sha(args.site)
    import torch
    from retrain import inference
    from retrain.data import ImageDataset
    from retrain.age import enabled
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    member = json.loads(Path(args.export / 'fp16' / 'model.json').read_text())
    model, job, metadata, origin, selected_device = inference._load_model(args.export, device=device)
    if selected_device.type != 'cuda':
        raise SystemExit('model was not placed on the pinned device')
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
        raise SystemExit('historical dataset cannot honor configured preprocessing')
    age_on = enabled(job)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('dev', 'official'):
        output = out_dir / f'{split}.{args.member}.shard{args.shard}of{args.shards}.csv'
        if output.exists():
            print(json.dumps({'status': 'hp_shard_skipped_existing', 'split': split}), flush=True)
            continue
        rows, mapping = build_rows(args.data_root, split, args.dev_indices, args.map_cache)
        dataset = ImageDataset(args.data_root, rows, runtime['image_size'], False,
                               runtime['normalization'], **kwargs)
        assigned = [index for index in range(len(rows)) if index % args.shards == args.shard]
        model.eval()
        logits = {}
        started = time.time()
        with torch.inference_mode():
            for start in range(0, len(assigned), args.batch_size):
                chunk = assigned[start:start + args.batch_size]
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
        name = f'{split}.{args.member}.shard{args.shard}of{args.shards}'
        with (out_dir / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['name', 'label', 'logit_orig', 'logit_hflip'])
            for index in sorted(logits):
                row = rows[index]
                writer.writerow([row['name'], row['label'], repr(logits[index][0]), repr(logits[index][1])])
            stream.flush()
        sidecar = {'member': args.member, 'export': str(args.export), 'export_sha256': sha(args.export / 'fp16' / 'model.pt'),
                   'split': split, 'rows_total': len(rows), 'rows_assigned': len(assigned),
                   'shard': args.shard, 'shards': args.shards, 'precision': 'fp16',
                   'batch_size': args.batch_size, 'seconds': round(time.time() - started, 3),
                   'csv_sha256': sha(out_dir / (name + '.csv')), 'test_label_values_read': False,
                   'TTA': 'orig+hflip', 'device': str(device)}
        (out_dir / (name + '.json')).write_text(json.dumps(sidecar, indent=1) + '\n')
        print(json.dumps({'status': 'hp_shard_done', 'split': split, 'rows': len(assigned),
                          'seconds': sidecar['seconds']}), flush=True)


def load_shard_population(shard_dir, member, split):
    directory = Path(shard_dir)
    sidecars = sorted(directory.glob(f'{split}.{member}.shard*.json'))
    if not sidecars:
        raise SystemExit(f'no shards for {member}/{split} in {directory}')
    shards = []
    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text())
        if payload['csv_sha256'] != sha(directory / sidecar.name.replace('.json', '.csv')):
            raise SystemExit('shard csv sha differs: ' + sidecar.name)
        shards.append(payload)
    total = sum(payload['rows_assigned'] for payload in shards)
    rows = {}
    for payload in shards:
        name = sidecar_name(payload, split, member)
        with (directory / name).open(newline='') as stream:
            for record in csv.DictReader(stream):
                key = record['name'].casefold()
                if key in rows:
                    raise SystemExit('duplicate row across shards: ' + record['name'])
                rows[key] = (record['label'], float(record['logit_orig']), float(record['logit_hflip']))
    return rows, total


def sidecar_name(payload, split, member):
    return f'{split}.{member}.shard{payload["shard"]}of{payload["shards"]}.csv'


def load_draft_population(shard_dir, member, split):
    """Load F09 draft-decode shards: <split>.<member>.draft.shardIofN.{csv,json}."""
    directory = Path(shard_dir)
    sidecars = sorted(directory.glob(f'{split}.{member}.draft.shard*.json'))
    if not sidecars:
        raise SystemExit(f'no draft shards for {member}/{split} in {directory}')
    rows = {}
    total = 0
    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text())
        csv_path = sidecar.with_suffix('.csv')
        if payload['csv_sha256'] != sha(csv_path):
            raise SystemExit('draft shard csv sha differs: ' + sidecar.name)
        if payload.get('decode') not in ('draft-half', 'draft-quarter'):
            raise SystemExit('unexpected draft decode identity: ' + str(payload.get('decode')))
        total += payload['rows_assigned']
        with csv_path.open(newline='') as stream:
            for record in csv.DictReader(stream):
                key = record['name'].casefold()
                if key in rows:
                    raise SystemExit('duplicate draft row across shards: ' + record['name'])
                rows[key] = (record['label'], float(record['logit']))
    if total != len(rows):
        raise SystemExit(f'{member}/{split} shard rows_assigned {total} differs from rows {len(rows)}')
    return rows, total


def _zstat(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {'mean': mean, 'std': math.sqrt(variance)}


def cmd_combo_draft(args):
    """Anchor-member comparison under the F09 draft4 decode using existing shards."""
    anchor = args.anchor
    members = json.loads(Path(args.members).read_text())
    if not isinstance(members, list) or anchor in members:
        raise SystemExit('members file must be a JSON list of ids distinct from the anchor')
    order = [anchor] + members
    populations = {'dev': {}, 'official': {}}
    for split in ('dev', 'official'):
        for member in order:
            directory = args.anchor_shard_dir if member == anchor else args.shard_dir
            rows, _ = load_draft_population(directory, member, split)
            populations[split][member] = rows
    dev_keys = sorted(populations['dev'][anchor])
    z = {}
    for member in order:
        rows = populations['dev'][member]
        if set(rows) != set(dev_keys):
            raise SystemExit(f'{member} dev coverage differs from {anchor}')
        z[member] = _zstat([rows[key][1] for key in dev_keys])
        if z[member]['std'] <= 0:
            raise SystemExit('degenerate dev logits for ' + member)

    def score(subset, split):
        rows = populations[split]
        reference = rows[anchor]
        common = set(reference)
        for member in subset[1:]:
            common &= set(rows[member])
        common = sorted(common)
        if len(common) != len(reference):
            return None
        labels_split = [int(rows[subset[0]][key][0]) for key in common]
        fused = [sum((rows[member][key][1] - z[member]['mean']) / z[member]['std']
                     for member in subset) / len(subset) for key in common]
        return {'auroc': auroc(fused, labels_split), 'ap': ap(fused, labels_split),
                'rows': len(common)}

    entries = []
    subsets = [[anchor]] + [[anchor, member] for member in members]
    if args.triple_with:
        missing = [member for member in args.triple_with if member not in members]
        if missing:
            raise SystemExit('triple members absent from members file: ' + ','.join(missing))
        subsets.append([anchor] + list(args.triple_with))
    for subset in subsets:
        entry = {'members': subset, 'size': len(subset)}
        for split in ('dev', 'official'):
            entry[split] = score(subset, split)
        if entry['dev'] and entry['official']:
            entry['mean_auroc'] = (entry['dev']['auroc'] + entry['official']['auroc']) / 2.0
        entries.append(entry)
    ranked = sorted([entry for entry in entries if entry.get('mean_auroc') is not None],
                    key=lambda entry: -entry['mean_auroc'])
    sanity = None
    if args.sanity_pair:
        missing = [member for member in args.sanity_pair if member not in members]
        if missing:
            raise SystemExit('sanity members absent from members file: ' + ','.join(missing))
        sanity = {'members': list(args.sanity_pair)}
        for split in ('dev', 'official'):
            sanity[split] = score(list(args.sanity_pair), split)
        if sanity['dev'] and sanity['official']:
            sanity['mean_auroc'] = (sanity['dev']['auroc'] + sanity['official']['auroc']) / 2.0
    receipt = {'status': 'hp_draft_combo_complete', 'decode': 'draft4', 'anchor': anchor, 'z': z,
               'baseline': args.baseline, 'threshold': args.baseline + 0.0005,
               'pairs_evaluated': len(members), 'table': ranked,
               'best_pair': ranked[0]['members'] if ranked else None, 'sanity_pair': sanity,
               'official_read': True, 'official_use': 'authorized_validation_only',
               'test_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(args.output, receipt)}, indent=1)[:6000])
    print('RECEIPT', args.output)


def cmd_combo(args):
    members = json.loads(Path(args.members).read_text())
    frozen = json.loads(Path(args.frozen_z).read_text())
    populations = {}
    for split in ('dev', 'official'):
        populations[split] = {}
        for member in members:
            rows, total = load_shard_population(args.shard_dir, member, split)
            populations[split][member] = rows
    dev = populations['dev']['HP']
    keys = sorted(dev)
    labels = [int(dev[key][0]) for key in keys]
    z = {}
    for member in members:
        rows = populations['dev'][member]
        common = [key for key in keys if key in rows]
        if len(common) != len(keys):
            raise SystemExit(f'{member} dev coverage differs')
        values = [rows[key][1] for key in keys]
        z[member] = {'mean': sum(values) / len(values),
                     'std': math.sqrt(sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values))}
        if z[member]['std'] <= 0:
            raise SystemExit('degenerate dev logits for ' + member)
    for member, constants in frozen.items():
        if member in members and member != 'HP':
            z[member] = dict(constants)
    results = []
    names = list(members)
    subsets = []
    for mask in range(1, 1 << len(names)):
        subset = [names[position] for position in range(len(names)) if mask & (1 << position)]
        if 2 <= len(subset) <= 4:
            subsets.append(subset)
    subsets.append(names)
    seen = set()
    for subset in subsets:
        key = '|'.join(sorted(subset))
        if key in seen:
            continue
        seen.add(key)
        entry = {'members': subset, 'size': len(subset)}
        for split in ('dev', 'official'):
            rows = populations[split]
            common = None
            for member in subset:
                member_keys = set(rows[member])
                common = member_keys if common is None else (common & member_keys)
            common = sorted(common)
            if len(common) != len(keys):
                entry[split] = None
                continue
            member_labels = [int(rows[subset[0]][key][0]) for key in common]
            fused = [sum((rows[member][key][1] - z[member]['mean']) / z[member]['std'] for member in subset) / len(subset)
                     for key in common]
            entry[split] = {'auroc': auroc(fused, member_labels), 'ap': ap(fused, member_labels),
                            'rows': len(common)}
        if entry['dev'] is not None and entry['official'] is not None:
            entry['mean_auroc'] = (entry['dev']['auroc'] + entry['official']['auroc']) / 2.0
        results.append(entry)
    results = [entry for entry in results if entry.get('mean_auroc') is not None]
    results.sort(key=lambda entry: -entry['mean_auroc'])
    receipt = {'status': 'hp_combo_search_complete', 'z': z,
               'frozen_members': sorted(frozen), 'baseline': args.baseline,
               'threshold': args.baseline + 0.0005,
               'top': results[:24], 'combos_evaluated': len(results),
               'official_read': True, 'official_use': 'authorized_validation_only',
               'test_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(args.output, receipt)}, indent=1)[:4000])
    print('RECEIPT', args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--run-dir', required=True)
    export.add_argument('--site', required=True)
    export.add_argument('--variant', choices=('ema', 'online'), required=True)
    export.add_argument('--output', required=True)
    forward = sub.add_parser('forward')
    forward.add_argument('--export', required=True)
    forward.add_argument('--site', required=True)
    forward.add_argument('--member', default='HP')
    forward.add_argument('--data-root', required=True)
    forward.add_argument('--dev-indices', required=True)
    forward.add_argument('--map-cache', default=None)
    forward.add_argument('--device', default='cuda:0')
    forward.add_argument('--batch-size', type=int, default=64)
    forward.add_argument('--shard', type=int, default=0)
    forward.add_argument('--shards', type=int, default=1)
    forward.add_argument('--out-dir', required=True)
    combo = sub.add_parser('combo')
    combo.add_argument('--shard-dir', required=True)
    combo.add_argument('--members', required=True)
    combo.add_argument('--frozen-z', required=True)
    combo.add_argument('--baseline', type=float, required=True)
    combo.add_argument('--output', required=True)
    combo_draft = sub.add_parser('combo-draft')
    combo_draft.add_argument('--shard-dir', required=True)
    combo_draft.add_argument('--anchor-shard-dir', required=True)
    combo_draft.add_argument('--anchor', default='HP')
    combo_draft.add_argument('--members', required=True)
    combo_draft.add_argument('--triple-with', nargs='+', default=None)
    combo_draft.add_argument('--baseline', type=float, required=True)
    combo_draft.add_argument('--sanity-pair', nargs=2, default=None)
    combo_draft.add_argument('--output', required=True)
    args = parser.parse_args()
    {'export': cmd_export, 'forward': cmd_forward, 'combo': cmd_combo,
     'combo-draft': cmd_combo_draft}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

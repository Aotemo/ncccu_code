"""Compare fp32 and fp16-stored selected weights on fixed internal-dev chunks."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from f04_export_member import construct_member, eligible_row, verify_published
from f04_size_ensemble import sha


def paired_logits(model32, model16, images, age_features, job, forward):
    import torch
    if model32.training or model16.training:
        raise ValueError('paired prediction requires evaluation mode')
    with torch.inference_mode():
        left = forward(model32, images, age_features, job).reshape(-1).float()
        right = forward(model16, images, age_features, job).reshape(-1).float()
    if (left.shape != right.shape or left.numel() != images.shape[0]
            or not torch.isfinite(left).all() or not torch.isfinite(right).all()):
        raise FloatingPointError('invalid paired dev logits')
    return left, right


def verify_prediction_blob(source, metadata):
    tree = json.loads((metadata / 'REMOTE-TREE.json').read_text())
    verified = {}
    for name in ('tools/diagnostics/f04_predict_member.py', 'tools/diagnostics/paired_evidence.py'):
        path = source / name
        matches = [item for item in tree['tree'] if item['path'] == name]
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        if path.is_symlink() or len(matches) != 1 or matches[0]['sha'] != blob or matches[0]['mode'] != mode:
            raise ValueError('published paired-prediction dependency differs: ' + name)
        verified[name] = sha(path)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'remote-metadata', 'inventory', 'export-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('git-commit', 'inventory-sha256', 'export-sha256', 'round-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--offset', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2] or sha(args.inventory) != args.inventory_sha256:
        raise ValueError('source or sealed inventory differs')
    git_files = verify_published(source, args.remote_metadata, args.git_commit)
    git_files.update(verify_prediction_blob(source, args.remote_metadata))
    if sha(args.export_receipt) != args.export_sha256:
        raise ValueError('export receipt SHA differs')
    exported = json.loads(args.export_receipt.read_text())
    if (exported.get('status') != 'F04_complete_inference_state_exported_and_strictly_reloaded'
            or exported.get('round_id') != args.round_id or exported.get('inventory', {}).get('sha256') != args.inventory_sha256
            or exported.get('official_read') is not False or exported.get('GPU_initialized') is not False):
        raise ValueError('sealed export identity differs')
    for precision in ('fp32', 'fp16'):
        file = exported['exports'][precision]; path = Path(file['path'])
        if path.is_symlink() or path.stat().st_size != file['bytes'] or sha(path) != file['sha256']:
            raise ValueError('exported weight file differs')
    config = json.loads((source / 'configs/F03-THRESHOLD-EQUAL.json').read_text())
    inventory = json.loads(args.inventory.read_text())
    row = eligible_row(inventory, args.round_id, config['approved_general_public_weight_sha256'])
    sys.path.insert(0, str(source / 'tools/pretraining'))
    from source_chain import cpu_context
    context = cpu_context()
    model32, job, site, runtime, science = construct_member(row)
    import torch
    from torch.utils.data import DataLoader, Subset
    from retrain.age import forward, unpack_batch
    from retrain.data import ImageDataset
    from retrain.engine import _loader_options
    from retrain.ema import tensor_digest
    import paired_evidence
    if tensor_digest(model32.state_dict()) != row['selected']['tensor_sha256']:
        raise ValueError('original selected tensor differs')
    model16 = copy.deepcopy(model32)
    model32.load_state_dict(torch.load(exported['exports']['fp32']['path'], map_location='cpu', weights_only=True), strict=True)
    model16.load_state_dict(torch.load(exported['exports']['fp16']['path'], map_location='cpu', weights_only=True), strict=True)
    if tensor_digest(model32.state_dict()) != row['selected']['tensor_sha256']:
        raise ValueError('fp32 export does not restore the selected state')
    model32.eval(); model16.eval()
    reference = paired_evidence._reference(inventory['manifest'])
    manifest = json.loads(Path(inventory['manifest']['path']).read_text())
    if len(reference['dev']) != 5140 or sum(r['label'] for r in reference['dev']) != 167:
        raise ValueError('fixed internal-dev population differs')
    if args.offset < 0 or not 1 <= args.count <= 128 or args.offset + args.count > len(reference['dev']):
        raise ValueError('explicit bounded internal-dev chunk required')
    work = Path(site['work_root']).resolve(strict=True); output = args.output.resolve()
    if not output.is_relative_to(work / 'technical-prep') or output.exists():
        raise ValueError('fresh technical-prep output required')
    data_root = Path(site['data_root']).resolve(strict=True)
    cache = Path(site['prepared_cache']).resolve(strict=True)
    if (data_root != (work / science['paths']['data_root']).resolve(strict=True)
            or cache != (work / science['paths']['prepared_cache']).resolve(strict=True)):
        raise ValueError('frozen train data or prepared cache differs')
    os.environ['TRAIN_IMAGE_CACHE'] = str(cache)
    dataset = ImageDataset(data_root, manifest['rows'], int(job['image_size']), training=False,
                           normalization=job.get('normalization', 'imagenet'),
                           prepared_size=int(job.get('prepared_size', 0)), age=job.get('age'),
                           input_preprocessing=job.get('input_preprocessing'))
    indices = manifest['dev_indices'][args.offset:args.offset + args.count]
    loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, **_loader_options(0, job))
    device = torch.device('cpu'); predictions = []; started = time.time()
    for batch in loader:
        images, labels, row_indices, age = unpack_batch(batch, job, device)
        left, right = paired_logits(model32, model16, images, age, job, forward)
        for index, label, score32, score16 in zip(row_indices.tolist(), labels.tolist(), left.tolist(), right.tolist()):
            predictions.append({'name': manifest['rows'][index]['name'], 'label': int(label),
                                'fp32': float(score32), 'fp16': float(score16)})
    expected = reference['dev'][args.offset:args.offset + args.count]
    if (len(predictions) != args.count or any((actual['name'], actual['label']) != (wanted['name'], wanted['label'])
            for actual, wanted in zip(predictions, expected)) or torch.cuda.is_initialized()):
        raise ValueError('dev chunk identity or CPU-only execution differs')
    output.mkdir(parents=True)
    path = output / 'PRIVATE-PAIRED-DEV.csv'
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'label', 'fp32', 'fp16'])
        writer.writeheader(); writer.writerows(predictions)
    result = {'status': 'F04_paired_internal_dev_chunk_completed_no_fusion', 'at': time.time(),
              'round_id': args.round_id, 'offset': args.offset, 'count': args.count,
              'inventory_sha256': args.inventory_sha256, 'export_sha256': args.export_sha256,
              'git_commit': args.git_commit, 'git_files': git_files, 'execution': context,
              'prediction': {'path': str(path), 'sha256': sha(path)}, 'batch_size': 8,
              'storage_precision': ['fp32', 'fp16'], 'inference_arithmetic': 'fp32',
              'model_forward_calls_per_batch': 2, 'TTA': False, 'official_read': False,
              'GPU_initialized': False, 'seconds': time.time() - started}
    receipt = output / 'CHUNK.json'
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'round_id': args.round_id, 'offset': args.offset,
                      'count': args.count, 'seconds': result['seconds'], 'receipt_sha256': sha(receipt)}, indent=2), flush=True)


if __name__ == '__main__':
    main()

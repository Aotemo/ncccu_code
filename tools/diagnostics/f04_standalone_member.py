"""Prepare one F04 fp16 member for checkpoint-free strict inference loading."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from f04_export_member import construct_member, eligible_row, verify_published
from f04_size_ensemble import sha


def export_metadata(job, model_sha256, tensor_sha256, *, variant='ema'):
    """Build the existing frozen-inference format, including the age contract."""
    if (not isinstance(job, dict) or not job.get('model') or variant not in ('ema', 'online')
            or not all(re.fullmatch(r'[0-9a-f]{64}', value or '')
                       for value in (model_sha256, tensor_sha256))):
        raise ValueError('complete model and state identities are required')
    return {'format': 'glaucoma-frozen-inference-v1', 'model_sha256': model_sha256,
            'job': copy.deepcopy(job), 'age': copy.deepcopy(job.get('age')),
            'checkpoint': {'variant': variant, 'tensor_sha256': tensor_sha256},
            'inference': {'threshold': .5, 'precision': 'fp32'}}


def verify_own_blob(source, metadata):
    tree = json.loads((metadata / 'REMOTE-TREE.json').read_text())
    name = 'tools/diagnostics/f04_standalone_member.py'
    path = source / name
    matches = [entry for entry in tree['tree'] if entry['path'] == name]
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    mode = '100755' if path.stat().st_mode & 0o111 else '100644'
    if path.is_symlink() or len(matches) != 1 or matches[0]['sha'] != blob or matches[0]['mode'] != mode:
        raise ValueError('published standalone source differs')
    return sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'remote-metadata', 'inventory', 'preflight', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('git-commit', 'inventory-sha256', 'preflight-sha256', 'round-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    if (source != Path(__file__).resolve().parents[2]
            or sha(args.inventory) != args.inventory_sha256 or sha(args.preflight) != args.preflight_sha256):
        raise ValueError('published source or sealed F04 inputs differ')
    git_files = verify_published(source, args.remote_metadata, args.git_commit)
    git_files['tools/diagnostics/f04_standalone_member.py'] = verify_own_blob(source, args.remote_metadata)
    preflight = json.loads(args.preflight.read_text())
    if (preflight.get('status') != 'F04_fp16_size_members_sealed_before_blend'
            or preflight.get('old_inventory_sha256') != args.inventory_sha256
            or args.round_id not in preflight['fp16']['members']
            or preflight['fp16']['bytes'] > 500_000_000):
        raise ValueError('member is not in the sealed size-limited F04 candidate')
    export_ref = preflight['exports'][args.round_id]
    export_path = Path(export_ref['path'])
    if export_path.is_symlink() or sha(export_path) != export_ref['sha256']:
        raise ValueError('sealed F04 export receipt differs')
    exported = json.loads(export_path.read_text())
    weight = exported['exports']['fp16']; weight_path = Path(weight['path'])
    if (exported.get('status') != 'F04_complete_inference_state_exported_and_strictly_reloaded'
            or exported.get('round_id') != args.round_id
            or exported['inventory']['sha256'] != args.inventory_sha256
            or weight_path.is_symlink() or weight_path.stat().st_size != weight['bytes']
            or weight['bytes'] != export_ref['sizes']['fp16_bytes'] or sha(weight_path) != weight['sha256']):
        raise ValueError('fp16 file or original export identity differs')
    config = json.loads((source / 'configs/F03-THRESHOLD-EQUAL.json').read_text())
    inventory = json.loads(args.inventory.read_text())
    row = eligible_row(inventory, args.round_id, config['approved_general_public_weight_sha256'])
    sys.path.insert(0, str(source / 'tools/pretraining'))
    from source_chain import cpu_context
    context = cpu_context()
    original, job, site, runtime, _ = construct_member(row)
    import torch
    from retrain.age import forward
    from retrain.ema import tensor_digest
    from retrain.inference import _load_model
    state = torch.load(weight_path, map_location='cpu', weights_only=True)
    state_sha = tensor_digest(state)
    original.load_state_dict(state, strict=True)
    original.eval()
    restored_sha = tensor_digest(original.state_dict())
    work = Path(site['work_root']).resolve(strict=True)
    output = args.output.resolve()
    if not output.is_relative_to(work / 'technical-prep') or output.exists():
        raise ValueError('fresh technical-prep output required')
    output.mkdir(parents=True)
    os.link(weight_path, output / 'model.pt', follow_symlinks=False)
    if not os.path.samefile(weight_path, output / 'model.pt') or sha(output / 'model.pt') != weight['sha256']:
        raise ValueError('fp16 hard-link identity differs')
    selected_variant = row['selected']['variant']
    metadata = export_metadata(job, weight['sha256'], state_sha, variant=selected_variant)
    with (output / 'model.json').open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False); stream.write('\n')
    loaded, loaded_job, _, origin, device = _load_model(output, device='cpu', variant=selected_variant)
    if (loaded_job != job or str(device) != 'cpu' or origin['model_sha256'] != weight['sha256']
            or origin['variant'] != selected_variant or origin['tensor_sha256'] != state_sha
            or tensor_digest(loaded.state_dict()) != restored_sha):
        raise ValueError('checkpoint-free model construction or strict state differs')
    size = int(job['image_size'])
    synthetic = torch.zeros((1, 3, size, size), dtype=torch.float32)
    age = torch.zeros((1, 2), dtype=torch.float32) if (job.get('age') or {}).get('enabled') else None
    with torch.inference_mode():
        expected = forward(original, synthetic, age, job).reshape(-1).float()
        actual = forward(loaded, synthetic, age, job).reshape(-1).float()
    if (expected.shape != (1,) or actual.shape != (1,) or not torch.isfinite(expected).all()
            or not torch.isfinite(actual).all() or not torch.allclose(expected, actual, rtol=1e-6, atol=1e-6)
            or torch.cuda.is_initialized()):
        raise ValueError('synthetic checkpoint-free single-view forward differs')
    result = {'status': 'F04_fp16_standalone_strict_load_and_synthetic_forward_passed',
              'at': time.time(), 'round_id': args.round_id, 'execution': context,
              'git_commit': args.git_commit, 'git_files': git_files,
              'preflight_sha256': args.preflight_sha256, 'inventory_sha256': args.inventory_sha256,
              'selected_variant': selected_variant,
              'export_receipt_sha256': export_ref['sha256'], 'frozen_source_commit': runtime['git_commit'],
              'model': {'path': str(output / 'model.pt'), 'bytes': weight['bytes'],
                        'sha256': weight['sha256'], 'fp16_tensor_sha256': state_sha,
                        'restored_fp32_tensor_sha256': restored_sha, 'hardlink_to_export': True},
              'metadata': {'path': str(output / 'model.json'), 'sha256': sha(output / 'model.json')},
              'synthetic_max_abs_logit_difference': float((expected - actual).abs().max().item()),
              'synthetic_forward_only': True, 'dev_read': False, 'official_read': False,
              'TTA': False, 'GPU_initialized': False}
    receipt = output / 'STANDALONE.json'
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'round_id': args.round_id,
                      'fp16_tensor_sha256': state_sha, 'receipt_sha256': sha(receipt)}, indent=2), flush=True)


if __name__ == '__main__':
    main()

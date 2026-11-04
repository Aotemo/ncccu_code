"""Export one sealed F04 candidate with its own frozen inference runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from ensemble_dev import f03_candidate_gate
from f04_size_ensemble import export_state_files, sha


def eligible_row(inventory, round_id, approved):
    if (inventory.get('status') != 'read_only_F03_candidate_discovery_no_fusion'
            or inventory.get('official_read') is not False):
        raise ValueError('sealed internal-dev discovery required')
    matches = [row for row in inventory['members'] if row.get('round_id') == round_id]
    if len(matches) != 1:
        raise ValueError('unique candidate identity required')
    row = matches[0]
    if f03_candidate_gate(row, .988, approved)[0] != 'eligible' or len(row.get('checkpoint_candidates', [])) != 1:
        raise ValueError('candidate not eligible or selected checkpoint not unique')
    return row


def verify_published(source, metadata, commit):
    source, metadata = Path(source), Path(metadata)
    ref = json.loads((metadata / 'REMOTE-REF.json').read_text())
    remote = json.loads((metadata / 'REMOTE-COMMIT.json').read_text())
    tree = json.loads((metadata / 'REMOTE-TREE.json').read_text())
    if (ref['object']['sha'] != commit or remote['sha'] != commit or remote['tree']['sha'] != tree['sha']
            or tree['truncated'] is not False):
        raise ValueError('published commit/tree differs')
    entries = {item['path']: item for item in tree['tree'] if item['type'] == 'blob'}
    names = ('tools/diagnostics/f04_size_ensemble.py', 'tools/diagnostics/f04_export_member.py',
             'tools/diagnostics/ensemble_dev.py', 'tests/test_f04_size_ensemble.py',
             'configs/F03-THRESHOLD-EQUAL.json', 'todo/auroc-999/F04-SIZE-LIMIT-20261104.md')
    result = {}
    for name in names:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('published source missing or symbolic')
        raw = path.read_bytes()
        digest = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        if entries[name]['sha'] != digest or entries[name]['mode'] != mode:
            raise ValueError('published Git blob differs: ' + name)
        result[name] = sha(path)
    return result


def construct_member(row):
    """Strictly reconstruct the pinned selected model in a fresh process."""
    freeze_path = Path(row['spec']['freeze']['path'])
    if sha(freeze_path) != row['spec']['freeze']['sha256']:
        raise ValueError('frozen model metadata differs')
    freeze = json.loads(freeze_path.read_text()); root = freeze_path.parent; bundle = root / 'bundle'
    for relative, expected in freeze['source_files'].items():
        if sha(bundle / relative) != expected:
            raise ValueError('frozen source differs: ' + relative)
    checkpoint = row['checkpoint_candidates'][0]
    if sha(checkpoint['sidecar']['path']) != checkpoint['sidecar']['sha256']:
        raise ValueError('selected sidecar SHA differs')
    if json.loads(Path(checkpoint['sidecar']['path']).read_text())['sha256'] != row['selected_blob_sha256']:
        raise ValueError('selected sidecar blob differs')
    science_path = Path(row['science']['path'])
    if sha(science_path) != row['science']['sha256']:
        raise ValueError('selected science SHA differs')
    science = json.loads(science_path.read_text())
    if (science['public_source'] != row['public_source'] or science['job'].get('initial_checkpoint') is not None
            or science['candidate'].get('weight_sha256') != science['public_source']['sha256']):
        raise ValueError('public source or fresh initialization differs')
    site_path = root / 'SITE_BINDING.json'; runtime = json.loads((root / 'RUNTIME_BINDING.json').read_text())
    if sha(site_path) != runtime['site_binding_sha256']:
        raise ValueError('historical frozen site differs')
    site = json.loads(site_path.read_text())
    if 'retrain' in sys.modules:
        raise ValueError('each historical model requires a fresh interpreter')
    os.environ.update(NCCCU_SITE_BINDING=str(site_path), NCCCU_SITE_BINDING_SHA256=sha(site_path))
    sys.path.insert(0, str(bundle))
    import torch
    torch.set_num_threads(max(1, int(os.environ['NCCCU_CPU_THREADS']) - 2)); torch.set_num_interop_threads(1)
    from retrain.inference import _load_model
    from retrain.ema import tensor_digest
    selected = row['selected']
    model, job, metadata, origin, device = _load_model(checkpoint['path'], device='cpu', variant=selected['variant'])
    if (origin['sha256'] != row['selected_blob_sha256'] or tensor_digest(model.state_dict()) != selected['tensor_sha256']
            or origin['successful_updates'] != selected['successful_updates'] or str(device) != 'cpu'):
        raise ValueError('selected model identity differs')
    for key in ('model', 'image_size', 'prepared_size', 'normalization', 'precision'):
        if job.get(key) != science['job'].get(key):
            raise ValueError('frozen job differs: ' + key)
    return model, job, site, runtime, science


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'remote-metadata', 'inventory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('git-commit', 'inventory-sha256', 'round-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2] or sha(args.inventory) != args.inventory_sha256:
        raise ValueError('source or sealed inventory differs')
    git_files = verify_published(source, args.remote_metadata, args.git_commit)
    config = json.loads((source / 'configs/F03-THRESHOLD-EQUAL.json').read_text())
    inventory = json.loads(args.inventory.read_text())
    row = eligible_row(inventory, args.round_id, config['approved_general_public_weight_sha256'])
    sys.path.insert(0, str(source / 'tools/pretraining'))
    from source_chain import cpu_context
    context = cpu_context()
    model, job, site, runtime, science = construct_member(row)
    work = Path(site['work_root']).resolve(strict=True); output = args.output.resolve()
    if not output.is_relative_to(work / 'technical-prep') or output.exists():
        raise ValueError('fresh technical-prep export directory required')
    output.mkdir(parents=True)
    files = export_state_files(model, output)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('GPU initialized during CPU export')
    result = {'status': 'F04_complete_inference_state_exported_and_strictly_reloaded',
              'at': time.time(), 'execution': context, 'git_commit': args.git_commit, 'git_files': git_files,
              'inventory': {'path': str(args.inventory), 'sha256': args.inventory_sha256},
              'round_id': args.round_id, 'selected': row['selected'], 'frozen_source_commit': runtime['git_commit'],
              'freeze': row['spec']['freeze'], 'science': row['science'], 'checkpoint': row['checkpoint_candidates'][0],
              'selected_blob_sha256': row['selected_blob_sha256'], 'exports': files,
              'model_constructed': True, 'forward_calls': 0, 'GPU_initialized': False, 'official_read': False}
    path = output / 'EXPORT.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'round_id': args.round_id,
                      'parameter_count': files['parameter_count'], 'fp32': files['fp32'],
                      'fp16': files['fp16'], 'receipt_sha256': sha(path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()

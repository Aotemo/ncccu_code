"""One fixed F04 internal-dev verification using existing model and safety chains."""
import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pinned(path, expected):
    path = Path(path)
    if path.is_symlink() or not re.fullmatch(r'[0-9a-f]{64}', expected or '') or sha(path) != expected:
        raise ValueError('pinned file differs: ' + str(path))
    return json.loads(path.read_text())


def write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def build_plan(members):
    members = list(members)
    if len(members) != 5 or len(set(members)) != 5 or any(not isinstance(x, str) or not x for x in members):
        raise ValueError('the fixed five distinct F04 members are required')
    result = []
    for member, world in zip(members, (2, 2, 2, 1, 1)):
        for rank in range(world):
            result.append({'device_rank': len(result), 'member': member, 'rank': rank, 'world': world})
    return result


def require_safe_predecessor(safe, expected_root, now):
    at = safe.get('observed_at')
    if (safe.get('phase') != 'terminal' or safe.get('formal_root') != str(expected_root)
            or type(at) not in (int, float) or not math.isfinite(at) or not 0 <= now - at <= 120
            or safe.get('first_success_receipt_read') is not True
            or safe.get('done', {}).get('present') is not True
            or safe.get('done', {}).get('status') != 'completed'
            or safe.get('safety', {}).get('safe') is not True
            or safe.get('signals_sent') is not False or safe.get('model_forward_executed') is not False
            or safe.get('official_read') is not False):
        raise ValueError('fresh completed predecessor safety proof required')
    return safe


def canonical_member(rows, indices, packets, world):
    from retrain.parallel_evaluation import canonical_predictions
    return canonical_predictions(rows, indices, packets, 8, world)


def guard_modules(source):
    """Import current safety helpers under a private name; historical retrain remains isolated."""
    package = Path(source) / 'runtime/bundle/retrain'
    name = '_f04_current_guard'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, package / '__init__.py',
                                                       submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return (importlib.import_module(name + '.site_binding'),
            importlib.import_module(name + '.execution_policy'),
            importlib.import_module(name + '.persistence'),
            importlib.import_module(name + '.distributed_launcher'))


def verify_own_blob(source, metadata):
    tree = json.loads((Path(metadata) / 'REMOTE-TREE.json').read_text())
    name = 'tools/diagnostics/f04_gpu_full_dev.py'
    path = Path(source) / name
    matches = [entry for entry in tree['tree'] if entry['path'] == name]
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    mode = '100755' if path.stat().st_mode & 0o111 else '100644'
    if path.is_symlink() or len(matches) != 1 or matches[0]['sha'] != blob or matches[0]['mode'] != mode:
        raise ValueError('published F04 GPU source differs')
    return sha(path)


def prepare(args):
    """Seal the five already-selected files; preparation never initializes a GPU."""
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('explicit published source root differs')
    sys.path.insert(0, str(source / 'tools/pretraining'))
    from f04_export_member import eligible_row, verify_published
    from source_chain import cpu_context
    import paired_evidence
    context = cpu_context()
    files = verify_published(source, args.remote_metadata, args.git_commit)
    files['tools/diagnostics/f04_gpu_full_dev.py'] = verify_own_blob(source, args.remote_metadata)
    inventory = read_pinned(args.inventory, args.inventory_sha256)
    preflight = read_pinned(args.preflight, args.preflight_sha256)
    if (preflight.get('status') != 'F04_fp16_size_members_sealed_before_blend'
            or preflight.get('old_inventory_sha256') != args.inventory_sha256
            or preflight['fp16']['bytes'] > 500_000_000 or preflight.get('official_read') is not False):
        raise ValueError('fixed F04 preflight differs')
    members = preflight['fp16']['members']; plan = build_plan(members)
    reference = paired_evidence._reference(inventory['manifest'])
    manifest = read_pinned(inventory['manifest']['path'], inventory['manifest']['sha256'])
    if (len(manifest['dev_indices']) != 5140 or sum(row['label'] for row in reference['dev']) != 167
            or len(args.standalone_receipt) != 5):
        raise ValueError('fixed complete internal dev or five receipts required')
    config = json.loads((source / 'configs/F03-THRESHOLD-EQUAL.json').read_text())
    receipts = {}; work_roots = set()
    for path in args.standalone_receipt:
        path = path.resolve(strict=True); record = json.loads(path.read_text())
        member = record.get('round_id')
        if member not in members or member in receipts:
            raise ValueError('standalone receipt identity differs')
        row = eligible_row(inventory, member, config['approved_general_public_weight_sha256'])
        ref = preflight['exports'][member]
        exported = read_pinned(ref['path'], ref['sha256'])
        model = record.get('model', {}); metadata = record.get('metadata', {})
        model_path = Path(model['path']); metadata_path = Path(metadata['path'])
        if (record.get('status') != 'F04_fp16_standalone_strict_load_and_synthetic_forward_passed'
                or record.get('preflight_sha256') != args.preflight_sha256
                or record.get('inventory_sha256') != args.inventory_sha256
                or record.get('export_receipt_sha256') != ref['sha256']
                or record.get('official_read') is not False or record.get('GPU_initialized') is not False
                or record.get('dev_read') is not False or record.get('TTA') is not False
                or model.get('hardlink_to_export') is not True
                or model_path.is_symlink() or model_path.stat().st_size != model['bytes']
                or model['bytes'] != ref['sizes']['fp16_bytes'] or sha(model_path) != model['sha256']
                or model['sha256'] != exported['exports']['fp16']['sha256']
                or metadata_path.is_symlink() or sha(metadata_path) != metadata['sha256']):
            raise ValueError('standalone model or source receipt differs')
        freeze = read_pinned(exported['freeze']['path'], exported['freeze']['sha256'])
        root = Path(exported['freeze']['path']).parent; bundle = root / 'bundle'
        for relative, expected in freeze['source_files'].items():
            if sha(bundle / relative) != expected:
                raise ValueError('historical source differs: ' + relative)
        site_path = root / 'SITE_BINDING.json'
        runtime = json.loads((root / 'RUNTIME_BINDING.json').read_text())
        site = read_pinned(site_path, runtime['site_binding_sha256'])
        work_roots.add(str(Path(site['work_root']).resolve(strict=True)))
        receipts[member] = {'round_id': member, 'receipt': {'path': str(path), 'sha256': sha(path)},
            'model': model, 'metadata': metadata, 'freeze': exported['freeze'],
            'bundle': str(bundle), 'site': {'path': str(site_path), 'sha256': runtime['site_binding_sha256']},
            'data_root': site['data_root'], 'cache': site['prepared_cache'], 'selected': row['selected']}
    if set(receipts) != set(members) or sum(receipts[name]['model']['bytes'] for name in members) != preflight['fp16']['bytes']:
        raise ValueError('five-member serialized byte total differs')
    output = args.output.resolve()
    if len(work_roots) != 1:
        raise ValueError('member work roots differ')
    work = Path(next(iter(work_roots)))
    if output.exists() or not output.is_relative_to(work / 'technical-prep'):
        raise ValueError('fresh technical-prep GPU preflight required')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'F04_full_dev_gpu_inputs_sealed_no_GPU', 'at': time.time(),
              'git_commit': args.git_commit, 'source_root': str(source), 'git_files': files,
              'inventory': {'path': str(args.inventory), 'sha256': args.inventory_sha256},
              'preflight': {'path': str(args.preflight), 'sha256': args.preflight_sha256},
              'manifest': inventory['manifest'], 'members': [receipts[name] for name in members], 'plan': plan,
              'weight_bytes': preflight['fp16']['bytes'], 'dev_count': 5140, 'dev_positive': 167,
              'batch_size': 8, 'arithmetic': 'fp32', 'TTA': False, 'official_read': False,
              'GPU_initialized': False, 'execution': context}
    write_new(output, result)
    print(json.dumps({'status': result['status'], 'members': len(members), 'receipt_sha256': sha(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare',))
    for name in ('source-root', 'remote-metadata', 'inventory', 'preflight', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('git-commit', 'inventory-sha256', 'preflight-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--standalone-receipt', type=Path, action='append', required=True)
    args = parser.parse_args()
    prepare(args)


if __name__ == '__main__':
    main()

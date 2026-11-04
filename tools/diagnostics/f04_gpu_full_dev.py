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
import subprocess
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


def validate_current_preflight(refreshed, current, current_sha, safe, inputs):
    """Bind the fixed F04 screening to a census taken after the safe predecessor."""
    discovered_at, screened_at, safe_at = current.get('at'), refreshed.get('at'), safe.get('observed_at')
    if (current.get('status') != 'read_only_F03_candidate_discovery_no_fusion'
            or current.get('official_read') is not False
            or refreshed.get('status') != 'F04_fp16_size_members_sealed_before_blend'
            or refreshed.get('official_read') is not False
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in (discovered_at, screened_at, safe_at))
            or not safe_at <= discovered_at <= screened_at
            or refreshed.get('inventory_sha256') != current_sha
            or refreshed.get('old_inventory_sha256') != inputs['inventory']['sha256']
            or refreshed.get('fp16', {}).get('members') != [x['round_id'] for x in inputs['members']]
            or refreshed.get('fp16', {}).get('bytes') != inputs['weight_bytes']):
        raise ValueError('freshly screened F04 inventory, fixed members or storage differs')
    predecessor = Path(safe['formal_root']).name
    selected_auroc = (safe.get('selected') or {}).get('auroc')
    matching = [row for row in current.get('members', []) if row.get('round_id') == predecessor]
    observed = (matching[0].get('selected') or {}).get('auroc') if len(matching) == 1 else None
    if (type(selected_auroc) not in (int, float) or not math.isfinite(selected_auroc)
            or type(observed) not in (int, float) or not math.isfinite(observed)
            or not math.isclose(observed, selected_auroc, rel_tol=0, abs_tol=1e-12)):
        raise ValueError('safe predecessor selected result absent or differs in the fresh census')
    if selected_auroc >= .988 and predecessor not in refreshed.get('eligible_order', []):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from f04_bounded_fusion import preregistered_exclusion
        reason = preregistered_exclusion(predecessor)
        excluded = [row for row in refreshed.get('excluded', [])
                    if row.get('round_id') == predecessor and row.get('reason') == reason]
        if reason is None or len(excluded) != 1:
            raise ValueError('eligible predecessor is absent from the refreshed F04 selection')
    return refreshed


def validate_evaluation_window(window, safe_sha, site_sha, round_id, now, free, *, reserve=1024 ** 3):
    """Keep the prepared next round and the peer's full reserved storage intact."""
    at, required, recorded = window.get('at'), window.get('required_free_bytes'), window.get('free_bytes')
    if (window.get('status') != 'cross_batch_safe_window_and_capacity_verified'
            or window.get('predecessor_safe_sha256') != safe_sha or window.get('site_sha256') != site_sha
            or window.get('round_id') != round_id or window.get('activation_reserved') is not False
            or window.get('GPU_started') is not False or type(at) not in (int, float)
            or not math.isfinite(at) or not 0 <= now - at <= 120
            or any(type(value) is not int for value in (required, recorded, free, reserve))
            or required <= 0 or reserve <= 0 or recorded <= required + reserve or free <= required + reserve):
        raise ValueError('fresh protected next-round/peer capacity window required')
    return window


def validate_proof_barrier(proofs, launches, parent, verify_devices, identity):
    """Release only eight supervised, live, identity-matched accelerator ranks."""
    if (not isinstance(parent, dict) or type(parent.get('pid')) is not int
            or not str(parent.get('start_ticks', '')).isdecimal()
            or len(proofs) != 8 or len(launches) != 8
            or {row.get('rank') for row in proofs} != set(range(8))
            or {row.get('rank') for row in launches} != set(range(8))):
        raise ValueError('eight supervised rank identities required')
    proofs = sorted(proofs, key=lambda row: row['rank'])
    launches = sorted(launches, key=lambda row: row['rank'])
    if len({row.get('pid') for row in proofs}) != 8:
        raise ValueError('rank process identities must be distinct')
    for rank, (proof, launch) in enumerate(zip(proofs, launches)):
        actual = identity(proof.get('pid'))
        if (proof.get('parent') != parent or proof.get('world_size') != 8
                or proof.get('device') != 'cuda:' + str(rank)
                or type(proof.get('pid')) is not int or proof['pid'] <= 0
                or not str(proof.get('start_ticks', '')).isdecimal()
                or launch.get('pid') != proof['pid']
                or str(launch.get('start_ticks')) != str(proof['start_ticks'])
                or actual.get('pid') != proof['pid'] or actual.get('active') is not True
                or actual.get('parent_pid') != parent['pid']
                or str(actual.get('start_ticks')) != str(proof['start_ticks'])):
            raise ValueError('rank supervisor, parent or live process identity differs')
    verify_devices(proofs)
    return proofs


def validate_worker_release(release, rank, proof_sha, spec_sha, parent, identity):
    """A worker cannot read data or forward before its own proven parent releases it."""
    proof_map = release.get('proof_sha256', {})
    if (release.get('status') != 'all_rank_device_proofs_passed'
            or release.get('parent') != parent or release.get('spec_sha256') != spec_sha
            or type(rank) is not int or not 0 <= rank < 8
            or set(proof_map) != {str(i) for i in range(8)}
            or any(not re.fullmatch(r'[0-9a-f]{64}', value or '') for value in proof_map.values())
            or proof_map[str(rank)] != proof_sha):
        raise ValueError('complete proof-bound supervised release required')
    actual = identity(parent['pid'])
    if (actual.get('pid') != parent['pid'] or actual.get('active') is not True
            or str(actual.get('start_ticks')) != str(parent['start_ticks'])):
        raise ValueError('GPU owner exited before worker release')
    return release


def process_identity(pid):
    if type(pid) is not int or pid <= 0:
        return {'pid': pid, 'active': False}
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'parent_pid': int(fields[1]),
                'start_ticks': fields[19], 'active': fields[0] not in ('Z', 'X')}
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return {'pid': pid, 'active': False}


def parent_death_guard(parent):
    import ctypes
    current = process_identity(parent['pid'])
    if (os.getppid() != parent['pid'] or not current.get('active')
            or current.get('start_ticks') != str(parent['start_ticks'])):
        raise ValueError('expected supervising parent is not alive')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent['pid']:
        raise RuntimeError('parent-death termination could not be bound')


def reserve_evaluation_attempt(path, request):
    path = Path(path)
    path.mkdir(mode=0o700, exist_ok=False)
    write_new(path / 'REQUEST.json', request)
    digest = sha(path / 'REQUEST.json')
    write_new(path / 'CLAIM.json', {'at': time.time(), 'request_sha256': digest, 'one_shot': True})
    return digest


def worker_batch_records(rows, expected, actual, labels, logits, probabilities, rank, batch):
    expected, actual = list(expected), list(actual)
    if (not expected or actual != expected or len(set(expected)) != len(expected)
            or any(type(index) is not int or not 0 <= index < len(rows) for index in expected)
            or not all(len(values) == len(expected) for values in (labels, logits, probabilities))
            or type(rank) is not int or rank < 0 or type(batch) is not int or batch < 0):
        raise ValueError('evaluation batch identity or dimensions differ')
    result = []
    for offset, (index, label, logit, probability) in enumerate(zip(expected, labels, logits, probabilities)):
        if (label != rows[index]['label'] or not math.isfinite(float(logit))
                or not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1):
            raise ValueError('evaluation label or finite score differs')
        result.append({'index': index, 'rank': rank, 'batch': batch, 'offset': offset,
                       'name': rows[index]['name'], 'label': int(label),
                       'logit': float(logit), 'probability': float(probability)})
    return result


def canonical_member(rows, indices, packets, world):
    from retrain.parallel_evaluation import canonical_predictions
    return canonical_predictions(rows, indices, packets, 8, world)


def evaluation_dataset(dataset_class, data_root, rows, job):
    """Keep each frozen runtime's default when no custom preprocessing was registered."""
    options = {'normalization': job.get('normalization', 'imagenet'),
               'prepared_size': int(job.get('prepared_size', 0)), 'age': job.get('age')}
    if job.get('input_preprocessing') is not None:
        options['input_preprocessing'] = job['input_preprocessing']
    return dataset_class(data_root, rows, int(job['image_size']), False, **options)


def compare_full_dev(labels, fp32, fp16, members):
    """Compare the same fixed members without retaining per-sample vectors."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from f04_size_ensemble import compare_predictions, equal_zscore
    build_plan(members)
    old = equal_zscore(labels, fp32, members)
    new = equal_zscore(labels, fp16, members)
    member_comparisons = {}
    for name in members:
        before = np.asarray(fp32[name], dtype=np.float64)
        after = np.asarray(fp16[name], dtype=np.float64)
        old_z = (before - old['normalization'][name]['mean']) / old['normalization'][name]['std']
        new_z = (after - new['normalization'][name]['mean']) / new['normalization'][name]['std']
        member_comparisons[name] = {**compare_predictions(labels, before, after),
                                    'max_abs_z_score_difference': float(np.max(np.abs(old_z - new_z)))}
    return {'members': member_comparisons,
            'fp32': {'metrics': old['metrics'], 'normalization': old['normalization']},
            'fp16': {'metrics': new['metrics'], 'normalization': new['normalization']},
            'ensemble_max_abs_z_score_difference': float(np.max(np.abs(old['scores'] - new['scores'])))}


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


def worker(args):
    """One isolated historical-runtime shard; no data or model before the eight-proof barrier."""
    import types
    spec = read_pinned(args.worker_spec, args.worker_spec_sha256)
    root = Path(spec['source_root']).resolve(strict=True)
    if root != Path(__file__).resolve().parents[2] or sha(__file__) != spec['worker_source_sha256']:
        raise ValueError('worker source differs from the published owner')
    rank = args.rank
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError('valid admitted GPU rank required')
    parent = spec['parent']
    parent_death_guard(parent)
    guard, execution, _, _ = guard_modules(root)
    site = guard.load_site(spec['site_binding'], expected_sha256=spec['site_binding_sha256'])
    if os.environ.get('NCCCU_SITE_BINDING') != spec['site_binding']:
        raise ValueError('worker site environment differs')
    device_name = 'cuda:' + str(rank)
    execution.bind_worker_cpu(spec, rank, device_name)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 8:
        raise ValueError('the admitted eight accelerators must be visible')
    torch.cuda.set_device(rank)
    context = types.SimpleNamespace(rank=rank, world_size=8, device=torch.device(device_name))
    proof = execution.rank_proof(context, spec)
    self_identity = process_identity(os.getpid())
    if self_identity.get('parent_pid') != parent['pid']:
        raise ValueError('worker was reparented before device proof')
    proof.update(start_ticks=self_identity['start_ticks'], parent=parent)
    output = Path(spec['output'])
    proof_path = output / 'proofs' / ('rank-' + str(rank) + '.json')
    write_new(proof_path, proof)
    own_sha = sha(proof_path)
    release_path = output / 'RELEASE.json'
    while not release_path.exists():
        current = process_identity(parent['pid'])
        if (os.getppid() != parent['pid'] or not current.get('active')
                or current.get('start_ticks') != str(parent['start_ticks'])
                or time.time() >= spec['deadline_at']):
            raise RuntimeError('owner or deadline failed before supervised release')
        time.sleep(.05)
    validate_worker_release(json.loads(release_path.read_text()), rank, own_sha,
                            args.worker_spec_sha256, parent, process_identity)
    sealed = read_pinned(spec['inputs']['path'], spec['inputs']['sha256'])
    assignment = sealed['plan'][rank]
    if assignment['device_rank'] != rank:
        raise ValueError('GPU rank assignment differs')
    matching = [member for member in sealed['members'] if member['round_id'] == assignment['member']]
    if len(matching) != 1:
        raise ValueError('worker member identity is ambiguous')
    member = matching[0]
    model_path, metadata_path = Path(member['model']['path']), Path(member['metadata']['path'])
    if (model_path.parent != metadata_path.parent or model_path.name != 'model.pt'
            or metadata_path.name != 'model.json' or model_path.is_symlink() or metadata_path.is_symlink()
            or model_path.stat().st_size != member['model']['bytes']
            or sha(model_path) != member['model']['sha256'] or sha(metadata_path) != member['metadata']['sha256']):
        raise ValueError('immutable exported inference state differs')
    manifest = read_pinned(sealed['manifest']['path'], sealed['manifest']['sha256'])
    rows, indices = manifest['rows'], manifest['dev_indices']
    if (len(indices) != 5140 or len(set(indices)) != 5140
            or sum(rows[i]['label'] == 1 for i in indices) != 167
            or any(not rows[i]['relative_path'].startswith('train/') for i in indices)):
        raise ValueError('only the complete sealed internal dev is permitted')
    bundle = Path(member['bundle']).resolve(strict=True)
    sys.path.insert(0, str(bundle))
    from retrain.inference import _load_model
    from retrain.parallel_evaluation import batch_assignments
    from retrain.data import ImageDataset
    from retrain.age import unpack_batch, forward
    from torch.utils.data import DataLoader
    os.environ['TRAIN_IMAGE_CACHE'] = str(Path(member['cache']) / 'train/sealed')
    os.environ['TRAIN_IMAGE_CACHE_READONLY'] = '1'
    precision = importlib.import_module('_f04_current_guard.precision')
    runtime_guard = importlib.import_module('_f04_current_guard.fresh')
    precision.configure(precision.RULE)
    model, job, _, origin, actual_device = _load_model(model_path.parent, device=device_name, variant='ema')
    if (str(actual_device) != device_name or origin.get('model_sha256') != member['model']['sha256']
            or origin.get('tensor_sha256') != member['model']['fp16_tensor_sha256']):
        raise ValueError('strict historical runtime or fp16 state differs')
    model.eval()
    assignments = batch_assignments(indices, 8, assignment['rank'], assignment['world'])
    dataset = evaluation_dataset(ImageDataset, member['data_root'], rows, job)
    loader = DataLoader(dataset, batch_sampler=[batch for _, batch in assignments], num_workers=0)
    records = []
    with torch.inference_mode():
        for (ordinal, expected), batch in zip(assignments, loader):
            parent_now = process_identity(parent['pid'])
            if (os.getppid() != parent['pid'] or not parent_now.get('active')
                    or parent_now.get('start_ticks') != str(parent['start_ticks'])
                    or runtime_guard.runtime_stop_reason(spec['deadline_at'], actual_device) is not None):
                raise RuntimeError('owner or deadline failed during inference')
            images, labels, actual_indices, age = unpack_batch(batch, job, actual_device)
            with torch.autocast(device_type='cuda', enabled=False):
                logits = forward(model, images, age, job).reshape(-1).float()
            probabilities = logits.sigmoid()
            records.extend(worker_batch_records(rows, expected, actual_indices.tolist(), labels.tolist(),
                logits.cpu().tolist(), probabilities.cpu().tolist(), assignment['rank'], ordinal))
    packet = {'rank': assignment['rank'], 'device_rank': rank, 'member': assignment['member'],
              'error': None, 'records': records, 'proof_sha256': own_sha,
              'input_sha256': spec['inputs']['sha256'], 'TTA': False, 'official_read': False}
    packet_path = output / 'packets' / ('rank-' + str(rank) + '.json')
    write_new(packet_path, packet)
    print(json.dumps({'status': 'worker_completed', 'rank': rank, 'records': len(records),
                      'packet_sha256': sha(packet_path)}), flush=True)


def release_supervised_ranks(output, spec_path, spec_sha, site, parent, deadline, stop):
    """Run alongside the existing supervisor; failed or dead ranks never receive a release."""
    guard, _, persistence, _ = guard_modules(Path(__file__).resolve().parents[2])
    proofs_root, launches_root = output / 'proofs', output / 'processes'
    try:
        while not stop.is_set() and time.time() < deadline:
            proof_paths = [proofs_root / ('rank-' + str(i) + '.json') for i in range(8)]
            launch_paths = [launches_root / ('rank-' + str(i) + '-launch.json') for i in range(8)]
            if all(path.is_file() for path in proof_paths + launch_paths):
                try:
                    proofs = [json.loads(path.read_text()) for path in proof_paths]
                    launches = [{**json.loads(path.read_text()), 'rank': i} for i, path in enumerate(launch_paths)]
                except (OSError, json.JSONDecodeError):
                    time.sleep(.05)
                    continue
                validate_proof_barrier(proofs, launches, parent,
                    lambda rows: guard.verify_rank_devices(rows, site), process_identity)
                release = {'status': 'all_rank_device_proofs_passed', 'at': time.time(), 'parent': parent,
                           'spec_sha256': spec_sha,
                           'proof_sha256': {str(i): sha(path) for i, path in enumerate(proof_paths)}}
                path = output / 'RELEASE.json'
                if path.exists():
                    raise FileExistsError('one-shot proof release already exists')
                persistence.atomic_json(path, release)
                return
            time.sleep(.05)
        raise TimeoutError('all eight supervised device proofs did not arrive in time')
    except Exception as error:
        write_new(output / 'BARRIER-FAILURE.json', {'at': time.time(), 'type': type(error).__name__,
                                                     'message': str(error)})
        stop.set()


def aggregate_packets(source, inputs, output, inputs_sha):
    """Reduce only the sealed internal-dev shards; row-level predictions remain private."""
    import numpy as np
    sys.path[:0] = [str(source / 'runtime/bundle'), str(source / 'tools/diagnostics')]
    import paired_evidence
    reference = paired_evidence._reference(inputs['manifest'])
    manifest = read_pinned(inputs['manifest']['path'], inputs['manifest']['sha256'])
    rows, indices = manifest['rows'], manifest['dev_indices']
    if len(indices) != 5140 or len(reference['dev']) != 5140:
        raise ValueError('complete sealed dev required')
    packets = []
    release = json.loads((output / 'RELEASE.json').read_text())
    for rank in range(8):
        path = output / 'packets' / ('rank-' + str(rank) + '.json')
        packet = json.loads(path.read_text())
        expected = inputs['plan'][rank]
        if (packet.get('device_rank') != rank or packet.get('member') != expected['member']
                or packet.get('rank') != expected['rank'] or packet.get('error') is not None
                or packet.get('input_sha256') != inputs_sha
                or packet.get('proof_sha256') != release.get('proof_sha256', {}).get(str(rank))
                or packet.get('TTA') is not False or packet.get('official_read') is not False):
            raise ValueError('worker shard identity differs')
        packets.append(packet)
    before, after = {}, {}
    members = [row['round_id'] for row in inputs['members']]
    for member in inputs['members']:
        name = member['round_id']
        selected = member['selected']
        spec = {'freeze': member['freeze'], 'completion': selected['completion']}
        old, details, _ = paired_evidence._selected(spec, reference)
        if any(details[key] != selected[key] for key in ('variant', 'cycle', 'successful_updates', 'tensor_sha256')):
            raise ValueError('original selected member changed')
        group = [packet for packet in packets if packet['member'] == name]
        world = next(row['world'] for row in inputs['plan'] if row['member'] == name)
        predictions = canonical_member(rows, indices, group, world)
        before[name] = old
        after[name] = [float(row['logit']) for row in predictions]
    result = compare_full_dev([row['label'] for row in reference['dev']], before, after, members)
    if any(not np.isfinite(value) for group in result['members'].values()
           for value in (group['max_abs_logit_difference'], group['max_abs_z_score_difference'])):
        raise ValueError('nonfinite full-dev precision difference')
    return result


def owner(args):
    """One gated evaluation attempt; next-round activation remains a separate one-shot chain."""
    parent_death_guard({'pid': args.controller_pid, 'start_ticks': args.controller_start_ticks})
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('explicit published F04 source differs')
    guard, _, persistence, launcher = guard_modules(source)
    site = guard.load_site(args.site, expected_sha256=args.site_sha256)
    os.environ['NCCCU_SITE_BINDING'] = str(args.site)
    os.environ['NCCCU_SITE_BINDING_SHA256'] = args.site_sha256
    # Dispatch already verified the initial full allocation and loaded the native
    # affinity guard; the detached child must inherit the admitted owner mask.
    guard.bind_cpu_role(site, 'owner', initial=False)
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        if os.environ.get(key) != ','.join(str(i) for i in range(8)):
            raise ValueError('explicit ordered accelerator visibility required')
    snapshot = json.loads(args.pod_live.read_text())
    ready = any(x.get('type') == 'Ready' and x.get('status') == 'True'
                for x in snapshot.get('status', {}).get('conditions', []))
    guard.verify_live(site, pod_uid=snapshot.get('metadata', {}).get('uid'), ready=ready,
                      observed_at=args.pod_live.stat().st_mtime)
    safe = read_pinned(args.safe, args.safe_sha256)
    require_safe_predecessor(safe, safe['formal_root'], time.time())
    proof = read_pinned(args.tree_proof, args.tree_proof_sha256)
    remote_ref = json.loads((args.remote_metadata / 'REMOTE-REF.json').read_text())
    remote_commit = json.loads((args.remote_metadata / 'REMOTE-COMMIT.json').read_text())
    remote_tree = json.loads((args.remote_metadata / 'REMOTE-TREE.json').read_text())
    if (proof.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
            or proof.get('commit') != args.git_commit or proof.get('tree') != args.git_tree
            or remote_ref.get('object', {}).get('sha') != args.git_commit
            or remote_commit.get('sha') != args.git_commit or remote_commit.get('tree', {}).get('sha') != args.git_tree
            or remote_tree.get('sha') != args.git_tree or remote_tree.get('truncated') is not False
            or proof.get('GPU_initialized') is not False or verify_own_blob(source, args.remote_metadata) != sha(__file__)):
        raise ValueError('published full-tree proof differs')
    inputs = read_pinned(args.inputs, args.inputs_sha256)
    if (inputs.get('status') != 'F04_full_dev_gpu_inputs_sealed_no_GPU'
            or inputs.get('plan') != build_plan([x['round_id'] for x in inputs['members']])
            or inputs.get('weight_bytes', 500_000_001) > 500_000_000
            or inputs.get('dev_count') != 5140 or inputs.get('dev_positive') != 167
            or inputs.get('batch_size') != 8 or inputs.get('TTA') is not False
            or inputs.get('official_read') is not False):
        raise ValueError('fixed F04 input inventory differs')
    refreshed = read_pinned(args.current_preflight, args.current_preflight_sha256)
    current_inventory = read_pinned(args.current_inventory, args.current_inventory_sha256)
    validate_current_preflight(refreshed, current_inventory, args.current_inventory_sha256, safe, inputs)
    ready_root = args.next_ready.resolve(strict=True)
    prep = read_pinned(ready_root / 'PREPARATION-RESULT.json', args.next_preparation_sha256)
    cpu_path = ready_root / 'formal-cpu/CPU-RECEIPT.json'
    deployment = ready_root / 'DEPLOYMENT.json'
    cpu = json.loads(cpu_path.read_text())
    if (prep.get('status') != 'formal_CPU_and_precise_git_deployment_ready'
            or prep.get('site_sha256') != args.site_sha256 or prep.get('round_id') != site['round_id']
            or sha(cpu_path) != prep.get('formal_cpu_sha256') or sha(deployment) != prep.get('deployment_sha256')
            or cpu.get('exit_code') != 0 or len(cpu.get('checks', [])) < 18
            or any(x.get('passed') is not True for x in cpu['checks'])
            or Path(site['round_root']).exists() or (ready_root / 'activation-audit').exists()):
        raise ValueError('fresh successor formal preparation differs')
    window = read_pinned(args.window, args.window_sha256)
    validate_evaluation_window(window, args.safe_sha256, args.site_sha256, site['round_id'],
                               time.time(), shutil.disk_usage(site['work_root']).free)
    grant_ref = site['cpu_execution']['allocation_receipt']
    grant = read_pinned(grant_ref['path'], grant_ref['sha256'])
    clock = json.loads((ready_root / 'formal-cpu/package/CLOCK_POLICY.json').read_text())
    if (not time.time() < args.deadline_at <= time.time() + 1800
            or grant['valid_until'] - args.deadline_at < clock['total_seconds'] + 60):
        raise ValueError('evaluation would consume the successor training lease')
    for member in inputs['members']:
        model, metadata = member['model'], member['metadata']
        standalone = read_pinned(member['receipt']['path'], member['receipt']['sha256'])
        refreshed_export = refreshed['exports'].get(member['round_id'], {})
        if (refreshed_export.get('sha256') != standalone.get('export_receipt_sha256')
                or Path(model['path']).is_symlink() or Path(metadata['path']).is_symlink()
                or Path(model['path']).stat().st_size != model['bytes']
                or sha(model['path']) != model['sha256'] or sha(metadata['path']) != metadata['sha256']):
            raise ValueError('member inference asset changed')
        freeze = read_pinned(member['freeze']['path'], member['freeze']['sha256'])
        for relative, digest in freeze['source_files'].items():
            if sha(Path(member['bundle']) / relative) != digest:
                raise ValueError('historical member runtime changed')
    if sha(inputs['manifest']['path']) != inputs['manifest']['sha256']:
        raise ValueError('internal dev manifest changed')
    sys.path[:0] = [str(source / 'runtime/bundle'), str(source / 'tools/activation')]
    import activate
    identities = safe['identity']
    previous = [identities['owner_expected'], *identities['ranks_expected']]
    if len(previous) != 9:
        raise ValueError('previous owner plus eight exact rank identities required')
    require_safe_predecessor(safe, safe['formal_root'], time.time())
    guard.verify_live(site, pod_uid=snapshot.get('metadata', {}).get('uid'), ready=ready,
                      observed_at=args.pod_live.stat().st_mtime)
    validate_evaluation_window(window, args.safe_sha256, args.site_sha256, site['round_id'],
                               time.time(), shutil.disk_usage(site['work_root']).free)
    admission = activate.admit_now(ready_root / 'formal-cpu/package', site, previous)
    output = args.output.resolve()
    if not output.is_relative_to(Path(site['work_root']) / 'technical-prep') or output.exists():
        raise ValueError('fresh isolated evaluation attempt required')
    output.mkdir(parents=True)
    (output / 'proofs').mkdir(); (output / 'packets').mkdir()
    parent_actual = process_identity(os.getpid())
    parent = {'pid': os.getpid(), 'start_ticks': parent_actual['start_ticks']}
    spec = {'source_root': str(source), 'worker_source_sha256': sha(__file__), 'output': str(output),
            'inputs': {'path': str(args.inputs), 'sha256': args.inputs_sha256},
            'site_binding': str(args.site), 'site_binding_sha256': args.site_sha256,
            'parent': parent, 'deadline_at': args.deadline_at,
            'devices': ['cuda:' + str(i) for i in range(8)],
            'cpu_masks': site['cpu_execution']['rank_masks'],
            'cpu_width': len(site['cpu_execution']['rank_masks'][0])}
    spec_path = output / 'WORKER-SPEC.json'; write_new(spec_path, spec); spec_sha = sha(spec_path)
    tasks = [{'name': 'rank-' + str(i), 'argv': [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
        'worker', '--worker-spec', str(spec_path), '--worker-spec-sha256', spec_sha, '--rank', str(i)]}
        for i in range(8)]
    stop = threading.Event()
    try:
        with persistence.RunLock(guard.instance_lock(site)):
            import run_round
            require_safe_predecessor(safe, safe['formal_root'], time.time())
            validate_evaluation_window(window, args.safe_sha256, args.site_sha256, site['round_id'],
                                       time.time(), shutil.disk_usage(site['work_root']).free)
            if run_round.device_users_before_start():
                raise RuntimeError('device handle appeared after acquiring the instance lock')
            resources = run_round.runtime_safety_snapshot()
            if len(resources['cards']) != 8 or any(card['busy_percent'] != 0 or card['vram_used_bytes'] >= 64 * 1024 ** 2
                                                   for card in resources['cards']):
                raise RuntimeError('accelerators not idle after acquiring the instance lock')
            barrier = threading.Thread(target=release_supervised_ranks,
                args=(output, spec_path, spec_sha, site, parent, min(args.deadline_at, time.time() + 180), stop), daemon=True)
            barrier.start()
            from retrain.fresh import start_runtime_guard
            monitor = start_runtime_guard(args.deadline_at, stop,
                lambda reason: persistence.atomic_json(output / 'RUNTIME-STOP.json', {'reason': reason, 'at': time.time()}))
            try:
                result = launcher.supervise_ranks(tasks, output / 'processes', 20, stop_event=stop, env=os.environ.copy())
            finally:
                stop.set(); barrier.join(timeout=5); monitor.join(timeout=2)
            if result['status'] != 'completed' or not (output / 'RELEASE.json').is_file():
                raise RuntimeError('supervised evaluation did not complete all eight ranks')
            released = []
            for rank in range(8):
                launch = json.loads((output / 'processes' / ('rank-' + str(rank) + '-launch.json')).read_text())
                current = process_identity(launch['pid'])
                if current.get('active') and current.get('start_ticks') == str(launch['start_ticks']):
                    raise RuntimeError('supervised evaluation rank remains active')
                released.append({'pid': launch['pid'], 'start_ticks': str(launch['start_ticks']), 'inactive': True})
            idle_until = min(args.deadline_at, time.time() + 45)
            while True:
                users = run_round.device_users_before_start()
                resources = run_round.runtime_safety_snapshot()
                idle = (not users and len(resources['cards']) == 8 and all(
                    card['busy_percent'] == 0 and card['vram_used_bytes'] < 64 * 1024 ** 2
                    for card in resources['cards']))
                if idle:
                    break
                if time.time() >= idle_until:
                    raise RuntimeError('accelerator release remains unknown after evaluation')
                time.sleep(.5)
            metrics = aggregate_packets(source, inputs, output, args.inputs_sha256)
            report = {'status': 'F04_full_internal_dev_fp16_storage_evaluated', 'at': time.time(),
                      'inputs_sha256': args.inputs_sha256, 'current_preflight_sha256': args.current_preflight_sha256,
                      'predecessor_safe_sha256': args.safe_sha256, 'site_sha256': args.site_sha256,
                      'weight_bytes': inputs['weight_bytes'], 'dev_count': 5140, 'dev_positive': 167,
                      'comparison': metrics, 'supervisor': result, 'admission': admission,
                      'released_ranks': released, 'device_handles_after': users, 'resources_after': resources,
                      'TTA': False, 'official_read': False, 'training_executed': False}
            write_new(output / 'SUMMARY.json', report)
            print(json.dumps({'status': report['status'], 'summary_sha256': sha(output / 'SUMMARY.json')}), flush=True)
    except Exception as error:
        if not (output / 'FAILURE.json').exists():
            write_new(output / 'FAILURE.json', {'status': 'failed', 'at': time.time(),
                                               'type': type(error).__name__, 'message': str(error),
                                               'official_read': False})
        raise


OWNER_PATH_OPTIONS = ('source-root', 'remote-metadata', 'site', 'safe', 'tree-proof', 'inputs',
                      'current-preflight', 'current-inventory', 'next-ready', 'window', 'pod-live', 'output')
OWNER_TEXT_OPTIONS = ('site-sha256', 'safe-sha256', 'tree-proof-sha256', 'git-commit', 'git-tree',
                      'inputs-sha256', 'current-preflight-sha256', 'current-inventory-sha256',
                      'next-preparation-sha256', 'window-sha256')


def dispatch(args):
    """Claim and start one detached controller; a transport timeout never permits a retry."""
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('published F04 source differs')
    guard, _, _, _ = guard_modules(source)
    site = guard.load_site(args.site, expected_sha256=args.site_sha256)
    os.environ.update(NCCCU_SITE_BINDING=str(args.site), NCCCU_SITE_BINDING_SHA256=args.site_sha256)
    guard.prepare_native_entry(site)
    guard.bind_cpu_role(site, 'owner', initial=True)
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        if os.environ.get(key) != ','.join(str(i) for i in range(8)):
            raise ValueError('explicit ordered accelerator visibility required')
    snapshot = json.loads(args.pod_live.read_text())
    ready = any(x.get('type') == 'Ready' and x.get('status') == 'True'
                for x in snapshot.get('status', {}).get('conditions', []))
    guard.verify_live(site, pod_uid=snapshot.get('metadata', {}).get('uid'), ready=ready,
                      observed_at=args.pod_live.stat().st_mtime)
    safe = read_pinned(args.safe, args.safe_sha256)
    require_safe_predecessor(safe, safe['formal_root'], time.time())
    window = read_pinned(args.window, args.window_sha256)
    validate_evaluation_window(window, args.safe_sha256, args.site_sha256, site['round_id'],
                               time.time(), shutil.disk_usage(site['work_root']).free)
    claim = args.claim_dir.resolve()
    if (not claim.is_relative_to(Path(site['work_root']) / 'technical-prep')
            or args.output.resolve() != claim / 'evaluation' or claim.exists() or claim.is_symlink()):
        raise ValueError('fresh isolated one-shot evaluation claim required')
    request = {name: str(getattr(args, name.replace('-', '_'))) for name in (*OWNER_PATH_OPTIONS, *OWNER_TEXT_OPTIONS)}
    request.update(deadline_at=args.deadline_at, source_sha256=sha(__file__))
    request_sha = reserve_evaluation_attempt(claim, request)
    argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'controller',
            '--claim-dir', str(claim), '--request-sha256', request_sha]
    try:
        with (claim / 'CONTROLLER-private.log').open('x') as log:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=source, env=os.environ.copy(), start_new_session=True)
        identity = process_identity(process.pid)
        write_new(claim / 'CONTROLLER-SPAWN.json', {'at': time.time(), 'pid': process.pid,
                                                    'start_ticks': identity.get('start_ticks'),
                                                    'request_sha256': request_sha})
    except Exception as error:
        write_new(claim / 'DISPATCH-FAILURE.json', {'at': time.time(), 'type': type(error).__name__,
                                                    'message': str(error)})
        raise
    print(json.dumps({'status': 'one_shot_controller_spawned', 'claim_dir': str(claim),
                      'request_sha256': request_sha}), flush=True)


def controller(args):
    """Retain owner exit status after transport disconnect; no automatic restart."""
    claim = args.claim_dir.resolve(strict=True)
    request = read_pinned(claim / 'REQUEST.json', args.request_sha256)
    if (request.get('source_sha256') != sha(__file__)
            or Path(request['source-root']).resolve(strict=True) != Path(__file__).resolve().parents[2]
            or Path(request['output']).resolve() != claim / 'evaluation'):
        raise ValueError('detached request/source differs')
    current = process_identity(os.getpid())
    argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'owner']
    for name in (*OWNER_PATH_OPTIONS, *OWNER_TEXT_OPTIONS):
        argv.extend(['--' + name, request[name]])
    argv.extend(['--deadline-at', str(request['deadline_at']), '--controller-pid', str(os.getpid()),
                 '--controller-start-ticks', current['start_ticks']])
    try:
        with (claim / 'OWNER-private.log').open('x') as log:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=os.environ.copy())
        identity = process_identity(process.pid)
        write_new(claim / 'OWNER-SPAWN.json', {'at': time.time(), 'pid': process.pid,
                                               'start_ticks': identity.get('start_ticks'),
                                               'request_sha256': args.request_sha256})
        try:
            code = process.wait(timeout=max(1, request['deadline_at'] - time.time() + 30))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); code = process.wait()
        write_new(claim / 'RESULT.json', {'status': 'owner_exit0' if code == 0 else 'owner_failed',
                                         'at': time.time(), 'exit_code': code,
                                         'request_sha256': args.request_sha256, 'official_read': False})
    except Exception as error:
        write_new(claim / 'RESULT.json', {'status': 'controller_failed', 'at': time.time(),
                                         'type': type(error).__name__, 'message': str(error),
                                         'request_sha256': args.request_sha256, 'official_read': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    initial = modes.add_parser('prepare')
    for name in ('source-root', 'remote-metadata', 'inventory', 'preflight', 'output'):
        initial.add_argument('--' + name, type=Path, required=True)
    for name in ('git-commit', 'inventory-sha256', 'preflight-sha256'):
        initial.add_argument('--' + name, required=True)
    initial.add_argument('--standalone-receipt', type=Path, action='append', required=True)
    child = modes.add_parser('worker')
    child.add_argument('--worker-spec', type=Path, required=True)
    child.add_argument('--worker-spec-sha256', required=True)
    child.add_argument('--rank', type=int, required=True)
    for mode in ('owner', 'dispatch'):
        run = modes.add_parser(mode)
        for name in OWNER_PATH_OPTIONS:
            run.add_argument('--' + name, type=Path, required=True)
        for name in OWNER_TEXT_OPTIONS:
            run.add_argument('--' + name, required=True)
        run.add_argument('--deadline-at', type=float, required=True)
        if mode == 'owner':
            run.add_argument('--controller-pid', type=int, required=True)
            run.add_argument('--controller-start-ticks', required=True)
        else:
            run.add_argument('--claim-dir', type=Path, required=True)
    detached = modes.add_parser('controller')
    detached.add_argument('--claim-dir', type=Path, required=True)
    detached.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'worker': worker, 'owner': owner, 'dispatch': dispatch,
     'controller': controller}[args.mode](args)


if __name__ == '__main__':
    main()

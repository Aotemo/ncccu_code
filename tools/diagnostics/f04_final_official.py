"""Frozen F04 ensemble identity and one irreversible official reservation.

This adapter never opens official input by itself. The existing GPU evaluation
chain may read it only after validating the canonical reservation.
"""
import hashlib
import copy
import argparse
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
import re
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_final_freeze(value):
    if (not isinstance(value, dict) or value.get('format') != 'f04-final-frozen-v1'
            or value.get('status') != 'F04_ensemble_final_frozen'
            or value.get('official_evaluations_allowed') != 1
            or value.get('official_metadata_used_for_selection') is not False
            or value.get('TTA') is not False):
        raise ValueError('complete final F04 freeze without official selection or TTA required')
    members = value.get('members')
    if not isinstance(members, list) or len(members) != 5:
        raise ValueError('the five fixed final members are required')
    names, paths, total = set(), set(), 0
    for member in members:
        if not isinstance(member, dict):
            raise ValueError('member identity must be an object')
        name, model, metadata, normalization = (member.get('round_id'), member.get('model'),
                                                 member.get('metadata'), member.get('normalization'))
        if (not isinstance(name, str) or not name or name in names
                or not isinstance(model, dict) or not isinstance(metadata, dict)
                or not isinstance(normalization, dict)):
            raise ValueError('unique member with complete inference identity required')
        for item in (model, metadata):
            path = item.get('path')
            if (not isinstance(path, str) or not Path(path).is_absolute() or path in paths
                    or not _sha256(item.get('sha256'))):
                raise ValueError('distinct absolute pinned inference files required')
            paths.add(path)
        if (not _sha256(model.get('fp16_tensor_sha256')) or type(model.get('bytes')) is not int
                or model['bytes'] <= 0 or not _finite(normalization.get('mean'))
                or not _finite(normalization.get('std')) or normalization['std'] <= 0):
            raise ValueError('fp16 tensor, exact bytes and finite dev moments required')
        names.add(name)
        total += model['bytes']
    if type(value.get('weight_bytes')) is not int or value['weight_bytes'] != total or total > 500_000_000:
        raise ValueError('final serialized member weights exceed or differ from the size limit')
    official = value.get('official')
    if (not isinstance(official, dict) or not isinstance(official.get('manifest_path'), str)
            or not Path(official['manifest_path']).is_absolute()
            or not _sha256(official.get('manifest_sha256'))
            or not isinstance(official.get('data_root'), str) or not Path(official['data_root']).is_absolute()):
        raise ValueError('the frozen official manifest and data-root identities are required')
    dev = value.get('full_dev')
    if (not isinstance(dev, dict) or dev.get('count') != 5140 or dev.get('positive') != 167
            or dev.get('official_read') is not False or not _sha256(dev.get('summary_sha256'))
            or any(not _finite(dev.get(key)) or not 0 <= dev[key] <= 1
                   for key in ('fp32_auroc', 'fp16_auroc', 'fp16_ap'))):
        raise ValueError('complete pinned internal fp16 and fp32 validation required')
    if dev['fp32_auroc'] - dev['fp16_auroc'] > .0005:
        raise ValueError('fp16 AUROC decline requires coordinator decision before freeze')
    return value


def freeze_from_verified_dev(inputs, preflight, summary, identity, official, runtime):
    """Build a fixed freeze only from the final five-member full-dev GPU receipt."""
    if (not isinstance(identity, dict) or any(not _sha256(identity.get(name)) for name in
            ('inputs_sha256', 'preflight_sha256', 'lineage_sha256', 'summary_sha256'))
            or not isinstance(inputs, dict) or inputs.get('status') != 'F04_full_dev_gpu_inputs_sealed_no_GPU'
            or inputs.get('official_read') is not False or inputs.get('TTA') is not False
            or not isinstance(preflight, dict)
            or preflight.get('status') != 'F04_fp16_size_members_sealed_before_blend'
            or preflight.get('official_read') is not False or not isinstance(summary, dict)
            or summary.get('status') != 'F04_full_internal_dev_fp16_storage_evaluated'
            or summary.get('inputs_sha256') != identity['inputs_sha256']
            or summary.get('current_preflight_sha256') != identity['preflight_sha256']
            or summary.get('final_lineage_sha256') != identity['lineage_sha256']
            or summary.get('dev_count') != 5140 or summary.get('dev_positive') != 167
            or summary.get('TTA') is not False or summary.get('official_read') is not False
            or summary.get('training_executed') is not False
            or summary.get('supervisor', {}).get('status') != 'completed'
            or summary.get('device_handles_after') != []):
        raise ValueError('pinned final internal-dev evaluation and its lineage are required')
    released = summary.get('released_ranks')
    cards = summary.get('resources_after', {}).get('cards')
    if (not isinstance(released, list) or len(released) != 8
            or any(not isinstance(row, dict) or row.get('inactive') is not True
                   or type(row.get('pid')) is not int or row['pid'] <= 0
                   or not str(row.get('start_ticks', '')).isdecimal() for row in released)
            or len({row['pid'] for row in released}) != 8
            or not isinstance(cards, list) or len(cards) != 8
            or any(not isinstance(card, dict) or card.get('busy_percent') != 0
                   or type(card.get('vram_used_bytes')) is not int
                   or not 0 <= card['vram_used_bytes'] < 64 * 1024 ** 2 for card in cards)):
        raise ValueError('all eight evaluation ranks and devices must be released')
    members = inputs.get('members')
    if not isinstance(members, list) or len(members) != 5 or any(not isinstance(row, dict) for row in members):
        raise ValueError('five fixed inference members are required')
    ids = [member.get('round_id') for member in members]
    if (len(set(ids)) != 5 or any(not isinstance(name, str) or not name for name in ids)
            or preflight.get('fp16', {}).get('members') != ids
            or preflight.get('fp16', {}).get('bytes') != inputs.get('weight_bytes')
            or summary.get('weight_bytes') != inputs.get('weight_bytes')):
        raise ValueError('final selected order and serialized total changed')
    comparison = summary.get('comparison', {})
    fp32 = comparison.get('fp32', {}).get('metrics', {})
    half = comparison.get('fp16', {})
    metrics, norms = half.get('metrics', {}), half.get('normalization', {})
    if not isinstance(norms, dict) or set(norms) != set(ids):
        raise ValueError('one full-dev fp16 normalization per selected member is required')
    final_members = []
    for member in members:
        selected, source = member.get('selected'), member.get('freeze')
        if (not isinstance(selected, dict) or selected.get('variant') not in ('ema', 'online')
                or type(selected.get('cycle')) is not int or selected['cycle'] < 1
                or type(selected.get('successful_updates')) is not int or selected['successful_updates'] < 1
                or not _sha256(selected.get('tensor_sha256')) or not isinstance(source, dict)
                or not isinstance(source.get('path'), str) or not Path(source['path']).is_absolute()
                or not _sha256(source.get('sha256')) or not isinstance(member.get('bundle'), str)
                or not Path(member['bundle']).is_absolute() or member.get('data_root') != official.get('data_root')):
            raise ValueError('the exact selected variant and historical inference runtime are required')
        item = copy.deepcopy(member)
        item['normalization'] = copy.deepcopy(norms[member['round_id']])
        final_members.append(item)
    if (not isinstance(runtime, dict)
            or any(not isinstance(runtime.get(name), str) or not re.fullmatch(r'[0-9a-f]{40}', runtime[name])
                   for name in ('code_commit', 'code_tree'))
            or any(not _sha256(runtime.get(name)) for name in ('gpu_owner_sha256', 'official_adapter_sha256'))
            or any(not isinstance(runtime.get(name), str) or not runtime[name]
                   for name in ('python_version', 'torch_version', 'pillow_version'))):
        raise ValueError('pinned inference source and runtime versions are required')
    frozen = {'format': 'f04-final-frozen-v1', 'status': 'F04_ensemble_final_frozen',
              'official_evaluations_allowed': 1, 'official_metadata_used_for_selection': False,
              'TTA': False, 'weight_bytes': inputs['weight_bytes'], 'members': final_members,
              'full_dev': {'count': 5140, 'positive': 167, 'fp32_auroc': fp32.get('auroc'),
                           'fp32_ap': fp32.get('ap'), 'fp16_auroc': metrics.get('auroc'),
                           'fp16_ap': metrics.get('ap'), 'official_read': False,
                           'summary_sha256': identity['summary_sha256']},
              'identity': copy.deepcopy(identity), 'official': copy.deepcopy(official),
              'runtime': copy.deepcopy(runtime)}
    return validate_final_freeze(frozen)


def freeze_from_pinned_receipts(inputs_path, inputs_sha, preflight_path, preflight_sha,
                                summary_path, summary_sha, lineage_sha, official, runtime):
    """Build only from exact, durable internal-dev receipts; never read official data."""
    def read(path, digest):
        path = Path(path)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or not _sha256(digest) or sha(path) != digest):
            raise ValueError('pinned internal F04 receipt differs')
        return json.loads(path.read_text())
    inputs = read(inputs_path, inputs_sha)
    preflight = read(preflight_path, preflight_sha)
    summary = read(summary_path, summary_sha)
    identity = {'inputs_path': str(inputs_path), 'inputs_sha256': inputs_sha,
                'preflight_path': str(preflight_path), 'preflight_sha256': preflight_sha,
                'summary_path': str(summary_path), 'summary_sha256': summary_sha,
                'lineage_sha256': lineage_sha}
    return freeze_from_verified_dev(inputs, preflight, summary, identity, official, runtime)


def verify_member_assets(frozen):
    """Recheck all final fp16 files, selected variants, age and historical code."""
    validate_final_freeze(frozen)
    verified = []
    for member in frozen['members']:
        model, metadata = member['model'], member['metadata']
        model_path, metadata_path = Path(model['path']), Path(metadata['path'])
        selected, historical = member.get('selected'), member.get('freeze')
        bundle = Path(member.get('bundle', ''))
        if (not isinstance(selected, dict) or selected.get('variant') not in ('ema', 'online')
                or not isinstance(historical, dict) or not isinstance(historical.get('path'), str)
                or not _sha256(historical.get('sha256')) or not bundle.is_absolute() or bundle.is_symlink()
                or not bundle.is_dir() or model_path.parent != metadata_path.parent
                or model_path.name != 'model.pt' or metadata_path.name != 'model.json'
                or model_path.is_symlink() or metadata_path.is_symlink()
                or not model_path.is_file() or not metadata_path.is_file()
                or model_path.stat().st_size != model['bytes'] or sha(model_path) != model['sha256']
                or sha(metadata_path) != metadata['sha256']):
            raise ValueError('frozen inference files or selected source differ')
        data = json.loads(metadata_path.read_text())
        if (data.get('format') != 'glaucoma-frozen-inference-v1'
                or data.get('model_sha256') != model['sha256']
                or data.get('checkpoint', {}).get('variant') != selected['variant']
                or data.get('checkpoint', {}).get('tensor_sha256') != model['fp16_tensor_sha256']
                or not isinstance(data.get('job'), dict) or data.get('age') != data['job'].get('age')):
            raise ValueError('frozen model metadata, variant, tensor or age contract differs')
        freeze_path = Path(historical['path'])
        if freeze_path.is_symlink() or not freeze_path.is_file() or sha(freeze_path) != historical['sha256']:
            raise ValueError('historical selected freeze differs')
        prior = json.loads(freeze_path.read_text())
        if (any(prior.get('selected', {}).get(key) != selected.get(key)
                for key in ('variant', 'cycle', 'successful_updates', 'tensor_sha256'))
                or not isinstance(prior.get('source_files'), dict) or not prior['source_files']):
            raise ValueError('historical selected identity or source manifest differs')
        for relative, expected in prior['source_files'].items():
            path = PurePosixPath(relative)
            if (path.is_absolute() or not path.parts or any(part in ('', '.', '..') for part in path.parts)
                    or not _sha256(expected)):
                raise ValueError('historical source path or digest is invalid')
            actual = bundle.joinpath(*path.parts)
            if (any(bundle.joinpath(*path.parts[:i]).is_symlink() for i in range(1, len(path.parts) + 1))
                    or not actual.is_file() or sha(actual) != expected):
                raise ValueError('historical inference source differs')
        verified.append(member['round_id'])
    return verified


def reserve_official_once(freeze_path, freeze_sha256, work_root, output_root):
    """Exclusively reserve the fixed final directory before any official read."""
    work = Path(work_root).resolve(strict=True)
    directory = work / 'final' / 'F04-20261104'
    freeze_path = Path(freeze_path)
    if (directory.is_symlink() or not directory.is_dir() or directory.resolve() != directory
            or freeze_path.is_symlink() or freeze_path.resolve(strict=True) != directory / 'FREEZE.json'
            or not _sha256(freeze_sha256) or sha(freeze_path) != freeze_sha256):
        raise ValueError('canonical pinned final freeze required')
    output = Path(output_root).resolve()
    if not output.is_relative_to(directory) or output == directory:
        raise ValueError('official output must stay under the unique final directory')
    frozen = json.loads(freeze_path.read_text())
    validate_final_freeze(frozen)
    marker = directory / 'OFFICIAL-ONCE.json'
    record = {'status': 'reserved_do_not_retry', 'at': time.time(),
              'freeze_sha256': freeze_sha256, 'freeze_path': str(freeze_path.resolve()),
              'output_root': str(output), 'official_read_before_reservation': False}
    with marker.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return marker


def validate_official_population(manifest, records, csv_sha256):
    """Validate already-read official inputs against the existing sealed contract."""
    rows = manifest.get('rows') if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get('training_eligible') is not False
            or 'fit_indices' in manifest or not _sha256(csv_sha256)
            or manifest.get('source_identity', {}).get('valid_csv_sha256') != csv_sha256
            or not isinstance(rows, list) or len(rows) != 5849 or len(records) != 5849):
        raise ValueError('complete isolated official manifest and CSV identity required')
    names, positive = set(), 0
    for row, record in zip(rows, records):
        if not isinstance(row, dict) or not isinstance(record, dict):
            raise ValueError('official row must be an object')
        name, relative, label = row.get('name'), row.get('relative_path'), row.get('label')
        if (not isinstance(name, str) or not name or name.casefold() in names
                or not isinstance(relative, str) or '\\' in relative
                or len(PurePosixPath(relative).parts) != 2 or PurePosixPath(relative).parts[0] != 'valid'
                or PurePosixPath(relative).name.casefold() != name.casefold()
                or type(label) is not int or label not in (0, 1) or not _sha256(row.get('sha256'))
                or any(key not in row or row[key] != record.get(key) for key in ('name', 'label', 'age'))):
            raise ValueError('official row identity, label, age or path differs')
        names.add(name.casefold())
        positive += label
    if positive != 191:
        raise ValueError('official class coverage differs from 191 positive and 5658 negative')
    return rows


def read_official_inputs_once(freeze_path, freeze_sha256, work_root, output_root,
                              manifest_path, manifest_sha256, data_root, read_csv):
    """Reserve the one input read before opening the official manifest or CSV."""
    work = Path(work_root).resolve(strict=True)
    directory = work / 'final' / 'F04-20261104'
    freeze = Path(freeze_path)
    marker = directory / 'OFFICIAL-ONCE.json'
    if (freeze.is_symlink() or freeze.resolve(strict=True) != directory / 'FREEZE.json'
            or not _sha256(freeze_sha256) or sha(freeze) != freeze_sha256
            or marker.is_symlink() or not marker.is_file()):
        raise ValueError('canonical official reservation is missing or differs')
    frozen = validate_final_freeze(json.loads(freeze.read_text()))
    reservation = json.loads(marker.read_text())
    output = Path(output_root).resolve()
    if (reservation.get('status') != 'reserved_do_not_retry'
            or reservation.get('freeze_sha256') != freeze_sha256
            or reservation.get('freeze_path') != str(freeze.resolve())
            or reservation.get('output_root') != str(output)
            or reservation.get('official_read_before_reservation') is not False):
        raise ValueError('official reservation does not match the frozen attempt')
    official = frozen['official']
    if (str(manifest_path) != official['manifest_path'] or manifest_sha256 != official['manifest_sha256']
            or str(data_root) != official['data_root'] or not callable(read_csv)):
        raise ValueError('official input arguments differ from the final freeze')
    read_marker = directory / 'OFFICIAL-INPUT-READ-ONCE.json'
    with read_marker.open('x') as stream:
        json.dump({'status': 'reserved_do_not_retry', 'at': time.time(),
                   'freeze_sha256': freeze_sha256, 'output_root': str(output)}, stream)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    manifest = Path(manifest_path)
    if manifest.is_symlink() or not _sha256(manifest_sha256) or sha(manifest) != manifest_sha256:
        raise ValueError('sealed official manifest identity differs')
    sealed = json.loads(manifest.read_text())
    records, csv_sha = read_csv(Path(data_root), 'valid')
    return {'rows': validate_official_population(sealed, records, csv_sha),
            'csv_sha256': csv_sha, 'manifest_sha256': manifest_sha256}


def ensemble_with_frozen_moments(frozen, vectors):
    """Apply only the already-frozen internal-dev moments; never fit on official data."""
    import numpy as np
    validate_final_freeze(frozen)
    members = frozen['members']
    if not isinstance(vectors, dict) or set(vectors) != {row['round_id'] for row in members}:
        raise ValueError('all and only frozen member vectors are required')
    normalized = []
    length = None
    for member in members:
        values = np.asarray(vectors[member['round_id']], dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError('finite one-dimensional member scores required')
        if length is None:
            length = len(values)
        if len(values) != length:
            raise ValueError('member score coverage differs')
        moments = member['normalization']
        normalized.append((values - moments['mean']) / moments['std'])
    scores = np.stack(normalized).mean(axis=0, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite fused score')
    return scores.tolist()


def seal_from_published_receipts(args):
    """Create the one canonical freeze using only published code and internal receipts."""
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('explicit published source differs')
    sys.path[:0] = [str(source / 'tools/pretraining'), str(source / 'tools/diagnostics')]
    from source_chain import cpu_context
    from f04_bounded_fusion import verify_blob
    import f04_gpu_full_dev as gpu
    cpu = cpu_context()
    proof = gpu.read_pinned(args.tree_proof, args.tree_proof_sha256)
    ref = json.loads((args.remote_metadata / 'REMOTE-REF.json').read_text())
    commit = json.loads((args.remote_metadata / 'REMOTE-COMMIT.json').read_text())
    tree = json.loads((args.remote_metadata / 'REMOTE-TREE.json').read_text())
    if (proof.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
            or proof.get('commit') != args.git_commit or proof.get('tree') != args.git_tree
            or ref.get('ref') != 'refs/heads/main' or ref.get('object', {}).get('sha') != args.git_commit
            or commit.get('sha') != args.git_commit or commit.get('tree', {}).get('sha') != args.git_tree
            or tree.get('sha') != args.git_tree or tree.get('truncated') is not False
            or proof.get('GPU_initialized') is not False):
        raise ValueError('published full-tree receipt differs')
    adapter_sha = verify_blob(source, args.remote_metadata, 'tools/diagnostics/f04_final_official.py')
    owner_sha = verify_blob(source, args.remote_metadata, 'tools/diagnostics/f04_gpu_full_dev.py')
    verify_blob(source, args.remote_metadata, 'configs/F04-FINAL-FREEZE-20261104.json')
    if (adapter_sha != sha(__file__) or
            verify_blob(source, args.remote_metadata, 'records/analysis/20261104-f04/GPU-INPUTS-v3.json')
            != args.inputs_sha256):
        raise ValueError('published freeze code or GPU inputs differ')
    policy = json.loads((source / 'configs/F04-FINAL-FREEZE-20261104.json').read_text())
    safes = gpu.load_final_lineage(args.final_lineage, args.final_lineage_sha256,
                                   policy.get('last_training_rounds'))
    inputs = gpu.read_pinned(args.inputs, args.inputs_sha256)
    inventory = gpu.read_pinned(args.current_inventory, args.current_inventory_sha256)
    preflight = gpu.read_pinned(args.current_preflight, args.current_preflight_sha256)
    for safe in safes:
        gpu.validate_current_preflight(preflight, inventory, args.current_inventory_sha256, safe, inputs)
    site = gpu.read_pinned(args.site, args.site_sha256)
    config = site.get('science', {}).get('configuration', {})
    science = gpu.read_pinned(config.get('path', ''), config.get('sha256', ''))
    if (policy.get('max_fp16_member_weight_bytes') != 500_000_000
            or policy.get('official_evaluations_allowed') != 1 or policy.get('tta') is not False
            or site.get('data_root') != inputs['members'][0]['data_root']
            or not _sha256(science.get('official_manifest_sha256'))
            or not Path(site.get('official_manifest', '')).is_absolute()):
        raise ValueError('published final policy or official metadata identity differs')
    official = {'manifest_path': site['official_manifest'],
                'manifest_sha256': science['official_manifest_sha256'], 'data_root': site['data_root']}
    import PIL
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('freeze preparation must not initialize a GPU')
    runtime = {'code_commit': args.git_commit, 'code_tree': args.git_tree,
               'gpu_owner_sha256': owner_sha, 'official_adapter_sha256': adapter_sha,
               'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
               'pillow_version': PIL.__version__}
    frozen = freeze_from_pinned_receipts(args.inputs, args.inputs_sha256,
        args.current_preflight, args.current_preflight_sha256, args.summary, args.summary_sha256,
        args.final_lineage_sha256, official, runtime)
    verified = verify_member_assets(frozen)
    directory = Path(site['work_root']) / 'final/F04-20261104'
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or directory.resolve() != directory:
        raise ValueError('canonical final directory differs')
    if any((directory / name).exists() for name in ('OFFICIAL-ONCE.json', 'OFFICIAL-INPUT-READ-ONCE.json')):
        raise ValueError('official evaluation already claimed or read')
    path = directory / 'FREEZE.json'
    with path.open('x') as stream:
        json.dump(frozen, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'status': 'F04_final_frozen_before_official', 'freeze': str(path),
        'freeze_sha256': sha(path), 'members': verified, 'weight_bytes': frozen['weight_bytes'],
        'fp16_auroc': frozen['full_dev']['fp16_auroc'], 'fp16_ap': frozen['full_dev']['fp16_ap'],
        'official_read': False, 'GPU_initialized': False, 'cpu_uid': cpu['uid']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze'])
    for name in ('source-root', 'remote-metadata', 'tree-proof', 'inputs', 'current-preflight',
                 'current-inventory', 'summary', 'final-lineage', 'site'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('tree-proof-sha256', 'inputs-sha256', 'current-preflight-sha256',
                 'current-inventory-sha256', 'summary-sha256', 'final-lineage-sha256', 'site-sha256',
                 'git-commit', 'git-tree'):
        parser.add_argument('--' + name, required=True)
    seal_from_published_receipts(parser.parse_args())


if __name__ == '__main__':
    main()

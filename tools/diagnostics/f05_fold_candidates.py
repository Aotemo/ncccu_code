"""Augment a sealed F03 census with registered F05 folds; never select or forward.

All calls belong to the verified CPU channel.  Real discovery must use a
published, tree-verified source and a previously sealed candidate policy.
The original F03 artifact remains untouched; consumers must call
``verify_augmented`` on the separately sealed result before using it.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re

import paired_evidence
import f05_final_selection as final

KIND = 'f05-augmented-candidate-discovery-v1'
STATUS = 'read_only_F03_candidate_discovery_no_fusion'
ROUNDS = tuple(
    f'r{265 + number}-small-f05fold{number}-seed20261029-pairsigmoid-24k-p4-9010-20261105'
    for number in range(3)
)


def _reference(path, work):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or not path.resolve(strict=True).is_relative_to(work)):
        raise ValueError('regular evidence file inside work root required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def _read(reference, work):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or not isinstance(reference.get('path'), str) or not final._is_sha(reference.get('sha256'))):
        raise ValueError('pinned file path and SHA required')
    final._inside_work(reference, work)
    path = Path(reference['path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular evidence file inside work root required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('evidence bytes differ from the pinned SHA')
    try:
        result = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, OSError) as error:
        raise ValueError('valid evidence JSON required') from error
    if not isinstance(result, dict):
        raise ValueError('evidence JSON object required')
    return result


def _f03(reference, work, allow_fixture):
    inventory = _read(reference, work)
    if (inventory.get('status') != STATUS or inventory.get('official_read') is not False
            or inventory.get('forward_calls') != 0 or inventory.get('GPU_initialized') is not False
            or not isinstance(inventory.get('members'), list)
            or (allow_fixture and inventory.get('fixture_only') is not True)
            or (not allow_fixture and inventory.get('fixture_only') is True)
            or inventory.get('artifact_kind') == KIND or inventory.get('augmented') is True):
        raise ValueError('original sealed F03 inventory required')
    manifest_ref = inventory.get('manifest')
    manifest = _read(manifest_ref, work)
    if not allow_fixture and manifest_ref['sha256'] != final.BASE_MANIFEST_SHA:
        raise ValueError('original registered fit/dev manifest required')
    names = [row.get('round_id') if isinstance(row, dict) else None for row in inventory['members']]
    if (any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('F03 members must have unique round identities')
    for row in inventory['members']:
        number = _fold_number(row['round_id'])
        if number is not None and (row['round_id'] != ROUNDS[number]
                                   or row.get('status') != 'unknown_or_excluded'):
            raise ValueError('only an existing explicit unknown fold may be preserved from F03')
    reference_value = paired_evidence._reference(manifest_ref)
    if not allow_fixture and (len(manifest['fit_indices']) != 46256 or len(manifest['dev_indices']) != 5140
            or sum(manifest['rows'][i]['label'] for i in manifest['dev_indices']) != 167):
        raise ValueError('original fixed fit/dev population differs')
    return inventory, manifest, reference_value


def _fold_number(name):
    if not isinstance(name, str):
        return None
    match = re.match(r'^r(265|266|267)(?:-|$)', name, re.IGNORECASE)
    return int(match.group(1)) - 265 if match else None


def _unknown(name, reason, **details):
    return {'round_id': name, 'status': 'unknown_or_excluded', 'reason': reason, **details}


def _subdirs(path):
    """Explicit scans surface filesystem errors instead of silently truncating glob results."""
    with os.scandir(path) as entries:
        result = []
        for entry in entries:
            if entry.is_symlink():
                raise ValueError('symlink prevents complete checkpoint enumeration')
            if entry.is_dir(follow_symlinks=False):
                result.append(Path(entry.path))
        return sorted(result, key=lambda item: item.name)


def _checkpoints(round_root, selected_blob_sha, work):
    if not final._is_sha(selected_blob_sha):
        raise ValueError('selected checkpoint blob SHA is unknown')
    outputs = round_root / 'outputs'
    if outputs.is_symlink() or not outputs.is_dir():
        raise ValueError('checkpoint output root cannot be fully enumerated')
    matches = []
    for job in _subdirs(outputs):
        for run in _subdirs(job):
            folder = run / 'checkpoints'
            if folder.is_symlink():
                raise ValueError('checkpoint directory is a symlink')
            if not folder.exists():
                continue
            if not folder.is_dir():
                raise ValueError('checkpoint path is not a directory')
            with os.scandir(folder) as entries:
                files = sorted(entries, key=lambda item: item.name)
            for entry in files:
                if not re.fullmatch(r'checkpoint-.*\.pt', entry.name):
                    continue
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError('checkpoint path is not a regular file')
                path = Path(entry.path)
                side = _reference(Path(str(path) + '.json'), work)
                record = _read(side, work)
                if not final._is_sha(record.get('sha256')):
                    raise ValueError('checkpoint sidecar does not identify a blob')
                if record['sha256'] == selected_blob_sha:
                    matches.append({'path': str(path), 'sidecar': side, 'bytes': entry.stat(follow_symlinks=False).st_size})
    return matches


def _fold_row(root, number, base_ref, base, reference_value, work, allow_fixture):
    if root.name != ROUNDS[number] or root.is_symlink() or not root.is_dir():
        raise ValueError('registered fold round identity differs')
    done = _read(_reference(root / 'control/done.json', work), work)
    if done.get('status') != 'completed':
        raise ValueError('fold has not reached its completed terminal')
    freeze_ref = _reference(root / 'SINGLE-MODEL-FREEZE.json', work)
    completion_ref = _reference(root / 'evidence/TRAIN_RESULT.json', work)
    science_ref = _reference(root / 'SCIENCE.json', work)
    site_ref = _reference(root / 'SITE_BINDING.json', work)
    runtime_ref = _reference(root / 'RUNTIME_BINDING.json', work)
    freeze, science, site = (_read(ref, work) for ref in (freeze_ref, science_ref, site_ref))
    train_ref = _reference(Path(site['metadata_root']) / 'manifest.json', work)
    spec = {'id': root.name, 'freeze': freeze_ref, 'completion': completion_ref}
    row = {'round_id': root.name, 'spec': spec, 'science': science_ref,
           'fold_binding': {'fold': number, 'site': site_ref,
                            'runtime_binding': runtime_ref, 'training_manifest': train_ref}}
    # This independently checks both SCIENCE copies, runtime/site, original and
    # derived files, sealed roles and the original dev rows before any score read.
    population = final.validate_candidate_population(row, base_ref, work_root=work,
                                                      allow_fixture=allow_fixture, _verified_base=base)
    if population.get('kind') != 'threefold' or population.get('fold') != number:
        raise ValueError('registered fold population did not verify')
    fold_reference = {**reference_value, 'manifest_sha256': population['training_manifest_sha256']}
    _, selected, _ = paired_evidence._selected(spec, fold_reference)
    blob = freeze.get('blob_sha256')
    checkpoints = _checkpoints(root, blob, work)
    status = ('prediction_verified_weight_pending' if len(checkpoints) == 1 else
              'selected_checkpoint_missing' if not checkpoints else 'selected_checkpoint_ambiguous')
    row.update(status=status, selected=selected, public_source=science.get('public_source'),
               checkpoint_candidates=checkpoints, selected_blob_sha256=blob)
    return row


def _other_row(root, base_ref, registry, work):
    done_path = root / 'control/done.json'
    if not done_path.exists():
        # Running or unrelated round directories are not complete candidates. A
        # partially frozen round is retained as unknown rather than discarded.
        if (root / 'SINGLE-MODEL-FREEZE.json').exists() or (root / 'evidence/TRAIN_RESULT.json').exists():
            return _unknown(root.name, 'round_has_partial_completion_evidence')
        return None
    done = _read(_reference(done_path, work), work)
    if done.get('status') != 'completed':
        return None
    freeze_ref = _reference(root / 'SINGLE-MODEL-FREEZE.json', work)
    completion_ref = _reference(root / 'evidence/TRAIN_RESULT.json', work)
    freeze, completion = _read(freeze_ref, work), _read(completion_ref, work)
    manifest_sha = freeze['experiment91']['manifest_sha256']
    if completion.get('status') != 'completed' or not final._is_sha(manifest_sha):
        raise ValueError('completed round freeze or declared manifest SHA is unknown')
    if manifest_sha == base_ref['sha256']:
        return _unknown(root.name, 'original_manifest_round_missing_from_F03')
    if registry is None or manifest_sha in registry:
        return _unknown(root.name, 'registered_fold_manifest_identity_unknown_or_misbound')
    return {'round_id': root.name, 'status': 'manifest_mismatch_verified',
            'reason': 'freeze_declared_training_manifest_mismatch',
            'spec': {'id': root.name, 'freeze': freeze_ref, 'completion': completion_ref}}


def discover(f03_ref, *, work_root, allow_fixture=False):
    """Read a pinned F03 census and add exactly one row per registered fold."""
    work = Path(work_root).resolve(strict=True)
    original, base, reference_value = _f03(f03_ref, work, allow_fixture)
    rounds = work / 'rounds'
    if rounds.is_symlink() or not rounds.is_dir():
        raise ValueError('round directory cannot be completely enumerated')
    roots = _subdirs(rounds)
    seen = {row['round_id'] for row in original['members']}
    additions, registry = [], []
    for number, expected in enumerate(ROUNDS):
        if expected in seen:
            # Preserve the original F03 unknown byte-for-byte; do not silently
            # promote or duplicate it during augmentation.
            continue
        matching = [root for root in roots if _fold_number(root.name) == number]
        if len(matching) != 1 or matching[0].name != expected:
            additions.append(_unknown(expected, 'registered_fold_missing_or_ambiguous',
                                      observed_rounds=[root.name for root in matching]))
            continue
        try:
            row = _fold_row(matching[0], number, original['manifest'], base, reference_value, work, allow_fixture)
            additions.append(row)
            if row['status'] in ('prediction_verified_weight_pending', 'selected_checkpoint_missing'):
                registry.append(row['fold_binding']['training_manifest']['sha256'])
        except Exception as error:
            additions.append(_unknown(expected, 'registered_fold_unverified', error_type=type(error).__name__,
                                      error=str(error)))
    registry = set(registry) if len(registry) == 3 and len(set(registry)) == 3 else None
    for root in roots:
        if root.name in seen or _fold_number(root.name) is not None:
            continue
        try:
            row = _other_row(root, original['manifest'], registry, work)
        except Exception as error:
            row = _unknown(root.name, 'completed_round_unverified', error_type=type(error).__name__, error=str(error))
        if row is not None:
            additions.append(row)
    members = copy.deepcopy(original['members']) + additions
    names = [row['round_id'] for row in members]
    if len(names) != len(set(names)):
        raise ValueError('augmented census has duplicate identities')
    result = copy.deepcopy(original)
    result.update(artifact_kind=KIND, augmented=True, source_f03_inventory=copy.deepcopy(f03_ref), members=members,
                  additional_forward_calls=0, official_read=False, GPU_initialized=False)
    return result


def verify_augmented(reference, *, work_root, allow_fixture=False):
    """Verify both artifacts, original membership and exact registered fold census."""
    work = Path(work_root).resolve(strict=True)
    result = _read(reference, work)
    if (result.get('status') != STATUS or result.get('artifact_kind') != KIND
            or result.get('augmented') is not True or result.get('official_read') is not False
            or result.get('additional_forward_calls') != 0 or result.get('GPU_initialized') is not False
            or (allow_fixture and result.get('fixture_only') is not True)
            or (not allow_fixture and result.get('fixture_only') is True)):
        raise ValueError('explicit sealed augmented F05 census required')
    source = result.get('source_f03_inventory')
    if not isinstance(source, dict) or source.get('path') == reference['path']:
        raise ValueError('independent original F03 artifact required')
    original, _, _ = _f03(source, work, allow_fixture)
    members = result.get('members')
    if (any(result.get(key) != value for key, value in original.items() if key != 'members')
            or not isinstance(members, list)
            or members[:len(original['members'])] != original['members']):
        raise ValueError('original F03 manifest or membership was altered')
    names = [row.get('round_id') if isinstance(row, dict) else None for row in members]
    if (any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names))
            or any(sum(_fold_number(name) == number for name in names) != 1 for number in range(3))):
        raise ValueError('three registered fold identities must occur exactly once')
    for number, expected in enumerate(ROUNDS):
        if not any(row['round_id'] == expected for row in members):
            raise ValueError('registered fold row identity differs')
    # Re-scan the same source before consumption so a newly discovered
    # unknown, ambiguous checkpoint, or changed round cannot be dropped from
    # a hand-edited census while retaining the original F03 SHA.
    if result != discover(source, work_root=work, allow_fixture=allow_fixture):
        raise ValueError('augmented discovery no longer matches the complete source census')
    return result

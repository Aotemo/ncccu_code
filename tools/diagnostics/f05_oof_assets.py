"""Derive versioned, fit-only metadata for the sealed F05 OOF folds."""
import copy
import hashlib
import json


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def _record(name, raw):
    return {'path':name, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}


def _full_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def construct_assets(fold, base_contract, schedule, *, base_manifest_sha256,
                     sealed_folds_sha256, base_freeze_sha256):
    """Build fold metadata after the caller has verified its sealed source roles."""
    from f05_oof import training_manifest
    from retrain.age import make_spec
    from retrain.persistence import canonical_digest

    if (not isinstance(fold, dict) or not isinstance(base_contract, dict)
            or not isinstance(base_contract.get('source_identity'), dict)
            or base_contract['source_identity'] != fold.get('source_identity')
            or not isinstance(schedule, bytes)
            or not all(_full_sha(value) for value in
                       (base_manifest_sha256, sealed_folds_sha256, base_freeze_sha256))):
        raise ValueError('explicit original source, schedule and sealed SHA identities required')
    try:
        schedule_value = json.loads(schedule)
    except (TypeError, ValueError) as error:
        raise ValueError('original schedule JSON required') from error
    if not isinstance(schedule_value, dict) or not schedule_value:
        raise ValueError('original schedule must be a nonempty object')
    train = training_manifest(fold)
    rows = fold['rows']
    held_groups = {rows[i]['group'] for i in fold['audit_indices']}
    dev_groups = {rows[i]['group'] for i in fold['dev_indices']}
    if held_groups & dev_groups:
        raise ValueError('held-out and dev groups overlap')
    age = make_spec(train)
    split = fold.get('split', {})
    if (split.get('kind') != 'fit_only_3fold' or split.get('folds') != 3
            or type(split.get('fold')) is not int or not 0 <= split['fold'] < 3
            or split.get('seed') != 20261121):
        raise ValueError('registered three-fold split identity required')
    files = {'manifest.json':_encoded(train), 'AGE.json':_encoded(age), 'SCHEDULE.json':schedule}
    contract = {'schema_version':1, 'status':'FROZEN_FIT_ONLY_FOLD_METADATA',
                'source_identity':copy.deepcopy(base_contract['source_identity']),
                'new_manifest':_record('manifest.json', files['manifest.json']),
                'new_manifest_canonical_sha256':canonical_digest(train),
                'fit_indices_canonical_sha256':canonical_digest(train['fit_indices']),
                'dev_indices_canonical_sha256':canonical_digest(train['dev_indices']),
                'age':_record('AGE.json', files['AGE.json']),
                'age_contract_sha256':age['contract_sha256'],
                'schedule':_record('SCHEDULE.json', schedule),
                'f05_fold':{'fold':split['fold'], 'folds':3, 'seed':split['seed'],
                            'base_manifest_sha256':base_manifest_sha256,
                            'base_freeze_sha256':base_freeze_sha256,
                            'sealed_folds_sha256':sealed_folds_sha256,
                            'heldout_indices_sha256':canonical_digest(fold['audit_indices']),
                            'fit_indices_sha256':canonical_digest(train['fit_indices']),
                            'dev_indices_sha256':canonical_digest(train['dev_indices'])},
                'official_read':False}
    files['CONTRACT.json'] = _encoded(contract)
    freeze = {'schema_version':1, 'status':'SEALED_FIT_ONLY_FOLD',
              'files':{name:_record(name, raw) for name, raw in sorted(files.items())}}
    files['FREEZE.json'] = _encoded(freeze)
    return files

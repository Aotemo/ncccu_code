"""Validate fit-only F05 fold metadata without exposing held-out rows to training."""
import copy
import hashlib
import json
from pathlib import Path

from .age import make_spec
from .persistence import canonical_digest

ORIGINAL_FREEZE_SHA = '90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6'
ORIGINAL_MANIFEST_SHA = '7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377'
THREE_FOLD_SEAL_SHA = 'eb880273832b81be61830a069d5ae9ba8dc2e52be4af2f911f7a6b1ec9e7f6d8'
F05_FOLD_PROFILES = tuple(
    'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-'
    f'seed20261029-scale085-pairsigmoid-f05fold{number}-v1' for number in range(3))


def validate_fold_configuration(site, science):
    """Require an explicit, original-data-bound policy for registered OOF rounds."""
    profile = science.get('candidate_profile')
    if profile not in F05_FOLD_PROFILES:
        if 'f05_fold' in science:
            raise ValueError('fold policy is forbidden for other profiles')
        return None
    number = F05_FOLD_PROFILES.index(profile)
    label = 'r' + str(265 + number)
    expected_round = f'{label}-small-f05fold{number}-seed20261029-pairsigmoid-24k-p4-9010-20261105'
    policy = science.get('f05_fold')
    keys = {'version', 'fold', 'folds', 'split_seed', 'base_metadata_root', 'base_freeze_sha256',
            'base_manifest_sha256', 'sealed_folds_path', 'sealed_folds_sha256', 'derived_freeze_sha256'}
    if (not isinstance(policy, dict) or set(policy) != keys or site.get('round_id') != expected_round
            or policy['version'] != 'fit-only-three-fold-v1' or type(policy['fold']) is not int
            or policy['fold'] != number or policy['folds'] != 3 or policy['split_seed'] != 20261121
            or policy['base_freeze_sha256'] != ORIGINAL_FREEZE_SHA
            or policy['base_manifest_sha256'] != ORIGINAL_MANIFEST_SHA
            or policy['sealed_folds_sha256'] != THREE_FOLD_SEAL_SHA):
        raise ValueError('registered fold/round/source policy differs')
    for key in ('base_metadata_root', 'sealed_folds_path'):
        if not isinstance(policy[key], str) or not policy[key]:
            raise ValueError('relative fold source paths required')
        path = Path(policy[key])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('fold source path escapes work root')
    derived_sha = policy['derived_freeze_sha256']
    if (not isinstance(derived_sha, str) or len(derived_sha) != 64
            or any(c not in '0123456789abcdef' for c in derived_sha)):
        raise ValueError('derived freeze SHA required')
    return number


def _bound_work_path(work_root, relative, *, technical):
    if not isinstance(relative, str) or not relative:
        raise ValueError('relative work path required')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or (technical and path.parts[0] != 'technical-prep'):
        raise ValueError('fold path escapes registered work location')
    result = work_root
    for part in path.parts:
        result = result / part
        if result.is_symlink():
            raise ValueError('fold source path cannot traverse a symlink')
    if not result.exists():
        raise ValueError('fold source path does not exist')
    return result


def verify_bound_fold(site, science, base_metadata_root=None):
    """Check site, migration and immutable source identities before consuming a fold."""
    try:
        number = validate_fold_configuration(site, science)
        if number is None:
            raise ValueError('registered fold profile required')
        work = Path(site['work_root']).resolve(strict=True)
        policy = science['f05_fold']
        base = _bound_work_path(work, policy['base_metadata_root'], technical=False)
        fold = _bound_work_path(work, science['paths']['metadata_root'], technical=True)
        sealed = _bound_work_path(work, policy['sealed_folds_path'], technical=True)
        if (Path(site['metadata_root']) != fold
                or (base_metadata_root is not None and Path(base_metadata_root) != base)):
            raise ValueError('site or migration metadata root differs')
        return verify_assets(fold, base, sealed, fold_number=number,
                             base_freeze_sha256=policy['base_freeze_sha256'],
                             base_manifest_sha256=policy['base_manifest_sha256'],
                             sealed_folds_sha256=policy['sealed_folds_sha256'],
                             derived_freeze_sha256=policy['derived_freeze_sha256'])
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('complete original migration and site-bound fold required') from error


def _read_verified(path, expected_sha):
    path = Path(path)
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha)
            or path.is_symlink() or not path.is_file()):
        raise ValueError('regular file and full expected SHA required')
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError('cannot read bound fold metadata') from error
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('bound fold metadata SHA differs: ' + path.name)
    return raw


def verify_assets(fold_root, base_root, sealed_path, *, fold_number,
                  base_freeze_sha256, base_manifest_sha256,
                  sealed_folds_sha256, derived_freeze_sha256):
    """Bind a training fold to the original freeze and the independent fit-only seal."""
    try:
        base_root, fold_root = Path(base_root), Path(fold_root)
        if (base_root.is_symlink() or fold_root.is_symlink()
                or not base_root.is_dir() or not fold_root.is_dir()
                or type(fold_number) is not int or not 0 <= fold_number < 3):
            raise ValueError('explicit original and fold directories are required')
        base_freeze = json.loads(_read_verified(base_root / 'FREEZE.json', base_freeze_sha256))
        derived_freeze = json.loads(_read_verified(fold_root / 'FREEZE.json', derived_freeze_sha256))
        seal = json.loads(_read_verified(sealed_path, sealed_folds_sha256))
        names = ('manifest.json', 'CONTRACT.json', 'AGE.json', 'SCHEDULE.json')
        if (derived_freeze.get('status') != 'SEALED_FIT_ONLY_FOLD'
                or set(derived_freeze['files']) != set(names)
                or seal.get('status') != 'three_fold_fit_only_indices_sealed_no_training'
                or seal.get('seed') != 20261121 or seal.get('base_manifest_sha256') != base_manifest_sha256
                or seal.get('GPU_initialized') is not False or seal.get('training_started') is not False
                or seal.get('official_read') is not False or len(seal['folds']) != 3):
            raise ValueError('three-fold freeze or seal identity differs')
        base_raw, fold_raw = {}, {}
        for name in names:
            base_raw[name] = _read_verified(base_root / name, base_freeze['files'][name]['sha256'])
            record = derived_freeze['files'][name]
            fold_raw[name] = _read_verified(fold_root / name, record['sha256'])
            if record.get('path') != name or record.get('bytes') != len(fold_raw[name]):
                raise ValueError('derived file identity differs')
        if (hashlib.sha256(base_raw['manifest.json']).hexdigest() != base_manifest_sha256
                or base_raw['SCHEDULE.json'] != fold_raw['SCHEDULE.json']):
            raise ValueError('base manifest or original schedule differs')
        base = json.loads(base_raw['manifest.json'])
        train = json.loads(fold_raw['manifest.json'])
        base_contract = json.loads(base_raw['CONTRACT.json'])
        contract = json.loads(fold_raw['CONTRACT.json'])
        sealed = seal['folds'][fold_number]
        roles = verify_roles(base, train, sealed, fold_number=fold_number, split_seed=20261121)
        fold_identity = {'fold':fold_number, 'folds':3, 'seed':20261121,
                         'base_manifest_sha256':base_manifest_sha256,
                         'base_freeze_sha256':base_freeze_sha256,
                         'sealed_folds_sha256':sealed_folds_sha256,
                         'heldout_indices_sha256':roles['heldout_indices_sha256'],
                         'fit_indices_sha256':roles['fit_indices_sha256'],
                         'dev_indices_sha256':roles['dev_indices_sha256']}
        age = make_spec(train)
        if (json.loads(fold_raw['AGE.json']) != age
                or contract.get('status') != 'FROZEN_FIT_ONLY_FOLD_METADATA'
                or contract.get('official_read') is not False
                or contract.get('source_identity') != base_contract.get('source_identity')
                or contract.get('source_identity') != base['source_identity']
                or contract.get('new_manifest_canonical_sha256') != canonical_digest(train)
                or contract.get('fit_indices_canonical_sha256') != roles['fit_indices_sha256']
                or contract.get('dev_indices_canonical_sha256') != roles['dev_indices_sha256']
                or contract.get('age_contract_sha256') != age['contract_sha256']
                or contract.get('f05_fold') != fold_identity):
            raise ValueError('derived contract or fit-only age differs')
        for key, name in (('new_manifest', 'manifest.json'), ('age', 'AGE.json'),
                          ('schedule', 'SCHEDULE.json')):
            if contract.get(key) != derived_freeze['files'][name]:
                raise ValueError('derived contract file reference differs')
        return dict(roles, fold_number=fold_number, derived_freeze_sha256=derived_freeze_sha256,
                    base_freeze_sha256=base_freeze_sha256, sealed_folds_sha256=sealed_folds_sha256)
    except (KeyError, TypeError, IndexError, json.JSONDecodeError, OSError) as error:
        raise ValueError('complete bound three-fold metadata required') from error


def verify_roles(base, train, sealed, *, fold_number, split_seed):
    """Reconstruct the sealed fold and reject any change to original data roles."""
    try:
        if (type(fold_number) is not int or not 0 <= fold_number < 3
                or split_seed != 20261121 or not isinstance(base, dict)
                or not isinstance(train, dict) or not isinstance(sealed, dict)
                or sealed['fold'] != fold_number or sealed['round'] != 'R' + str(265 + fold_number)
                or train['rows'] != base['rows'] or train['source_identity'] != base['source_identity']
                or train['dev_indices'] != base['dev_indices'] or train['audit_indices'] != []
                or train['fit_indices'] != sealed['fit_indices']
                or train['split'] != {'kind':'fit_only_3fold', 'seed':split_seed,
                                      'fold':fold_number, 'folds':3, 'base_split':base.get('split')}
                or train['split_method'] != 'content-group-label-age-size-v1'):
            raise ValueError('fold identity, source rows or dev role differs')
        fit, held, dev = train['fit_indices'], sealed['audit_indices'], train['dev_indices']
        if (not all(isinstance(role, list) and role for role in (fit, held, dev))
                or any(type(i) is not int or not 0 <= i < len(base['rows']) for role in (fit, held, dev) for i in role)
                or any(len(set(role)) != len(role) for role in (fit, held, dev))
                or set(fit) & set(held) or set(fit) & set(dev) or set(held) & set(dev)
                or set(fit) | set(held) != set(base['fit_indices'])
                or set(base['fit_indices']) & set(base['dev_indices'])
                or set(base['fit_indices']) | set(base['dev_indices']) != set(range(len(base['rows'])))):
            raise ValueError('fold roles do not partition original fit and fixed dev')
        groups = [{base['rows'][i]['group'] for i in role} for role in (fit, held, dev)]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('content group crosses fold roles')
        if (sealed['fit_indices_sha256'] != canonical_digest(fit)
                or sealed['audit_indices_sha256'] != canonical_digest(held)
                or sealed['dev_indices_sha256'] != canonical_digest(dev)
                or sealed['age'] != make_spec(train)):
            raise ValueError('sealed indices or fit-only age differs')
        reconstructed = copy.deepcopy(train)
        reconstructed['audit_indices'] = list(held)
        reconstructed['counts']['audit'] = copy.deepcopy(sealed['counts']['audit'])
        if (reconstructed['counts'] != sealed['counts']
                or canonical_digest(reconstructed) != sealed['manifest_canonical_sha256']):
            raise ValueError('fold manifest differs from presealed metadata')
        return {'fit_count':len(fit), 'heldout_count':len(held), 'dev_count':len(dev),
                'fit_indices_sha256':sealed['fit_indices_sha256'],
                'heldout_indices_sha256':sealed['audit_indices_sha256'],
                'dev_indices_sha256':sealed['dev_indices_sha256'],
                'age_contract_sha256':sealed['age']['contract_sha256']}
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('complete sealed fit-only fold metadata required') from error

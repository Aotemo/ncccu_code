"""Derive versioned, fit-only metadata for the sealed F05 OOF folds."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def _record(name, raw):
    return {'path':name, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}


def _full_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_registered_policy(config, manifest_sha, sealed_sha):
    if (not isinstance(config, dict) or not _full_sha(manifest_sha) or not _full_sha(sealed_sha)
            or config.get('base_manifest_sha256') != manifest_sha
            or config.get('sealed_folds_sha256') != sealed_sha or config.get('folds') != 3
            or config.get('fold_seed') != 20261121 or config.get('official_read') is not False):
        raise ValueError('registered base manifest or three-fold policy differs')
    return True


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


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('site', 'sealed-folds', 'tree-proof', 'remote-metadata', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('site-sha256', 'sealed-folds-sha256', 'tree-proof-sha256', 'git-commit'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(key) != '' for key in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))):
        raise ValueError('guarded hidden-device Pod CPU required')
    source = Path(__file__).resolve().parents[2]
    for path, expected in ((args.site, args.site_sha256), (args.sealed_folds, args.sealed_folds_sha256),
                           (args.tree_proof, args.tree_proof_sha256)):
        if path.is_symlink() or not _full_sha(expected) or _file_sha(path) != expected:
            raise ValueError('site, sealed fold or source proof SHA differs')
    site, seal, proof = (json.loads(path.read_text()) for path in
                         (args.site, args.sealed_folds, args.tree_proof))
    work = Path(site['work_root']).resolve(strict=True)
    if (site['pod']['uid'] != os.environ['NCCCU_POD_UID']
            or not args.output.is_absolute() or args.output.exists()
            or not args.output.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')
            or not args.sealed_folds.resolve(strict=True).is_relative_to(work / 'technical-prep')):
        raise ValueError('current Pod or fresh technical-prep identity differs')
    ref, commit, tree = (json.loads((args.remote_metadata / name).read_text()) for name in
                         ('REMOTE-REF.json', 'REMOTE-COMMIT.json', 'REMOTE-TREE.json'))
    if (proof.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
            or proof.get('source_root') != str(source) or proof.get('pod_uid') != site['pod']['uid']
            or proof.get('commit') != args.git_commit or ref.get('object', {}).get('sha') != args.git_commit
            or commit.get('sha') != args.git_commit or proof.get('tree') != tree.get('sha')
            or commit.get('tree', {}).get('sha') != tree.get('sha') or tree.get('truncated') is not False):
        raise ValueError('published Pod-local source identity differs')
    blobs = {entry['path']:entry for entry in tree['tree'] if entry.get('type') == 'blob'}
    names = ('tools/diagnostics/f05_oof.py', 'tools/diagnostics/f05_oof_assets.py',
             'configs/F05-FOLDS-3.json', 'runtime/bundle/retrain/data.py',
             'runtime/bundle/retrain/age.py', 'runtime/bundle/retrain/persistence.py')
    for name in names:
        path = source / name; raw = path.read_bytes()
        git_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        if path.is_symlink() or name not in blobs or blobs[name]['sha'] != git_blob or blobs[name]['mode'] != mode:
            raise ValueError('fold implementation differs from published Git: ' + name)
    sys.path[:0] = [str(source / 'tools/diagnostics'), str(source / 'runtime/bundle')]
    from f05_oof import build_fold_manifests, verify_sealed_folds
    from retrain.experiment91 import FREEZE_SHA
    config = json.loads((source / 'configs/F05-FOLDS-3.json').read_text())
    base_root = Path(site['metadata_root'])
    freeze_path = base_root / 'FREEZE.json'
    if freeze_path.is_symlink() or _file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError('original 9:1 metadata freeze differs')
    freeze = json.loads(freeze_path.read_text())
    for name in ('manifest.json', 'CONTRACT.json', 'AGE.json', 'SCHEDULE.json'):
        path = base_root / name
        if path.is_symlink() or _file_sha(path) != freeze['files'][name]['sha256']:
            raise ValueError('original frozen metadata differs: ' + name)
    manifest_path = base_root / 'manifest.json'
    validate_registered_policy(config, _file_sha(manifest_path), args.sealed_folds_sha256)
    base = json.loads(manifest_path.read_text())
    if (len(base['fit_indices']) != config['fit_count'] or len(base['dev_indices']) != config['dev_count']):
        raise ValueError('original fit/dev counts differ')
    folds = build_fold_manifests(base, config['fold_seed'])
    verify_sealed_folds(folds, seal, config['fold_seed'])
    base_contract = json.loads((base_root / 'CONTRACT.json').read_text())
    schedule = (base_root / 'SCHEDULE.json').read_bytes()
    all_files = [construct_assets(fold, base_contract, schedule,
                 base_manifest_sha256=config['base_manifest_sha256'],
                 sealed_folds_sha256=args.sealed_folds_sha256, base_freeze_sha256=FREEZE_SHA)
                 for fold in folds]
    args.output.mkdir(exist_ok=False)
    summaries = []
    for number, files in enumerate(all_files):
        folder = args.output / ('fold-' + str(number)); folder.mkdir()
        for name, raw in files.items():
            with (folder / name).open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        summaries.append({'fold':number, 'round':config['rounds'][number],
                          'counts':folds[number]['counts'],
                          'files':{name:{'path':str(folder / name), 'sha256':_file_sha(folder / name)}
                                   for name in sorted(files)}})
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        raise ValueError('GPU unexpectedly initialized')
    result = {'status':'three_fold_fit_only_metadata_derived_before_training', 'at':time.time(),
              'git_commit':args.git_commit, 'site_sha256':args.site_sha256,
              'base_freeze_sha256':FREEZE_SHA, 'base_manifest_sha256':config['base_manifest_sha256'],
              'sealed_folds_sha256':args.sealed_folds_sha256, 'folds':summaries,
              'GPU_initialized':False, 'training_started':False, 'official_read':False}
    result_path = args.output / 'DERIVATION.json'
    with result_path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'status':result['status'], 'folds':summaries,
                      'receipt_sha256':_file_sha(result_path)}), flush=True)


if __name__ == '__main__':
    main()

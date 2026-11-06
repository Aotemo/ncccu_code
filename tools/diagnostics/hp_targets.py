"""HP hard-positive fit row weights from the pinned F05 three-fold OOF (CPU only).

Every artifact is written with exclusive creation; nothing is overwritten.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False).encode() + b'\n'
    with path.open('xb') as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def load_verifier(source_root):
    verifier_path = Path(source_root) / 'tools' / 'diagnostics' / 'f05_oof_targets.py'
    if verifier_path.is_symlink() or not verifier_path.is_file():
        raise SystemExit('published F05 OOF verifier is missing')
    specification = importlib.util.spec_from_file_location('_hp_f05_verifier', verifier_path)
    verifier = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(verifier)
    return verifier


def cmd_build(args):
    from retrain.data import build_f05_hardpos_targets
    verifier = load_verifier(args.source_root)
    summary_ref = {'path': str(Path(args.summary).resolve()), 'sha256': args.summary_sha256}
    if sha(summary_ref['path']) != args.summary_sha256:
        raise SystemExit('pinned summary bytes differ')
    result = build_f05_hardpos_targets(summary_ref, args.work_root, verifier=verifier)
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit('HP target output directory already exists')
    output.mkdir(parents=True)
    table_path = output / 'HARDPOS-TARGETS-private.json'
    table_sha = write_new(table_path, result['table'])
    extended = dict(result['summary'])
    extended['hardpos_targets'] = {'path': str(table_path), 'sha256': table_sha}
    extended['hardpos_stats'] = result['hardpos_stats']
    extended['hardpos_source_summary'] = summary_ref
    summary_path = output / 'HP-SUMMARY.json'
    hp_summary_sha = write_new(summary_path, extended)
    receipt = {'status': 'hp_hardpos_targets_built_not_trained',
               'rule': result['hardpos_stats'],
               'summary': summary_ref,
               'hp_summary': {'path': str(summary_path), 'sha256': hp_summary_sha},
               'hardpos_targets': {'path': str(table_path), 'sha256': table_sha},
               'fold_sources': result['fold_sources'],
               'base': result['base'], 'folds': result['folds'],
               'manifest_digest': result['manifest_digest'],
               'fit_count': result['fit_count'], 'dev_count': result['dev_count'],
               'weights': {'hard': 3.0, 'other': 1.0},
               'GPU_initialized': False, 'official_read': False, 'dev_read': False,
               'test_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(output / 'HP-TARGETS-RECEIPT.json', receipt)}, indent=1))


def cmd_verify(args):
    from retrain.data import verify_f05_oof_for_training
    verifier = load_verifier(args.source_root)
    summary_ref = {'path': str(Path(args.summary).resolve()), 'sha256': args.summary_sha256}
    if sha(summary_ref['path']) != args.summary_sha256:
        raise SystemExit('HP summary bytes differ')
    proof = verify_f05_oof_for_training(summary_ref, args.work_root, verifier=verifier)
    if proof.get('targets', {}).get('hardpos') is None:
        raise SystemExit('verified proof lacks the hardpos target reference')
    proof_bytes = json.dumps(proof, indent=1, sort_keys=True, allow_nan=False).encode() + b'\n'
    receipt = {'status': 'hp_hardpos_targets_verified_not_launched',
               'summary': summary_ref,
               'proof_sha256': hashlib.sha256(proof_bytes).hexdigest(),
               'hardpos_stats': json.loads(Path(summary_ref['path']).read_text())['hardpos_stats'],
               'official_read': False, 'dev_read': False, 'test_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(args.receipt, receipt),
                                'evidence_sha256': write_new(args.evidence, proof)}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--summary', required=True)
    build.add_argument('--summary-sha256', required=True)
    build.add_argument('--work-root', required=True)
    build.add_argument('--source-root', required=True)
    build.add_argument('--output-dir', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--summary', required=True)
    verify.add_argument('--summary-sha256', required=True)
    verify.add_argument('--work-root', required=True)
    verify.add_argument('--source-root', required=True)
    verify.add_argument('--evidence', required=True)
    verify.add_argument('--receipt', required=True)
    args = parser.parse_args()
    {'build': cmd_build, 'verify': cmd_verify}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Call one approved explicit activation and reconcile its actual receipt on the Pod."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Importing this module is read-only. The actual activation is guarded by main().
sys.path.insert(0, str(Path(__file__).resolve().parent))
from activate import parser, read, sha, write, require, ensure_new_activation, pinned


def main():
    argument_parser = parser()
    argument_parser.description = __doc__
    argument_parser.add_argument('--audit-root', type=Path, required=True)
    args = argument_parser.parse_args()
    prep = args.prep_root.resolve(strict=True); package = prep / 'package'
    sys.path.insert(0, str(package / 'bundle'))
    from retrain.site_binding import load_site, require_cpu_channel, verify_live, evidence_path
    site = load_site(args.site_binding, expected_sha256=args.site_binding_sha256)
    require_cpu_channel(site)
    verify_live(site, pod_uid=args.pod_uid, ready=args.ready == 'True', observed_at=args.observed_at)
    audit = evidence_path(args.audit_root, site['work_root'])
    require(audit.resolve().is_relative_to(Path(site['work_root']) / 'technical-prep'), 'audit must be an isolated preparation output')
    require(site.get('scope') == 'formal_exp_v3', 'technical snapshots cannot activate')
    manifest = pinned(prep / 'PACKAGE-MANIFEST.json', args.package_manifest_sha256, site['work_root'])
    activation = prep / 'activate.py'
    require(sha(activation) == manifest['activate_script_sha256'], 'actual activation script differs')
    ensure_new_activation(prep, site['round_root'])
    audit.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, '-B', str(activation), '--activate-formal']
    for name in ('prep_root', 'authorization_reference', 'site_binding', 'site_binding_sha256',
                 'package_manifest_sha256', 'cpu_prep_receipt_sha256', 'deployment_receipt',
                 'deployment_receipt_sha256', 'gpu_admission', 'gpu_admission_sha256',
                 'site_preflight', 'site_preflight_sha256', 'pod_uid', 'ready', 'observed_at'):
        argv.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    # An attempted call is irreversible even when the child exits before clock creation.
    write(audit / 'ACTIVATION-CALL-STARTED.json', {'at': time.time(), 'argv': argv,
        'authorization_reference': args.authorization_reference, 'site_binding_sha256': args.site_binding_sha256,
        'package_manifest_sha256': args.package_manifest_sha256, 'activation_script_sha256': sha(activation)})
    with (audit / 'ACTIVATION.log').open('x') as stream:
        completed = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=False)
    write(audit / 'ACTIVATION-EXIT.json', {'at': time.time(), 'exit_code': completed.returncode})
    require(completed.returncode == 0, 'activation did not exit0; preserve its same clock/root and do not retry')
    formal = Path(site['round_root'])
    receipt_path = formal / 'evidence/FORMAL_BINDING_RECEIPT.json'
    receipt = read(receipt_path); runtime = read(formal / 'RUNTIME_BINDING.json')
    require(receipt.get('exit_code') == 0 and receipt.get('status') == 'formally_bound_not_dispatched'
            and receipt.get('GPU_started') is False and receipt['root'] == str(formal)
            and receipt['source_digest'] == manifest['bundle_source_digest'] == runtime['source_digest']
            and receipt['clock']['authorization_reference'] == args.authorization_reference
            and receipt['clock'] == read(formal / 'control/round-clock.json') == read(prep / 'FORMAL-CLOCK.json')
            and receipt['git_commit'] == site['git_commit'] == runtime['git_commit']
            and receipt['site_binding_sha256'] == args.site_binding_sha256
            and sha(formal / 'RUNTIME_BINDING.json') == receipt['binding_sha256'], 'actual formal activation receipt differs')
    for name in ('dispatch.json', 'pod-admission.json', 'controller-start.json'):
        path = formal / 'control' / name
        require(not path.exists() and not path.is_symlink(), 'activation unexpectedly dispatched GPU work')
    for relative, digest in runtime['source_files'].items():
        path = evidence_path(formal / relative, formal)
        require(sha(path) == digest, 'actual bound source changed: ' + relative)
    result = {'status': 'activation_exit0_and_actual_binding_verified', 'exit_code': 0,
        'root': str(formal), 'clock': receipt['clock'], 'source_digest': receipt['source_digest'],
        'git_commit': site['git_commit'], 'site_binding_sha256': args.site_binding_sha256,
        'formal_receipt_path': str(receipt_path), 'formal_receipt_sha256': sha(receipt_path),
        'runtime_binding_sha256': sha(formal / 'RUNTIME_BINDING.json'),
        'predecessor_handoff': read(formal / 'control/predecessor-handoff.json'),
        'site_receipt_sha256': args.site_preflight_sha256, 'GPU_started': False}
    write(audit / 'FORMAL-BINDING-AUDIT.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

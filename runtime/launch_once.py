"""One-shot owner spawn from an explicit, verified formal deployment."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

R = Path(__file__).resolve().parent


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def ensure_new_dispatch(root):
    for name in ('dispatch.json', 'controller-start.json', 'pod-admission.json'):
        path = Path(root) / 'control' / name
        if path.exists() or path.is_symlink():
            raise FileExistsError('one-shot dispatch already reserved: ' + str(path))


def load_fit_context(path, expected_sha256):
    """Validate a separate diagnostic deployment; never load a checkpoint here."""
    from retrain.site_binding import load_site, sha256
    path = Path(path)
    if path != R / 'FIT-CENSUS.json' or path.is_symlink() or sha256(path) != expected_sha256:
        raise ValueError('explicit diagnostic context differs')
    value = json.loads(path.read_text())
    site = load_site(R / 'SITE_BINDING.json', expected_sha256=value['site_binding_sha256'])
    if (value['mode'] != 'fit-census' or value['round_id'] != site['round_id']
            or value['git_commit'] != site['git_commit'] or Path(site['round_root']).resolve() != R
            or not value['authorization_reference'] or not value['token']):
        raise ValueError('diagnostic execution identity differs')
    for relative, digest in value['files'].items():
        source = R / relative
        if source.is_symlink() or not source.resolve().is_relative_to(R) or sha256(source) != digest:
            raise ValueError('diagnostic deployment differs: ' + relative)
    if not {'launch_once.py', 'run_training.py', 'bundle/parallel_evaluate.py', 'run_round.py', 'component_gate.py'} <= value['files'].keys():
        raise ValueError('diagnostic entry pins missing')
    reference = value['freeze']; frozen_path = Path(reference['path'])
    if frozen_path.is_symlink() or sha256(frozen_path) != reference['sha256']:
        raise ValueError('original selected freeze changed')
    frozen = json.loads(frozen_path.read_text()); selected = value['selected']
    checkpoint = Path(selected['checkpoint'])
    if checkpoint.is_symlink() or not checkpoint.is_file() or value['output'] != str(R / 'evidence/FIT-CENSUS-AGGREGATE.json'):
        raise ValueError('diagnostic requires the explicit file and its own aggregate path')
    if (frozen.get('cpu_fixture_only') or frozen['variant'] != selected['variant']
            or frozen['checkpoint'] != selected['checkpoint'] or frozen['blob_sha256'] != selected['blob_sha256']
            or frozen['tensor_sha256'] != selected['tensor_sha256']
            or any(frozen['selected'][key] != selected[key] for key in ('cycle', 'successful_updates', 'variant', 'tensor_sha256'))
            or frozen['selected']['auroc'] != selected['dev_auroc']):
        raise ValueError('diagnostic must use the original unique selector')
    # The existing consumer modules remain byte-identical to the selected run.
    for relative, digest in frozen['source_files'].items():
        if relative.startswith('retrain/') and value['files'].get('bundle/' + relative) != digest:
            raise ValueError('selected-run consumer source differs: ' + relative)
    manifest = value['manifest']
    if Path(manifest['path']).is_symlink() or sha256(manifest['path']) != manifest['sha256']:
        raise ValueError('sealed fit manifest changed')
    if (value['cache'] != str(Path(site['prepared_cache']) / 'train/sealed')
            or value['data_root'] != site['data_root']):
        raise ValueError('diagnostic data/cache source differs')
    value['_frozen'] = frozen
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-binding', type=Path, required=True)
    parser.add_argument('--site-binding-sha256', required=True)
    parser.add_argument('--pod-uid', required=True)
    parser.add_argument('--ready', choices=['True'], required=True)
    parser.add_argument('--observed-at', type=float, required=True)
    parser.add_argument('--mode', choices=('train', 'fit-census'), default='train')
    parser.add_argument('--fit-context', type=Path)
    parser.add_argument('--fit-context-sha256')
    args = parser.parse_args()
    sys.path.insert(0, str(R / 'bundle'))
    from retrain.site_binding import load_site, verify_live, bind_cpu_role, prepare_native_entry
    site = load_site(args.site_binding, expected_sha256=args.site_binding_sha256)
    verify_live(site, pod_uid=args.pod_uid, ready=args.ready == 'True', observed_at=args.observed_at)
    if args.site_binding.resolve() != R / 'SITE_BINDING.json' or Path(site['round_root']).resolve() != R:
        raise ValueError('launcher is outside the admitted formal round')
    if site.get('scope') != 'formal_exp_v3':
        raise ValueError('launcher requires a formal deployment scope')
    prepare_native_entry(site)
    bind_cpu_role(site, 'owner', initial=True)
    os.environ.update(NCCCU_SITE_BINDING=str(args.site_binding), NCCCU_SITE_BINDING_SHA256=args.site_binding_sha256)
    if args.mode == 'fit-census':
        census = load_fit_context(args.fit_context, args.fit_context_sha256)
        clock = {'token': census['token'], 'deadline_at': None}
    else:
        if args.fit_context is not None or args.fit_context_sha256 is not None:
            raise ValueError('training cannot consume a diagnostic context')
        from run_round import verify_bound
        verify_bound()
        clock = json.loads((R / 'control/round-clock.json').read_text())
        if (not clock.get('token') or clock['round_id'] != site['round_id']
                or time.time() >= clock['implementation_deadline_at']):
            raise ValueError('formal clock is absent, expired or belongs to another round')
    ensure_new_dispatch(R)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        if env.get(key) != ','.join(str(rank) for rank in range(len(site['gpus']))):
            raise RuntimeError('GPU entry requires explicit ordered device visibility')
    # Exclusive admission is consumed before spawning; a failed spawn cannot be retried.
    write(R / 'control/pod-admission.json', {'pod_uid': args.pod_uid, 'ready': True,
        'observed_at': args.observed_at, 'hostname': socket.gethostname(), 'python': sys.executable,
        'version': sys.version, 'site_binding_sha256': args.site_binding_sha256,
        'owner_cpu_mask': sorted(os.sched_getaffinity(0)),
        'launcher_nice': os.getpriority(os.PRIO_PROCESS, 0)})
    argv = [sys.executable, '-B', '-u', str(R / 'run_training.py'), '--token', clock['token']]
    if args.mode == 'fit-census':
        argv += ['--mode', args.mode, '--fit-context', str(args.fit_context), '--fit-context-sha256', args.fit_context_sha256]
    with (R / 'evidence/controller.log').open('x') as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=R, env=env, start_new_session=True)
    value = {'at': time.time(), 'pid': process.pid, 'argv': argv, 'token': clock['token'],
        'deadline_at': clock['deadline_at'], 'site_binding_sha256': args.site_binding_sha256,
        'meaning': 'spawn only; first update requires every admitted rank initial barrier proof'}
    write(R / 'control/dispatch.json', value)
    print(json.dumps(value if args.mode == 'train' else {'mode': args.mode, 'pid': process.pid,
        'meaning': 'spawn only; the independent census terminal result is required'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

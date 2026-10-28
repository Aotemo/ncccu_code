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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-binding', type=Path, required=True)
    parser.add_argument('--site-binding-sha256', required=True)
    parser.add_argument('--pod-uid', required=True)
    parser.add_argument('--ready', choices=['True'], required=True)
    parser.add_argument('--observed-at', type=float, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(R / 'bundle'))
    from retrain.site_binding import load_site, verify_live, bind_cpu_role
    site = load_site(args.site_binding, expected_sha256=args.site_binding_sha256)
    verify_live(site, pod_uid=args.pod_uid, ready=args.ready == 'True', observed_at=args.observed_at)
    if args.site_binding.resolve() != R / 'SITE_BINDING.json' or Path(site['round_root']).resolve() != R:
        raise ValueError('launcher is outside the admitted formal round')
    if site.get('scope') != 'formal_exp_v3':
        raise ValueError('launcher requires a formal deployment scope')
    bind_cpu_role(site, 'owner', initial=True)
    os.environ.update(NCCCU_SITE_BINDING=str(args.site_binding), NCCCU_SITE_BINDING_SHA256=args.site_binding_sha256)
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
    with (R / 'evidence/controller.log').open('x') as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=R, env=env, start_new_session=True)
    value = {'at': time.time(), 'pid': process.pid, 'argv': argv, 'token': clock['token'],
        'deadline_at': clock['deadline_at'], 'site_binding_sha256': args.site_binding_sha256,
        'meaning': 'spawn only; first update requires every admitted rank initial barrier proof'}
    write(R / 'control/dispatch.json', value)
    print(json.dumps(value, indent=2), flush=True)


if __name__ == '__main__':
    main()

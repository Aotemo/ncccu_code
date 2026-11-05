"""Spawn one F06 training process durably inside the Pod and publish its exact identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True)
    parser.add_argument('--site-sha256', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--control-mask', required=True)
    parser.add_argument('--owner-mask', required=True)
    parser.add_argument('--native-library', required=True)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--verification-cache', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        raise SystemExit('training command required')
    log = Path(args.log)
    receipt = Path(args.receipt)
    if log.exists() or receipt.exists():
        raise SystemExit('fresh log/receipt required')
    environment = os.environ.copy()
    environment.update(NCCCU_SITE_BINDING=args.site, NCCCU_SITE_BINDING_SHA256=args.site_sha256,
                       TRAIN_IMAGE_CACHE=args.train_cache,
                       TRAIN_IMAGE_CACHE_READONLY='1',
                       TRAIN_VERIFICATION_CACHE=args.verification_cache,
                       LD_PRELOAD=args.native_library, NCCCU_NATIVE_CPU_MASK=args.owner_mask,
                       XFORMERS_DISABLED='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('MODE', 'RUN_NAME', 'OUTPUT_ROOT', 'DATA_ROOT', 'RESUME', 'DEVICE_IDS'):
        environment.pop(key, None)
    argv = ['taskset', '-c', args.control_mask, *command]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('xb') as stream:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, env=environment,
                                   start_new_session=True, cwd='/')
    stat = Path('/proc') / str(process.pid) / 'stat'
    fields = stat.read_text().rsplit(')', 1)[1].split()
    value = {'status': 'f06_training_spawned', 'at': time.time(), 'pid': process.pid,
             'start_ticks': fields[19], 'argv': argv, 'log': str(log),
             'site': args.site, 'site_sha256': args.site_sha256,
             'control_mask': args.control_mask, 'owner_mask': args.owner_mask}
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n'
    with receipt.open('x') as stream:
        stream.write(payload)
    print(json.dumps(value | {'receipt_sha256': hashlib.sha256(payload.encode()).hexdigest()}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

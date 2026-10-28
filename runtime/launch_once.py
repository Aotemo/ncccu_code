"""Original one-shot owner spawn; invoke only after formal activation and routing."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12'
assert sys.argv[1:] == ['eec33bd6-b567-40ef-b866-633e272ea019', 'True']
clock = json.loads((R / 'control/round-clock.json').read_text())
assert clock['token'] and time.time() < clock['implementation_deadline_at']


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


assert not (R / 'control/dispatch.json').exists() and not (R / 'control/controller-start.json').exists()
write(R / 'control/pod-admission.json', {'pod_uid': sys.argv[1], 'ready': True, 'observed_at': time.time(),
    'hostname': socket.gethostname(), 'python': sys.executable, 'version': sys.version, 'launcher_nice': os.getpriority(os.PRIO_PROCESS, 0)})
env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
    if env.get(key) == '':
        raise RuntimeError('GPU owner inherited CPU-hidden devices')
argv = [sys.executable, '-B', '-u', str(R / 'run_training.py'), '--token', clock['token']]
with (R / 'evidence/controller.log').open('x') as log:
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=R, env=env, start_new_session=True)
value = {'at': time.time(), 'pid': process.pid, 'argv': argv, 'token': clock['token'], 'deadline_at': clock['deadline_at'],
         'meaning': 'spawn only; first training update requires actual four-rank initial barrier proof'}
write(R / 'control/dispatch.json', value)
print(json.dumps(value, indent=2), flush=True)

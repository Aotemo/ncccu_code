"""Bind existing read-only observer to actual formally bound R92 token; no dispatch."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
F = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R92-TINY1K-9010')
assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
clock = json.loads((F / 'control/round-clock.json').read_text())
receipt = json.loads((HERE / 'FORMAL-BINDING-AUDIT.json').read_text())
assert clock == receipt['clock'] and receipt['exit_code'] == 0
template = (HERE / 'observe_template.py').read_text()
assert template.count('__ACTUAL_R92_TOKEN__') == 1 and clock['token']
source = template.replace('__ACTUAL_R92_TOKEN__', clock['token'])
compile(source, str(HERE / 'observe_run.py'), 'exec')
with (HERE / 'observe_run.py').open('x') as stream:
    stream.write(source)
proof = {'status': 'read_only_observer_bound', 'exit_code': 0, 'token': clock['token'], 'root': str(F),
    'observer': str(HERE / 'observe_run.py'), 'sha256': hashlib.sha256(source.encode()).hexdigest(),
    'template_sha256': hashlib.sha256(template.encode()).hexdigest(), 'model_loaded': False,
    'lock_acquired': False, 'GPU_spawned': False}
with (HERE / 'OBSERVER-DEPLOYMENT.json').open('x') as stream:
    json.dump(proof, stream, indent=2); stream.write('\n')
print(json.dumps(proof), flush=True)

"""Extract small experiment conclusions and official single-model candidates on Pod."""
import json
import os
from pathlib import Path
import re
import socket
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2')
assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
entries = []
failures = []
def inspect(value, location=''):
    if isinstance(value, dict):
        metric = next((value[k] for k in ('auroc', 'auc', 'roc_auc') if isinstance(value.get(k), (int, float))), None)
        if metric is not None:
            yield {'location': location, 'metric': {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool, type(None)))}}
        for key, child in value.items():
            yield from inspect(child, location + '/' + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from inspect(child, location + '/' + str(index))
def blobs(value):
    if isinstance(value, str) and value.endswith(('.pt', '.pth', '.safetensors')):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from blobs(child)
    elif isinstance(value, list):
        for child in value:
            yield from blobs(child)
skip = {'.git', '__pycache__', 'data', 'datasets', 'train', 'valid', 'images', 'site-packages', 'source-tree', 'bundle', 'retrain', 'code'}
for campaign in (ROOT / 'auroc998-20261025', ROOT / 'auroc999-20261027', HERE / 'local-results'):
    for base, dirs, files in os.walk(campaign, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(('prepared', 'CPU-FIXTURE'))]
        for name in files:
            path = Path(base) / name
            if not (name.startswith('RESULT-') and name.endswith('.json') or name in ('RESULT.json', 'VALID_RESULT.json', 'OFFICIAL-RESULT.json')):
                continue
            try:
                if path.stat().st_size > 16 * 1024**2:
                    continue
                value = json.loads(path.read_text())
                metrics = list(inspect(value))
                if metrics:
                    entries.append({'path': str(path), 'top_keys': list(value)[:35], 'metrics': metrics,
                                    'checkpoint_paths': sorted(set(blobs(value)))})
            except Exception as exc:
                failures.append({'path': str(path), 'error': repr(exc)})
result = {'status': 'conclusions_extracted', 'entries': entries, 'failures': failures}
with (HERE / 'EXPERIMENT-CONCLUSIONS-v2.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({'status': result['status'], 'result_files': len(entries), 'failures': failures}), flush=True)
for entry in entries:
    official = [m for m in entry['metrics'] if any(str(v) == '5849' for k, v in m['metric'].items())
                or any(word in m['location'].lower() for word in ('valid', 'official', 'dev5k'))]
    # Report one line per result, with no per-sample data.
    print(json.dumps({'path': entry['path'], 'top_keys': entry['top_keys'], 'official_metrics': official[:8],
                      'checkpoint_paths': entry['checkpoint_paths'][:7]}), flush=True)

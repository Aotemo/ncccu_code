"""Summarize the Pod inventory and inspect checkpoint indexes without mutations."""
import collections
import json
import os
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2')
assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
data = json.loads((HERE / 'STORAGE-INVENTORY.json').read_text())
groups = collections.defaultdict(lambda: {'files': 0, 'bytes': 0})
for item in data['large_files']:
    relative = Path(item['path']).relative_to(ROOT)
    key = '/'.join(relative.parts[:3])
    groups[key]['files'] += 1
    groups[key]['bytes'] += item['bytes']
indexes = []
seen = set()
def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for part in value.values():
            yield from strings(part)
    elif isinstance(value, list):
        for part in value:
            yield from strings(part)
for item in data['large_files']:
    folder = Path(item['path']).parent
    if folder in seen or folder.name not in ('checkpoints', 'initial-checkpoint'):
        continue
    seen.add(folder)
    path = folder / 'index.json'
    if not path.is_file():
        continue
    value = json.loads(path.read_text())
    references = {Path(s).name for s in strings(value) if s.endswith(('.pt', '.pth', '.ckpt'))}
    blobs = [{'path': str(p), 'bytes': p.stat().st_size, 'referenced': p.name in references}
             for p in sorted(folder.glob('*.pt')) if p.is_file()]
    indexes.append({'path': str(path), 'content': value, 'blobs': blobs,
                    'completion_exists': (folder.parent / 'complete.json').is_file()})
result = {'inventory_errors': data['errors'], 'process_scan_denied_count': len(data['process_scan_denied']),
          'large_files': len(data['large_files']), 'large_bytes': sum(x['bytes'] for x in data['large_files']),
          'groups': dict(sorted(groups.items(), key=lambda x: x[1]['bytes'], reverse=True)),
          'indexes': indexes}
with (HERE / 'CHECKPOINT-REVIEW.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != 'indexes'}), flush=True)
for value in indexes:
    refs = {k: v.get('file') if isinstance(v, dict) else v
            for k, v in value['content'].items() if k in ('last', 'previous', 'best')}
    refs['protected'] = {k: v.get('file') for k, v in value['content'].get('protected', {}).items()}
    print(json.dumps({'index': value['path'], 'complete': value['completion_exists'], 'references': refs,
                      'blob_count': len(value['blobs']), 'unreferenced_count': sum(not b['referenced'] for b in value['blobs'])}), flush=True)

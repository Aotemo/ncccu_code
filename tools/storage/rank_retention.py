"""Report full-official metrics and precise freeze schemas for retention."""
import json
import os
from pathlib import Path
import socket
import sys
HERE = Path(__file__).resolve().parent
assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
data = json.loads((HERE / 'EXPERIMENT-CONCLUSIONS-v2.json').read_text())
rows = []
for entry in data['entries']:
    for m in entry['metrics']:
        value = m['metric']; loc = m['location'].lower()
        if value.get('count', value.get('n', value.get('rows'))) != 5849:
            continue
        if any(x in loc for x in ('strata', 'reference', 'compar', 'baseline', 'delta', 'bootstrap')):
            continue
        rows.append({'auroc': value.get('auroc', value.get('auc')), 'ap': value.get('ap'),
                     'file': entry['path'], 'location': m['location']})
unique = {}
for row in rows:
    key = (row['file'], row['auroc'])
    unique.setdefault(key, row)
ranked = sorted(unique.values(), key=lambda x: (-x['auroc'], x['file']))
with (HERE / 'OFFICIAL-RANKING-v2.json').open('x') as stream:
    json.dump(ranked, stream, indent=2)
print(json.dumps({'result_files': len(data['entries']), 'full_official_rows': len(ranked)}), flush=True)
for row in ranked[:25]:
    print(json.dumps(row), flush=True)
for entry in data['entries']:
    if '/RESULT-R' in entry['path'] and any('/r' + str(n) + '/' in entry['path'] for n in (17, 16, 24, 23, 15)):
        original = json.loads(Path(entry['path']).read_text())
        selected = {k: v for k, v in original.items() if any(w in k.lower() for w in ('freeze', 'frozen', 'select', 'checkpoint'))}
        print(json.dumps({'schema_file': entry['path'], 'selected_fields': selected})[:14000], flush=True)

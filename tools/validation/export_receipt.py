"""Package existing R92 small originals on Pod; no model or new inference."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tarfile

F = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R92-TINY1K-9010')
P = Path(__file__).resolve().parent
R = P / 'result'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
receipt = json.loads((R / 'FINAL-RECEIPT.json').read_text())
assert receipt['status'] == 'experiment_evidence_verified' and receipt['exit_code'] == 0
with (R / 'DEV-METRICS.csv').open('x', newline='') as stream:
    fields = ['cycle', 'variant', 'successful_updates', 'auroc', 'ap', 'error_twice', 'tensor_sha256']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: e[k] for k in fields} for e in receipt['dev_entries'])
zone = timezone(timedelta(hours=8))
clock = receipt['clock']
times = {'T0': clock['first_implementation_at'], 'binding_D': clock['implementation_deadline_at'],
         'round_D': clock['deadline_at'], 'done': receipt['done']['ended_at']}
for phase, value in receipt['components'].items():
    times.update({phase + '_start': value['start_at'], phase + '_end': value['end_at'],
                  phase + '_H': value['deadline_at'], phase + '_soft': value['deadline_at'] - 150})
timing = {'times': {k: {'epoch': v, 'Asia_Shanghai': datetime.fromtimestamp(v, zone).isoformat()} for k, v in times.items()},
          'round_T0_to_done_seconds': receipt['done']['ended_at'] - clock['first_implementation_at'],
          'passed_check_count': sum(c['passed'] for c in receipt['checks']), 'check_count': len(receipt['checks']),
          'dev_entries': len(receipt['dev_entries']), 'official_forward_invocations': 1}
write(R / 'TIMING.json', timing)
entries = []
for p in sorted(F.rglob('*')):
    if p.is_file() and not p.is_symlink() and p.stat().st_size < 2 * 1024**2:
        entries.append((p, 'formal/' + str(p.relative_to(F))))
entries += [(R / name, 'audit/' + name) for name in ('FINAL-RECEIPT.json', 'OUTPUT-MANIFEST.json', 'DEV-METRICS.csv', 'TIMING.json')]
entries += [(P / name, 'audit/' + name) for name in ('finalize_receipt.py', 'export_receipt.py')]
manifest = {'formal_root': str(F), 'files': [{'server_path': str(p), 'archive_path': rel, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p, rel in entries],
            'large_checkpoint_blobs_retained_on_Pod': True, 'exit_code': 0}
write(R / 'DELIVERY-MANIFEST.json', manifest)
artifact = R / 'SMALL-EVIDENCE.tar.gz'
with tarfile.open(artifact, 'x:gz') as archive:
    for p, rel in entries:
        archive.add(p, arcname=rel, recursive=False)
    archive.add(R / 'DELIVERY-MANIFEST.json', arcname='DELIVERY-MANIFEST.json', recursive=False)
value = {'status': 'small_originals_packaged', 'artifact': str(artifact), 'artifact_sha256': sha(artifact),
         'delivery_manifest': str(R / 'DELIVERY-MANIFEST.json'), 'delivery_manifest_sha256': sha(R / 'DELIVERY-MANIFEST.json'),
         'final_receipt': str(R / 'FINAL-RECEIPT.json'), 'final_receipt_sha256': sha(R / 'FINAL-RECEIPT.json'),
         'file_count': len(entries), 'export_script_sha256': sha(Path(__file__)), 'timing': timing, 'exit_code': 0}
write(R / 'DELIVERY-RECEIPT.json', value)
print(json.dumps(value, indent=2), flush=True)

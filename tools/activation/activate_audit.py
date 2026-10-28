"""Call the approved R92 activation once and reconcile its real receipt on Pod."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P = HERE.parent / 'resume9010-A-r92-prep-20261028-attempt3'
F = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R92-TINY1K-9010')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
site_path = HERE / 'SITE-PREFLIGHT-OWNER.json'
if not site_path.exists():
    site_path = HERE / 'SITE-PREFLIGHT.json'
site = read(site_path)
assert site['exit_code'] == 0 and site['Ready'] and site['pod_uid'] == 'eec33bd6-b567-40ef-b866-633e272ea019'
for limit in site['declared_lease_limits']:
    if limit['deadline_at'] is not None:
        assert limit['deadline_at'] - time.time() >= 12600
assert not site['lease_annotations'], 'Unresolved provider lease annotation needs reconciliation before clock'
owners = site['ownerReferences']
if owners and len(site['declared_lease_limits']) == 2:
    assert site['declared_lease_limits'][1]['uid'] in {x['uid'] for x in owners}
elif owners:
    # The Pod has no declared active deadline. Reading its upper Volcano Job is
    # RBAC-forbidden; preserve this limitation without claiming an unlimited
    # provider lease or inventing a new training/authorization gate.
    assert site['declared_lease_limits'][0]['activeDeadlineSeconds'] is None
    assert 'Forbidden' in (HERE / 'OWNER-LEASE-UNAVAILABLE.log').read_text()
reference_text = (HERE / 'R92-FORMAL-ROUTE.md').read_text()
match = re.search(r'authorization-reference / 本次实际 messageId：`([0-9a-f-]{36})`', reference_text)
assert match, 'Actual formal routing messageId must come from parent file'
reference = match.group(1)
assert sha(P / 'activate_r92.py') == read(P / 'PACKAGE-MANIFEST.json')['activate_script_sha256']
assert not F.exists() and not (P / 'FORMAL-CLOCK.json').exists()
write(HERE / 'ACTIVATION-CALL-STARTED.json', {'at': time.time(), 'authorization_reference': reference,
    'authorization_file_sha256': sha(HERE / 'R92-FORMAL-ROUTE.md'), 'site_receipt_sha256': sha(site_path),
    'activation_script': str(P / 'activate_r92.py'), 'activation_script_sha256': sha(P / 'activate_r92.py')})
argv = [sys.executable, '-B', str(P / 'activate_r92.py'), '--activate-formal', '--authorization-reference', reference,
        '--pod-uid', site['pod_uid'], '--ready', 'True']
with (HERE / 'ACTIVATION.log').open('x') as stream:
    completed = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=False)
(HERE / 'ACTIVATION-EXIT.txt').write_text(str(completed.returncode) + '\n')
assert completed.returncode == 0, 'Approved activation did not exit0; preserve same clock/root and do not retry'
receipt = read(F / 'evidence/FORMAL_BINDING_RECEIPT.json')
assert receipt['exit_code'] == 0 and receipt['status'] == 'formally_bound_not_dispatched' and receipt['GPU_started'] is False
assert receipt['source_digest'] == '57ef83209692eb79e682e04e2b3fe7a060225856f673505ab67396080216d51e'
assert receipt['clock']['authorization_reference'] == reference and receipt['clock'] == read(F / 'control/round-clock.json')
assert sha(F / 'RUNTIME_BINDING.json') == receipt['binding_sha256']
assert not (F / 'control/dispatch.json').exists()
write(HERE / 'FORMAL-BINDING-AUDIT.json', {'status': 'activation_exit0_and_actual_binding_verified', 'exit_code': 0,
    'root': str(F), 'clock': receipt['clock'], 'source_digest': receipt['source_digest'],
    'formal_receipt_path': str(F / 'evidence/FORMAL_BINDING_RECEIPT.json'),
    'formal_receipt_sha256': sha(F / 'evidence/FORMAL_BINDING_RECEIPT.json'),
    'runtime_binding_sha256': sha(F / 'RUNTIME_BINDING.json'), 'predecessor_handoff': read(F / 'control/predecessor-handoff.json'),
    'site_receipt_sha256': sha(site_path), 'GPU_started': False,
    'upper_Volcano_lease_readable': len(site['declared_lease_limits']) == 2,
    'Pod_declares_deadline': site['declared_lease_limits'][0]['activeDeadlineSeconds'] is not None})
print(json.dumps({'status': 'activation_exit0_and_actual_binding_verified', 'exit_code': 0, 'clock': receipt['clock'],
    'root': str(F), 'formal_receipt_sha256': sha(F / 'evidence/FORMAL_BINDING_RECEIPT.json'), 'GPU_started': False}), flush=True)

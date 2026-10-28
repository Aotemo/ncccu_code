"""R92 live identity/dependency/declared lease check; no GPU/lock/clock mutation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
UID = 'eec33bd6-b567-40ef-b866-633e272ea019'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
    assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
    import torch, torchvision, numpy, PIL
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    assert (sys.version_info[:3], torch.__version__, torchvision.__version__, numpy.__version__, PIL.__version__) == ((3, 12, 7), '2.10.0', '0.25.0', '1.26.4', '12.3.0')
    assert not torch.cuda.is_initialized()
    pod = read(HERE / 'POD-LIVE.json')
    assert pod['metadata']['uid'] == UID and pod['metadata']['namespace'] == 'acfwvv9ejn'
    assert pod['status']['phase'] == 'Running'
    assert any(c['type'] == 'Ready' and c['status'] == 'True' for c in pod['status']['conditions'])
    now = time.time(); limits = []
    records = [('pod', pod)]
    if (HERE / 'OWNER-LIVE.json').exists():
        records.append(('owner', read(HERE / 'OWNER-LIVE.json')))
    for kind, record in records:
        spec, status = record.get('spec', {}), record.get('status', {})
        duration = spec.get('activeDeadlineSeconds')
        start = status.get('startTime')
        deadline = None
        if duration is not None:
            assert type(duration) is int and duration > 0 and start
            deadline = datetime.datetime.fromisoformat(start.replace('Z', '+00:00')).timestamp() + duration
            assert deadline - now >= 12600, 'Declared remaining lease cannot cover R92 total budget'
        limits.append({'resource': kind, 'kind': record.get('kind'), 'name': record['metadata']['name'],
            'uid': record['metadata']['uid'], 'startTime': start, 'activeDeadlineSeconds': duration,
            'deadline_at': deadline, 'remaining_seconds': deadline - now if deadline else None})
    annotations = {kind + ':' + k: v for kind, record in records for k, v in record['metadata'].get('annotations', {}).items()
                   if any(x in k.lower() for x in ('deadline', 'lease', 'expire', 'timeout', 'duration'))}
    receipt = {'status': 'site_identity_dependencies_and_declared_lease_checked', 'at': now, 'exit_code': 0,
        'pod_uid': UID, 'Ready': True, 'python': sys.executable, 'version': sys.version,
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': numpy.__version__, 'PIL': PIL.__version__,
        'nice': os.getpriority(os.PRIO_PROCESS, 0), 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'CUDA_initialized': torch.cuda.is_initialized(), 'declared_lease_limits': limits,
        'lease_annotations': annotations, 'ownerReferences': pod['metadata'].get('ownerReferences', []),
        'pod_metadata_sha256': sha(HERE / 'POD-LIVE.json'), 'script_sha256': sha(__file__),
        'formal_T0_created': False, 'training_lock_acquired': False, 'GPU_started': False}
    name = 'SITE-PREFLIGHT-OWNER.json' if len(records) > 1 else 'SITE-PREFLIGHT.json'
    with (HERE / name).open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Explicit live Pod/dependency/declared lease check; never create a GPU context or clock."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pinned(path, digest):
    if path.is_symlink() or sha(path) != digest:
        raise ValueError('live metadata file SHA differs: ' + str(path))
    return json.loads(path.read_text())


def declared_limits(pod, owner, *, now, required_seconds):
    records = [('pod', pod)]
    owners = pod['metadata'].get('ownerReferences', [])
    if owner is not None:
        if owner['metadata']['uid'] not in {item['uid'] for item in owners}:
            raise ValueError('upper resource is not an actual Pod owner')
        records.append(('owner', owner))
    limits = []
    for kind, record in records:
        spec, status = record.get('spec', {}), record.get('status', {})
        duration = spec.get('activeDeadlineSeconds'); start = status.get('startTime'); deadline = None
        if duration is not None:
            if type(duration) is not int or duration <= 0 or not start:
                raise ValueError('declared lease is incomplete')
            deadline = datetime.datetime.fromisoformat(start.replace('Z', '+00:00')).timestamp() + duration
            if deadline - now < required_seconds:
                raise ValueError('declared remaining lease cannot cover the frozen round budget')
        limits.append({'resource': kind, 'kind': record.get('kind'), 'name': record['metadata']['name'],
            'uid': record['metadata']['uid'], 'startTime': start, 'activeDeadlineSeconds': duration,
            'deadline_at': deadline, 'remaining_seconds': deadline - now if deadline is not None else None})
    annotations = {kind + ':' + key: value for kind, record in records
        for key, value in record['metadata'].get('annotations', {}).items()
        if any(word in key.lower() for word in ('deadline', 'lease', 'expire', 'timeout', 'duration'))}
    if annotations:
        raise ValueError('provider lease annotations require explicit reconciliation before activation')
    return limits, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--site-binding', type=Path, required=True)
    parser.add_argument('--site-binding-sha256', required=True)
    parser.add_argument('--pod-live', type=Path, required=True)
    parser.add_argument('--pod-live-sha256', required=True)
    parser.add_argument('--owner-live', type=Path)
    parser.add_argument('--owner-live-sha256')
    parser.add_argument('--observed-at', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.prep_root / 'package/bundle'))
    from retrain.site_binding import load_site, require_cpu_channel, verify_live, evidence_path
    site = load_site(args.site_binding, expected_sha256=args.site_binding_sha256)
    require_cpu_channel(site)
    for path in (args.prep_root, args.site_binding, args.pod_live, args.output):
        evidence_path(path, site['work_root'])
    if bool(args.owner_live) != bool(args.owner_live_sha256):
        raise ValueError('owner metadata path and SHA must be supplied together')
    pod = read_pinned(args.pod_live, args.pod_live_sha256)
    owner = None
    if args.owner_live:
        evidence_path(args.owner_live, site['work_root'])
        owner = read_pinned(args.owner_live, args.owner_live_sha256)
    if (pod['metadata']['name'] != site['pod']['name'] or pod['metadata']['namespace'] != site['pod']['namespace']
            or pod['status']['phase'] != 'Running'
            or site['pod']['container'] not in {item['name'] for item in pod['spec']['containers']}):
        raise ValueError('live Pod name/namespace/phase/container differs')
    ready = any(item['type'] == 'Ready' and item['status'] == 'True' for item in pod['status']['conditions'])
    verify_live(site, pod_uid=pod['metadata']['uid'], ready=ready, observed_at=args.observed_at)
    policy_path = args.prep_root / 'package/CLOCK_POLICY.json'
    policy = json.loads(policy_path.read_text())
    required = policy['total_seconds']
    if type(required) is not int or required <= 0 or policy['round_id'] != site['round_id']:
        raise ValueError('explicit positive round budget required')
    now = time.time()
    limits, annotations = declared_limits(pod, owner, now=now, required_seconds=required)
    import torch, torchvision, numpy, PIL
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS'])); torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError('CPU preflight initialized an accelerator')
    verify_live(site, pod_uid=pod['metadata']['uid'], ready=ready, observed_at=args.observed_at)
    receipt = {'schema_version': 1, 'status': 'site_identity_dependencies_and_declared_lease_checked',
        'at': time.time(), 'observed_at': args.observed_at, 'exit_code': 0, 'pod_uid': site['pod']['uid'],
        'Ready': True, 'python': sys.executable, 'version': sys.version,
        'site_binding_sha256': args.site_binding_sha256, 'round_id': site['round_id'],
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': numpy.__version__, 'PIL': PIL.__version__,
        'nice': os.getpriority(os.PRIO_PROCESS, 0), 'cpu_mask': sorted(os.sched_getaffinity(0)),
        'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'CUDA_initialized': False, 'declared_lease_limits': limits, 'required_seconds': required,
        'lease_annotations': annotations, 'ownerReferences': pod['metadata'].get('ownerReferences', []),
        'upper_lease_readable': owner is not None,
        'upper_lease_status': 'checked' if owner is not None else 'unknown_not_an_unlimited_lease',
        'pod_metadata_sha256': args.pod_live_sha256, 'owner_metadata_sha256': args.owner_live_sha256,
        'clock_policy_sha256': sha(policy_path), 'script_sha256': sha(__file__),
        'formal_T0_created': False, 'training_lock_acquired': False, 'GPU_started': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

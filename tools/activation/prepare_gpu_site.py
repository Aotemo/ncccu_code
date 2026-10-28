"""Prepare a future GPU CPU allocation after migration and the peer CPU tree exit."""
import argparse
import copy
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(path, digest):
    require(Path(path).is_absolute() and not Path(path).is_symlink() and sha(path) == digest, 'input receipt SHA differs')
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def partition_cpus(resources, owner):
    available = sorted(set(resources['affinity']) & set(resources['cpuset']))
    quota, period = resources['cpu_max']
    require(int(period) > 0 and (quota == 'max' or int(quota) > 0), 'invalid current CPU quota')
    capacity = Fraction(len(available)) if quota == 'max' else min(Fraction(len(available)), Fraction(int(quota), int(period)))
    total = int(capacity // 5)
    require(total >= 9, '20% CPU budget cannot supply eight ranks and a controller')
    width = (total - 1) // 8
    mask = available[:total]
    return {'owner_id': owner, 'effective_cpus': int(capacity) if capacity.denominator == 1 else float(capacity),
        'total_fraction_cap': '1/5', 'owner_mask': mask, 'control_mask': mask[8 * width:],
        'rank_masks': [mask[rank * width:(rank + 1) * width] for rank in range(8)],
        'rank_threads': 1, 'interop_threads': 1, 'loader_workers': int(width >= 2)}


def verify_peer_exit(value, uid, owner):
    require(type(value.get('exit_code')) is int and type(value.get('child_exit_code')) is int
            and value['exit_code'] == value['child_exit_code'] == 0
            and value.get('task_tree_exited') is True and value.get('affinity_violations') == []
            and isinstance(value.get('executor_id'), str)
            and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', value['executor_id'])
            and value['executor_id'] != owner and value.get('identity', {}).get('NCCCU_POD_UID') == uid,
            'peer CPU executor has not proved clean exit in the current Pod')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('technical-site', 'technical-site-sha256', 'relocation-receipt', 'relocation-receipt-sha256',
                 'released-cpu-receipt', 'released-cpu-receipt-sha256', 'git-commit', 'remote-commit', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--valid-seconds', required=True, type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime/bundle'))
    from retrain.site_binding import (load_site, require_cpu_channel, evidence_path, live_cpu_resources,
                                      validate_site, validate_gpu_allocation)
    site = load_site(args.technical_site, expected_sha256=args.technical_site_sha256)
    require_cpu_channel(site)
    require(site.get('scope') == 'cpu_technical_prep', 'explicit prior technical site required')
    require(re.fullmatch('[0-9a-f]{40}', args.git_commit) and args.git_commit == args.remote_commit,
            'the actual Git commit must equal the connector-verified remote head')
    require(1 <= args.valid_seconds <= 21600, 'explicit GPU allocation TTL must be at most six hours')
    relocation = pinned(evidence_path(args.relocation_receipt, site['work_root']), args.relocation_receipt_sha256)
    require(relocation.get('passed') is True and relocation.get('phase') == 'complete'
            and relocation.get('execution', {}).get('uid') == site['pod']['uid']
            and relocation.get('source_modified_by_tool') is False
            and relocation.get('historical_FREEZE_bytes_preserved') is True
            and relocation.get('official_manifest_semantics_read') is False,
            'complete current migration/cache relocation receipt required')
    for key, field in {'data_root': 'dataset_root', 'metadata_root': 'stage1_root',
                       'public_weight': 'public_pretrained', 'public_provenance': 'public_metadata',
                       'train_identity': 'train_identity', 'prepared_cache': 'prepared_cache_root',
                       'official_manifest': 'official_manifest'}.items():
        require(site[key] == relocation['paths'][field], 'relocation target differs: ' + key)
    owner = os.environ['NCCCU_EXECUTOR_ID']
    peer = pinned(evidence_path(args.released_cpu_receipt, site['work_root']), args.released_cpu_receipt_sha256)
    verify_peer_exit(peer, site['pod']['uid'], owner)
    allocation = evidence_path(peer['allocation_path'], site['work_root'])
    require(sha(allocation) == peer['allocation_sha256'], 'peer CPU allocation pin differs')
    lease = allocation.parent / ('.cpu-task-' + peer['executor_id'] + '.lock')
    descriptor = os.open(lease, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed_exit_at = time.time()
        resources = live_cpu_resources()
        # The current preparation task has its own narrow mask. Observe the
        # container init affinity for the future uncontracted GPU entry; that
        # entry must independently match all original resources before binding.
        resources['affinity'] = sorted(os.sched_getaffinity(1))
        policy = partition_cpus(resources, owner)
        resources['effective_cpus'] = policy['effective_cpus']
        output = evidence_path(args.output, site['work_root'])
        require('technical-prep' in output.parts, 'GPU input artifacts must stay in technical-prep')
        output.mkdir(parents=True, exist_ok=False)
        release_path = output / 'CPU-EXECUTOR-RELEASE.json'
        write(release_path, {'executor_id': peer['executor_id'], 'pod_uid': site['pod']['uid'],
            'exit_code': 0, 'task_tree_exited': True, 'live_pids': [], 'ended_at': observed_exit_at,
            'ended_at_semantics': 'clean_exit_observed_and_CPU_task_lease_free_at_this_time',
            'original_cpu_receipt': {'path': args.released_cpu_receipt, 'sha256': args.released_cpu_receipt_sha256},
            'CPU_task_lease_inode': os.fstat(descriptor).st_ino, 'signals_sent': False})
        created = time.time()
        grant = {'schema_version': 1, 'stage': 'gpu-owner',
            'execution': {key: site['pod'][field] for key, field in (
                ('namespace', 'namespace'), ('pod', 'name'), ('uid', 'uid'), ('container', 'container'), ('python', 'python'))},
            'resources': resources, 'total_fraction_cap': '1/5', 'owner_id': owner,
            'owner_mask': policy['owner_mask'], 'created_at': created, 'valid_until': created + args.valid_seconds,
            'affinity_observation': 'current_container_PID1; actual_uncontracted_GPU_entry_must_recheck',
            'released_executors': [{'executor_id': peer['executor_id'], 'receipt_path': str(release_path),
                                    'receipt_sha256': sha(release_path)}]}
        grant_path = output / 'GPU-CPU-ALLOCATION.json'; write(grant_path, grant)
        policy['allocation_receipt'] = {'path': str(grant_path), 'sha256': sha(grant_path)}
        formal = copy.deepcopy(site)
        formal.update(scope='formal_exp_v3', git_commit=args.git_commit, cpu_execution=policy,
            source_state={'kind': 'remote_commit_frozen', 'git_commit': args.git_commit},
            relocation_receipt={'path': args.relocation_receipt, 'sha256': args.relocation_receipt_sha256})
        validate_site(formal)
        validate_gpu_allocation(formal, grant, resources, initial=True, now=created)
        site_path = output / 'SITE_BINDING.json'; write(site_path, formal)
        result = {'status': 'future_GPU_inputs_prepared_not_dispatched', 'site_binding': str(site_path),
            'site_binding_sha256': sha(site_path), 'allocation': str(grant_path), 'allocation_sha256': sha(grant_path),
            'effective_cpus': policy['effective_cpus'], 'owner_mask': policy['owner_mask'],
            'rank_masks': policy['rank_masks'], 'control_mask': policy['control_mask'],
            'training_lock_acquired': False, 'GPU_started': False, 'formal_clock_created': False}
        write(output / 'INPUTS-RECEIPT.json', result)
        print(json.dumps(result, indent=2))
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    main()

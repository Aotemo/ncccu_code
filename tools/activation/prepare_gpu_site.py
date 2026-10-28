"""Prepare disjoint configured validation/training CPU pools from live resources."""
import argparse
import copy
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


def verify_native_guard(receipt_path, receipt_sha256, resource_policy, work_root):
    from retrain.site_binding import evidence_path
    receipt_path = evidence_path(receipt_path, work_root)
    require(re.fullmatch('[0-9a-f]{64}', receipt_sha256 or '') and receipt_path.is_file(),
            'explicit native guard build receipt path/SHA required')
    value = pinned(receipt_path, receipt_sha256)
    checks = value.get('checks')
    require(type(value.get('schema_version')) is int and value['schema_version'] == 1
            and value.get('passed') is True and value.get('scope') == 'compute_pod_native_affinity_guard'
            and isinstance(checks, list) and checks
            and all(isinstance(row, dict) and row.get('passed') is True for row in checks),
            'successful native affinity guard CPU build checks required')
    try:
        policy_path = evidence_path(resource_policy['path'], work_root)
        require(policy_path.is_file() and sha(policy_path) == resource_policy['sha256']
                and json.loads(policy_path.read_text()) == resource_policy['value'],
                'native guard resource policy pin differs')
        relative = Path(resource_policy['value']['native_affinity_guard_source'])
        require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
                'native guard source must be relative to the explicit Git source root')
        source_root = policy_path.parent.parent
        source = evidence_path(source_root / relative, source_root)
        recorded_source = evidence_path(value['source']['path'], work_root)
        artifact = evidence_path(value['artifact']['path'], work_root)
        require(artifact.resolve().is_relative_to(Path(work_root).resolve() / 'technical-prep'),
                'native guard artifact must stay in technical-prep')
        require(source.is_file() and recorded_source.is_file() and artifact.is_file()
                and re.fullmatch('[0-9a-f]{64}', value['source']['sha256'] or '')
                and sha(source) == sha(recorded_source) == value['source']['sha256'],
                'native guard Git/build source SHA differs')
        require(re.fullmatch('[0-9a-f]{64}', value['artifact']['sha256'] or '')
                and sha(artifact) == value['artifact']['sha256'], 'native guard artifact SHA differs')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('incomplete native affinity guard binding') from error
    return {'path': str(artifact), 'sha256': value['artifact']['sha256'],
            'source': {'path': str(source), 'sha256': value['source']['sha256']},
            'build_receipt': {'path': str(receipt_path), 'sha256': receipt_sha256}}


def partition_cpus(resources, owner, resource_policy, world):
    available = sorted(set(resources['affinity']) & set(resources['cpuset']))
    quota, period = resources['cpu_max']
    require(int(period) > 0 and (quota == 'max' or int(quota) > 0), 'invalid current CPU quota')
    capacity = Fraction(len(available)) if quota == 'max' else min(Fraction(len(available)), Fraction(int(quota), int(period)))
    settings = resource_policy['value']
    fractions = {name: Fraction(settings['cpu_fractions'][name]) for name in ('validation', 'training')}
    require(all(0 < part <= 1 for part in fractions.values()) and sum(fractions.values()) <= 1,
            'configured CPU fractions exceed the device budget')
    validation_count = int(capacity * fractions['validation'])
    total = int(capacity * fractions['training'])
    require(validation_count >= 1 and total >= world + 1, 'configured pools cannot supply validation/ranks/controller')
    width = (total - 1) // world
    mask = available[validation_count:validation_count + total]
    return {'owner_id': owner, 'effective_cpus': int(capacity) if capacity.denominator == 1 else float(capacity),
        'total_fraction_cap': settings['cpu_fractions']['training'], 'resource_policy': resource_policy,
        'validation_mask': available[:validation_count], 'owner_mask': mask, 'control_mask': mask[world * width:],
        'rank_masks': [mask[rank * width:(rank + 1) * width] for rank in range(world)],
        'rank_threads': settings['rank_threads'], 'interop_threads': settings['interop_threads'],
        'loader_workers': min(settings['loader_workers'], max(0, width - settings['rank_threads']))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('technical-site', 'technical-site-sha256', 'relocation-receipt', 'relocation-receipt-sha256',
                 'validation-cpu-allocation', 'validation-cpu-allocation-sha256',
                 'native-guard-receipt', 'native-guard-receipt-sha256',
                 'resource-policy', 'resource-policy-sha256', 'git-commit', 'remote-commit', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--valid-seconds', required=True, type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime/bundle'))
    from retrain.site_binding import (load_site, require_cpu_channel, evidence_path, live_cpu_resources,
                                      validate_site, validate_gpu_allocation, load_resource_policy, verify_validation_pool)
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
    validation = pinned(evidence_path(args.validation_cpu_allocation, site['work_root']), args.validation_cpu_allocation_sha256)
    resource_policy = load_resource_policy(args.resource_policy, args.resource_policy_sha256)
    native_guard = verify_native_guard(args.native_guard_receipt, args.native_guard_receipt_sha256,
                                       resource_policy, site['work_root'])
    resources = live_cpu_resources()
    # Observe uncontracted container resources; the GPU entry rechecks them.
    resources['affinity'] = sorted(os.sched_getaffinity(1))
    policy = partition_cpus(resources, owner, resource_policy, len(site['gpus']))
    policy['native_affinity_guard'] = native_guard
    verify_validation_pool(validation, site['pod']['uid'], policy['validation_mask'], policy['owner_mask'])
    resources['effective_cpus'] = policy['effective_cpus']
    output = evidence_path(args.output, site['work_root'])
    require('technical-prep' in output.parts, 'GPU input artifacts must stay in technical-prep')
    output.mkdir(parents=True, exist_ok=False)
    created = time.time()
    grant = {'schema_version': 1, 'stage': 'gpu-owner',
            'execution': {key: site['pod'][field] for key, field in (
                ('namespace', 'namespace'), ('pod', 'name'), ('uid', 'uid'), ('container', 'container'), ('python', 'python'))},
            'resources': resources, 'total_fraction_cap': policy['total_fraction_cap'], 'owner_id': owner,
            'resource_policy': resource_policy, 'validation_mask': policy['validation_mask'],
            'native_affinity_guard': copy.deepcopy(native_guard),
            'owner_mask': policy['owner_mask'], 'created_at': created, 'valid_until': created + args.valid_seconds,
            'affinity_observation': 'current_container_PID1; actual_uncontracted_GPU_entry_must_recheck',
            'validation_allocation': {'path': args.validation_cpu_allocation,
                                      'sha256': args.validation_cpu_allocation_sha256}}
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


if __name__ == '__main__':
    main()

"""F06 official-5849 fine-tuning preparation: warm start, config, site grant and fresh spec.

CPU only. Every artifact is written with exclusive creation; nothing is overwritten.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

WARM_TENSOR_SHA = 'f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85'
EXPECTED_UPDATES = 4800
FOLD_ROWS = [1950, 1950, 1949]
FOLD_POSITIVE = [64, 64, 63]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False).encode() + b'\n'
    with path.open('xb') as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def bundle_path():
    return Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'


def cmd_warmstart(args):
    sys.path.insert(0, str(bundle_path()))
    import torch
    from retrain.ema import tensor_digest
    from retrain.persistence import load_file
    checkpoint = Path(args.checkpoint)
    payload = load_file(checkpoint, map_location='cpu')
    if payload.get('successful_updates') != EXPECTED_UPDATES:
        raise SystemExit('source checkpoint update count differs')
    ema = payload.get('ema') or {}
    state = ema.get('model')
    if (not isinstance(state, dict) or ema.get('update_count') != EXPECTED_UPDATES
            or tensor_digest(state) != WARM_TENSOR_SHA):
        raise SystemExit('registered R120 EMA tensor identity differs')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'warmstart.pt'
    if target.exists():
        raise SystemExit('warm-start export already exists')
    torch.save({name: tensor.detach().cpu().clone() for name, tensor in state.items()}, target)
    receipt = {'status': 'f06_warm_start_exported', 'format': 'exported-fp32-state-dict-v1',
               'path': str(target), 'sha256': sha(target), 'tensor_sha256': tensor_digest(state),
               'tensors': len(state), 'variant': 'ema', 'updates': EXPECTED_UPDATES,
               'source_model': str(checkpoint), 'source_model_sha256': sha(checkpoint),
               'GPU_initialized': False, 'official_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(output / 'WARMSTART.json', receipt)}, indent=1))


def cmd_config(args):
    template = json.loads(Path(args.template).read_text())
    fold = int(args.fold)
    if not 0 <= fold < 3:
        raise SystemExit('fold outside 0..2')
    warm = json.loads(Path(args.warmstart).read_text())
    if warm['tensor_sha256'] != WARM_TENSOR_SHA or sha(warm['path']) != warm['sha256']:
        raise SystemExit('warm-start receipt/bytes differ')
    folds_sha = sha(args.folds)
    folds = json.loads(Path(args.folds).read_text())
    if (folds.get('format') != 'f06-official-folds-v1' or folds_sha != args.folds_sha256
            or folds['population'] != {'rows': 5849, 'positive': 191, 'negative': 5658}):
        raise SystemExit('fold registration identity differs')
    cfg = copy.deepcopy(template)
    cfg['run_name'] = f"{args.round_id}-8c-convnext_small-s{20261106 + fold}-b64-p8-fp32"
    cfg['output_root'] = str(Path(args.round_root) / 'outputs')
    cfg['data']['split'] = {'kind': 'official_cv', 'seed': 20261024, 'valid_fraction': .1,
                            'folds': 3, 'fold': fold, 'inner_fraction': .1, 'include_valid': True,
                            'folds_path': str(Path(args.folds).resolve()), 'folds_sha256': folds_sha,
                            'expected_fold_counts': [[FOLD_ROWS[i], FOLD_POSITIVE[i]] for i in range(3)]}
    job = cfg['jobs'][0]
    job.pop('experiment91', None)
    job.update(seed=20261106 + fold, cycles=15, warmup_cycles=1, min_cycles=1, patience=4,
               backbone_lr=1e-5, head_lr=1e-4,
               initialization={'kind': 'checkpoint_warm_start', 'format': 'exported-fp32-state-dict-v1',
                               'path': warm['path'], 'sha256': warm['sha256'],
                               'tensor_sha256': warm['tensor_sha256'], 'source_model': warm['source_model']},
               f06={'version': 'f06-official-ft-v1', 'fold': fold, 'folds_sha256': folds_sha})
    output = Path(args.output)
    print(json.dumps({'status': 'f06_train_config_written', 'path': str(output),
                      'sha256': write_new(output, cfg), 'fold': fold,
                      'run_name': cfg['run_name']}, indent=1))


def cmd_site(args):
    sys.path.insert(0, str(bundle_path()))
    sys.path.insert(0, str(Path(args.source_root) / 'tools' / 'activation'))
    from prepare_gpu_site import partition_cpus, verify_native_guard
    from retrain.site_binding import (evidence_path, load_resource_policy, live_cpu_resources,
                                      sha256, validate_gpu_allocation, validate_site,
                                      verify_validation_pool)
    technical = json.loads(Path(args.technical_site).read_text())
    if sha(args.technical_site) != args.technical_site_sha256:
        raise SystemExit('technical site SHA differs')
    if technical['pod']['uid'] != args.pod_uid:
        raise SystemExit('technical site belongs to another Pod')
    policy = load_resource_policy(args.resource_policy, args.resource_policy_sha256)
    native = verify_native_guard(args.native_guard_receipt, args.native_guard_receipt_sha256,
                                 policy, technical['work_root'])
    validation = json.loads(Path(args.validation_cpu_allocation).read_text())
    if sha(args.validation_cpu_allocation) != args.validation_cpu_allocation_sha256:
        raise SystemExit('validation allocation SHA differs')
    resources = live_cpu_resources()
    resources['affinity'] = sorted(os.sched_getaffinity(1))
    policy_value = partition_cpus(resources, os.environ['NCCCU_EXECUTOR_ID'], policy,
                                  len(technical['gpus']))
    policy_value['native_affinity_guard'] = native
    verify_validation_pool(validation, technical['pod']['uid'], policy_value['validation_mask'],
                           policy_value['owner_mask'])
    resources['effective_cpus'] = policy_value['effective_cpus']
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    created = time.time()
    grant = {'schema_version': 1, 'stage': 'gpu-owner',
             'execution': {key: technical['pod'][field] for key, field in (
                 ('namespace', 'namespace'), ('pod', 'name'), ('uid', 'uid'),
                 ('container', 'container'), ('python', 'python'))},
             'resources': resources, 'total_fraction_cap': policy_value['total_fraction_cap'],
             'owner_id': policy_value['owner_id'], 'resource_policy': policy,
             'validation_mask': policy_value['validation_mask'],
             'native_affinity_guard': copy.deepcopy(native),
             'owner_mask': policy_value['owner_mask'], 'created_at': created,
             'valid_until': created + args.valid_seconds,
             'affinity_observation': 'current_container_PID1; actual_uncontracted_GPU_entry_must_recheck',
             'validation_allocation': {'path': args.validation_cpu_allocation,
                                       'sha256': args.validation_cpu_allocation_sha256}}
    grant_path = output / 'GPU-CPU-ALLOCATION.json'
    write_new(grant_path, grant)
    policy_value['allocation_receipt'] = {'path': str(grant_path), 'sha256': sha(grant_path)}
    site = copy.deepcopy(technical)
    site.update(scope='formal_exp_v3', round_id=args.round_id, round_root=args.round_root,
                git_commit=args.git_commit, source_state={'kind': 'remote_commit_frozen',
                                                          'git_commit': args.git_commit},
                cpu_execution=policy_value)
    site['science'] = copy.deepcopy(technical['science'])
    site['science']['binding_version'] = 'f06-official-ft-v1'
    site['science']['configuration'] = {'path': args.science_config, 'sha256': sha(args.science_config)}
    site_path = output / 'SITE_BINDING.json'
    validate_site(site)
    validate_gpu_allocation(site, grant, resources, initial=True, now=created)
    print(json.dumps({'status': 'f06_gpu_site_written', 'site': str(site_path),
                      'site_sha256': write_new(site_path, site),
                      'allocation': str(grant_path), 'allocation_sha256': sha(grant_path),
                      'effective_cpus': policy_value['effective_cpus'],
                      'owner_mask': policy_value['owner_mask'],
                      'rank_masks': policy_value['rank_masks'],
                      'control_mask': policy_value['control_mask'],
                      'validation_mask': policy_value['validation_mask'],
                      'native_guard': native}, indent=1))


def cmd_spec(args):
    sys.path.insert(0, str(bundle_path()))
    os.environ['NCCCU_SITE_BINDING'] = args.site
    os.environ['NCCCU_SITE_BINDING_SHA256'] = sha(args.site)
    import train
    from retrain.config import job_contract, load_config
    from retrain.data import build_manifest, load_source
    from retrain.ema import RULE as EMA_RULE
    from retrain.execution_policy import resolve_topology
    from retrain.fresh import resolve_steps_per_cycle, validate_fresh
    from retrain.models import validate_initialization
    from retrain.persistence import canonical_digest
    from retrain.precision import RULE as PRECISION_RULE
    cfg = load_config(args.config)
    job, plan, policy = resolve_topology(cfg, available_accelerators=8)
    if job.get('f06') is None:
        raise SystemExit('F06 job binding required')
    source = load_source(cfg['data']['root'], cfg['data']['identity_path'], True, True,
                         cfg['data']['expected_count'])
    manifest = build_manifest(source, cfg['data']['split'])
    fold = job['f06']['fold']
    dev_positive = manifest['counts']['dev']['positive']
    if (len(manifest['dev_indices']) != FOLD_ROWS[fold] or dev_positive != FOLD_POSITIVE[fold]
            or len(manifest['fit_indices']) != 46256 + 5849 - FOLD_ROWS[fold]
            or set(manifest['fit_indices']) & set(manifest['dev_indices'])):
        raise SystemExit('manifest fold counts differ from the registered split')
    negatives = sum(not manifest['rows'][i]['label'] for i in manifest['fit_indices'])
    steps = resolve_steps_per_cycle(job, negatives, fresh_spec={'purpose': 'train'})
    if steps != 400:
        raise SystemExit('registered F06 steps-per-cycle differs')
    hashes, digest = train.source_identity()
    plan_path = Path(args.plan_out)
    plan_value = {'format': 'f06-fresh-plan-v1', 'round_id': cfg['run_name'], 'fold': fold,
                  'config_sha256': sha(args.config), 'folds_sha256': job['f06']['folds_sha256'],
                  'warm_start_tensor_sha256': job['initialization']['tensor_sha256'],
                  'schedule': {'steps_per_cycle': 400, 'total_updates': 6000,
                               'warmup_updates': 400, 'stop_after_updates': 6000}}
    plan_sha = write_new(plan_path, plan_value)
    fresh = {'format': 'glaucoma-fresh-init-v1', 'purpose': 'train',
             'plan_version': 'f06-official-ft-v1', 'plan_sha256': plan_sha,
             'source_digest': digest, 'job': job_contract(job),
             'schedule': {'steps_per_cycle': 400, 'total_updates': 6000,
                          'warmup_updates': 400, 'stop_after_updates': 6000},
             'execution_policy': policy, 'ema': copy.deepcopy(EMA_RULE),
             'precision_protocol': copy.deepcopy(PRECISION_RULE),
             'initialization': validate_initialization(job['initialization']),
             'amp_initial_scale': 1.0, 'cache_readonly_required': True}
    identity = {'format': 'glaucoma-retrain-v1', 'source_digest': digest,
                'manifest_digest': canonical_digest(manifest), 'job': job_contract(job),
                'initialization': fresh['initialization'], 'phase_parent_sha256': None,
                'distributed': plan, 'fresh': fresh}
    validate_fresh(fresh, identity, job, plan, steps, mode='train')
    fresh_path = Path(args.fresh_out)
    fresh_sha = write_new(fresh_path, fresh)
    receipt = {'status': 'f06_fresh_spec_validated_not_launched', 'fold': fold,
               'config': str(args.config), 'config_sha256': sha(args.config),
               'plan': str(plan_path), 'plan_sha256': plan_sha,
               'fresh_spec': str(fresh_path), 'fresh_spec_sha256': fresh_sha,
               'source_digest': digest, 'manifest_digest': identity['manifest_digest'],
               'fit_rows': len(manifest['fit_indices']), 'dev_rows': len(manifest['dev_indices']),
               'dev_positive': dev_positive, 'negatives': negatives, 'steps_per_cycle': steps,
               'topology': plan, 'GPU_initialized': False, 'official_read': True,
               'test_labels_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(args.receipt, receipt)}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    warm = sub.add_parser('warmstart')
    warm.add_argument('--checkpoint', required=True)
    warm.add_argument('--output-dir', required=True)
    config = sub.add_parser('config')
    config.add_argument('--template', required=True)
    config.add_argument('--folds', required=True)
    config.add_argument('--folds-sha256', required=True)
    config.add_argument('--warmstart', required=True)
    config.add_argument('--fold', required=True)
    config.add_argument('--round-id', required=True)
    config.add_argument('--round-root', required=True)
    config.add_argument('--output', required=True)
    site = sub.add_parser('site')
    site.add_argument('--source-root', required=True)
    site.add_argument('--technical-site', required=True)
    site.add_argument('--technical-site-sha256', required=True)
    site.add_argument('--pod-uid', required=True)
    site.add_argument('--resource-policy', required=True)
    site.add_argument('--resource-policy-sha256', required=True)
    site.add_argument('--native-guard-receipt', required=True)
    site.add_argument('--native-guard-receipt-sha256', required=True)
    site.add_argument('--validation-cpu-allocation', required=True)
    site.add_argument('--validation-cpu-allocation-sha256', required=True)
    site.add_argument('--round-id', required=True)
    site.add_argument('--round-root', required=True)
    site.add_argument('--git-commit', required=True)
    site.add_argument('--science-config', required=True)
    site.add_argument('--valid-seconds', type=int, required=True)
    site.add_argument('--output-dir', required=True)
    spec = sub.add_parser('spec')
    spec.add_argument('--config', required=True)
    spec.add_argument('--site', required=True)
    spec.add_argument('--plan-out', required=True)
    spec.add_argument('--fresh-out', required=True)
    spec.add_argument('--receipt', required=True)
    args = parser.parse_args()
    {'warmstart': cmd_warmstart, 'config': cmd_config, 'site': cmd_site,
     'spec': cmd_spec}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

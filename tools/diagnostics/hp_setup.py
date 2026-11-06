"""HP hard-positive / OHEM warm-start preparation: config, fresh spec and GPU site.

CPU only, mirrors the F06 flow. Every artifact is written with exclusive
creation; nothing is overwritten. The warm start reuses the registered F06
export of the R120 selected EMA.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PAIR_SIGMOID_SPEC = {'version': 'binary-pairwise-sigmoid-v1', 'kind': 'binary_pairwise_sigmoid',
                     'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}
OHEM_SPEC = {'version': 'binary-pairwise-sigmoid-ohem-top25-v1', 'kind': 'binary_pairwise_sigmoid_ohem',
             'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
             'hard_fraction': 0.25, 'tie_break': 'WORLD_pair_row_major_stable'}
HP_SEED = 20261203
OHEM_SEED = 20261204
F11_SEED = 20261205
SCHEDULE = {'steps_per_cycle': 400, 'total_updates': 6000, 'warmup_updates': 400, 'stop_after_updates': 6000}
R120_WARM_TENSOR = 'f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85'
R120_WARM_FILE = '2ac4a025cdef2835127ee619c7f37a6d094b62eefdb1943005437a49e570dfa3'
F11_WARM = ('6251937d3d778ed4be5d0cc11fc505fc4c2cc897d0ae18ff31ddfb6c8dce8a73', 'f78ab30c559dba43fd3af434b0144ce1f54fae5e4eb2020f1001d1d5002bec6e')


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


def cmd_config(args):
    warm = json.loads(Path(args.warmstart).read_text())
    expected = F11_WARM if args.mode == 'f11' else (R120_WARM_TENSOR, R120_WARM_FILE)
    if (warm.get('tensor_sha256') != expected[0]
            or sha(warm['path']) != warm['sha256']
            or warm['sha256'] != expected[1]):
        raise SystemExit('warm-start receipt/bytes differ from the registered R120 EMA export')
    age = json.loads(Path(args.age_contract).read_text())
    if not age.get('enabled') or age['contract']['fit_count'] != 46256:
        raise SystemExit('registered original-fit age contract required')
    seed = {'hp': HP_SEED, 'ohem': OHEM_SEED, 'f11': F11_SEED}[args.mode]
    job = {
        'name': 'convnext-small', 'model': 'convnext_small', 'seed': seed,
        'training_scale': [.85, 1.0],
        'loss': dict(PAIR_SIGMOID_SPEC if args.mode == 'hp' else OHEM_SPEC),
        'image_size': 512, 'prepared_size': 512, 'physical_batch': 32, 'effective_batch': 64,
        'positive_per_batch': 8, 'precision': 'fp32', 'cycles': 15, 'warmup_cycles': 1,
        'min_cycles': 1, 'patience': 4, 'min_delta': 0.0, 'stop_after_cycles': 0,
        'backbone_lr': 1e-5, 'head_lr': 1e-4, 'weight_decay': 1e-4, 'dropout': .2,
        'stochastic_depth': .1, 'normalization': 'imagenet', 'final_norm_eps': 1e-6,
        'head_lr_scope': 'last_linear', 'mixup': None, 'attention_pool': None,
        'initial_checkpoint': None, 'frozen_bn': False, 'benchmark_warmup': 50,
        'benchmark_updates': 100, 'checkpoint_seconds': 300, 'checkpoint_updates': 0,
        'log_every': 25, 'prefetch_factor': 2, 'workers': 0, 'eval_workers': 0, 'split': {},
        'age': age,
        'initialization': {'kind': 'checkpoint_warm_start', 'format': 'exported-fp32-state-dict-v1',
                           'path': warm['path'], 'sha256': warm['sha256'],
                           'tensor_sha256': warm['tensor_sha256'], 'source_model': warm['source_model']},
        ('hp' if args.mode == 'hp' else ('f11' if args.mode == 'f11' else 'hp')):
            ({'version': 'hp-r120-warmstart-fit-only-v1'} if args.mode != 'f11'
             else {'version': 'f11-official-merge-r249-warmstart-v1'}),
    }
    if args.mode == 'hp':
        for name in ('summary', 'verification', 'target'):
            reference = getattr(args, 'f05_' + name)
            if not reference or not Path(reference).is_absolute():
                raise SystemExit('HP mode requires the prepared F05 hardpos binding references')
        job['f05_targets'] = {
            'version': 'f05-oof-training-v1', 'mode': 'hardpos',
            'summary': {'path': args.f05_summary, 'sha256': sha(args.f05_summary)},
            'verification': {'path': args.f05_verification, 'sha256': sha(args.f05_verification)},
            'target': {'path': args.f05_target, 'sha256': sha(args.f05_target)}}
    elif any(getattr(args, 'f05_' + name) for name in ('summary', 'verification', 'target')):
        raise SystemExit('OHEM mode consumes no F05 targets')
    cfg = {
        'schema_version': 1, 'mode': 'train', 'resume': False,
        'run_name': args.round_id + '-8c-convnext_small-s' + str(seed) + '-b64-p8-fp32',
        'output_root': str(Path(args.round_root) / 'outputs'),
        'data': {'expected_count': 51396, 'expected_valid_count': 5849,
                 'identity_path': args.identity_path, 'root': args.data_root,
                 'split': ({'kind': 'official_merge', 'seed': 20261024, 'valid_fraction': .1,
                            'inner_fraction': .1, 'include_valid': True, 'fold': 0, 'folds': 3}
                           if args.mode == 'f11' else
                           {'kind': 'holdout', 'seed': 20261024, 'valid_fraction': .1,
                            'inner_fraction': .1, 'include_valid': False, 'fold': 0, 'folds': 3}),
                 'verify_images': True},
        'execution': {'cpu_cores_per_job': 1,
                      'devices': ['cuda:' + str(index) for index in range(8)],
                      'evaluation_physical_batch': 32, 'grace_seconds': 150,
                      'heartbeat_seconds': 15, 'legacy_six_card_exception': False,
                      'per_rank_microbatch': 0, 'technical_smoke': False},
        'jobs': [job]}
    print(json.dumps({'status': 'hp_train_config_written', 'mode': args.mode,
                      'path': str(args.output), 'sha256': write_new(args.output, cfg),
                      'run_name': cfg['run_name']}, indent=1))


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
    plan['stop_after_updates'] = SCHEDULE['stop_after_updates']
    if (job.get('hp') is None and job.get('f11') is None) or job.get('f06') is not None:
        raise SystemExit('HP job binding required')
    source = load_source(cfg['data']['root'], cfg['data']['identity_path'], True, True,
                         cfg['data']['expected_count'])
    manifest = build_manifest(source, cfg['data']['split'])
    expected_fit = 46256 + (5849 if job.get('f11') is not None else 0)
    if (len(manifest['fit_indices']) != expected_fit or len(manifest['dev_indices']) != 5140
            or manifest['counts']['dev']['positive'] != 167
            or set(manifest['fit_indices']) & set(manifest['dev_indices'])):
        raise SystemExit('manifest holdout counts differ from the registered 9:1 split')
    negatives = sum(not manifest['rows'][i]['label'] for i in manifest['fit_indices'])
    steps = resolve_steps_per_cycle(job, negatives, fresh_spec={'purpose': 'train'})
    if steps != SCHEDULE['steps_per_cycle']:
        raise SystemExit('registered HP steps-per-cycle differs')
    hashes, digest = train.source_identity()
    plan_path = Path(args.plan_out)
    plan_value = {'format': 'hp-fresh-plan-v1', 'round_id': cfg['run_name'],
                  'config_sha256': sha(args.config),
                  'warm_start_tensor_sha256': job['initialization']['tensor_sha256'],
                  'schedule': dict(SCHEDULE)}
    plan_sha = write_new(plan_path, plan_value)
    fresh = {'format': 'glaucoma-fresh-init-v1', 'purpose': 'train',
             'plan_version': 'hp-hardpos-fit-only-v1', 'plan_sha256': plan_sha,
             'source_digest': digest, 'job': job_contract(job),
             'schedule': dict(SCHEDULE), 'execution_policy': policy,
             'ema': copy.deepcopy(EMA_RULE), 'precision_protocol': copy.deepcopy(PRECISION_RULE),
             'initialization': validate_initialization(job['initialization']),
             'amp_initial_scale': 1.0, 'cache_readonly_required': True}
    identity = {'format': 'glaucoma-retrain-v1', 'source_digest': digest,
                'manifest_digest': canonical_digest(manifest), 'job': job_contract(job),
                'initialization': fresh['initialization'], 'phase_parent_sha256': None,
                'distributed': plan, 'fresh': fresh}
    validate_fresh(fresh, identity, job, plan, steps, mode='train')
    fresh_path = Path(args.fresh_out)
    fresh_sha = write_new(fresh_path, fresh)
    receipt = {'status': 'hp_fresh_spec_validated_not_launched', 'mode': 'hp' if job.get('f05_targets') else 'ohem',
               'config': str(args.config), 'config_sha256': sha(args.config),
               'plan': str(plan_path), 'plan_sha256': plan_sha,
               'fresh_spec': str(fresh_path), 'fresh_spec_sha256': fresh_sha,
               'source_digest': digest, 'manifest_digest': identity['manifest_digest'],
               'fit_rows': len(manifest['fit_indices']), 'dev_rows': len(manifest['dev_indices']),
               'dev_positive': manifest['counts']['dev']['positive'], 'negatives': negatives,
               'steps_per_cycle': steps, 'topology': plan,
               'GPU_initialized': False, 'official_read': False, 'test_labels_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(args.receipt, receipt)}, indent=1))


def cmd_site(args):
    sys.path.insert(0, str(bundle_path()))
    sys.path.insert(0, str(Path(args.source_root) / 'tools' / 'activation'))
    from prepare_gpu_site import partition_cpus, verify_native_guard
    from retrain.site_binding import (live_cpu_resources, load_resource_policy, sha256,
                                      validate_gpu_allocation, validate_site,
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
    site['science']['binding_version'] = 'hp-hardpos-v1'
    site['science']['configuration'] = {'path': args.science_config, 'sha256': sha(args.science_config)}
    site_path = output / 'SITE_BINDING.json'
    validate_site(site)
    validate_gpu_allocation(site, grant, resources, initial=True, now=created)
    print(json.dumps({'status': 'hp_gpu_site_written', 'site': str(site_path),
                      'site_sha256': write_new(site_path, site),
                      'allocation': str(grant_path), 'allocation_sha256': sha(grant_path),
                      'effective_cpus': policy_value['effective_cpus'],
                      'owner_mask': policy_value['owner_mask'],
                      'rank_masks': policy_value['rank_masks'],
                      'control_mask': policy_value['control_mask'],
                      'validation_mask': policy_value['validation_mask'],
                      'native_guard': native}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    config = sub.add_parser('config')
    config.add_argument('--mode', choices=('hp', 'ohem', 'f11'), required=True)
    config.add_argument('--round-id', required=True)
    config.add_argument('--round-root', required=True)
    config.add_argument('--data-root', required=True)
    config.add_argument('--identity-path', required=True)
    config.add_argument('--warmstart', required=True)
    config.add_argument('--age-contract', required=True)
    config.add_argument('--f05-summary', default=None)
    config.add_argument('--f05-verification', default=None)
    config.add_argument('--f05-target', default=None)
    config.add_argument('--output', required=True)
    spec = sub.add_parser('spec')
    spec.add_argument('--config', required=True)
    spec.add_argument('--site', required=True)
    spec.add_argument('--plan-out', required=True)
    spec.add_argument('--fresh-out', required=True)
    spec.add_argument('--receipt', required=True)
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
    args = parser.parse_args()
    {'config': cmd_config, 'spec': cmd_spec, 'site': cmd_site}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

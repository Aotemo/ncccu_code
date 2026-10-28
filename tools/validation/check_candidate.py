"""Production Tiny22K binary+age CPU admission; no data forward or formal state."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('explicit_prepare', HERE.parent / 'activation/prepare.py')
build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
CHECKS = []


def group_lrs_match(groups, job):
    expected = {'backbone': job['backbone_lr'], 'head': job['head_lr'],
                'age': job['age']['contract']['lr']}
    return all(group['role'] in expected and group['lr'] == group['initial_lr'] == expected[group['role']]
               for group in groups)


def check(name, condition, actual=None):
    CHECKS.append({'name': name, 'passed': bool(condition), 'actual': actual})
    if not condition:
        raise ValueError(name)


def rejected(name, operation):
    try:
        operation()
    except ValueError as error:
        check(name, True, str(error))
    else:
        check(name, False)


def create_cpu_fresh_model(job):
    """Use the production seed/precision/factory path for CPU initial identity."""
    import torch
    from retrain.engine import _seed
    from retrain.models import create_model
    from retrain.precision import RULE, configure
    configure(RULE); _seed(job['seed'], torch.device('cpu'))
    return create_model(job, initialize=True)


def main(args):
    package, cfg, job, topology, policy, digest = build.build(args)
    from retrain.site_binding import load_site, require_cpu_channel
    site = load_site(); require_cpu_channel(site)
    import torch
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS'])); torch.set_num_interop_threads(1)
    check('CPU_only_before_model', not torch.cuda.is_initialized())
    from retrain import experiment91 as ex
    from retrain.engine import _optimizer, _lr_factor
    from retrain.ema import EMA, tensor_digest, validate_state
    from retrain.config import job_contract
    from retrain.fresh import validate_fresh
    from retrain.precision import RULE, configure, runtime_proof
    from retrain.persistence import CheckpointStore, canonical_digest
    sys.path.insert(0, str(package))
    import r92_initial_gate as initial_gate
    model = create_cpu_fresh_model(job)
    structure = ex.describe_model(model); public = model.public_initialization_receipt
    check('public1000_strict_exact_finite_FP32_then_fresh_binary', bool(public)
          and public['all_public_tensors_exact'] and public['all_public_tensors_FP32_finite']
          and public['public_tensor_count'] == 182 and public['full_head_shape'] == [1000, 768]
          and not public['strict_missing_keys'] and not public['strict_unexpected_keys'], public)
    check('actual_LN23_GELU18_native_avgpool_noBN_noq_binary_agezero',
          len(structure['layer_norms']) == 23 and all(row['eps'] == 1e-6 for row in structure['layer_norms'])
          and structure['batch_norm_modules'] == structure['buffer_count'] == 0
          and not dict(model.named_buffers()) and model.age_metadata == job['age']
          and model.age_weight.numel() == 2 and bool((model.age_weight == 0).all())
          and sum(isinstance(module, torch.nn.GELU) and module.approximate == 'none'
                  for module in model.modules()) == 18, structure)
    wrong = copy.deepcopy(job); wrong['final_norm_eps'] = 1e-12
    rejected('old_R92_final_LN_job_rejected', lambda: ex.validate_job(wrong))
    model.network.classifier[0].eps = 1e-12
    rejected('wrong_actual_final_LN_rejected', lambda: ex.describe_model(model))
    model.network.classifier[0].eps = 1e-6
    optimizer = _optimizer(model, job)
    named = dict(model.named_parameters()); names = {id(parameter): name for name, parameter in named.items()}
    groups = [{'role': group['role'], 'lr': group['lr'], 'initial_lr': group['initial_lr'],
        'weight_decay': group['weight_decay'], 'parameters': [
            {'name': names[id(parameter)], 'shape': list(parameter.shape), 'numel': parameter.numel()}
            for parameter in group['params']]} for group in optimizer.param_groups]
    members = [parameter for group in optimizer.param_groups for parameter in group['params']]
    roles = {role: sum(row['numel'] for group in groups if group['role'] == role for row in group['parameters'])
             for role in ('backbone', 'head', 'age')}
    head = {id(parameter) for parameter in model.network.classifier[-1].parameters()}
    check('unique_groups_emptyAdam_last_linear_LN_backbone', not optimizer.state
        and len(named) == len(members) == len({id(parameter) for parameter in members}) == 183
        and {id(parameter) for parameter in members} == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        and roles == {'backbone': 27820128, 'head': 769, 'age': 2}
        and {id(parameter) for group in optimizer.param_groups if group['role'] == 'head' for parameter in group['params']} == head
        and any(row['name'] == 'network.classifier.0.weight' for group in groups if group['role'] == 'backbone' for row in group['parameters'])
        and group_lrs_match(groups, job)
        and all(group['weight_decay'] == (1e-4 if group['role'] == 'age' or any(
            named[row['name']].ndim > 1 and not row['name'].endswith('.bias') for row in group['parameters']) else 0.) for group in groups),
        {'trainable_tensors': len(named), 'elements_by_role': roles, 'group_count': len(groups)})
    model_digest = tensor_digest(model.state_dict()); ema = EMA(model, ex.ema_rule())
    check('initial_EMA_exact_and_outside_optimizer', tensor_digest(ema.model.state_dict()) == model_digest
          and model_digest == ex.science_configuration()['initial_model_tensor_sha256']
          and not {id(parameter) for parameter in ema.model.parameters()} & {id(parameter) for parameter in members})
    fresh = build.make_fresh(package, job, topology, policy, digest, public['initialization'])
    identity = {'format': 'glaucoma-retrain-v1', 'source_digest': digest,
        'manifest_digest': ex.binding()['manifest_digest'], 'job': job_contract(job),
        'initialization': public['initialization'], 'phase_parent_sha256': None,
        'distributed': topology, 'fresh': fresh, 'experiment91': ex.binding()}
    schedule = ex.schedule()
    total, warmup = schedule['total_updates'], schedule['warmup_updates']
    validate_fresh(fresh, identity, job, topology, schedule['steps_per_cycle'], mode='train')
    check('exact_fixed_science_and_eight_rank_fresh', ex.validate_runtime(identity, job, fresh=fresh)
          and fresh['schedule'] == schedule
          and (topology['world_size'], topology['per_rank_batch'], topology['accumulation']) == (8, 8, 1))
    expected = {'source_digest': digest, 'candidate': ex.candidate(), 'model_tensor_sha256': model_digest,
        'batch_norm_modules': 0, 'buffer_count': 0, 'buffer_sha256': structure['buffer_sha256'],
        'candidate_normalization': structure, 'optimizer_state_entries': 0, 'trainable_tensors': len(named),
        'parameter_elements_by_role': roles, 'no_attention_q': True}
    initial = {'format_version': 1, 'cpu_fixture_only': True, 'identity': identity,
        'job': job, 'mode': 'train', 'age': job['age'], 'manifest_digest': ex.binding()['manifest_digest'],
        'fresh': fresh, 'successful_updates': 0, 'attempted_updates': 0, 'evaluated_cycle': 0,
        'schedule': {**ex.schedule(), 'successful_updates': 0}, 'model': model.state_dict(),
        'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': {},
        'candidate_normalization': structure}
    store = CheckpointStore(args.output / 'CPU-FIXTURE-NOT-FOR-TRAINING/initial-checkpoint')
    checkpoint = store.save(initial); loaded = store.load('last', map_location='cpu')
    ex.validate_checkpoint(loaded); validate_state(loaded['ema'], loaded['model'], 0, ex.ema_rule())
    with torch.no_grad():
        model.network.classifier[-1].bias.add_(1.)
    result = model.load_state_dict(loaded['model'], strict=True)
    check('CPU_initial_checkpoint_strict_roundtrip_emptyAdam', not result.missing_keys and not result.unexpected_keys
          and tensor_digest(model.state_dict()) == model_digest and loaded['optimizer']['state'] == {}
          and loaded['successful_updates'] == loaded['attempted_updates'] == 0 and loaded['cpu_fixture_only'] is True,
          {'checkpoint': str(checkpoint), 'sha256': build.sha(checkpoint), 'fixture_only': True})
    initials = [{'rank': rank, 'model_sha256': model_digest, 'optimizer_state_entries': 0,
        'bn_modules': 0, 'scaler': {}, 'precision_runtime': {**runtime_proof(RULE, device='cpu'),
            'scaler_enabled': False, 'scaler_state_dict': {}, 'effective_scale': 1.0},
        'candidate_normalization': structure, 'cpu_fixture_only': True} for rank in range(8)]
    check('initial_gate_eight_rank_coverage_CPU_fixture_only',
        initial_gate.verify_records(initials, groups, groups, expected, expected_world=8) == model_digest)
    rejected('initial_gate_missing_rank_rejected', lambda: initial_gate.verify_records(initials[:-1], groups, groups, expected, expected_world=8))
    wrong_initial = copy.deepcopy(initials); wrong_initial[0]['model_sha256'] = '0' * 64
    rejected('initial_gate_wrong_tensor_rejected', lambda: initial_gate.verify_records(wrong_initial, groups, groups, expected, expected_world=8))
    factors = {str(update): _lr_factor(update, total, warmup)
               for update in (0, warmup - 1, warmup, warmup + 1, total)}
    check('LR_successful_update_boundaries', factors['0'] == 1 / warmup
          and factors[str(warmup - 1)] == factors[str(warmup)] == 1 and factors[str(warmup + 1)] < 1
          and math.isclose(factors[str(total)], .01, abs_tol=1e-15), factors)
    scalar = torch.nn.Linear(1, 1, bias=False); scalar.weight.data.zero_()
    scalar_ema = EMA(scalar, ex.ema_rule()); boundaries = {}
    for update in range(1, total + 1):
        scalar.weight.data.fill_(float(update)); scalar_ema.update(scalar, update)
        if update in (warmup - 1, warmup, warmup + 1, total):
            state = scalar_ema.state_dict(); validate_state(state, scalar.state_dict(), update, ex.ema_rule())
            boundaries[str(update)] = {key: value for key, value in state.items() if key != 'model'}
            if update <= warmup:
                check('EMA_warmcopy_at_' + str(update), torch.equal(scalar.weight, scalar_ema.model.weight))
            elif update == warmup + 1:
                check('EMA_first_blend', math.isclose(scalar_ema.model.weight.item(), warmup + 1 - ex.ema_rule()['decay'], abs_tol=.0002))
    check('EMA_successful_clock_counts_at_maximum', boundaries[str(total)]['copy_count'] == warmup
          and boundaries[str(total)]['blend_count'] == total - warmup
          and boundaries[str(total)]['update_count'] == total, boundaries)
    check('CPU_only_after_all_checks', not torch.cuda.is_initialized())
    result = {'status': 'completed', 'exit_code': 0, 'scope': 'production_binary_age_CPU_only',
        'science_configuration': site['science']['configuration'],
        'checks': CHECKS, 'source_digest': digest, 'candidate': ex.candidate(), 'initial_state_control': expected,
        'job_contract': job_contract(job), 'topology': topology, 'site_binding_sha256': build.sha(package / 'SITE_BINDING.json'),
        'source_files': build.read(package / 'SOURCE-MANIFEST.reference.json')['after'],
        'python': sys.executable, 'pod_uid': os.environ['NCCCU_POD_UID'], 'cpu_mask': sorted(os.sched_getaffinity(0)),
        'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(), 'gpu_initialized': False,
        'cpu_allocation': os.environ['NCCCU_CPU_ALLOCATION'], 'cpu_allocation_sha256': os.environ['NCCCU_CPU_ALLOCATION_SHA256'],
        'asset_receipt': str(args.asset_receipt), 'asset_receipt_sha256': args.asset_receipt_sha256,
        'training_updates': 0, 'data_images_read': False, 'official_semantics_read': False,
        'production_barrier_released': False, 'formal_clock_created': False, 'training_lock_acquired': False}
    original = args.output / 'CPU-RECEIPT.json'; build.write(original, result)
    evidence = package / 'evidence'
    build.write(package / 'FRESH-SPEC.template.json', fresh)
    build.write(evidence / 'CPU_PARAMETER_GROUPS.json', groups)
    build.write(evidence / 'FRESH_EMA_CONTRACT.json', {'rule': ex.ema_rule()})
    build.write(evidence / 'MODEL_CPU_ADMISSION.template.json', {**result,
        'formal_train_config_sha256': None, 'original_receipt_path': str(original), 'original_receipt_sha256': build.sha(original)})
    files = {str(path.relative_to(package)): build.sha(path) for path in sorted(package.rglob('*')) if path.is_file()}
    manifest = args.output / 'PACKAGE-MANIFEST.json'
    build.write(manifest, {'schema_version': 1, 'formal_root': site['round_root'], 'package_root': str(package),
        'git_commit': site['git_commit'], 'files': files, 'bundle_source_digest': digest,
        'activate_script_sha256': build.sha(args.output / 'activate.py')})
    prep = {'status': 'prepared_no_formal_clock_no_GPU', 'exit_code': 0, 'source_digest': digest,
        'initial_state_control': expected, 'cpu_receipt_path': str(original), 'cpu_receipt_sha256': build.sha(original),
        'package_manifest_sha256': build.sha(manifest), 'package_root': str(package), 'formal_root': site['round_root'],
        'source_state': site.get('source_state'), 'GPU_started': False, 'formal_clock_created': False}
    build.write(args.output / 'CPU-PREP-RECEIPT.json', prep)
    print(json.dumps({'status': result['status'], 'checks': len(CHECKS), 'passed': sum(row['passed'] for row in CHECKS),
        'cpu_receipt': str(original), 'cpu_receipt_sha256': build.sha(original),
        'prep_receipt': str(args.output / 'CPU-PREP-RECEIPT.json'), 'prep_receipt_sha256': build.sha(args.output / 'CPU-PREP-RECEIPT.json'),
        'initial_tensor_sha256': model_digest, 'GPU_started': False}, indent=2))


if __name__ == '__main__':
    arguments = build.parser().parse_args()
    started = time.time()
    try:
        main(arguments)
    except BaseException as error:
        if arguments.output.is_dir():
            build.write(arguments.output / 'FAILURE.json', {'status': 'failed', 'exit_code': 1,
                'checks': CHECKS, 'error': repr(error), 'traceback': traceback.format_exc(), 'wall_seconds': time.time() - started})
        raise

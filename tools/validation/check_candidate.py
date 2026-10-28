"""One affected Tiny CPU runner; fixtures never authorize a production barrier."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / 'result'
CHECKS = []
GROUP = ''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def check(name, condition, actual=None):
    CHECKS.append({'group': GROUP, 'name': name, 'passed': bool(condition), 'actual': actual})
    assert condition, name


def rejected(name, fn):
    try:
        fn()
    except ValueError as error:
        check(name, True, {'error': str(error)})
    else:
        check(name, False)


def main():
    global GROUP
    assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
    assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
    assert all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    assert not torch.cuda.is_initialized()
    import prepare_r92 as build
    build.main()
    package = HERE / 'package'
    sys.path[:0] = [str(package / 'bundle'), str(package)]
    from retrain import experiment91 as ex, single91 as gate
    from retrain.config import load_config, job_contract
    from retrain.execution_policy import resolve_topology
    from retrain.models import create_model
    from retrain.engine import _optimizer, _seed
    from retrain.ema import EMA, tensor_digest, validate_state, commit_pair, FROZEN_BN_RULE
    from retrain.persistence import CheckpointStore, atomic_json, canonical_digest
    from retrain.precision import RULE, configure, runtime_proof
    import r92_initial_gate as initial_gate
    source_map = read(package / 'SOURCE-MANIFEST.reference.json')
    source = source_map['source_digest']
    assert canonical_digest(source_map['after']) == source
    assert all(sha(package / 'bundle' / r) == h for r, h in source_map['after'].items())
    cfg = load_config(package / 'TRAIN_CONFIG.json', environ={})
    job, topology, policy = resolve_topology(cfg, available_accelerators=4)
    fresh = read(package / 'FRESH-SPEC.template.json'); identity = read(HERE / 'CPU-IDENTITY.json')
    GROUP = '1_public_Tiny_contract_factory_groups_checkpoint'
    check('candidate2_runtime_same9010_and_new_noBN_rule', ex.validate_runtime(identity, job, fresh=fresh)
        and ex.ema_rule()['warmcopy_updates'] == ex.schedule()['warmup_updates'] == 1600
        and 'frozen_bn_buffers' not in ex.ema_rule())
    wrong = copy.deepcopy(job); wrong['experiment91'] = read(build.PREVIOUS / 'TRAIN_CONFIG.json')['jobs'][0]['experiment91']
    rejected('R50_candidate_binding_rejected', lambda: ex.validate_job(wrong))
    wrong = copy.deepcopy(job); wrong['final_norm_eps'] = 1e-6
    rejected('wrong_final_LN_job_rejected', lambda: ex.validate_job(wrong))
    _seed(job['seed'], torch.device('cpu')); configure(RULE)
    model = create_model(job, initialize=True)
    normalization = ex.describe_model(model)
    public = model.public_initialization_receipt
    check('actual_public1000_strict_exact_FP32_finite_then_binary', bool(public)
        and public['all_public_tensors_exact'] and public['all_public_tensors_FP32_finite']
        and public['full_head_shape'] == [1000, 768] and not public['strict_missing_keys'] and not public['strict_unexpected_keys'], public)
    write(OUT / 'PUBLIC-STRICT.json', public)
    write(OUT / 'ACTUAL-TINY-STRUCTURE.json', normalization)
    check('native_avgpool_768_noq_zeroBN_actual_LN_SD_and_zeroage', normalization['batch_norm_modules'] == 0
        and normalization['buffer_count'] == 0 and model.age_metadata == job['age']
        and bool((model.age_weight == 0).all()), normalization)
    model.network.classifier[0].eps = 1e-6
    rejected('wrong_actual_LN_rejected', lambda: ex.describe_model(model))
    model.network.classifier[0].eps = 1e-12
    parameters = [{'name': n, 'shape': list(p.shape), 'numel': p.numel(), 'dtype': str(p.dtype), 'requires_grad': p.requires_grad}
                  for n, p in model.named_parameters()]
    write(OUT / 'PARAMETERS.json', parameters)
    optimizer = _optimizer(model, job)
    names = {id(p): n for n, p in model.named_parameters()}
    groups = [{'role': g['role'], 'lr': g['lr'], 'initial_lr': g['initial_lr'], 'weight_decay': g['weight_decay'],
               'parameters': [{'name': names[id(p)], 'shape': list(p.shape), 'numel': p.numel()} for p in g['params']]}
              for g in optimizer.param_groups]
    write(OUT / 'PARAMETER-GROUPS.json', groups)
    write(package / 'evidence/CPU_PARAMETER_GROUPS.json', groups)
    members = [p for g in optimizer.param_groups for p in g['params']]
    role_elements = {role: sum(p['numel'] for g in groups if g['role'] == role for p in g['parameters']) for role in ('backbone', 'head', 'age')}
    named = dict(model.named_parameters())
    expected_head = {id(p) for p in model.network.classifier[-1].parameters()}
    group_ok = (not optimizer.state and len(members) == len({id(p) for p in members}) == len(parameters)
        and {id(p) for p in members} == {id(p) for p in model.parameters() if p.requires_grad}
        and all(g['initial_lr'] == (2e-4 if g['role'] == 'backbone' else 1e-3) for g in groups)
        and all(g['weight_decay'] == (1e-4 if g['role'] == 'age' or any(named[p['name']].ndim > 1 and not p['name'].endswith('.bias') for p in g['parameters']) else 0.) for g in groups)
        and {id(p) for g in optimizer.param_groups if g['role'] == 'head' for p in g['params']} == expected_head
        and any(p['name'] == 'network.classifier.0.weight' for g in groups if g['role'] == 'backbone' for p in g['parameters']))
    check('actual_unique_groups_emptyAdam_lastlinear_LN_backbone', group_ok,
          {'trainable_tensors': len(parameters), 'elements_by_role': role_elements, 'groups': len(groups)})
    initial_tensor = tensor_digest(model.state_dict()); ema = EMA(model, ex.ema_rule())
    check('fresh_online_EMA_same_and_disjoint_optimizer', tensor_digest(ema.model.state_dict()) == initial_tensor
        and not {id(p) for p in ema.model.parameters()} & {id(p) for p in members})
    expected = {'source_digest': source, 'candidate': ex.candidate(), 'model_tensor_sha256': initial_tensor,
        'batch_norm_modules': 0, 'buffer_count': 0, 'buffer_sha256': normalization['buffer_sha256'],
        'candidate_normalization': normalization, 'optimizer_state_entries': 0,
        'trainable_tensors': len(parameters), 'parameter_elements_by_role': role_elements, 'no_attention_q': True}
    write(OUT / 'INITIAL-STATE.json', expected)
    fixture = HERE / 'CPU-FIXTURE-NOT-FOR-TRAINING'; fixture.mkdir()
    s = ex.schedule()
    payload = {'format_version': 1, 'cpu_fixture_only': True, 'identity': identity, 'job': job, 'mode': 'train',
        'age': copy.deepcopy(job['age']), 'manifest_digest': ex.binding()['manifest_digest'], 'fresh': fresh,
        'successful_updates': 0, 'attempted_updates': 0, 'schedule': {**s, 'successful_updates': 0},
        'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': {},
        'candidate_normalization': normalization, 'evaluated_cycle': 0}
    initial_store = CheckpointStore(fixture / 'initial-checkpoint')
    initial_path = initial_store.save(payload)
    loaded = initial_store.load('last'); ex.validate_checkpoint(loaded)
    check('initial_checkpoint_roundtrip_emptyAdam_newage_tensor', tensor_digest(loaded['model']) == initial_tensor
        and loaded['optimizer']['state'] == {} and loaded['age'] == job['age'] and loaded['candidate_normalization'] == normalization,
        {'blob_sha256': sha(initial_path), 'fixture': True})
    del loaded

    GROUP = '2_Tiny_noBN_EMA_and_initial_gate'
    ema.update(model, 1)
    state = ema.state_dict()
    check('new_rule_single_EMA_update_and_state_validation', validate_state(state, model.state_dict(), 1, ex.ema_rule()) is state
        and state['update_count'] == state['copy_count'] == 1 and state['blend_count'] == 0
        and not dict(model.named_buffers()) and not dict(ema.model.named_buffers()))
    corrupted = {**state, 'rule': FROZEN_BN_RULE}
    rejected('R50_EMA_rule_rejected_for_Tiny_checkpoint', lambda: validate_state(corrupted, model.state_dict(), 1, ex.ema_rule()))
    fake = 'network.fake.running_mean'
    badstate = {**state, 'model': {**state['model'], fake: torch.zeros(1)}}
    badreference = {**model.state_dict(), fake: torch.zeros(1)}
    rejected('noBN_validator_rejects_BN_even_matching_keys', lambda: validate_state(badstate, badreference, 1, ex.ema_rule()))
    saved_age = model.age_metadata; model.age_metadata = {**saved_age, 'contract_sha256': '0' * 64}
    rejected('immutable_age_preserved_for_Tiny_EMA', lambda: ema.update(model, 2))
    model.age_metadata = saved_age
    initials = [{'rank': r, 'model_sha256': initial_tensor, 'optimizer_state_entries': 0, 'bn_modules': 0,
                 'scaler': {}, 'precision_runtime': runtime_proof(RULE, device=torch.device('cpu')),
                 'candidate_normalization': normalization, 'cpu_fixture_only': True} for r in range(4)]
    check('noBN_gate_actual_CPU_reference_positive_fixture', initial_gate.verify_records(initials, groups, groups, expected) == initial_tensor
        and [p.name for p in initial_gate.required_paths(fixture)] == ['initial-ranks.json', 'parameter-groups.json', 'index.json'],
        {'four_rank_fixture_only': True, 'production_barrier_released': False})
    wrong_initials = copy.deepcopy(initials); wrong_initials[0]['model_sha256'] = '0' * 64
    rejected('initial_gate_wrong_tensor_rejected', lambda: initial_gate.verify_records(wrong_initials, groups, groups, expected))

    GROUP = '3_Tiny_exact_protection_freeze_official_boundary'
    # Exercise the newly enabled Tiny evaluation branch with synthetic tensors,
    # real Tiny model and its exact digest. No dataset images or official reads.
    from retrain import parallel_evaluation as pe
    class FixtureDataset:
        def __init__(self, root, rows, size, **kwargs): self.rows = rows; self.size = size
        def __getitem__(self, index):
            return torch.zeros(3, self.size, self.size), float(self.rows[index]['label']), index, torch.zeros(2)
    class Context:
        rank = 0; world_size = 1; primary = True
        def values(self, value): return [value]
        def broadcast(self, value): return value
    synthetic_rows = [{'name': 'CPU-negative', 'label': 0}, {'name': 'CPU-positive', 'label': 1}]
    original_dataset = pe.ImageDataset; pe.ImageDataset = FixtureDataset
    eval_job = {**job, 'eval_workers': 0}
    tiny_csv = fixture / 'tiny-branch.csv'
    try:
        metric = pe.evaluate(model, synthetic_rows, [0, 1], '/CPU-FIXTURE-NOT-DATA', eval_job, torch.device('cpu'), Context(), tiny_csv,
            variant='online', state_identity={'successful_updates': 0, 'attempted_updates': 0})
    finally:
        pe.ImageDataset = original_dataset
    check('Tiny_parallel_evaluation_exact_error_prediction_tensor', type(metric.get('error_twice')) is int
        and metric.get('prediction_sha256') == sha(tiny_csv) and metric.get('actual_tensor_sha256') == initial_tensor, metric)
    rows = ex.metadata()['manifest.json']['rows']
    dev_rows = [rows[i] for i in ex.metadata()['manifest.json']['dev_indices']]
    for cycle in range(1, 9):
        for variant in ('online', 'ema'):
            path = fixture / 'predictions' / variant / f'dev-cycle-{cycle:04d}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['name', 'label', 'logit', 'probability']); writer.writeheader()
                for row in dev_rows:
                    logit = float(row['label']) if cycle == 1 else 0.
                    writer.writerow({'name': row['name'], 'label': row['label'], 'logit': logit, 'probability': .5})
            atomic_json(str(path) + '.precision.json', {'variant': variant, 'view': 'identity', 'cpu_fixture_only': True,
                'state_identity': {'successful_updates': cycle * s['steps_per_cycle'], 'attempted_updates': cycle * s['steps_per_cycle']},
                'actual_tensor_sha256': initial_tensor})
    entries = gate.collect_dev_entries(fixture)
    pair1 = {v: {k: e[k] for k in ('auroc', 'ap', 'error_twice', 'prediction_sha256', 'tensor_sha256')}
             for v in ('online', 'ema') for e in entries if e['cycle'] == 1 and e['variant'] == v}
    transaction = {'evaluated_cycle': 0, 'best_metrics': None, 'ema_best_metrics': None,
                   'early_stop': {'reference_auroc': None, 'bad_cycles': 0, 'stopped': False}}
    committed, improved = commit_pair(transaction, pair1, 1, 800, job)
    # A misleading float AUROC must not beat a worse integer pair count.
    pair2 = {v: {**pair1[v], 'error_twice': 2, 'auroc': 1.0000000000000002} for v in ('online', 'ema')}
    second, improved2 = commit_pair(committed, pair2, 2, 1600, job)
    check('Tiny_commit_uses_exact_order_and_protects_both_variants', all(improved.values()) and not any(improved2.values())
        and second['best_metrics'] == committed['best_metrics'] and second['ema_best_metrics'] == committed['ema_best_metrics'])
    ema.successful = 800
    payload.update(successful_updates=800, attempted_updates=800, evaluated_cycle=1, schedule={**s, 'successful_updates': 800}, ema=ema.state_dict())
    store = CheckpointStore(fixture / 'checkpoints')
    selected_blob = store.save(payload, protect=tuple('best_' + v for v in ('online', 'ema') if improved[v]))
    later = {**payload, 'successful_updates': 1600, 'attempted_updates': 1600,
             'schedule': {**s, 'successful_updates': 1600}, 'ema': {**payload['ema'], 'update_count': 1600, 'copy_count': 1600, 'last_update_success': 1600}}
    store.save(later, protect=())
    check('both_best_blob_records_survive_later_unimproved_save', all(store._index()['protected']['best_' + v]['file'] == selected_blob.name for v in ('online', 'ema')) and selected_blob.is_file())
    atomic_json(fixture / 'complete.json', {'status': 'completed', 'cpu_fixture_only': True,
        'successful_updates': 6400, 'evaluated_cycle': 8, 'skipped_updates': 0})
    frozen_path = fixture / 'SINGLE-MODEL-FREEZE.json'
    frozen = gate.freeze_training(fixture, frozen_path, technical_fixture=True)
    reloaded, actual_job, origin, device = gate.load_frozen_model(frozen_path, 'cpu', allow_fixture=True)
    actual = {'variant': origin['variant'], 'tensor_sha256': tensor_digest(reloaded.state_dict())}
    gate.validate_frozen_checkpoint(frozen, payload, actual)
    check('order2_noBN_freeze_protected_EMA_strict_reload_actual_tensor', frozen['experiment_order'] == 2
        and frozen['variant'] == 'ema' and frozen['selected']['cycle'] == 1 and frozen['selected']['error_twice'] == 0
        and actual_job == job and actual['tensor_sha256'] == initial_tensor and ex.describe_model(reloaded) == normalization,
        {'fixture_only': True, 'selected': frozen['selected'], 'blob_sha256': frozen['blob_sha256'], 'actual': actual})
    for key in ('candidate', 'source', 'variant', 'LN'):
        bad = copy.deepcopy(frozen)
        if key == 'candidate': bad['experiment91']['candidate']['model'] = 'resnet50_imagenet'
        if key == 'source': bad['identity']['initialization']['weights_sha256'] = '0' * 64; bad['initialization'] = bad['identity']['initialization']
        if key == 'variant': bad['variant'] = 'online'
        if key == 'LN': bad['candidate_normalization']['layer_norms'][-1]['eps'] = 1e-6
        path = fixture / ('BAD-' + key + '.json'); gate.write_exclusive(path, bad)
        rejected('Tiny_official_identity_rejects_' + key, lambda path=path: gate.validate_freeze(path, allow_fixture=True))
    rejected('CPU_fixture_still_barred_before_official_reservation', lambda: gate.reserve_official(frozen_path, fixture / 'official-output'))
    del reloaded

    GROUP = '4_wrapper_templates_budget_no_GPU'
    import run_round as guard_module
    status_snapshot = {'at': time.time(), 'cards': guard_module.card_snapshot(), 'device_users': guard_module.device_users_before_start(),
        'training_admission': False, 'training_lock_acquired': False}
    write(OUT / 'GPU-READONLY-STATUS.json', status_snapshot)
    check('current_no_GPU_device_users_and_idle_four_cards', not status_snapshot['device_users']
        and len(status_snapshot['cards']) == 4 and all(c['busy_percent'] == 0 and c['vram_used_bytes'] < 64 * 1024**2 for c in status_snapshot['cards']), status_snapshot)
    owner = (package / 'run_training.py').read_text(); guard = (package / 'run_round.py').read_text()
    activation = (HERE / 'activate_r92.py').read_text(); gate_text = (package / 'r92_initial_gate.py').read_text()
    check('both_Tiny_initial_consumers_use_native_without_BN_file', 'bn-initial-ranks.json' not in owner + gate_text
        and 'from r92_initial_gate import' in guard and 'initial-checkpoint/index.json' in gate_text)
    clock = read(package / 'CLOCK_POLICY.json')
    check('policy_derived_full_tail_windows_and_R91_all_predecessors',
        clock['total_seconds'] == 12600 and clock['binding_seconds'] == 1800 and clock['train_hard_seconds'] == 7200
        and clock['freeze_hard_seconds'] == 300 and clock['official_hard_seconds'] == 1200 and clock['cleanup_seconds'] == 150
        and "clock['deadline_at']-1200" not in owner and "clock['deadline_at']-750" not in owner
        and "clock['deadline_at']-policy['freeze_hard_seconds']-policy['official_hard_seconds']-policy['cleanup_seconds']" in owner
        and "clock['deadline_at']-policy['official_hard_seconds']-policy['cleanup_seconds']" in owner
        and "for phase in ('train', 'freeze', 'official')" in activation and 'len(expected) == 8' in activation
        and 'rounds/R91-R50-IMAGENETV2-9010' in activation and 'N15-R50' not in activation)
    environment = read(HERE / 'environment/ENVIRONMENT-RECEIPT.json')
    historical = [c for c in environment['historical_Tiny_costs'] if c['job']['image_size'] == 512]
    budget = {'status': 'DRAFT_NO_FORMAL_CLOCK_OR_GPU_PROBE', 'policy': clock, 'historical_Tiny512': historical,
        'linear_update_scaled_seconds': [{'root': c['root'], 'seconds': c['cost']['wall_seconds_this_process'] / c['cost']['successful_updates'] * 6400} for c in historical],
        'limitations': 'Historical different split/attention/LR; no current throughput guarantee. Formal live Pod lease/resources remain required.',
        'T0': None, 'D': None, 'H': None, 'token': None, 'new_GPU_probe': False}
    write(OUT / 'BUDGET-DRAFT.json', budget)
    build_receipt = read(HERE / 'BUILD-RECEIPT.json')
    check('unchanged_owner_functions_and_strict_once_reservation_preserved', build_receipt['preserved_owner_function_bodies']
        and build_receipt['preserved_guard_function_bodies']
        and 'marker = Path(value[\'checkpoint\']).parent.parent / \'OFFICIAL-ONCE.json\'' in (package / 'bundle/retrain/single91.py').read_text())
    check('no_formal_clock_token_lock_or_GPU', not build.FORMAL.exists() and not (HERE / 'FORMAL-CLOCK.json').exists()
        and not torch.cuda.is_initialized(), {'T0': None, 'token': None, 'production_four_rank_gate_executed': False,
        'official_reads': 0, 'official_forwards': 0, 'training_lock_acquired': False})
    result = {'status': 'passed', 'exit_code': 0, 'checks': CHECKS, 'groups_completed': [1, 2, 3, 4],
        'source_digest': source, 'initial_state_control': expected, 'fixture_only': True, 'public_model_loads': 1,
        'checkpoint_model_constructed_initialize_false': 1, 'actual_EMA_updates': 1, 'CPU_optimizer_updates': 0,
        'official_data_read': False, 'official_forward_calls': 0, 'GPU_initialized': False,
        'formal_clock_created': False, 'training_lock_acquired': False, 'new_GPU_probe': False}
    write(OUT / 'R92-CPU-RECEIPT.json', result)
    write(package / 'evidence/MODEL_CPU_ADMISSION.template.json', {'status': 'completed', 'formal_train_config_sha256': None,
        'gpu_initialized': False, 'checks': CHECKS, 'source_digest': source, 'job_contract': job_contract(job), 'topology': topology,
        'initial_state_control': expected, 'original_receipt_path': str(OUT / 'R92-CPU-RECEIPT.json'),
        'original_receipt_sha256': sha(OUT / 'R92-CPU-RECEIPT.json'), 'scope': 'four affected Tiny groups; production native four-rank proof remains future'})
    files = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*')) if p.is_file()}
    write(package / 'RUNTIME_BINDING.template.json', {'round_id': build.ROUND, 'implementation_ready': False,
        'source_digest': source, 'bundle_source_files': source_map['after'], 'source_files': files,
        'public_weight_sha256': ex.WEIGHT_SHA, 'clock_sha256': None, 'freeze_sha256': None,
        'effective_output_root': str(build.FORMAL / 'outputs'), 'initial_state_control': expected,
        'fresh_spec_sha256': {'train_template': sha(package / 'FRESH-SPEC.template.json')}})
    files = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*')) if p.is_file()}
    write(HERE / 'PACKAGE-MANIFEST.json', {'formal_root': str(build.FORMAL), 'package_root': str(package), 'files': files,
        'bundle_source_digest': source, 'activate_script_sha256': sha(HERE / 'activate_r92.py')})
    prep = {'status': 'prepared_no_formal_clock_no_GPU', 'exit_code': 0, 'source_digest': source,
        'package_root': str(package), 'formal_root_proposed': str(build.FORMAL), 'cpu_receipt_sha256': sha(OUT / 'R92-CPU-RECEIPT.json'),
        'package_manifest_sha256': sha(HERE / 'PACKAGE-MANIFEST.json'), 'initial_state_control': expected,
        'CPU_checks_passed': len(CHECKS), 'actual_training_updates': 0, 'actual_official_forwards': 0,
        'production_four_rank_gate_executed': False, 'live_GPU_admission_run': False, 'instance_lock_acquired': False,
        'T0': None, 'token': None, 'clock_policy_draft': clock,
        'official_cache': str(Path(read(package / 'ASSETS.json')['prepared512_cache']['remote_root']) / 'valid/sealed'),
        'official_device': 'cuda:0', 'environment': environment['environment']}
    write(HERE / 'R92-PREP-RECEIPT.json', prep)
    return prep


if __name__ == '__main__':
    OUT.mkdir(); code = 1; started = time.time()
    try:
        receipt = main(); code = 0
    except Exception as error:
        receipt = {'status': 'failed', 'exit_code': 1, 'checks': CHECKS, 'error': repr(error), 'traceback': traceback.format_exc()}
        write(OUT / 'FAILURE.json', receipt); traceback.print_exc()
    (OUT / 'EXIT-CODE.txt').write_text(str(code) + '\n')
    write(OUT / 'OUTPUT-MANIFEST.json', {'exit_code': code, 'started_at': started, 'ended_at': time.time(),
        'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(HERE.rglob('*'))
                  if p.is_file() and p.suffix not in ('.pt', '.pth') and 'CPU-FIXTURE-NOT-FOR-TRAINING' not in str(p)],
        'large_fixture_checkpoint_retained_on_Pod': True})
    print(json.dumps({'status': receipt['status'], 'exit_code': code, 'checks': len(CHECKS), 'passed': sum(x['passed'] for x in CHECKS),
                      'result': str(OUT), 'formal_clock_created': False, 'GPU_started': False}), flush=True)
    sys.exit(code)

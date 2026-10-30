"""Actual admitted-world noBN initial proof; load native checkpoint, never a new model."""
import json
from pathlib import Path
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def required_paths(folder):
    return [Path(folder) / name for name in ('initial-ranks.json', 'parameter-groups.json', 'initial-checkpoint/index.json')]


def verify_records(initials, actual_groups, reference_groups, expected, *, expected_world):
    from retrain.experiment91 import candidate, validate_normalization
    require(expected['candidate'] == candidate()
            and expected['batch_norm_modules'] == candidate()['batch_norm_modules']
            and expected['buffer_count'] == candidate()['batch_norm_buffers'],
            'CPU expected candidate/noBN identity differs')
    require(type(expected_world) is int and expected_world == 8
            and len(initials) == expected_world and all(type(x.get('rank')) is int for x in initials)
            and {x['rank'] for x in initials} == set(range(expected_world)),
            'all admitted initial ranks required')
    for rank in initials:
        require(rank['model_sha256'] == expected['model_tensor_sha256'] and rank['optimizer_state_entries'] == 0
                and rank['bn_modules'] == expected['batch_norm_modules'] and rank['scaler'] == {} and not rank.get('attention_pool'), 'fresh tensor/Adam/BN/noq differs')
        require(rank.get('candidate_normalization') == expected['candidate_normalization'], 'actual rank LN/noBN differs')
        require(rank.get('parameter_trainability') == expected.get('parameter_trainability'),
                'actual rank frozen/trainable/optimizer identity differs')
        validate_normalization(rank['candidate_normalization'])
        precision = rank.get('precision_runtime', {})
        require(precision.get('precision') == 'fp32' and precision.get('effective_scale') == 1.0
                and precision.get('scaler_enabled') is False and precision.get('scaler_state_dict') == {}
                and precision.get('autocast_enabled') is False, 'rank FP32 differs')
    require(len(actual_groups) == len(reference_groups), 'parameter group count differs')
    names = set(); roles = {}
    for current, prior in zip(actual_groups, reference_groups):
        require({k: v for k, v in current.items() if k != 'lr'} == {k: v for k, v in prior.items() if k != 'lr'}, 'actual group members/shape/decay/peak differ')
        role = current['role']
        for p in current['parameters']:
            require(p['name'] not in names and p['name'] not in ('attention_pool_weight', 'q'), 'duplicate/q parameter')
            names.add(p['name']); roles[role] = roles.get(role, 0) + p['numel']
    require(len(names) == expected['trainable_tensors'] and roles == expected['parameter_elements_by_role'], 'actual Tiny parameter topology differs')
    return expected['model_tensor_sha256']


def maybe_release(root, atomic, sha):
    root = Path(root); cfg = read(root / 'TRAIN_CONFIG.json')
    evidence = root / 'evidence'; control = root / 'control'
    barrier_path = control / 'train-startup-barrier.json'; receipt_path = evidence / 'ACTUAL_FRESH_INITIAL.json'
    if barrier_path.exists():
        require(receipt_path.is_file(), 'barrier exists without native initial receipt')
        return True
    folder = root / 'outputs' / cfg['run_name'] / cfg['jobs'][0]['name']
    paths = required_paths(folder)
    if not all(p.is_file() for p in paths):
        return False
    sys.path.insert(0, str(root / 'bundle'))
    from retrain.site_binding import load_site, verify_rank_devices
    site = load_site(); world = len(site['gpus'])
    require(cfg['execution']['devices'] == ['cuda:' + str(rank) for rank in range(world)], 'configured ranks differ from site')
    verify_rank_devices(read(folder / 'rank-device-proof.json'), site)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'owner initial gate must remain CPU')
    from retrain.persistence import CheckpointStore
    from retrain.ema import tensor_digest, validate_state
    from retrain.experiment91 import validate_checkpoint, ema_rule
    expected = read(evidence / 'MODEL_CPU_ADMISSION.json')['initial_state_control']
    full_digest = verify_records(read(paths[0]), read(paths[1]), read(evidence / 'CPU_PARAMETER_GROUPS.json'), expected,
                                 expected_world=world)
    fresh = read(control / 'train-fresh-spec.json')
    source = read(root / 'SOURCE-MANIFEST.reference.json')['source_digest']
    require(time.time() < read(control / 'train-phase-start.json')['soft_deadline_at'], 'startup budget expired')
    store = CheckpointStore(folder / 'initial-checkpoint'); payload = store.load('last', map_location='cpu')
    validate_checkpoint(payload)
    require(not payload.get('cpu_fixture_only') and payload['successful_updates'] == payload['attempted_updates'] == 0
            and payload['optimizer']['state'] == {} and payload['scaler'] == {}, 'native initial not production public fresh')
    require(payload['fresh'] == fresh and payload['identity']['fresh'] == fresh
            and payload['identity']['source_digest'] == source == expected['source_digest']
            and payload['identity']['initialization'] == fresh['initialization'], 'native fresh source differs')
    state = payload['model']
    require('attention_pool_weight' not in state and 'q' not in state and int(torch.count_nonzero(state['age_weight'])) == 0, 'noq/zeroage state differs')
    require(tensor_digest(state) == full_digest and payload['candidate_normalization'] == expected['candidate_normalization'], 'native tensor/actual LN differs')
    require(payload.get('parameter_trainability') == expected.get('parameter_trainability'),
            'native frozen/trainable/optimizer identity differs')
    buffers = {k: v for k, v in state.items() if k.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))}
    require(len(buffers) == expected['buffer_count'] and tensor_digest(buffers) == expected['buffer_sha256'],
            'native immutable BN buffers differ')
    validate_state(payload['ema'], state, 0, ema_rule())
    require(payload['ema']['rule'] == fresh['ema'] == ema_rule() and tensor_digest(payload['ema']['model']) == full_digest, 'initial EMA differs')
    require(not torch.cuda.is_initialized(), 'gate initialized CUDA')
    record = read(paths[2])['last']
    receipt = {'status': 'completed', 'at': time.time(), 'rank_count': world, 'optimizer_state_entries': 0,
        'candidate': expected['candidate'], 'source_digest': source, 'model_tensor_sha256': full_digest,
        'batch_norm_modules': expected['batch_norm_modules'], 'buffer_count': expected['buffer_count'], 'buffer_sha256': expected['buffer_sha256'],
        'candidate_normalization': expected['candidate_normalization'], 'parameter_tensors': expected['trainable_tensors'],
        'parameter_elements_by_role': expected['parameter_elements_by_role'],
        'initial_files_sha256': {str(p.relative_to(root)): sha(p) for p in paths},
        'native_initial_blob': str(store.directory / record['file']), 'native_initial_blob_sha256': record['sha256'],
        'public_weights_reloaded_by_gate': False, 'new_model_created_by_gate': False, 'CUDA_initialized': False}
    if expected.get('parameter_trainability') is not None:
        receipt['parameter_trainability'] = expected['parameter_trainability']
    atomic(receipt_path, receipt, exclusive=True)
    barrier = fresh['startup_barrier']
    atomic(barrier_path, {'round_id': barrier['round_id'], 'plan_sha256': fresh['plan_sha256'], 'source_digest': source,
        'purpose': 'train', 'run_ids': barrier['run_ids'], 'world_size': world, 'initial_rank_count': world, 'all_initials_passed': True,
        'rank_initial_proof_sha256': sha(paths[0]), 'initials_sha256': {cfg['run_name']: sha(paths[0])},
        'actual_initial_receipt_sha256': sha(receipt_path)}, exclusive=True)
    return True

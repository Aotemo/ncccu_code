"""Consume this run's native four-rank zero-step evidence; never construct a model."""
import json
from pathlib import Path
import sys
import time

SOURCE = '76cb37a6cd69238d6269c02e019c79a3349b58fad017f9a71eca867dba624b01'
INITIAL = 'b88683d2cd788e74127e86225bed1cbbbec2fbe3baa56e3b172a33b8ef92ea96'
PUBLIC_BN = '21b9ef124340ced1d13ffe0ea1ef8214f800c05ea0f7bdb9572f06d26475b210'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def verify_records(initials, bn, actual_groups, reference_groups, expected):
    require(expected['model_tensor_sha256'] == INITIAL and expected['expected_frozen_bn_sha256'] == PUBLIC_BN, 'CPU expected initial identity differs')
    require(len(initials) == 4 and {x['rank'] for x in initials} == {0, 1, 2, 3}, 'four initial ranks required')
    require(len(bn) == 4 and {x['rank'] for x in bn} == {0, 1, 2, 3}, 'four BN ranks required')
    require(all(x['online'] == x['ema'] == PUBLIC_BN for x in bn), 'public fixed BN differs')
    for rank in initials:
        require(rank['model_sha256'] == INITIAL and rank['optimizer_state_entries'] == 0
                and rank['bn_modules'] == 53 and rank['scaler'] == {} and not rank.get('attention_pool'), 'fresh tensor/Adam/BN/no-q differs')
        precision = rank.get('precision_runtime', {})
        require(precision.get('precision') == 'fp32' and precision.get('effective_scale') == 1.0
                and precision.get('scaler_enabled') is False and precision.get('scaler_state_dict') == {}
                and precision.get('autocast_enabled') is False, 'rank FP32 differs')
    require(len(actual_groups) == len(reference_groups), 'parameter group count differs')
    names = set(); roles = {}
    for current, prior in zip(actual_groups, reference_groups):
        require({k: v for k, v in current.items() if k != 'lr'} == {k: v for k, v in prior.items() if k != 'lr'}, 'group members/shape/decay/peak differ')
        role = current['role']
        require(current['initial_lr'] == (2e-5 if role == 'backbone' else 1e-3), 'actual peak LR differs')
        for parameter in current['parameters']:
            require(parameter['name'] not in names and parameter['name'] != 'attention_pool_weight', 'duplicate or q parameter')
            names.add(parameter['name'])
            roles[role] = roles.get(role, 0) + parameter['numel']
    require(len(names) == 162 and roles == {'backbone': 23508032, 'head': 2049, 'age': 2}, 'ordinary R50 parameter topology differs')
    return INITIAL


def maybe_release(root, atomic, sha):
    root = Path(root); cfg = read(root / 'TRAIN_CONFIG.json')
    evidence = root / 'evidence'; control = root / 'control'
    barrier_path = control / 'train-startup-barrier.json'
    receipt_path = evidence / 'ACTUAL_FRESH_INITIAL.json'
    if barrier_path.exists():
        require(receipt_path.is_file(), 'barrier exists without initial receipt')
        return True
    folder = root / 'outputs' / cfg['run_name'] / cfg['jobs'][0]['name']
    paths = [folder / name for name in ('initial-ranks.json', 'bn-initial-ranks.json', 'parameter-groups.json', 'initial-checkpoint/index.json')]
    if not all(p.is_file() for p in paths):
        return False
    sys.path.insert(0, str(root / 'bundle'))
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized(), 'owner initial gate must remain CPU')
    from retrain.persistence import CheckpointStore
    from retrain.ema import tensor_digest, validate_state
    from retrain.experiment91 import validate_checkpoint, ema_rule
    initials, bn, groups = (read(p) for p in paths[:3])
    expected = read(evidence / 'MODEL_CPU_ADMISSION.json')['initial_state_control']
    full_digest = verify_records(initials, bn, groups, read(evidence / 'CPU_PARAMETER_GROUPS.json'), expected)
    fresh = read(control / 'train-fresh-spec.json')
    require(time.time() < read(control / 'train-phase-start.json')['soft_deadline_at'], 'startup budget expired')
    store = CheckpointStore(folder / 'initial-checkpoint')
    payload = store.load('last', map_location='cpu')
    validate_checkpoint(payload)
    require(not payload.get('cpu_fixture_only') and payload['successful_updates'] == payload['attempted_updates'] == 0
            and payload['optimizer']['state'] == {} and payload['scaler'] == {}, 'initial envelope not production public fresh')
    require(payload['fresh'] == fresh and payload['identity']['fresh'] == fresh
            and payload['identity']['source_digest'] == SOURCE
            and payload['identity']['initialization'] == fresh['initialization'], 'native fresh source differs')
    state = payload['model']
    require('attention_pool_weight' not in state and int(torch.count_nonzero(state['age_weight'])) == 0, 'no-q/zero-age state differs')
    require(tensor_digest(state) == full_digest, 'native checkpoint vs actual four-rank digest differs')
    fixed = {k: v for k, v in state.items() if k.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))}
    require(len(fixed) == 159 and tensor_digest(fixed) == PUBLIC_BN, 'native public BN differs')
    validate_state(payload['ema'], state, 0, ema_rule())
    require(payload['ema']['rule'] == fresh['ema'] == ema_rule() and tensor_digest(payload['ema']['model']) == INITIAL, 'initial EMA differs')
    require(not torch.cuda.is_initialized(), 'owner initial gate initialized CUDA')
    index = read(paths[3]); record = index['last']
    receipt = {'status': 'completed', 'at': time.time(), 'rank_count': 4, 'optimizer_state_entries': 0,
        'source_digest': SOURCE, 'model_tensor_sha256': INITIAL, 'frozen_bn_sha256': PUBLIC_BN,
        'parameter_tensors': 162, 'parameter_elements': sum(p['numel'] for g in groups for p in g['parameters']),
        'initial_files_sha256': {str(p.relative_to(root)): sha(p) for p in paths},
        'native_initial_blob': str(store.directory / record['file']), 'native_initial_blob_sha256': record['sha256'],
        'public_weights_reloaded_by_gate': False, 'new_model_created_by_gate': False, 'CUDA_initialized': False}
    atomic(receipt_path, receipt, exclusive=True)
    barrier = fresh['startup_barrier']
    atomic(barrier_path, {'round_id': barrier['round_id'], 'plan_sha256': fresh['plan_sha256'], 'source_digest': SOURCE,
        'purpose': 'train', 'run_ids': barrier['run_ids'], 'world_size': 4, 'initial_rank_count': 4, 'all_initials_passed': True,
        'six_rank_initial_proof_sha256': sha(paths[0]), 'initials_sha256': {cfg['run_name']: sha(paths[0])},
        'actual_initial_receipt_sha256': sha(receipt_path)}, exclusive=True)
    return True

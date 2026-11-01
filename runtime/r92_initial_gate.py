"""Actual admitted-world noBN initial proof; load native checkpoint, never a new model."""
import json
import copy
import hashlib
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
            and expected['buffer_count'] == candidate().get('static_buffer_count', candidate().get('static_index_buffers', candidate()['batch_norm_buffers'])),
            'CPU expected candidate/noBN identity differs')
    require(type(expected_world) is int and expected_world == 8
            and len(initials) == expected_world and all(type(x.get('rank')) is int for x in initials)
            and {x['rank'] for x in initials} == set(range(expected_world)),
            'all admitted initial ranks required')
    for rank in initials:
        require(rank['model_sha256'] == expected['model_tensor_sha256'] and rank['optimizer_state_entries'] == 0
                and rank['bn_modules'] == expected['batch_norm_modules'] and rank['scaler'] == {}
                and rank.get('attention_pool') == expected.get('attention_pool')
                and rank.get('stage3_residual') == expected.get('stage3_residual'),
                'fresh tensor/Adam/BN/attention identity differs')
        require(rank.get('candidate_normalization') == expected['candidate_normalization'], 'actual rank LN/noBN differs')
        require(rank.get('parameter_trainability') == expected.get('parameter_trainability'),
                'actual rank frozen/trainable/optimizer identity differs')
        if expected.get('low_rank_optimizer') is not None:
            require(rank.get('low_rank_optimizer') == expected['low_rank_optimizer'],
                    'actual rank pointwise initial Adam coverage differs')
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
            require(p['name'] not in names and p['name'] != 'q'
                    and (p['name'] != 'attention_pool_weight' or expected.get('attention_pool') is not None)
                    and (p['name'] != 'stage3_residual_weight' or expected.get('stage3_residual') is not None),
                    'duplicate/unregistered attention parameter')
            names.add(p['name']); roles[role] = roles.get(role, 0) + p['numel']
    require(len(names) == expected['trainable_tensors'] and roles == expected['parameter_elements_by_role'], 'actual Tiny parameter topology differs')
    require(('attention_pool_weight' in names) == (expected.get('attention_pool') is not None),
            'native content pooling optimizer membership differs')
    require(('stage3_residual_weight' in names) == (expected.get('stage3_residual') is not None),
            'native stage3 residual optimizer membership differs')
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
    fresh = read(control / 'train-fresh-spec.json')
    expected = copy.deepcopy(read(evidence / 'MODEL_CPU_ADMISSION.json')['initial_state_control'])
    ssl = None
    if fresh.get('ssl_recipe') is not None:
        ssl_path = folder / 'ssl-stage/SSL-RESULT.json'
        public = read(evidence / 'SSL-PUBLIC-INITIAL.json')
        ssl = read(ssl_path)
        asset = folder / 'ssl-stage/derived-encoder.pt'
        initials = read(paths[0])
        from retrain.site_binding import load_site
        site = load_site()
        clock = read(control / 'round-clock.json')
        require(public['status'] == 'completed' and public['recipe'] == fresh['ssl_recipe']
                and public['public_tensor_sha256'] == expected['public_tensor_sha256']
                and public['public_encoder_tensor_sha256'] == expected['public_encoder_tensor_sha256']
                and ssl['status'] == 'completed' and ssl['recipe'] == fresh['ssl_recipe']
                and ssl['successful_updates'] == 1600 and ssl['dev_evaluations'] == 0
                and ssl['public_tensor_sha256'] == public['public_tensor_sha256']
                and ssl['public_encoder_tensor_sha256'] == public['public_encoder_tensor_sha256']
                and ssl['ssl_rank0_non_encoder_sha256'] == public['ssl_rank0_non_encoder_sha256']
                and ssl['round_id'] == site['round_id']
                and ssl['attempt_token_sha256'] == hashlib.sha256(clock['token'].encode()).hexdigest()
                and ssl['git_commit'] == site['git_commit']
                and ssl['science_sha256'] == site['science']['configuration']['sha256']
                and ssl['source_digest'] == fresh['source_digest']
                and ssl['manifest_digest'] == fresh['experiment91']['manifest_digest']
                and ssl['public_weight_sha256'] == fresh['initialization']['weights_sha256']
                and ssl['world_size'] == world and ssl['attempted_updates'] == 1600
                and ssl['skipped_updates'] == 0
                and len(ssl['rank_proofs']) == world
                and {item['rank'] for item in ssl['rank_proofs']} == set(range(world))
                and all(item['derived_tensor_sha256'] == ssl['derived_tensor_sha256']
                        and item['head_age_unchanged'] is True for item in ssl['rank_proofs'])
                and ssl['asset_path'] == str(asset) and not asset.is_symlink()
                and sha(asset) == ssl['asset_sha256'] and len(initials) == world,
                'R115 actual public-to-fit-only-derived lineage differs')
        require(initials[0].get('supervised_factory_non_encoder_sha256') == expected['public_non_encoder_sha256']
                and all(item.get('supervised_factory_rng_restored') is True for item in initials),
                'R115 native R107-seeded head/age factory or rank RNG reset differs')
        expected['model_tensor_sha256'] = initials[0]['model_sha256']
    full_digest = verify_records(read(paths[0]), read(paths[1]), read(evidence / 'CPU_PARAMETER_GROUPS.json'), expected,
                                 expected_world=world)
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
    if ssl is not None:
        asset_state = torch.load(ssl['asset_path'], map_location='cpu', weights_only=True)
        subset = {name: tensor for name, tensor in state.items()
                  if name.startswith('network.features.') or name.startswith('network.classifier.0.')}
        remainder = {name: tensor for name, tensor in state.items() if name not in subset}
        require(asset_state.keys() == subset.keys()
                and all(torch.equal(asset_state[name], subset[name]) for name in subset)
                and tensor_digest(asset_state) == ssl['derived_tensor_sha256']
                and tensor_digest(remainder) == read(evidence / 'MODEL_CPU_ADMISSION.json')['initial_state_control']['public_non_encoder_sha256']
                and payload.get('ssl_derived') == ssl,
                'R115 native supervised initial does not equal fixed SSL encoder plus public fresh head/age')
    age_enabled = payload['job']['age']['enabled']
    has_attention = expected.get('attention_pool') is not None
    has_stage3 = expected.get('stage3_residual') is not None
    require(('attention_pool_weight' in state) == has_attention and 'q' not in state
            and (not has_attention or (tuple(state['attention_pool_weight'].shape) == (1, 768)
                                       and state['attention_pool_weight'].dtype == torch.float32
                                       and int(torch.count_nonzero(state['attention_pool_weight'])) == 0))
            and ('stage3_residual_weight' in state) == has_stage3
            and (not has_stage3 or (tuple(state['stage3_residual_weight'].shape) == (1, 384)
                                    and state['stage3_residual_weight'].dtype == torch.float32
                                    and int(torch.count_nonzero(state['stage3_residual_weight'])) == 0))
            and ('age_weight' in state) == age_enabled
            and (not age_enabled or int(torch.count_nonzero(state['age_weight'])) == 0),
            'attention/resolved-age state differs')
    require(tensor_digest(state) == full_digest and payload['candidate_normalization'] == expected['candidate_normalization'], 'native tensor/actual LN differs')
    require(payload.get('parameter_trainability') == expected.get('parameter_trainability'),
            'native frozen/trainable/optimizer identity differs')
    if expected.get('low_rank_optimizer') is not None:
        require(payload.get('low_rank_optimizer') == expected['low_rank_optimizer'],
                'native pointwise initial Adam coverage differs')
    from retrain.single91 import immutable_buffers
    buffers = immutable_buffers(state)
    require(len(buffers) == expected['buffer_count'] and tensor_digest(buffers) == expected['buffer_sha256'],
            'native immutable model buffers differ')
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
    if ssl is not None:
        receipt['ssl_derived'] = ssl
        receipt['ssl_receipt_sha256'] = sha(folder / 'ssl-stage/SSL-RESULT.json')
    if expected.get('parameter_trainability') is not None:
        receipt['parameter_trainability'] = expected['parameter_trainability']
    atomic(receipt_path, receipt, exclusive=True)
    barrier = fresh['startup_barrier']
    atomic(barrier_path, {'round_id': barrier['round_id'], 'plan_sha256': fresh['plan_sha256'], 'source_digest': source,
        'purpose': 'train', 'run_ids': barrier['run_ids'], 'world_size': world, 'initial_rank_count': world, 'all_initials_passed': True,
        'rank_initial_proof_sha256': sha(paths[0]), 'initials_sha256': {cfg['run_name']: sha(paths[0])},
        'actual_initial_receipt_sha256': sha(receipt_path)}, exclusive=True)
    return True

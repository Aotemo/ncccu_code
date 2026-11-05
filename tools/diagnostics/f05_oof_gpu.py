"""Fit-only fold roles and supervised packet checks for the F04 GPU owner."""
import math


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _round_name(fold):
    if type(fold) is not int or not 0 <= fold < 3:
        raise ValueError('one of the three registered fit-only folds is required')
    return (f'r{265 + fold}-small-f05fold{fold}-seed20261029-'
            'pairsigmoid-24k-p4-9010-20261105')


def build_plan(round_id, fold):
    if round_id != _round_name(fold):
        raise ValueError('fold and registered round differ')
    return [{'device_rank': rank, 'member': round_id, 'rank': rank, 'world': 8}
            for rank in range(8)]


def validate_inputs(inputs, policy):
    """Validate the immutable identity of one presealed fit-only evaluation."""
    from pathlib import Path
    if (not isinstance(inputs, dict) or not isinstance(policy, dict)
            or policy.get('version') != 'f05-fit-only-oof-gpu-20261105-v1'
            or policy.get('folds') != 3 or policy.get('rounds') != ['R265', 'R266', 'R267']
            or policy.get('population_order') != ['train', 'heldout']
            or policy.get('world_size') != 8 or policy.get('batch_size') != 8
            or policy.get('view') != 'identity' or policy.get('model_mode') != 'eval'
            or policy.get('storage') != 'fp32' or policy.get('original_fit_count') != 46256
            or policy.get('dev_count') != 5140 or policy.get('minimum_free_bytes') != 20 * 1024 ** 3
            or policy.get('evaluation_reserve_bytes') != 1024 ** 3
            or any(policy.get(key) is not False for key in ('tta', 'dev_read', 'official_read'))
            or inputs.get('status') != 'F05_fit_only_oof_inputs_sealed_no_GPU'
            or any(inputs.get(key) is not False for key in ('TTA', 'dev_read', 'official_read', 'GPU_initialized'))):
        raise ValueError('fixed predeclared fit-only OOF policy and input are required')
    fold = inputs.get('fold')
    round_id = _round_name(fold)
    members = inputs.get('members')
    if (not isinstance(members, list) or len(members) != 1 or not isinstance(members[0], dict)
            or members[0].get('round_id') != round_id
            or inputs.get('plan') != build_plan(round_id, fold)
            or inputs.get('base_manifest_sha256') != policy.get('base_manifest_sha256')
            or inputs.get('folds_sha256') != policy.get('folds_sha256')):
        raise ValueError('one registered fold and its eight-rank plan are required')
    keys = ('manifest', 'folds', 'training_manifest', 'site', 'science', 'runtime_binding',
            'freeze', 'completion', 'safe', 'checkpoint', 'sidecar', 'local_checkpoint')
    for key in keys:
        value = inputs.get(key)
        if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
                or not isinstance(value['path'], str) or not Path(value['path']).is_absolute()
                or not _digest(value['sha256'])):
            raise ValueError('complete pinned OOF source references are required')
    if (inputs['manifest']['sha256'] != policy['base_manifest_sha256']
            or inputs['folds']['sha256'] != policy['folds_sha256']
            or inputs.get('training_manifest_sha256') != inputs['training_manifest']['sha256']
            or inputs.get('checkpoint_sha256') != inputs['checkpoint']['sha256']
            or inputs['local_checkpoint']['sha256'] != inputs['checkpoint']['sha256']
            or members[0].get('site') != inputs['site'] or members[0].get('freeze') != inputs['freeze']
            or members[0].get('selected') != inputs.get('selected')):
        raise ValueError('fold source and historical runtime references differ')
    selected = inputs.get('selected')
    if (not isinstance(selected, dict) or set(selected) != {'variant', 'tensor_sha256', 'successful_updates'}
            or selected.get('variant') not in ('ema', 'online') or not _digest(selected.get('tensor_sha256'))
            or type(selected.get('successful_updates')) is not int or selected['successful_updates'] <= 0):
        raise ValueError('unique selected variant, tensor and update are required')
    return inputs


def verified_roles(base, training, sealed, *, fold):
    """Recheck the presealed original-fit partition and fit-only age."""
    from retrain.f05_folds import verify_roles
    _round_name(fold)
    verify_roles(base, training, sealed, fold_number=fold, split_seed=20261121)
    train, heldout = list(sealed['fit_indices']), list(sealed['audit_indices'])
    if (train != sorted(train) or heldout != sorted(heldout)
            or any({base['rows'][index]['label'] for index in role} != {0, 1}
                   for role in (train, heldout))):
        raise ValueError('ordered complete binary fold populations required')
    return {'train': train, 'heldout': heldout}


def selected_checkpoint_sidecar(outputs, checkpoint_sha):
    """Find the unique sidecar declaring the frozen selected blob."""
    import json
    import os
    from pathlib import Path
    outputs = Path(outputs)
    if not _digest(checkpoint_sha) or outputs.is_symlink() or not outputs.is_dir():
        raise ValueError('historical checkpoint directory and frozen digest required')
    candidates = []
    def scan_error(error):
        raise ValueError('checkpoint enumeration failed') from error
    for directory, dirs, files in os.walk(outputs, onerror=scan_error, followlinks=False):
        directory = Path(directory)
        if any((directory / name).is_symlink() for name in dirs):
            raise ValueError('checkpoint enumeration cannot follow a directory link')
        names = set(files)
        for name in names:
            if name.startswith('checkpoint-') and name.endswith('.pt'):
                path = directory / name
                if path.is_symlink() or not path.is_file() or name + '.json' not in names:
                    raise ValueError('checkpoint without a regular matching sidecar')
            if not (name.startswith('checkpoint-') and name.endswith('.pt.json')):
                continue
            sidecar_path = directory / name
            if (sidecar_path.is_symlink() or not sidecar_path.is_file()
                    or name[:-5] not in names or (directory / name[:-5]).is_symlink()):
                raise ValueError('checkpoint sidecar lacks a regular matching blob')
            sidecar = json.loads(sidecar_path.read_text())
            if sidecar.get('sha256') == checkpoint_sha:
                candidates.append(sidecar_path)
    if len(candidates) != 1:
        raise ValueError('exactly one selected checkpoint sidecar must match the frozen digest')
    return candidates[0]


def prepare(gpu, args):
    """Seal one completed fold and a verified Pod-local selected checkpoint copy."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import time

    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('published OOF source differs')
    sys.path[:0] = [str(source / 'runtime/bundle'), str(source / 'tools/diagnostics'),
                    str(source / 'tools/pretraining')]
    from f04_bounded_fusion import verify_blob
    from f04_export_member import verify_published
    from retrain.site_binding import load_site
    from source_chain import cpu_context
    context = cpu_context()
    files = verify_published(source, args.remote_metadata, args.git_commit)
    for name in ('tools/diagnostics/f04_gpu_full_dev.py', 'tools/diagnostics/f05_oof_gpu.py',
                 'configs/F05-OOF-GPU-20261105.json'):
        files[name] = verify_blob(source, args.remote_metadata, name)
    tree = gpu.read_pinned(args.tree_proof, args.tree_proof_sha256)
    if (tree.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
            or tree.get('source_root') != str(source) or tree.get('commit') != args.git_commit
            or tree.get('pod_uid') != context['uid'] or tree.get('GPU_initialized') is not False
            or tree.get('official_read') is not False):
        raise ValueError('the current published Pod-local source tree is required')
    execution_site = load_site(args.execution_site, expected_sha256=args.execution_site_sha256)
    if execution_site['pod']['uid'] != context['uid']:
        raise ValueError('execution site does not belong to the verified Pod')
    work = Path(execution_site['work_root']).resolve(strict=True)

    def pinned_ref(path, expected=None):
        path = Path(path)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or not path.resolve(strict=True).is_relative_to(work)):
            raise ValueError('source must be a regular file within the verified work root')
        digest = gpu.sha(path)
        if expected is not None and digest != expected:
            raise ValueError('predeclared source bytes differ')
        return {'path': str(path), 'sha256': digest}

    safe_ref = pinned_ref(args.safe, args.safe_sha256)
    safe = gpu.read_pinned(safe_ref['path'], safe_ref['sha256'])
    root = Path(safe['formal_root'])
    if (not root.is_absolute() or root.is_symlink() or root.name != _round_name(args.fold)
            or not root.resolve(strict=True).is_relative_to(work / 'rounds')):
        raise ValueError('registered completed fold root required')
    site_ref = pinned_ref(root / 'SITE_BINDING.json')
    site = gpu.read_pinned(site_ref['path'], site_ref['sha256'])
    if site.get('work_root') != str(work) or site.get('data_root') != execution_site.get('data_root'):
        raise ValueError('historical fold and execution site data roots differ')
    science_ref = site['science']['configuration']
    if pinned_ref(science_ref['path'], science_ref['sha256']) != science_ref:
        raise ValueError('historical science binding differs')
    science = gpu.read_pinned(science_ref['path'], science_ref['sha256'])
    policy = json.loads((source / 'configs/F05-OOF-GPU-20261105.json').read_text())
    fold_policy = science['f05_fold']
    if fold_policy.get('fold') != args.fold:
        raise ValueError('science fold differs')
    manifest_ref = pinned_ref(work / fold_policy['base_metadata_root'] / 'manifest.json',
                              policy['base_manifest_sha256'])
    folds_ref = pinned_ref(work / fold_policy['sealed_folds_path'], policy['folds_sha256'])
    freeze_ref = pinned_ref(root / 'SINGLE-MODEL-FREEZE.json')
    freeze = gpu.read_pinned(freeze_ref['path'], freeze_ref['sha256'])
    selected = {key: freeze['selected'][key] for key in ('variant', 'tensor_sha256', 'successful_updates')}
    training_ref = pinned_ref(Path(site['metadata_root']) / 'manifest.json',
                              freeze['experiment91']['manifest_sha256'])
    checkpoint_sha = freeze['blob_sha256']
    if not _digest(checkpoint_sha):
        raise ValueError('selected checkpoint digest missing')
    outputs = root / 'outputs'
    sidecar_path = selected_checkpoint_sidecar(outputs, checkpoint_sha)
    checkpoint = Path(str(sidecar_path)[:-5])
    sidecar_ref = pinned_ref(sidecar_path)
    metadata = gpu.read_pinned(sidecar_ref['path'], sidecar_ref['sha256'])
    if (metadata.get('file') != checkpoint.name or metadata.get('format') != 'retraining-container-checkpoint'
            or metadata.get('format_version') != 1 or checkpoint.is_symlink() or not checkpoint.is_file()
            or not checkpoint.resolve(strict=True).is_relative_to(outputs.resolve(strict=True))):
        raise ValueError('selected checkpoint sidecar or location differs')
    local_root = (Path('/tmp/technical-prep') / context['uid']).resolve(strict=True)
    local_dir = args.local_dir
    if (not local_dir.is_absolute() or local_dir.exists() or local_dir.is_symlink()
            or not local_dir.parent.resolve(strict=True).is_relative_to(local_root)):
        raise ValueError('fresh Pod-local selected copy directory required')
    output = args.output
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')):
        raise ValueError('fresh private OOF inputs output required')
    local_dir.mkdir()
    local = local_dir / checkpoint.name
    digest = hashlib.sha256()
    with checkpoint.open('rb') as incoming, local.open('xb') as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b''):
            digest.update(block); outgoing.write(block)
        outgoing.flush(); os.fsync(outgoing.fileno())
    if digest.hexdigest() != checkpoint_sha or gpu.sha(local) != checkpoint_sha:
        raise ValueError('source or local selected checkpoint bytes differ')
    local_sidecar = Path(str(local) + '.json')
    with sidecar_path.open('rb') as incoming, local_sidecar.open('xb') as outgoing:
        outgoing.write(incoming.read()); outgoing.flush(); os.fsync(outgoing.fileno())
    if gpu.sha(local_sidecar) != sidecar_ref['sha256']:
        raise ValueError('local checkpoint sidecar differs')
    checkpoint_ref = {'path': str(checkpoint), 'sha256': checkpoint_sha}
    member = {'round_id': _round_name(args.fold), 'site': site_ref, 'freeze': freeze_ref,
              'selected': selected, 'bundle': str(root / 'bundle'), 'data_root': site['data_root'],
              'cache': site['prepared_cache']}
    folds = gpu.read_pinned(folds_ref['path'], folds_ref['sha256'])
    inputs = {'status': 'F05_fit_only_oof_inputs_sealed_no_GPU', 'at': time.time(),
              'git_commit': args.git_commit, 'source_root': str(source), 'git_files': files,
              'fold': args.fold, 'manifest': manifest_ref, 'folds': folds_ref,
              'base_manifest_sha256': manifest_ref['sha256'], 'folds_sha256': folds_ref['sha256'],
              'training_manifest': training_ref, 'training_manifest_sha256': training_ref['sha256'],
              'site': site_ref, 'science': science_ref, 'runtime_binding': pinned_ref(root / 'RUNTIME_BINDING.json'),
              'freeze': freeze_ref, 'completion': pinned_ref(root / 'evidence/TRAIN_RESULT.json'),
              'safe': safe_ref, 'checkpoint': checkpoint_ref, 'checkpoint_sha256': checkpoint_sha,
              'sidecar': sidecar_ref, 'local_checkpoint': {'path': str(local), 'sha256': checkpoint_sha},
              'age_contract_sha256': folds['folds'][args.fold]['age']['contract_sha256'],
              'selected': selected, 'members': [member], 'plan': build_plan(member['round_id'], args.fold),
              'TTA': False, 'dev_read': False, 'official_read': False, 'GPU_initialized': False,
              'execution': context}
    validate_inputs(inputs, policy)
    roles = verify_assets(gpu, inputs, work_root=work, pod_uid=context['uid'])
    gpu.write_new(output, inputs)
    print(json.dumps({'status': inputs['status'], 'fold': args.fold, 'inputs_sha256': gpu.sha(output),
                      'checkpoint_sha256': checkpoint_sha,
                      'role_counts': {key: len(value) for key, value in roles.items()},
                      'dev_read': False, 'official_read': False}), flush=True)


def verify_assets(gpu, inputs, *, work_root, pod_uid):
    """Recheck the historical fold, selected checkpoint and private local copy."""
    from pathlib import Path
    from retrain.f05_folds import verify_bound_fold

    work = Path(work_root).resolve(strict=True)

    def pinned(reference):
        if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
            raise ValueError('explicit file and digest required')
        path = Path(reference['path'])
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or not path.resolve(strict=True).is_relative_to(work)):
            raise ValueError('historical evidence is outside the verified work root')
        return gpu.read_pinned(path, reference['sha256'])

    base, seal, training, site, science, runtime, freeze, completion, safe, sidecar = (
        pinned(inputs[name]) for name in ('manifest', 'folds', 'training_manifest', 'site',
            'science', 'runtime_binding', 'freeze', 'completion', 'safe', 'sidecar'))
    fold = inputs['fold']; selected = inputs['selected']; round_id = _round_name(fold)
    member = inputs['members'][0]
    root = Path(inputs['freeze']['path']).parent
    if (root.name != round_id or Path(inputs['freeze']['path']).name != 'SINGLE-MODEL-FREEZE.json'
            or Path(inputs['site']['path']) != root / 'SITE_BINDING.json'
            or Path(inputs['runtime_binding']['path']) != root / 'RUNTIME_BINDING.json'
            or Path(inputs['completion']['path']) != root / 'evidence/TRAIN_RESULT.json'
            or Path(inputs['training_manifest']['path']) != Path(site['metadata_root']) / 'manifest.json'
            or site.get('round_id') != round_id or runtime.get('site_binding_sha256') != inputs['site']['sha256']
            or site.get('science', {}).get('configuration') != inputs['science']
            or freeze.get('cpu_fixture_only') is not False or freeze.get('precision') != 'fp32'
            or freeze.get('view') != 'identity'
            or freeze.get('blob_sha256') != inputs['checkpoint']['sha256']
            or freeze.get('experiment91', {}).get('manifest_sha256') != inputs['training_manifest']['sha256']
            or any(freeze.get('selected', {}).get(key) != value for key, value in selected.items())
            or completion.get('status') != 'completed'
            or type(completion.get('successful_updates')) is not int
            or completion['successful_updates'] < selected['successful_updates']
            or safe.get('phase') != 'terminal' or safe.get('formal_root') != str(root)
            or safe.get('done', {}).get('present') is not True or safe.get('done', {}).get('status') != 'completed'
            or safe.get('safety', {}).get('safe') is not True
            or safe.get('first_success_receipt_read') is not True
            or safe.get('signals_sent') is not False or safe.get('model_forward_executed') is not False
            or safe.get('official_read') is not False
            or any(safe.get('selected', {}).get(key) != value for key, value in selected.items()
                   if key in ('variant', 'successful_updates'))):
        raise ValueError('completed selected fold, historical site or safety differs')
    checkpoint = Path(inputs['checkpoint']['path'])
    if (not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_file()
            or not checkpoint.resolve(strict=True).is_relative_to((root / 'outputs').resolve(strict=True))
            or Path(inputs['sidecar']['path']) != Path(str(checkpoint) + '.json')
            or sidecar.get('file') != checkpoint.name or sidecar.get('format') != 'retraining-container-checkpoint'
            or type(sidecar.get('format_version')) is not int or sidecar['format_version'] != 1
            or sidecar.get('sha256') != inputs['checkpoint']['sha256']
            or gpu.sha(checkpoint) != inputs['checkpoint']['sha256']):
        raise ValueError('selected checkpoint bytes or adjacent sidecar differ')
    local = Path(inputs['local_checkpoint']['path'])
    local_root = Path('/tmp/technical-prep') / pod_uid
    if (not local.is_absolute() or local.is_symlink() or not local.is_file()
            or not local.resolve(strict=True).is_relative_to(local_root.resolve(strict=True))
            or local.name != checkpoint.name or gpu.sha(local) != inputs['checkpoint']['sha256']
            or gpu.sha(Path(str(local) + '.json')) != inputs['sidecar']['sha256']):
        raise ValueError('Pod-local selected checkpoint copy differs')
    files = freeze.get('source_files')
    bundle = root / 'bundle'
    if (not isinstance(files, dict) or not files or bundle.is_symlink() or not bundle.is_dir()
            or member.get('bundle') != str(bundle) or member.get('data_root') != site.get('data_root')
            or member.get('cache') != site.get('prepared_cache')):
        raise ValueError('historical runtime or fit cache identity differs')
    for name, digest in files.items():
        relative = Path(name)
        path = bundle / relative
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.is_file()
                or not path.resolve(strict=True).is_relative_to(bundle.resolve(strict=True))
                or not _digest(digest) or gpu.sha(path) != digest):
            raise ValueError('historical runtime source bytes differ')
    if (len(base['fit_indices']) != 46256 or len(base['dev_indices']) != 5140
            or seal.get('status') != 'three_fold_fit_only_indices_sealed_no_training'
            or seal.get('base_manifest_sha256') != inputs['manifest']['sha256']
            or seal.get('seed') != 20261121 or seal.get('training_started') is not False
            or seal.get('official_read') is not False or len(seal.get('folds', [])) != 3):
        raise ValueError('original manifest or registered three-fold seal differs')
    roles = verified_roles(base, training, seal['folds'][fold], fold=fold)
    bound = verify_bound_fold(site, science)
    if (bound.get('fold_number') != fold
            or bound.get('fit_indices_sha256') != seal['folds'][fold]['fit_indices_sha256']
            or bound.get('heldout_indices_sha256') != seal['folds'][fold]['audit_indices_sha256']
            or bound.get('age_contract_sha256') != inputs.get('age_contract_sha256')):
        raise ValueError('site-bound fold roles or fit-only age differ')
    return roles


def validate_evaluation_window(window, peer, peer_site, *, safe_sha, site_sha,
                               peer_observation_sha, peer_site_sha, current_uid,
                               now, free, peer_limit):
    """Require a fresh, conservatively budgeted peer observation before evaluation."""
    gib = 1024 ** 3
    if (not isinstance(window, dict) or not isinstance(peer, dict) or not isinstance(peer_site, dict)
            or not all(_digest(value) for value in (safe_sha, site_sha, peer_observation_sha, peer_site_sha))
            or not isinstance(current_uid, str) or not current_uid
            or type(now) not in (int, float) or not math.isfinite(now)
            or type(free) is not int or free < 0 or type(peer_limit) is not int or peer_limit < 12 * gib):
        raise ValueError('current peer, capacity and pinned evidence are required')
    at, observed = window.get('at'), peer.get('observed_at')
    if (type(at) not in (int, float) or type(observed) not in (int, float)
            or not math.isfinite(at) or not math.isfinite(observed)
            or not 0 <= now - at <= 120 or not 0 <= now - observed <= 120
            or window.get('status') != 'f05_evaluation_capacity_verified'
            or window.get('predecessor_safe_sha256') != safe_sha or window.get('site_sha256') != site_sha
            or window.get('peer_observation_sha256') != peer_observation_sha
            or window.get('peer_site_sha256') != peer_site_sha
            or window.get('peer_round_id') != peer_site.get('round_id')
            or peer.get('formal_root') != peer_site.get('round_root')
            or not isinstance(peer_site.get('pod'), dict)
            or not peer_site['pod'].get('uid') or peer_site['pod']['uid'] == current_uid
            or peer.get('signals_sent') is not False or peer.get('model_forward_executed') is not False
            or peer.get('official_read') is not False or window.get('official_read') is not False
            or window.get('training_activated') is not False or window.get('GPU_started') is not False):
        raise ValueError('fresh independent peer and read-only evaluation identity required')
    if peer.get('phase') == 'running':
        identity = peer.get('identity') or {}
        if identity.get('owner_status') != 'active' or identity.get('rank_status') != ['active'] * 8:
            raise ValueError('running peer owner and eight ranks must be observed')
        reserve = peer_limit
    elif (peer.get('phase') == 'terminal' and peer.get('done', {}).get('present') is True
          and peer.get('done', {}).get('status') in ('completed', 'failed', 'cancelled')
          and peer.get('safety', {}).get('safe') is True):
        reserve = 0
    else:
        raise ValueError('peer state is neither actively observed nor safely terminal')
    required = 21 * gib + reserve
    recorded = window.get('free_bytes')
    if (window.get('peer_storage_limit_bytes') != peer_limit or window.get('peer_reserved_bytes') != reserve
            or window.get('required_free_bytes') != required or type(recorded) is not int
            or recorded <= required or free <= required):
        raise ValueError('20 GiB, evaluation output and full peer capacity must remain available')
    return window


def prepare_evaluation_window(gpu, args):
    """Seal a fresh capacity window using existing round and peer observations."""
    import json
    import os
    from pathlib import Path
    import shutil
    import sys
    import time

    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('published evaluation source differs')
    sys.path[:0] = [str(source / 'runtime/bundle'), str(source / 'tools/diagnostics'),
                    str(source / 'tools/pretraining')]
    from f04_bounded_fusion import verify_blob
    from f04_export_member import verify_published
    from retrain.site_binding import load_site
    from source_chain import cpu_context
    context = cpu_context()
    verify_published(source, args.remote_metadata, args.git_commit)
    for name in ('tools/diagnostics/f04_gpu_full_dev.py', 'tools/diagnostics/f05_oof_gpu.py',
                 'configs/F05-OOF-GPU-20261105.json'):
        verify_blob(source, args.remote_metadata, name)
    proof = gpu.read_pinned(args.tree_proof, args.tree_proof_sha256)
    if (proof.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
            or proof.get('source_root') != str(source) or proof.get('commit') != args.git_commit
            or proof.get('pod_uid') != context['uid'] or proof.get('GPU_initialized') is not False
            or proof.get('official_read') is not False):
        raise ValueError('published Pod-local source proof required')
    site = load_site(args.site, expected_sha256=args.site_sha256)
    peer_site = load_site(args.peer_site, expected_sha256=args.peer_site_sha256)
    if site['pod']['uid'] != context['uid'] or context['uid'] != os.environ.get('NCCCU_POD_UID'):
        raise ValueError('evaluation resource site is not this Pod')
    safe = gpu.read_pinned(args.safe, args.safe_sha256)
    gpu.require_safe_predecessor(safe, safe['formal_root'], time.time())
    predecessor = Path(safe['formal_root'])
    previous_site_path = predecessor / 'SITE_BINDING.json'
    previous_site = json.loads(previous_site_path.read_text())
    previous_runtime = json.loads((predecessor / 'RUNTIME_BINDING.json').read_text())
    if (previous_site_path.is_symlink() or previous_site.get('pod', {}).get('uid') != context['uid']
            or previous_runtime.get('site_binding_sha256') != gpu.sha(previous_site_path)):
        raise ValueError('safe predecessor belongs to a different or unbound Pod')
    peer = gpu.read_pinned(args.peer_observation, args.peer_observation_sha256)
    peer_root = Path(peer_site['round_root'])
    peer_site_path = peer_root / 'SITE_BINDING.json'
    peer_runtime = json.loads((peer_root / 'RUNTIME_BINDING.json').read_text())
    if (peer_site_path != args.peer_site or peer_runtime.get('site_binding_sha256') != args.peer_site_sha256
            or peer_runtime.get('git_commit') != peer_site.get('git_commit')):
        raise ValueError('peer runtime is not bound to the observed peer site')
    policy_path = peer_root / 'STORAGE_POLICY.json'
    policy = json.loads(policy_path.read_text())
    if policy_path.is_symlink() or policy.get('schema_version') != 1 or not isinstance(policy.get('round_overrides'), dict):
        raise ValueError('peer storage policy differs')
    limit = policy['round_overrides'].get(peer_site['round_id'], policy.get('default_round_storage_limit_bytes'))
    if type(limit) is not int or limit < 12 * 1024 ** 3:
        raise ValueError('registered peer storage budget required')
    peer_safe = (peer.get('phase') == 'terminal' and peer.get('done', {}).get('present') is True
                 and peer.get('safety', {}).get('safe') is True)
    reserve = 0 if peer_safe else limit
    free = shutil.disk_usage(site['work_root']).free
    output = args.output
    work = Path(site['work_root']).resolve(strict=True)
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')):
        raise ValueError('fresh isolated evaluation window path required')
    window = {'status': 'f05_evaluation_capacity_verified', 'at': time.time(),
              'predecessor_safe_sha256': args.safe_sha256, 'site_sha256': args.site_sha256,
              'peer_observation_sha256': args.peer_observation_sha256,
              'peer_site_sha256': args.peer_site_sha256, 'peer_round_id': peer_site['round_id'],
              'peer_observation': {'path': str(args.peer_observation), 'sha256': args.peer_observation_sha256},
              'peer_site': {'path': str(args.peer_site), 'sha256': args.peer_site_sha256},
              'peer_storage_policy': {'path': str(policy_path), 'sha256': gpu.sha(policy_path)},
              'peer_storage_limit_bytes': limit, 'peer_reserved_bytes': reserve,
              'required_free_bytes': 21 * 1024 ** 3 + reserve, 'free_bytes': free,
              'training_activated': False, 'GPU_started': False, 'official_read': False,
              'execution': context, 'git_commit': args.git_commit}
    validate_evaluation_window(window, peer, peer_site, safe_sha=args.safe_sha256,
        site_sha=args.site_sha256, peer_observation_sha=args.peer_observation_sha256,
        peer_site_sha=args.peer_site_sha256, current_uid=context['uid'], now=time.time(),
        free=free, peer_limit=limit)
    gpu.write_new(output, window)
    print({'status': window['status'], 'required_free_bytes': window['required_free_bytes'],
           'free_bytes': free, 'window_sha256': gpu.sha(output)}, flush=True)


def make_scores(base, sealed, packets, *, fold, role, round_id, inputs_sha, proof_shas,
                selected, checkpoint_sha, base_manifest_sha, folds_sha):
    """Reduce eight independently proved shards to one private score object."""
    from retrain.parallel_evaluation import canonical_predictions
    from retrain.persistence import canonical_digest
    build_plan(round_id, fold)
    if (role not in ('train', 'heldout') or sealed.get('fold') != fold
            or sealed.get('round') != 'R' + str(265 + fold)
            or not all(_digest(value) for value in (inputs_sha, checkpoint_sha,
                                                    base_manifest_sha, folds_sha))
            or not isinstance(selected, dict)
            or set(selected) != {'variant', 'tensor_sha256', 'successful_updates'}
            or selected['variant'] not in ('ema', 'online') or not _digest(selected['tensor_sha256'])
            or type(selected['successful_updates']) is not int or selected['successful_updates'] <= 0
            or not isinstance(proof_shas, dict) or set(proof_shas) != {str(i) for i in range(8)}
            or not all(_digest(value) for value in proof_shas.values())):
        raise ValueError('fixed fold, selected checkpoint and all device proofs required')
    key = 'fit_indices' if role == 'train' else 'audit_indices'
    digest_key = 'fit_indices_sha256' if role == 'train' else 'audit_indices_sha256'
    indices = sealed[key]
    if (not isinstance(indices, list) or not indices or indices != sorted(indices)
            or any(type(index) is not int for index in indices) or len(indices) != len(set(indices))
            or canonical_digest(indices) != sealed[digest_key]
            or not set(indices) <= set(base['fit_indices'])
            or set(indices) & set(base['dev_indices'])):
        raise ValueError('complete ordered sealed fit-only role required')
    load = {'selected_model_sha256': checkpoint_sha,
            'selected_tensor_sha256': selected['tensor_sha256'],
            'variant': selected['variant'], 'successful_updates': selected['successful_updates'],
            'strict_state_load': True, 'storage': 'fp32'}
    if not isinstance(packets, list) or len(packets) != 8:
        raise ValueError('eight supervised rank packets required')
    for packet in packets:
        rank = packet.get('rank')
        if (type(rank) is not int or not 0 <= rank < 8 or packet.get('device_rank') != rank
                or packet.get('member') != round_id or packet.get('fold') != fold
                or packet.get('role') != role or packet.get('error') is not None
                or packet.get('proof_sha256') != proof_shas[str(rank)]
                or packet.get('input_sha256') != inputs_sha or packet.get('model_load') != load
                or packet.get('TTA') is not False or packet.get('official_read') is not False):
            raise ValueError('fold rank, load, role or supervised proof differs')
    records = canonical_predictions(base['rows'], indices, packets, 8, 8)
    rows = [{'index': index, 'label': row['label'], 'logit': float(row['logit'])}
            for index, row in zip(indices, records)]
    return {'format': 'f05-fold-scores-v1', 'fold': fold, 'role': role,
            'base_manifest_sha256': base_manifest_sha, 'folds_sha256': folds_sha,
            'selected_model_sha256': checkpoint_sha,
            'selected_tensor_sha256': selected['tensor_sha256'],
            'selected_variant': selected['variant'], 'role_indices_sha256': sealed[digest_key],
            'rows': rows}


def aggregate(gpu, inputs, output, inputs_path, inputs_sha, tree_proof, tree_proof_sha):
    """Seal both supervised populations and a consumer-verifiable parent receipt."""
    from pathlib import Path

    output = Path(output)

    def ref(path):
        path = Path(path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError('an immutable supervised output file is required')
        return {'path': str(path), 'sha256': gpu.sha(path)}

    gpu.read_pinned(tree_proof, tree_proof_sha)
    input_ref = {'path': str(inputs_path), 'sha256': inputs_sha}
    if gpu.read_pinned(input_ref['path'], input_ref['sha256']) != inputs:
        raise ValueError('producer inputs changed during supervised inference')
    spec_ref, release_ref = ref(output / 'WORKER-SPEC.json'), ref(output / 'RELEASE.json')
    spec = gpu.read_pinned(spec_ref['path'], spec_ref['sha256'])
    release = gpu.read_pinned(release_ref['path'], release_ref['sha256'])
    if (spec.get('inputs') != input_ref or spec.get('oof_mode') is not True
            or release.get('status') != 'all_rank_device_proofs_passed'
            or release.get('spec_sha256') != spec_ref['sha256']
            or release.get('parent') != spec.get('parent')):
        raise ValueError('supervised worker release or input identity differs')
    base = gpu.read_pinned(inputs['manifest']['path'], inputs['manifest']['sha256'])
    folds = gpu.read_pinned(inputs['folds']['path'], inputs['folds']['sha256'])
    training = gpu.read_pinned(inputs['training_manifest']['path'], inputs['training_manifest']['sha256'])
    fold, selected = inputs['fold'], inputs['selected']
    sealed = folds['folds'][fold]
    verified_roles(base, training, sealed, fold=fold)
    proof_refs = [ref(output / 'proofs' / ('rank-' + str(rank) + '.json')) for rank in range(8)]
    proof_shas = {str(rank): value['sha256'] for rank, value in enumerate(proof_refs)}
    if release.get('proof_sha256') != proof_shas:
        raise ValueError('rank proof bytes do not match the supervised release')
    for rank, proof_ref in enumerate(proof_refs):
        proof = gpu.read_pinned(proof_ref['path'], proof_ref['sha256'])
        if proof.get('rank') != rank or proof.get('parent') != spec.get('parent'):
            raise ValueError('rank proof and owner identity differ')
    scores, packet_refs, counts = {}, {}, {}
    for role in ('train', 'heldout'):
        refs = [ref(output / 'packets' / ('rank-' + str(rank) + '-' + role + '.json'))
                for rank in range(8)]
        packets = [gpu.read_pinned(value['path'], value['sha256']) for value in refs]
        score = make_scores(base, sealed, packets, fold=fold, role=role,
                            round_id=inputs['members'][0]['round_id'], inputs_sha=inputs_sha,
                            proof_shas=proof_shas, selected=selected,
                            checkpoint_sha=inputs['checkpoint']['sha256'],
                            base_manifest_sha=inputs['manifest']['sha256'],
                            folds_sha=inputs['folds']['sha256'])
        path = output / (role + '-scores-private.json')
        gpu.write_new(path, score)
        counts[role] = len(score['rows'])
        scores[role] = {**ref(path), 'role_indices_sha256': score['role_indices_sha256'],
                        'count': counts[role]}
        packet_refs[role] = refs
    parent = {'format': 'f05-fold-inference-receipt-v1', 'status': 'completed_fit_only_selected_logits',
              'fold': fold, 'round_id': inputs['members'][0]['round_id'],
              'base_manifest_sha256': inputs['manifest']['sha256'], 'folds_sha256': inputs['folds']['sha256'],
              'age_contract_sha256': inputs['age_contract_sha256'], 'selected': selected,
              'producer_inputs': input_ref, 'worker_spec': spec_ref, 'release': release_ref,
              'source_tree_proof': {'path': str(tree_proof), 'sha256': tree_proof_sha},
              'rank_proofs': proof_refs, 'train_packets': packet_refs['train'],
              'heldout_packets': packet_refs['heldout'], 'scores': scores, 'forward_counts': counts,
              'world_size': 8, 'batch_size': 8, 'model_mode': 'eval', 'view': 'identity',
              'TTA': False, 'dev_read': False, 'official_read': False}
    for name in ('training_manifest', 'site', 'science', 'runtime_binding', 'freeze', 'completion',
                 'safe', 'checkpoint', 'sidecar'):
        parent[name] = inputs[name]
    parent_path = output / 'FOLD-INFERENCE.json'
    gpu.write_new(parent_path, parent)
    return {'fold': fold, 'forward_counts': counts, 'fold_receipt': ref(parent_path),
            'dev_read': False, 'official_read': False}


def worker(gpu, spec, inputs, assignment, rank, own_sha, parent, output, device_name, torch):
    """Run the two presealed fit roles after the existing eight-device barrier."""
    import importlib
    import os
    from pathlib import Path
    from retrain.age import forward, unpack_batch
    from retrain.data import ImageDataset
    from retrain.ema import tensor_digest
    from retrain.inference import _load_model
    from retrain.parallel_evaluation import batch_assignments
    from torch.utils.data import DataLoader

    if (spec.get('oof_mode') is not True or spec.get('soup_mode') is not False
            or spec.get('official') is not None or inputs.get('TTA') is not False
            or inputs.get('dev_read') is not False or inputs.get('official_read') is not False):
        raise ValueError('only the sealed fit-only worker mode is permitted')
    fold, member = inputs['fold'], inputs['members'][0]
    if assignment != build_plan(member['round_id'], fold)[rank]:
        raise ValueError('fold rank assignment differs')
    base = gpu.read_pinned(inputs['manifest']['path'], inputs['manifest']['sha256'])
    folds = gpu.read_pinned(inputs['folds']['path'], inputs['folds']['sha256'])
    training = gpu.read_pinned(inputs['training_manifest']['path'], inputs['training_manifest']['sha256'])
    roles = verified_roles(base, training, folds['folds'][fold], fold=fold)
    selected = inputs['selected']
    local = Path(inputs['local_checkpoint']['path'])
    sidecar = Path(str(local) + '.json')
    local_root = Path('/tmp/technical-prep') / os.environ['NCCCU_POD_UID']
    if (not local.is_absolute() or local.is_symlink() or sidecar.is_symlink()
            or not local.resolve(strict=True).is_relative_to(local_root.resolve(strict=True))
            or local.name != Path(inputs['checkpoint']['path']).name
            or gpu.sha(local) != inputs['checkpoint']['sha256']
            or gpu.sha(sidecar) != inputs['sidecar']['sha256']):
        raise ValueError('local selected checkpoint or adjacent sidecar differs')
    gpu.bind_evaluation_cache(member, False, output)
    precision = importlib.import_module('_f04_current_guard.precision')
    runtime_guard = importlib.import_module('_f04_current_guard.fresh')
    precision.configure(precision.RULE)
    model, job, _, origin, actual_device = _load_model(local, device=device_name,
                                                       variant=selected['variant'])
    loaded_state = model.state_dict()
    floating = {str(value.dtype) for value in loaded_state.values() if value.is_floating_point()}
    if (str(actual_device) != device_name or origin.get('sha256') != inputs['checkpoint']['sha256']
            or origin.get('variant') != selected['variant']
            or origin.get('tensor_sha256') != selected['tensor_sha256']
            or origin.get('successful_updates') != selected['successful_updates']
            or floating != {'torch.float32'} or tensor_digest(loaded_state) != selected['tensor_sha256']):
        raise ValueError('strictly loaded selected checkpoint, variant, update or dtype differs')
    model.eval()
    if model.training:
        raise ValueError('selected model did not enter evaluation mode')
    load = {'selected_model_sha256': origin['sha256'],
            'selected_tensor_sha256': origin['tensor_sha256'], 'variant': origin['variant'],
            'successful_updates': origin['successful_updates'], 'strict_state_load': True, 'storage': 'fp32'}
    for role in ('train', 'heldout'):
        assignments = batch_assignments(roles[role], 8, assignment['rank'], assignment['world'])
        dataset = gpu.evaluation_dataset(ImageDataset, member['data_root'], base['rows'], job)
        loader = DataLoader(dataset, batch_sampler=[batch for _, batch in assignments], num_workers=0)
        records = []
        with torch.inference_mode():
            for (ordinal, expected), batch in zip(assignments, loader):
                current = gpu.process_identity(parent['pid'])
                if (os.getppid() != parent['pid'] or not current.get('active')
                        or current.get('start_ticks') != str(parent['start_ticks'])
                        or runtime_guard.runtime_stop_reason(spec['deadline_at'], actual_device) is not None):
                    raise RuntimeError('owner or deadline failed during fit-only inference')
                images, labels, actual_indices, age = unpack_batch(batch, job, actual_device)
                with torch.autocast(device_type='cuda', enabled=False):
                    logits = forward(model, images, age, job).reshape(-1).float()
                probabilities = logits.sigmoid()
                records.extend(gpu.worker_batch_records(base['rows'], expected, actual_indices.tolist(),
                    labels.tolist(), logits.cpu().tolist(), probabilities.cpu().tolist(), rank, ordinal))
        packet = {'rank': rank, 'device_rank': rank, 'member': member['round_id'],
                  'fold': fold, 'role': role, 'error': None, 'records': records,
                  'proof_sha256': own_sha, 'input_sha256': spec['inputs']['sha256'],
                  'model_load': load, 'TTA': False, 'official_read': False}
        path = output / 'packets' / ('rank-' + str(rank) + '-' + role + '.json')
        gpu.write_new(path, packet)
        print({'status': 'fit_only_worker_role_completed', 'rank': rank, 'role': role,
               'records': len(records), 'packet_sha256': gpu.sha(path)}, flush=True)

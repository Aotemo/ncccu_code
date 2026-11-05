"""Build an isolated explicit package; this entry never creates a formal clock."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def limit_cpu_threads():
    """Match every preparation subprocess to the explicit allocation's core count."""
    raw = os.environ.get('NCCCU_CPU_THREADS')
    if not raw or not raw.isdecimal():
        return None
    count = int(raw)
    import torch
    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)
    return count


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def prepare_f05_targets(source, site, science, package):
    """Independently verify the OOF chain once, before a package can be admitted."""
    from retrain import experiment91 as ex
    from retrain.data import verify_f05_oof_for_training
    policy = ex.validate_f05_training_science(science, site)
    if policy is None:
        return None
    import importlib.util
    verifier_path = Path(source) / 'tools/diagnostics/f05_oof_targets.py'
    if verifier_path.is_symlink() or not verifier_path.is_file():
        raise ValueError('published F05 OOF verifier is missing')
    specification = importlib.util.spec_from_file_location('_f05_prepared_oof_targets', verifier_path)
    verifier = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(verifier)
    work = Path(site['work_root']).resolve(strict=True)
    summary = {'path': str(work / policy['summary_relative_path']), 'sha256': policy['summary_sha256']}
    proof = verify_f05_oof_for_training(summary, work, verifier=verifier)
    if proof['manifest_digest'] != ex.binding()['manifest_digest']:
        raise ValueError('verified OOF and training manifest identities differ')
    mode = policy['mode']
    target_name = 'CLEAN-TARGETS-private.json' if mode == 'clean' else 'DISTILL-TARGETS-private.json'
    if Path(proof['targets'][mode]['path']) != Path(summary['path']).parent / target_name:
        raise ValueError('verified F05 target is not in the fixed OOF output directory')
    proof_path = Path(package) / 'evidence/F05-OOF-VERIFICATION.json'
    write(proof_path, proof)
    return {'version': 'f05-oof-training-v1', 'mode': mode, 'summary': summary,
            'verification': {'path': str(Path(site['round_root']) / 'evidence/F05-OOF-VERIFICATION.json'),
                             'sha256': sha(proof_path)}, 'target': proof['targets'][mode]}


def science_binding_path(source, binding, work_root):
    source = Path(source)
    if binding is None:
        return source
    binding, work_root = Path(binding), Path(work_root)
    if (not binding.is_absolute() or not work_root.is_absolute() or binding.is_symlink()
            or not binding.is_file() or not binding.resolve().is_relative_to(work_root.resolve() / 'technical-prep')
            or sha(binding) != sha(source)):
        raise ValueError('science binding must be an exact regular technical-prep byte copy')
    for parent in (binding.parent, *binding.parents):
        if parent == work_root.parent:
            break
        if parent.is_symlink():
            raise ValueError('science binding cannot traverse a symlink')
    return binding


def verify_fold_migration(site, science, migrated):
    """Keep the original migration binding when using a derived OOF fold."""
    from retrain.f05_folds import validate_fold_configuration, verify_bound_fold
    if validate_fold_configuration(site, science) is None:
        return False
    try:
        base_root = migrated['paths']['stage1_root']
    except (KeyError, TypeError) as error:
        raise ValueError('original stage1 migration binding required') from error
    verify_bound_fold(site, science, base_root)
    return True


def prepare_local_checkpoint_storage(site, source, package):
    """Install only an explicitly pinned per-Pod scratch policy in a new package."""
    reference = site.get('local_checkpoint_storage')
    if reference is None:
        return
    from retrain.local_checkpoints import POLICY, _local_mount, mapped_directory
    policy = Path(source) / 'configs/F05-LOCAL-CHECKPOINT-POLICY.json'
    if (policy.is_symlink() or not policy.is_file() or read(policy) != POLICY
            or sha(policy) != reference['policy']['sha256']):
        raise ValueError('versioned local checkpoint policy differs')
    uid = site['pod']['uid']; round_id = site['round_id']; root = Path(site['round_root'])
    binding = {'version': POLICY['version'], 'pod_uid': uid, 'round_id': round_id,
               'round_root': str(root),
               'local_root': str(Path(POLICY['local_root_prefix']) / uid / POLICY['local_subdirectory'] / round_id)}
    mapped_directory(root / 'outputs' / 'fixture' / 'checkpoints', binding, site)
    _local_mount('/tmp')
    raw = (json.dumps(binding, indent=2, allow_nan=False) + '\n').encode()
    if (reference['binding']['path'] != str(root / 'LOCAL-CHECKPOINT-BINDING.json')
            or reference['policy']['path'] != str(root / 'LOCAL-CHECKPOINT-POLICY.json')
            or hashlib.sha256(raw).hexdigest() != reference['binding']['sha256']):
        raise ValueError('local checkpoint package reference differs')
    for name, body in (('LOCAL-CHECKPOINT-BINDING.json', raw),
                       ('LOCAL-CHECKPOINT-POLICY.json', policy.read_bytes())):
        with (Path(package) / name).open('xb') as stream:
            stream.write(body)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source-root', type=Path, required=True)
    result.add_argument('--science-config', type=Path, required=True,
                        help='explicit repository-relative versioned science configuration')
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--site-binding', type=Path)
    result.add_argument('--site-binding-sha256')
    result.add_argument('--science-binding-path', type=Path)
    result.add_argument('--work-root', type=Path)
    result.add_argument('--round-id')
    result.add_argument('--git-base')
    result.add_argument('--pcis')
    result.add_argument('--asset-receipt', type=Path, required=True)
    result.add_argument('--asset-receipt-sha256', required=True)
    return result


def technical_site(args, science, science_path):
    from retrain.experiment91 import candidate_from_configuration, BINDING_VERSION
    from retrain.site_binding import validate_site, require_cpu_channel
    require_cpu_channel()
    if (not args.work_root or not args.work_root.is_absolute() or not args.round_id
            or not args.git_base or not args.pcis):
        raise ValueError('technical preparation requires explicit work root, round, Git base and observed PCI order')
    root = args.work_root.resolve()
    paths = {}
    for key, relative in science['paths'].items():
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('configured asset path must stay inside the work root')
        paths[key] = str(root / name)
    cards = []
    for ordinal, pci in enumerate(args.pcis.split(',')):
        device = Path('/sys/bus/pci/devices') / pci
        cards.append({'ordinal': ordinal, 'pci': pci,
                      'vram_total_bytes': int((device / 'mem_info_vram_total').read_text())})
    return validate_site({'schema_version': 1, 'scope': 'cpu_technical_prep',
        'round_id': args.round_id, 'git_commit': args.git_base,
        'source_state': {'kind': 'uncommitted_cpu_snapshot', 'base_commit': args.git_base},
        'work_root': str(root), 'round_root': str(root / 'rounds' / args.round_id),
        'pod': {'namespace': os.environ['NCCCU_POD_NAMESPACE'], 'name': os.environ['NCCCU_POD_NAME'],
                'uid': os.environ['NCCCU_POD_UID'], 'container': os.environ['NCCCU_CONTAINER'],
                'python': os.environ['NCCCU_PYTHON']}, 'gpus': cards, **paths,
        'science': {'binding_version': science['candidate_profile'], 'candidate': candidate_from_configuration(science),
                    'configuration': {'path': str(science_path), 'sha256': sha(science_path)}}})


def build(args):
    limit_cpu_threads()
    source = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    if 'technical-prep' not in output.parts or 'technical-prep' not in source.parts:
        raise ValueError('CPU build requires an explicit isolated technical-prep snapshot and output')
    relative = args.science_config
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'configs':
        raise ValueError('science configuration must be explicit and repository-relative under configs')
    science_path = source / relative
    if science_path.is_symlink() or not science_path.is_file():
        raise ValueError('science configuration must be a regular source file')
    science = read(science_path)
    sys.path[:0] = [str(source / 'runtime/bundle')]
    from retrain.site_binding import load_site, require_cpu_channel
    if bool(args.site_binding) != bool(args.site_binding_sha256):
        raise ValueError('site binding path and SHA must be supplied together')
    site = load_site(args.site_binding, expected_sha256=args.site_binding_sha256) if args.site_binding else None
    work_root = site['work_root'] if site else args.work_root
    science_reference = science_binding_path(science_path, args.science_binding_path, work_root)
    if site is None:
        site = technical_site(args, science, science_reference)
    require_cpu_channel(site)
    if site['science'].get('configuration') != {'path': str(science_reference), 'sha256': sha(science_path)}:
        raise ValueError('site science pin must identify this exact Git source configuration')
    if sha(args.asset_receipt) != args.asset_receipt_sha256:
        raise ValueError('explicit migrated-asset receipt SHA differs')
    migrated = read(args.asset_receipt)
    if (migrated.get('passed') is not True or migrated.get('phase') != 'small'
            or migrated.get('execution', {}).get('uid') != site['pod']['uid']
            or migrated.get('source_modified_by_tool') is not False
            or migrated.get('historical_FREEZE_bytes_preserved') is not True):
        raise ValueError('successful current Pod small-asset copy required')
    from retrain.experiment91 import verify_public_asset
    public_asset = verify_public_asset(science, site)
    mapping = {'data_root': 'dataset_root', 'train_identity': 'train_identity',
        'metadata_root': 'stage1_root', 'public_weight': 'public_pretrained',
        'public_provenance': 'public_metadata', 'prepared_cache': 'prepared_cache_root',
        'official_manifest': 'official_manifest'}
    if verify_fold_migration(site, science, migrated):
        mapping.pop('metadata_root')
    if public_asset:
        mapping.pop('public_weight'); mapping.pop('public_provenance')
    from retrain.preprocessing import verify_native_cache_asset,verify_composite_cache_asset
    composite=verify_composite_cache_asset(science,site)
    if verify_native_cache_asset(science,site) is not None:
        mapping.pop('prepared_cache')
    if composite is not None:
        if composite['global_cache_root']!=migrated['paths']['prepared_cache_root']:raise ValueError('global paired-cache root differs from original migration')
        mapping.pop('prepared_cache')
    if any(site[key] != migrated['paths'][field] for key, field in mapping.items()):
        raise ValueError('site asset paths differ from the exact migration binding')
    if sha(migrated['manifest']['path']) != migrated['manifest']['sha256']:
        raise ValueError('migration file-manifest SHA differs')
    entries = [json.loads(line) for line in Path(migrated['manifest']['path']).read_text().splitlines()]
    if (len(entries) != migrated['files'] or any(row['sha256'] != row['target_sha256']
            or row['source_modified_by_tool'] is not False for row in entries)):
        raise ValueError('migration source/target byte proof differs')
    identity = next(row for row in entries if row['target_path'] == site['train_identity'])
    if sha(site['train_identity']) != identity['sha256']:
        raise ValueError('current migrated train identity differs')
    output.mkdir(parents=True, exist_ok=False)
    package = output / 'package'; package.mkdir()
    prepare_local_checkpoint_storage(site, source, package)
    # Old R92 clocks, manifests, evidence and templates are never imported.
    for path in (source / 'runtime').rglob('*.py'):
        target = package / path.relative_to(source / 'runtime')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    for name in ('activate.py', 'activate_audit.py', 'site_preflight.py'):
        shutil.copyfile(source / 'tools/activation' / name, output / name)
    shutil.copyfile(source / 'runtime/PLAN.md', package / 'PLAN.md')
    shutil.copyfile(science_path, package / 'SCIENCE.json')
    if sha(package / 'SCIENCE.json') != site['science']['configuration']['sha256']:
        raise ValueError('package science bytes differ from the pinned source configuration')
    policy = read(source / 'runtime/CLOCK_POLICY.json'); policy['round_id'] = site['round_id']
    write(package / 'CLOCK_POLICY.json', policy)
    # Freeze the centralized storage limit with this formal package. Historical
    # packages without this file retain the original 12 GiB monitor limit.
    shutil.copyfile(source / 'configs/ROUND-STORAGE-POLICY.json', package / 'STORAGE_POLICY.json')
    write(package / 'SITE_BINDING.json', site)
    os.environ['NCCCU_SITE_BINDING'] = str(package / 'SITE_BINDING.json')
    os.environ['NCCCU_SITE_BINDING_SHA256'] = sha(package / 'SITE_BINDING.json')
    from retrain import experiment91 as ex
    from retrain.config import load_config
    from retrain.execution_policy import resolve_topology
    from retrain.persistence import canonical_digest
    if science['candidate_profile'] not in ex.SUPPORTED_BINDINGS or science['schedule'] != ex.schedule():
        raise ValueError('science profile or configured successful-update schedule differs')
    job = copy.deepcopy(science['job'])
    job.update(age=ex.resolved_age_spec(), experiment91=ex.binding(),
        initialization={'kind': ex.public_source().get('kind','imagenet'), 'format': ex.public_source()['format'],
            'path': site['public_weight'], 'provenance': site['public_provenance'], 'sha256': ex.public_source()['sha256']})
    f05_binding = prepare_f05_targets(source, site, science, package)
    if f05_binding is not None:
        job['f05_targets'] = f05_binding
    if job['initialization']['format'] in ('torchvision-convnext-small-dinov3-backbone-state-dict-v1',
                                          'torchvision-convnext-large-dinov3-backbone-state-dict-v1'):
        job['initialization']['expected_backbone_state_keys']=342
    if job['initialization']['format']in ('timm-vits16-dinov3-native-state-dict-v1','timm-vitb16-dinov3-native-state-dict-v1'):
        job['initialization']['expected_backbone_state_keys']=162
    cfg = {'schema_version': 1, 'mode': 'train', 'resume': False,
        'run_name': site['round_id'] + f"-8c-{job['model']}-s{job['seed']}-b64-p{job['positive_per_batch']}-fp32",
        'output_root': str(Path(site['round_root']) / 'outputs'),
        'data': {'root': site['data_root'], 'identity_path': site['train_identity'],
            'expected_count': 51396, 'expected_valid_count': 5849, 'verify_images': True,
            'split': science['data_split']},
        'execution': {'devices': ['cuda:' + str(rank) for rank in range(len(site['gpus']))],
            'cpu_cores_per_job': 1, 'grace_seconds': 150, 'heartbeat_seconds': 15,
            'legacy_six_card_exception': False, 'per_rank_microbatch': 0,
            'evaluation_physical_batch': 32, 'technical_smoke': False}, 'jobs': [job]}
    write(package / 'TRAIN_CONFIG.json', cfg)
    cfg = load_config(package / 'TRAIN_CONFIG.json', environ={})
    resolved, topology, execution = resolve_topology(cfg, available_accelerators=len(site['gpus']))
    ex.validate_job(resolved)
    if (topology['world_size'], topology['per_rank_batch'], topology['accumulation'],
            topology['reference_physical_batch']) != (8, 8, 1, 32):
        raise ValueError('world8/G64/reference32 topology differs')
    files = {str(path.relative_to(package / 'bundle')): sha(path)
             for path in sorted((package / 'bundle').rglob('*.py'))}
    source_digest = canonical_digest(files)
    write(package / 'SOURCE-MANIFEST.reference.json', {'after': files, 'source_digest': source_digest,
        'source_state': site.get('source_state'), 'git_commit': site['git_commit']})
    write(package / 'EMA-TINY.json', ex.ema_rule())
    assets = {'schema_version': 1, 'round_id': site['round_id'], 'data_root': site['data_root'],
        'age': resolved['age'], 'experiment91': ex.binding(), 'schedule': ex.schedule(), 'ema': ex.ema_rule(),
        'manifests': {'fit_dev': {'remote_path': str(Path(site['metadata_root']) / 'manifest.json'),
            'sha256': ex.binding()['manifest_sha256'], 'canonical_sha256': ex.binding()['manifest_digest']},
            'official_valid': {'remote_path': site['official_manifest'],
                              'sha256': science['official_manifest_sha256']}},
        'prepared512_cache': {'remote_root': site['prepared_cache'], 'pillow': '12.3.0',
                              'prepared_size': resolved['prepared_size'], 'read_only': True},
        'public_weight': {'remote_path': site['public_weight'], 'sha256': ex.public_source()['sha256']},
        'public_provenance': {'remote_path': site['public_provenance'], 'sha256': ex.public_source()['provenance_sha256']},
        'official_manifest_content_read': False}
    if composite is not None:
        assets['global_prepared512_cache']={'remote_root':composite['global_cache_root'],'prepared_size':512,'read_only':True,'pillow':'12.3.0'}
    if f05_binding is not None:
        assets['f05_targets'] = f05_binding
    write(package / 'ASSETS.json', assets)
    return package, cfg, resolved, topology, execution, source_digest


def make_fresh(package, job, topology, execution, source_digest, initialization):
    from retrain import experiment91 as ex
    from retrain.config import job_contract
    from retrain.precision import RULE
    site = read(package / 'SITE_BINDING.json'); cfg = read(package / 'TRAIN_CONFIG.json')
    topology['stop_after_updates'] = ex.schedule()['stop_after_updates']
    result = {'format': 'glaucoma-fresh-init-v1', 'purpose': 'train',
        'plan_version': ex.science_configuration()['candidate_profile'], 'plan_sha256': sha(package / 'PLAN.md'),
        'source_digest': source_digest, 'job': job_contract(job), 'schedule': ex.schedule(),
        'execution_policy': execution, 'ema': ex.ema_rule(), 'precision_protocol': RULE,
        'experiment91': ex.binding(), 'initialization': initialization, 'amp_initial_scale': 1.0,
        'cache_readonly_required': True, 'startup_barrier': {
            'path': str(Path(site['round_root']) / 'control/train-startup-barrier.json'),
            'round_id': site['round_id'], 'run_ids': [cfg['run_name']]}}
    ssl_recipe=ex.science_configuration().get('ssl_recipe')
    if ssl_recipe is not None:
        from retrain.ssl_stage import validate_recipe
        validate_recipe(ssl_recipe,job,ex.metadata()['manifest.json'],len(site['gpus']))
        result['ssl_recipe']=ssl_recipe
        result['ssl_startup_barrier']=str(Path(site['round_root']) / 'control/ssl-startup-barrier.json')
    return result


if __name__ == '__main__':
    package, *_ = build(parser().parse_args())
    print(json.dumps({'status': 'package_only_no_clock_no_GPU', 'package': str(package)}))

"""Build an isolated explicit package; this entry never creates a formal clock."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source-root', type=Path, required=True)
    result.add_argument('--science-config', type=Path, required=True,
                        help='explicit repository-relative versioned science configuration')
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--site-binding', type=Path)
    result.add_argument('--site-binding-sha256')
    result.add_argument('--work-root', type=Path)
    result.add_argument('--round-id')
    result.add_argument('--git-base')
    result.add_argument('--pcis')
    result.add_argument('--asset-receipt', type=Path, required=True)
    result.add_argument('--asset-receipt-sha256', required=True)
    return result


def technical_site(args, science, science_path):
    from retrain.experiment91 import CANDIDATE, BINDING_VERSION
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
        'science': {'binding_version': BINDING_VERSION, 'candidate': copy.deepcopy(CANDIDATE),
                    'configuration': {'path': str(science_path), 'sha256': sha(science_path)}}})


def build(args):
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
    site = (load_site(args.site_binding, expected_sha256=args.site_binding_sha256)
            if args.site_binding else technical_site(args, science, science_path))
    require_cpu_channel(site)
    if site['science'].get('configuration') != {'path': str(science_path), 'sha256': sha(science_path)}:
        raise ValueError('site science pin must identify this exact Git source configuration')
    if sha(args.asset_receipt) != args.asset_receipt_sha256:
        raise ValueError('explicit migrated-asset receipt SHA differs')
    migrated = read(args.asset_receipt)
    if (migrated.get('passed') is not True or migrated.get('phase') != 'small'
            or migrated.get('execution', {}).get('uid') != site['pod']['uid']
            or migrated.get('source_modified_by_tool') is not False
            or migrated.get('historical_FREEZE_bytes_preserved') is not True):
        raise ValueError('successful current Pod small-asset copy required')
    mapping = {'data_root': 'dataset_root', 'train_identity': 'train_identity',
        'metadata_root': 'stage1_root', 'public_weight': 'public_pretrained',
        'public_provenance': 'public_metadata', 'prepared_cache': 'prepared_cache_root',
        'official_manifest': 'official_manifest'}
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
    write(package / 'SITE_BINDING.json', site)
    os.environ['NCCCU_SITE_BINDING'] = str(package / 'SITE_BINDING.json')
    os.environ['NCCCU_SITE_BINDING_SHA256'] = sha(package / 'SITE_BINDING.json')
    from retrain import experiment91 as ex
    from retrain.config import load_config
    from retrain.execution_policy import resolve_topology
    from retrain.persistence import canonical_digest
    if science['candidate_profile'] != ex.BINDING_VERSION or science['schedule'] != ex.schedule():
        raise ValueError('science profile or fixed S800/6400/1600 schedule differs')
    job = copy.deepcopy(science['job'])
    job.update(age=ex.metadata()['AGE.json'], experiment91=ex.binding(),
        initialization={'kind': 'imagenet', 'format': 'torchvision-1000-state-dict-v1',
            'path': site['public_weight'], 'provenance': site['public_provenance'], 'sha256': ex.WEIGHT_SHA})
    cfg = {'schema_version': 1, 'mode': 'train', 'resume': False,
        'run_name': site['round_id'] + '-8c-tiny22k-s20261027-b64-p8-fp32',
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
        raise ValueError('world8/G64/P8/reference32 topology differs')
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
                              'prepared_size': 512, 'read_only': True},
        'public_weight': {'remote_path': site['public_weight'], 'sha256': ex.WEIGHT_SHA},
        'public_provenance': {'remote_path': site['public_provenance'], 'sha256': ex.PROVENANCE_SHA},
        'official_manifest_content_read': False}
    write(package / 'ASSETS.json', assets)
    return package, cfg, resolved, topology, execution, source_digest


def make_fresh(package, job, topology, execution, source_digest, initialization):
    from retrain import experiment91 as ex
    from retrain.config import job_contract
    from retrain.precision import RULE
    site = read(package / 'SITE_BINDING.json'); cfg = read(package / 'TRAIN_CONFIG.json')
    topology['stop_after_updates'] = ex.schedule()['stop_after_updates']
    return {'format': 'glaucoma-fresh-init-v1', 'purpose': 'train',
        'plan_version': ex.BINDING_VERSION, 'plan_sha256': sha(package / 'PLAN.md'),
        'source_digest': source_digest, 'job': job_contract(job), 'schedule': ex.schedule(),
        'execution_policy': execution, 'ema': ex.ema_rule(), 'precision_protocol': RULE,
        'experiment91': ex.binding(), 'initialization': initialization, 'amp_initial_scale': 1.0,
        'cache_readonly_required': True, 'startup_barrier': {
            'path': str(Path(site['round_root']) / 'control/train-startup-barrier.json'),
            'round_id': site['round_id'], 'run_ids': [cfg['run_name']]}}


if __name__ == '__main__':
    package, *_ = build(parser().parse_args())
    print(json.dumps({'status': 'package_only_no_clock_no_GPU', 'package': str(package)}))

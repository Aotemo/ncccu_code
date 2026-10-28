"""Inspect the pinned Tiny22K source chain on the compute Pod CPU only.

Metadata rejects known changes; current content hashes prove the bytes. Historical
receipts keep their original evidence semantics. No downloads, weight writes,
dataset reads, model construction, or GPU API calls are performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys


REPOSITORY = 'timm/convnext_tiny.fb_in22k_ft_in1k'
REVISION = 'bc48a87f119bae1b06418f5340bfa97ef75d4609'
ORIGINAL_SHA = '8cb65dc650a7fb25eaadbff3eb69559d02f83cdc5afdfb3785d1889cdc3e3518'
CONVERTED_SHA = '1ec1dd0b7045237eafe1c8f1becd58456cc811326c97ec416ae8b2e5ede860ed'
PROVENANCE_SHA = '40c14c8bd780d4feb88ae96d7e9def1ef78499daf93ae4412eff78502780233f'
README_SHA = '5829109fcf28ec11af3f28fe4477a8319a4aaf39aa67a3ab7b4499e8d1720d4d'
TRAINING_DATA = 'ImageNet-22K pretraining -> ImageNet-1K fine-tuning'
SOURCE_URL = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/pytorch_model.bin'
FILE_IDENTITY_FIELDS = ('path', 'resolved', 'bytes', 'device', 'inode', 'mtime_ns', 'ctime_ns')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            f'regular absolute file required: {path}')
    stat = path.stat()
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': stat.st_size,
            'device': stat.st_dev, 'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns}


def require_complete_identity(record):
    require(isinstance(record, dict) and all(key in record for key in FILE_IDENTITY_FIELDS),
            'incomplete historical file identity; independent new evidence required')
    require(all(isinstance(record[key], str) and Path(record[key]).is_absolute()
                for key in ('path', 'resolved'))
            and all(type(record[key]) is int for key in FILE_IDENTITY_FIELDS[2:]),
            'invalid historical file identity; independent new evidence required')


def verify_current_content(path, expected_sha):
    """Hash the current regular object, observing identity before/after the read."""
    before = file_identity(path)
    digest, count = hashlib.sha256(), 0
    stat_fields = {'bytes': 'st_size', 'device': 'st_dev', 'inode': 'st_ino',
                   'mtime_ns': 'st_mtime_ns', 'ctime_ns': 'st_ctime_ns'}
    with Path(path).open('rb') as stream:
        opened = os.fstat(stream.fileno())
        require(all(before[key] == getattr(opened, field) for key, field in stat_fields.items()),
                'opened file identity differs from the observed path')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            count += len(block)
        finished = os.fstat(stream.fileno())
        require(all(before[key] == getattr(finished, field) for key, field in stat_fields.items()),
                'file identity changed during current content verification')
    after = file_identity(path)
    require(after == before and count == before['bytes'], 'file identity/length changed during current content verification')
    actual_sha = digest.hexdigest()
    require(actual_sha == expected_sha, 'current content SHA differs from pinned source: ' + str(path))
    return dict(after, sha256=actual_sha, sha256_evidence='fresh_pod_sha256', bytes_hashed=count)


def cpu_context():
    for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        require(os.environ.get(name) == '', f'{name} must explicitly hide all GPUs')
    assigned_raw = os.environ.get('NCCCU_CPU_THREADS', '')
    require(assigned_raw.isdecimal() and int(assigned_raw) > 0, 'explicit allocated NCCCU_CPU_THREADS required')
    assigned = int(assigned_raw)
    allocation_path = os.environ.get('NCCCU_CPU_ALLOCATION', '')
    executor = os.environ.get('NCCCU_EXECUTOR_ID', '')
    require(bool(executor), 'explicit allocated executor identity required')
    require(Path(allocation_path).is_absolute() and Path(allocation_path).is_file(),
            'verified channel CPU allocation file required')
    allocation_sha = sha256(allocation_path)
    require(os.environ.get('NCCCU_CPU_ALLOCATION_SHA256') == allocation_sha,
            'current CPU allocation differs from the explicitly pinned channel SHA')
    mask_raw = os.environ.get('NCCCU_CPU_MASK', '')
    require(bool(mask_raw), 'verified channel CPU mask required')
    mask = json.loads(mask_raw)
    require(isinstance(mask, list) and bool(mask)
            and all(type(cpu) is int and cpu >= 0 for cpu in mask)
            and len(set(mask)) == len(mask), 'CPU mask must be a nonempty unique JSON integer array')
    # The owner channel validates total effective CPU, all executor shares and
    # disjoint masks. This consumer never allocates a second global 20% budget.
    affinity = sorted(os.sched_getaffinity(0))
    require(affinity == sorted(mask) and len(mask) == assigned,
            'actual affinity/mask differs from assigned CPU thread budget')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        require(os.environ.get(name) == assigned_raw, f'{name} differs from assigned CPU budget')
    require(os.getpriority(os.PRIO_PROCESS, 0) == 19, 'nice 19 required')
    require(socket.gethostname() == os.environ.get('NCCCU_POD_NAME'), 'Pod hostname differs')
    require(bool(os.environ.get('NCCCU_POD_UID')), 'explicit verified Pod UID required')
    require(os.path.realpath(sys.executable) == os.path.realpath(os.environ.get('NCCCU_PYTHON', '')),
            'verified interpreter differs')
    return {'at': datetime.now(timezone.utc).isoformat(), 'scope': 'compute_pod_cpu',
            'namespace': os.environ['NCCCU_POD_NAMESPACE'], 'pod': socket.gethostname(),
            'uid': os.environ['NCCCU_POD_UID'], 'container': os.environ['NCCCU_CONTAINER'],
            'python': sys.executable, 'python_version': sys.version.split()[0], 'nice': 19,
            'gpu_visibility': {k: os.environ[k] for k in
                               ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
            'thread_limits': {k: os.environ[k] for k in
                              ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'cpu_allocation': {'path': allocation_path, 'sha256': allocation_sha,
                               'executor_id': executor, 'threads': assigned,
                               'channel_mask': mask, 'actual_affinity': affinity,
                               'global_allocation_validated_by': 'owner CPU execution channel'},
            'dependencies': {k: importlib.metadata.version(k) for k in
                             ('torch', 'torchvision', 'numpy', 'Pillow', 'timm')}}


def configure_torch_threads(context):
    import torch
    torch.set_num_threads(context['cpu_allocation']['threads'])
    torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == context['cpu_allocation']['threads']
            and torch.get_num_interop_threads() == 1, 'actual torch thread budget differs')
    context['cpu_allocation'].update(torch_intraop_threads=torch.get_num_threads(),
                                     torch_interop_threads=torch.get_num_interop_threads())
    require(not torch.cuda.is_initialized(), 'GPU already initialized')


def read_pinned_receipt(path, expected_sha):
    require(sha256(path) == expected_sha, f'inventory receipt SHA differs: {path}')
    return json.loads(Path(path).read_text())


def inherited_file(path, record, expected_sha):
    """Reject changed metadata, then prove current bytes; stat alone is insufficient."""
    actual = file_identity(path)
    require_complete_identity(record)
    require(record.get('path') == str(path) and record.get('exists') is True,
            f'inventory path/missing differs: {path}')
    require(record.get('sha256') == expected_sha, f'prior SHA differs: {path}')
    for key in FILE_IDENTITY_FIELDS:
        require(record[key] == actual[key],
                f'current {key} differs from inventory; independent new evidence required: {path}')
    if 'sha_matches' in record:
        require(record['sha_matches'] is True, f'prior verification failed: {path}')
    return verify_current_content(path, expected_sha)


def validate_metadata(provenance, manifest, model_api, config, download, readme):
    expected = {'model': 'convnext_tiny', 'format': 'torchvision-1000-state-dict-v1',
                'source_url': SOURCE_URL, 'revision': REVISION, 'training_data': TRAINING_DATA,
                'original_sha256': ORIGINAL_SHA, 'sha256': CONVERTED_SHA,
                'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6,
                'license_file': '../original/LICENSE-apache-2.0.txt',
                'redistribution_confirmed': True}
    for key, value in expected.items():
        require(provenance.get(key) == value, f'provenance differs: {key}')
    require(manifest.get('repo') == REPOSITORY and manifest.get('revision') == REVISION,
            'source manifest repository/revision differs')
    require(model_api.get('id') == REPOSITORY and model_api.get('sha') == REVISION
            and model_api.get('private') is False and model_api.get('gated') is False,
            'archived source registry identity differs')
    require(model_api.get('cardData', {}).get('license') == 'apache-2.0', 'model card license differs')
    sibling = [s for s in model_api['siblings'] if s['rfilename'] == 'pytorch_model.bin']
    require(len(sibling) == 1 and sibling[0].get('lfs', {}).get('sha256') == ORIGINAL_SHA
            and sibling[0]['size'] == 114420885, 'registry LFS weight identity differs')
    for name in ('config.json', 'README.md'):
        entries = [s for s in manifest['files'] if s['file'] == name]
        require(len(entries) == 1 and entries[0]['url'] ==
                f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
                and entries[0]['sha256'] == provenance['source_file_hashes'][name],
                f'pinned source URL/SHA differs: {name}')
    require(download.get('repository') == REPOSITORY and download.get('revision') == REVISION
            and download.get('source_url') == SOURCE_URL and download.get('actual_sha256') == ORIGINAL_SHA
            and download.get('bytes') == 114420885 and download.get('training_data') == TRAINING_DATA,
            'download receipt differs')
    require((config.get('architecture'), config.get('num_classes'), config.get('num_features')) ==
            ('convnext_tiny', 1000, 768), 'public architecture config differs')
    cfg = config['pretrained_cfg']
    require(cfg.get('tag') == 'fb_in22k_ft_in1k' and cfg.get('num_classes') == 1000
            and cfg.get('first_conv') == 'stem.0' and cfg.get('classifier') == 'head.fc'
            and cfg.get('mean') == [.485, .456, .406] and cfg.get('std') == [.229, .224, .225]
            and cfg.get('fixed_input_size') is False, 'public non-tensor preprocessing config differs')
    require('license: apache-2.0' in readme and 'datasets:\n- imagenet-1k\n- imagenet-22k' in readme
            and 'Pretrained on ImageNet-22k and fine-tuned on ImageNet-1k by paper authors.' in readme,
            'pinned model card data/recipe/license differs')
    return expected


def inspect(args):
    context = cpu_context()
    environment = read_pinned_receipt(args.environment, args.environment_sha)
    supplement = read_pinned_receipt(args.supplement, args.supplement_sha)
    require(environment['uid'] == context['uid'] and environment['pod'] == context['pod']
            and supplement['pod'] == context['pod'], 'inventory belongs to a different Pod')
    assets = Path(args.assets)
    provenance_path = assets / 'converted/provenance.json'
    env_index = {r['path']: r for r in environment['assets']}
    original_index = {r['path']: r for r in supplement['public_original']}
    provenance_identity = inherited_file(provenance_path, env_index[str(provenance_path)], PROVENANCE_SHA)
    provenance = json.loads(provenance_path.read_text())
    identities = {}
    for name, expected_sha in provenance['source_file_hashes'].items():
        original = assets / 'original' / name
        if name == 'README.md':
            restored = Path(args.restored_readme)
            require(sha256(restored) == expected_sha == README_SHA, 'restored README is not exact original bytes')
            identities[name] = dict(file_identity(restored), sha256=README_SHA,
                                    sha256_evidence='new_file_hashed_in_this_Pod',
                                    historical_path=str(original), historical_path_exists=original.exists())
        else:
            identities[name] = inherited_file(original, original_index[str(original)], expected_sha)
    converted = assets / 'converted/convnext-tiny-tv-bc48a87f.pt'
    identities['converted'] = inherited_file(converted, env_index[str(converted)], CONVERTED_SHA)
    docs = assets / 'original'
    read = lambda name: json.loads((docs / name).read_text())
    expected = validate_metadata(provenance, read('source-manifest.json'), read('model-api.json'),
                                 read('config.json'), read('download-receipt.json'),
                                 Path(args.restored_readme).read_text())
    converter = dict(file_identity(args.historical_converter), sha256=sha256(args.historical_converter))
    require(converter['sha256'] == provenance['converter_sha256'], 'historical converter SHA differs')
    snapshot = Path(args.runtime_snapshot)
    code_files = {str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob('*.py'))}
    require('bundle/retrain/models.py' in code_files, 'fixed runtime snapshot missing models.py')
    return {'version': 'tiny22k-source-chain-v1', 'passed': True, 'execution': context,
            'candidate': expected, 'provenance': provenance_identity, 'files': identities,
            'public_pretrained_config': read('config.json')['pretrained_cfg'],
            'source_config_norm_eps': 'not present in config.json; requires actual source factory inspection',
            'source_registry': {'path': str(docs / 'model-api.json'),
                                'revision': REVISION, 'license': 'apache-2.0'},
            'readme_recovery': {'method': 'historical_local_copy_transferred_verbatim_then_Pod_SHA_verified',
                                'transport_source': args.readme_origin,
                                'destination': str(Path(args.restored_readme)), 'sha256': README_SHA,
                                'upstream_url': f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/README.md'},
            'historical_converter': converter,
            'inventory_references': {'environment': {'path': args.environment, 'sha256': args.environment_sha},
                                     'supplement': {'path': args.supplement, 'sha256': args.supplement_sha}},
            'runtime_snapshot': {'root': str(snapshot), 'git_commit': args.snapshot_commit, 'files': code_files},
            'weight_blob_hashes_recomputed': True, 'historical_assets_modified': False,
            'datasets_read': False, 'model_constructed': False, 'GPU_work': False,
            'license_scope': 'archived public package declares Apache-2.0; upstream MIT notice preserved; '
                             'no new assertion about underlying ImageNet rights or competition approval',
            'pending': ['actual source/target factory strict load and mapping',
                        'A candidate runtime integration, fresh age/optimizer and formal deployment']}


def write_new_json(path, value):
    path = Path(path)
    require(path.is_absolute() and path.parent.is_dir(), 'existing absolute output directory required')
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(path), 'sha256': sha256(path), 'passed': value['passed']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'environment', 'environment-sha', 'supplement', 'supplement-sha',
                 'restored-readme', 'readme-origin', 'historical-converter', 'runtime-snapshot',
                 'snapshot-commit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    write_new_json(args.output, inspect(args))


if __name__ == '__main__':
    main()

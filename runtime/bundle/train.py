#!/usr/bin/env python3
"""Platform entry: inspect, prepare, benchmark or train. No shell wrapper required."""
import os
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import traceback
import uuid

from retrain.config import BUNDLE, load_config, job_contract


def environment_report(cfg):
    report = {'mode': cfg['mode'], 'python': sys.version, 'executable': sys.executable,
              'data_root': cfg['data']['root'], 'output_root': cfg['output_root'], 'errors': [],
              'versions': {}, 'devices': [], 'cpu_count': os.cpu_count()}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn'):
        try:
            report['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['errors'].append('Missing dependency: '+name)
    try:
        import torch
        import torchvision
        report['torchvision_importable'] = True
        if not hasattr(torch, 'amp') or not hasattr(torch.amp, 'GradScaler'):
            report['errors'].append('Runtime must provide torch.amp.GradScaler(device_type)')
        for architecture in ('efficientnet_v2_s', 'convnext_tiny'):
            if not hasattr(torchvision.models, architecture):
                report['errors'].append('torchvision missing model: '+architecture)
        report['torch_cuda_version'], report['torch_hip_version'] = torch.version.cuda, torch.version.hip
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            report['devices'].append({'logical_index': i, 'name': prop.name, 'total_memory': prop.total_memory})
        for device in cfg['execution']['devices']:
            if device != 'cpu' and int(device.split(':')[1]) >= len(report['devices']):
                report['errors'].append('Assigned logical device unavailable: '+device)
    except Exception as exc:
        report['errors'].append(f'Cannot inspect torch runtime: {exc}')
    if hasattr(os, 'sched_getaffinity'):
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    for candidate in ('/proc/meminfo', '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/memory.max'):
        p = Path(candidate)
        if p.is_file():
            report[candidate] = p.read_text()[:4096]
    for candidate in (cfg['data']['root'], cfg['output_root'], '/dev/shm'):
        if Path(candidate).is_dir():
            report.setdefault('disk_free_bytes', {})[candidate] = shutil.disk_usage(candidate).free
    out = Path(cfg['output_root'])
    if not out.is_dir():
        report['errors'].append('Persistent OUTPUT_ROOT must already be mounted: '+str(out))
    else:
        try:
            with tempfile.TemporaryFile(dir=out) as f:
                f.write(b'write-check'); f.flush(); os.fsync(f.fileno())
        except OSError as exc:
            report['errors'].append('OUTPUT_ROOT is not writable: '+str(exc))
    return report


def source_identity():
    from retrain.persistence import canonical_digest, sha256_file
    files = [*sorted(BUNDLE.glob('*.py')), *sorted((BUNDLE/'retrain').glob('*.py'))]
    hashes = {str(p.relative_to(BUNDLE)): sha256_file(p) for p in files}
    return hashes, canonical_digest(hashes)


def _worker(args):
    spec = json.loads(Path(args.worker_spec).read_text())
    if args.cpu_affinity and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {int(c) for c in args.cpu_affinity.split(',')})
    import torch
    torch.set_num_threads(1)
    from retrain.engine import run_training
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    job = spec['job']
    job['device'] = args.device
    manifest = json.loads(Path(spec['manifest_path']).read_text())
    result = run_training(job, manifest, spec['data_root'], spec['run_dir'], spec['identity'],
                          resume=spec['resume'], mode=spec['mode'], stop_event=stop)
    return 0 if result['status'] == 'completed' else 130


def run(cfg):
    from retrain.persistence import RunLock, atomic_json, canonical_digest, sha256_file
    from retrain.data import load_source, build_manifest
    from retrain.models import validate_initialization
    from retrain.orchestration import supervise
    report = environment_report(cfg)
    splits = []
    for job in cfg['jobs']:
        split = dict(cfg['data']['split']); split.update(job['split']); splits.append(split)
    include_valid = any(s['kind'] == 'refit' and s.get('include_valid') for s in splits)
    if cfg['mode'] == 'inspect':
        try:
            source = load_source(cfg['data']['root'], cfg['data']['identity_path'], False, False,
                                 cfg['data']['expected_count'])
            report['data_check'] = {'rows': len(source['rows']), 'identity': source['identity'],
                                    'image_bytes_verified': False}
        except Exception as exc:
            report['errors'].append('Data inspection: '+str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 1 if report['errors'] else 0
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report['errors']:
        raise ValueError('Environment inspection failed; correct the listed errors before running')
    if cfg['mode'] in ('train', 'benchmark') and not cfg['data']['verify_images']:
        raise ValueError('Real training requires verify_images=true; each image must be verified before use')
    out = Path(cfg['output_root'])
    cache_directory = os.environ.get('TRAIN_VERIFICATION_CACHE') or str(out/'.data-verification-cache')
    if not Path(cache_directory).is_absolute():
        raise ValueError('TRAIN_VERIFICATION_CACHE must be an absolute persistent directory')
    os.environ['TRAIN_VERIFICATION_CACHE'] = cache_directory
    verification_mode = 'on-access' if cfg['data']['verify_images'] else 'metadata-only'
    root = out/cfg['run_name']
    with RunLock(out/('.'+cfg['run_name']+'.lock')):
        if root.exists() != cfg['resume']:
            raise ValueError('New runs require a new RUN_NAME; resume requires the existing RUN_NAME and RESUME=1')
        print(json.dumps({'event': 'data_verification_started', 'verify_images': cfg['data']['verify_images'],
                          'verification_mode': verification_mode, 'cache_directory': cache_directory}), flush=True)
        source = load_source(cfg['data']['root'], cfg['data']['identity_path'], cfg['data']['verify_images'],
                             include_valid, cfg['data']['expected_count'])
        print(json.dumps({'event': 'data_verification_ready', 'verification_mode': verification_mode,
                          'cache_directory': cache_directory, 'rows': len(source['rows']),
                          'all_images_verified': False}), flush=True)
        if (include_valid and cfg['data']['expected_valid_count'] is not None
                and len(source['valid_rows']) != cfg['data']['expected_valid_count']):
            raise ValueError('Official valid row count does not match expected_valid_count')
        manifests = [build_manifest(source, s) for s in splits]
        file_hashes, source_digest = source_identity()
        identities = {}
        for job, manifest in zip(cfg['jobs'], manifests):
            proof = validate_initialization(job['initialization'])
            parent = sha256_file(job['initial_checkpoint']) if job['initial_checkpoint'] else None
            identities[job['name']] = {'format': 'glaucoma-retrain-v1', 'source_digest': source_digest,
                'manifest_digest': canonical_digest(manifest), 'job': job_contract(job),
                'initialization': proof, 'phase_parent_sha256': parent}
        contract = {'version': 1, 'purpose': 'benchmark' if cfg['mode'] == 'benchmark' else 'train',
                    'runtime_versions': report['versions'], 'jobs': identities}
        if cfg['resume']:
            old = json.loads((root/'contract.json').read_text())
            if old != contract:
                raise ValueError('Resume rejected: source, data, initialization or scientific configuration changed')
        else:
            # Publish a complete initial contract in one rename. A killed first
            # launch must not leave a visible run that cannot be resumed.
            staging = out/('.'+cfg['run_name']+'.initializing-'+uuid.uuid4().hex)
            staging.mkdir()
            atomic_json(staging/'contract.json', contract)
            atomic_json(staging/'source-files.json', file_hashes)
            for relative in file_hashes:
                destination = staging/'code'/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(BUNDLE/relative, destination)
                with destination.open('rb') as copied:
                    os.fsync(copied.fileno())
            asset = staging/'code/assets/train_identity.json'
            asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cfg['data']['identity_path'] or BUNDLE/'assets/train_identity.json', asset)
            with asset.open('rb') as copied:
                os.fsync(copied.fileno())
            os.replace(staging, root)
            directory_fd = os.open(out, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        (root/'complete.json').unlink(missing_ok=True)
        atomic_json(root/'config.json', cfg)
        atomic_json(root/'environment.json', report)
        atomic_json(root/'status.json', {'status': 'preparing', 'mode': cfg['mode']})
        tasks = []
        affinities = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
        nslots = len(cfg['execution']['devices'])
        width = min(cfg['execution']['cpu_cores_per_job'], max(1, len(affinities)//nslots))
        slot_affinities = [affinities[s*width:(s+1)*width] for s in range(nslots)]
        for job, manifest in zip(cfg['jobs'], manifests):
            run_dir = root/job['name']; run_dir.mkdir(exist_ok=True)
            manifest_path = run_dir/'manifest.json'
            atomic_json(manifest_path, manifest)
            atomic_json(run_dir/'job.json', job)
            spec = {'job': job, 'manifest_path': str(manifest_path), 'data_root': cfg['data']['root'],
                    'run_dir': str(run_dir), 'identity': identities[job['name']],
                    'resume': cfg['resume'] and (run_dir/'checkpoints/index.json').exists(),
                    'mode': cfg['mode']}
            path = run_dir/'worker-spec.json'; atomic_json(path, spec)
            tasks.append({'name': job['name'], 'run_dir': str(run_dir), 'mode': cfg['mode'],
                          'argv': [sys.executable, '-u', str(BUNDLE/'train.py'), '--worker-spec', str(path)]})
        if cfg['mode'] == 'prepare':
            atomic_json(root/'status.json', {'status': 'prepared', 'mode': 'prepare'})
            print(f'Prepared {root}; launch MODE=train RESUME=1 with the same config.', flush=True)
            return 0
        try:
            result = supervise(tasks, cfg['execution']['devices'], root/'logs',
                               cfg['execution']['grace_seconds'], cfg['execution']['heartbeat_seconds'],
                               slot_affinities=slot_affinities)
            result['mode'] = cfg['mode']
            atomic_json(root/'status.json', result)
            if result['status'] == 'completed':
                atomic_json(root/'complete.json', result)
            return {'completed': 0, 'interrupted': 130, 'failed': 1}[result['status']]
        except BaseException as exc:
            atomic_json(root/'status.json', {'status': 'failed', 'error': str(exc), 'mode': cfg['mode']})
            raise


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=os.environ.get('TRAIN_CONFIG'))
    p.add_argument('--mode', choices=('inspect', 'prepare', 'benchmark', 'train'))
    p.add_argument('--run-name'); p.add_argument('--data-root'); p.add_argument('--output-root')
    p.add_argument('--devices', help='Visible logical IDs, e.g. 0,1; cpu is for local validation only')
    p.add_argument('--workers', type=int)
    p.add_argument('--resume', action='store_true', default=None)
    p.add_argument('--worker-spec', help=argparse.SUPPRESS)
    p.add_argument('--device', help=argparse.SUPPRESS)
    p.add_argument('--cpu-affinity', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_spec:
        return _worker(args)
    overrides = {k: getattr(args, k) for k in ('mode', 'run_name', 'data_root', 'output_root', 'devices', 'workers', 'resume')}
    return run(load_config(args.config, overrides=overrides))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}, ensure_ascii=False), file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)

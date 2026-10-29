"""One exclusive eight-card fit-only fixed-public-reference BN calibration.

This entry is deployed with the public control. It is run only after that round
has safely exited, with its own future CPU allocation and the same instance lock.
It writes 106 floating buffers, never a trained model or sample predictions.
"""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / 'runtime/bundle'))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def context(args):
    from retrain.site_binding import load_site
    site = load_site(args.execution_site, expected_sha256=args.execution_site_sha256)
    deployment = read(args.deployment)
    if (sha(args.deployment) != args.deployment_sha256
            or deployment['git_commit'] != site['git_commit']
            or deployment['pod_uid'] != site['pod']['uid']
            or deployment['files']['tools/validation/calibrate_r50_bn.py'] != sha(__file__)
            or deployment['files']['configs/R112-SCIENCE.json'] != sha(SOURCE / 'configs/R112-SCIENCE.json')
            or Path(deployment['source_root']).resolve() != SOURCE):
        raise ValueError('actual frozen calibration source/site identity differs')
    reference = Path(args.reference_root).resolve(strict=True)
    binding = read(reference / 'evidence/FORMAL_BINDING_RECEIPT.json')
    reference_site = load_site(reference / 'SITE_BINDING.json',
                               expected_sha256=binding['site_binding_sha256'])
    if (reference_site['git_commit'] != site['git_commit']
            or reference_site['pod'] != site['pod']
            or reference_site['science'] != site['science']):
        raise ValueError('public control and future calibration identities differ')
    return site, reference


def worker(args, site, reference):
    from retrain.site_binding import bind_cpu_role
    bind_cpu_role(site, 'rank', rank=args.rank)
    # Use the public arm's immutable science/data bindings; CPU execution stays
    # on this calibration's future site and was applied before any native import.
    os.environ['NCCCU_SITE_BINDING'] = str(reference / 'SITE_BINDING.json')
    os.environ['NCCCU_SITE_BINDING_SHA256'] = sha(reference / 'SITE_BINDING.json')
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    device = torch.device('cuda', args.rank); torch.cuda.set_device(device)
    from retrain import experiment91 as ex
    from retrain.bn_calibration import calibrate_fit_bn
    from retrain.config import load_config
    from retrain.data import ImageDataset
    from retrain.ema import tensor_digest
    from retrain.engine import _seed
    from retrain.models import create_model
    from retrain.precision import RULE, configure
    cfg = load_config(reference / 'TRAIN_CONFIG.json', environ={}); job = cfg['jobs'][0]
    science = ex.science_configuration()
    if job['model'] != 'resnet50_imagenet' or job['bn_statistics'] != {'kind': 'public'}:
        raise ValueError('calibration must start from the public R50 fresh arm')
    configure(RULE); _seed(job['seed'], torch.device('cpu'))
    model = create_model(job, initialize=True)
    parameters = tensor_digest(dict(model.named_parameters()))
    if (tensor_digest(model.state_dict()) != science['initial_model_tensor_sha256']
            or parameters != science['initial_parameters_sha256']
            or tensor_digest(dict(model.named_buffers())) != science['initial_buffer_sha256']):
        raise ValueError('public fresh parameter/buffer controls differ')
    manifest = ex.metadata()['manifest.json']; indices = sorted(manifest['fit_indices'])
    if len(set(indices)) != len(indices) or len(indices) != job['age']['contract']['fit_count']:
        raise ValueError('exact unique fit membership required')
    # No dev rows, labels, ages or classification head are supplied to the pass.
    local = indices[args.rank::len(site['gpus'])]
    rows = [{key: manifest['rows'][i][key] for key in ('relative_path', 'sha256')} for i in local]
    assets = read(reference / 'ASSETS.json')
    os.environ['TRAIN_IMAGE_CACHE'] = str(Path(assets['prepared512_cache']['remote_root']) / 'train/sealed')
    os.environ['TRAIN_IMAGE_CACHE_READONLY'] = '1'
    os.environ['TRAIN_VERIFICATION_CACHE'] = str(args.output / 'verification-unused-readonly')
    dataset = ImageDataset(cfg['data']['root'], rows, job['image_size'], False,
                           job['normalization'], job['prepared_size'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
        drop_last=False, num_workers=site['cpu_execution']['loader_workers'])
    model.to(device)
    spec = read(SOURCE / 'configs/R112-SCIENCE.json')['job']['bn_statistics']['spec']
    dist.init_process_group('nccl', init_method=args.rendezvous, rank=args.rank,
                            world_size=len(site['gpus']), timeout=timedelta(seconds=180))
    try:
        receipt = calibrate_fit_bn(model, (batch[0].to(device) for batch in loader), spec)
        if (not receipt['passed'] or receipt['global_images'] != len(indices)
                or receipt['local_images'] != len(local) or receipt['layer_count'] != 53
                or receipt['floating_buffer_count'] != 106 or receipt['counter_buffer_count'] != 53
                or parameters != tensor_digest(dict(model.named_parameters()))):
            raise ValueError('actual fit pass/parameter/BN aggregate proof differs')
        report = {'rank': args.rank, 'device': str(device), 'local_fit_count': len(local),
                  'parameters_sha256': parameters, 'calibration': receipt}
        write(args.output / ('rank-' + str(args.rank) + '.json'), report)
        if args.rank == 0:
            artifact = args.asset_output / 'BN-BUFFERS.pt'
            state = {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()
                     if name.endswith(('.running_mean', '.running_var'))}
            if len(state) != 106: raise ValueError('exact floating BN subset required')
            with artifact.open('xb') as stream: torch.save(state, stream)
            write(args.asset_output / 'CALIBRATION-RECEIPT.json', {
                'schema_version': 1, 'scope': 'public_r50_fit_derived_bn', 'passed': True,
                'git_commit': site['git_commit'], 'pod_uid': site['pod']['uid'],
                'deployment': {'path': str(args.deployment), 'sha256': args.deployment_sha256},
                'source': {'weight_sha256': science['public_source']['sha256'],
                    'parameters_sha256': parameters, 'public_bn_sha256': science['initial_buffer_sha256']},
                'fit': {key: job['age']['contract'][key] for key in
                        ('fit_indices_sha256', 'fit_rows_identity_sha256')},
                'artifact': {'path': str(artifact), 'sha256': sha(artifact)}, 'calibration': receipt})
    finally:
        dist.destroy_process_group()


def owner(args, site, reference):
    from retrain.site_binding import bind_cpu_role, instance_lock, verify_live, verify_gpu_cpu
    from gpu_admission import identity, wait_children
    bind_cpu_role(site, 'owner', initial=True)
    policy, _ = verify_gpu_cpu(site)
    allocation = read(policy['allocation_receipt']['path'])
    if allocation['valid_until'] < time.time() + args.max_seconds + 30:
        raise ValueError('future calibration allocation must cover the entire bounded pass and drain')
    verify_live(site, pod_uid=args.pod_uid, ready=args.ready == 'True', observed_at=args.observed_at)
    root = Path(site['work_root']).resolve()
    for path in (args.output, args.asset_output):
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError('calibration output must remain inside the explicit work root')
        path.mkdir(parents=True, exist_ok=False)
    lock = instance_lock(site); lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(reference))
        import run_round
        if run_round.device_users_before_start(): raise ValueError('devices occupied or unknown')
        cards = run_round.runtime_safety_snapshot()['cards']
        if len(cards) != 8 or any(c['busy_percent'] or c['vram_used_bytes'] >= 64 * 1024**2 for c in cards):
            raise ValueError('eight physical devices must be idle after public control safe exit')
        if read(reference / 'evidence/ROUND_RESULT.json')['status'] != 'completed':
            raise ValueError('public control must complete its original full round before calibration')
        stop = []
        def interrupted(sig, _frame): stop.append('owner_signal:' + str(sig))
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        children = []; supervision = None; started = time.time()
        try:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
            write(args.output / 'STARTED.json', {'at': started, 'deadline': started + args.max_seconds,
                'git_commit': site['git_commit'], 'pod_uid': site['pod']['uid'], 'reference_root': str(reference),
                'execution_site': str(args.execution_site), 'execution_site_sha256': args.execution_site_sha256})
            for rank in range(8):
                if stop: break
                argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:],
                        '--rank', str(rank), '--rendezvous', 'tcp://127.0.0.1:' + str(port)]
                with (args.output / ('rank-' + str(rank) + '.log')).open('x') as log:
                    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                child = {'name': 'rank-' + str(rank), 'process': process, 'identity': identity(process.pid)}
                children.append(child)
                if child['identity'] is None: raise ValueError('new calibration rank identity unknown')
            supervision = wait_children(children, started + args.max_seconds, {'site': site},
                                        stop_reason=lambda: stop[0] if stop else None)
            if supervision['status'] != 'completed' or len(children) != 8:
                raise ValueError('calibration ranks did not complete: ' + repr(supervision))
            reports = [read(args.output / ('rank-' + str(rank) + '.json')) for rank in range(8)]
            if len({r['calibration']['derived_bn_digest_after'] for r in reports}) != 1:
                raise ValueError('eight calibrated BN identities differ')
            write(args.output / 'RESULT.json', {'passed': True, 'supervision': supervision,
                'calibration': reports[0]['calibration'], 'rank_devices': [r['device'] for r in reports],
                'local_fit_counts': [r['local_fit_count'] for r in reports],
                'asset_receipt': str(args.asset_output / 'CALIBRATION-RECEIPT.json'),
                'asset_receipt_sha256': sha(args.asset_output / 'CALIBRATION-RECEIPT.json')})
        except BaseException as error:
            if supervision is None:
                supervision = wait_children(children, time.time(), {'site': site}, initial_reason=repr(error))
            write(args.output / 'FAILED.json', {'passed': False, 'error': repr(error), 'supervision': supervision})
            raise
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('execution-site', 'execution-site-sha256', 'deployment', 'deployment-sha256',
                'reference-root', 'output', 'asset-output', 'pod-uid', 'ready'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--observed-at', type=float, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--rendezvous')
    args = parser.parse_args()
    for key in ('execution_site', 'deployment', 'reference_root', 'output', 'asset_output'):
        setattr(args, key, Path(getattr(args, key)))
    site, reference = context(args)
    if len(site['gpus']) != 8 or not 0 < args.max_seconds <= 21600:
        raise ValueError('one bounded eight-card calibration required')
    if args.rank is None:
        from retrain.site_binding import prepare_native_entry
        prepare_native_entry(site)
        owner(args, site, reference)
    else:
        worker(args, site, reference)


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""One frozen model, configured1/2/4/8 ranks; six only an explicit legacy exception."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['XFORMERS_DISABLED'] = '1'


def require_frozen_valid(spec):
    from n05_gates import valid_gate
    from run_round import verify_bound
    verify_bound()
    from aggregate import sha256_file
    context = {'config_sha256': sha256_file(ROOT / 'SCIENCE_CONFIG.json'),
               'assets_sha256': sha256_file(ROOT / 'ASSETS.json'),
               'train_result_sha256': sha256_file(ROOT / 'evidence/TRAIN_RESULT.json')}
    frozen = valid_gate(ROOT, context)
    checkpoint = frozen['checkpoint']
    assets = json.loads((ROOT / 'ASSETS.json').read_text())
    if (frozen.get('runtime_binding_sha256') != sha256_file(ROOT / 'RUNTIME_BINDING.json')
            or frozen['candidate_id'] != 'J1' or spec.get('split') != 'valid'
            or Path(spec['source']).resolve() != Path(checkpoint['path']).resolve()
            or spec['source_sha256'] != checkpoint['sha256'] or spec['variant'] != checkpoint['variant']
            or Path(spec['output']).resolve() != (ROOT / 'predictions/valid/resnet50.csv').resolve()
            or Path(spec['manifest']).resolve() != Path(assets['manifests']['official_valid']['remote_path']).resolve()
            or spec['manifest_sha256'] != assets['manifests']['official_valid']['sha256']
            or spec['data_root'] != assets['data_root'] or spec['precision'] != 'fp32'
            or spec['expected_count'] != 5849):
        raise ValueError('valid evaluator arguments differ from the frozen N08 candidate')
    return frozen


def worker(args):
    import torch
    from retrain.distributed import RankContext
    from retrain.execution_policy import rank_proof
    from retrain.persistence import atomic_json, canonical_digest
    from retrain import inference
    from retrain.parallel_evaluation import evaluate
    spec = json.loads(Path(args.worker_spec).read_text())
    require_frozen_valid(spec)
    world = len(spec['devices']); rank = args.rank
    if not 0 <= rank < world or args.device != spec['devices'][rank]:
        raise ValueError('standalone evaluator rank/device differs from configured world')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(spec['cpu_masks'][rank]))
    torch.set_num_threads(1)
    ctx = RankContext(rank, world, args.device, spec['init_method'], timeout_seconds=90)
    try:
        before = time.monotonic()
        details = {}
        proofs = ctx.values(rank_proof(ctx, spec))
        if {proof['physical_pci'] for proof in proofs} != {'0000:09:00.0','0000:77:00.0','0000:85:00.0','0000:f5:00.0'}:
            raise ValueError('actual inference ranks differ from the four admitted PCI GPUs')
        if spec['split'] == 'valid':
            report = inference.run_evaluation(spec['source'], spec['data_root'], spec['output'], device=args.device,
                variant=spec['variant'], expected_count=spec['expected_count'], precision=spec['precision'],
                eval_manifest=spec['manifest'], eval_manifest_sha256=spec['manifest_sha256'], context=ctx,
                expected_source_sha256=spec['source_sha256'], telemetry=details,deadline_at=spec['deadline_at'])
        else:
            error = None
            try:
                load_started = time.monotonic()
                model, job, metadata, origin, device = inference._load_model(spec['source'], args.device, variant=spec['variant'])
                if origin.get('sha256') != spec['source_sha256']:
                    raise ValueError('loaded immutable source SHA differs from the admitted frozen checkpoint')
                manifest = json.loads(Path(spec['manifest']).read_text())
                if spec['split']=='outer':
                    if manifest.get('training_eligible') is not False or 'fit_indices' in manifest or manifest.get('training_manifest_digest')!=origin.get('manifest_digest'):
                        raise ValueError('detached outer manifest differs from frozen training identity')
                    rows,indices=manifest['rows'],list(range(len(manifest['rows'])))
                    if len(manifest.get('source_indices',[]))!=len(rows) or len(set(manifest['source_indices']))!=len(rows):
                        raise ValueError('outer canonical source indices missing or duplicated')
                else:
                    if canonical_digest(manifest) != origin.get('manifest_digest'):
                        raise ValueError('development manifest differs from checkpoint identity')
                    rows, indices = manifest['rows'], manifest['dev_indices']
                if len(indices) != spec['expected_count']:
                    raise ValueError('complete development coverage differs')
                runtime = inference._runtime(job, metadata, precision=spec['precision'])
                inference._check_precision(device, runtime['precision'])
                policy = origin.get('identity', {}).get('fresh', {}).get('execution_policy', {})
                job.update(device=args.device, precision=runtime['precision'],
                           physical_batch=policy.get('evaluation_physical_batch', job.get('physical_batch', 32)))
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                load_seconds = time.monotonic() - load_started
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
            failures = ctx.values({'rank': rank, 'error': error})
            if any(value['error'] for value in failures):
                raise RuntimeError('coordinated standalone dev setup failure: ' + str(failures))
            metrics = evaluate(model, rows, indices, spec['data_root'], job, device, ctx, spec['output'],
                variant=spec['variant'], state_identity={'source': origin, 'variant': spec['variant']}, telemetry=details)
            report = {'metrics': metrics, 'source': origin, 'split': spec['split'], 'world_size': world,
                      'actual_evaluation_physical_batch': job['physical_batch'], 'model_load_seconds_primary': load_seconds}
        memory = {'rank': rank, 'elapsed_seconds': time.monotonic() - before}
        if ctx.device.type == 'cuda':
            memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(ctx.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(ctx.device))
        current = Path('/sys/fs/cgroup/memory.current')
        if current.exists():
            memory['sampled_cgroup_memory_current'] = int(current.read_text())
        memory = ctx.values(memory)
        if ctx.primary:
            atomic_json(Path(spec['output']).with_suffix('.native.json'), {'report': report, 'telemetry': details,
                'rank_devices': proofs, 'rank_memory_and_elapsed': memory, 'source_sha256': spec['source_sha256'],
                'manifest_sha256': spec['manifest_sha256'], 'actual_GPU_count': world if ctx.device.type == 'cuda' else 0,
                'legacy_six_card_exception': spec['legacy_six_card_exception'], 'new_training': False})
        return 0
    finally:
        ctx.close()


def run(args):
    import torch
    from retrain.execution_policy import allocate_cpus
    from retrain.distributed_launcher import supervise_ranks
    from retrain.fresh import start_runtime_guard
    from retrain.persistence import RunLock, atomic_json, sha256_file
    configured = json.loads(Path(args.config).read_text())['execution'] if args.config else {}
    devices = args.devices.split(',') if args.devices else configured.get('devices', ['cpu'])
    legacy = args.legacy_six_card_exception or configured.get('legacy_six_card_exception', False)
    if devices != ['cuda:0','cuda:1','cuda:2','cuda:3'] or torch.cuda.device_count()!=4 or legacy:
        raise ValueError('N03 valid evaluation requires the same four visible DCUs')
    require_frozen_valid(vars(args))
    if args.world_size is not None and args.world_size != len(devices):
        raise ValueError('requested world and configured device list differ')
    if args.deadline_at is None or not time.time() < args.deadline_at:
        raise ValueError('an unexpired bounded evaluation deadline is required')
    if sha256_file(args.source) != args.source_sha256 or sha256_file(args.manifest) != args.manifest_sha256:
        raise ValueError('admitted immutable source/manifest SHA differs')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('standalone output already exists')
    width, masks = allocate_cpus(len(devices))
    output.parent.mkdir(parents=True,exist_ok=True)
    control = output.parent / (output.stem + '.ranks')
    stop = threading.Event()
    with RunLock(control.with_suffix('.lock')):
        control.mkdir(parents=True, exist_ok=False)
        from retrain.n05_rendezvous import new_path, finish
        rendezvous = new_path()
        spec = {key: getattr(args, key) for key in ('source', 'source_sha256', 'manifest', 'manifest_sha256',
            'data_root', 'output', 'split', 'variant', 'precision', 'expected_count', 'deadline_at')}
        spec.update(devices=devices, cpu_width=width, cpu_masks=masks, init_method='file://' + str(rendezvous),
                    legacy_six_card_exception=legacy)
        path = control / 'worker-spec.json'; atomic_json(path, spec)
        tasks = [{'name': 'rank-' + str(rank), 'argv': [sys.executable, '-u', str(Path(__file__).resolve()),
            '--worker-spec', str(path), '--rank', str(rank), '--device', device]} for rank, device in enumerate(devices)]
        guardian = start_runtime_guard(args.deadline_at, stop,
            lambda reason: atomic_json(control / 'runtime-stop.json', {'at': time.time(), 'reason': reason}))
        try:
            result = supervise_ranks(tasks, control / 'processes', 20, stop_event=stop, env=os.environ.copy())
        finally:
            stop.set(); guardian.join(timeout=2); finish(rendezvous, control / 'processes')
        atomic_json(control / 'result.json', result)
        print(json.dumps({'result': result, 'world_size': len(devices), 'output': str(output)}), flush=True)
        return 0 if result['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--devices'); parser.add_argument('--world-size', type=int)
    parser.add_argument('--legacy-six-card-exception', action='store_true')
    for name in ('source', 'source-sha256', 'manifest', 'manifest-sha256', 'data-root', 'output'):
        parser.add_argument('--' + name)
    parser.add_argument('--split', choices=('dev', 'outer', 'valid'), default='valid')
    parser.add_argument('--variant', choices=('online', 'ema'), default='ema')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'))
    parser.add_argument('--expected-count', type=int, default=5849)
    parser.add_argument('--deadline-at', type=float)
    parser.add_argument('--worker-spec'); parser.add_argument('--rank', type=int); parser.add_argument('--device')
    args = parser.parse_args()
    return worker(args) if args.worker_spec else run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        import traceback
        print(json.dumps({'status': 'failed', 'error': str(error)}), flush=True)
        traceback.print_exc()
        raise SystemExit(1)

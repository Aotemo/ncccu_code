"""Fail-closed benchmark/cycle gates and resumable single-container dispatch."""
import csv
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

from .persistence import (CheckpointStore, RunLock, atomic_json, canonical_digest,
                          sha256_file)

_OWNED = {}


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON value: ' + value)
    result = json.loads(Path(path).read_text(), parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError('JSON object required: ' + str(path))
    return result


def completed_jobs(root, mode, expected_contract=None):
    root = Path(root)
    contract, config = read_json(root/'contract.json'), read_json(root/'config.json')
    if expected_contract is not None and contract != expected_contract:
        raise ValueError('completed stage contract differs from expected source/data/config identity')
    marker = read_json(root/'complete.json')
    names = [j['name'] for j in config['jobs']]
    if (not names or len(names) != len(set(names)) or set(contract['jobs']) != set(names)
            or marker.get('status') != 'completed' or marker.get('mode') != mode
            or marker.get('exit_codes') != {n: 0 for n in names}
            or contract.get('purpose') != ('benchmark' if mode == 'benchmark' else 'train')):
        raise ValueError('missing/mismatched completed stage contract')
    jobs = {}
    for name in names:
        record = read_json(root/name/'complete.json')
        if record.get('status') != 'completed' or record.get('mode') != mode:
            raise ValueError('job is not completed in required mode: ' + name)
        jobs[name] = record
    return contract, jobs


def benchmark_gate(root, baseline, min_speedup=1.2, max_wait_fraction=.75, expected_contract=None):
    _, jobs = completed_jobs(root, 'benchmark', expected_contract)
    results = {}
    for name, record in jobs.items():
        bench, timing = record['benchmark'], record['timing']
        seconds = float(bench['seconds_per_update'])
        old = float(baseline[name]['seconds_per_update'])
        training, waiting = float(timing['training_seconds']), float(timing['data_wait_seconds'])
        if (record['successful_updates'] != 250 or bench['warmup_updates'] != 50
                or bench['measured_updates'] != 200 or not math.isfinite(seconds) or seconds <= 0
                or not math.isfinite(old) or old <= 0 or not math.isfinite(training) or training <= 0
                or not math.isfinite(waiting) or not 0 <= waiting <= training
                or old/seconds < min_speedup or waiting/training > max_wait_fraction):
            raise ValueError('benchmark throughput/data-wait gate failed: ' + name)
        memory = record.get('memory', {})
        if memory.get('total_bytes') and memory.get('peak_reserved_bytes', 0) > .85 * memory['total_bytes']:
            raise ValueError('benchmark memory margin failed: ' + name)
        results[name] = {'seconds_per_update': seconds, 'speedup': old/seconds,
                         'data_wait_fraction': waiting/training,
                         'successful_updates': record['successful_updates']}
    return {'passed': True, 'jobs': results}


def cycle_gate(root, expected_contract=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    root = Path(root)
    contract, jobs = completed_jobs(root, 'train', expected_contract)
    result = {}
    for name, record in jobs.items():
        folder = root/name
        manifest = read_json(folder/'manifest.json')
        payload = CheckpointStore(folder/'checkpoints').load('last', map_location='cpu')
        job, schedule = payload['job'], payload['schedule']
        steps = int(schedule['steps_per_cycle'])
        metrics = record.get('best_metrics')
        expected = {manifest['rows'][i]['name']: int(manifest['rows'][i]['label'])
                    for i in manifest['dev_indices']}
        if (job.get('stop_after_cycles') != 1 or record['successful_updates'] != steps
                or record['evaluated_cycle'] != 1 or payload['successful_updates'] != steps
                or payload['evaluated_cycle'] != 1 or payload.get('identity') != contract['jobs'][name]
                or payload.get('manifest_digest') != canonical_digest(manifest)
                or schedule['total_updates'] != job['cycles'] * steps
                or schedule['warmup_updates'] != job['warmup_cycles'] * steps
                or schedule.get('stop_after_updates') != steps
                or not payload.get('model') or not payload.get('optimizer', {}).get('state')
                or not {'python', 'numpy', 'torch', 'cuda'} <= set(payload.get('rng', {}))
                or (job['precision'] == 'fp16' and not payload.get('scaler'))
                or not metrics or metrics['count'] != len(expected)
                or metrics['positive_count'] != sum(expected.values())
                or metrics['negative_count'] != len(expected) - sum(expected.values())):
            raise ValueError('cycle checkpoint/validation completion gate failed: ' + name)
        covered = set(payload['coverage']['indices'])
        negatives = {i for i in manifest['fit_indices'] if not manifest['rows'][i]['label']}
        if not negatives <= covered:
            raise ValueError('cycle negative coverage is incomplete: ' + name)
        rows = list(csv.DictReader((folder/'predictions/dev-cycle-0001.csv').open()))
        if (len(rows) != len(expected) or len({r['name'] for r in rows}) != len(expected)
                or {r['name']: int(r['label']) for r in rows} != expected):
            raise ValueError('prediction count/label coverage mismatch: ' + name)
        logits = [float(r['logit']) for r in rows]
        probabilities = [float(r['probability']) for r in rows]
        labels = [int(r['label']) for r in rows]
        if (not all(math.isfinite(x) for x in logits)
                or not all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities)
                or not all(math.isfinite(float(metrics[k])) and 0 <= metrics[k] <= 1
                           for k in ['auroc', 'ap', 'pr_auc'])):
            raise ValueError('validation has nonfinite/invalid metrics: ' + name)
        auc, ap = roc_auc_score(labels, logits), average_precision_score(labels, logits)
        if abs(auc - metrics['auroc']) > 1e-10 or abs(ap - metrics['ap']) > 1e-10:
            raise ValueError('independent validation metric mismatch: ' + name)
        result[name] = {'successful_updates': steps, 'validation_count': len(rows),
                        'auroc': float(auc), 'ap': float(ap), 'negative_coverage': len(negatives),
                        'checkpoint_verified': True, 'schedule_total_updates': schedule['total_updates']}
    return {'passed': True, 'jobs': result}


def stage_expectations(plan, source_digest):
    """Derive current contracts from sealed CSV/content-group metadata."""
    from .config import load_config, job_contract
    from .data import load_source, build_manifest
    from .models import validate_initialization
    configs, expected, common = {}, {}, None
    versions = {n: importlib.metadata.version(n) for n in
                ['torch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn']}
    for phase in ['benchmark', 'cycle', 'formal']:
        cfg = load_config(plan['stages'][phase], environ={})
        if any(job.get('f05_targets') is not None for job in cfg['jobs']):
            raise ValueError('F05 targets require the registered eight-rank fresh training entry')
        mode = 'benchmark' if phase == 'benchmark' else 'train'
        if cfg['mode'] != mode or str(Path(cfg['output_root'])/cfg['run_name']) != plan['outputs'][phase]:
            raise ValueError('stage mode/output differs from expected plan contract')
        if cfg['data']['split']['kind'] != 'holdout' or cfg['data']['split']['include_valid']:
            raise ValueError('this pipeline permits only the sealed train holdout')
        source = load_source(cfg['data']['root'], cfg['data']['identity_path'], True, False,
                             cfg['data']['expected_count'])
        jobs = {}
        for job in cfg['jobs']:
            if job['initialization']['kind'] != 'random' or job['initial_checkpoint'] is not None:
                raise ValueError('every stage must start from fresh random initialization')
            if job['stop_after_cycles'] != (1 if phase == 'cycle' else 0):
                raise ValueError('stage stopping budget differs from expected contract')
            split = dict(cfg['data']['split']); split.update(job['split'])
            manifest = build_manifest(source, split)
            jobs[job['name']] = {'format': 'glaucoma-retrain-v1', 'source_digest': source_digest,
                'manifest_digest': canonical_digest(manifest), 'job': job_contract(job),
                'initialization': validate_initialization(job['initialization']), 'phase_parent_sha256': None}
        comparable = copy.deepcopy(jobs)
        for record in comparable.values():
            record['job'].pop('stop_after_cycles')
        if common is not None and common != comparable:
            raise ValueError('benchmark/cycle/formal scientific contracts differ')
        common = comparable
        configs[phase] = cfg
        expected[phase] = {'version': 1, 'purpose': 'benchmark' if mode == 'benchmark' else 'train',
                           'runtime_versions': versions, 'jobs': jobs}
    return configs, expected


def verify_resume_proof(path, identity, expected_contract, completed_root):
    proof = read_json(path)
    _, completed = completed_jobs(completed_root, 'benchmark', expected_contract)
    if proof.get('identity') != identity or set(proof.get('jobs', {})) != set(expected_contract['jobs']):
        raise ValueError('benchmark resume proof has wrong pipeline identity')
    for name, record in proof['jobs'].items():
        if (record.get('identity') != expected_contract['jobs'][name]
                or record.get('checkpoint_verified') is not True or record.get('interrupted') is not True
                or not 25 <= record.get('successful_updates', -1) < completed[name]['successful_updates']
                or record.get('attempted_updates', -1) < record['successful_updates']):
            raise ValueError('benchmark stop/resume proof is incomplete: ' + name)
    return {'passed': True, 'identity': identity}


def matching_processes(argv):
    matches = []
    proc = Path('/proc')
    if proc.is_dir():
        for directory in proc.iterdir():
            if not directory.name.isdigit() or int(directory.name) == os.getpid():
                continue
            try:
                args = directory.joinpath('cmdline').read_bytes().split(b'\0')
                args = [x.decode() for x in args if x]
                if args[1:] == list(argv)[1:] and args:
                    matches.append(int(directory.name))
            except (OSError, UnicodeError):
                pass
    else:
        # Local macOS test/inspection fallback. Production containers use /proc.
        output = subprocess.check_output(['ps', '-axww', '-o', 'pid=', '-o', 'command='], text=True)
        suffix = shlex.join(list(argv)[1:])
        for line in output.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) == 2 and fields[1].endswith(' ' + suffix) and int(fields[0]) != os.getpid():
                matches.append(int(fields[0]))
    return matches


def launch_stage(argv, state_path, log_path, env=None):
    """Persist intent before spawn; reconcile a crash before PID publication."""
    state_path, log_path = Path(state_path), Path(log_path)
    if state_path.exists() and read_json(state_path).get('argv') != list(argv):
        raise ValueError('stage launch command differs from its persistent intent')
    matches = matching_processes(argv)
    if len(matches) > 1:
        raise RuntimeError('multiple matching stage processes; refusing another launch')
    state = {'argv': list(argv), 'host': socket.gethostname(), 'at': time.time()}
    if matches:
        state.update(status='running', pid=matches[0], attached=True)
        atomic_json(state_path, state)
        return state
    state.update(status='starting')
    atomic_json(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab') as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)
    _OWNED[proc.pid] = proc
    state.update(status='running', pid=proc.pid, attached=False)
    atomic_json(state_path, state)
    return state


def alive(pid):
    if pid in _OWNED:
        return _OWNED[pid].poll() is None
    try:
        os.kill(pid, 0)
        path = Path('/proc')/str(pid)/'stat'
        return not path.is_file() or path.read_text().split()[2] != 'Z'
    except ProcessLookupError:
        return False


def run_pipeline(plan_path):
    import train
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    _, digest = train.source_identity()
    if plan['source_digest'] != digest:
        raise ValueError('pipeline source seal mismatch')
    for path, expected in plan['file_sha256'].items():
        if sha256_file(path) != expected:
            raise ValueError('pipeline configuration/evidence seal mismatch: ' + path)
    identity = canonical_digest(plan)
    control = Path(plan['control_root'])
    control.mkdir(parents=True, exist_ok=True)
    baseline_records = read_json(plan['baseline'])
    baseline = {n: {'seconds_per_update': d['benchmark']['seconds_per_update']}
                for n, d in baseline_records.items()}
    os.environ['TRAIN_VERIFICATION_CACHE'] = plan['verification_cache']
    configs, expected = stage_expectations(plan, digest)
    stop = threading.Event()
    previous = {s: signal.signal(s, lambda *_: stop.set()) for s in [signal.SIGTERM, signal.SIGINT]}
    try:
        with RunLock(control/'pipeline.lock'):
            state_path = control/'pipeline-state.json'
            if state_path.exists() and read_json(state_path)['identity'] != identity:
                raise ValueError('pipeline identity changed; use a new control root')
            for phase in ['benchmark', 'cycle', 'formal']:
                config_path = Path(plan['stages'][phase])
                cfg = configs[phase]
                root = Path(cfg['output_root'])/cfg['run_name']
                argv = [sys.executable, '-u', str(Path(train.__file__).resolve()),
                        '--config', str(config_path), '--mode', cfg['mode'], '--run-name', cfg['run_name']]
                if phase == 'cycle':
                    verify_resume_proof(control/'benchmark-resume-proof.json', identity,
                                        expected['benchmark'], plan['outputs']['benchmark'])
                    receipt = benchmark_gate(Path(plan['outputs']['benchmark']), baseline,
                                             expected_contract=expected['benchmark'])
                    atomic_json(control/'benchmark-gate.json', receipt)
                elif phase == 'formal':
                    receipt = cycle_gate(Path(plan['outputs']['cycle']), expected['cycle'])
                    atomic_json(control/'cycle-gate.json', receipt)
                if (root/'complete.json').is_file():
                    completed_jobs(root, cfg['mode'], expected[phase])
                    continue
                if (root/'status.json').is_file() and read_json(root/'status.json').get('status') == 'failed':
                    raise ValueError('stage has recorded failure: ' + phase)
                env = os.environ.copy()
                env.update(MODE=cfg['mode'], RUN_NAME=cfg['run_name'],
                           RESUME='1' if (root/'contract.json').is_file() else '0',
                           TRAIN_CONFIG=str(config_path), TRAIN_IMAGE_CACHE=plan['cache_root'],
                           TRAIN_VERIFICATION_CACHE=plan['verification_cache'], PYTHONDONTWRITEBYTECODE='1')
                launched = launch_stage(argv, control/(phase+'-launch.json'),
                                        control/(phase+'-console.log'), env)
                pid, stopping_at, last_report = launched['pid'], None, 0
                restart_file = control/'benchmark-resume-proof.json'
                restarting = False
                while alive(pid):
                    if stop.is_set() and stopping_at is None:
                        os.kill(pid, signal.SIGTERM)
                        stopping_at = time.monotonic()
                    if phase == 'benchmark' and not restart_file.exists() and not restarting:
                        records = []
                        for name in [j['name'] for j in cfg['jobs']]:
                            try:
                                records.append(read_json(root/name/'status.json'))
                            except (OSError, ValueError):
                                pass
                        if (len(records) == len(cfg['jobs'])
                                and all(30 <= x.get('successful_updates', 0) < 250 for x in records)):
                            os.kill(pid, signal.SIGTERM)
                            restarting = True
                            stopping_at = time.monotonic()
                    if stopping_at is not None and time.monotonic() - stopping_at > 180:
                        raise RuntimeError('stage did not stop within its safe-save grace period')
                    if time.monotonic() - last_report > 10:
                        jobs = {}
                        for name in [j['name'] for j in cfg['jobs']]:
                            try:
                                d = read_json(root/name/'status.json')
                                jobs[name] = {k: d[k] for k in ['status', 'successful_updates', 'evaluated_cycle']}
                            except (OSError, ValueError, KeyError):
                                jobs[name] = {'status': 'starting'}
                        state = {'identity': identity, 'phase': phase, 'status': 'running',
                                 'pid': pid, 'host': socket.gethostname(), 'at': time.time(), 'jobs': jobs}
                        atomic_json(state_path, state)
                        print(json.dumps(state), flush=True)
                        last_report = time.monotonic()
                    time.sleep(1)
                if stop.is_set():
                    atomic_json(state_path, {'identity': identity, 'phase': phase, 'status': 'interrupted'})
                    return 130
                if restarting:
                    proof = {}
                    for name in [j['name'] for j in cfg['jobs']]:
                        payload = CheckpointStore(root/name/'checkpoints').load(map_location='cpu')
                        if not 25 <= payload['successful_updates'] < 250 or not payload['optimizer']['state']:
                            raise ValueError('benchmark safe-stop checkpoint is incomplete')
                        interrupted = read_json(root/name/'status.json')
                        if (payload['identity'] != expected[phase]['jobs'][name]
                                or interrupted.get('status') != 'interrupted'):
                            raise ValueError('benchmark safe-stop identity/status mismatch')
                        proof[name] = {'successful_updates': payload['successful_updates'],
                                       'attempted_updates': payload['attempted_updates'], 'checkpoint_verified': True,
                                       'identity': payload['identity'], 'interrupted': True}
                    atomic_json(restart_file, {'identity': identity, 'jobs': proof})
                    env['RESUME'] = '1'
                    # Re-enter the same fixed stage; the source/config identities
                    # and completion gate are unchanged after the resume proof.
                    launched = launch_stage(argv, control/(phase+'-launch.json'),
                                            control/(phase+'-console.log'), env)
                    pid = launched['pid']
                    while alive(pid):
                        if stop.is_set():
                            os.kill(pid, signal.SIGTERM)
                            return 130
                        time.sleep(2)
                completed_jobs(root, cfg['mode'], expected[phase])
                atomic_json(control/(phase+'-verified.json'),
                            {'identity': identity, 'phase': phase, 'completed_at': time.time()})
            atomic_json(state_path, {'identity': identity, 'phase': 'complete', 'status': 'completed'})
            return 0
    except BaseException as exc:
        atomic_json(control/'pipeline-error.json', {'identity': identity, 'error': str(exc), 'at': time.time()})
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

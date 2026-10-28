"""One frozen eight-rank initialization/FP32 collective admission: zero steps, forwards or samples.

Only the standard-library outer auditor publishes GPU-ADMISSION.json, after its
owner child and all observed ranks have actually exited. Every attempt has one
exclusive technical-prep directory. No formal training clock/token is created.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time

VISIBLE = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')
ZERO_COUNTS = ('successful_updates', 'attempted_updates', 'optimizer_steps', 'forward_calls', 'data_samples_read')
ZERO_FLAGS = ('data_images_read', 'csv_read', 'cache_read', 'dev_evaluated', 'official_evaluated')
INPUT_NAMES = ('prep_root', 'site_binding', 'site_binding_sha256', 'package_manifest',
    'package_manifest_sha256', 'deployment_receipt', 'deployment_receipt_sha256',
    'cpu_prep_receipt', 'cpu_prep_receipt_sha256', 'pod_uid', 'ready', 'observed_at')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(path, root=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and not path.is_symlink(), 'explicit nonsymlink absolute path required')
    if root is not None:
        base = Path(root).resolve()
        require(path.resolve().is_relative_to(base) and path.resolve() != base, 'path escapes the explicit work root')
    return path


def pinned(path, digest, root=None):
    path = safe_path(path, root)
    require(re.fullmatch(r'[0-9a-f]{64}', digest or '') and sha(path) == digest, 'input SHA differs: ' + str(path))
    return json.loads(path.read_text())


def write(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication; an existing marker can never be replaced.
    finally:
        Path(temporary).unlink(missing_ok=True)


def zero_work():
    return {**{key: 0 for key in ZERO_COUNTS}, **{key: False for key in ZERO_FLAGS}}


def validate_zero_work(value):
    require(isinstance(value, dict) and all(type(value.get(key)) is int and value[key] == 0 for key in ZERO_COUNTS)
            and all(value.get(key) is False for key in ZERO_FLAGS), 'admission must have zero steps, forwards, samples and evaluations')


def validate_contract(site, manifest, deployment, preparation, cpu, *, manifest_sha):
    require(site.get('schema_version') == 1 and site.get('scope') == 'formal_exp_v3'
            and site.get('source_state', {}).get('kind') != 'uncommitted_cpu_snapshot'
            and re.fullmatch(r'[0-9a-f]{40}', site.get('git_commit', '') or ''), 'formal committed site required')
    require(manifest.get('schema_version') == 1 and manifest.get('git_commit') == site['git_commit']
            and len(site.get('gpus', [])) == 8
            and [card.get('ordinal') for card in site['gpus']] == list(range(8))
            and len({card['pci'] for card in site['gpus']}) == 8, 'package Git/eight-rank identity differs')
    require(deployment.get('schema_version') == 1 and deployment.get('status') == 'frozen_from_remote_commit'
            and deployment.get('scope') == 'formal_exp_v3' and deployment.get('git_commit') == site['git_commit']
            and deployment.get('remote_commit') == site['git_commit']
            and deployment.get('remote_url') == 'https://github.com/Aotemo/ncccu_code'
            and deployment.get('pod_uid') == site['pod']['uid']
            and deployment.get('package_manifest_sha256') == manifest_sha, 'remote Git/Pod/package evidence differs')
    source = manifest.get('bundle_source_digest')
    require(re.fullmatch(r'[0-9a-f]{64}', source or '')
            and preparation.get('status') == 'prepared_no_formal_clock_no_GPU' and preparation.get('exit_code') == 0
            and preparation.get('package_manifest_sha256') == manifest_sha
            and preparation.get('source_digest') == source and preparation.get('GPU_started') is False
            and preparation.get('formal_clock_created') is False, 'CPU preparation/package identity differs')
    expected = cpu.get('initial_state_control', {})
    require(cpu.get('status') == 'completed' and cpu.get('exit_code') == 0
            and cpu.get('source_digest') == source == expected.get('source_digest')
            and expected == preparation.get('initial_state_control') and expected.get('candidate') == site['science']['candidate']
            and re.fullmatch(r'[0-9a-f]{64}', expected.get('model_tensor_sha256', '') or '')
            and expected.get('optimizer_state_entries') == expected.get('batch_norm_modules') == expected.get('buffer_count') == 0
            and expected.get('no_attention_q') is True and expected.get('candidate_normalization')
            and expected.get('trainable_tensors') == 183
            and expected.get('parameter_elements_by_role') == {'backbone': 27820128, 'head': 769, 'age': 2}
            and cpu.get('gpu_initialized') is False and cpu.get('formal_train_config_sha256') is None
            and cpu.get('checks') and all(item.get('passed') is True for item in cpu['checks'])
            and preparation.get('cpu_receipt_sha256') == cpu.get('original_receipt_sha256')
            and cpu.get('training_updates') == 0 and cpu.get('data_images_read') is False
            and cpu.get('official_semantics_read') is False and cpu.get('formal_clock_created') is False
            and cpu.get('training_lock_acquired') is False, 'production CPU initial-state evidence differs')
    topology = cpu.get('topology', {})
    require(tuple(topology.get(key) for key in ('world_size', 'per_rank_batch', 'accumulation',
            'reference_physical_batch', 'global_effective_batch', 'global_positive_quota')) == (8, 8, 1, 32, 64, 8),
            'CPU admission topology differs')
    return expected


def verify_entry(deployment, entry, work_root):
    source = safe_path(deployment['source_root'], work_root)
    frozen = safe_path(source / 'tools/validation/gpu_admission.py', source)
    entry = safe_path(entry)
    digest = deployment['files'].get('tools/validation/gpu_admission.py')
    require(re.fullmatch(r'[0-9a-f]{64}', digest or '') and sha(entry) == sha(frozen) == digest,
            'the executed GPU admission tool is not the pinned Git deployment')


def reserve_output(path, work_root):
    path = safe_path(path, work_root)
    require(path.resolve().is_relative_to(Path(work_root).resolve() / 'technical-prep'),
            'GPU admission output must be a new technical-prep directory')
    if path.exists() or path.is_symlink():
        raise FileExistsError('attempt directory already exists; do not retry: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def identity(pid):
    try:
        process = Path('/proc') / str(pid)
        fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]), 'pgrp': int(fields[2]),
                'start_ticks': fields[19], 'argv': [part.decode('utf-8', 'replace')
                    for part in (process / 'cmdline').read_bytes().split(b'\0') if part]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def exact(expected, actual):
    return bool(actual and actual.get('pid') == expected.get('pid')
        and str(actual.get('start_ticks')) == str(expected.get('start_ticks'))
        and actual.get('argv') == expected.get('argv'))


def signal_exact(expected, signum):
    actual = identity(expected['pid'])
    if not exact(expected, actual) or actual['state'] in ('Z', 'X'):
        return {'pid': expected['pid'], 'signal': signum, 'sent': False, 'actual': actual}
    try:
        os.kill(expected['pid'], signum)
        return {'pid': expected['pid'], 'signal': signum, 'sent': True}
    except ProcessLookupError:
        return {'pid': expected['pid'], 'signal': signum, 'sent': False, 'actual': None}


def verify_exits(owner_exit, rank_exits, identities, observations, *, auditor_pid):
    require(type(owner_exit) is int and owner_exit == 0
            and set(rank_exits) == {'rank-' + str(rank) for rank in range(8)}
            and all(type(code) is int and code == 0 for code in rank_exits.values()), 'owner/every rank must actually exit0')
    require(len(identities) >= 9 and len({item['pid'] for item in identities}) == len(identities)
            and auditor_pid not in {item['pid'] for item in identities}, 'live auditor must not claim its own exit')
    for item in identities:
        require(type(item.get('pid')) is int and item['pid'] > 0 and str(item.get('start_ticks', '')).isdecimal()
                and isinstance(item.get('argv'), list) and item['argv']
                and item['pid'] in observations, 'exact process exit observation is missing')
        actual = observations[item['pid']]
        if actual is not None:
            require(isinstance(actual, dict) and actual.get('pid') == item['pid']
                    and str(actual.get('start_ticks', '')).isdecimal(), 'process exit observation is unknown')
            require(str(actual['start_ticks']) != str(item['start_ticks']),
                    'an owner/rank/helper birth identity still exists, including an unexpected exec')


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_context(inputs):
    """Pin the two stdlib helpers before importing any deployed code."""
    prep = safe_path(inputs['prep_root']); package = prep / 'package'
    site = pinned(inputs['site_binding'], inputs['site_binding_sha256'])
    root = safe_path(site['work_root'])
    require(site.get('scope') == 'formal_exp_v3' and site.get('source_state', {}).get('kind') != 'uncommitted_cpu_snapshot',
            'CPU technical snapshots cannot enter GPU admission')
    require(Path(inputs['site_binding']).resolve() == package / 'SITE_BINDING.json'
            and Path(inputs['package_manifest']).resolve() == prep / 'PACKAGE-MANIFEST.json'
            and Path(inputs['cpu_prep_receipt']).resolve() == prep / 'CPU-PREP-RECEIPT.json',
            'explicit inputs must name this one preparation package')
    safe_path(prep, root)
    manifest = pinned(inputs['package_manifest'], inputs['package_manifest_sha256'], root)
    deployment = pinned(inputs['deployment_receipt'], inputs['deployment_receipt_sha256'], root)
    preparation = pinned(inputs['cpu_prep_receipt'], inputs['cpu_prep_receipt_sha256'], root)
    # Scope/Git/Pod rejection precedes importing package helpers.
    require(deployment.get('scope') == 'formal_exp_v3' and deployment.get('status') == 'frozen_from_remote_commit'
            and deployment.get('git_commit') == deployment.get('remote_commit') == site.get('git_commit')
            and deployment.get('pod_uid') == site['pod']['uid']
            and deployment.get('package_manifest_sha256') == inputs['package_manifest_sha256'],
            'successful remote Git/frozen deployment required before importing helpers')
    verify_entry(deployment, Path(__file__).resolve(), root)
    source = safe_path(deployment['source_root'], root)
    for relative in ('bundle/retrain/__init__.py', 'bundle/retrain/site_binding.py'):
        digest = manifest['files'].get(relative)
        path = safe_path(package / relative, package)
        original = safe_path(source / 'runtime' / relative, source)
        require(re.fullmatch(r'[0-9a-f]{64}', digest or '')
                and deployment['files'].get('runtime/' + relative) == digest
                and sha(path) == sha(original) == digest, 'unverified site helper cannot be imported')
    activation_path = safe_path(prep / 'activate.py', prep)
    require(sha(activation_path) == manifest['activate_script_sha256']
            == deployment['files'].get('tools/activation/activate.py')
            == sha(safe_path(source / 'tools/activation/activate.py', source)),
            'unverified activation helper cannot be imported')
    require(manifest['files'].get('SITE_BINDING.json') == inputs['site_binding_sha256'], 'site pin differs from package manifest')
    require(not any(name == 'retrain' or name.startswith('retrain.') for name in sys.modules),
            'admission helpers require a fresh interpreter, without a different cached runtime')
    sys.path.insert(0, str(package / 'bundle'))
    from retrain import site_binding as api
    site = api.load_site(inputs['site_binding'], expected_sha256=inputs['site_binding_sha256'])
    api.verify_live(site, pod_uid=inputs['pod_uid'], ready=inputs['ready'] == 'True', observed_at=inputs['observed_at'])
    activation = import_file('_pinned_zero_work_activation', activation_path)
    source_digest = activation.verify_package(package, manifest, site, activation_script=activation_path)
    activation.verify_deployment(deployment, site, manifest, inputs['package_manifest_sha256'], package, prep)
    cpu = json.loads((package / 'evidence/MODEL_CPU_ADMISSION.template.json').read_text())
    validate_contract(site, manifest, deployment, preparation, cpu, manifest_sha=inputs['package_manifest_sha256'])
    original = pinned(cpu['original_receipt_path'], cpu['original_receipt_sha256'], root)
    require(original.get('exit_code') == 0 and original.get('status') == 'completed'
            and original.get('source_digest') == source_digest
            and original.get('initial_state_control') == cpu['initial_state_control']
            and original.get('gpu_initialized') is False and original.get('training_updates') == 0,
            'original successful production CPU receipt differs')
    config = json.loads((package / 'TRAIN_CONFIG.json').read_text())
    require(config['mode'] == 'train' and config['resume'] is False and len(config['jobs']) == 1
            and config['execution']['devices'] == ['cuda:' + str(rank) for rank in range(8)]
            and config['jobs'][0]['initialization']['kind'] == 'imagenet'
            and config['jobs'][0].get('initial_checkpoint') is None, 'fresh single-model eight-rank config required')
    activation.ensure_new_activation(prep, site['round_root'])
    require(all(os.environ.get(key) == '0,1,2,3,4,5,6,7' for key in VISIBLE),
            'GPU entry needs explicit ordinal visibility; CPU-hidden devices are forbidden')
    os.environ.update(NCCCU_SITE_BINDING=str(inputs['site_binding']),
                      NCCCU_SITE_BINDING_SHA256=inputs['site_binding_sha256'], PYTHONDONTWRITEBYTECODE='1')
    return {'api': api, 'activation': activation, 'site': site, 'cpu': cpu, 'manifest': manifest,
            'deployment': deployment, 'package': package, 'source_digest': source_digest}


def read_spec(args):
    spec = pinned(args.spec, args.spec_sha256)
    output = safe_path(spec['output'])
    require(Path(args.spec).resolve() == output / 'SPEC.json' and spec['schema_version'] == 1
            and spec['kind'] == 'eight-rank-zero-forward-initialization'
            and spec['entry_sha256'] == sha(__file__), 'child specification/source identity differs')
    validate_zero_work(spec['zero_work'])
    require(type(spec['started_at']) in (int, float) and math.isfinite(spec['started_at'])
            and spec['deadline_at'] == spec['started_at'] + 240
            and spec['owner_wait_until'] == spec['started_at'] + 270
            and spec['hard_deadline_at'] == spec['started_at'] + 300
            and time.time() < spec['deadline_at'], 'original bounded admission deadline expired')
    endpoint = re.fullmatch(r'tcp://127\.0\.0\.1:([1-9][0-9]{0,4})', spec['init_method'])
    require(endpoint and int(endpoint.group(1)) <= 65535, 'only the one reserved loopback TCP rendezvous is admitted')
    started = pinned(output / 'STARTED.json', spec['started_sha256'])
    require(started['auditor'] == spec['auditor'], 'child belongs to a different outer audit attempt')
    return spec


def wait_release(spec, spec_sha, role, rank=None):
    output = Path(spec['output'])
    name = 'owner-launch.json' if role == 'owner' else 'rank-' + str(rank) + '-launch.json'
    gate = output / name
    until = min(time.time() + 5, spec['deadline_at'])
    while not gate.exists():
        require(not gate.is_symlink() and time.time() < until, 'exact child release was not published')
        time.sleep(.02)
    require(not gate.is_symlink(), 'child release symlink forbidden')
    value = json.loads(gate.read_text()); actual = identity(os.getpid())
    parent = spec['auditor'] if role == 'owner' else json.loads((output / 'owner-launch.json').read_text())['identity']
    require(value.get('spec_sha256') == spec_sha and value.get('role') == role and value.get('rank') == rank
            and exact(value['identity'], actual) and actual['ppid'] == parent['pid']
            and exact(parent, identity(parent['pid'])), 'child PID/start/argv/parent release differs')


def launch_child(argv, name, output, spec_sha, *, role, rank=None):
    with (output / (name + '.log')).open('x') as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            cwd=output, env=os.environ.copy(), start_new_session=True)
    actual = identity(process.pid)
    require(actual and actual['argv'] == argv and actual['pgrp'] == process.pid, 'new child identity could not be established')
    write(output / (name + '-launch.json'), {'at': time.time(), 'identity': actual,
        'spec_sha256': spec_sha, 'role': role, 'rank': rank})
    return {'name': name, 'process': process, 'identity': actual}


def audit_owned_tree(context, known):
    from retrain.execution_policy import audit_cpu_tree
    return audit_cpu_tree(os.getpid(), known, context['site'])


def wait_children(children, deadline, context, *, stop_reason=None, initial_reason=None,
                  kill_after=10, grace_seconds=25, until=None):
    """Bound only these new exact children; retain every observed descendant identity."""
    known = {}; observed = {}; signals = []; reason = initial_reason; term_at = None; kill_at = None; checked_at = 0
    while True:
        requested = stop_reason() if stop_reason is not None else None
        reason = reason or requested
        codes = {child['name']: child['process'].poll() for child in children}
        if reason is None and any(code is not None and code != 0 for code in codes.values()):
            reason = 'child_exit_nonzero'
        if all(code is not None for code in codes.values()):
            break
        now = time.time()
        if reason is None and now >= deadline:
            reason = 'absolute_deadline'
        if now - checked_at >= 1:
            try:
                snapshot = audit_owned_tree(context, known)
                for pid in snapshot['live_pids']:
                    if pid != os.getpid():
                        current = identity(pid)
                        if current and current['state'] not in ('Z', 'X') and current['argv']:
                            observed[(pid, current['start_ticks'])] = current
            except Exception as error:
                reason = reason or 'CPU_tree_unknown:' + repr(error)
            checked_at = now
        if reason is not None and term_at is None:
            for child in children:
                try:
                    signals.append(signal_exact(child['identity'], signal.SIGTERM))
                except Exception as error:
                    signals.append({'pid': child['identity']['pid'], 'signal': signal.SIGTERM,
                                    'sent': False, 'error': repr(error)})
            term_at = now
        stop_at = min(term_at + grace_seconds, until if until is not None else float('inf')) if term_at is not None else None
        if term_at is not None and now >= min(term_at + kill_after, stop_at - 1) and kill_at is None:
            for child in children:
                try:
                    signals.append(signal_exact(child['identity'], signal.SIGKILL))
                except Exception as error:
                    signals.append({'pid': child['identity']['pid'], 'signal': signal.SIGKILL,
                                    'sent': False, 'error': repr(error)})
            kill_at = now
        if term_at is not None and now >= stop_at:
            break
        time.sleep(.05)
    codes = {}; reaped = {}; reap_errors = {}; live = []
    for child in children:
        name = child['name']; codes[name] = child['process'].poll(); reaped[name] = False
        try:
            codes[name] = child['process'].wait(timeout=0)
            reaped[name] = True
        except (subprocess.TimeoutExpired, ChildProcessError, OSError) as error:
            reap_errors[name] = repr(error); live.append(child['identity'])
    status = ('resource_unknown' if not all(reaped.values()) else
              'completed' if reason is None and all(code == 0 for code in codes.values()) else 'failed')
    return {'exit_codes': codes, 'reason': reason, 'signals': signals,
        'observed_processes': list(observed.values()), 'reaped': reaped, 'reap_errors': reap_errors,
        'live_owned_children': live, 'status': status}


def data_guard(site):
    """Allow scientific/source metadata, forbid every sample/CSV/cache read."""
    roots = [Path(site[key]).resolve() for key in ('data_root', 'prepared_cache')]
    official = Path(site['official_manifest']).resolve()
    counters = {'forbidden_read_attempts': 0}
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if (path == official or any(path.is_relative_to(root) for root in roots)
                or path.suffix.lower() in ('.csv', '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
            counters['forbidden_read_attempts'] += 1
            raise PermissionError('zero-work admission forbids sample/CSV/cache access: ' + str(path))
    sys.addaudithook(audit)
    return counters


def run_rank(args, spec, context):
    api = context['api']; site = context['site']; rank = args.rank
    require(type(rank) is int and 0 <= rank < 8, 'invalid admitted rank')
    api.bind_cpu_role(site, 'rank', rank=rank)  # Before any torch/TV/numpy import.
    reads = data_guard(site)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from retrain.config import load_config, job_contract
    from retrain.execution_policy import resolve_topology, rank_proof
    from retrain.distributed import RankContext
    from retrain.engine import _seed, _optimizer
    from retrain.models import create_model
    from retrain import experiment91 as experiment
    from retrain.ema import EMA, tensor_digest, validate_state
    from retrain.precision import RULE, configure, runtime_proof
    configure(RULE)
    config = load_config(context['package'] / 'TRAIN_CONFIG.json', environ={})
    job, topology, _ = resolve_topology(config, available_accelerators=8)
    topology['stop_after_updates'] = 6400
    require(job_contract(job) == context['cpu']['job_contract'] and topology == context['cpu']['topology'],
            'native job/topology differs from CPU proof')
    require(torch.cuda.device_count() == 8, 'eight actual visible accelerators required')
    ctx = RankContext(rank, 8, 'cuda:' + str(rank), spec['init_method'],
                      timeout_seconds=max(1, min(90, int(spec['deadline_at'] - time.time()))))
    try:
        proof = rank_proof(ctx, {'cpu_masks': site['cpu_execution']['rank_masks']})
        _seed(job['seed'], ctx.device)
        model = create_model(job, initialize=True)
        public = model.public_initialization_receipt
        require(public and public['all_public_tensors_exact'] and public['all_public_tensors_FP32_finite']
                and public['public_tensor_count'] == 182 and public['full_head_shape'] == [1000, 768]
                and not public['strict_missing_keys'] and not public['strict_unexpected_keys'],
                'native public1000 strict/finite identity differs')
        expected = context['cpu']['initial_state_control']; work = zero_work()
        def reject_forward(*_args, **_kwargs):
            work['forward_calls'] += 1
            raise RuntimeError('model forward is prohibited in this initialization-only admission')
        for module in model.modules():
            module.register_forward_pre_hook(reject_forward)
        require(experiment.describe_model(model) == expected['candidate_normalization']
                and tensor_digest(model.state_dict()) == expected['model_tensor_sha256']
                and not dict(model.named_buffers()) and bool((model.age_weight == 0).all()),
                'native CPU model/normalization/age identity differs')
        model = model.to(ctx.device)
        optimizer = _optimizer(model, job)
        def reject_step(*_args, **_kwargs):
            work['optimizer_steps'] += 1
            raise RuntimeError('optimizer step is prohibited in this admission')
        optimizer.step = reject_step
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        groups = [{'role': group['role'], 'lr': group['lr'], 'initial_lr': group['initial_lr'],
            'weight_decay': group['weight_decay'], 'parameters': [
                {'name': names[id(parameter)], 'shape': list(parameter.shape), 'numel': parameter.numel()}
                for parameter in group['params']]} for group in optimizer.param_groups]
        reference_groups = json.loads((context['package'] / 'evidence/CPU_PARAMETER_GROUPS.json').read_text())
        require(groups == reference_groups and not optimizer.state, 'native GPU parameter roles/empty Adam differ')
        ema = EMA(model, experiment.ema_rule())
        state = model.state_dict(); ema_state = ema.state_dict()
        validate_state(ema_state, state, 0, experiment.ema_rule())
        finite = all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in state.values())
        digest = tensor_digest(state); ema_digest = tensor_digest(ema.model.state_dict())
        require(finite and digest == ema_digest == expected['model_tensor_sha256']
                and not optimizer.state and int(torch.count_nonzero(model.age_weight)) == 0
                and ema_state['update_count'] == ema_state['copy_count'] == ema_state['blend_count'] == 0,
                'GPU FP32/finite/zero-age/online=EMA empty-state identity differs')
        probe = torch.tensor(float(rank), dtype=torch.float32, device=ctx.device)
        torch.distributed.all_reduce(probe, op=torch.distributed.ReduceOp.SUM)
        torch.cuda.synchronize(ctx.device)
        require(probe.dtype == torch.float32 and bool(torch.isfinite(probe).all()) and probe.item() == 28.,
                'known FP32 collective sum differs from 0+1+...+7=28')
        validate_zero_work(work)
        require(reads['forbidden_read_attempts'] == 0 and time.time() < spec['deadline_at'], 'admission deadline/data guard failed')
        thread_masks = [sorted(os.sched_getaffinity(int(task.name))) for task in Path('/proc/self/task').iterdir()]
        require(all(mask and set(mask) <= set(site['cpu_execution']['rank_masks'][rank]) for mask in thread_masks),
                'a native rank thread escaped its assigned CPU mask')
        record = {'rank': rank, 'rank_proof': proof, 'identity': identity(os.getpid()), **work,
            'model_tensor_sha256': digest, 'ema_tensor_sha256': ema_digest,
            'initial_state_control': expected, 'public_initialization': public, 'parameter_groups': groups,
            'precision_runtime': runtime_proof(RULE, device=ctx.device), 'finite_values_verified': True,
            'collective': {'dtype': 'torch.float32', 'input': rank, 'sum': probe.item(), 'expected': 28.},
            'optimizer_state_entries': len(optimizer.state), 'ema_counts': {key: ema_state[key]
                for key in ('update_count', 'copy_count', 'blend_count', 'last_update_success')},
            'scientific_metadata_read': True, 'checkpoint_reused': False, 'thread_cpu_masks': thread_masks}
        records = ctx.values(record)
        api.verify_rank_devices([item['rank_proof'] for item in records], site)
        require({item['rank'] for item in records} == set(range(8)) and all(
            item['initial_state_control'] == expected and item['model_tensor_sha256'] == item['ema_tensor_sha256']
                == expected['model_tensor_sha256'] and item['collective']['sum'] == 28. for item in records),
                'the eight actual initializations/collectives disagree')
        write(Path(spec['output']) / ('rank-' + str(rank) + '-result.json'), record)
        ctx.barrier()
        return 0
    finally:
        ctx.close()


def run_owner(args, spec, context):
    output = Path(spec['output']); children = []; stopping = {'reason': None, 'signals': []}
    def request_stop(signum, _frame):
        stopping['reason'] = stopping['reason'] or 'signal_' + str(signum)
        stopping['signals'].append(signum)
    previous = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    result = None
    try:
        context['api'].bind_cpu_role(context['site'], 'control')
        require(time.time() < spec['deadline_at'], 'owner deadline expired before starting ranks')
        for rank in range(8):
            if stopping['reason'] is not None:
                break
            require(time.time() < spec['deadline_at'], 'owner deadline expired while starting ranks')
            argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--internal-role', 'rank',
                    '--spec', str(args.spec), '--spec-sha256', args.spec_sha256, '--rank', str(rank)]
            children.append(launch_child(argv, 'rank-' + str(rank), output, args.spec_sha256, role='rank', rank=rank))
        result = wait_children(children, spec['deadline_at'], context, stop_reason=lambda: stopping['reason'],
            until=min(spec['deadline_at'] + 25, spec['hard_deadline_at'] - 3))
    except BaseException as error:
        result = wait_children(children, time.time(), context, stop_reason=lambda: stopping['reason'],
            initial_reason='owner_exception:' + repr(error), until=min(time.time() + 25, spec['hard_deadline_at'] - 3))
    try:
        if set(result['exit_codes']) != {'rank-' + str(rank) for rank in range(8)} or stopping['reason'] is not None:
            result['reason'] = result['reason'] or stopping['reason'] or 'incomplete_rank_launch'
            if result['status'] == 'completed':
                result['status'] = 'failed'
        write(output / 'OWNER-RESULT.json', {**result, 'owner': identity(os.getpid()), **zero_work(),
            'stop_signals': stopping['signals'], 'ended_at': time.time(),
            'meaning': 'own ranks wait/reap recorded; outer auditor must still wait for this owner'})
        return 0 if result['status'] == 'completed' else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def inactive_observations(processes):
    return {item['pid']: identity(item['pid']) for item in processes}


def merge_processes(*groups):
    result = {}
    for group in groups:
        for item in group:
            key = (item['pid'], str(item['start_ticks']))
            previous = result.get(key)
            require(previous is None or previous['argv'] == item['argv'], 'observed process argv identity changed')
            result[key] = item
    require(len({item['pid'] for item in result.values()}) == len(result), 'PID recycling within this attempt is ambiguous')
    return list(result.values())


def clean_survivors(processes, until):
    signals = []
    for item in processes:
        if exact(item, identity(item['pid'])):
            signals.append(signal_exact(item, signal.SIGTERM))
    kill_at = min(time.time() + 3, until - 1)
    while time.time() < kill_at and any(exact(item, identity(item['pid'])) for item in processes):
        time.sleep(.05)
    for item in processes:
        if exact(item, identity(item['pid'])):
            signals.append(signal_exact(item, signal.SIGKILL))
    while time.time() < until and any(exact(item, identity(item['pid'])) for item in processes):
        time.sleep(.05)
    return signals


def run_outer(args):
    started_at = time.time()
    require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
            'outer auditor requires a fresh standard-library-only process')
    site_input = pinned(args.site_binding, args.site_binding_sha256)
    root = safe_path(site_input['work_root']); output = safe_path(args.output, root)
    prep = safe_path(args.prep_root, root)
    require(not output.resolve().is_relative_to(prep / 'package')
            and not output.resolve().is_relative_to(Path(site_input['round_root'])), 'attempt must not mutate a frozen package/round')
    require(not output.resolve().is_relative_to(Path(__file__).resolve().parents[2]),
            'attempt must not mutate the source checkout containing this tool')
    reserve_output(output, root)
    inputs = {name: str(getattr(args, name)) if isinstance(getattr(args, name), Path) else getattr(args, name)
              for name in INPUT_NAMES}
    auditor = identity(os.getpid()); lock_fd = None; owner = None; context = None; known_processes = []
    supervision = None; owner_result = {}; interrupted_drain = None
    write(output / 'STARTED.json', {'status': 'reserved_do_not_retry', 'started_at': started_at,
        'inputs': inputs, 'auditor': auditor, 'GPU_started': False, **zero_work()})
    def interrupted(signum, _frame):
        raise InterruptedError('outer auditor received signal ' + str(signum))
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        deployment_input = pinned(args.deployment_receipt, args.deployment_receipt_sha256, root)
        require(not output.resolve().is_relative_to(Path(deployment_input['source_root']).resolve()),
                'attempt must not mutate the frozen source checkout')
        context = load_context(inputs)
        api = context['api']; site = context['site']
        allocation, _ = api.verify_gpu_cpu(site, initial=True)
        allocation_value = pinned(allocation['allocation_receipt']['path'],
                                  allocation['allocation_receipt']['sha256'], root)
        require(allocation_value['valid_until'] >= started_at + 300, 'CPU allocation cannot cover this bounded GPU admission')
        api.bind_cpu_role(site, 'owner', initial=True); api.bind_cpu_role(site, 'control')
        lock = api.instance_lock(site); lock.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        lock_info = os.fstat(lock_fd)
        require(stat.S_ISREG(lock_info.st_mode) and lock_info.st_size == 0, 'actual empty instance lock required')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(context['package']))
        import run_round as guard
        require(not guard.device_users_before_start(), 'GPU device ownership is occupied or unknown')
        resources = guard.runtime_safety_snapshot()
        require(len(resources['cards']) == 8 and all(card['busy_percent'] == 0
                and card['vram_used_bytes'] < 64 * 1024**2 for card in resources['cards']), 'all eight physical GPUs must be idle')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
        api.verify_live(site, pod_uid=args.pod_uid, ready=args.ready == 'True', observed_at=args.observed_at)
        spec = {'schema_version': 1, 'kind': 'eight-rank-zero-forward-initialization',
            'inputs': inputs, 'output': str(output), 'auditor': auditor, 'started_at': started_at,
            'deadline_at': started_at + 240, 'owner_wait_until': started_at + 270, 'hard_deadline_at': started_at + 300,
            'started_sha256': sha(output / 'STARTED.json'), 'entry_sha256': sha(__file__),
            'zero_work': zero_work(), 'init_method': 'tcp://127.0.0.1:' + str(port)}
        require(time.time() < spec['deadline_at'], 'preflight consumed the bounded initialization window')
        write(output / 'SPEC.json', spec)
        spec_path = output / 'SPEC.json'; spec_sha = sha(spec_path)
        argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--internal-role', 'owner',
                '--spec', str(spec_path), '--spec-sha256', spec_sha]
        owner = launch_child(argv, 'owner', output, spec_sha, role='owner')
        # The owner's own rank drain can take 25s. Keep it alive for that entire
        # window before a forced stop, within this attempt's original 300s end.
        supervision = wait_children([owner], spec['owner_wait_until'], context,
            kill_after=27, grace_seconds=29, until=spec['hard_deadline_at'] - 1)
        owner_exit = owner['process'].poll()
        if owner_exit is not None:
            owner_exit = owner['process'].wait(timeout=0)
        rank_launches = [json.loads(path.read_text())['identity']
                        for path in sorted(output.glob('rank-*-launch.json'))]
        owner_result_path = output / 'OWNER-RESULT.json'
        owner_result = json.loads(owner_result_path.read_text()) if owner_result_path.is_file() else {}
        known_processes = merge_processes([owner['identity']], rank_launches,
            supervision['observed_processes'], owner_result.get('observed_processes', []))
        cleanup = clean_survivors(known_processes, min(started_at + 298, time.time() + 4))
        observations = inactive_observations(known_processes)
        write(output / 'EXIT-AUDIT.json', {'owner_exit_code': owner_exit, 'owner_supervision': supervision,
            'rank_exit_codes': owner_result.get('exit_codes', {}), 'processes': known_processes,
            'actual_after_wait': observations, 'cleanup_signals': cleanup, 'auditor': auditor,
            'lock_inode': lock_info.st_ino, 'lock_device': lock_info.st_dev, 'at': time.time()})
        require(supervision['status'] == owner_result.get('status') == 'completed', 'owner/rank supervision was not completed')
        require(owner_result.get('reaped') == {'rank-' + str(rank): True for rank in range(8)}
                and owner_result.get('live_owned_children') == [], 'every native rank must have a recorded wait/reap')
        verify_exits(owner_exit, owner_result['exit_codes'], known_processes, observations, auditor_pid=os.getpid())
        records = [json.loads((output / ('rank-' + str(rank) + '-result.json')).read_text()) for rank in range(8)]
        for rank, record in enumerate(records):
            validate_zero_work(record)
            require(record['rank'] == rank and exact(rank_launches[rank], record['identity'])
                    and record['initial_state_control'] == context['cpu']['initial_state_control']
                    and record['model_tensor_sha256'] == record['ema_tensor_sha256']
                        == context['cpu']['initial_state_control']['model_tensor_sha256']
                    and record['finite_values_verified'] is True and record['collective']
                        == {'dtype': 'torch.float32', 'input': rank, 'sum': 28., 'expected': 28.},
                    'actual rank result does not match its native initialization/collective')
        require(not guard.device_users_before_start(), 'GPU handles remain after owner/rank exit')
        final_resources = guard.runtime_safety_snapshot()
        require(all(card['busy_percent'] == 0 and card['vram_used_bytes'] < 64 * 1024**2
                    for card in final_resources['cards']), 'GPU resources have not returned to idle')
        result = {'schema_version': 1, 'status': 'completed', 'exit_code': 0,
            'git_commit': site['git_commit'], 'pod_uid': site['pod']['uid'],
            'package_manifest_sha256': args.package_manifest_sha256,
            'bundle_source_digest': context['source_digest'], 'candidate': site['science']['candidate'],
            'initial_state_control': context['cpu']['initial_state_control'],
            'rank_proofs': [record['rank_proof'] for record in records], 'released_processes': known_processes,
            'all_processes_exited': True, 'live_processes': [], 'fit_only': True,
            'fresh_initial_state_verified': True, 'fp32_collective_verified': True, 'finite_values_verified': True,
            'wall_seconds': time.time() - started_at, 'scientific_metadata_read': True,
            'checkpoint_reused': False, 'owner_exit_code': owner_exit, 'rank_exit_codes': owner_result['exit_codes'],
            'exit_audit_sha256': sha(output / 'EXIT-AUDIT.json'), 'spec_sha256': spec_sha,
            'script_sha256': sha(__file__), 'deployment_receipt_sha256': args.deployment_receipt_sha256,
            'cpu_prep_receipt_sha256': args.cpu_prep_receipt_sha256,
            'cpu_allocation_sha256': allocation['allocation_receipt']['sha256'],
            'instance_lock': {'path': str(lock), 'device': lock_info.st_dev, 'inode': lock_info.st_ino},
            'before': resources, 'after': final_resources, **zero_work()}
        validate_zero_work(result)
        context['activation'].verify_gpu_admission(result, site, manifest_sha=args.package_manifest_sha256,
            source_digest=context['source_digest'], cpu=context['cpu'])
        require(time.time() - started_at <= 300, 'outer audit exceeded the 300-second attempt budget')
        require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules), 'outer auditor imported torch')
        write(output / 'GPU-ADMISSION.json', result)
        print(json.dumps({'status': result['status'], 'receipt': str(output / 'GPU-ADMISSION.json'),
                          'sha256': sha(output / 'GPU-ADMISSION.json'), 'wall_seconds': result['wall_seconds']}), flush=True)
        return 0
    except BaseException as error:
        cleanup_error = None
        try:
            if owner is not None:
                if owner['process'].poll() is None:
                    interrupted_drain = wait_children([owner], time.time(), context,
                        initial_reason='outer_exception:' + repr(error), kill_after=27, grace_seconds=29,
                        until=started_at + 299)
                launch_identities = [json.loads(path.read_text())['identity'] for path in output.glob('rank-*-launch.json')]
                known_processes = merge_processes(known_processes, [owner['identity']], launch_identities)
                clean_survivors(known_processes, min(started_at + 299, time.time() + 4))
                if owner['process'].poll() is not None:
                    owner['process'].wait(timeout=0)
                owner_result_path = output / 'OWNER-RESULT.json'
                if owner_result_path.is_file():
                    owner_result = json.loads(owner_result_path.read_text())
            observations = inactive_observations(known_processes)
        except BaseException as cleanup_failure:
            cleanup_error = repr(cleanup_failure)
            observations = {'status': 'unknown', 'error': cleanup_error}
        failure = {'status': 'failed_or_resource_unknown_do_not_retry', 'exit_code': 1, 'error': repr(error),
            'wall_seconds': time.time() - started_at, 'processes': known_processes,
            'actual': observations, 'cleanup_error': cleanup_error,
            'owner_exit_code': owner['process'].poll() if owner else None,
            'outer_supervision': supervision, 'interrupted_owner_drain': interrupted_drain,
            'outer_reason': supervision.get('reason') if supervision else repr(error),
            'owner_reason': owner_result.get('reason'), 'rank_exit_codes': owner_result.get('exit_codes', {}),
            'rank_reaped': owner_result.get('reaped', {}),
            'GPU_success_receipt_written': (output / 'GPU-ADMISSION.json').exists(), 'formal_clock_created': False}
        write(output / 'FAILURE.json', failure)
        raise
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN); os.close(lock_fd)
        signal.signal(signal.SIGTERM, previous_term)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('prep-root', 'site-binding', 'package-manifest', 'deployment-receipt', 'cpu-prep-receipt', 'output', 'spec'):
        result.add_argument('--' + name, type=Path)
    for name in ('site-binding-sha256', 'package-manifest-sha256', 'deployment-receipt-sha256',
                 'cpu-prep-receipt-sha256', 'pod-uid', 'spec-sha256'):
        result.add_argument('--' + name)
    result.add_argument('--ready', choices=['True'])
    result.add_argument('--observed-at', type=float)
    result.add_argument('--internal-role', choices=['owner', 'rank'], help=argparse.SUPPRESS)
    result.add_argument('--rank', type=int, help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.internal_role:
        require(all(getattr(args, name) is not None for name in INPUT_NAMES + ('output',)),
                'every explicit path/SHA/live identity argument is required')
        # The first invocation has no marker/clock yet. Re-exec with the exact
        # CPU-verified library before reserving this one GPU attempt.
        inputs = {name: str(getattr(args, name)) if isinstance(getattr(args, name), Path) else getattr(args, name)
                  for name in INPUT_NAMES}
        context = load_context(inputs)
        context['api'].prepare_native_entry(context['site'])
        return run_outer(args)
    require(args.spec is not None and args.spec_sha256, 'internal child needs a pinned parent specification')
    spec = read_spec(args)
    wait_release(spec, args.spec_sha256, args.internal_role, args.rank)
    context = load_context(spec['inputs'])
    try:
        return run_owner(args, spec, context) if args.internal_role == 'owner' else run_rank(args, spec, context)
    except BaseException as error:
        name = 'owner' if args.internal_role == 'owner' else 'rank-' + str(args.rank)
        write(Path(spec['output']) / (name + '-FAILURE.json'), {'status': 'failed', 'error': repr(error),
            'identity': identity(os.getpid()), 'at': time.time()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())

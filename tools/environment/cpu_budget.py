"""Allocate the pinned validation CPU share and audit each executor's inherited task tree."""
import argparse
import datetime
import fcntl
from fractions import Fraction
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_resource_policy(value):
    try:
        if type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise ValueError('resource policy schema differs')
        fractions = value['cpu_fractions']
        if (not isinstance(fractions, dict) or set(fractions) != {'validation', 'training'}
                or any(not isinstance(item, str) for item in fractions.values())):
            raise ValueError('explicit validation/training fraction strings required')
        shares = [Fraction(fractions[key]) for key in ('validation', 'training')]
        idle = value.get('idle_borrow')
        if idle is None:
            if any(share <= 0 for share in shares) or sum(shares) > 1:
                raise ValueError('resource policy fractions must be positive with sum at most one')
        elif (shares != [Fraction(1), Fraction(0)] or not isinstance(idle, dict)
              or set(idle) != {'enabled', 'safe_proof_max_age_seconds', 'device_check_interval_seconds',
                               'idle_vram_bytes_exclusive'}
              or idle.get('enabled') is not True
              or any(type(idle.get(key)) is not int or idle[key] <= 0 for key in
                     ('safe_proof_max_age_seconds', 'device_check_interval_seconds', 'idle_vram_bytes_exclusive'))):
            raise ValueError('idle CPU borrowing requires explicit complete 1/0 policy')
        native = value['native_affinity_environment']
        if (not isinstance(native, dict)
                or set(native) != {'AMD_CPU_AFFINITY', 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG',
                                   'NCCL_IGNORE_CPU_AFFINITY'}
                or any(item not in ('0', '1') for item in native.values())
                or any(type(value[key]) is not int or value[key] <= 0
                       for key in ('rank_threads', 'interop_threads'))
                or type(value['loader_workers']) is not int or value['loader_workers'] < 0):
            raise ValueError('explicit native affinity and thread/worker policy required')
    except (KeyError, TypeError, AttributeError, ZeroDivisionError) as error:
        raise ValueError('incomplete or invalid resource policy') from error
    return value


def load_resource_policy(reference=None):
    """Read env only for planning; task launches supply their allocation's exact pin."""
    if reference is None:
        path = os.environ.get('NCCCU_RESOURCE_POLICY')
        expected = os.environ.get('NCCCU_RESOURCE_POLICY_SHA256')
    else:
        if not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'value'}:
            raise ValueError('allocation requires the complete pinned resource policy')
        path, expected = reference['path'], reference['sha256']
    if (not isinstance(path, str) or not Path(path).is_absolute() or '..' in Path(path).parts
            or Path(path).is_symlink() or not Path(path).is_file()
            or not isinstance(expected, str) or len(expected) != 64
            or any(character not in '0123456789abcdef' for character in expected)):
        raise ValueError('explicit absolute resource policy file and SHA required')
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('resource policy SHA differs')
    value = validate_resource_policy(json.loads(content))
    if reference is not None and value != reference['value']:
        raise ValueError('resource policy content differs from the allocation pin')
    return {'path': str(path), 'sha256': expected, 'value': value}


def parse_cpuset(text):
    result = set()
    for part in text.strip().split(','):
        if not part:
            continue
        ends = part.split('-')
        if len(ends) == 1:
            result.add(int(ends[0]))
        elif len(ends) == 2 and 0 <= int(ends[0]) <= int(ends[1]):
            result.update(range(int(ends[0]), int(ends[1]) + 1))
        else:
            raise ValueError('invalid effective cpuset')
    if not result or min(result) < 0:
        raise ValueError('empty or invalid effective cpuset')
    return sorted(result)


def measure(resource_policy=None):
    policy = load_resource_policy(resource_policy)
    fraction = Fraction(policy['value']['cpu_fractions']['validation'])
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = parse_cpuset(Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text())
    available = sorted(set(affinity) & set(cpuset))
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if int(period) <= 0:
        raise ValueError('invalid cgroup CPU period')
    capacity = Fraction(len(available)) if quota == 'max' else min(Fraction(len(available)), Fraction(int(quota), int(period)))
    if capacity <= 0:
        raise ValueError('effective CPU capacity is not positive')
    total = int(capacity * fraction)
    return {'affinity': affinity, 'cpuset': cpuset, 'available': available,
            'cpu_max': [quota, period], 'effective_cpu_fraction': str(capacity),
            'effective_cpus': float(capacity), 'max_total_threads': total,
            'validation_mask': available[:total], 'resource_policy': policy}


def validation_pool(resources):
    try:
        policy = resources['resource_policy']
        if not isinstance(policy, dict) or set(policy) != {'path', 'sha256', 'value'}:
            raise ValueError('resources require an explicit pinned policy')
        validate_resource_policy(policy['value'])
        available = resources['available']
        capacity = Fraction(resources['effective_cpu_fraction'])
        total = int(capacity * Fraction(policy['value']['cpu_fractions']['validation']))
        if (not isinstance(available, list) or not available
                or any(type(cpu) is not int or cpu < 0 for cpu in available)
                or available != sorted(set(available)) or not 0 < capacity <= len(available)
                or type(resources['max_total_threads']) is not int
                or resources['max_total_threads'] != total
                or resources['validation_mask'] != available[:total]):
            raise ValueError('validation pool differs from the measured policy/capacity')
    except (KeyError, TypeError, AttributeError, ZeroDivisionError) as error:
        raise ValueError('incomplete measured validation CPU pool') from error
    return total, resources['validation_mask']


def identity():
    result = {key: os.environ[key] for key in
              ('NCCCU_POD_NAMESPACE', 'NCCCU_POD_NAME', 'NCCCU_POD_UID', 'NCCCU_CONTAINER', 'NCCCU_PYTHON')}
    if (socket.gethostname() != result['NCCCU_POD_NAME']
            or os.path.realpath(sys.executable) != os.path.realpath(result['NCCCU_PYTHON'])
            or os.getpriority(os.PRIO_PROCESS, 0) != 19):
        raise ValueError('fresh Pod/interpreter/nice19 identity differs')
    return result


def validate_idle_evidence(site, proof, pod_identity, live, *, now, max_age, max_vram):
    """Borrow only from a recently completed owner and presently idle devices."""
    try:
        expected = {row['pci'] for row in site['gpus']}
        identity_proof, safety = proof['identity'], proof['safety']
        processes = [identity_proof['owner_expected'], *identity_proof['ranks_expected']]
        ranks = identity_proof['ranks']
        observed = proof['observed_at']
        if (site['pod']['uid'] != pod_identity['NCCCU_POD_UID'] or len(expected) != 8
                or proof['formal_root'] != site['round_root'] or proof['phase'] != 'terminal'
                or type(observed) not in (int, float) or not math.isfinite(observed)
                or not 0 <= now - observed <= max_age or proof['signals_sent'] is not False
                or proof['model_forward_executed'] is not False or proof['official_read'] is not False
                or proof['first_success_receipt_read'] is not True
                or proof['done'] != {'present': True, 'status': 'completed'}
                or identity_proof['owner_status'] != 'exited'
                or identity_proof['owner'].get('active') is not False
                or identity_proof['owner'].get('readable') is not True
                or len(processes) != 9 or len({item['pid'] for item in processes}) != 9
                or any(type(item['pid']) is not int or item['pid'] <= 0
                       or not str(item['start_ticks']).isdecimal() for item in processes)
                or len(ranks) != 8 or any(row.get('active') is not False or row.get('readable') is not True
                                           for row in ranks)
                or safety['safe'] is not True or safety['device_users'] != []
                or safety['device_scan_denied'] != 0 or safety['all_expected_cards_seen'] is not True
                or safety['lock']['readable'] is not True or safety['lock']['holders'] != []):
            raise ValueError('idle borrow terminal proof is missing, stale or unsafe')
        for rows in (safety['cards'], live['cards']):
            if (not isinstance(rows, list) or len(rows) != 8
                    or {row['pci'] for row in rows} != expected
                    or any(row['busy_percent'] != 0 for row in rows)):
                raise ValueError('idle borrow card set or utilization differs')
        if (live['device_users'] != [] or live['device_scan_denied'] != 0
                or live['active_terminal_processes'] != []
                or live['lock']['readable'] is not True or live['lock']['holders'] != []
                or any(type(row['vram_used_bytes']) is not int or not 0 <= row['vram_used_bytes'] < max_vram
                       for row in live['cards'])):
            raise ValueError('a GPU user, unreadable device or live terminal process blocks borrowing')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('incomplete idle borrowing evidence') from error
    return live


def snapshot_idle_devices(site, proof):
    """Read process handles and sysfs without initializing CUDA/HIP."""
    users, denied = [], 0
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] in ('Z', 'X'):
                continue
            for descriptor in (path / 'fd').iterdir():
                try:
                    target = os.readlink(descriptor)
                    if target == '/dev/kfd' or target.startswith('/dev/dri/renderD'):
                        users.append({'pid': int(path.name), 'start_ticks': fields[19], 'device': target})
                        break
                except (FileNotFoundError, ProcessLookupError):
                    pass
                except PermissionError:
                    denied += 1
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            denied += 1
    expected = {row['pci'] for row in site['gpus']}; cards = []
    for path in Path('/sys/class/drm').glob('card[0-9]*'):
        if not path.name[4:].isdigit():
            continue
        device = (path / 'device').resolve()
        if device.name in expected:
            cards.append({'pci': device.name, 'busy_percent': int((device / 'gpu_busy_percent').read_text()),
                          'vram_used_bytes': int((device / 'mem_info_vram_used').read_text())})
    active = []
    for item in [proof['identity']['owner_expected'], *proof['identity']['ranks_expected']]:
        path = Path('/proc') / str(item['pid']) / 'stat'
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if fields[0] not in ('Z', 'X') and fields[19] == str(item['start_ticks']):
                active.append(item)
        except (FileNotFoundError, ProcessLookupError):
            pass
    lock_path = Path(site['work_root']) / 'control' / ('.instance-owner-' + site['pod']['uid'] + '.lock')
    if lock_path.is_file():
        inode = lock_path.stat().st_ino
        holders = [line for line in Path('/proc/locks').read_text().splitlines()
                   if len(line.split()) > 5 and line.split()[5].endswith(':' + str(inode))]
    else:
        holders = []
    return {'at': time.time(), 'device_users': users, 'device_scan_denied': denied,
            'active_terminal_processes': active, 'cards': cards,
            'lock': {'readable': True, 'holders': holders}}


def read_idle_reference(policy, pod_identity, *, now, reference=None):
    """Bind an idle-only allocation to explicit site and terminal proof files."""
    settings = policy['value'].get('idle_borrow')
    if settings is None:
        return None
    if reference is None:
        reference = {key: {'path': os.environ.get('NCCCU_CPU_IDLE_' + key.upper(), ''),
                           'sha256': os.environ.get('NCCCU_CPU_IDLE_' + key.upper() + '_SHA256', '')}
                     for key in ('site', 'safety')}
    if not isinstance(reference, dict) or set(reference) != {'site', 'safety'}:
        raise ValueError('explicit idle site and safety references required')
    values = {}
    for key, ref in reference.items():
        path = Path(ref.get('path', '')) if isinstance(ref, dict) else Path('')
        digest = ref.get('sha256') if isinstance(ref, dict) else None
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or not isinstance(digest, str) or len(digest) != 64 or sha(path) != digest):
            raise ValueError('idle site or safety SHA differs')
        values[key] = json.loads(path.read_text())
    live = snapshot_idle_devices(values['site'], values['safety'])
    validate_idle_evidence(values['site'], values['safety'], pod_identity, live, now=now,
                           max_age=settings['safe_proof_max_age_seconds'],
                           max_vram=settings['idle_vram_bytes_exclusive'])
    return {'references': reference, 'live': live}


def build_allocation(resources, owners, pod_identity, *, now, valid_seconds):
    if not 1 <= len(owners) <= 2 or len(set(owners)) != len(owners) or any(not owner.strip() for owner in owners):
        raise ValueError('one or two explicit, distinct existing executor IDs required')
    if type(valid_seconds) is not int or not 1 <= valid_seconds <= 21600:
        raise ValueError('explicit allocation validity must be within six hours')
    total, pool = validation_pool(resources)
    if total < len(owners):
        raise ValueError('configured validation CPU cap cannot give each executor one CPU')
    allocations, start = {}, 0
    for index, owner in enumerate(owners):
        width = total // len(owners) + int(index < total % len(owners))
        mask = pool[start:start + width]
        allocations[owner] = {'threads': width, 'mask': mask}
        start += width
    return {'schema_version': 1, 'identity': pod_identity, 'resources': resources,
            'resource_policy': resources['resource_policy'],
            'total_fraction_cap': str(Fraction(resources['resource_policy']['value']['cpu_fractions']['validation'])),
            'max_total_threads': total, 'allocations': allocations,
            'created_at': now, 'valid_until': now + valid_seconds, 'signals_allowed': False,
            'tree_enforcement': 'disjoint inherited CPU affinity; one active task lease per executor'}


def validate_allocation(value, resources, pod_identity, owner, *, now):
    total, pool = validation_pool(resources)
    expected_cap = str(Fraction(resources['resource_policy']['value']['cpu_fractions']['validation']))
    if (value.get('schema_version') != 1 or value.get('identity') != pod_identity
            or value.get('resource_policy') != resources['resource_policy']
            or value.get('total_fraction_cap') != expected_cap
            or not value.get('created_at', float('inf')) <= now <= value.get('valid_until', -1)):
        raise ValueError('CPU allocation identity is missing, expired or belongs to another Pod')
    allocations = value.get('allocations', {})
    if not isinstance(allocations, dict) or not 1 <= len(allocations) <= 2 or owner not in allocations:
        raise ValueError('explicit executor allocation is missing')
    masks = []
    for item in allocations.values():
        mask = item.get('mask')
        if (not isinstance(mask, list) or not mask or any(type(cpu) is not int or cpu < 0 for cpu in mask)
                or len(set(mask)) != len(mask) or item.get('threads') != len(mask)):
            raise ValueError('CPU allocation mask or thread count differs')
        masks.extend(mask)
    if (len(set(masks)) != len(masks) or not set(masks) <= set(pool)
            or len(masks) > total
            or value.get('max_total_threads') != len(masks)):
        raise ValueError('overlapping CPUs or allocation escapes the live validation pool')
    return allocations[owner]


def snapshot_tree(root_pid, known, mask):
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            processes[int(path.name)] = (int(fields[1]), fields[19], fields[0])
        except (FileNotFoundError, ProcessLookupError):
            continue
    selected = {root_pid} | {pid for pid, ticks in known.items()
                                if pid in processes and processes[pid][1] == ticks}
    while True:
        expanded = selected | {pid for pid, (parent, _, _) in processes.items() if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    threads, runnable, violations, live = 0, 0, [], []
    for pid in selected:
        if pid not in processes or processes[pid][2] in ('Z', 'X'):
            continue
        known[pid] = processes[pid][1]; live.append(pid)
        try:
            for task in (Path('/proc') / str(pid) / 'task').iterdir():
                try:
                    allowed = set(os.sched_getaffinity(int(task.name)))
                    fields = (task / 'stat').read_text().rsplit(')', 1)[1].split()
                    threads += 1; runnable += int(fields[0] == 'R')
                    if not allowed or not allowed <= set(mask):
                        violations.append({'pid': pid, 'tid': int(task.name), 'affinity': sorted(allowed)})
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {'live_pids': sorted(live), 'threads': threads, 'runnable_threads': runnable,
            'affinity_violations': violations}


def task_lease_path(pod_identity, owner):
    # Leases serialize processes in this Pod. NFS flock RPCs can stall even
    # with LOCK_NB; keep the lease local and stable across allocation renewals.
    key = hashlib.sha256((pod_identity['NCCCU_POD_UID'] + '\0' + owner).encode()).hexdigest()
    return Path('/tmp') / ('.ncccu-cpu-task-' + key + '.lock')


def run_task(args):
    allocation_path = args.allocation.resolve(strict=True)
    if sha(allocation_path) != args.allocation_sha:
        raise ValueError('explicit CPU allocation SHA differs')
    value = json.loads(allocation_path.read_text())
    if value.get('guard_sha256') != sha(__file__):
        raise ValueError('CPU guard differs from the explicitly allocated version')
    if not isinstance(value.get('resource_policy'), dict):
        raise ValueError('CPU allocation is missing the explicit resource policy pin')
    policy = load_resource_policy(value['resource_policy'])
    observed = measure(policy); pod_identity = identity()
    owner = args.owner
    granted = validate_allocation(value, observed, pod_identity, owner, now=time.time())
    idle = policy['value'].get('idle_borrow')
    idle_at_start = None
    if idle is not None:
        if not isinstance(value.get('idle_references'), dict):
            raise ValueError('idle allocation is missing its pinned safety references')
        idle_at_start = read_idle_reference(policy, pod_identity, now=value['created_at'],
                                            reference=value['idle_references'])['live']
    mask = granted['mask']; threads = granted['threads']
    os.sched_setaffinity(0, set(mask))
    os.environ.update({key: str(threads) for key in THREAD_ENV})
    os.environ.update(policy['value']['native_affinity_environment'])
    os.environ.update(NCCCU_CPU_THREADS=str(threads), NCCCU_CPU_MASK=json.dumps(mask),
                      NCCCU_RESOURCE_POLICY=policy['path'], NCCCU_RESOURCE_POLICY_SHA256=policy['sha256'],
                      NCCCU_CPU_INTEROP_THREADS=str(policy['value']['interop_threads']),
                      NCCCU_CPU_ALLOCATION=str(allocation_path), NCCCU_EXECUTOR_ID=owner,
                      NCCCU_CPU_ALLOCATION_SHA256=sha(allocation_path), NCCCU_CPU_WORKERS='0',
                      OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        if os.environ.get(key) != '':
            raise ValueError('CPU task must inherit hidden accelerators')
    lease_path = task_lease_path(pod_identity, owner)
    descriptor = os.open(lease_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = allocation_path.parent / 'tasks'; tasks.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=owner + '-', dir=tasks))
    if args.operation == 'probe':
        code = ('import importlib.metadata as m,json,os,sys,torch;'
                'n=int(os.environ["NCCCU_CPU_THREADS"]);torch.set_num_threads(n);'
                'torch.set_num_interop_threads(int(os.environ["NCCCU_CPU_INTEROP_THREADS"]));'
                'assert not torch.cuda.is_initialized();'
                'print(json.dumps({"python":sys.executable,"python_version":sys.version.split()[0],'
                '"dependencies":{k:m.version(k) for k in ("torch","torchvision","numpy","Pillow")},'
                '"affinity":sorted(os.sched_getaffinity(0)),"threads":torch.get_num_threads(),'
                '"interop_threads":torch.get_num_interop_threads(),"gpu_initialized":False},indent=2))')
        command = [sys.executable, '-B', '-c', code]
    else:
        if not args.script.is_absolute() or args.script.suffix != '.py':
            raise ValueError('explicit absolute Python script required')
        command = [sys.executable, '-B', str(args.script), *args.script_args]
    known = {}; peak_threads = peak_runnable = 0; violations = []
    idle_failure = None; sent_signal = False; next_idle_check = time.monotonic()
    start = time.time()
    with (output / 'OUTPUT.txt').open('x') as stream:
        options = {'cwd': output, 'env': os.environ.copy(), 'stdout': stream,
                   'stderr': subprocess.STDOUT, 'pass_fds': (descriptor,)}
        if idle is not None:
            options['start_new_session'] = True
        process = subprocess.Popen(command, **options)
        while True:
            state = snapshot_tree(process.pid, known, mask)
            peak_threads = max(peak_threads, state['threads'])
            peak_runnable = max(peak_runnable, state['runnable_threads'])
            violations.extend(state['affinity_violations'])
            code = process.poll()
            if code is not None and not state['live_pids']:
                break
            if idle is not None and idle_failure is None and time.monotonic() >= next_idle_check:
                try:
                    read_idle_reference(policy, pod_identity, now=value['created_at'],
                                        reference=value['idle_references'])
                except Exception as error:
                    idle_failure = {'at': time.time(), 'type': type(error).__name__, 'message': str(error)}
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        sent_signal = True
                    except ProcessLookupError:
                        pass
                next_idle_check = time.monotonic() + idle['device_check_interval_seconds']
            time.sleep(.2)
    receipt = {'scope': 'compute_pod_cpu_task_tree', 'identity': pod_identity, 'executor_id': owner,
               'allocation_path': str(allocation_path), 'allocation_sha256': sha(allocation_path),
               'lease_path': str(lease_path), 'lease_scope': 'compute_pod_executor_across_preparation_bindings',
               'guard_sha256': sha(__file__), 'resource_policy': policy,
               'mask': mask, 'threads': threads, 'nice': 19, 'command': command,
               'max_total_threads': value['max_total_threads'], 'actual_resources_before_affinity': observed,
               'peak_task_tree_threads': peak_threads, 'peak_task_tree_runnable_threads': peak_runnable,
               'affinity_violations': violations, 'task_tree_exited': True, 'signals_sent': sent_signal,
               'signals_scope': 'own_cpu_task_process_group' if sent_signal else None,
               'idle_borrow': idle is not None, 'idle_live_at_start': idle_at_start,
               'idle_interruption': idle_failure,
               'child_exit_code': code, 'wall_seconds': time.time() - start,
               'physical_cpu_envelope_enforced': not violations,
               'runnable_thread_budget_respected': peak_runnable <= threads,
               'exit_code': code if not violations and peak_runnable <= threads and idle_failure is None else 2,
               'output_sha256': sha(output / 'OUTPUT.txt')}
    with (output / 'CPU-RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    os.close(descriptor)
    print((output / 'OUTPUT.txt').read_text(), end='')
    print(json.dumps({'cpu_receipt': str(output / 'CPU-RECEIPT.json'),
                      'cpu_receipt_sha256': sha(output / 'CPU-RECEIPT.json'), 'exit_code': receipt['exit_code']}, indent=2))
    return receipt['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--output', type=Path, required=True)
    plan.add_argument('--owner', action='append', required=True)
    plan.add_argument('--valid-seconds', type=int, required=True)
    for name in ('probe', 'run'):
        task = sub.add_parser(name)
        task.add_argument('--allocation', type=Path, required=True)
        task.add_argument('--allocation-sha', required=True)
        task.add_argument('--owner', required=True)
        if name == 'run':
            task.add_argument('script', type=Path)
            task.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation != 'plan':
        return run_task(args)
    if not args.output.is_absolute() or 'technical-prep' not in args.output.parts:
        raise ValueError('allocation must be an explicit technical-prep output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output.parent)
    pod_identity = identity(); resources = measure()
    if any(os.environ.get(key) != '' for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')):
        raise ValueError('CPU planning requires hidden accelerators')
    value = build_allocation(resources, args.owner, pod_identity, now=time.time(), valid_seconds=args.valid_seconds)
    idle = read_idle_reference(resources['resource_policy'], pod_identity, now=value['created_at'])
    if idle is not None:
        value['idle_references'] = idle['references']
        value['idle_live_at_plan'] = idle['live']
    value['guard_sha256'] = sha(__file__)
    value['python_version'] = sys.version.split()[0]
    value['dependencies'] = {name: metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps({'allocation': str(args.output), 'sha256': sha(args.output), 'budget': value}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

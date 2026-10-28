"""Allocate at most 20% of live Pod CPUs and audit each executor's inherited task tree."""
import argparse
import datetime
import fcntl
from fractions import Fraction
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def measure():
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = parse_cpuset(Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text())
    available = sorted(set(affinity) & set(cpuset))
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if int(period) <= 0:
        raise ValueError('invalid cgroup CPU period')
    capacity = Fraction(len(available)) if quota == 'max' else min(Fraction(len(available)), Fraction(int(quota), int(period)))
    if capacity <= 0:
        raise ValueError('effective CPU capacity is not positive')
    return {'affinity': affinity, 'cpuset': cpuset, 'available': available,
            'cpu_max': [quota, period], 'effective_cpu_fraction': str(capacity),
            'effective_cpus': float(capacity), 'max_total_threads': int(capacity // 5)}


def identity():
    result = {key: os.environ[key] for key in
              ('NCCCU_POD_NAMESPACE', 'NCCCU_POD_NAME', 'NCCCU_POD_UID', 'NCCCU_CONTAINER', 'NCCCU_PYTHON')}
    if (socket.gethostname() != result['NCCCU_POD_NAME']
            or os.path.realpath(sys.executable) != os.path.realpath(result['NCCCU_PYTHON'])
            or os.getpriority(os.PRIO_PROCESS, 0) != 19):
        raise ValueError('fresh Pod/interpreter/nice19 identity differs')
    return result


def build_allocation(resources, owners, pod_identity, *, now, valid_seconds):
    if len(owners) != 2 or len(set(owners)) != 2 or any(not owner.strip() for owner in owners):
        raise ValueError('two explicit, distinct existing executor IDs required')
    if type(valid_seconds) is not int or not 1 <= valid_seconds <= 21600:
        raise ValueError('explicit allocation validity must be within six hours')
    total = resources['max_total_threads']
    if total < len(owners):
        raise ValueError('20% CPU cap cannot give each executor one CPU')
    allocations, start = {}, 0
    for index, owner in enumerate(owners):
        width = total // len(owners) + int(index < total % len(owners))
        mask = resources['available'][start:start + width]
        allocations[owner] = {'threads': width, 'mask': mask}
        start += width
    return {'schema_version': 1, 'identity': pod_identity, 'resources': resources,
            'total_fraction_cap': '1/5', 'max_total_threads': total, 'allocations': allocations,
            'created_at': now, 'valid_until': now + valid_seconds, 'signals_allowed': False,
            'tree_enforcement': 'disjoint inherited CPU affinity; one active task lease per executor'}


def validate_allocation(value, resources, pod_identity, owner, *, now):
    if (value.get('schema_version') != 1 or value.get('identity') != pod_identity
            or value.get('total_fraction_cap') != '1/5'
            or not value.get('created_at', float('inf')) <= now <= value.get('valid_until', -1)):
        raise ValueError('CPU allocation identity is missing, expired or belongs to another Pod')
    allocations = value.get('allocations', {})
    if len(allocations) != 2 or owner not in allocations:
        raise ValueError('explicit executor allocation is missing')
    masks = []
    for item in allocations.values():
        mask = item.get('mask')
        if (not isinstance(mask, list) or not mask or any(type(cpu) is not int or cpu < 0 for cpu in mask)
                or len(set(mask)) != len(mask) or item.get('threads') != len(mask)):
            raise ValueError('CPU allocation mask or thread count differs')
        masks.extend(mask)
    if (len(set(masks)) != len(masks) or not set(masks) <= set(resources['available'])
            or len(masks) > resources['max_total_threads']
            or value.get('max_total_threads') != len(masks)):
        raise ValueError('overlapping CPUs or total allocation exceeds the live 20% cap')
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


def run_task(args):
    allocation_path = args.allocation.resolve(strict=True)
    if sha(allocation_path) != args.allocation_sha:
        raise ValueError('explicit CPU allocation SHA differs')
    value = json.loads(allocation_path.read_text())
    if value.get('guard_sha256') != sha(__file__):
        raise ValueError('CPU guard differs from the explicitly allocated version')
    observed = measure(); pod_identity = identity()
    owner = args.owner
    granted = validate_allocation(value, observed, pod_identity, owner, now=time.time())
    mask = granted['mask']; threads = granted['threads']
    os.sched_setaffinity(0, set(mask))
    os.environ.update({key: str(threads) for key in THREAD_ENV})
    os.environ.update(NCCCU_CPU_THREADS=str(threads), NCCCU_CPU_MASK=json.dumps(mask),
                      NCCCU_CPU_ALLOCATION=str(allocation_path), NCCCU_EXECUTOR_ID=owner,
                      NCCCU_CPU_ALLOCATION_SHA256=sha(allocation_path), NCCCU_CPU_WORKERS='0',
                      OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        if os.environ.get(key) != '':
            raise ValueError('CPU task must inherit hidden accelerators')
    lease_path = allocation_path.parent / ('.cpu-task-' + owner + '.lock')
    descriptor = os.open(lease_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = allocation_path.parent / 'tasks'; tasks.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=owner + '-', dir=tasks))
    if args.operation == 'probe':
        code = ('import importlib.metadata as m,json,os,sys,torch;'
                'n=int(os.environ["NCCCU_CPU_THREADS"]);torch.set_num_threads(n);torch.set_num_interop_threads(1);'
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
    start = time.time()
    with (output / 'OUTPUT.txt').open('x') as stream:
        process = subprocess.Popen(command, cwd=output, env=os.environ.copy(),
                                   stdout=stream, stderr=subprocess.STDOUT, pass_fds=(descriptor,))
        while True:
            state = snapshot_tree(process.pid, known, mask)
            peak_threads = max(peak_threads, state['threads'])
            peak_runnable = max(peak_runnable, state['runnable_threads'])
            violations.extend(state['affinity_violations'])
            code = process.poll()
            if code is not None and not state['live_pids']:
                break
            time.sleep(.2)
    receipt = {'scope': 'compute_pod_cpu_task_tree', 'identity': pod_identity, 'executor_id': owner,
               'allocation_path': str(allocation_path), 'allocation_sha256': sha(allocation_path),
               'guard_sha256': sha(__file__),
               'mask': mask, 'threads': threads, 'nice': 19, 'command': command,
               'max_total_threads': value['max_total_threads'], 'actual_resources_before_affinity': observed,
               'peak_task_tree_threads': peak_threads, 'peak_task_tree_runnable_threads': peak_runnable,
               'affinity_violations': violations, 'task_tree_exited': True, 'signals_sent': False,
               'child_exit_code': code, 'wall_seconds': time.time() - start,
               'physical_cpu_envelope_enforced': not violations,
               'runnable_thread_budget_respected': peak_runnable <= threads,
               'exit_code': code if not violations and peak_runnable <= threads else 2,
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

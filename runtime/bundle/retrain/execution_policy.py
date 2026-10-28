"""Configured single-model topology, independent from the immutable R9 policy."""
import copy
from fractions import Fraction
import os
from pathlib import Path

def resolve_topology(cfg, *, available_accelerators):
    from .distributed import layout
    devices = cfg['execution']['devices']
    world = len(devices)
    legacy = cfg['execution'].get('legacy_six_card_exception', False)
    if world not in (1, 2, 4, 8) and not (world == 6 and legacy is True):
        raise ValueError('future worlds are1/2/4/8; six requires the explicit legacy/current exception')
    cpu_only = all(device == 'cpu' for device in devices)
    if not cpu_only:
        if any(not device.startswith('cuda:') for device in devices) or len(set(devices)) != world:
            raise ValueError('one unique accelerator per configured rank required')
        ordinals = [int(device.split(':')[1]) for device in devices]
        if any(index < 0 or index >= available_accelerators for index in ordinals):
            raise ValueError('configured accelerator is unavailable; no output may be created')
    reference = cfg['jobs'][0]
    micro = cfg['execution'].get('per_rank_microbatch', 0)
    plan = layout(reference['physical_batch'], reference['effective_batch'], reference['positive_per_batch'], world,
                  per_rank_microbatch=micro or None)
    job = copy.deepcopy(reference)
    job['physical_batch'] = plan['per_rank_batch']
    evaluation = cfg['execution'].get('evaluation_physical_batch', 32)
    if type(evaluation) is not int or evaluation <= 0:
        raise ValueError('evaluation batch must be explicit and positive')
    policy = {'version': 'single-model-configured-world-v1', 'world_size': world,
              'legacy_six_card_exception': world == 6 and legacy is True,
              'per_rank_microbatch': micro, 'evaluation_physical_batch': evaluation,
              'technical_smoke': cfg['execution'].get('technical_smoke',False),
              'reference_physical_batch': plan['reference_physical_batch'], 'global_effective_batch': plan['global_effective_batch'],
              'global_positive_quota': plan['global_positive_quota'], 'normalization_policy': plan['normalization_policy']}
    return job, plan, policy


def allocate_cpus(world, available=None, quota_cores=None):
    from .site_binding import load_site, verify_gpu_cpu
    site = load_site()
    policy, _ = verify_gpu_cpu(site)
    if type(world) is not int or world != len(site['gpus']):
        raise ValueError('CPU allocation world differs from the explicit site')
    if available is not None and not set(policy['owner_mask']) <= set(available):
        raise ValueError('explicit CPU allocation exceeds the supplied available CPU set')
    if quota_cores is not None and len(policy['owner_mask']) > int(Fraction(str(quota_cores)) * Fraction(policy['total_fraction_cap'])):
        raise ValueError('explicit CPU allocation exceeds the configured training quota')
    return len(policy['rank_masks'][0]), copy.deepcopy(policy['rank_masks'])


def validate_worker_spec(spec, rank, device, requested_mask):
    plan = spec['identity']['distributed']
    world = plan['world_size']
    if (world not in (1, 2, 4, 8) and not (world == 6 and spec['execution_policy'].get('legacy_six_card_exception'))
            or spec['execution_policy']['world_size'] != world or len(spec['devices']) != world
            or len(spec['cpu_masks']) != world or type(rank) is not int or not 0 <= rank < world
            or device != spec['devices'][rank] or list(requested_mask) != spec['cpu_masks'][rank]
            or len(requested_mask) != spec['cpu_width'] or spec['cpu_width'] < 1
            or spec['job']['physical_batch'] != plan['per_rank_batch']
            or len({cpu for mask in spec['cpu_masks'] for cpu in mask}) != world * spec['cpu_width']):
        raise ValueError('worker rank/device/world/job/CPU identity differs from configured topology')
    return world


def bind_worker_cpu(spec, rank, device):
    from .site_binding import load_site, sha256, bind_cpu_role
    site = load_site()
    if (spec.get('site_binding_sha256') != sha256(os.environ['NCCCU_SITE_BINDING'])
            or type(rank) is not int or not 0 <= rank < len(site['gpus'])
            or device != 'cuda:' + str(rank)
            or spec['devices'] != ['cuda:' + str(n) for n in range(len(site['gpus']))]
            or spec['cpu_masks'] != site['cpu_execution']['rank_masks']
            or spec['cpu_width'] != len(site['cpu_execution']['rank_masks'][0])):
        raise ValueError('worker site/CPU/device identity differs from its explicit allocation')
    return bind_cpu_role(site, 'rank', rank=rank)


def audit_cpu_tree(root_pid, known, site):
    """Observe exact descendants, including reparented children seen in prior samples."""
    from .site_binding import verify_gpu_cpu
    policy, _ = verify_gpu_cpu(site)
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
    live, violations = [], []
    threads = 0
    for pid in sorted(selected):
        if pid not in processes or processes[pid][2] in ('Z', 'X'):
            continue
        known[pid] = processes[pid][1]; live.append(pid)
        try:
            for task in (Path('/proc') / str(pid) / 'task').iterdir():
                try:
                    mask = sorted(os.sched_getaffinity(int(task.name))); threads += 1
                    if not mask or not set(mask) <= set(policy['owner_mask']):
                        violations.append({'pid': pid, 'tid': int(task.name), 'mask': mask})
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
    if violations:
        raise ValueError('GPU owner descendant escaped its allocated CPU envelope: ' + repr(violations))
    return {'live_pids': live, 'threads': threads, 'owner_mask': policy['owner_mask'],
            'affinity_violations': violations, 'allocation_sha256': policy['allocation_receipt']['sha256']}


def validate_general_fresh(spec, identity, job, topology):
    from .distributed import layout
    policy = spec['execution_policy']
    world = policy['world_size']
    if world not in (1, 2, 4, 8) and not (world == 6 and policy.get('legacy_six_card_exception') is True):
        raise ValueError('general fresh topology is outside the configured user policy')
    expected = layout(policy['reference_physical_batch'], job['effective_batch'], job['positive_per_batch'], world,
                      per_rank_microbatch=policy.get('per_rank_microbatch') or None)
    expected['stop_after_updates'] = spec['schedule']['stop_after_updates']
    if (topology != expected or policy['global_effective_batch'] != job['effective_batch']
            or policy['global_positive_quota'] != job['positive_per_batch']
            or policy['normalization_policy'] != topology['normalization_policy']
            or type(policy['evaluation_physical_batch']) is not int or policy['evaluation_physical_batch'] <= 0
            or job['physical_batch'] != topology['per_rank_batch']):
        raise ValueError('general fresh global/microbatch/evaluation/topology identity differs')


def rank_proof(ctx, spec):
    import ctypes
    import ctypes.util
    import time
    import torch
    requested = spec['cpu_masks'][ctx.rank]
    actual = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    if actual is not None and actual != sorted(requested):
        raise ValueError('actual worker CPU affinity differs from its configured disjoint mask')
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError('rank torch threads differ from the explicit one-thread contract')
    native_affinity = {key: os.environ.get(key) for key in
        ('AMD_CPU_AFFINITY', 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG', 'NCCL_IGNORE_CPU_AFFINITY')}
    if ctx.device.type == 'cuda':
        from .site_binding import load_site, native_guard_loaded
        cpu_policy = load_site()['cpu_execution']
        expected_native = cpu_policy['resource_policy']['value']['native_affinity_environment']
        if native_affinity != expected_native:
            raise ValueError('GPU native helper affinity inheritance policy differs from configuration')
        if not native_guard_loaded(cpu_policy):
            raise ValueError('actual GPU rank has not loaded the pinned NUMA compatibility library')
    physical = None
    total = None
    if ctx.device.type == 'cuda':
        if torch.cuda.current_device() != ctx.device.index:
            raise ValueError('actual current GPU differs from configured ordinal')
        properties = torch.cuda.get_device_properties(ctx.device)
        total = properties.total_memory
        if torch.version.hip:
            library = ctypes.CDLL(ctypes.util.find_library('amdhip64') or 'libamdhip64.so')
            function = library.hipDeviceGetPCIBusId
        else:
            library = ctypes.CDLL(ctypes.util.find_library('cudart') or 'libcudart.so')
            function = library.cudaDeviceGetPCIBusId
        function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        function.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(64)
        if function(buffer, 64, ctx.device.index):
            raise ValueError('native GPU PCI identity query failed')
        physical = buffer.value.decode().lower()
        if physical.count(':') == 1:
            physical = '0000:' + physical
    return {'rank': ctx.rank, 'world_size': ctx.world_size, 'pid': os.getpid(), 'device': str(ctx.device),
            'physical_pci': physical, 'total_memory_bytes': total, 'requested_cpu_ids': requested,
            'actual_cpu_ids': actual, 'intraop_threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'native_affinity_environment': native_affinity,
            'native_affinity_guard': cpu_policy.get('native_affinity_guard') if ctx.device.type == 'cuda' else None,
            'native_affinity_guard_loaded': native_guard_loaded(cpu_policy) if ctx.device.type == 'cuda' else False,
            'at': time.time(), 'CPU_only_software_proof': ctx.device.type != 'cuda'}

"""Configured single-model topology, independent from the immutable R9 policy."""
import copy
import os
from pathlib import Path

from .distributed import layout


def resolve_topology(cfg, *, available_accelerators):
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
    available = sorted(set(available if available is not None else
        (os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else range(os.cpu_count() or 1))))
    if quota_cores is None:
        path = Path('/sys/fs/cgroup/cpu.max')
        if path.is_file():
            quota, period = path.read_text().split()
            if quota != 'max':
                quota_cores = int(quota) / int(period)
    count = min(len(available), int(quota_cores)) if quota_cores is not None else len(available)
    if type(world) is not int or world <= 0 or count < world:
        raise ValueError('actual CPU affinity/quota cannot supply one CPU per configured rank')
    width = max(1, (count - 2) // world)
    masks = [available[rank * width:(rank + 1) * width] for rank in range(world)]
    if any(len(mask) != width for mask in masks):
        raise ValueError('empty or incomplete rank CPU affinity')
    return width, masks


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


def validate_general_fresh(spec, identity, job, topology):
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
            'actual_cpu_ids': actual, 'at': time.time(), 'CPU_only_software_proof': ctx.device.type != 'cuda'}

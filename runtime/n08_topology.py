"""Resolve and fail closed on the frozen N08 reference and rank batch contract."""
from retrain.execution_policy import resolve_topology


def resolve_n08_topology(cfg):
    if (len(cfg['jobs']) != 1 or cfg['jobs'][0]['physical_batch'] != 32
            or cfg['jobs'][0]['effective_batch'] != 64 or cfg['jobs'][0]['positive_per_batch'] != 8
            or cfg['execution']['devices'] != ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
            or cfg['execution']['per_rank_microbatch'] != 0):
        raise ValueError('N08 reference batch, quota or configured devices differ')
    job, topology, policy = resolve_topology(cfg, available_accelerators=4)
    expected = {'world_size': 4, 'reference_physical_batch': 32, 'per_rank_batch': 16,
                'accumulation': 1, 'global_effective_batch': 64, 'global_positive_quota': 8}
    if (any(topology.get(key) != value for key, value in expected.items())
            or job['physical_batch'] != 16):
        raise ValueError('N08 resolved rank topology differs')
    return job, topology, policy

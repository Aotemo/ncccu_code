"""Resolve the fixed reference batch against an explicitly admitted world."""
from retrain.execution_policy import resolve_topology


def resolve_n08_topology(cfg, *, available_accelerators, expected_world):
    if type(expected_world) is not int or expected_world not in (4, 8) or available_accelerators != expected_world:
        raise ValueError('visible accelerators differ from the explicitly admitted world')
    if (len(cfg['jobs']) != 1 or cfg['jobs'][0]['physical_batch'] != 32
            or cfg['jobs'][0]['effective_batch'] != 64 or cfg['jobs'][0]['positive_per_batch'] != 8
            or cfg['execution']['devices'] != ['cuda:' + str(rank) for rank in range(expected_world)]
            or cfg['execution']['per_rank_microbatch'] != 0):
        raise ValueError('N08 reference batch, quota or configured devices differ')
    job, topology, policy = resolve_topology(cfg, available_accelerators=available_accelerators)
    expected = {'world_size': expected_world, 'reference_physical_batch': 32, 'per_rank_batch': 64 // expected_world,
                'accumulation': 1, 'global_effective_batch': 64, 'global_positive_quota': 8}
    if (any(topology.get(key) != value for key, value in expected.items())
            or job['physical_batch'] != 64 // expected_world):
        raise ValueError('N08 resolved rank topology differs')
    return job, topology, policy

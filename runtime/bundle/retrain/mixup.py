"""Registered training-only rank-local16 Mixup, independent of all global RNGs."""
import hashlib
import random

RULE = {'version': 'rank-local-mixup01-v1', 'enabled': True, 'alpha': 0.1, 'probability': 1.0, 'scope': 'training_only', 'world_size': 4, 'local_batch': 16, 'pairing': 'uniform permutation of all16 local positions; self-pairs allowed; class-agnostic', 'lambda': 'one Beta(0.1,0.1) scalar per attempted update/rank/micro; no folding or clipping', 'rng': 'isolated random.Random(int.from_bytes(sha256(UTF8 colonjoin(job_seed,mixup01-v1,attempted_updates_before_increment,rank,micro)).digest()[:8],big)); first betavariate(.1,.1), then shuffle(list(range(16)))', 'apply': 'after original augmentation/normalization before forward/autocast; identical lambda/permutation to FP32 images, float labels and both FP32 age features', 'loss': 'mean soft-label BCE; pos_weight1; no smoothing or extra classweight', 'age_missing_feature': 'synthetic mixed missingness may be fractional; original raw fit/eval age contracts unchanged', 'sampling': 'preserve original primary sampleIDs and attempt cursor; rank0/2 4P12N and rank1/3 0P16N unchanged; scalar+permutation preserves local positive mass/global8', 'coverage': 'original primary IDs as before; partners from same16, never double-count exposures; no partner reload/cross-rank image transfer', 'resume': 'derive mixing from attempted cursor including skips; no global Python/numpy/torch RNG consumption; bind mixup fields in identity/fresh/checkpoint; reject changedworld/config on resume', 'validation': 'no mixing on dev/officialvalid/test; original fullbatch32 inference', 'limitation': 'local16 Mixup, not uniformglobal64 pairing; rank1/3 mix only negatives', 'telemetry': 'Use existing log intervals for reproduciblelambda/permutation/primary-partnerIDs andlabelmass; no perstep heavyweightcollective/image logging.'}


def validate_spec(spec):
    if not isinstance(spec, dict) or spec != RULE or spec.get('enabled') is not True:
        raise ValueError('Mixup rule differs from rank-local-mixup01-v1')
    return spec


def validate_execution(job, plan):
    if job.get('mixup') is None:
        return
    validate_spec(job['mixup'])
    wanted = {'world_size': 4, 'global_effective_batch': 64, 'global_positive_quota': 8,
              'reference_physical_batch': 32, 'per_rank_batch': 16, 'accumulation': 1}
    if (any(plan.get(key) != value for key, value in wanted.items())
            or job['physical_batch'] != 16 or job['model'] != 'convnext_tiny'
            or not (job.get('age') or {}).get('enabled')):
        raise ValueError('Mixup requires the registered Tiny/age world4 local16 layout')


def apply(spec, images, labels, age_features, *, job_seed, attempted, rank, micro,
          world_size, training):
    if spec is None or not training:
        return images, labels, age_features, None
    validate_spec(spec)
    if (world_size != 4 or any(type(v) is not int or v < 0 for v in (job_seed, attempted, rank, micro))
            or not 0 <= rank < 4):
        raise ValueError('Mixup topology/cursor differs from the registered attempt/rank/micro')
    import torch
    if (images.ndim != 4 or images.shape[0] != 16 or labels.shape != (16,)
            or age_features is None or age_features.shape != (16, 2)
            or any(value.dtype != torch.float32 or value.device != images.device
                   for value in (images, labels, age_features))):
        raise ValueError('Mixup needs aligned local16 FP32 images/labels/two age features')
    key = ':'.join(map(str, (job_seed, 'mixup01-v1', attempted, rank, micro)))
    seed = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'big')
    rng = random.Random(seed)
    lam = rng.betavariate(.1, .1)
    permutation = list(range(16))
    rng.shuffle(permutation)
    index = torch.tensor(permutation, device=images.device, dtype=torch.long)

    def blend(value):
        return value * lam + value.index_select(0, index) * (1 - lam)

    proof = {'version': spec['version'], 'seed': seed, 'lambda': lam, 'permutation': permutation,
             'attempted_before_increment': attempted, 'rank': rank, 'micro': micro}
    return blend(images), blend(labels), blend(age_features), proof

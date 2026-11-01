"""Original fixed local8 feature-statistics mixing; no external package or RNG state."""
import copy
import hashlib
import random

import torch
from torch import nn

MIXSTYLE_RULE = {'version': 'convnext-small-local8-mixstyle13-v1', 'sites': [1, 3],
    'probability': .5, 'alpha': .1, 'epsilon': 1e-6, 'variance_correction': 1,
    'scope': 'rank_local8', 'salt': 'mixstyle13-local8-v1',
    'rng': 'sha256-first8-big-private-python-v1'}


def validate_job(job):
    policy = job.get('mixstyle_policy')
    if policy is None:
        return None
    if (policy != MIXSTYLE_RULE or job.get('model') != 'convnext_small'
            or job.get('precision') != 'fp32' or job.get('head_lr_scope') != 'last_linear'
            or not (job.get('age') or {}).get('enabled')
            or job.get('effective_batch') != 64 or job.get('positive_per_batch') != 8
            or any(job.get(key) is not None for key in
                ('mixup','tri3_downsample_policy','output_regularization','low_rank_policy',
                 'layerwise_lr_policy','freeze_features_before','final_downsample_stride',
                 'attention_pool','stage3_residual','head_warmup_policy','gradient_clip_policy',
                 'training_views'))):
        raise ValueError('MixStyle requires the fixed FP32 full Small age/local8 policy')
    return policy


def mix_statistics(x, lambdas, permutation):
    if (x.dtype != torch.float32 or x.ndim != 4 or x.shape[0] != 8
            or x.shape[2]*x.shape[3] < 2 or lambdas.shape != (8,1,1,1)
            or lambdas.dtype != torch.float32 or permutation.shape != (8,)
            or permutation.dtype != torch.int64
            or lambdas.device != x.device or permutation.device != x.device):
        raise ValueError('FP32 local8 feature/statistic coefficient shapes differ')
    mu = x.mean(dim=[2,3], keepdim=True)
    variance = x.var(dim=[2,3], correction=1, keepdim=True)
    sigma = (variance + 1e-6).sqrt()
    mu, sigma = mu.detach(), sigma.detach()
    normalized = (x-mu)/sigma
    mixed_sigma = lambdas*sigma + (1-lambdas)*sigma[permutation]
    mixed_mu = lambdas*mu + (1-lambdas)*mu[permutation]
    return normalized*mixed_sigma + mixed_mu


def apply(x, policy, *, training, seed, context, site):
    if policy is None or not training:
        return x
    if policy != MIXSTYLE_RULE or not isinstance(context, dict):
        raise ValueError('exact MixStyle policy and explicit attempt context required')
    attempted, rank, micro = (context.get(key) for key in ('attempted_before_increment','rank','micro'))
    if (any(type(value) is not int or value < 0 for value in (seed,attempted,rank,micro))
            or rank >= 8 or site not in (1,3)
            or x.dtype != torch.float32 or x.ndim != 4 or x.shape[0] != 8):
        raise ValueError('MixStyle seed/attempt/rank/micro/site or local batch differs')
    key = ':'.join(map(str, (seed, MIXSTYLE_RULE['salt'], attempted, rank, micro, site)))
    integer = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'big')
    private = random.Random(integer)
    gate = private.random()
    coefficients, permutation = None, None
    active = gate <= .5
    if active:
        coefficients = [private.betavariate(.1,.1) for _ in range(8)]
        permutation = list(range(8)); private.shuffle(permutation)
    trace = context.get('trace')
    if trace is not None:
        if not isinstance(trace, list):
            raise ValueError('MixStyle trace belongs to the caller list')
        trace.append({'attempted_before_increment':attempted,'rank':rank,'micro':micro,
            'site_index':site,'rng_seed_u64':integer,'gate_draw':gate,'active':active,
            'lambdas':coefficients,'permutation':permutation})
    if not active:
        return x
    return mix_statistics(x, torch.tensor(coefficients,dtype=torch.float32,device=x.device).reshape(8,1,1,1),
                          torch.tensor(permutation,dtype=torch.int64,device=x.device))


def _sites(model):
    from torchvision.models.convnext import CNBlock
    features = model.network.features
    if (type(features) is not nn.Sequential or len(features) != 8
            or features[0][0].out_channels != 96 or features[0][0].stride != (4,4)
            or features[2][1].in_channels != 96 or features[2][1].out_channels != 192
            or features[2][1].stride != (2,2) or features[2][1].kernel_size != (2,2)):
        raise ValueError('MixStyle native early Small transitions differ')
    result = []
    for index, channels in ((1,96),(3,192)):
        stage = features[index]
        if (type(stage) is not nn.Sequential or len(stage) != 3
                or any(not isinstance(block,CNBlock)
                       or block.block[3].in_features != channels
                       or block.block[3].out_features != 4*channels
                       or block.block[5].in_features != 4*channels
                       or block.block[5].out_features != channels for block in stage)):
            raise ValueError('MixStyle sites must follow the two complete native stages')
        result.append({'feature_index':index,'feature_name':f'network.features.{index}',
                       'channels':channels,'blocks':len(stage)})
    return result


def attach_mixstyle(model, job):
    policy = validate_job(job)
    if policy is None:
        return model
    _sites(model)
    model.mixstyle_policy = copy.deepcopy(policy)
    model.mixstyle_seed = job['seed']
    return model


def describe_mixstyle(model, job):
    policy = validate_job(job)
    if policy is None:
        return None
    if getattr(model,'mixstyle_policy',None) != policy or getattr(model,'mixstyle_seed',None) != job['seed']:
        raise ValueError('MixStyle reconstruction lacks the saved policy/seed')
    return {'policy':copy.deepcopy(policy),'sites':_sites(model),
            'training_only':True,'persistent_rng_state':False}


def training_image_logits(model, images, context):
    features = images
    for index, module in enumerate(model.network.features):
        features = module(features)
        if index in (1,3):
            features = apply(features,model.mixstyle_policy,training=True,seed=model.mixstyle_seed,
                             context=context,site=index)
    return model.network.classifier(model.network.avgpool(features)).flatten()

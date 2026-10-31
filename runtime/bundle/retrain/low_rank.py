"""Fixed Small pointwise rank-8 factors, always evaluated without weight merging.

Original implementation using PyTorch operations. A per-module CPU generator
initializes A; B starts at zero. Constructing factors never consumes model RNG.
"""
import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

LOW_RANK_RULE = {'version': 'convnext-small-pointwise-lora8-v1', 'rank': 8, 'alpha': 8,
                 'targets': 'all-36-block-pointwise-3-5', 'base': 'frozen', 'merge': False}


def validate_policy(policy):
    if policy is not None and policy != LOW_RANK_RULE:
        raise ValueError('pointwise low-rank policy differs from the fixed rank-8 protocol')
    return policy


def validate_job(job):
    policy = validate_policy(job.get('low_rank_policy'))
    if policy is None:
        return None
    if (job.get('model') != 'convnext_small' or job.get('precision') != 'fp32'
            or job.get('head_lr_scope') != 'last_linear'
            or not (job.get('age') or {}).get('enabled')
            or any(job.get(key) is not None for key in (
                'layerwise_lr_policy', 'head_warmup_policy', 'gradient_clip_policy',
                'attention_pool', 'stage3_residual', 'freeze_features_before',
                'final_downsample_stride'))):
        raise ValueError('fixed pointwise factors require the FP32 Small GAP+age model')
    return policy


def target_shapes():
    return {f'network.features.{stage}.{block}.block.{index}': shape
            for stage, count, width in ((1, 3, 96), (3, 3, 192), (5, 27, 384), (7, 3, 768))
            for block in range(count)
            for index, shape in ((3, (width, 4 * width)), (5, (4 * width, width)))}


class PointwiseLowRankLinear(nn.Linear):
    """Keep the original weight/bias names and objects; no train/eval mutation."""
    def __init__(self, linear, seed, module_name):
        if (type(linear) is not nn.Linear or linear.weight.device.type != 'cpu'
                or linear.weight.dtype != torch.float32):
            raise ValueError('attach factors to an original CPU FP32 Linear')
        # Calling Linear.__init__ would initialize a new base and consume RNG.
        nn.Module.__init__(self)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight.requires_grad_(False)
        self.bias = linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        rank = LOW_RANK_RULE['rank']
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32))
        material = f'{seed}:{LOW_RANK_RULE["version"]}:{module_name}'.encode('utf-8')
        factor_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], 'big') % (2 ** 63)
        generator = torch.Generator(device='cpu').manual_seed(factor_seed)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
        self.train(linear.training)

    def forward(self, value):
        return F.linear(value, self.weight, self.bias) + F.linear(F.linear(value, self.lora_A), self.lora_B)


def attach_low_rank(model, job):
    if validate_job(job) is None:
        return model
    targets = target_shapes()
    modules = dict(model.named_modules())
    for name, (inputs, outputs) in targets.items():
        module = modules.get(name)
        if (type(module) is not nn.Linear or (module.in_features, module.out_features) != (inputs, outputs)
                or module.weight.device.type != 'cpu' or module.weight.dtype != torch.float32):
            raise ValueError('native Small pointwise target differs: ' + name)
    model.network.requires_grad_(False)
    model.network.classifier[-1].requires_grad_(True)
    for name in targets:
        parent, child = name.rsplit('.', 1)
        setattr(modules[parent], child, PointwiseLowRankLinear(modules[name], job['seed'], name))
    model.low_rank_spec = copy.deepcopy(job['low_rank_policy'])
    model.low_rank_initial_frozen_sha256 = frozen_digest(model)
    low_rank_proof(model, job)
    return model


def optimizer_group_identity(model, optimizer):
    named = {id(parameter): name for name, parameter in model.named_parameters()}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    members = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if len(members) != len(expected) or {id(p) for p in members} != expected:
        raise ValueError('low-rank optimizer must cover each trainable parameter exactly once')
    result = []
    for group in optimizer.param_groups:
        names = [named[id(parameter)] for parameter in group['params']]
        shapes = [list(parameter.shape) for parameter in group['params']]
        if group.get('param_names') != names or group.get('param_shapes') != shapes:
            raise ValueError('low-rank optimizer names/shapes differ from actual positions')
        result.append({key: copy.deepcopy(group[key]) for key in
                       ('role', 'initial_lr', 'weight_decay', 'param_names', 'param_shapes')})
    return result


def bind_optimizer(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for group in optimizer.param_groups:
        group['param_names'] = [names[id(parameter)] for parameter in group['params']]
        group['param_shapes'] = [list(parameter.shape) for parameter in group['params']]
    optimizer_group_identity(model, optimizer)
    return optimizer


def frozen_digest(model):
    from .ema import tensor_digest
    head = {id(parameter) for parameter in model.network.classifier[-1].parameters()}
    return tensor_digest({name: parameter for name, parameter in model.named_parameters()
                          if id(parameter) not in head and name != 'age_weight'
                          and not name.endswith(('.lora_A', '.lora_B'))})


def optimizer_update_proof(model, optimizer, successful):
    """Read actual Adam state and compare it to the committed successful cursor."""
    if type(successful) is not int or successful < 0:
        raise ValueError('nonnegative successful cursor required')
    optimizer_group_identity(model, optimizer)
    parameters = {parameter for group in optimizer.param_groups for parameter in group['params']}
    if set(optimizer.state) != (parameters if successful else set()):
        raise ValueError('low-rank Adam state coverage differs from the actual trainable set')
    observed_steps = []
    for parameter, state in optimizer.state.items():
        step = state.get('step')
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or not bool(torch.isfinite(step).all()) or float(step) != successful):
            raise ValueError('low-rank actual Adam step differs from successful cursor')
        observed_steps.append(int(step.item()))
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state.get(key)
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise ValueError('low-rank actual Adam moment shape/dtype/finite differs')
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    if any(parameter.grad is not None or parameter in optimizer.state for parameter in frozen):
        raise ValueError('frozen low-rank base acquired gradients or Adam state')
    return {'successful_updates': successful, 'optimizer_state_entries': len(optimizer.state),
            'observed_Adam_steps': sorted(set(observed_steps)),
            'trainable_parameter_tensors': len(parameters), 'frozen_parameter_tensors': len(frozen),
            'frozen_gradients_absent': True, 'frozen_Adam_absent': True}


def validate_optimizer_state(model, optimizer, saved):
    expected = optimizer_group_identity(model, optimizer)
    groups = saved.get('param_groups')
    current = optimizer.state_dict()['param_groups']
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise ValueError('saved low-rank optimizer group count differs')
    positions = []
    shapes = {}
    for identity, stored, actual in zip(expected, groups, current):
        if (any(stored.get(key) != value for key, value in identity.items())
                or stored.get('params') != actual['params']):
            raise ValueError('saved low-rank optimizer named parameter positions differ')
        positions.extend(stored['params'])
        shapes.update(zip(stored['params'], identity['param_shapes']))
    if len(positions) != len(set(positions)) or not set(saved.get('state', {})).issubset(positions):
        raise ValueError('saved low-rank Adam state has an unbound parameter position')
    for position, state in saved.get('state', {}).items():
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = state.get(key)
            if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != shapes[position]
                    or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())):
                raise ValueError('saved low-rank Adam moment shape/dtype/finite differs')
    return expected


def describe_low_rank(model, job):
    """Structural base/factor roles also apply to the wholly frozen EMA readout."""
    if validate_job(job) is None:
        return None
    if getattr(model, 'low_rank_spec', None) != job['low_rank_policy']:
        raise ValueError('model low-rank protocol differs')
    modules = dict(model.named_modules())
    targets = target_shapes()
    actual_targets = {name for name, module in modules.items() if isinstance(module, PointwiseLowRankLinear)}
    if actual_targets != set(targets) or dict(model.named_buffers()):
        raise ValueError('low-rank targets or no-buffer structure differ')
    named = dict(model.named_parameters())
    head = {id(parameter) for parameter in model.network.classifier[-1].parameters()}
    expected = sorted(name for name, parameter in named.items()
                      if id(parameter) in head or name == 'age_weight'
                      or any(name == target + suffix for target in targets for suffix in ('.lora_A', '.lora_B')))
    trainable = expected
    frozen = sorted(set(named) - set(trainable))
    rows = []
    for name, (inputs, outputs) in targets.items():
        module = modules[name]
        if (tuple(module.weight.shape) != (outputs, inputs)
                or tuple(module.lora_A.shape) != (8, inputs)
                or tuple(module.lora_B.shape) != (outputs, 8)
                or any(parameter.dtype != torch.float32 for parameter in module.parameters())):
            raise ValueError('low-rank factor/base shapes or FP32 differ')
        rows.append({'name': name, 'in_features': inputs, 'out_features': outputs,
                     'A_shape': [8, inputs], 'B_shape': [outputs, 8]})
    actual_frozen = frozen_digest(model)
    if actual_frozen != getattr(model, 'low_rank_initial_frozen_sha256', None):
        raise ValueError('low-rank frozen base differs from its attachment reference')
    result = {'low_rank_policy': copy.deepcopy(job['low_rank_policy']), 'targets': rows,
              'initial_frozen_tensor_sha256': model.low_rank_initial_frozen_sha256,
              'frozen_tensor_sha256': actual_frozen,
              'all_parameter_names': sorted(named), 'frozen_parameter_names': frozen,
              'trainable_parameter_names': trainable,
              'all_parameter_elements': sum(p.numel() for p in named.values()),
              'frozen_parameter_elements': sum(named[n].numel() for n in frozen),
              'trainable_parameter_elements': sum(named[n].numel() for n in trainable)}
    return result


def low_rank_proof(model, job, optimizer=None):
    if validate_job(job) is None:
        return None
    result = describe_low_rank(model, job)
    frozen = sorted(name for name, parameter in model.named_parameters() if not parameter.requires_grad)
    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if frozen != result['frozen_parameter_names'] or trainable != result['trainable_parameter_names']:
        raise ValueError('low-rank actual gradient mask differs from structural roles')
    if optimizer is not None:
        result['optimizer_groups'] = optimizer_group_identity(model, optimizer)
        result['optimizer_parameter_names'] = sorted(n for g in result['optimizer_groups'] for n in g['param_names'])
    return result


def validate_checkpoint(payload, job):
    if validate_job(job) is None:
        return
    from .config import job_contract
    from .ema import tensor_digest
    if (job_contract(payload['job']) != job_contract(job)
            or payload.get('identity', {}).get('job') != job_contract(job)):
        raise ValueError('checkpoint low-rank job contract differs')
    proof = payload.get('parameter_trainability', {})
    state = payload['model']
    factors = {target + suffix for target in target_shapes() for suffix in ('.lora_A', '.lora_B')}
    head = {'network.classifier.3.weight', 'network.classifier.3.bias', 'age_weight'}
    trainable = sorted(factors | head)
    frozen = sorted(set(state) - set(trainable))
    if (proof.get('low_rank_policy') != job['low_rank_policy']
            or proof.get('all_parameter_names') != sorted(state)
            or proof.get('frozen_parameter_names') != frozen
            or proof.get('trainable_parameter_names') != trainable
            or proof.get('optimizer_parameter_names') != trainable
            or not set(trainable).issubset(state) or len(state) != 489 or len(frozen) != 342):
        raise ValueError('checkpoint low-rank frozen/trainable identity differs')
    digest = tensor_digest({name: state[name] for name in frozen})
    if digest != proof.get('initial_frozen_tensor_sha256') or digest != proof.get('frozen_tensor_sha256'):
        raise ValueError('checkpoint frozen base differs from its initial reference')
    if payload.get('ema') is not None:
        ema = payload['ema']['model']
        if state.keys() != ema.keys() or any(not torch.equal(state[n], ema[n]) for n in frozen):
            raise ValueError('checkpoint EMA frozen base differs from online')


def restore_base_reference(model, job, payload):
    if validate_job(job) is None:
        return
    validate_checkpoint(payload, job)
    digest = frozen_digest(model)
    if digest != payload['parameter_trainability']['initial_frozen_tensor_sha256']:
        raise ValueError('strictly loaded low-rank base differs from checkpoint reference')
    model.low_rank_initial_frozen_sha256 = digest

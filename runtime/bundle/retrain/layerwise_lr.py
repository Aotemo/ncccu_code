"""R130 explicit ConvNeXt-Small layerwise LR scales on top of existing groups."""
LAYERWISE_LR_RULE = {
    'version': 'convnext-small-layerwise-lr-decay08-v1',
    'kind': 'optimizer_group_layerwise_lr_decay',
    'mapping': 'facebook-convnext-small-group12-tv-v1',
    'layer_decay': 0.8, 'num_layers': 12,
    'backbone_scale': 'layer_decay**(13-layer_id)',
    'head_scale': 1.0, 'age_scale': 1.0,
    'schedule': 'original_global_successful_cursor',
    'weight_decay': 'original_per_parameter_rule',
    'parameter_membership': 'unique_trainable_optimizer_parameters',
    'precision': 'fp32'}


def _validate(policy):
    if not isinstance(policy, dict) or policy != LAYERWISE_LR_RULE:
        raise ValueError('layerwise LR policy differs from the frozen protocol')
    return policy


def _layer_id(name, role):
    if role == 'age':
        if name != 'age_weight':
            raise ValueError('age role must contain only age_weight')
        return 13
    if role == 'head':
        if not name.startswith('network.classifier.3.'):
            raise ValueError('head role must contain only the final binary Linear')
        return 13
    if role != 'backbone':
        raise ValueError('unsupported optimizer role')
    if name.startswith('network.classifier.0.'):
        return 13
    if name.startswith('network.features.0.'):
        return 0
    if name.startswith('network.features.1.'):
        return 1
    if name.startswith('network.features.2.') or name.startswith('network.features.3.'):
        return 2
    if name.startswith('network.features.4.'):
        return 3
    if name.startswith('network.features.5.'):
        block = int(name.split('.')[3])
        if not 0 <= block <= 26:
            raise ValueError('stage-three block outside the Small mapping')
        return 3 + block // 3
    if name.startswith('network.features.6.') or name.startswith('network.features.7.'):
        return 12
    raise ValueError('parameter is outside the fixed Small layer map: ' + name)


def _scale(role, layer_id):
    if role == 'backbone':
        return LAYERWISE_LR_RULE['layer_decay'] ** (
            LAYERWISE_LR_RULE['num_layers'] + 1 - layer_id)
    return LAYERWISE_LR_RULE['head_scale'] if role == 'head' else LAYERWISE_LR_RULE['age_scale']


def build_layerwise_groups(model, base_groups, policy):
    if policy is None:
        return base_groups
    _validate(policy)
    names = {id(parameter): name for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    base_ids = [id(parameter) for group in base_groups for parameter in group['params']]
    if len(base_ids) != len(set(base_ids)) or set(base_ids) != set(names):
        raise ValueError('base groups do not uniquely cover all trainable parameters')
    result = []
    for group in base_groups:
        role = group.get('role')
        buckets = {}
        for parameter in group['params']:
            name = names[id(parameter)]
            layer_id = _layer_id(name, role)
            buckets.setdefault(layer_id, []).append(parameter)
        for layer_id in sorted(buckets):
            parameters = buckets[layer_id]
            child = {key: value for key, value in group.items() if key != 'params'}
            base = float(group['initial_lr'])
            scale = _scale(role, layer_id)
            child.update({'params': parameters, 'layer_id': layer_id,
                          'lr_scale': scale, 'base_initial_lr': base,
                          'initial_lr': base * scale, 'lr': base * scale,
                          'param_names': [names[id(parameter)] for parameter in parameters]})
            result.append(child)
    return result


def group_identity(model, groups, policy):
    if policy is None:
        return None
    _validate(policy)
    names = {id(parameter): name for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    seen = set()
    result = []
    for group in groups:
        role = group.get('role')
        layer_id = group.get('layer_id')
        scale = group.get('lr_scale')
        base = group.get('base_initial_lr')
        initial = group.get('initial_lr')
        param_names = group.get('param_names')
        if (role not in ('backbone', 'head', 'age') or type(layer_id) is not int
                or not isinstance(scale, (int, float)) or not isinstance(base, (int, float))
                or not isinstance(initial, (int, float)) or float(base) <= 0
                or not isinstance(param_names, list)
                or len(param_names) != len(group['params'])
                or not isinstance(group.get('weight_decay'), (int, float))):
            raise ValueError('layerwise group metadata differs')
        for parameter, name in zip(group['params'], param_names):
            if names.get(id(parameter)) != name or id(parameter) in seen:
                raise ValueError('layerwise parameter identity differs')
            seen.add(id(parameter))
            if _layer_id(name, role) != layer_id:
                raise ValueError('layerwise layer id differs')
            if abs(_scale(role, layer_id) - float(scale)) > 1e-12:
                raise ValueError('layerwise scale differs')
        if abs(float(base) * float(scale) - float(initial)) > 1e-12:
            raise ValueError('layerwise initial_lr does not equal base*scale')
        result.append({'role': role, 'layer_id': layer_id, 'lr_scale': float(scale),
                       'base_initial_lr': float(base), 'initial_lr': float(initial),
                       'weight_decay': float(group['weight_decay']),
                       'param_names': list(param_names)})
    if seen != set(names):
        raise ValueError('layerwise groups do not uniquely cover all trainable parameters')
    return result

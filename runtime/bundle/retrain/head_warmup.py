"""R128 fixed head/age-only 1600-update warmup before the full optimizer roles."""
import torch

HEAD_WARMUP_RULE = {
    'version': 'head-age-only1600-then-full-v1', 'kind': 'optimizer_role_head_warmup',
    'head_only_successful_updates': 1600, 'roles': ['backbone', 'head', 'age'],
    'warmup_active_roles': ['head', 'age'],
    'backbone_action': 'grad_none_after_world_finite',
    'computed_gradient_check': 'all_roles_before_mask',
    'active_clipping': 'combined_global_l2', 'max_norm': 1.0, 'norm_type': 2.0,
    'epsilon': 1e-6, 'nonfinite': 'world_consensus_before_mutation',
    'parameter_membership': 'unique_optimizer_parameters',
    'optimizer_groups': 'unchanged',
    'backbone_lr_schedule': 'original_global_successful_cursor',
    'ema_schedule': 'original_absolute_successful_cursor', 'precision': 'fp32'}


def _validate_policy(policy):
    if not isinstance(policy, dict) or policy != HEAD_WARMUP_RULE:
        raise ValueError('head warmup policy differs from the frozen protocol')
    return policy


def _norm(pieces, device):
    if not pieces:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.linalg.vector_norm(torch.stack(pieces), ord=2)


def build_update_groups(optimizer, policy):
    if policy is None:
        return None
    roles = _validate_policy(policy)['roles']
    groups = {role: [] for role in roles}
    seen = set()
    for group in optimizer.param_groups:
        role = group.get('role')
        if role not in groups:
            raise ValueError('optimizer parameter group has no supported role')
        for parameter in group['params']:
            if id(parameter) in seen:
                raise ValueError('optimizer parameter appears more than once across roles')
            seen.add(id(parameter))
            groups[role].append(parameter)
    if len(seen) != sum(len(group['params']) for group in optimizer.param_groups):
        raise ValueError('optimizer parameter membership is not unique')
    return groups


def mask_and_clip_gradients_(role_groups, successful_updates, any_bad=None):
    if type(successful_updates) is not int or successful_updates < 0:
        raise ValueError('successful update cursor must be a nonnegative integer')
    next_update = successful_updates + 1
    warmup = next_update <= HEAD_WARMUP_RULE['head_only_successful_updates']
    active_roles = (list(HEAD_WARMUP_RULE['warmup_active_roles']) if warmup
                    else list(HEAD_WARMUP_RULE['roles']))
    norms, pieces = {}, {}
    with torch.no_grad():
        device = next((parameter.device for role in HEAD_WARMUP_RULE['roles']
                       for parameter in role_groups[role]), torch.device('cpu'))
        for role in HEAD_WARMUP_RULE['roles']:
            parameters = role_groups[role]
            pieces[role] = [torch.linalg.vector_norm(parameter.grad.detach().float(), ord=2)
                            for parameter in parameters if parameter.grad is not None]
            norms[role] = _norm(pieces[role], device)
        active_pieces = [value for role in active_roles for value in pieces[role]]
        active_global = _norm(active_pieces, device)
        finite = bool(torch.stack([norms[role] for role in HEAD_WARMUP_RULE['roles']]
                                  + [active_global]).isfinite().all())
    local_bad = not finite
    world_bad = bool(any_bad(local_bad)) if any_bad is not None else local_bad
    info = {'phase': 'warmup' if warmup else 'full', 'next_successful_update': next_update,
            'active_roles': active_roles, 'role_norms': norms,
            'active_global_norm': active_global, 'clip_coefficient': None}
    if local_bad or world_bad:
        return False, info
    coefficient = torch.clamp(1.0 / (active_global + HEAD_WARMUP_RULE['epsilon']),
                              max=HEAD_WARMUP_RULE['max_norm'])
    info['clip_coefficient'] = coefficient
    if warmup:
        for parameter in role_groups['backbone']:
            parameter.grad = None
    for role in active_roles:
        for parameter in role_groups[role]:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return True, info


def expected_optimizer_steps(successful_updates):
    if type(successful_updates) is not int or successful_updates < 0:
        raise ValueError('successful update cursor must be a nonnegative integer')
    warmup = HEAD_WARMUP_RULE['head_only_successful_updates']
    if successful_updates == 0:
        return {'backbone': None, 'head': None, 'age': None}
    if successful_updates <= warmup:
        return {'backbone': None, 'head': successful_updates, 'age': successful_updates}
    return {'backbone': successful_updates - warmup, 'head': successful_updates,
            'age': successful_updates}

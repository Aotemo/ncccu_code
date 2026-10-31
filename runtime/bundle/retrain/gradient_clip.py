"""R126 optional optimizer-rolewise L2 clipping; None keeps the legacy global path."""
import torch

ROLEWISE_CLIP_RULE = {
    'version': 'optimizer-rolewise-l2-norm1-v1', 'kind': 'optimizer_rolewise_l2',
    'roles': ['backbone', 'head', 'age'], 'max_norm': 1.0, 'norm_type': 2.0,
    'epsilon': 1e-6, 'merge_decay_groups': True,
    'scope': 'ddp_averaged_unscaled_gradients',
    'nonfinite': 'world_consensus_before_mutation',
    'global_second_clip': False, 'parameter_membership': 'unique_optimizer_parameters'}


def _validate_policy(policy):
    if not isinstance(policy, dict) or policy != ROLEWISE_CLIP_RULE:
        raise ValueError('gradient clip policy differs from the frozen rolewise protocol')
    return policy


def build_role_groups(optimizer, policy):
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


def measure_role_norms(role_groups):
    norms = {}
    with torch.no_grad():
        for role in ROLEWISE_CLIP_RULE['roles']:
            parameters = role_groups[role]
            device = parameters[0].device if parameters else torch.device('cpu')
            pieces = [torch.linalg.vector_norm(parameter.grad.detach().float(), ord=2)
                      for parameter in parameters if parameter.grad is not None]
            if pieces:
                norms[role] = torch.linalg.vector_norm(torch.stack(pieces), ord=2)
            else:
                norms[role] = torch.zeros((), dtype=torch.float32, device=device)
    return norms


def apply_role_clip_(role_groups, norms):
    roles = ROLEWISE_CLIP_RULE['roles']
    if set(norms) != set(roles):
        raise ValueError('rolewise clip requires all three norms')
    if any(not isinstance(norms[role], torch.Tensor) or norms[role].ndim != 0
           for role in roles):
        raise ValueError('rolewise clip requires three scalar norms')
    if not bool(torch.stack([norms[role].detach().float() for role in roles]).isfinite().all()):
        raise ValueError('rolewise clip requires finite local norms')
    coefficients = {role: torch.clamp(
        1.0 / (norms[role].detach().float() + ROLEWISE_CLIP_RULE['epsilon']),
        max=ROLEWISE_CLIP_RULE['max_norm']) for role in roles}
    for role in roles:
        coefficient = coefficients[role]
        for parameter in role_groups[role]:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return coefficients

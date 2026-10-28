"""Frozen FP32/no-scaling runtime protocol; no data, model or RNG mutation."""
RULE = {'autocast_enabled': False,
 'cuda_matmul_allow_tf32': False,
 'cudnn_allow_tf32': False,
 'cyclic_evaluation_precision': 'fp32',
 'effective_scale': 1.0,
 'float32_matmul_precision': 'highest',
 'nonfinite_gradient': 'hard_stop_before_optimizer_step',
 'official_valid_precision': 'fp32',
 'resume': 'same world4/full identity/precision protocol only; native empty '
           'scaler dictionary required; no conversion from R13',
 'scaler_enabled': False,
 'scaler_state_dict': {},
 'training_precision': 'fp32',
 'unexpected_skip': 'hard_stop',
 'version': 'fp32-no-loss-scaling-v1'}


def validate_protocol(protocol, precision='fp32'):
    if precision != 'fp32' or protocol != RULE:
        raise ValueError('FP32 precision protocol differs from the frozen no-scaling rule')


def validate_scaler_state(state, precision):
    if precision == 'fp32' and state != {}:
        raise ValueError('FP32 requires the native empty disabled GradScaler state')


def validate_observation(proof):
    expected = {'precision': 'fp32', 'autocast_enabled': False,
        'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False,
        'cudnn_allow_tf32': False}
    if not isinstance(proof, dict) or any(proof.get(k) != v for k, v in expected.items()):
        raise ValueError('actual FP32 runtime settings differ')
    if 'scaler_enabled' in proof:
        if proof['scaler_enabled'] is not False or proof.get('effective_scale') != 1.0:
            raise ValueError('FP32 requires disabled scaler and effective scale1')
        validate_scaler_state(proof.get('scaler_state_dict'), 'fp32')


def runtime_proof(protocol, *, scaler=None, device=None):
    validate_protocol(protocol)
    import torch
    kind = torch.device(device or 'cpu').type
    proof = {'precision': 'fp32', 'autocast_enabled': torch.is_autocast_enabled(kind),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
    if scaler is not None:
        proof.update(scaler_enabled=scaler.is_enabled(), effective_scale=scaler.get_scale(),
                     scaler_state_dict=scaler.state_dict())
    validate_observation(proof)
    return proof


def configure(protocol):
    validate_protocol(protocol)
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_autocast_enabled('cpu', False)
    torch.set_autocast_enabled('cuda', False)
    return runtime_proof(protocol)

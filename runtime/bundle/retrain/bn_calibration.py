"""One fit-only pass over fixed public R50 BN activations; Torch imports lazily.

The caller owns fit membership, deterministic eval pixels, WORLD sharding and
asset provenance. This module accepts image tensors only, never rows or labels.
The resulting moments describe the public reference, not a fixed point of the
network after all BN buffers have been replaced.
"""


_SPEC = {
    'version': 'fit-bn-fixed-reference-v1',
    'estimator': 'pooled_input_moments',
    'reference': 'public_bn_eval',
    'accumulation_dtype': 'float64',
    'variance': 'unbiased',
    'output_dtype': 'float32',
}


def validate_calibration_spec(spec):
    if not isinstance(spec, dict) or spec != _SPEC:
        raise ValueError('BN calibration requires the fixed-reference six-field contract')
    return dict(spec)


def _merge_moments(left, right):
    """Chan merge of (Python integer count, FP64 channel mean, FP64 M2)."""
    a, mean_a, m2_a = left
    b, mean_b, m2_b = right
    if not a:
        return right
    if not b:
        return left
    total = a + b
    delta = mean_b - mean_a
    return (total, mean_a + delta * (b / total),
            m2_a + m2_b + delta.square() * (a * b / total))


def calibrate_fit_bn(model, batches, spec):
    """Replace only BN means/variances after a complete fixed-reference pass.

    All collective operations use default WORLD and occur after local iteration.
    ``global_batches`` sums physical forwards across ranks, not training updates.
    Element counts include N*H*W per channel and are transmitted as exact FP64
    integers (never int32). No classification head, age branch or optimizer runs.
    """
    spec = validate_calibration_spec(spec)
    import torch
    import torch.distributed as dist
    from torch import nn
    from .ema import tensor_digest

    if not hasattr(model, 'backbone'):
        raise ValueError('R50 BN calibration requires model.backbone')
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not layers or any(isinstance(module, nn.SyncBatchNorm)
                         or not module.track_running_stats for _, module in layers):
        raise ValueError('calibration requires standard BN with running buffers')
    device = layers[0][1].running_mean.device
    if any(module.running_mean.device != device or module.running_var.device != device
           or module.running_mean.dtype != torch.float32
           or module.running_var.dtype != torch.float32 for _, module in layers):
        raise ValueError('BN reference must be FP32 on a single device')

    def bn_state():
        return {name + '.' + suffix: getattr(module, suffix)
                for name, module in layers
                for suffix in ('running_mean', 'running_var', 'num_batches_tracked')}

    public = {name: value.detach().clone() for name, value in bn_state().items()}
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    modes = [(module, module.training) for module in model.modules()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    before = tensor_digest(public)
    moments = [(0, torch.zeros_like(module.running_mean, dtype=torch.float64),
                torch.zeros_like(module.running_mean, dtype=torch.float64)) for _, module in layers]
    hooks = []
    local_images = local_batches = 0
    success = False

    def collect(index, module, args):
        values = args[0]
        if (not torch.is_tensor(values) or values.ndim != 4 or values.dtype != torch.float32
                or values.device != device or values.shape[1] != module.num_features):
            raise ValueError('R50 BN input must be FP32 NCHW on the reference device')
        count = values.shape[0] * values.shape[2] * values.shape[3]
        variance, mean = torch.var_mean(values.detach().to(torch.float64),
                                       dim=(0, 2, 3), unbiased=False)
        moments[index] = _merge_moments(moments[index], (count, mean, variance * count))

    def packed_count(value):
        if type(value) is not int or value < 0 or value > 2 ** 53:
            raise ValueError('moment count is outside exact FP64 integer transport')
        return torch.tensor([value], device=device, dtype=torch.float64)

    def unpack_count(value):
        number = float(value.item())
        if not number.is_integer() or number < 0 or number > 2 ** 53:
            raise ValueError('WORLD moment count is not an exact nonnegative integer')
        return int(number)

    try:
        for module, _ in modes:
            module.training = False
        for index, (_, module) in enumerate(layers):
            hooks.append(module.register_forward_pre_hook(
                lambda current, args, index=index: collect(index, current, args)))
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            for images in batches:
                if (not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] <= 0
                        or images.dtype != torch.float32 or images.device != device):
                    raise ValueError('calibration batches must be nonempty FP32 NCHW tensors')
                model.backbone(images)
                local_images += images.shape[0]
                local_batches += 1

            # Public running buffers must remain the reference for every forward.
            if any(not torch.equal(value, public[name]) for name, value in bn_state().items()):
                raise ValueError('reference BN buffers changed during calibration')
            if (dict(model.named_parameters()).keys() != parameters.keys()
                    or any(not torch.equal(value, parameters[name])
                           for name, value in model.named_parameters())):
                raise ValueError('a parameter changed during calibration')

            payload = [packed_count(local_images), packed_count(local_batches)]
            for count, mean, m2 in moments:
                payload.extend((packed_count(count), mean, m2))
            payload = torch.cat(payload).contiguous()
            distributed = dist.is_available() and dist.is_initialized()
            world_size = dist.get_world_size() if distributed else 1
            parts = [torch.empty_like(payload) for _ in range(world_size)]
            if distributed:
                dist.all_gather(parts, payload)
            else:
                parts[0] = payload

            # Every rank merges the gathered summaries in rank order.
            merged = [(0, torch.zeros_like(mean), torch.zeros_like(m2))
                      for _, mean, m2 in moments]
            global_images = global_batches = 0
            for part in parts:
                global_images += unpack_count(part[0])
                global_batches += unpack_count(part[1])
                offset = 2
                for index, (_, module) in enumerate(layers):
                    channels = module.num_features
                    count = unpack_count(part[offset])
                    mean = part[offset + 1:offset + 1 + channels]
                    m2 = part[offset + 1 + channels:offset + 1 + 2 * channels]
                    merged[index] = _merge_moments(merged[index], (count, mean, m2))
                    offset += 1 + 2 * channels

            # Check all layers and the FP32 casts before modifying any buffer.
            replacements = []
            for count, mean, m2 in merged:
                if count <= 1 or count < global_images:
                    raise ValueError('pooled BN variance needs a valid per-channel element count')
                variance = m2 / (count - 1)
                mean32, variance32 = mean.float(), variance.float()
                if (not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(m2).all())
                        or not bool(torch.isfinite(variance).all()) or not bool((variance >= 0).all())
                        or not bool(torch.isfinite(mean32).all())
                        or not bool(torch.isfinite(variance32).all())
                        or not bool((variance32 >= 0).all())):
                    raise FloatingPointError('pooled BN statistics are nonfinite or negative')
                replacements.append((mean32, variance32))
            for (_, module), (mean, variance) in zip(layers, replacements):
                module.running_mean.copy_(mean)
                module.running_var.copy_(variance)
            success = True
    finally:
        for hook in hooks:
            hook.remove()
        if not success:
            with torch.no_grad():
                for name, value in bn_state().items():
                    value.copy_(public[name])
        # Restore individual flags; model.train() would overwrite child modes.
        for module, training in modes:
            module.training = training
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    parameters_unchanged = all(torch.equal(value, parameters[name])
                               for name, value in model.named_parameters())
    counters_unchanged = all(torch.equal(module.num_batches_tracked,
                                         public[name + '.num_batches_tracked'])
                             for name, module in layers)
    restored = (all(module.training == training for module, training in modes)
                and torch.equal(torch.get_rng_state(), cpu_rng)
                and (cuda_rng is None or all(torch.equal(a, b) for a, b in
                     zip(torch.cuda.get_rng_state_all(), cuda_rng))))
    if not (parameters_unchanged and counters_unchanged and restored):
        raise RuntimeError('BN calibration leaked parameter, counter, mode or RNG state')
    return {
        'schema_version': 1,
        'scope': 'fit_bn_fixed_reference_calibration',
        'passed': True,
        'spec': spec,
        'layer_count': len(layers),
        'floating_buffer_count': 2 * len(layers),
        'counter_buffer_count': len(layers),
        'world_size': world_size,
        'local_images': local_images,
        'global_images': global_images,
        'local_batches': local_batches,
        'global_batches': global_batches,
        'per_channel_count': {name: item[0] for (name, _), item in zip(layers, merged)},
        'reference_bn_digest_before': before,
        'derived_bn_digest_after': tensor_digest(bn_state()),
        'parameters_unchanged': parameters_unchanged,
        'counters_unchanged': counters_unchanged,
        'mode_rng_restored': restored,
    }

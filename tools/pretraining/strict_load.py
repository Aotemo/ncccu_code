"""CPU-only source/converted/full1000 strict acceptance, without any forward.

This is source-asset acceptance. It does not activate a run or certify the final
candidate runtime's age, optimizer, or distributed initialization.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re

from source_chain import (configure_torch_threads, cpu_context, file_identity, read_pinned_receipt, require,
                          sha256, write_new_json, REVISION)
from stage_binding import validate_execution_binding


DEPTHS = (3, 3, 9, 3)
TIMM_GELU_FORWARD_SHA = '375052b2c2b44c570274dee239db60f55afda25d413c98c827561f63a0ca5a77'


def gelu_semantics(module):
    """Recognize only the inspected exact GELU implementations; never execute them."""
    import torch
    from torch.nn import functional as functional
    from timm.layers.activations import GELU as TimmGELU
    cls = type(module)
    qualified = cls.__module__ + '.' + cls.__qualname__
    forward = module.forward.__func__
    require(forward is cls.forward and forward.__globals__.get('F') is functional,
            'activation forward/function alias differs: ' + qualified)
    forward_sha = hashlib.sha256(inspect.getsource(forward).encode()).hexdigest()
    if cls is torch.nn.GELU:
        require(module.approximate == 'none', 'GELU approximate must be none')
    elif cls is TimmGELU:
        require(forward_sha == TIMM_GELU_FORWARD_SHA,
                'timm GELU forward differs from inspected return F.gelu(input)')
        require(re.search(r"approximate\s*=\s*['\"]none['\"]", functional.gelu.__doc__ or '') is not None,
                'actual functional GELU default is not documented as none')
    else:
        raise ValueError('unsupported activation implementation: ' + qualified)
    return {'qualname': qualified, 'repr': repr(module), 'verified_approximate': 'none',
            'forward_source_sha256': forward_sha}


def target_key(key):
    prefixes = {'stem.0': 'features.0.0', 'stem.1': 'features.0.1',
                'head.norm': 'classifier.0', 'head.fc': 'classifier.2'}
    for prefix, target in prefixes.items():
        if key in (prefix + '.weight', prefix + '.bias'):
            return target + key[len(prefix):], False
    match = re.fullmatch(r'stages\.([1-3])\.downsample\.([01])\.(weight|bias)', key)
    if match:
        return f'features.{2 * int(match[1])}.{match[2]}.{match[3]}', False
    match = re.fullmatch(r'stages\.([0-3])\.blocks\.(\d+)\.(conv_dw|norm|mlp\.fc1|mlp\.fc2)\.(weight|bias)', key)
    if match:
        stage, block = int(match[1]), int(match[2])
        require(block < DEPTHS[stage], 'block outside Tiny depth')
        operation = {'conv_dw': 0, 'norm': 2, 'mlp.fc1': 3, 'mlp.fc2': 5}[match[3]]
        return f'features.{2 * stage + 1}.{block}.block.{operation}.{match[4]}', False
    match = re.fullmatch(r'stages\.([0-3])\.blocks\.(\d+)\.gamma', key)
    if match:
        stage, block = int(match[1]), int(match[2])
        require(block < DEPTHS[stage], 'gamma block outside Tiny depth')
        return f'features.{2 * stage + 1}.{block}.layer_scale', True
    raise ValueError('unsupported public tensor: ' + key)


def check_tensor(source, actual, expected_shape, gamma=False):
    """Reject changed values/dtypes/shapes; gamma permits only [C] -> [C,1,1]."""
    import torch
    require(isinstance(source, torch.Tensor) and source.dtype == torch.float32
            and bool(torch.isfinite(source).all()), 'source nonfinite/non-FP32 tensor')
    require(not gamma or source.ndim == 1, 'gamma must be [C]')
    mapped = source.reshape(-1, 1, 1) if gamma else source
    require(isinstance(actual, torch.Tensor) and actual.dtype == torch.float32
            and actual.shape == mapped.shape == expected_shape
            and bool(torch.isfinite(actual).all()) and torch.equal(actual, mapped),
            'converted value/shape/dtype differs')


def check_mapping(source, converted, expected):
    require(isinstance(source, dict) and isinstance(converted, dict)
            and len(source) == len(converted) == len(expected) == 182, 'Tiny must contain exactly 182 tensors')
    names, result, gammas = set(), [], 0
    for key, value in source.items():
        name, gamma = target_key(key)
        require(name not in names and name in expected and name in converted, 'mapping collision or missing key: ' + key)
        names.add(name)
        actual = converted[name]
        check_tensor(value, actual, expected[name].shape, gamma)
        gammas += int(gamma)
        result.append({'source': key, 'target': name, 'source_shape': list(value.shape),
                       'target_shape': list(actual.shape), 'gamma_reshape': gamma,
                       'role': 'public1000_head_to_be_discarded' if key.startswith('head.fc.') else 'backbone',
                       'FP32_finite_exact': True})
    require(names == set(converted) == set(expected) and gammas == 18, 'mapping coverage/gamma count differs')
    return result


def normalization(model):
    import torch
    return [{'name': name, 'eps': float(layer.eps), 'normalized_shape': list(layer.normalized_shape)}
            for name, layer in model.named_modules() if isinstance(layer, torch.nn.LayerNorm)]


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def unchanged(identity):
    actual = file_identity(identity['path'])
    require(actual == {k: identity[k] for k in actual},
            'file stat changed after source-chain inspection: ' + identity['path'])


def accept(args, completed):
    context = cpu_context()
    chain = read_pinned_receipt(args.source_receipt, args.source_receipt_sha)
    require(bool(args.stage_binding) == bool(args.stage_binding_sha), 'stage binding path and SHA must be supplied together')
    stage = read_pinned_receipt(args.stage_binding, args.stage_binding_sha) if args.stage_binding else None
    current_assets = validate_execution_binding(chain, {'path': args.source_receipt, 'sha256': args.source_receipt_sha}, context, stage)
    snapshot = chain['runtime_snapshot']
    import torch
    import torchvision
    import timm
    from torchvision.models import convnext_tiny
    configure_torch_threads(context)
    require(not torch.cuda.is_initialized(), 'GPU already initialized')
    source_state = torch.load(current_assets['pytorch_model.bin']['path'], map_location='cpu', weights_only=True)
    converted = torch.load(current_assets['converted']['path'], map_location='cpu', weights_only=True)
    # pretrained=False and a supplied local config forbid implicit weight downloads.
    source_model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=False,
                                    pretrained_cfg=chain['public_pretrained_config'], drop_path_rate=.1)
    cfg = chain['public_pretrained_config']
    require(cfg['input_size'] == [3, 224, 224] and cfg['test_input_size'] == [3, 288, 288]
            and cfg['interpolation'] == 'bicubic' and cfg['crop_pct'] == .875
            and cfg['test_crop_pct'] == 1. and cfg['crop_mode'] == 'center',
            'pinned public inference preprocessing differs')
    source_loaded = source_model.load_state_dict(source_state, strict=True)
    full = convnext_tiny(weights=None, stochastic_depth_prob=.1)
    full.classifier[0].eps = chain['candidate']['final_norm_eps']
    mapping = check_mapping(source_state, converted, full.state_dict())
    target_loaded = full.load_state_dict(converted, strict=True)
    actual = full.state_dict()
    require(not source_loaded.missing_keys and not source_loaded.unexpected_keys
            and not target_loaded.missing_keys and not target_loaded.unexpected_keys,
            'strict load missing/unexpected keys')
    require(all(torch.equal(actual[k], converted[k]) for k in converted), 'loaded public tensor differs')
    completed['public_strict_mapping'] = {'source_tensors': len(source_state), 'target_tensors': len(converted),
                                         'source_missing': source_loaded.missing_keys,
                                         'source_unexpected': source_loaded.unexpected_keys,
                                         'target_missing': target_loaded.missing_keys,
                                         'target_unexpected': target_loaded.unexpected_keys,
                                         'all_mapped_values_FP32_finite_exact': True}
    source_norms, target_norms = normalization(source_model), normalization(full)
    require(len(source_norms) == len(target_norms) == 23
            and all(n['eps'] == chain['candidate']['internal_norm_eps'] for n in source_norms + target_norms),
            'actual source/target LN config differs')
    completed['source_layer_norms'] = source_norms
    completed['target_layer_norms'] = target_norms
    require(source_model.num_features == full.classifier[-1].in_features == 768
            and source_model.num_classes == full.classifier[-1].out_features == 1000
            and source_model.head.global_pool.pool_type == 'avg'
            and isinstance(full.avgpool, torch.nn.AdaptiveAvgPool2d) and full.avgpool.output_size == 1,
            'source/target native pool/features/classes differ')
    source_activations = [gelu_semantics(block.mlp.act)
                          for stage in source_model.stages for block in stage.blocks]
    target_activations = [gelu_semantics(block.block[4])
                          for index in (1, 3, 5, 7) for block in full.features[index]]
    source_gelu = [m['verified_approximate'] for m in source_activations]
    target_gelu = [m['verified_approximate'] for m in target_activations]
    require(source_gelu == target_gelu == ['none'] * 18, 'GELU configuration differs')
    convolutions = []
    target_modules = dict(full.named_modules())
    for name, module in source_model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        target_name = target_key(name + '.weight')[0].removesuffix('.weight')
        target = target_modules[target_name]
        fields = ('kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode')
        require(all(getattr(module, k) == getattr(target, k) for k in fields),
                'source/target non-tensor convolution config differs: ' + name)
        convolutions.append({'source': name, 'target': target_name,
                             **{k: getattr(module, k) for k in fields}})
    require(len(convolutions) == 22, 'Tiny convolution count differs')
    source_drop_path = [getattr(block.drop_path, 'drop_prob', 0.)
                        for stage in source_model.stages for block in stage.blocks]
    from torchvision.ops import StochasticDepth
    target_drop_path = [m.p for m in full.modules() if isinstance(m, StochasticDepth)]
    require(len(source_drop_path) == len(target_drop_path) == 18
            and all(abs(a - b) < 1e-8 for a, b in zip(source_drop_path, target_drop_path)),
            'source/target stochastic depth differs')
    require(sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in source_model.modules()) == 0
            and sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in full.modules()) == 0
            and not list(source_model.named_buffers()) and not list(full.named_buffers()),
            'unexpected BN/buffers')
    public_digest = tensor_digest(actual)
    # This demonstrates the allowed readout transition, not final runtime admission.
    torch.manual_seed(20261027)
    rng_before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    old_head = full.classifier[-1]
    new_head = torch.nn.Linear(old_head.in_features, 1)
    full.classifier = torch.nn.Sequential(*list(full.classifier.children())[:-1],
                                         torch.nn.Dropout(.2), new_head)
    binary_state = full.state_dict()
    backbone = [k for k in converted if not k.startswith('classifier.2.')]
    require(len(backbone) == 180 and all(torch.equal(binary_state[k], converted[k]) for k in backbone),
            'binary replacement changed public backbone')
    require(new_head.weight.data_ptr() != old_head.weight.data_ptr()
            and new_head.weight.shape == (1, 768) and new_head.bias.shape == (1,)
            and all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in new_head.parameters()),
            'fresh binary head identity differs')
    require(not torch.cuda.is_initialized(), 'GPU initialized during CPU acceptance')
    for identity in current_assets.values():
        unchanged(identity)
    map_path = Path(args.mapping_output)
    write_new_json(map_path, {'passed': True, 'revision': REVISION, 'mapping': mapping})
    return {'version': 'tiny22k-cpu-strict-load-v1', 'passed': True, 'execution': context,
            'source_chain_receipt': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
            'historical_source_execution': chain['execution'],
            'stage_binding': {'path': args.stage_binding, 'sha256': args.stage_binding_sha} if stage else None,
            'assets': current_assets,
            'content_verification': {'kind': 'fresh_pod_sha256', 'asset_count': len(current_assets),
                                     'bytes_hashed': sum(value['bytes_hashed'] for value in current_assets.values()),
                                     'metadata_postcheck_is_not_a_content_proof': True},
            'runtime_snapshot': snapshot,
            'source_factory': {'implementation': str(inspect.getsourcefile(type(source_model))),
                               'implementation_sha256': sha256(inspect.getsourcefile(type(source_model))),
                               'timm_version': timm.__version__, 'pretrained': False,
                               'missing_keys': source_loaded.missing_keys, 'unexpected_keys': source_loaded.unexpected_keys,
                               'layer_norms': source_norms, 'global_pool': 'avg', 'num_features': 768,
                               'stochastic_depth_probabilities': source_drop_path,
                               'GELU_modules': source_activations,
                               'GELU_approximate': source_gelu, 'batch_norm_modules': 0, 'buffers': 0},
            'target_factory': {'implementation': str(inspect.getsourcefile(type(full))),
                               'implementation_sha256': sha256(inspect.getsourcefile(type(full))),
                               'torchvision_version': torchvision.__version__, 'weights': None,
                               'missing_keys': target_loaded.missing_keys, 'unexpected_keys': target_loaded.unexpected_keys,
                               'layer_norms': target_norms, 'native_avgpool': 1, 'full_head_shape': [1000, 768],
                               'GELU_modules': target_activations,
                               'GELU_approximate': target_gelu, 'batch_norm_modules': 0, 'buffers': 0,
                               'stochastic_depth': [{'name': n, 'p': m.p, 'mode': m.mode}
                                                    for n, m in full.named_modules() if isinstance(m, StochasticDepth)]},
            'conversion': {'source_tensor_count': 182, 'converted_tensor_count': 182,
                           'backbone_tensor_count': 180, 'gamma_reshapes': 18, 'all_values_FP32_finite_exact': True,
                           'mapping_path': str(map_path), 'mapping_sha256': sha256(map_path),
                           'loaded_public_tensor_sha256': public_digest},
            'matching_non_tensor_convolution_config': convolutions,
            'readout_demonstration': {'scope': 'source_acceptance_only_not_candidate_runtime',
                                      'discarded_public_head_keys': ['classifier.2.weight', 'classifier.2.bias'],
                                      'backbone_tensors_preserved': 180, 'factory': 'torch.nn.Linear_native',
                                      'seed': 20261027, 'rng_sha256_before_new_head': rng_before,
                                      'head_shapes': [[1, 768], [1]], 'dropout': .2,
                                      'head_tensor_sha256': tensor_digest(new_head.state_dict()),
                                      'no_age_attached': True},
            'configuration_scope': {'source_norm_eps_observed': 1e-6, 'R92_final_norm_eps': 1e-12,
                                     'candidate_final_norm_eps': 1e-6, 'candidate_internal_norm_eps': 1e-6,
                                     'candidate_image_size': 512, 'source_train_size': cfg['input_size'],
                                     'source_test_size': cfg['test_input_size'], 'source_size_is_not_training_contract': True,
                                     'candidate_stochastic_depth_override': .1, 'candidate_binary_dropout_override': .2},
            'GPU_initialized': torch.cuda.is_initialized(), 'forward_calls': 0, 'datasets_read': False,
            'validation_source_sha256': {name: sha256(Path(__file__).with_name(name))
                                         for name in ('source_chain.py', 'strict_load.py', 'stage_binding.py')},
            'weight_files_written': False, 'weight_blob_hashes_recomputed': True,
            'candidate_runtime_admission': 'pending A fixed-source integration including age/optimizer/parameter groups',
            'pending': ['A candidate runtime strict source integration and fresh age/optimizer receipt',
                        'GitHub synchronization and exp-v3 formal asset/deployment identity']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'source-receipt-sha', 'mapping-output', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--stage-binding')
    parser.add_argument('--stage-binding-sha')
    args = parser.parse_args()
    completed = {}
    try:
        result = accept(args, completed)
    except Exception as error:
        write_new_json(args.output, {'passed': False, 'execution': cpu_context(),
                                     'error_type': type(error).__name__, 'error': str(error),
                                     'completed_observations': completed})
        raise
    write_new_json(args.output, result)


if __name__ == '__main__':
    main()

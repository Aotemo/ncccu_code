"""DINO v1 ResNet-50 backbone-only CPU extraction and strict TorchVision acceptance."""
import argparse
import json
from pathlib import Path

from source_chain import (configure_torch_threads, cpu_context, read_pinned_receipt,
                          require, sha256, verify_current_content, write_new_json)
from dino_resnet_source import REPO, TRAINING_DATA, WEIGHT_URL


TARGET_STATE_COUNT = 318
BN_COUNT = 53


def strip_prefixes(state):
    require(isinstance(state, dict) and state, 'state dict is empty')
    result, used, mapping = {}, [], {}
    for key, value in state.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in ('module.', 'backbone.'):
                if name.startswith(prefix):
                    if prefix not in used:
                        used.append(prefix)
                    name = name[len(prefix):]
                    changed = True
        require(name not in result, 'duplicate key after prefix stripping: ' + name)
        result[name] = value
        if name != key:
            mapping[key] = name
    return result, used, mapping


def partition_state(state, target_keys):
    unexpected = sorted(key for key in state if key not in target_keys)
    missing = sorted(key for key in target_keys if key not in state)
    derived = [key for key in missing if key.endswith('.num_batches_tracked')]
    blocking = [key for key in missing if key not in derived]
    return unexpected, derived, blocking


def positive_pair(value):
    if isinstance(value, (tuple, list)):
        require(len(value) == 2, 'pair must have two entries')
        pair = (int(value[0]), int(value[1]))
    else:
        require(isinstance(value, int) and value > 0, 'value must be a positive int or pair')
        pair = (value, value)
    require(pair[0] > 0 and pair[1] > 0, 'pair entries must be positive')
    return pair


def accept(args, completed):
    import torch
    import torchvision
    from torchvision.models.resnet import resnet50
    from torch import nn
    context = cpu_context()
    source = read_pinned_receipt(args.source_receipt, args.source_receipt_sha)
    weight_receipt = read_pinned_receipt(args.weight_receipt, args.weight_receipt_sha)
    require(source.get('passed') is True and source.get('repo') == REPO
            and weight_receipt.get('passed') is True and weight_receipt.get('repo') == REPO
            and source.get('revision') == weight_receipt.get('revision')
            and source.get('weight', {}).get('url') == WEIGHT_URL
            and weight_receipt.get('weight', {}).get('url') == WEIGHT_URL,
            'DINO source metadata/weight receipt identity differs')
    require(all(source['execution'][key] == context[key]
                for key in ('namespace', 'pod', 'uid', 'container', 'python'))
            and weight_receipt['execution']['uid'] == context['uid'],
            'DINO source execution identity differs')
    require(source['execution']['cpu_allocation']['path'] == context['cpu_allocation']['path']
            and source['execution']['cpu_allocation']['sha256'] == context['cpu_allocation']['sha256']
            and source['execution']['cpu_allocation']['executor_id'] == context['cpu_allocation']['executor_id'],
            'DINO CPU allocation changed; explicit new binding required')
    weight_record = weight_receipt['weight']
    weight_file = verify_current_content(weight_record['path'], weight_record['sha256'])
    require(weight_file['bytes'] == weight_record['bytes'], 'DINO weight byte count changed')
    configure_torch_threads(context)
    require(not torch.cuda.is_initialized(), 'GPU already initialized')
    raw = torch.load(weight_file['path'], map_location='cpu', weights_only=True)
    require(isinstance(raw, dict) and raw, 'checkpoint is not a nonempty dict')
    require(not any(key in raw and isinstance(raw[key], dict) for key in ('teacher', 'student', 'model')),
            'checkpoint is not the fixed plain backbone-only state: container keys present')
    state, prefixes_actual, prefix_key_map = strip_prefixes(raw)
    target = resnet50(weights=None)
    target.fc = nn.Identity()
    reference = target.state_dict()
    target_keys = set(reference)
    unexpected, derived, blocking = partition_state(state, target_keys)
    require(not unexpected and not blocking,
            'checkpoint is not the fixed plain backbone-only state: %r %r' % (unexpected[:8], blocking[:8]))
    completed['source_accounting'] = {'raw_type': type(raw).__name__,
                                      'prefixes_actual': prefixes_actual,
                                      'prefix_key_map': prefix_key_map,
                                      'keys_after_prefix_strip': len(state),
                                      'derived_counter_keys': derived,
                                      'unexpected_keys': unexpected, 'blocking_missing_keys': blocking}
    completed_state = dict(state)
    for key in derived:
        require(key not in completed_state, 'derived counter already present')
        completed_state[key] = torch.zeros((), dtype=torch.long)
    require(set(completed_state) == target_keys, 'completed source key set differs from target')
    present_counters = [key for key in target_keys if key.endswith('.num_batches_tracked')
                        and key not in derived]
    counter_values = [int(completed_state[key]) for key in sorted(present_counters)]
    for key, tensor in completed_state.items():
        require(isinstance(tensor, torch.Tensor), 'source value is not a tensor: ' + key)
        require(tensor.dtype == reference[key].dtype and tuple(tensor.shape) == tuple(reference[key].shape),
                'source dtype/shape differs from target state: ' + key)
        if tensor.dtype.is_floating_point:
            require(tensor.dtype == torch.float32, 'source float tensor is not FP32: ' + key)
            require(bool(torch.isfinite(tensor).all()), 'source tensor has nonfinite values: ' + key)
        else:
            require(key.endswith('.num_batches_tracked') and tensor.dtype == torch.long
                    and tuple(tensor.shape) == (),
                    'only int64 scalar num_batches_tracked may be non-floating: ' + key)
    loaded = target.load_state_dict(completed_state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, 'DINO target strict load differs')
    actual = target.state_dict()
    require(all(torch.equal(completed_state[key], actual[key]) for key in completed_state),
            'loaded DINO backbone differs from source tensors')
    batch_norms = [module for module in target.modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    require(len(batch_norms) == BN_COUNT and all(module.eps == 1e-5 and module.affine is True
                                                 and module.track_running_stats is True for module in batch_norms),
            'DINO target BatchNorm configuration differs')
    native_maxpool_stride = target.maxpool.stride
    require(positive_pair(target.conv1.stride) == (2, 2)
            and positive_pair(native_maxpool_stride) == (2, 2)
            and positive_pair(target.layer1[0].conv2.stride) == (1, 1)
            and target.layer1[0].downsample is not None
            and positive_pair(target.layer1[0].downsample[0].stride) == (1, 1)
            and all(positive_pair(layer[0].conv2.stride) == (2, 2) for layer in (target.layer2, target.layer3, target.layer4))
            and all(layer[0].downsample is not None and positive_pair(layer[0].downsample[0].stride) == (2, 2)
                    for layer in (target.layer2, target.layer3, target.layer4)),
            'DINO target stem/stage stride or downsample differs')
    native_output_size = (target.avgpool.output_size
                          if isinstance(target.avgpool, nn.AdaptiveAvgPool2d) else None)
    normalized_output_size = positive_pair(native_output_size) if native_output_size is not None else None
    require(isinstance(target.fc, nn.Identity) and target.layer4[-1].conv3.out_channels == 2048
            and isinstance(target.avgpool, nn.AdaptiveAvgPool2d)
            and normalized_output_size == (1, 1),
            'DINO target native head/features/GAP differs')
    public_parameters = sum(parameter.numel() for parameter in target.parameters())
    state_tensors = len(actual)
    parameter_tensors = sum(1 for name, _ in target.named_parameters())
    buffer_tensors = sum(1 for _ in target.named_buffers())
    structure = {'features': 2048, 'public_state_tensors': state_tensors,
                 'parameter_tensors': parameter_tensors, 'buffer_tensors': buffer_tensors,
                 'public_parameters': public_parameters, 'batch_norm_layers': BN_COUNT,
                 'batch_norm_eps': 1e-5, 'batch_norm_affine': True, 'fc': 'Identity',
                 'public_1000_head': False, 'derived_num_batches_tracked': len(derived),
                 'present_num_batches_tracked': len(present_counters),
                 'stem_stride': [2, 2], 'maxpool_stride': [2, 2], 'stage_strides': [1, 2, 2, 2],
                 'downsample_strides': [1, 2, 2, 2],
                 'maxpool_stride_native': (native_maxpool_stride if isinstance(native_maxpool_stride, int)
                                           else list(native_maxpool_stride)),
                 'maxpool_stride_normalized': list(positive_pair(native_maxpool_stride)),
                 'avgpool_output_size_native': (native_output_size if isinstance(native_output_size, int)
                                                else list(native_output_size)),
                 'avgpool_output_size_normalized': list(normalized_output_size)}
    require(state_tensors == TARGET_STATE_COUNT, 'DINO backbone state tensor count differs')
    converted_root = Path(args.assets) / 'converted'
    require(converted_root.is_absolute() and '/exp-v3/' in str(converted_root), 'exp-v3 converted root required')
    converted_root.mkdir(parents=True, exist_ok=True)
    converted_path = converted_root / 'dino-resnet50-backbone.pt'
    with converted_path.open('xb') as stream:
        torch.save(completed_state, stream)
    converted_sha = sha256(converted_path)
    reloaded = torch.load(converted_path, map_location='cpu', weights_only=True)
    require(set(reloaded) == target_keys and all(torch.equal(reloaded[key], actual[key]) for key in reloaded),
            'reloaded DINO backbone differs')
    mapping_path = Path(args.output).with_name('MAPPING.json')
    write_new_json(mapping_path, {'schema_version': 1, 'passed': True, 'repo': REPO,
                                 'revision': source['revision'],
                                 'mapping': [{'target': key,
                                              'shape': list(actual[key].shape),
                                              'dtype': str(actual[key].dtype),
                                              'role': ('derived_num_batches_tracked' if key in derived
                                                       else 'public_backbone'),
                                              'from_source_bytes': key in state}
                                             for key in sorted(actual)]})
    provenance = {'schema_version': 1, 'passed': True, 'model': 'resnet50',
                  'format': 'torchvision-resnet50-backbone-state-dict-v1', 'repo': REPO,
                  'revision': source['revision'], 'weight_url': WEIGHT_URL,
                  'training_data': TRAINING_DATA,
                  'original_sha256': weight_record['sha256'], 'sha256': converted_sha,
                  'final_norm_eps': None, 'internal_norm_eps': None,
                  'license_file': '../original/LICENSE', 'redistribution_confirmed': True,
                  'license_basis': source['license'],
                  'source_metadata': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
                  'source_weight': {'path': args.weight_receipt, 'sha256': args.weight_receipt_sha},
                  'source_file_hashes': {name: record['sha256'] for name, record in source['files'].items()},
                  'prefixes_actual': prefixes_actual,
                  'prefix_key_map': prefix_key_map,
                  'derived_num_batches_tracked': derived,
                  'public_1000_head': False,
                  'mapping': {'path': str(mapping_path), 'sha256': sha256(mapping_path)},
                  'structure': structure}
    provenance_path = converted_root / 'provenance.json'
    write_new_json(provenance_path, provenance)
    require(not torch.cuda.is_initialized(), 'GPU initialized during DINO acceptance')
    return {'schema_version': 1, 'passed': True, 'scope': 'dino_resnet50_backbone_cpu_strict',
            'execution': context, 'model': 'resnet50', 'repo': REPO, 'revision': source['revision'],
            'structure': structure, 'assets': {'weight': weight_file},
            'source_metadata': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
            'source_weight': {'path': args.weight_receipt, 'sha256': args.weight_receipt_sha},
            'converted': {'path': str(converted_path), 'sha256': converted_sha},
            'provenance': {'path': str(provenance_path), 'sha256': sha256(provenance_path)},
            'mapping': {'path': str(mapping_path), 'sha256': sha256(mapping_path)},
            'strict_load': {'missing': loaded.missing_keys, 'unexpected': loaded.unexpected_keys,
                            'state_tensors': state_tensors, 'all_values_exact': True},
            'prefixes_actual': prefixes_actual,
            'prefix_key_map': prefix_key_map,
            'derived_num_batches_tracked': derived,
            'num_batches_tracked_observed': {'count': len(counter_values),
                                             'min': min(counter_values) if counter_values else None,
                                             'max': max(counter_values) if counter_values else None,
                                             'zero_count': sum(value == 0 for value in counter_values)},
            'target_factory': {'name': 'torchvision.models.resnet.resnet50', 'weights': None,
                               'fc': 'Identity', 'torchvision_version': torchvision.__version__,
                               'batch_norm_eps': 1e-5},
            'mapping_notes': {'fc': 'plain backbone only; fc/head/container keys are a format mismatch',
                              'counter_policy': 'missing num_batches_tracked derived as int64 zero; recorded, not public'},
            'forward_calls': 0, 'datasets_read': False,
            'GPU_initialized_before': False, 'GPU_initialized_after': torch.cuda.is_initialized(),
            'weight_blob_hashes_recomputed': True,
            'validation_source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                         ('dino_resnet_source.py', 'dino_resnet_strict.py', 'source_chain.py')},
            'production_512_head_age_bn_ema_admission': 'pending A affected formal acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'source-receipt-sha', 'weight-receipt', 'weight-receipt-sha',
                 'assets', 'output', 'handoff'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    completed = {}
    try:
        result = accept(args, completed)
    except Exception as error:
        write_new_json(args.output, {'schema_version': 1, 'passed': False, 'execution': cpu_context(),
                                     'error_type': type(error).__name__, 'error': str(error),
                                     'completed_observations': completed})
        raise
    write_new_json(args.output, result)
    handoff = {key: result[key] for key in ('schema_version', 'passed', 'model', 'repo', 'revision',
                                           'execution', 'converted', 'provenance', 'structure',
                                           'source_metadata', 'source_weight')}
    handoff['strict_load'] = {'path': args.output, 'sha256': sha256(args.output)}
    write_new_json(args.handoff, handoff)
    print(json.dumps({'handoff': {'path': args.handoff, 'sha256': sha256(args.handoff)},
                      'strict_load': handoff['strict_load'], 'structure': handoff['structure'],
                      'converted': handoff['converted'], 'provenance': handoff['provenance']}, indent=2))


if __name__ == '__main__':
    main()

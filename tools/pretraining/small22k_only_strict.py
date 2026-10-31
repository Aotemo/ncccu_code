"""Small 22K-only full-public-head CPU strict acceptance; never run a forward."""
import argparse
import inspect
import json
from pathlib import Path

from source_chain import (configure_torch_threads, cpu_context, file_identity, read_pinned_receipt,
                          require, sha256, verify_current_content, write_new_json)
from strict_load import check_tensor, gelu_semantics, normalization, target_key, tensor_digest
from small22k_only_source import FORMAT, MODEL, REPO, TAG, TRAINING_DATA


DEPTHS = (3, 3, 27, 3)
MODEL_TAG = MODEL + '.' + TAG


def config_num_classes(config):
    value = config.get('num_classes') if isinstance(config, dict) else None
    cfg = config.get('pretrained_cfg') if isinstance(config, dict) else None
    require(type(value) is int and value > 0 and isinstance(cfg, dict)
            and type(cfg.get('num_classes')) is int and cfg['num_classes'] == value,
            'config/pretrained_cfg num_classes must be the same positive integer')
    return value


def small22k_mapping(source, converted, expected, num_classes):
    require(len(source) == len(converted) == len(expected) == 344,
            'Small full-public tensor count differs')
    names, rows, gammas = set(), [], 0
    for key, value in sorted(source.items()):
        name, gamma = target_key(key, DEPTHS)
        require(name not in names and name in expected and name in converted,
                'Small mapping missing/collision: ' + key)
        names.add(name)
        check_tensor(value, converted[name], expected[name].shape, gamma)
        gammas += int(gamma)
        rows.append({'source': key, 'target': name, 'source_shape': list(value.shape),
                     'target_shape': list(converted[name].shape), 'gamma_reshape': gamma,
                     'role': ('public_K_head_to_be_discarded' if key.startswith('head.fc.')
                              else 'backbone'),
                     'FP32_finite_exact': True})
    require(names == set(expected) == set(converted) and gammas == 36,
            'Small mapping coverage/gamma count differs')
    require({'head.fc.weight', 'head.fc.bias'} <= set(source)
            and tuple(source['head.fc.weight'].shape) == (num_classes, 768)
            and tuple(source['head.fc.bias'].shape) == (num_classes,)
            and {'classifier.0.weight', 'classifier.0.bias'} <= names,
            'public K head / final norm mapping differs')
    return rows


def accept(args, completed):
    context = cpu_context()
    chain = read_pinned_receipt(args.source_receipt, args.source_receipt_sha)
    require(chain.get('passed') is True and chain.get('repo') == REPO
            and chain.get('model') == MODEL and chain.get('tag') == TAG
            and chain.get('format') == FORMAT and chain.get('training_data') == TRAINING_DATA
            and chain.get('registry', {}).get('id') == REPO
            and (chain.get('registry') or {}).get('private') is False
            and (chain.get('registry') or {}).get('gated') is False
            and (chain.get('registry') or {}).get('license') == 'apache-2.0',
            'Small 22K-only source receipt identity differs')
    require(isinstance(chain.get('revision'), str)
            and len(chain['revision']) == 40
            and all(character in '0123456789abcdef' for character in chain['revision']),
            'source receipt revision differs')
    num_classes = chain.get('num_classes')
    require(type(num_classes) is int and num_classes > 0, 'source receipt K differs')
    require(all(chain['execution'][key] == context[key]
                for key in ('namespace', 'pod', 'uid', 'container', 'python')),
            'source execution identity differs')
    require(chain['execution']['cpu_allocation']['path'] == context['cpu_allocation']['path']
            and chain['execution']['cpu_allocation']['sha256'] == context['cpu_allocation']['sha256']
            and chain['execution']['cpu_allocation']['executor_id'] == context['cpu_allocation']['executor_id'],
            'CPU allocation changed; explicit new binding required')
    assets = {name: verify_current_content(record['path'], record['sha256'])
              for name, record in chain['files'].items()}
    cfg = json.loads(Path(assets['config.json']['path']).read_text())
    require(config_num_classes(cfg) == num_classes and cfg.get('architecture') == MODEL
            and cfg.get('num_features') == 768
            and cfg['pretrained_cfg'].get('tag') == TAG,
            'actual config K/architecture/tag differs')
    actual_preprocessing = {key: cfg['pretrained_cfg'].get(key) for key in
                            ('mean', 'std', 'input_size', 'test_input_size', 'interpolation',
                             'crop_pct', 'test_crop_pct', 'crop_mode')}
    require(chain.get('preprocessing') == actual_preprocessing,
            'source receipt preprocessing differs from actual config')
    import torch
    import torchvision
    import timm
    from torchvision.models import convnext_small
    from torchvision.ops import StochasticDepth
    configure_torch_threads(context)
    gpu_before = torch.cuda.is_initialized()
    require(not gpu_before, 'GPU already initialized')
    source_path = assets['pytorch_model.bin']['path']
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    require(isinstance(source, dict), 'bin is not a plain state dict')
    source_model = timm.create_model(MODEL_TAG, pretrained=False,
                                     pretrained_cfg=cfg['pretrained_cfg'],
                                     drop_path_rate=.1, num_classes=num_classes)
    target = convnext_small(weights=None, stochastic_depth_prob=.1)
    target.classifier[-1] = torch.nn.Linear(768, num_classes)
    source_loaded = source_model.load_state_dict(source, strict=True)
    converted = {}
    for key, value in source.items():
        name, gamma = target_key(key, DEPTHS)
        require(name not in converted, 'Small mapping collision')
        converted[name] = value.reshape(-1, 1, 1).contiguous() if gamma else value
    rows = small22k_mapping(source, converted, target.state_dict(), num_classes)
    converted_root = Path(args.assets) / 'converted'
    require(converted_root.is_absolute() and '/exp-v3/' in str(converted_root),
            'exp-v3 converted root required')
    converted_root.mkdir(parents=True, exist_ok=True)
    converted_path = converted_root / ('convnext-small-tv-in22konly-' + chain['revision'][:8] + '.pt')
    with converted_path.open('xb') as stream:
        torch.save(converted, stream)
    converted_sha = sha256(converted_path)
    reloaded = torch.load(converted_path, map_location='cpu', weights_only=True)
    small22k_mapping(source, reloaded, target.state_dict(), num_classes)
    target_loaded = target.load_state_dict(reloaded, strict=True)
    require(not source_loaded.missing_keys and not source_loaded.unexpected_keys
            and not target_loaded.missing_keys and not target_loaded.unexpected_keys,
            'Small strict missing/unexpected keys')
    loaded_state = target.state_dict()
    require(all(torch.equal(value, loaded_state[key]) for key, value in reloaded.items()),
            'loaded target differs')
    completed['strict'] = {'source_missing': source_loaded.missing_keys,
                           'source_unexpected': source_loaded.unexpected_keys,
                           'target_missing': target_loaded.missing_keys,
                           'target_unexpected': target_loaded.unexpected_keys,
                           'public_state_tensors': len(reloaded),
                           'all_FP32_finite_exact': True}
    source_norms, target_norms = normalization(source_model), normalization(target)
    require(len(source_norms) == len(target_norms) == 41
            and all(row['eps'] == 1e-6 for row in source_norms + target_norms),
            'Small LN differs')
    source_activations = [gelu_semantics(block.mlp.act)
                          for stage in source_model.stages for block in stage.blocks]
    target_activations = [gelu_semantics(block.block[4])
                          for index in (1, 3, 5, 7) for block in target.features[index]]
    require(len(source_activations) == len(target_activations) == 36, 'Small GELU count differs')
    source_drop = [float(getattr(block.drop_path, 'drop_prob', 0.))
                   for stage in source_model.stages for block in stage.blocks]
    target_drop = [float(module.p) for module in target.modules()
                   if isinstance(module, StochasticDepth)]
    require(len(source_drop) == len(target_drop) == 36
            and abs(target_drop[-1] - .1) < 1e-8
            and all(abs(a - b) < 1e-8 for a, b in zip(source_drop, target_drop)),
            'Small SD config differs')
    require(source_model.num_features == target.classifier[-1].in_features == 768
            and source_model.num_classes == target.classifier[-1].out_features == num_classes
            and source_model.head.global_pool.pool_type == 'avg'
            and isinstance(target.avgpool, torch.nn.AdaptiveAvgPool2d)
            and (target.avgpool.output_size == 1
                 or (isinstance(target.avgpool.output_size, (tuple, list))
                     and tuple(target.avgpool.output_size) == (1, 1))),
            'Small pool/features/head differs')
    require(not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                    for model in (source_model, target) for module in model.modules())
            and not list(source_model.named_buffers()) and not list(target.named_buffers()),
            'Small BN/buffers present')
    target_modules = dict(target.named_modules())
    convolution_config = []
    for name, module in source_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            destination = target_key(name + '.weight', DEPTHS)[0].removesuffix('.weight')
            fields = ('kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode')
            require(all(getattr(module, key) == getattr(target_modules[destination], key)
                        for key in fields), 'Small convolution non-tensor config differs: ' + name)
            convolution_config.append({'source': name, 'target': destination,
                                       **{key: getattr(module, key) for key in fields}})
    require(len(convolution_config) == 40, 'Small convolution count differs')
    public_parameters = sum(parameter.numel() for parameter in target.parameters())
    backbone_parameters = sum(parameter.numel() for name, parameter in target.named_parameters()
                              if not name.startswith('classifier.2.'))
    backbone_tensors = sum(not key.startswith('classifier.2.') for key in reloaded)
    require(public_parameters == backbone_parameters + num_classes * (768 + 1)
            and backbone_tensors == 342 and len(reloaded) == 344,
            'Small parameter/tensor accounting differs')
    structure = {'features': 768, 'num_classes': num_classes, 'public_state_tensors': len(reloaded),
                 'backbone_state_tensors': backbone_tensors, 'head_state_tensors': 2,
                 'backbone_parameters': backbone_parameters,
                 'public_parameters': public_parameters, 'full_head_shape': [num_classes, 768],
                 'layer_norms': len(target_norms), 'gelu_modules': len(target_activations),
                 'stochastic_depth_modules': len(target_drop), 'convolutions': len(convolution_config)}
    mapping_path = Path(args.output).with_name('MAPPING.json')
    write_new_json(mapping_path, {'schema_version': 1, 'passed': True, 'repo': REPO,
                                 'tag': TAG, 'revision': chain['revision'],
                                 'num_classes': num_classes, 'mapping': rows})
    provenance = {'schema_version': 1, 'passed': True, 'model': MODEL, 'format': FORMAT,
                  'repo': REPO, 'tag': TAG, 'revision': chain['revision'],
                  'source_url': f"https://huggingface.co/{REPO}/resolve/{chain['revision']}/pytorch_model.bin",
                  'training_data': TRAINING_DATA, 'num_classes': num_classes,
                  'original_sha256': assets['pytorch_model.bin']['sha256'],
                  'sha256': converted_sha, 'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6,
                  'license_file': '../original/LICENSE-apache-2.0.txt',
                  'redistribution_confirmed': True, 'license_basis': chain['license'],
                  'source_file_hashes': {name: record['sha256'] for name, record in assets.items()},
                  'source_chain': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
                  'mapping': {'path': str(mapping_path), 'sha256': sha256(mapping_path)},
                  'structure': structure,
                  'public_head_disposition': {
                      'full_K_head_verified': True,
                      'production_discards': ['head.fc.weight', 'head.fc.bias'],
                      'mapped_target': ['classifier.2.weight', 'classifier.2.bias'],
                      'enters_training_or_inference': False}}
    provenance_path = converted_root / 'provenance.json'
    write_new_json(provenance_path, provenance)
    implementation_pins = {}
    for label, obj in (('timm_convnext', type(source_model)), ('torchvision_convnext', type(target)),
                       ('timm_GELU', type(source_model.stages[0].blocks[0].mlp.act)),
                       ('torch_GELU', torch.nn.GELU)):
        path = inspect.getsourcefile(obj)
        implementation_pins[label] = {'path': path, 'sha256': sha256(path)}
    for identity in assets.values():
        require(file_identity(identity['path']) == {key: identity[key]
                                                    for key in file_identity(identity['path'])},
                'asset metadata changed during acceptance')
    require(not torch.cuda.is_initialized(), 'GPU initialized during Small acceptance')
    return {'schema_version': 1, 'passed': True,
            'scope': 'small22k_only_full_public_head_cpu_strict', 'execution': context,
            'model': MODEL, 'repo': REPO, 'tag': TAG, 'revision': chain['revision'],
            'num_classes': num_classes, 'structure': structure,
            'public_parameters': public_parameters, 'assets': assets,
            'original': {'path': source_path, 'sha256': assets['pytorch_model.bin']['sha256']},
            'converted': {'path': str(converted_path), 'sha256': converted_sha},
            'provenance': {'path': str(provenance_path), 'sha256': sha256(provenance_path)},
            'source_chain': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
            'mapping': {'path': str(mapping_path), 'sha256': sha256(mapping_path)},
            'strict_load': completed['strict'], 'source_implementation_pins': implementation_pins,
            'source_factory': {'name': 'timm.create_model', 'model': MODEL_TAG, 'pretrained': False,
                               'drop_path_rate': .1, 'num_classes': num_classes,
                               'timm_version': timm.__version__, 'layer_norms': source_norms,
                               'GELU': source_activations, 'stochastic_depth': source_drop,
                               'BN': 0, 'buffers': 0},
            'target_factory': {'name': 'torchvision.models.convnext_small', 'weights': None,
                               'stochastic_depth_prob': .1, 'replaced_head_classes': num_classes,
                               'torchvision_version': torchvision.__version__,
                               'layer_norms': target_norms, 'GELU': target_activations,
                               'stochastic_depth': target_drop, 'BN': 0, 'buffers': 0},
            'matching_convolution_config': convolution_config,
            'source_preprocessing': chain.get('preprocessing'),
            'loaded_public_tensor_sha256': tensor_digest(reloaded),
            'source_card_preprocessing_is_not_candidate_preprocessing': True,
            'public_head_disposition': provenance['public_head_disposition'],
            'candidate_overrides': chain['candidate'], 'forward_calls': 0, 'datasets_read': False,
            'GPU_initialized_before': gpu_before,
            'GPU_initialized_after': torch.cuda.is_initialized(),
            'weight_blob_hashes_recomputed': True,
            'validation_source_sha256': {name: sha256(Path(__file__).with_name(name))
                                         for name in ('small22k_only_source.py',
                                                      'small22k_only_strict.py',
                                                      'strict_load.py', 'source_chain.py')},
            'production_binary_age_optimizer_admission': 'pending A, no binary demonstration here'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'source-receipt-sha', 'assets', 'output', 'handoff'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    completed = {}
    try:
        result = accept(args, completed)
    except Exception as error:
        write_new_json(args.output, {'schema_version': 1, 'passed': False,
                                     'execution': cpu_context(),
                                     'error_type': type(error).__name__, 'error': str(error),
                                     'completed_observations': completed})
        raise
    write_new_json(args.output, result)
    handoff = {key: result[key] for key in ('schema_version', 'passed', 'model', 'repo', 'tag',
                                            'revision', 'num_classes', 'execution', 'original',
                                            'converted', 'provenance', 'structure', 'source_chain')}
    handoff['strict_load'] = {'path': args.output, 'sha256': sha256(args.output)}
    write_new_json(args.handoff, handoff)
    print(json.dumps({'handoff': {'path': args.handoff, 'sha256': sha256(args.handoff)},
                      'strict_load': handoff['strict_load'], 'structure': handoff['structure'],
                      'original': handoff['original'], 'converted': handoff['converted'],
                      'provenance': handoff['provenance']}, indent=2))


if __name__ == '__main__':
    main()

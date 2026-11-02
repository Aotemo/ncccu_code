"""Base22K full1000 CPU conversion/strict acceptance; never run a forward."""
import argparse
import inspect
import json
from pathlib import Path

from source_chain import (configure_torch_threads, cpu_context, file_identity, read_pinned_receipt,
                          require, sha256, verify_current_content, write_new_json)
from strict_load import check_tensor, gelu_semantics, normalization, target_key, tensor_digest
from base_source import MODEL, REPO, TRAINING_DATA, SOURCE_PROFILES


DEPTHS = (3, 3, 27, 3)


def base_mapping(source, converted, expected):
    require(len(source) == len(converted) == len(expected) == 344, 'Base full1000 tensor count differs')
    names, rows = set(), []
    for key, value in sorted(source.items()):
        name, gamma = target_key(key, DEPTHS)
        require(name not in names and name in expected and name in converted,
                'Base mapping missing/collision: ' + key)
        names.add(name)
        check_tensor(value, converted[name], expected[name].shape, gamma)
        rows.append({'source': key, 'target': name, 'source_shape': list(value.shape),
                     'target_shape': list(converted[name].shape), 'gamma_reshape': gamma,
                     'role': 'public1000_head_to_be_discarded' if key.startswith('head.fc.') else 'backbone',
                     'FP32_finite_exact': True})
    require(names == set(expected) == set(converted)
            and sum(row['gamma_reshape'] for row in rows) == 36, 'Base mapping coverage differs')
    return rows


def accept(args, completed):
    context = cpu_context()
    profile = SOURCE_PROFILES[getattr(args, 'source_profile', 'base224')]
    repo, training_data = profile['repo'], profile['training_data']
    model_name, features = profile.get('model', MODEL), profile.get('features', 1024)
    source_model_name = repo.split('/', 1)[1]
    chain = read_pinned_receipt(args.source_receipt, args.source_receipt_sha)
    require(chain.get('passed') is True and chain.get('repo') == repo and chain.get('model') == model_name
            and chain.get('training_data') == training_data,
            'Base source receipt identity differs')
    revision = chain.get('revision')
    require(isinstance(revision, str) and len(revision) == 40
            and all(character in '0123456789abcdef' for character in revision),
            'Base source receipt lacks one immutable commit')
    require(profile.get('revision', revision) == revision, 'fixed Base profile revision differs')
    require(all(chain['execution'][key] == context[key]
                for key in ('namespace', 'pod', 'uid', 'container', 'python')),
            'Base source execution identity differs')
    require(chain['execution']['cpu_allocation']['path'] == context['cpu_allocation']['path']
            and chain['execution']['cpu_allocation']['sha256'] == context['cpu_allocation']['sha256']
            and chain['execution']['cpu_allocation']['executor_id'] == context['cpu_allocation']['executor_id'],
            'Base CPU allocation changed; explicit new binding required')
    assets = {name: verify_current_content(record['path'], record['sha256'])
              for name, record in chain['files'].items()}
    cfg = json.loads(Path(assets['config.json']['path']).read_text())
    require(cfg['architecture'] == model_name and cfg['num_classes'] == 1000 and cfg['num_features'] == features,
            'actual Base config differs')
    import torch
    import torchvision
    import timm
    from torchvision import models as torchvision_models
    from torchvision.ops import StochasticDepth
    configure_torch_threads(context)
    gpu_before = torch.cuda.is_initialized()
    require(not gpu_before, 'GPU already initialized')
    source_path = assets['pytorch_model.bin']['path']
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    require(isinstance(source, dict), 'bin is not a plain state dict')
    # Both factories receive explicit SD .1. No pretrained/network factory load.
    source_model = timm.create_model(source_model_name, pretrained=False,
                                    pretrained_cfg=cfg['pretrained_cfg'], drop_path_rate=.1)
    target = getattr(torchvision_models, model_name)(weights=None, stochastic_depth_prob=.1)
    source_loaded = source_model.load_state_dict(source, strict=True)
    converted = {}
    for key, value in source.items():
        name, gamma = target_key(key, DEPTHS)
        require(name not in converted, 'Base mapping collision')
        converted[name] = value.reshape(-1, 1, 1).contiguous() if gamma else value
    rows = base_mapping(source, converted, target.state_dict())
    converted_root = Path(args.assets) / 'converted'
    require(converted_root.is_absolute() and '/exp-v3/' in str(converted_root), 'exp-v3 converted root required')
    converted_root.mkdir(parents=True, exist_ok=True)
    converted_path = converted_root / f'{model_name.replace("_", "-")}-tv-{revision[:8]}.pt'
    with converted_path.open('xb') as stream:
        torch.save(converted, stream)
    converted_sha = sha256(converted_path)
    reloaded = torch.load(converted_path, map_location='cpu', weights_only=True)
    base_mapping(source, reloaded, target.state_dict())
    target_loaded = target.load_state_dict(reloaded, strict=True)
    require(not source_loaded.missing_keys and not source_loaded.unexpected_keys
            and not target_loaded.missing_keys and not target_loaded.unexpected_keys,
            'Base strict missing/unexpected keys')
    loaded_state = target.state_dict()
    require(all(torch.equal(value, loaded_state[key]) for key, value in reloaded.items()),
            'loaded target differs')
    completed['strict'] = {'source_missing': source_loaded.missing_keys, 'source_unexpected': source_loaded.unexpected_keys,
                           'target_missing': target_loaded.missing_keys, 'target_unexpected': target_loaded.unexpected_keys,
                           'public_state_tensors': len(reloaded), 'all_FP32_finite_exact': True}
    source_norms, target_norms = normalization(source_model), normalization(target)
    require(len(source_norms) == len(target_norms) == 41
            and all(row['eps'] == 1e-6 for row in source_norms + target_norms), 'Base LN differs')
    source_activations = [gelu_semantics(block.mlp.act)
                          for stage in source_model.stages for block in stage.blocks]
    target_activations = [gelu_semantics(block.block[4])
                          for index in (1, 3, 5, 7) for block in target.features[index]]
    require(len(source_activations) == len(target_activations) == 36, 'Base GELU count differs')
    source_drop = [float(getattr(block.drop_path, 'drop_prob', 0.))
                   for stage in source_model.stages for block in stage.blocks]
    target_drop = [float(module.p) for module in target.modules() if isinstance(module, StochasticDepth)]
    require(len(source_drop) == len(target_drop) == 36 and abs(target_drop[-1] - .1) < 1e-8
            and all(abs(a - b) < 1e-8 for a, b in zip(source_drop, target_drop)), 'Base SD config differs')
    require(source_model.num_features == target.classifier[-1].in_features == features
            and source_model.num_classes == target.classifier[-1].out_features == 1000
            and source_model.head.global_pool.pool_type == 'avg'
            and isinstance(target.avgpool, torch.nn.AdaptiveAvgPool2d) and target.avgpool.output_size == 1,
            'Base pool/features/head differs')
    if getattr(args, 'source_profile', 'base224') in ('baseclip384', 'large224'):
        require(isinstance(source_model.norm_pre, torch.nn.Identity)
                and isinstance(source_model.head.pre_logits, torch.nn.Identity),
                'CLIP Base final norm/head placement differs from native TorchVision')
    widths = [target.features[0][0].out_channels, target.features[2][1].out_channels,
              target.features[4][1].out_channels, target.features[6][1].out_channels]
    require(widths == profile.get('widths', [128,256,512,1024]), 'native ConvNeXt stage widths differ')
    require(not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                    for model in (source_model, target) for m in model.modules())
            and not list(source_model.named_buffers()) and not list(target.named_buffers()),
            'Base BN/buffers present')
    target_modules = dict(target.named_modules())
    convolution_config = []
    for name, module in source_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            dest = target_key(name + '.weight', DEPTHS)[0].removesuffix('.weight')
            fields = ('kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode')
            require(all(getattr(module, k) == getattr(target_modules[dest], k) for k in fields),
                    'Base convolution non-tensor config differs: ' + name)
            convolution_config.append({'source': name, 'target': dest,
                                       **{k: getattr(module, k) for k in fields}})
    require(len(convolution_config) == 40, 'Base convolution count differs')
    public_parameters = sum(p.numel() for p in target.parameters())
    backbone_parameters = sum(p.numel() for n, p in target.named_parameters() if not n.startswith('classifier.2.'))
    backbone_tensors = sum(not key.startswith('classifier.2.') for key in reloaded)
    structure = {'features': target.classifier[-1].in_features, 'public_state_tensors': len(reloaded),
                 'backbone_state_tensors': backbone_tensors, 'backbone_parameters': backbone_parameters,
                 'public_parameters': public_parameters,
                 'layer_norms': len(target_norms), 'gelu_modules': len(target_activations),
                 'stochastic_depth_modules': len(target_drop), 'full_head_shape': list(target.classifier[-1].weight.shape)}
    require(backbone_tensors == 342 and len(reloaded) == 344, 'Base converted backbone/state count differs')
    mapping_path = Path(args.output).with_name('MAPPING.json')
    write_new_json(mapping_path, {'schema_version': 1, 'passed': True, 'repo': repo,
                                 'revision': revision, 'mapping': rows})
    provenance = {'schema_version': 1, 'passed': True, 'model': model_name, 'format': 'torchvision-1000-state-dict-v1', 'repo': repo,
                  'revision': revision, 'source_url': f'https://huggingface.co/{repo}/resolve/{revision}/pytorch_model.bin',
                  'training_data': training_data,
                  'original_sha256': assets['pytorch_model.bin']['sha256'], 'sha256': converted_sha,
                  'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6,
                  'license_file': '../original/LICENSE-apache-2.0.txt', 'redistribution_confirmed': True,
                  'license_basis': chain['license'],
                  'source_file_hashes': {k: v['sha256'] for k, v in assets.items()},
                  'source_chain': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
                  'mapping': {'path': str(mapping_path), 'sha256': sha256(mapping_path)},
                  'structure': structure}
    provenance_path = converted_root / 'provenance.json'
    write_new_json(provenance_path, provenance)
    implementation_pins = {}
    for label, obj in (('timm_convnext', type(source_model)), ('torchvision_convnext', type(target)),
                       ('timm_GELU', type(source_model.stages[0].blocks[0].mlp.act)),
                       ('torch_GELU', torch.nn.GELU)):
        path = inspect.getsourcefile(obj)
        implementation_pins[label] = {'path': path, 'sha256': sha256(path)}
    for identity in assets.values():
        require(file_identity(identity['path']) == {k: identity[k] for k in file_identity(identity['path'])},
                'asset metadata changed during acceptance')
    require(not torch.cuda.is_initialized(), 'GPU initialized during Base acceptance')
    return {'schema_version': 1, 'passed': True,
            'scope': ('large22k_full1000_cpu_strict' if model_name == 'convnext_large' else
                      'base22k_full1000_cpu_strict' if getattr(args, 'source_profile', 'base224') == 'base224' else 'baseclip_full1000_cpu_strict'),
            'execution': context,
            'model': model_name, 'repo': repo, 'revision': revision, 'structure': structure,
            'public_parameters': public_parameters, 'assets': assets,
            'original': {'path': source_path, 'sha256': assets['pytorch_model.bin']['sha256']},
            'converted': {'path': str(converted_path), 'sha256': converted_sha},
            'provenance': {'path': str(provenance_path), 'sha256': sha256(provenance_path)},
            'source_chain': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
            'mapping': {'path': str(mapping_path), 'sha256': sha256(mapping_path)},
            'strict_load': completed['strict'], 'source_implementation_pins': implementation_pins,
            'source_factory': {'name': 'timm.create_model', 'model': source_model_name,
                               'pretrained': False, 'drop_path_rate': .1, 'timm_version': timm.__version__,
                               'layer_norms': source_norms, 'GELU': source_activations,
                               'stochastic_depth': source_drop, 'BN': 0, 'buffers': 0},
            'target_factory': {'name': 'torchvision.models.' + model_name, 'weights': None,
                               'stochastic_depth_prob': .1, 'torchvision_version': torchvision.__version__,
                               'layer_norms': target_norms, 'GELU': target_activations,
                               'stochastic_depth': target_drop, 'BN': 0, 'buffers': 0},
            'matching_convolution_config': convolution_config,
            'loaded_public_tensor_sha256': tensor_digest(reloaded),
            'source_card_preprocessing_is_not_candidate_preprocessing': True,
            'candidate_overrides': chain['candidate'], 'forward_calls': 0, 'datasets_read': False,
            'GPU_initialized_before': gpu_before, 'GPU_initialized_after': torch.cuda.is_initialized(),
            'weight_blob_hashes_recomputed': True,
            'validation_source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                         ('base_source.py', 'base_strict.py', 'strict_load.py', 'source_chain.py')},
            'production_binary_age_optimizer_admission': 'pending A, no binary demonstration here'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'source-receipt-sha', 'assets', 'output', 'handoff'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-profile', choices=tuple(SOURCE_PROFILES), default='base224')
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
                                           'execution', 'original', 'converted', 'provenance', 'structure', 'source_chain')}
    handoff['strict_load'] = {'path': args.output, 'sha256': sha256(args.output)}
    write_new_json(args.handoff, handoff)
    print(json.dumps({'handoff': {'path': args.handoff, 'sha256': sha256(args.handoff)},
                      'strict_load': handoff['strict_load'], 'structure': handoff['structure'],
                      'original': handoff['original'], 'converted': handoff['converted'],
                      'provenance': handoff['provenance']}, indent=2))


if __name__ == '__main__':
    main()

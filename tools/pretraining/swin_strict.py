"""Swin-B 22K full1000 CPU conversion/strict acceptance; never run a forward on GPU."""
import argparse
from functools import partial
import inspect
import json
from pathlib import Path
import re

from source_chain import (configure_torch_threads, cpu_context, file_identity, read_pinned_receipt,
                          require, sha256, verify_current_content, write_new_json)
from strict_load import check_tensor, normalization, tensor_digest
from swin_source import MODEL, REPO, TRAINING_DATA


DEPTHS = (2, 2, 18, 2)
HEADS = (4, 8, 16, 32)
WINDOW = 7
EMBED = 128
STATIC_INDEX_COUNT = 24
SOURCE_MASK_COUNT = 11
SOURCE_TENSOR_COUNT = 329
TARGET_STATE_COUNT = 353


def swin_target_key(key):
    prefixes = {'patch_embed.proj': 'features.0.0', 'patch_embed.norm': 'features.0.2',
                'norm': 'norm', 'head.fc': 'head'}
    for prefix, target in prefixes.items():
        if key in (prefix + '.weight', prefix + '.bias'):
            return target + key[len(prefix):]
    match = re.fullmatch(r'layers\.([0-3])\.blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)', key)
    if match:
        stage, block = int(match[1]), int(match[2])
        require(block < DEPTHS[stage], 'block outside configured Swin depth')
        return f'features.{2 * stage + 1}.{block}.{match[3]}.{match[4]}'
    match = re.fullmatch(r'layers\.([0-3])\.blocks\.(\d+)\.attn\.(qkv|proj)\.(weight|bias)', key)
    if match:
        stage, block = int(match[1]), int(match[2])
        require(block < DEPTHS[stage], 'attention block outside configured Swin depth')
        return f'features.{2 * stage + 1}.{block}.attn.{match[3]}.{match[4]}'
    match = re.fullmatch(r'layers\.([0-3])\.blocks\.(\d+)\.attn\.relative_position_bias_table', key)
    if match:
        stage, block = int(match[1]), int(match[2])
        require(block < DEPTHS[stage], 'relative bias block outside configured Swin depth')
        return f'features.{2 * stage + 1}.{block}.attn.relative_position_bias_table'
    match = re.fullmatch(r'layers\.([0-3])\.blocks\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)', key)
    if match:
        stage, block = int(match[1]), int(match[2])
        require(block < DEPTHS[stage], 'MLP block outside configured Swin depth')
        target = '0' if match[3] == 'fc1' else '3'
        return f'features.{2 * stage + 1}.{block}.mlp.{target}.{match[4]}'
    match = re.fullmatch(r'layers\.([1-3])\.downsample\.norm\.(weight|bias)', key)
    if match:
        return f'features.{2 * int(match[1])}.norm.{match[2]}'
    match = re.fullmatch(r'layers\.([1-3])\.downsample\.reduction\.weight', key)
    if match:
        return f'features.{2 * int(match[1])}.reduction.weight'
    raise ValueError('unsupported public Swin tensor: ' + key)


def relative_position_index(window=WINDOW):
    """Torchvision's deterministic int64 index; never trained and never an EMA/BN statistic."""
    import torch
    coords_h = torch.arange(window)
    coords_w = torch.arange(window)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window - 1
    relative_coords[:, :, 1] += window - 1
    relative_coords[:, :, 0] *= 2 * window - 1
    return relative_coords.sum(-1).flatten()


def index_names():
    return [f'features.{2 * stage + 1}.{block}.attn.relative_position_index'
            for stage, depth in enumerate(DEPTHS) for block in range(depth)]


def source_index_names():
    return [f'layers.{stage}.blocks.{block}.attn.relative_position_index'
            for stage, depth in enumerate(DEPTHS) for block in range(depth)]


def pair(value):
    return value if isinstance(value, tuple) else (value, value)


def drop_branch_probability(module, drop_path_type):
    import torch
    if isinstance(module, torch.nn.Identity):
        return 0.0, 'torch.nn.Identity'
    require(isinstance(module, drop_path_type), 'unexpected drop branch type: ' + type(module).__name__)
    value = getattr(module, 'drop_prob', None)
    require(isinstance(value, float) and 0.0 <= value <= 1.0, 'DropPath branch lacks a real drop_prob')
    return float(value), type(module).__module__ + '.' + type(module).__qualname__


def swin_mapping(source, converted, expected, indices):
    import torch
    require(len(source) == SOURCE_TENSOR_COUNT, 'Swin source parameter tensor count differs')
    names, rows = set(), []
    for key, value in sorted(source.items()):
        name = swin_target_key(key)
        require(name not in names and name in expected and name in converted,
                'Swin mapping missing/collision: ' + key)
        names.add(name)
        check_tensor(value, converted[name], expected[name].shape)
        rows.append({'source': key, 'target': name, 'source_shape': list(value.shape),
                     'target_shape': list(converted[name].shape), 'gamma_reshape': False,
                     'role': ('public1000_head_to_be_discarded' if key.startswith('head.fc.')
                              else 'trainable_relative_position_bias' if key.endswith('relative_position_bias_table')
                              else 'backbone'),
                     'FP32_finite_exact': True})
    index_set = set(index_names())
    require(names == set(converted) - index_set == set(expected) - index_set,
            'Swin parameter mapping coverage differs')
    require(set(indices) == index_set and len(indices) == STATIC_INDEX_COUNT, 'Swin static index set differs')
    for name in sorted(indices):
        require(expected[name].dtype == indices[name].dtype == torch.int64
                and torch.equal(expected[name], indices[name])
                and torch.equal(converted[name], indices[name]),
                'Swin deterministic relative position index differs: ' + name)
    require(len(converted) == len(expected) == TARGET_STATE_COUNT, 'Swin target state count differs')
    return rows


def accept(args, completed):
    if getattr(args, 'variant', 'base') == 'v2-small':
        return accept_v2(args, completed)
    import torch
    import torchvision
    import timm
    from timm.layers import DropPath as TimmDropPath
    from torchvision.models.swin_transformer import SwinTransformer
    from torchvision.ops import StochasticDepth
    context = cpu_context()
    chain = read_pinned_receipt(args.source_receipt, args.source_receipt_sha)
    require(chain.get('passed') is True and chain.get('repo') == REPO and chain.get('model') == MODEL,
            'Swin source receipt identity differs')
    revision = chain.get('revision')
    require(isinstance(revision, str) and len(revision) == 40
            and all(character in '0123456789abcdef' for character in revision),
            'Swin source receipt lacks one immutable commit')
    require(all(chain['execution'][key] == context[key]
                for key in ('namespace', 'pod', 'uid', 'container', 'python')),
            'Swin source execution identity differs')
    require(chain['execution']['cpu_allocation']['path'] == context['cpu_allocation']['path']
            and chain['execution']['cpu_allocation']['sha256'] == context['cpu_allocation']['sha256']
            and chain['execution']['cpu_allocation']['executor_id'] == context['cpu_allocation']['executor_id'],
            'Swin CPU allocation changed; explicit new binding required')
    assets = {name: verify_current_content(record['path'], record['sha256'])
              for name, record in chain['files'].items()}
    cfg = json.loads(Path(assets['config.json']['path']).read_text())
    require(cfg['architecture'] == MODEL and cfg['num_classes'] == 1000 and cfg['num_features'] == 1024,
            'actual Swin config differs')
    configure_torch_threads(context)
    require(not torch.cuda.is_initialized(), 'GPU already initialized')
    source_path = assets['pytorch_model.bin']['path']
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    require(isinstance(source, dict), 'bin is not a plain state dict')
    source_model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=False,
                                    pretrained_cfg=cfg['pretrained_cfg'], img_size=224, drop_path_rate=.1,
                                    drop_rate=0., attn_drop_rate=0.)
    require(tuple(source_model.patch_embed.proj.kernel_size) == (4, 4)
            and tuple(source_model.patch_embed.proj.stride) == (4, 4)
            and source_model.patch_embed.proj.out_channels == EMBED, 'Swin patch embedding differs')
    for stage, depth in enumerate(DEPTHS):
        require(len(source_model.layers[stage].blocks) == depth, 'Swin source depth differs')
        dim = EMBED * 2 ** stage
        for block in source_model.layers[stage].blocks:
            require(block.norm1.normalized_shape[0] == dim and block.attn.num_heads == HEADS[stage]
                    and pair(block.window_size) == (WINDOW, WINDOW)
                    and block.mlp.fc1.out_features == 4 * dim, 'Swin source block structure differs')
    index = relative_position_index()
    index_matrix = index.reshape(WINDOW * WINDOW, WINDOW * WINDOW)
    native_buffers = dict(source_model.named_buffers())
    native_index = {key: value for key, value in native_buffers.items()
                    if key.endswith('.relative_position_index')}
    native_mask = {key: value for key, value in native_buffers.items() if key.endswith('.attn_mask')}
    require(set(native_index) == set(source_index_names()) and len(native_index) == STATIC_INDEX_COUNT,
            'timm native relative position buffers differ')
    require(len(native_mask) == SOURCE_MASK_COUNT
            and len(native_buffers) == len(native_index) + len(native_mask),
            'timm native shifted-window masks or total named buffers differ')
    source_index_keys = {key for key in source if key.endswith('.relative_position_index')}
    source_mask_keys = {key for key in source if key.endswith('.attn_mask')}
    require(source_index_keys == set(native_index), 'source relative position names differ from timm native')
    require(source_mask_keys == set(native_mask), 'source attn_mask names differ from timm native')
    for key in sorted(source_index_keys):
        require(source[key].dtype == torch.int64
                and tuple(source[key].shape) == (WINDOW * WINDOW, WINDOW * WINDOW)
                and torch.equal(source[key], index_matrix)
                and torch.equal(source[key].reshape(-1), index),
                'stored source relative position matrix differs from independent construction: ' + key)
    mask_rows = []
    for key in sorted(source_mask_keys):
        stored, native = source[key], native_mask[key]
        require(stored.dtype == native.dtype and tuple(stored.shape) == tuple(native.shape)
                and bool(torch.isfinite(stored).all()) and bool(torch.isfinite(native).all())
                and torch.equal(stored, native),
                'stored source 224 attn_mask differs from timm native: ' + key)
        mask_rows.append({'name': key, 'shape': list(stored.shape), 'dtype': str(stored.dtype),
                          'finite': True, 'equals_timm_native': True})
    source_buffer_keys = source_index_keys | source_mask_keys
    source_parameters = {key: value for key, value in source.items() if key not in source_buffer_keys}
    require(len(source_parameters) == SOURCE_TENSOR_COUNT
            and len(source) == SOURCE_TENSOR_COUNT + len(source_buffer_keys),
            'Swin source parameter/buffer accounting differs')
    source_loaded = source_model.load_state_dict(source_parameters, strict=True)
    require(not source_loaded.missing_keys and not source_loaded.unexpected_keys,
            'Swin strict trainable source load differs')
    target = SwinTransformer(patch_size=[4, 4], embed_dim=EMBED, depths=list(DEPTHS), num_heads=list(HEADS),
                             window_size=[WINDOW, WINDOW], mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0,
                             stochastic_depth_prob=.1, norm_layer=partial(torch.nn.LayerNorm, eps=1e-5),
                             num_classes=1000)
    indices = {name: index.clone() for name in index_names()}
    converted = {}
    for key, value in source_parameters.items():
        name = swin_target_key(key)
        require(name not in converted, 'Swin mapping collision')
        converted[name] = value
    require(not set(converted) & set(indices), 'Swin parameter/index name collision')
    converted.update(indices)
    rows = swin_mapping(source_parameters, converted, target.state_dict(), indices)
    target_loaded = target.load_state_dict(converted, strict=True)
    require(not target_loaded.missing_keys and not target_loaded.unexpected_keys, 'Swin target strict load differs')
    require(all(torch.equal(value, target.state_dict()[key]) for key, value in converted.items()),
            'loaded Swin target differs')

    source_norms, target_norms = normalization(source_model), normalization(target)
    require(len(source_norms) == len(target_norms) == 53
            and all(row['eps'] == 1e-5 for row in source_norms + target_norms), 'Swin LN count/eps differs')
    source_drop, drop_branch_rows = [], []
    block_index = 0
    for stage, depth in enumerate(DEPTHS):
        require(len(source_model.layers[stage].blocks) == depth, 'Swin source depth differs')
        for position, block in enumerate(source_model.layers[stage].blocks):
            require(hasattr(block, 'drop_path1') and hasattr(block, 'drop_path2'),
                    'timm Swin block lacks drop_path1/drop_path2')
            p1, t1 = drop_branch_probability(block.drop_path1, TimmDropPath)
            p2, t2 = drop_branch_probability(block.drop_path2, TimmDropPath)
            require(abs(p1 - p2) < 1e-12,
                    'two residual drop branches differ: layers.%d.blocks.%d' % (stage, position))
            expected = .1 * block_index / (sum(DEPTHS) - 1)
            require(abs(p1 - expected) < 1e-8,
                    'source drop-path schedule differs at layers.%d.blocks.%d' % (stage, position))
            source_drop.append(p1)
            drop_branch_rows.append({'block': f'layers.{stage}.blocks.{position}', 'type1': t1, 'type2': t2,
                                     'p1': p1, 'p2': p2,
                                     'identity_branches': int(t1 == 'torch.nn.Identity') + int(t2 == 'torch.nn.Identity'),
                                     'expected_linear_p': expected})
            block_index += 1
    target_drop = [float(module.p) for module in target.modules() if isinstance(module, StochasticDepth)]
    require(len(source_drop) == len(target_drop) == 24 and len(drop_branch_rows) == 24
            and abs(source_drop[-1] - .1) < 1e-8 and abs(target_drop[-1] - .1) < 1e-8
            and all(abs(a - b) < 1e-8 for a, b in zip(source_drop, target_drop)), 'Swin SD schedule differs')
    dropouts = [module.p for model in (source_model, target) for module in model.modules()
                if isinstance(module, torch.nn.Dropout)]
    require(dropouts and all(p == 0. for p in dropouts), 'Swin source/model dropout must be zero')
    require(not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                    for model in (source_model, target) for module in model.modules()), 'Swin BN present')
    source_buffers = dict(source_model.named_buffers())
    target_buffers = dict(target.named_buffers())
    require(set(source_buffers) == set(native_index) | set(native_mask)
            and len(source_buffers) == STATIC_INDEX_COUNT + SOURCE_MASK_COUNT,
            'source named buffers differ from 24 index + 11 mask')
    require(set(target_buffers) == set(indices) and len(target_buffers) == STATIC_INDEX_COUNT,
            'target named buffers differ from 24 persistent indices')
    require(all(source_buffers[key].dtype == torch.int64
                and tuple(source_buffers[key].shape) == (WINDOW * WINDOW, WINDOW * WINDOW)
                and torch.equal(source_buffers[key], index_matrix) for key in source_index_keys),
            'source native relative position matrices differ from independent construction')
    require(all(source_buffers[key].dtype == source[key].dtype
                and tuple(source_buffers[key].shape) == tuple(source[key].shape)
                and torch.equal(source_buffers[key], source[key]) for key in source_mask_keys),
            'source native 224 attn_masks differ from stored values')
    require(all(value.dtype == torch.int64 and tuple(value.shape) == (index.numel(),)
                and torch.equal(value, index) for value in target_buffers.values()),
            'target persistent relative position indices differ from independent construction')
    require(source_model.num_features == target.head.in_features == 1024
            and source_model.num_classes == target.head.out_features == 1000
            and isinstance(target.avgpool, torch.nn.AdaptiveAvgPool2d) and target.avgpool.output_size == 1,
            'Swin features/classes/pool differ')
    widths = [target.features[0][0].out_channels, target.features[2].reduction.out_features,
              target.features[4].reduction.out_features, target.features[6].reduction.out_features]
    require(widths == [128, 256, 512, 1024], 'Swin stage widths differ from 128/256/512/1024')

    torch.manual_seed(20261031)
    sample = torch.randn(2, 3, 224, 224)
    source_model.eval(); target.eval()
    with torch.no_grad():
        source_logits = source_model(sample)
        target_logits = target(sample)
    require(tuple(source_logits.shape) == tuple(target_logits.shape) == (2, 1000)
            and bool(torch.isfinite(source_logits).all()) and bool(torch.isfinite(target_logits).all()),
            'Swin full1000 logits shape/finite differs')
    difference = (source_logits - target_logits).abs()
    maximum, mean = float(difference.max()), float(difference.mean())
    require(maximum <= 1e-3 and mean <= 1e-4, 'Swin source/target 224 FP32 numerical parity failed')
    require(torch.equal(source_logits.argmax(1), target_logits.argmax(1)), 'Swin top-1 class differs')

    converted_root = Path(args.assets) / 'converted'
    require(converted_root.is_absolute() and '/exp-v3/' in str(converted_root), 'exp-v3 converted root required')
    converted_root.mkdir(parents=True, exist_ok=True)
    converted_path = converted_root / f'swin-base-tv-{revision[:8]}.pt'
    with converted_path.open('xb') as stream:
        torch.save(converted, stream)
    converted_sha = sha256(converted_path)
    reloaded = torch.load(converted_path, map_location='cpu', weights_only=True)
    swin_mapping(source_parameters, reloaded, target.state_dict(), indices)
    reloaded_loaded = target.load_state_dict(reloaded, strict=True)
    require(not reloaded_loaded.missing_keys and not reloaded_loaded.unexpected_keys
            and all(torch.equal(value, target.state_dict()[key]) for key, value in reloaded.items()),
            'reloaded Swin converted state differs')

    parameter_names = [key for key in reloaded if key not in set(indices)]
    head_names = {'head.weight', 'head.bias'}
    public_parameters = sum(p.numel() for p in target.parameters())
    backbone_parameters = sum(p.numel() for name, p in target.named_parameters() if name not in head_names)
    structure = {'features': 1024, 'public_state_tensors': len(reloaded),
                 'parameter_tensors': len(parameter_names), 'backbone_parameter_tensors': len(parameter_names) - 2,
                 'static_index_buffers': len(indices), 'static_index_dtype': 'torch.int64',
                 'public_parameters': public_parameters, 'backbone_parameters': backbone_parameters,
                 'layer_norms': len(target_norms), 'stochastic_depth_modules': len(target_drop),
                 'full_head_shape': [1000, 1024], 'window': [WINDOW, WINDOW]}
    require(structure['parameter_tensors'] == SOURCE_TENSOR_COUNT
            and structure['backbone_parameter_tensors'] == SOURCE_TENSOR_COUNT - 2, 'Swin tensor counts differ')
    mapping_path = Path(args.output).with_name('MAPPING.json')
    write_new_json(mapping_path, {'schema_version': 1, 'passed': True, 'repo': REPO,
                                 'revision': revision, 'mapping': rows,
                                 'static_indices': [{'name': name, 'dtype': 'torch.int64',
                                                     'shape': list(indices[name].shape),
                                                     'construction': 'torchvision_define_relative_position_index_equivalent'}
                                                    for name in sorted(indices)],
                                 'source_only_attn_masks': mask_rows})
    provenance = {'schema_version': 1, 'passed': True, 'model': MODEL,
                  'format': 'torchvision-1000-state-dict-v1', 'repo': REPO,
                  'revision': revision, 'source_url': f'https://huggingface.co/{REPO}/resolve/{revision}/pytorch_model.bin',
                  'training_data': TRAINING_DATA,
                  'original_sha256': assets['pytorch_model.bin']['sha256'], 'sha256': converted_sha,
                  'final_norm_eps': 1e-5, 'internal_norm_eps': 1e-5,
                  'license_file': '../original/LICENSE-MIT.txt', 'redistribution_confirmed': True,
                  'license_basis': chain['license'],
                  'source_file_hashes': {k: v['sha256'] for k, v in assets.items()},
                  'source_chain': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
                  'mapping': {'path': str(mapping_path), 'sha256': sha256(mapping_path)},
                  'structure': structure}
    provenance_path = converted_root / 'provenance.json'
    write_new_json(provenance_path, provenance)
    implementation_pins = {}
    for label, obj in (('timm_swin', type(source_model)), ('torchvision_swin', type(target)),
                       ('timm_GELU', type(source_model.layers[0].blocks[0].mlp.act)),
                       ('torch_GELU', torch.nn.GELU)):
        path = inspect.getsourcefile(obj)
        implementation_pins[label] = {'path': path, 'sha256': sha256(path)}
    for identity in assets.values():
        require(file_identity(identity['path']) == {k: identity[k] for k in file_identity(identity['path'])},
                'asset metadata changed during acceptance')
    require(not torch.cuda.is_initialized(), 'GPU initialized during Swin acceptance')
    return {'schema_version': 1, 'passed': True, 'scope': 'swinbase22k_full1000_cpu_strict', 'execution': context,
            'model': MODEL, 'repo': REPO, 'revision': revision, 'structure': structure,
            'assets': assets,
            'original': {'path': source_path, 'sha256': assets['pytorch_model.bin']['sha256']},
            'converted': {'path': str(converted_path), 'sha256': converted_sha},
            'provenance': {'path': str(provenance_path), 'sha256': sha256(provenance_path)},
            'source_chain': {'path': args.source_receipt, 'sha256': args.source_receipt_sha},
            'mapping': {'path': str(mapping_path), 'sha256': sha256(mapping_path)},
            'strict_load': completed.setdefault('strict', {
                'source_missing': source_loaded.missing_keys, 'source_unexpected': source_loaded.unexpected_keys,
                'target_missing': target_loaded.missing_keys, 'target_unexpected': target_loaded.unexpected_keys,
                'public_state_tensors': len(reloaded), 'all_FP32_finite_exact': True}),
            'source_implementation_pins': implementation_pins,
            'source_factory': {'name': 'timm.create_model', 'model': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
                               'pretrained': False, 'img_size': 224, 'drop_path_rate': .1,
                               'drop_rate': 0., 'attn_drop_rate': 0., 'timm_version': timm.__version__,
                               'layer_norms': source_norms, 'stochastic_depth': source_drop,
                               'BN': 0, 'state_dict_tensors': len(source_parameters),
                               'named_buffers_total': len(source_buffers),
                               'relative_position_index_buffers': len(source_index_keys),
                               'attn_mask_buffers': len(source_mask_keys)},
            'source_drop_branches': {'blocks': len(drop_branch_rows),
                                     'branch_modules': 2 * len(drop_branch_rows),
                                     'per_block': drop_branch_rows,
                                     'both_branches_same_probability': True,
                                     'linear_0_to_0.1': True,
                                     'equals_target_stochastic_depth': True,
                                     'identity_branches_explicitly_zero': sum(row['identity_branches'] for row in drop_branch_rows)},
            'target_factory': {'name': 'torchvision.models.swin_transformer.SwinTransformer', 'weights': None,
                               'patch_size': [4, 4], 'embed_dim': EMBED, 'depths': list(DEPTHS),
                               'num_heads': list(HEADS), 'window_size': [WINDOW, WINDOW], 'mlp_ratio': 4.0,
                               'dropout': 0., 'attention_dropout': 0., 'stochastic_depth_prob': .1,
                               'norm_eps': 1e-5, 'torchvision_version': torchvision.__version__,
                               'layer_norms': target_norms, 'stochastic_depth': target_drop,
                               'BN': 0, 'static_index_buffers': len(target_buffers)},
            'static_integer_indices': {'count': len(indices), 'dtype': 'torch.int64',
                                       'source_shape_per_block': [WINDOW * WINDOW, WINDOW * WINDOW] if source_index_keys else None,
                                       'target_shape_per_block': list(index.shape),
                                       'layout_relation': 'source_2d_row_major_flatten_equals_target_flat',
                                       'deterministic_construction': True, 'source_contains': bool(source_index_keys),
                                       'source_matches_actual_timm_native': True,
                                       'target_persistent': True,
                                       'is_batch_norm_statistic': False, 'is_ema_parameter': False},
            'source_tensor_accounting': {'total_state_tensors': len(source),
                                         'parameter_tensors': len(source_parameters),
                                         'relative_position_index': len(source_index_keys),
                                         'attn_mask': len(source_mask_keys),
                                         'named_buffers_total': len(source_buffers),
                                         'batch_norm_modules': 0},
            'source_only_224_attn_masks': {'count': len(source_mask_keys), 'per_block': mask_rows,
                                           'converted_to_target': False, 'enters_ema': False,
                                           'note': 'source-only shifted-window masks; target has only 24 persistent int64 flat indices'},
            'mapping_notes': {'patch_merging': 'V1 norm then reduction; source layers.{1,2,3}.downsample -> target features.{2,4,6}',
                              'qkv': 'single Linear(dim, 3*dim), order q/k/v',
                              'mlp': 'fc1->mlp.0, fc2->mlp.3 (torchvision inserts Dropout(0) at index 2)',
                              'norms': 'V1 pre-norm: norm1 before attention, norm2 before MLP'},
            'loaded_public_tensor_sha256': tensor_digest(reloaded),
            'numerical_parity_224': {'input': [2, 3, 224, 224], 'seed': 20261031, 'dtype': 'float32',
                                     'max_abs': maximum, 'mean_abs': mean,
                                     'max_abs_limit': 1e-3, 'mean_abs_limit': 1e-4,
                                     'top1_equal': True, 'finite': True},
            'source_card_preprocessing_is_not_candidate_preprocessing': True,
            'candidate_overrides': chain['candidate'], 'forward_calls': 2, 'datasets_read': False,
            'GPU_initialized_before': False, 'GPU_initialized_after': torch.cuda.is_initialized(),
            'weight_blob_hashes_recomputed': True,
            'validation_source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                         ('swin_source.py', 'swin_strict.py', 'strict_load.py', 'source_chain.py')},
            'production_binary_age_optimizer_admission': 'pending A, no binary or 512 demonstration here'}


def accept_v2(args, completed):
    """Fixed Small V2 extension of the same source/strict/handoff chain."""
    import torch, torchvision, timm
    from torchvision.models.swin_transformer import SwinTransformer, SwinTransformerBlockV2, PatchMergingV2
    from swin_source import V2_REPO, V2_MODEL, V2_REVISION
    context = cpu_context()
    chain = read_pinned_receipt(args.source_receipt, args.source_receipt_sha)
    require(chain.get('passed') is True and (chain.get('repo'), chain.get('model'), chain.get('revision'))
            == (V2_REPO, V2_MODEL, V2_REVISION), 'fixed Swin V2 source differs')
    require(all(chain['execution'].get(k)==v for k,v in context.items() if k!='at'),
            'Swin V2 source CPU execution changed')
    configure_torch_threads(context)
    assets = {n: verify_current_content(r['path'], r['sha256']) for n,r in chain['files'].items()}
    cfg = json.loads(Path(assets['config.json']['path']).read_text())
    source = torch.load(assets['pytorch_model.bin']['path'], map_location='cpu', weights_only=True)
    require(isinstance(source, dict) and len(source) == 436, 'Swin V2 complete source state differs')
    source_model = timm.create_model(V2_REPO.split('/')[1], pretrained=False, pretrained_cfg=cfg['pretrained_cfg'],
                                    img_size=256, drop_path_rate=.1, drop_rate=0., attn_drop_rate=0.)
    masks = {n:t for n,t in source.items() if n.endswith('.attn_mask')}
    require(len(masks) == 11, 'Swin V2 source mask coverage differs')
    native = dict(source_model.named_buffers())
    require(all(n in native and torch.equal(t,native[n]) for n,t in masks.items()), 'Swin V2 native masks differ')
    parameters = {n:t for n,t in source.items() if n not in masks}
    loaded_source = source_model.load_state_dict(parameters, strict=True)
    require(not loaded_source.missing_keys and not loaded_source.unexpected_keys, 'Swin V2 strict source load differs')
    target = SwinTransformer(patch_size=[4,4], embed_dim=96, depths=list(DEPTHS), num_heads=[3,6,12,24],
              window_size=[8,8], stochastic_depth_prob=.1, norm_layer=partial(torch.nn.LayerNorm,eps=1e-5),
              num_classes=1000, block=SwinTransformerBlockV2, downsample_layer=PatchMergingV2)
    expected=target.state_dict(); converted={}; rows=[]; consumed=set()
    for key,value in sorted(parameters.items()):
        if key in consumed:continue
        match=re.fullmatch(r'layers\.([0-3])\.blocks\.(\d+)\.attn\.(logit_scale|cpb_mlp\.[02]\.(?:weight|bias)|q_bias|v_bias)',key)
        if match:
            stage,block,tail=int(match[1]),int(match[2]),match[3]
            name=f'features.{2*stage+1}.{block}.attn.'
            if tail in ('q_bias','v_bias'):
                require(tail=='q_bias', 'Swin V2 q/v mapping order differs')
                other=key.removesuffix('q_bias')+'v_bias'
                require(other in parameters and parameters[other].shape==value.shape, 'Swin V2 v bias missing')
                mapped=torch.cat((value,torch.zeros_like(value),parameters[other]));name+='qkv.bias'
                consumed.add(other); sources=[key,other]
            else:name+=tail;mapped=value;sources=[key]
        else:name=swin_target_key(key);mapped=value;sources=[key]
        require(name in expected and name not in converted, 'Swin V2 mapping collision: '+key)
        check_tensor(mapped,mapped,expected[name].shape)
        converted[name]=mapped;consumed.add(key)
        rows.append({'sources':sources,'target':name,'shape':list(mapped.shape),'FP32_finite_exact':True,
                     'transform':'q_zero_k_v_concat' if len(sources)==2 else 'identity'})
    require(consumed==set(parameters) and len(converted)==401, 'Swin V2 trainable mapping coverage differs')
    static=dict(target.named_buffers())
    require(len(static)==48, 'Swin V2 target static count differs')
    for name,tensor in static.items():
        match=re.fullmatch(r'features\.([1357])\.(\d+)\.attn\.(relative_coords_table|relative_position_index)',name)
        require(match is not None, 'unregistered Swin V2 target buffer')
        source_name=f'layers.{(int(match[1])-1)//2}.blocks.{match[2]}.attn.{match[3]}'
        require(source_name in native, 'Swin V2 generated source buffer missing')
        original=native[source_name].reshape(-1) if name.endswith('relative_position_index') else native[source_name]
        require(torch.equal(tensor,original), 'Swin V2 deterministic source/target buffer differs: '+name)
        converted[name]=tensor.clone()
    require(set(converted)==set(expected) and len(converted)==449, 'Swin V2 complete target state differs')
    loaded_target=target.load_state_dict(converted,strict=True)
    require(not loaded_target.missing_keys and not loaded_target.unexpected_keys
            and all(torch.equal(t,target.state_dict()[n]) for n,t in converted.items()), 'Swin V2 strict target differs')
    source_norms,target_norms=normalization(source_model),normalization(target)
    require(len(source_norms)==len(target_norms)==53 and all(r['eps']==1e-5 for r in source_norms+target_norms),
            'Swin V2 native LN differs')
    torch.manual_seed(20261031); sample=torch.randn(2,3,256,256);source_model.eval();target.eval()
    with torch.inference_mode():a=source_model(sample);b=target(sample)
    difference=(a-b).abs();maximum,mean=float(difference.max()),float(difference.mean())
    require(a.shape==b.shape==(2,1000) and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            and maximum<=1e-3 and mean<=1e-4 and torch.equal(a.argmax(1),b.argmax(1)),
            'Swin V2 source/target FP32 numerical parity failed')
    destination=Path(args.assets)/'converted';destination.mkdir(parents=True,exist_ok=False)
    weight=destination/f'swin-v2-small-tv-{V2_REVISION[:8]}.pt'
    with weight.open('xb') as stream:torch.save(converted,stream)
    reloaded=torch.load(weight,map_location='cpu',weights_only=True)
    require(reloaded.keys()==converted.keys() and all(torch.equal(t,reloaded[n]) for n,t in converted.items()),
            'Swin V2 cold saved tensors differ')
    target.load_state_dict(reloaded,strict=True)
    structure={'features':768,'public_state_tensors':449,'parameter_tensors':401,'backbone_parameter_tensors':399,
       'static_index_buffers':24,'static_buffer_count':48,'static_index_dtype':'torch.int64',
       'static_float_buffers':24,'static_float_dtype':'torch.float32','public_parameters':sum(p.numel() for p in target.parameters()),
       'backbone_parameters':sum(p.numel() for n,p in target.named_parameters() if not n.startswith('head.')),
       'layer_norms':53,'stochastic_depth_modules':24,'full_head_shape':[1000,768],'window':[8,8]}
    mapping=destination/'tensor-mapping.json';write_new_json(mapping,{'passed':True,'rows':rows,'generated_static_buffers':sorted(static)})
    provenance={'passed':True,'model':V2_MODEL,'format':'torchvision-1000-state-dict-v1','revision':V2_REVISION,
       'source_url':chain['files']['pytorch_model.bin']['url'],'training_data':'ImageNet-1K',
       'original_sha256':assets['pytorch_model.bin']['sha256'],'sha256':sha256(weight),
       'final_norm_eps':1e-5,'internal_norm_eps':1e-5,'license_file':'../original/LICENSE-MIT.txt',
       'license':'MIT','license_sha256':assets['LICENSE-MIT.txt']['sha256'],'redistribution_confirmed':True,
       'source_chain':{'path':args.source_receipt,'sha256':args.source_receipt_sha},
       'mapping':{'path':str(mapping),'sha256':sha256(mapping)},'structure':structure}
    provenance_path=destination/'provenance.json';write_new_json(provenance_path,provenance)
    completed.update(source_strict=True,target_strict=True,full1000_FP32_parity=True)
    require(not torch.cuda.is_initialized(), 'Swin V2 initialized GPU')
    return {'schema_version':1,'passed':True,'scope':'swinv2small1k_full1000_cpu_strict','execution':context,
       'model':V2_MODEL,'repo':V2_REPO,'revision':V2_REVISION,'structure':structure,'assets':assets,
       'original':{'path':assets['pytorch_model.bin']['path'],'sha256':assets['pytorch_model.bin']['sha256']},
       'converted':{'path':str(weight),'sha256':sha256(weight)},
       'provenance':{'path':str(provenance_path),'sha256':sha256(provenance_path)},
       'source_chain':{'path':args.source_receipt,'sha256':args.source_receipt_sha},
       'mapping':{'path':str(mapping),'sha256':sha256(mapping)},'strict_load':completed,
       'source_implementation_pins':{label:{'path':inspect.getsourcefile(obj),'sha256':sha256(inspect.getsourcefile(obj))}
          for label,obj in (('timm_swin',type(source_model)),('torchvision_swin',type(target)))},
       'numerical_parity_256':{'max_abs':maximum,'mean_abs':mean,'max_abs_limit':1e-3,'mean_abs_limit':1e-4,'top1_equal':True},
       'GPU_initialized':False,'datasets_read':False,
       'validation_source_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ('swin_source.py','swin_strict.py','source_chain.py','strict_load.py')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'source-receipt-sha', 'assets', 'output', 'handoff'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--variant', choices=('base','v2-small'), default='base')
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

"""Fixed stateless Tri3 at the three native Small stage transitions.

Original expression using ordinary PyTorch padding/pooling, never fused or
merged. The public model and its binary head/age must already be constructed.
"""
import copy

import torch
from torch import nn
from torchvision.models.convnext import LayerNorm2d

TRI3_RULE = {'version': 'convnext-small-tri3-downsample-v1', 'targets': [2, 4, 6],
             'convolution': 'native2-stride1-pad0', 'filter': 'reflect1-box2s1-box2s2'}
TARGETS = ((2, 96, 192), (4, 192, 384), (6, 384, 768))


def validate_job(job):
    policy = job.get('tri3_downsample_policy')
    if policy is None:
        return None
    if (policy != TRI3_RULE or job.get('model') != 'convnext_small'
            or job.get('precision') != 'fp32' or job.get('head_lr_scope') != 'last_linear'
            or not (job.get('age') or {}).get('enabled')
            or any(job.get(key) is not None for key in
                   ('output_regularization', 'low_rank_policy', 'layerwise_lr_policy',
                    'freeze_features_before', 'final_downsample_stride', 'attention_pool',
                    'stage3_residual', 'head_warmup_policy', 'gradient_clip_policy'))):
        raise ValueError('Tri3 requires the fixed full FP32 Small age transition policy')
    return policy


def _targets(model, *, attached):
    found = []
    for index, inputs, outputs in TARGETS:
        stage = model.network.features[index]
        if (type(stage) is not nn.Sequential or list(stage._modules) != [str(i) for i in range(5 if attached else 2)]
                or type(stage[0]) is not LayerNorm2d or tuple(stage[0].normalized_shape) != (inputs,)
                or stage[0].eps != 1e-6 or type(stage[1]) is not nn.Conv2d):
            raise ValueError('native Small LN/convolution transition differs')
        conv = stage[1]
        if ((conv.in_channels, conv.out_channels) != (inputs, outputs)
                or conv.kernel_size != (2, 2) or conv.stride != ((1, 1) if attached else (2, 2))
                or conv.padding != (0, 0) or conv.dilation != (1, 1) or conv.groups != 1
                or conv.padding_mode != 'zeros' or conv.bias is None
                or conv.weight.dtype != torch.float32):
            raise ValueError('native Small 2x2 convolution identity differs')
        if attached:
            if (type(stage[2]) is not nn.ReflectionPad2d or stage[2].padding != (1, 1, 1, 1)
                    or any(type(stage[position]) is not nn.AvgPool2d
                           or stage[position].kernel_size != 2 or stage[position].stride != stride
                           or stage[position].padding != 0 or stage[position].ceil_mode is not False
                           or stage[position].count_include_pad is not True
                           or stage[position].divisor_override is not None
                           for position, stride in ((3, 1), (4, 2)))
                    or any(list(module.parameters()) or list(module.buffers()) for module in stage[2:])):
                raise ValueError('Tri3 reflection/two ordinary box filters differ')
        found.append((index, stage))
    return found


def attach_tri3(model, job):
    if validate_job(job) is None:
        return model
    targets = _targets(model, attached=False)  # Validate all three before any mutation.
    for _, stage in targets:
        stage[1].stride = (1, 1)
        for index, module in ((2, nn.ReflectionPad2d(1)),
                              (3, nn.AvgPool2d(2, stride=1, padding=0, ceil_mode=False)),
                              (4, nn.AvgPool2d(2, stride=2, padding=0, ceil_mode=False))):
            module.train(stage.training)
            stage.add_module(str(index), module)
    describe_tri3(model, job)
    return model


def describe_tri3(model, job):
    if validate_job(job) is None:
        return None
    rows = []
    for index, stage in _targets(model, attached=True):
        conv = stage[1]
        rows.append({'feature_index': index, 'normalization_name': f'network.features.{index}.0',
            'convolution_name': f'network.features.{index}.1',
            'input_channels': conv.in_channels, 'output_channels': conv.out_channels,
            'convolution_kernel': list(conv.kernel_size), 'convolution_stride': list(conv.stride),
            'convolution_padding': list(conv.padding),
            'tail': [{'kind': type(stage[2]).__name__, 'padding': list(stage[2].padding)},
                     *[{'kind': type(stage[p]).__name__, 'kernel_size': stage[p].kernel_size,
                        'stride': stage[p].stride, 'padding': stage[p].padding,
                        'ceil_mode': stage[p].ceil_mode, 'count_include_pad': stage[p].count_include_pad,
                        'divisor_override': stage[p].divisor_override} for p in (3, 4)]]})
    return {'policy': copy.deepcopy(job['tri3_downsample_policy']), 'targets': rows}

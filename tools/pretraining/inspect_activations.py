"""Describe actual source/target GELU modules without loading weights or forwarding."""
import argparse
import hashlib
import inspect
from pathlib import Path

from source_chain import configure_torch_threads, cpu_context, read_pinned_receipt, sha256, write_new_json


def describe(name, module, implementations):
    cls = type(module)
    qualified = cls.__module__ + '.' + cls.__qualname__
    if qualified not in implementations:
        path = inspect.getsourcefile(cls)
        forward = inspect.getsource(cls.forward)
        implementations[qualified] = {'path': path, 'sha256': sha256(path),
                                      'class_source': inspect.getsource(cls),
                                      'forward_source': forward,
                                      'forward_source_sha256': hashlib.sha256(forward.encode()).hexdigest()}
    return {'name': name, 'qualname': qualified, 'repr': repr(module),
            'approximate': getattr(module, 'approximate', None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'source-receipt-sha', 'failed-receipt', 'failed-receipt-sha', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    context = cpu_context()
    chain = read_pinned_receipt(args.source_receipt, args.source_receipt_sha)
    failed = read_pinned_receipt(args.failed_receipt, args.failed_receipt_sha)
    import torch
    import timm
    from torchvision.models import convnext_tiny
    configure_torch_threads(context)
    source = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=False,
                              pretrained_cfg=chain['public_pretrained_config'], drop_path_rate=.1)
    target = convnext_tiny(weights=None, stochastic_depth_prob=.1)
    target.classifier[0].eps = 1e-6
    implementations = {}
    source_acts = [describe(f'stages.{s}.blocks.{b}.mlp.act', block.mlp.act, implementations)
                   for s, stage in enumerate(source.stages) for b, block in enumerate(stage.blocks)]
    target_acts = [describe(f'features.{s}.{b}.block.4', block.block[4], implementations)
                   for s in (1, 3, 5, 7) for b, block in enumerate(target.features[s])]
    checker = Path(__file__).with_name('strict_load.py')
    frozen_checker = Path(args.output).parent / 'strict_load.failed1.py'
    with frozen_checker.open('xb') as stream:
        stream.write(checker.read_bytes())
    result = {'passed': True, 'scope': 'activation_observation_only_not_source_acceptance',
              'execution': context, 'timm_version': timm.__version__,
              'timm_creation': {'model': 'convnext_tiny.fb_in22k_ft_in1k', 'pretrained': False,
                                'pretrained_cfg': chain['public_pretrained_config'], 'drop_path_rate': .1},
              'source_activations': source_acts, 'target_activations': target_acts,
              'implementations': implementations,
              'failed_acceptance': {'path': args.failed_receipt, 'sha256': args.failed_receipt_sha,
                                    'receipt': failed, 'checker': str(frozen_checker),
                                    'checker_sha256': sha256(frozen_checker)},
              'diagnostic_source_sha256': sha256(__file__), 'weight_loads': 0, 'forward_calls': 0,
              'GPU_initialized': torch.cuda.is_initialized()}
    write_new_json(args.output, result)
    print({'source_types': sorted({v['qualname'] for v in source_acts}),
           'target_types': sorted({v['qualname'] for v in target_acts}),
           'source_repr': source_acts[0]['repr'], 'target_repr': target_acts[0]['repr'],
           'implementations': implementations}, flush=True)


if __name__ == '__main__':
    main()

"""Fetch exactly timm/convnext_small.fb_in22k at a caller-fixed revision."""
import argparse
import json
import re
from pathlib import Path

from source_chain import cpu_context, require, sha256, write_new_json
from small_source import fetch


REPO = 'timm/convnext_small.fb_in22k'
TAG = 'fb_in22k'
MODEL = 'convnext_small'
FORMAT = 'torchvision-small22k-only-full-head-state-dict-v1'
TRAINING_DATA = 'ImageNet-22K pretraining (no ImageNet-1K fine-tuning)'
LICENSE_URL = 'https://www.apache.org/licenses/LICENSE-2.0.txt'


def validate_revision(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value) is not None,
            'explicit 40-character lowercase hexadecimal revision required')
    return value


def num_classes(config):
    value = config.get('num_classes') if isinstance(config, dict) else None
    cfg = config.get('pretrained_cfg') if isinstance(config, dict) else None
    require(type(value) is int and value > 0 and isinstance(cfg, dict)
            and type(cfg.get('num_classes')) is int and cfg['num_classes'] == value,
            'config/pretrained_cfg num_classes must be the same positive integer')
    return value


def preprocessing(cfg):
    names = ('mean', 'std', 'input_size', 'test_input_size', 'interpolation',
             'crop_pct', 'test_crop_pct', 'crop_mode')
    return {name: cfg.get(name) for name in names}


def download(assets, revision, transport_dir=None):
    context = cpu_context()
    revision = validate_revision(revision)
    assets = Path(assets)
    require(assets.is_absolute() and '/exp-v3/' in str(assets), 'explicit exp-v3 asset root required')
    original = assets / 'original'
    original.mkdir(parents=True, exist_ok=True)
    if transport_dir:
        require(Path(transport_dir).is_absolute() and '/exp-v3/' in str(transport_dir),
                'explicit exp-v3 transport root required')
    transport = lambda name: str(Path(transport_dir) / name) if transport_dir else None
    base = f'https://huggingface.co/{REPO}/resolve/{revision}'
    api_url = f'https://huggingface.co/api/models/{REPO}/revision/{revision}?blobs=true'
    records = {}
    records['model-api.json'] = fetch(api_url, original / 'model-api.json',
                                      transport_file=transport('model-api.json'))
    api = json.loads((original / 'model-api.json').read_text())
    require(api.get('id') == REPO and api.get('sha') == revision
            and api.get('private') is False and api.get('gated') is False,
            'fixed-revision public registry identity differs')
    require(api.get('cardData', {}).get('license') == 'apache-2.0', 'registry license differs')
    siblings = [item for item in api['siblings'] if item['rfilename'] == 'pytorch_model.bin']
    require(len(siblings) == 1, 'exactly one bin entry required at fixed revision')
    blob = siblings[0]
    lfs = blob.get('lfs') or {}
    expected_sha, expected_size = lfs.get('sha256'), blob.get('size')
    require(isinstance(expected_sha, str) and len(expected_sha) == 64
            and type(expected_size) is int and expected_size > 0,
            'fixed-revision API must provide bin LFS SHA and size')
    for name in ('config.json', 'README.md'):
        records[name] = fetch(base + '/' + name, original / name, transport_file=transport(name))
    config = json.loads((original / 'config.json').read_text())
    K = num_classes(config)
    cfg = config['pretrained_cfg']
    require(config.get('architecture') == MODEL and config.get('num_features') == 768,
            'fixed-revision architecture/features differ')
    require(cfg.get('tag') == TAG and cfg.get('num_classes') == K
            and cfg.get('first_conv') == 'stem.0' and cfg.get('classifier') == 'head.fc',
            'fixed-revision tag/head mapping differs')
    require(isinstance(cfg.get('mean'), list) and len(cfg['mean']) == 3
            and isinstance(cfg.get('std'), list) and len(cfg['std']) == 3
            and all(type(v) in (int, float) for v in cfg['mean'] + cfg['std'])
            and all(v > 0 for v in cfg['std']),
            'fixed-revision normalization metadata differs')
    readme = (original / 'README.md').read_text().lower()
    require('license: apache-2.0' in readme and 'imagenet-22k' in readme,
            'fixed-revision model card license/22K recipe differs')
    records['LICENSE-apache-2.0.txt'] = fetch(
        LICENSE_URL, original / 'LICENSE-apache-2.0.txt', transport_file=transport('LICENSE-apache-2.0.txt'))
    license_text = (original / 'LICENSE-apache-2.0.txt').read_text()
    require('Apache License' in license_text and 'Version 2.0, January 2004' in license_text
            and 'END OF TERMS AND CONDITIONS' in license_text, 'official license text incomplete')
    records['pytorch_model.bin'] = fetch(
        base + '/pytorch_model.bin', original / 'pytorch_model.bin',
        expected_sha, expected_size, transport('pytorch_model.bin'))
    receipt = {'schema_version': 1, 'passed': True,
               'scope': 'small22k_only_fixed_revision_download', 'execution': context,
               'model': MODEL, 'repo': REPO, 'tag': TAG, 'revision': revision,
               'revision_source': 'caller_supplied_40hex_verified_by_fixed_revision_api',
               'training_data': TRAINING_DATA, 'format': FORMAT, 'num_classes': K,
               'registry': {'id': REPO, 'revision': revision, 'private': False,
                            'gated': False, 'license': 'apache-2.0'},
               'public_pretrained_config': cfg,
               'preprocessing': preprocessing(cfg),
               'files': records,
               'license': {'identifier': 'apache-2.0',
                           'declaration': 'fixed-revision model card and registry',
                           'full_text_url': LICENSE_URL, 'redistribution_confirmed': True},
               'candidate': {'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6,
                             'stochastic_depth': .1, 'dropout': .2, 'image_size': 512,
                             'source_card_preprocessing_is_not_candidate_preprocessing': True},
               'downloaded_weight_formats': ['pytorch_model.bin'],
               'validation_source_sha256': {
                   'small22k_only_source.py': sha256(__file__),
                   'source_chain.py': sha256(Path(__file__).with_name('source_chain.py')),
                   'small_source.py': sha256(Path(__file__).with_name('small_source.py'))},
               'forward_calls': 0, 'datasets_read': False, 'GPU_imported': False}
    write_new_json(original / 'download-receipt.json', receipt)
    receipt['download_receipt'] = {'path': str(original / 'download-receipt.json'),
                                    'sha256': sha256(original / 'download-receipt.json')}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True,
                        help='explicit 40-character lowercase commit discovered on the Pod')
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--transport-dir',
                        help='explicit authorized raw byte relay; parsing and hashes stay on Pod')
    args = parser.parse_args()
    try:
        result = download(args.assets, args.revision, args.transport_dir)
    except Exception as error:
        write_new_json(args.output, {'schema_version': 1, 'passed': False,
                                     'execution': cpu_context(), 'repo': REPO, 'tag': TAG,
                                     'revision': args.revision,
                                     'error_type': type(error).__name__, 'error': str(error)})
        raise
    write_new_json(args.output, result)


if __name__ == '__main__':
    main()

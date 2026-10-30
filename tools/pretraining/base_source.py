"""Fetch exactly the approved public Base revision on the guarded compute Pod CPU.

The single registry metadata read fixes one immutable commit; every file is then
fetched at that commit. Only the generic byte-fetch helper is reused from the
Small tool; no Small revision, SHA or size is inherited.
"""
import argparse
import json
from pathlib import Path

from source_chain import cpu_context, require, sha256, write_new_json
from small_source import fetch


REPO = 'timm/convnext_base.fb_in22k_ft_in1k'
MODEL = 'convnext_base'
TRAINING_DATA = 'ImageNet-22K pretraining -> ImageNet-1K fine-tuning'
API_URL = f'https://huggingface.co/api/models/{REPO}?blobs=true'
LICENSE_URL = 'https://www.apache.org/licenses/LICENSE-2.0.txt'


def download(assets, transport_dir=None):
    context = cpu_context()
    assets = Path(assets)
    require(assets.is_absolute() and '/exp-v3/' in str(assets), 'explicit exp-v3 asset root required')
    original = assets / 'original'
    original.mkdir(parents=True, exist_ok=True)
    if transport_dir:
        require(Path(transport_dir).is_absolute() and '/exp-v3/' in str(transport_dir),
                'explicit exp-v3 transport root required')
    transport = lambda name: str(Path(transport_dir) / name) if transport_dir else None
    api_record = fetch(API_URL, original / 'model-api.json', transport_file=transport('model-api.json'))
    api = json.loads((original / 'model-api.json').read_text())
    revision = api.get('sha')
    require(isinstance(revision, str) and len(revision) == 40
            and all(character in '0123456789abcdef' for character in revision),
            'single metadata read did not return one immutable commit')
    require(api.get('id') == REPO and api.get('private') is False and api.get('gated') is False,
            'fixed public non-gated registry identity differs')
    require(api.get('cardData', {}).get('license') == 'apache-2.0', 'registry license differs')
    require('384' not in REPO and 'v2' not in REPO.lower(), 'non-approved Base variant')
    siblings = [s for s in api['siblings'] if s['rfilename'] == 'pytorch_model.bin']
    require(len(siblings) == 1, 'exactly one pytorch_model.bin entry required at the pinned commit')
    blob = siblings[0]
    expected_sha, expected_size = blob.get('lfs', {}).get('sha256'), blob.get('size')
    require(isinstance(expected_sha, str) and len(expected_sha) == 64
            and type(expected_size) is int and expected_size > 0,
            'pinned commit must provide bin LFS SHA and size')
    base = f'https://huggingface.co/{REPO}/resolve/{revision}'
    records = {'model-api.json': api_record}
    for name in ('config.json', 'README.md'):
        records[name] = fetch(base + '/' + name, original / name, transport_file=transport(name))
    config = json.loads((original / 'config.json').read_text())
    cfg = config['pretrained_cfg']
    require((config.get('architecture'), config.get('num_classes'), config.get('num_features'))
            == (MODEL, 1000, 1024), 'public Base architecture config differs')
    require(cfg.get('tag') == 'fb_in22k_ft_in1k' and cfg.get('num_classes') == 1000
            and cfg.get('first_conv') == 'stem.0' and cfg.get('classifier') == 'head.fc'
            and cfg.get('mean') == [.485, .456, .406] and cfg.get('std') == [.229, .224, .225]
            and cfg.get('input_size') == [3, 224, 224] and cfg.get('test_input_size') == [3, 288, 288]
            and cfg.get('interpolation') == 'bicubic' and cfg.get('crop_pct') == .875
            and cfg.get('test_crop_pct') == 1. and cfg.get('crop_mode') == 'center',
            'public non-tensor Base configuration differs')
    readme = (original / 'README.md').read_text()
    require('license: apache-2.0' in readme and 'imagenet-1k' in readme and 'imagenet-22k' in readme
            and 'ImageNet-1k' in readme and 'ImageNet-22k' in readme and 'fine-tuned' in readme,
            'pinned Base model card license/datasets/recipe differ')
    records['LICENSE-apache-2.0.txt'] = fetch(LICENSE_URL, original / 'LICENSE-apache-2.0.txt',
                                               transport_file=transport('LICENSE-apache-2.0.txt'))
    license_text = (original / 'LICENSE-apache-2.0.txt').read_text()
    require('Apache License' in license_text and 'Version 2.0, January 2004' in license_text
            and 'END OF TERMS AND CONDITIONS' in license_text, 'official license text incomplete')
    records['pytorch_model.bin'] = fetch(base + '/pytorch_model.bin', original / 'pytorch_model.bin',
                                         expected_sha, expected_size, transport('pytorch_model.bin'))
    receipt = {'schema_version': 1, 'passed': True, 'scope': 'base22k_fixed_revision_download',
               'execution': context, 'model': MODEL, 'repo': REPO, 'revision': revision,
               'revision_source': 'single_registry_metadata_read_then_all_files_at_that_commit',
               'training_data': TRAINING_DATA, 'files': records,
               'public': True, 'private': False, 'gated': False,
               'lfs_weight': {'sha256': expected_sha, 'bytes': expected_size},
               'public_pretrained_config': cfg,
               'candidate': {'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6,
                             'stochastic_depth': .1, 'dropout': .2, 'image_size': 512,
                             'source_card_preprocessing_is_not_candidate_preprocessing': True},
               'license': {'identifier': 'apache-2.0', 'declaration': 'fixed-revision model card and registry',
                           'full_text_url': LICENSE_URL, 'redistribution_confirmed': True},
               'validation_source_sha256': {'base_source.py': sha256(__file__),
                                            'source_chain.py': sha256(Path(__file__).with_name('source_chain.py'))},
               'downloaded_weight_formats': ['pytorch_model.bin'], 'forward_calls': 0,
               'datasets_read': False, 'GPU_imported': False}
    write_new_json(original / 'download-receipt.json', receipt)
    receipt['download_receipt'] = {'path': str(original / 'download-receipt.json'),
                                   'sha256': sha256(original / 'download-receipt.json')}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--transport-dir', help='Explicit authorized raw byte relay; all parsing and hashes stay on Pod')
    args = parser.parse_args()
    try:
        result = download(args.assets, args.transport_dir)
    except Exception as error:
        write_new_json(args.output, {'schema_version': 1, 'passed': False, 'execution': cpu_context(),
                                     'error_type': type(error).__name__, 'error': str(error)})
        raise
    write_new_json(args.output, result)


if __name__ == '__main__':
    main()

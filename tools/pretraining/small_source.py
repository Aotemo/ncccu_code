"""Fetch exactly the approved Small revision on the guarded compute Pod CPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
from urllib.parse import urlparse

from source_chain import cpu_context, file_identity, require, sha256, write_new_json


REPO = 'timm/convnext_small.fb_in22k_ft_in1k'
REVISION = 'fe6b358aa409414cdb522ae3d43a2c9922f7cc1b'
MODEL = 'convnext_small'
TRAINING_DATA = 'ImageNet-22K pretraining -> ImageNet-1K fine-tuning'
BASE = f'https://huggingface.co/{REPO}/resolve/{REVISION}'
LICENSE_URL = 'https://www.apache.org/licenses/LICENSE-2.0.txt'
SOURCE_PROFILES = {
    'small224': {'repo': REPO, 'revision': REVISION, 'tag': 'fb_in22k_ft_in1k'},
    'small384': {'repo': 'timm/convnext_small.fb_in22k_ft_in1k_384',
                 'revision': '14a837353b6ffeced1c635dc5c8c799356e316f4',
                 'tag': 'fb_in22k_ft_in1k_384'},
}


def fetch(url, target, expected_sha=None, expected_bytes=None, transport_file=None):
    target = Path(target)
    require(not target.exists(), 'refusing to overwrite asset: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={'User-Agent': 'ncccu-public-source-verification/1'})
    started = datetime.now(timezone.utc).isoformat()
    # Retain failed attempts; a later authorized retry never rewrites them.
    output = tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.part-', delete=False)
    part = Path(output.name)
    with output:
        response = (Path(transport_file).open('rb') if transport_file else
                    urllib.request.urlopen(request, timeout=60))
        with response:
            if transport_file is None:
                require(response.status == 200, 'HTTP status differs')
            final_host = urlparse(response.url).netloc if transport_file is None else None
            advertised = response.headers.get('Content-Length') if transport_file is None else None
            for block in iter(lambda: response.read(1024 * 1024), b''):
                output.write(block)
                digest.update(block)
                count += len(block)
            if advertised is not None:
                require(count == int(advertised), 'HTTP content length differs')
    actual_sha = digest.hexdigest()
    require(expected_sha is None or actual_sha == expected_sha, 'fixed-revision LFS SHA differs')
    require(expected_bytes is None or count == expected_bytes, 'fixed-revision LFS size differs')
    part.rename(target)
    return {'url': url, 'started_at': started, 'completed_at': datetime.now(timezone.utc).isoformat(),
            'path': str(target), 'bytes': count, 'sha256': actual_sha, 'resolved_host': final_host,
            'transport': 'authorized_raw_byte_relay_verified_on_pod' if transport_file else 'compute_pod_https',
            'transport_path': str(transport_file) if transport_file else None,
            'expected_lfs_sha256': expected_sha, 'expected_lfs_bytes': expected_bytes,
            'identity': file_identity(target)}


def download(assets, transport_dir=None, source_profile='small224'):
    context = cpu_context()
    profile = SOURCE_PROFILES[source_profile]
    repo, revision = profile['repo'], profile['revision']
    base = f'https://huggingface.co/{repo}/resolve/{revision}'
    assets = Path(assets)
    require(assets.is_absolute() and '/exp-v3/' in str(assets), 'explicit exp-v3 asset root required')
    original = assets / 'original'
    original.mkdir(parents=True, exist_ok=True)
    if transport_dir:
        require(Path(transport_dir).is_absolute() and '/exp-v3/' in str(transport_dir),
                'explicit exp-v3 transport root required')
    transport = lambda name: str(Path(transport_dir) / name) if transport_dir else None
    api_url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
    api_record = fetch(api_url, original / 'model-api.json', transport_file=transport('model-api.json'))
    api = json.loads((original / 'model-api.json').read_text())
    require(api.get('id') == repo and api.get('sha') == revision
            and api.get('private') is False and api.get('gated') is False,
            'fixed-revision public registry identity differs')
    require(api.get('cardData', {}).get('license') == 'apache-2.0', 'registry license differs')
    siblings = [s for s in api['siblings'] if s['rfilename'] == 'pytorch_model.bin']
    require(len(siblings) == 1, 'one bin entry required at fixed revision')
    blob = siblings[0]
    lfs = blob.get('lfs', {})
    expected_sha, expected_size = lfs.get('sha256'), blob.get('size')
    require(isinstance(expected_sha, str) and len(expected_sha) == 64
            and type(expected_size) is int and expected_size > 0,
            'fixed-revision API must provide bin LFS SHA and size')
    records = {'model-api.json': api_record}
    for name in ('config.json', 'README.md'):
        records[name] = fetch(base + '/' + name, original / name, transport_file=transport(name))
    config = json.loads((original / 'config.json').read_text())
    cfg = config['pretrained_cfg']
    require((config.get('architecture'), config.get('num_classes'), config.get('num_features'))
            == (MODEL, 1000, 768), 'public architecture config differs')
    require(cfg.get('tag') == profile['tag'] and cfg.get('num_classes') == 1000
            and cfg.get('first_conv') == 'stem.0' and cfg.get('classifier') == 'head.fc'
            and cfg.get('mean') == [.485, .456, .406] and cfg.get('std') == [.229, .224, .225]
            and cfg.get('interpolation') == 'bicubic',
            'public non-tensor configuration differs')
    if source_profile == 'small224':
        require(cfg.get('input_size') == [3, 224, 224] and cfg.get('test_input_size') == [3, 288, 288]
                and cfg.get('crop_pct') == .875 and cfg.get('test_crop_pct') == 1.
                and cfg.get('crop_mode') == 'center', 'original public224 preprocessing differs')
    else:
        require(cfg.get('input_size') == [3, 384, 384] and cfg.get('crop_pct') == 1.
                and cfg.get('crop_mode') == 'squash', 'fixed public384 preprocessing differs')
    readme = (original / 'README.md').read_text()
    require('license: apache-2.0' in readme
            and 'Pretrained on ImageNet-22k and fine-tuned on ImageNet-1k by paper authors.' in readme,
            'fixed-revision model card license/recipe differs')
    records['LICENSE-apache-2.0.txt'] = fetch(LICENSE_URL, original / 'LICENSE-apache-2.0.txt',
                                            transport_file=transport('LICENSE-apache-2.0.txt'))
    license_text = (original / 'LICENSE-apache-2.0.txt').read_text()
    require('Apache License' in license_text and 'Version 2.0, January 2004' in license_text
            and 'END OF TERMS AND CONDITIONS' in license_text, 'official license text incomplete')
    records['pytorch_model.bin'] = fetch(base + '/pytorch_model.bin', original / 'pytorch_model.bin',
                                        expected_sha, expected_size, transport('pytorch_model.bin'))
    receipt = {'schema_version': 1, 'passed': True, 'scope': 'small22k_fixed_revision_download',
               'execution': context, 'model': MODEL, 'repo': repo, 'revision': revision,
               'training_data': TRAINING_DATA, 'files': records,
               'public_pretrained_config': cfg,
               'candidate': {'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6,
                             'stochastic_depth': .1, 'dropout': .2, 'image_size': 512,
                             'source_card_preprocessing_is_not_candidate_preprocessing': True},
               'license': {'identifier': 'apache-2.0', 'declaration': 'fixed-revision model card and registry',
                           'full_text_url': LICENSE_URL, 'redistribution_confirmed': True},
               'validation_source_sha256': {'small_source.py': sha256(__file__),
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
    parser.add_argument('--source-profile', choices=tuple(SOURCE_PROFILES), default='small224')
    args = parser.parse_args()
    try:
        result = download(args.assets, args.transport_dir, args.source_profile)
    except Exception as error:
        write_new_json(args.output, {'schema_version': 1, 'passed': False, 'execution': cpu_context(),
                                    'error_type': type(error).__name__, 'error': str(error)})
        raise
    write_new_json(args.output, result)


if __name__ == '__main__':
    main()

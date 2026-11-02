"""Local-only backbone initialization. No implicit model or dependency downloads."""
import hashlib
import json
from pathlib import Path
import re

DEIT3_VARIANT = 'deit3_small_patch16_384'
DEIT3_FORMAT = 'timm-deit3-native384-state-dict-v1'
DEIT3_LIBRARY = {'version':'1.0.27','source_sha256':{
    'deit.py':'7fa9a4f90fc3fafb09dfa5faa7e5196347eac599f0402190d7c89cbdc346d621',
    'vision_transformer.py':'f0f453b75952698059a1e82fa283a7a9cf2cf2217c33fc2c9cb1efc06947f552'}}

STEM_XY_RULE = {'version':'convnext-small-rgbxy512-v1','channels':['R','G','B','X','Y'],
                'canvas_size':512,'coordinates':'2*k/(size-1)-1',
                'position':'after_augmentation_and_normalization','xy_initialization':'zeros'}

IMAGENET_RESNET = {
    'model': 'resnet50',
    'provenance_model': 'resnet50_imagenet',
    'architecture': 'torchvision-resnet50',
    'sha256': '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca',
    'bytes': 102540417,
    'source_url': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
    'training_data': 'ImageNet-1K',
    'variant': 'IMAGENET1K_V2',
    'format': 'torchvision-1000-state-dict-v1',
    'code_license': 'BSD-3-Clause',
    'full_state_keys': 320,
    'backbone_keys': 318,
    'bn_modules': 53,
    'bn_buffers': 159,
}

DINO_RESNET = {
    'format': 'torchvision-resnet50-backbone-state-dict-v1',
    'sha256': '7a228bfac4d2eb2332c91f93d5b0daaffd93bb2df4a1b8ad2918b2acc156cc68',
    'provenance_sha256': 'fb01cc841bf1beccfc9a2dfacd49390b51e345d6dc1cfa5c637dec54d57c516b',
    'original_sha256': '156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c',
    'revision': '7c446df5b9f45747937fb0d72314eb9f7b66930a',
    'source_url': 'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth',
    'training_data': 'ImageNet self-supervised DINO v1',
}

SMALL_22K_ONLY_FORMAT = 'torchvision-small22k-only-full-head-state-dict-v1'


def validate_small22k_only_provenance(initialization):
    """Bind the accepted complete public K-head state before binary replacement."""
    from .experiment91 import public_source
    source = public_source()
    path, meta_path = Path(initialization['path']), Path(initialization['provenance'])
    if (initialization.get('kind') != 'imagenet'
            or initialization.get('format') != SMALL_22K_ONLY_FORMAT
            or source.get('format') != SMALL_22K_ONLY_FORMAT
            or initialization.get('sha256') != source.get('sha256')
            or not path.is_absolute() or not meta_path.is_absolute()
            or path.is_symlink() or meta_path.is_symlink()
            or not path.is_file() or not meta_path.is_file()
            or _hash(path) != source['sha256']
            or _hash(meta_path) != source['provenance_sha256']):
        raise ValueError('Small 22K-only public asset/provenance bytes differ')
    meta = json.loads(meta_path.read_text())
    expected = {key: source[key] for key in
                ('model', 'format', 'repo', 'tag', 'revision', 'training_data',
                 'num_classes', 'original_sha256', 'sha256')}
    if (any(meta.get(key) != value for key, value in expected.items())
            or meta.get('passed') is not True
            or meta.get('structure') != source['structure']
            or meta.get('redistribution_confirmed') is not True
            or meta.get('public_head_disposition', {}).get('full_K_head_verified') is not True):
        raise ValueError('Small 22K-only complete public provenance differs')
    license_path = (meta_path.parent / meta['license_file']).resolve()
    if license_path.is_symlink() or not license_path.is_file() or not license_path.read_text().strip():
        raise ValueError('Small 22K-only license file absent')
    return {'kind': 'imagenet', 'model': 'convnext_small', 'format': SMALL_22K_ONLY_FORMAT,
            'weights_sha256': source['sha256'], 'provenance_sha256': source['provenance_sha256'],
            'original_sha256': source['original_sha256'], 'source_url': meta['source_url'],
            'training_data': source['training_data'], 'source_commit': source['revision'],
            'num_classes': source['num_classes'], 'final_norm_eps': source['final_norm_eps'],
            'license_sha256': _hash(license_path), 'license_file': meta['license_file'],
            'code_license': 'Apache-2.0', 'weight_license_status': 'public_model_card_apache_2_0'}


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _provenance_text(meta, key):
    value = meta.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('ImageNet ResNet provenance requires '+key)
    return value.strip()


def check_imagenet_provenance(meta, root):
    """Validate the planning-confirmed public-asset provenance verbatim.

    The code license (BSD-3-Clause) is recorded separately.  The weights are
    dataset-derived with unresolved terms: redistribution is neither confirmed
    nor approved, so both must stay explicitly false/null.
    """
    if not isinstance(meta, dict):
        raise ValueError('ImageNet ResNet provenance must be a JSON object')
    for key, expected in (
            ('kind', 'imagenet'),
            ('model', IMAGENET_RESNET['provenance_model']),
            ('format', IMAGENET_RESNET['format']),
            ('sha256', IMAGENET_RESNET['sha256']),
            ('bytes', IMAGENET_RESNET['bytes']),
            ('source_url', IMAGENET_RESNET['source_url']),
            ('training_data', IMAGENET_RESNET['training_data']),
            ('weight_variant', IMAGENET_RESNET['variant']),
            ('code_license', IMAGENET_RESNET['code_license']),
            ('weight_license_status', 'dataset_derived_terms_not_fully_resolved'),
            ('local_research_authorized', True),
            ('competition_redistribution_approved', False),
            ('redistribution_confirmed', None)):
        if key not in meta or meta[key] != expected or type(meta[key]) is not type(expected):
            raise ValueError('ImageNet ResNet provenance differs: '+key)
    for key in ('code_license_url', 'source_registry_url', 'weights_terms_notice_url'):
        value = meta.get(key)
        if not isinstance(value, str) or not value.startswith('https://'):
            raise ValueError('ImageNet ResNet provenance requires an https '+key)
    for key in ('cpu_strict_receipt_sha256', 'download_receipt_sha256', 'license_sha256'):
        value = meta.get(key)
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('ImageNet ResNet provenance '+key+' is not a sha256')
    name = _provenance_text(meta, 'license_file')
    path = Path(root) / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('ImageNet ResNet license file missing: '+name)
    if _hash(path) != meta['license_sha256']:
        raise ValueError('ImageNet ResNet license bytes differ: '+name)
    if not path.read_text(encoding='utf-8', errors='replace').strip():
        raise ValueError('ImageNet ResNet license file is empty: '+name)
    return {'license_file': name, 'license_sha256': meta['license_sha256'],
            'code_license': meta['code_license'], 'code_license_url': meta['code_license_url'],
            'weight_license_status': meta['weight_license_status'],
            'weights_terms_notice_url': meta['weights_terms_notice_url'],
            'competition_redistribution_approved': False, 'redistribution_confirmed': None}


def validate_imagenet_provenance(initialization):
    path, meta_path = Path(initialization['path']), Path(initialization['provenance'])
    if (not path.is_absolute() or not meta_path.is_absolute() or path.is_symlink()
            or meta_path.is_symlink() or not path.is_file() or not meta_path.is_file()):
        raise ValueError('ImageNet ResNet requires fixed local weight and provenance files')
    if initialization.get('sha256') != IMAGENET_RESNET['sha256'] or _hash(path) != IMAGENET_RESNET['sha256']:
        raise ValueError('ImageNet ResNet public weight SHA differs')
    if path.stat().st_size != IMAGENET_RESNET['bytes']:
        raise ValueError('ImageNet ResNet public weight byte count differs')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    proof = check_imagenet_provenance(meta, meta_path.parent)
    if initialization.get('format', IMAGENET_RESNET['format']) != IMAGENET_RESNET['format']:
        raise ValueError('ImageNet ResNet state format differs')
    proof.update(kind='imagenet', weights_sha256=IMAGENET_RESNET['sha256'],
                 provenance_sha256=_hash(meta_path), model=IMAGENET_RESNET['provenance_model'],
                 source_url=IMAGENET_RESNET['source_url'], training_data=IMAGENET_RESNET['training_data'],
                 format=IMAGENET_RESNET['format'], variant=IMAGENET_RESNET['variant'])
    return proof


def validate_dino_resnet_provenance(initialization):
    """Bind the accepted backbone-only asset without importing its source model."""
    path, meta_path = Path(initialization['path']), Path(initialization['provenance'])
    if (initialization.get('kind') != 'imagenet' or initialization.get('format') != DINO_RESNET['format']
            or initialization.get('sha256') != DINO_RESNET['sha256']
            or not path.is_absolute() or not meta_path.is_absolute()
            or path.is_symlink() or meta_path.is_symlink() or not path.is_file() or not meta_path.is_file()
            or _hash(path) != DINO_RESNET['sha256']
            or _hash(meta_path) != DINO_RESNET['provenance_sha256']):
        raise ValueError('DINO R50 backbone asset or provenance identity differs')
    meta = json.loads(meta_path.read_text())
    expected = {'schema_version': 1, 'passed': True, 'model': 'resnet50',
        'format': DINO_RESNET['format'], 'repo': 'facebookresearch/dino',
        'revision': DINO_RESNET['revision'], 'weight_url': DINO_RESNET['source_url'],
        'training_data': DINO_RESNET['training_data'],
        'original_sha256': DINO_RESNET['original_sha256'], 'sha256': DINO_RESNET['sha256'],
        'public_1000_head': False, 'redistribution_confirmed': True,
        'license_file': '../original/LICENSE'}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('DINO R50 provenance fields differ')
    license_path = (meta_path.parent / expected['license_file']).resolve()
    if (not license_path.is_file() or license_path.is_symlink()
            or not license_path.read_text(encoding='utf-8').strip()):
        raise ValueError('DINO R50 source license file absent')
    return {'kind': 'imagenet', 'model': 'resnet50_imagenet',
        'format': DINO_RESNET['format'], 'weights_sha256': DINO_RESNET['sha256'],
        'provenance_sha256': DINO_RESNET['provenance_sha256'],
        'original_sha256': DINO_RESNET['original_sha256'],
        'source_url': DINO_RESNET['source_url'], 'training_data': DINO_RESNET['training_data'],
        'source_commit': DINO_RESNET['revision'], 'public_1000_head': False,
        'license_sha256': _hash(license_path)}


def copy_imagenet_materials(source, job, metadata, origin, destination, provenance=None):
    """Copy the supplied license; never assert redistribution clearance."""
    import shutil
    from .persistence import atomic_json
    existing = metadata.get('materials')
    initial = origin.get('identity', {}).get('initialization', {})
    if provenance is not None:
        proof_path = Path(provenance)
    elif existing:
        proof_path = Path(source) / existing['provenance']
        if not proof_path.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError('exported material path escapes its artifact')
    else:
        proof_path = Path(job['initialization']['provenance'])
    expected_proof = ((existing or {}).get('provenance_sha256') if existing and provenance is None
                      else initial.get('provenance_sha256') or (existing or {}).get('original_provenance_sha256'))
    expected_license = initial.get('license_sha256') or (existing or {}).get('license_sha256')
    if not expected_proof or not expected_license:
        raise ValueError('checkpoint/export lacks frozen provenance and license identity hashes')
    if _hash(proof_path) != expected_proof:
        raise ValueError('provenance hash differs from frozen training/material identity')
    meta = json.loads(proof_path.read_text(encoding='utf-8'))
    if job['initialization'].get('format') == SMALL_22K_ONLY_FORMAT:
        proof = validate_small22k_only_provenance(job['initialization'])
        license_path = (proof_path.parent / meta['license_file']).resolve()
    elif job['initialization'].get('format') == DINO_RESNET['format']:
        proof = validate_dino_resnet_provenance(job['initialization'])
        license_path = (proof_path.parent / meta['license_file']).resolve()
        if _hash(proof_path) != DINO_RESNET['provenance_sha256']:
            raise ValueError('DINO R50 export provenance differs')
        proof = {**proof, 'license_file': meta['license_file'],
                 'code_license': 'Apache-2.0', 'weight_license_status': 'author_statement_covers_weights'}
    else:
        proof = check_imagenet_provenance(meta, proof_path.parent)
        license_path = proof_path.parent / proof['license_file']
    if _hash(license_path) != expected_license:
        raise ValueError('license hash differs from frozen training/material identity')
    folder = Path(destination) / 'materials'
    folder.mkdir()
    shutil.copy2(license_path, folder / 'LICENSE.txt')
    atomic_json(folder / 'provenance.json', dict(meta))
    return {'provenance': 'materials/provenance.json', 'license': 'materials/LICENSE.txt',
            'license_sha256': _hash(folder / 'LICENSE.txt'),
            'code_license': proof['code_license'], 'weight_license_status': proof['weight_license_status'],
            'competition_redistribution_approved': False, 'redistribution_confirmed': None,
            'original_provenance_sha256': (existing or {}).get('original_provenance_sha256', _hash(proof_path)),
            'provenance_sha256': _hash(folder / 'provenance.json')}


def validate_initialization(initialization):
    if initialization.get('format') == DEIT3_FORMAT:
        from .experiment91 import public_source
        source=public_source();path=Path(initialization['path']);meta_path=Path(initialization['provenance'])
        if (initialization.get('kind')!='imagenet' or source.get('model')!=DEIT3_VARIANT
                or not path.is_absolute() or not meta_path.is_absolute() or path.is_symlink() or meta_path.is_symlink()
                or initialization.get('sha256')!=source['sha256'] or _hash(path)!=source['sha256']
                or _hash(meta_path)!=source['provenance_sha256']):
            raise ValueError('fixed native DeiT3 source identity differs')
        meta=json.loads(meta_path.read_text())
        if (any(meta.get(k)!=source[k]for k in ('model','format','revision','source_url','training_data','sha256',
                                               'original_sha256','native_library'))
                or meta['native_library']!=DEIT3_LIBRARY or meta.get('redistribution_confirmed')is not True
                or _hash(meta_path.parent/meta['license_file'])!=meta['license_sha256']):
            raise ValueError('native DeiT3 provenance/license differs')
        return {'kind':'imagenet','model':DEIT3_VARIANT,'format':DEIT3_FORMAT,'weights_sha256':source['sha256'],
                'provenance_sha256':source['provenance_sha256'],'final_norm_eps':1e-6,'internal_norm_eps':1e-6,
                'source_url':source['source_url'],'training_data':source['training_data'],'source_commit':source['revision'],
                'original_sha256':source['original_sha256'],'license_sha256':meta['license_sha256'],
                'native_library':DEIT3_LIBRARY}
    if initialization['kind'] == 'random':
        return {'kind': 'random'}
    if (initialization.get('kind') == 'imagenet'
            and initialization.get('format') == DINO_RESNET['format']):
        return validate_dino_resnet_provenance(initialization)
    if (initialization.get('kind') == 'imagenet'
            and initialization.get('format') == SMALL_22K_ONLY_FORMAT):
        return validate_small22k_only_provenance(initialization)
    if initialization['kind'] == 'retinal_vision_language':
        path, meta_path = Path(initialization['path']), Path(initialization['provenance'])
        fixed='62dbdc8020ac2d0412092d01617ebff7ebfb499420629f03426c3b8ad75758c2'
        if (not path.is_absolute() or not meta_path.is_absolute() or path.is_symlink() or meta_path.is_symlink()
                or initialization.get('sha256')!=fixed or _hash(path)!=fixed):
            raise ValueError('N05 FLAIR visual asset identity differs')
        meta=json.loads(meta_path.read_text())
        expected={'model':'flair_resnet50','kind':'retinal_vision_language',
                  'format':'flair-visual-safetensors-exact-extraction-v1','sha256':fixed,
                  'original_sha256':'050334cd934fa3126435c202c41f493ca222fc3fe3ea9c50cf21f67c792a1440',
                  'tensor_count':318,'license':'Apache-2.0',
                  'source_commit':'d6652d53389ff49e5f73efaccf4246e9de88d1a3'}
        if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('N05 public provenance differs')
        for filename,key in (('LICENSE','license_sha256'),('SOURCE-RECEIPT.json','source_receipt_sha256'),('VISION-EXTRACTION.json','extraction_sha256')):
            target=meta_path.parent/filename
            if target.is_symlink() or _hash(target)!=meta.get(key):raise ValueError('N05 provenance proof differs: '+filename)
        return {'kind':'retinal_vision_language','weights_sha256':fixed,'provenance_sha256':_hash(meta_path),
                'license_sha256':meta['license_sha256'],'model':'flair_resnet50','format':meta['format'],
                'source_commit':meta['source_commit'],'original_sha256':meta['original_sha256'],
                'source_url':meta['source_url'],'pretraining_overlap':'unknown'}
    if initialization['kind'] == 'self_supervised':
        if (initialization.get('format') != 'dinov2-official-backbone-state-dict-v1'
                or initialization.get('sha256') != 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'):
            raise ValueError('DINOv2 initialization identity differs from the N03 freeze')
        path, meta_path = Path(initialization['path']), Path(initialization['provenance'])
        if not path.is_absolute() or not meta_path.is_absolute() or _hash(path) != initialization['sha256']:
            raise ValueError('DINOv2 initialization requires the fixed local weight SHA')
        if _hash(meta_path) != 'e76029ce54fcf519d4006d72ee23d8b35af52dde01d8afa3c21a9ed17beab9dc':
            raise ValueError('DINOv2 frozen provenance SHA differs')
        meta = json.loads(meta_path.read_text())
        if (meta.get('model') != 'dinov2_vits14' or meta.get('training_data') != 'LVD-142M'
                or meta.get('commit') != '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
                or meta.get('sha256') != initialization['sha256']
                or meta.get('redistribution_confirmed') is not True
                or meta.get('license') != 'Apache-2.0'
                or not meta.get('source_url', '').startswith('https://')):
            raise ValueError('DINOv2 source / data / license provenance differs')
        license_path = (meta_path.parent / meta['license_file']).resolve()
        if not license_path.is_file() or _hash(license_path) != meta.get('license_sha256'):
            raise ValueError('DINOv2 license bytes differ')
        return {'kind': 'self_supervised', 'weights_sha256': initialization['sha256'],
                'provenance_sha256': _hash(meta_path), 'license_sha256': _hash(license_path),
                'model': meta['model'], 'source_url': meta['source_url'], 'training_data': meta['training_data'],
                'format': initialization['format'], 'source_commit': meta['commit'],
                'final_norm_eps':1e-6,'internal_norm_eps':1e-6}
    if (initialization['kind'] == 'imagenet'
            and initialization.get('sha256') == IMAGENET_RESNET['sha256']):
        return validate_imagenet_provenance(initialization)
    if initialization.get('format','torchvision-1000-state-dict-v1')!='torchvision-1000-state-dict-v1':
        raise ValueError('Only explicit native TorchVision1000 state_dict format is supported')
    path = Path(initialization['path'])
    if _hash(path) != initialization['sha256']:
        raise ValueError('Public weights SHA256 does not match configuration')
    meta_path = Path(initialization['provenance'])
    meta = json.loads(meta_path.read_text())
    for key in ('model', 'source_url', 'training_data', 'license_file', 'sha256'):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError('Weight provenance requires '+key)
    if meta['sha256'] != initialization['sha256'] or meta.get('redistribution_confirmed') is not True:
        raise ValueError('Weight hash/provenance and redistribution confirmation are required')
    convnext22 = meta['training_data'] == 'ImageNet-22K pretraining -> ImageNet-1K fine-tuning'
    baseclip = (meta['model'] == 'convnext_base' and meta['training_data'] ==
                'LAION-2B CLIP pretraining -> ImageNet-12K -> ImageNet-1K fine-tuning')
    if convnext22 or baseclip:
        # The existing Tiny source remains its exact historical whitelist.
        expected = {'model': 'convnext_tiny', 'format': 'torchvision-1000-state-dict-v1',
            'revision': 'bc48a87f119bae1b06418f5340bfa97ef75d4609',
            'sha256': '1ec1dd0b7045237eafe1c8f1becd58456cc811326c97ec416ae8b2e5ede860ed',
            'source_url': 'https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k/resolve/'
                          'bc48a87f119bae1b06418f5340bfa97ef75d4609/pytorch_model.bin',
            'original_sha256': '8cb65dc650a7fb25eaadbff3eb69559d02f83cdc5afdfb3785d1889cdc3e3518',
            'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6,
            'license_file': '../original/LICENSE-apache-2.0.txt'}
        provenance_sha = '40c14c8bd780d4feb88ae96d7e9def1ef78499daf93ae4412eff78502780233f'
        if meta['model'] in ('convnext_small', 'convnext_base', 'swin_base_patch4_window7_224'):
            from .experiment91 import public_source
            source = public_source()
            expected = {key: source[key] for key in expected}
            if expected['model'] != meta['model']:
                raise ValueError('public backbone provenance requires its pinned scientific source')
            provenance_sha = source['provenance_sha256']
            if baseclip:
                expected['training_data'] = source['training_data']
        if (any(meta.get(key) != value for key, value in expected.items())
                or _hash(meta_path) != provenance_sha
                or initialization.get('kind') != 'imagenet'
                or not path.is_absolute() or not meta_path.is_absolute()
                or path.is_symlink() or meta_path.is_symlink()):
            raise ValueError('Pinned ConvNeXt22K-to-1K source/revision/conversion/LN differs')
    elif meta['model'] == 'swinv2_small_window8_256':
        from .experiment91 import public_source
        source = public_source()
        if (source['model'] != meta['model'] or _hash(meta_path) != source['provenance_sha256']
                or any(meta.get(k) != source[k] for k in ('model','format','revision','source_url','training_data',
                         'original_sha256','sha256','final_norm_eps','internal_norm_eps','license_file'))
                or meta.get('license') != 'MIT' or not path.is_absolute() or not meta_path.is_absolute()
                or path.is_symlink() or meta_path.is_symlink()):
            raise ValueError('Pinned Swin V2 Small source/conversion/provenance differs')
    elif meta['training_data'] != 'ImageNet-1K' or not meta['source_url'].startswith('https://'):
        raise ValueError('Only documented ImageNet-1K or the pinned Tiny22K-to-1K source is supported')
    license_path = (meta_path.parent / meta['license_file']).resolve()
    if not license_path.is_file() or not license_path.read_text().strip():
        raise ValueError('Provide the applicable full weight license as license_file')
    if baseclip and _hash(license_path) != meta['source_file_hashes']['LICENSE-apache-2.0.txt']:
        raise ValueError('Pinned CLIP Base license bytes differ')
    if meta['model'] == 'swinv2_small_window8_256' and _hash(license_path) != source['license_sha256']:
        raise ValueError('Pinned Swin V2 license bytes differ')
    return {'kind': 'imagenet', 'weights_sha256': initialization['sha256'],
            'provenance_sha256': _hash(meta_path), 'license_sha256': _hash(license_path),
            'model': meta['model'], 'source_url': meta['source_url'], 'training_data': meta['training_data'],
            'format':initialization.get('format','torchvision-1000-state-dict-v1'),
            'original_sha256':meta.get('original_sha256'),'final_norm_eps':meta.get('final_norm_eps')}


def apply_feature_freeze(model, job):
    """Freeze the registered Small early features without changing mode or RNG."""
    if 'freeze_features_before' not in job:
        return ()
    if (job.get('model') != 'convnext_small' or type(job['freeze_features_before']) is not int
            or job['freeze_features_before'] != 4):
        raise ValueError('feature freezing supports only explicit Small features[:4]')
    selected = {id(p) for module in model.network.features[:job['freeze_features_before']]
                for p in module.parameters()}
    names = []
    for name, parameter in model.named_parameters():
        if id(parameter) in selected:
            parameter.requires_grad_(False)
            names.append(name)
    if not names:
        raise ValueError('registered early features contain no parameters')
    return tuple(sorted(names))


def feature_freeze_proof(model, job, optimizer):
    """Read the real online mask and its optimizer membership for initial proof."""
    if job.get('low_rank_policy') is not None:
        from .low_rank import low_rank_proof
        return low_rank_proof(model, job, optimizer)
    if 'freeze_features_before' not in job:
        return None
    selected = {id(p) for module in model.network.features[:job['freeze_features_before']]
                for p in module.parameters()}
    named = dict(model.named_parameters())
    expected = sorted(name for name, p in named.items() if id(p) in selected)
    frozen = sorted(name for name, p in named.items() if not p.requires_grad)
    trainable = sorted(name for name, p in named.items() if p.requires_grad)
    names = {id(p): name for name, p in named.items()}
    members = [names[id(p)] for group in optimizer.param_groups for p in group['params']]
    if not frozen or frozen != expected or sorted(members) != trainable or len(members) != len(set(members)):
        raise ValueError('actual early-feature mask or optimizer membership differs')
    return {'freeze_features_before': job['freeze_features_before'],
        'all_parameter_names': sorted(named), 'frozen_parameter_names': frozen,
        'trainable_parameter_names': trainable, 'optimizer_parameter_names': sorted(members),
        'all_parameter_elements': sum(p.numel() for p in named.values()),
        'frozen_parameter_elements': sum(named[name].numel() for name in frozen),
        'trainable_parameter_elements': sum(named[name].numel() for name in trainable)}


def _deit3_library():
    import inspect,timm
    import timm.models.deit as deit
    from timm.models.vision_transformer import VisionTransformer
    actual={'version':timm.__version__,'source_sha256':{
        'deit.py':_hash(inspect.getfile(deit)),
        'vision_transformer.py':_hash(inspect.getfile(VisionTransformer))}}
    if actual!=DEIT3_LIBRARY:raise ValueError('installed native DeiT3 library differs')
    return actual


def _create_deit3_binary(job, initialize=True):
    import torch
    from torch import nn
    from timm.models.vision_transformer import VisionTransformer
    from .ema import tensor_digest
    library=_deit3_library()
    class DeiTNetwork(VisionTransformer):
        def forward_head(self, x, pre_logits=False):
            features=super().forward_head(x,pre_logits=True)
            return features if pre_logits else self.classifier(features)
    network=DeiTNetwork(img_size=384,patch_size=16,embed_dim=384,depth=12,num_heads=6,
        no_embed_class=True,init_values=1e-6,num_classes=1000,drop_rate=0.,
        drop_path_rate=job.get('stochastic_depth',.1),device='cpu',dtype=torch.float32)
    public=None
    if initialize:
        proof=validate_initialization(job['initialization'])
        state=torch.load(job['initialization']['path'],map_location='cpu',weights_only=True)
        loaded=network.load_state_dict(state,strict=True);actual=network.state_dict()
        if (len(actual)!=176 or set(actual)!=set(state) or loaded.missing_keys or loaded.unexpected_keys
                or network.head.weight.shape!=(1000,384)
                or any(t.dtype!=torch.float32 or not bool(torch.isfinite(t).all()) or not torch.equal(actual[n],t)
                       for n,t in state.items())):
            raise ValueError('DeiT3 complete public384 strict/exact/FP32 proof differs')
        public={'initialization':proof,'strict_missing_keys':[],'strict_unexpected_keys':[],
            'public_tensor_count':176,'public_tensor_sha256':tensor_digest(state),'full_head_shape':[1000,384],
            'all_public_tensors_exact':True,'all_public_tensors_FP32_finite':True,
            'scope':'public384_before_position_resampling_and_binary_head'}
    network.set_input_size(img_size=512)
    pool_policy=job.get('deit_pool_policy')
    if pool_policy is not None:
        from .config import DEIT3_MEANPATCH_RULE
        if pool_policy!=DEIT3_MEANPATCH_RULE:raise ValueError('native DeiT3 meanpatch policy differs')
        network.global_pool='avg';network.pool_include_prefix=False
    network.head=nn.Identity();network.num_classes=1
    network.classifier=nn.Sequential(nn.Dropout(job.get('dropout',.2)),nn.Linear(384,1))
    class DeiTBinary(nn.Module):
        def __init__(self):
            super().__init__();self.network=network;self.native_library=library
            if pool_policy is not None:self.deit_pool_policy=dict(pool_policy)
        def forward(self, images, age_features=None):
            logits=self.network(images).flatten()
            if hasattr(self,'age_weight'):
                from .age import residual
                return residual(self,logits,age_features)
            return logits
    result=DeiTBinary()
    if job.get('age')is not None:
        from .age import attach_age
        attach_age(result,job['age'])
    if public is not None:
        public['derived_position']={'source_grid':[24,24],'runtime_grid':[32,32],
            'method':'native_set_input_size_bicubic_antialias_no_prefix',
            'tensor_sha256':tensor_digest({'pos_embed':network.pos_embed})}
        result.public_initialization_receipt=public
    return result


def describe_deit3(model):
    import torch
    nn=torch.nn;network=model.network;head=network.classifier[-1]
    policy=getattr(model,'deit_pool_policy',None)
    if policy is not None:
        from .config import DEIT3_MEANPATCH_RULE
        if policy!=DEIT3_MEANPATCH_RULE:raise ValueError('native DeiT3 saved pool policy differs')
    value={'native_library':model.native_library,'source_size':384,'image_size':list(network.patch_embed.img_size),
        'patch_size':list(network.patch_embed.patch_size),'grid_size':list(network.patch_embed.grid_size),
        'patch_weight_shape':list(network.patch_embed.proj.weight.shape),'pos_embed_shape':list(network.pos_embed.shape),
        'cls_token_shape':list(network.cls_token.shape),'no_embed_class':network.no_embed_class,
        'num_prefix_tokens':network.num_prefix_tokens,'global_pool':network.global_pool,'head_drop':network.head_drop.p,
        'head_shape':list(head.weight.shape),'head_dropout':network.classifier[0].p,
        'blocks':[{'heads':b.attn.num_heads,'qkv_shape':list(b.attn.qkv.weight.shape),'fused_attn':b.attn.fused_attn,
            'drop_path1':getattr(b.drop_path1,'drop_prob',0.),'drop_path2':getattr(b.drop_path2,'drop_prob',0.),
            'scale1':list(b.ls1.gamma.shape),'scale2':list(b.ls2.gamma.shape)}for b in network.blocks]}
    if (value['native_library']!=DEIT3_LIBRARY or value['image_size']!=[512,512] or value['patch_size']!=[16,16]
            or value['grid_size']!=[32,32] or value['patch_weight_shape']!=[384,3,16,16]
            or value['pos_embed_shape']!=[1,1024,384] or value['cls_token_shape']!=[1,1,384]
            or value['no_embed_class']is not True or value['num_prefix_tokens']!=1 or value['global_pool']!=('avg'if policy is not None else'token')
            or value['head_drop']!=0. or value['head_shape']!=[1,384] or value['head_dropout']!=.2
            or len(value['blocks'])!=12 or any(b['heads']!=6 or b['qkv_shape']!=[1152,384]
                or b['fused_attn']is not True or b['scale1']!=[384] or b['scale2']!=[384]for b in value['blocks'])
            or not isinstance(network.head,nn.Identity) or not isinstance(head,nn.Linear)
            or (policy is not None and (network.pool_include_prefix is not False or not isinstance(network.fc_norm,nn.Identity)))):
        raise ValueError('native DeiT3 binary512 graph differs')
    if policy is not None:value.update(pool_policy=dict(policy),pool_include_prefix=network.pool_include_prefix)
    return value


def _cosine_linear(linear):
    """Reuse the native parameters without advancing its initialization RNG."""
    import torch
    from .config import COSINE_HEAD_RULE
    class CosineLinear(torch.nn.Linear):
        def __init__(self, original):
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = original.in_features, original.out_features
            self.weight, self.bias = original.weight, original.bias
            self.cosine_head_spec = dict(COSINE_HEAD_RULE)
            self.train(original.training)
        def forward(self, inputs):
            from torch.nn import functional as F
            spec = self.cosine_head_spec
            return F.linear(F.normalize(inputs, dim=-1, eps=spec['eps']) * spec['scale'],
                            F.normalize(self.weight, dim=1, eps=spec['eps']), self.bias)
    return CosineLinear(linear)


def describe_cosine_head(model, job):
    from .config import COSINE_HEAD_RULE
    head = model.network.classifier[-1]
    if (job.get('cosine_head_policy') != COSINE_HEAD_RULE
            or getattr(head, 'cosine_head_spec', None) != COSINE_HEAD_RULE
            or type(head).__name__ != 'CosineLinear'
            or (head.in_features, head.out_features) != (768, 1)):
        raise ValueError('saved cosine16 head graph differs')
    return {'policy':dict(COSINE_HEAD_RULE), 'module':'CosineLinear',
            'in_features':768, 'out_features':1}


def create_model(job, initialize=True):
    if job.get('experiment91') is not None:
        from .experiment91 import validate_job
        validate_job(job)
    import torch
    from torch import nn
    if job['model'] == 'flair_resnet50':
        from .flair_model import create_flair_binary
        return create_flair_binary(job, initialize=initialize)
    if job['model'] == 'resnet50_imagenet':
        from .imagenet_resnet import create_imagenet_binary
        return create_imagenet_binary(job, initialize=initialize)
    if job['model'] == 'dinov2_vits14':
        from .dinov2_model import create_dinov2_binary
        return create_dinov2_binary(job, initialize=initialize)
    if job['model'] in ('swin_base', 'swin_v2_small'):
        from .swin_model import create_swin_binary
        return create_swin_binary(job, initialize=initialize)
    if job['model'] == 'deit3_small':
        return _create_deit3_binary(job, initialize=initialize)
    from torchvision import models
    name = job['model']
    kwargs = {'weights': None, 'stochastic_depth_prob': job.get('stochastic_depth', .1)}
    if name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(dropout=job.get('dropout', .2), **kwargs)
    elif name in ('convnext_tiny', 'convnext_small', 'convnext_base'):
        factory = {'convnext_tiny': models.convnext_tiny,
                   'convnext_small': models.convnext_small,
                   'convnext_base': models.convnext_base}[name]
        model = factory(**kwargs)
        model.classifier[0].eps=float(job.get('final_norm_eps',1e-6))
    else:
        raise ValueError('Unsupported backbone: '+name)
    public_receipt = None
    init = job.get('initialization', {'kind': 'random'})
    if initialize and init['kind'] != 'random':
        proof = validate_initialization(init)
        if proof['model'] != name:
            raise ValueError('Pretrained provenance model differs from requested architecture')
        if proof.get('final_norm_eps') is not None and float(job.get('final_norm_eps',1e-6))!=proof['final_norm_eps']:
            raise ValueError('Published final LayerNorm epsilon differs from the scientific job contract')
        state = torch.load(init['path'], map_location='cpu', weights_only=True)
        small22k_only = init.get('format') == SMALL_22K_ONLY_FORMAT
        if small22k_only:
            # The native factory already constructed its 1000 head.  A temporary
            # public K head must not advance the binary head/age RNG stream.
            with torch.random.fork_rng(devices=[]):
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                                 proof['num_classes'])
                loaded = model.load_state_dict(state, strict=True)
        else:
            loaded = model.load_state_dict(state, strict=True)
        if job.get('experiment91') is not None:
            from .ema import tensor_digest
            from .experiment91 import public_source
            actual = model.state_dict()
            if (proof['provenance_sha256'] != public_source()['provenance_sha256'] or loaded.missing_keys or loaded.unexpected_keys
                    or actual.keys() != state.keys()
                    or model.classifier[-1].out_features != (proof['num_classes'] if small22k_only else 1000)
                    or any(not torch.equal(actual[n], t) or t.dtype != torch.float32
                           or not bool(torch.isfinite(t).all()) for n, t in state.items())):
                raise ValueError('ConvNeXt complete public1000 strict/exact/FP32/finite proof differs')
            public_receipt = {'initialization': proof, 'strict_missing_keys': loaded.missing_keys,
                'strict_unexpected_keys': loaded.unexpected_keys, 'public_tensor_count': len(state),
                'public_tensor_sha256': tensor_digest(state), 'full_head_shape': list(model.classifier[-1].weight.shape),
                'all_public_tensors_exact': True, 'all_public_tensors_FP32_finite': True}
    # Keep native names so optimizer grouping and raw checkpoint state_dicts remain simple.
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, 1)
    if name in ('convnext_tiny', 'convnext_small', 'convnext_base'):
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1],
                                         nn.Dropout(job.get('dropout', .2)), model.classifier[-1])
    if job.get('cosine_head_policy') is not None:
        from .config import COSINE_HEAD_RULE
        if name != 'convnext_small' or job['cosine_head_policy'] != COSINE_HEAD_RULE:
            raise ValueError('cosine16 supports only the fixed native Small head')
        model.classifier[-1] = _cosine_linear(model.classifier[-1])
    class BinaryModel(nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network
        def forward(self, images, age_features=None, *, mixstyle_context=None):
            if getattr(self, 'stem_xy_policy', None) is not None:
                if images.ndim != 4 or images.shape[1:] != (3,512,512) or images.dtype != torch.float32:
                    raise ValueError('RGBXY requires the final FP32 RGB 512 canvas')
                coordinate=2*torch.arange(512,device=images.device,dtype=torch.float32)/511-1
                images=torch.cat((images,coordinate.view(1,1,1,512).expand(images.shape[0],1,512,512),
                                  coordinate.view(1,1,512,1).expand(images.shape[0],1,512,512)),dim=1)
            if self.training and getattr(self, 'mixstyle_policy', None) is not None:
                from .mixstyle import training_image_logits
                logits = training_image_logits(self, images, mixstyle_context)
            elif hasattr(self, 'attention_pool_weight'):
                from .attention_pool import image_logits
                logits = image_logits(self, images)
            elif hasattr(self, 'stage3_residual_weight'):
                from .stage3_residual import image_logits
                logits = image_logits(self, images)
            else:
                logits = self.network(images).flatten()
            if hasattr(self, 'age_weight'):
                from .age import residual
                return residual(self, logits, age_features)
            return logits
    result = BinaryModel(model)
    if job.get('age') is not None:
        from .age import attach_age
        attach_age(result, job['age'])
    if job.get('stem_xy_policy') is not None:
        stem=model.features[0][0]
        if (job['model']!='convnext_small' or job['stem_xy_policy']!=STEM_XY_RULE
                or not isinstance(stem,nn.Conv2d)
                or (stem.in_channels,stem.out_channels,stem.kernel_size,stem.stride,stem.padding,stem.dilation,stem.groups)
                    != (3,96,(4,4),(4,4),(0,0),(1,1),1)):
            raise ValueError('RGBXY supports only the existing native Small RGB stem')
        from .ema import tensor_digest
        rgb_before=tensor_digest({'rgb':stem.weight});bias_before=tensor_digest({'bias':stem.bias})
        with torch.no_grad():
            stem.weight.data=torch.cat((stem.weight.detach(),stem.weight.new_zeros(96,2,4,4)),dim=1)
        stem.in_channels=5
        result.stem_xy_policy={**STEM_XY_RULE,'channels':list(STEM_XY_RULE['channels'])}
        if public_receipt is not None:
            public_receipt['stem_xy_initialization']={'rgb_before':rgb_before,
                'rgb_after':tensor_digest({'rgb':stem.weight[:,:3]}),'bias_before':bias_before,
                'bias_after':tensor_digest({'bias':stem.bias}),'added_zero_elements':int(stem.weight[:,3:].numel())}
    from .config import GLOBAL_NATIVE_SIX_RULE
    if job.get('input_preprocessing')==GLOBAL_NATIVE_SIX_RULE:
        from copy import deepcopy
        from .ema import tensor_digest
        stem=model.features[0][0]
        if (name!='convnext_small' or not isinstance(stem,nn.Conv2d)
                or (stem.in_channels,stem.out_channels,stem.kernel_size,stem.stride,stem.padding,stem.dilation,stem.groups)
                    !=(3,96,(4,4),(4,4),(0,0),(1,1),1)):
            raise ValueError('native six-channel input requires the original Small RGB stem')
        rgb_before=tensor_digest({'rgb':stem.weight});bias_before=tensor_digest({'bias':stem.bias})
        with torch.no_grad():stem.weight.data=torch.cat((stem.weight.detach(),stem.weight.new_zeros(96,3,4,4)),dim=1)
        stem.in_channels=6;result.stem_native6_policy=deepcopy(GLOBAL_NATIVE_SIX_RULE)
        if public_receipt is not None:
            public_receipt['stem_native6_initialization']={'rgb_before':rgb_before,'rgb_after':tensor_digest({'rgb':stem.weight[:,:3]}),
                'bias_before':bias_before,'bias_after':tensor_digest({'bias':stem.bias}),'added_zero_elements':int(stem.weight[:,3:].numel())}
    if job.get('mixstyle_policy') is not None:
        from .mixstyle import attach_mixstyle
        attach_mixstyle(result, job)
    if job.get('tri3_downsample_policy') is not None:
        from .tri3_downsample import attach_tri3
        attach_tri3(result, job)
    if job.get('attention_pool') is not None:
        from .attention_pool import attach
        attach(result, job)
    if job.get('stage3_residual') is not None:
        from .stage3_residual import attach
        attach(result, job)
    from .spatial_stride import apply_final_downsample_stride
    apply_final_downsample_stride(result, job)
    apply_feature_freeze(result, job)
    if job.get('low_rank_policy') is not None:
        from .low_rank import attach_low_rank
        attach_low_rank(result, job)
    if job.get('experiment91') is not None:
        from .experiment91 import describe_model
        result.candidate_normalization = describe_model(result)
        result.public_initialization_receipt = public_receipt
    return result

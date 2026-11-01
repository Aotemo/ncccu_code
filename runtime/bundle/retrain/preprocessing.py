"""Versioned deterministic preparation and disposable local RGB caches.

Augmentation remains outside this cache. Original file state is checked even
on a hit; changed files must pass the original sealed SHA check again.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION

from .verification import (VerificationCache, _atomic_json, configured_cache,
                           lexical_path, source_fingerprint)

PREPARATION_VERSION = 'rgb-conservative-border-prepared-v2'


def bright_center_box(image, policy):
    """Fixed bright-region heuristic, expressed in the input RGB pixel space."""
    from PIL import ImageFilter
    from .config import BRIGHT_CENTER_RULE,NATIVE_BRIGHT_CENTER_RULE
    if policy not in (BRIGHT_CENTER_RULE,NATIVE_BRIGHT_CENTER_RULE) or image.mode!='RGB':
        raise ValueError('fixed RGB bright-center crop policy differs')
    width,height=image.size
    ratio=min(1.,policy['thumbnail_size']/max(width,height))
    thumbnail=image.resize((max(1,round(width*ratio)),max(1,round(height*ratio))),Image.Resampling.BILINEAR) if ratio<1 else image
    green=np.asarray(thumbnail.getchannel(policy['channel']).filter(ImageFilter.GaussianBlur(policy['blur_radius'])))
    tw,th=thumbnail.size;mx=int(tw*policy['margin_fraction']);my=int(th*policy['margin_fraction'])
    scores=green[my:th-my,mx:tw-mx]
    if scores.max()==scores.min():cx,cy=width/2,height/2
    else:
        y,x=np.unravel_index(int(scores.argmax()),scores.shape)
        cx=(x+mx+.5)*width/tw;cy=(y+my+.5)*height/th
    side=max(1,round(min(width,height)*policy['crop_fraction']))
    left=max(0,min(width-side,round(cx-side/2)));top=max(0,min(height-side,round(cy-side/2)))
    return left,top,left+side,top+side


def bright_center_crop(image, policy):
    return image.crop(bright_center_box(image,policy))


def global_native_inset(global_image, native_image, policy):
    from .config import GLOBAL_NATIVE_INSET_RULE
    if policy!=GLOBAL_NATIVE_INSET_RULE or global_image.mode!='RGB' or native_image.mode!='RGB':
        raise ValueError('fixed global/native inset policy differs')
    size=policy['global_size'];ratio=size/max(global_image.size)
    target=(max(1,round(global_image.width*ratio)),max(1,round(global_image.height*ratio)))
    image=Image.new('RGB',(size,size))
    image.paste(global_image.resize(target,Image.Resampling.BILINEAR),(round((size-target[0])/2),round((size-target[1])/2)))
    image.paste(native_image.resize((policy['inset_size'],policy['inset_size']),Image.Resampling.BILINEAR),tuple(policy['corner']))
    return image


def verify_composite_cache_asset(science, site):
    from .config import GLOBAL_NATIVE_INSET_RULE,GLOBAL_NATIVE_SIX_RULE
    policy=science.get('job',{}).get('input_preprocessing')
    if policy not in (GLOBAL_NATIVE_INSET_RULE,GLOBAL_NATIVE_SIX_RULE):return None
    six=policy==GLOBAL_NATIVE_SIX_RULE
    root=Path(site['work_root']).resolve(strict=True);ref=science.get('composite_cache_receipt',{});relative=Path(ref.get('path',''))
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:raise ValueError('explicit relative paired-cache receipt required')
    path=root/relative
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=ref.get('sha256'):raise ValueError('paired-cache receipt SHA differs')
    value=json.loads(path.read_text())
    expected={'format':'global-native-six-cache-pair-v1' if six else 'global-native-inset-cache-pair-v1','passed':True,'pod_uid':site['pod']['uid'],
        'data_root':site['data_root'],'metadata_root':site['metadata_root'],'native_cache_root':site['prepared_cache'],
        'manifest_sha256':hashlib.sha256((Path(site['metadata_root'])/'manifest.json').read_bytes()).hexdigest(),
        'fit_count':46256,'dev_count':5140,'row_count':51396,'verified_readonly_count_each':51396,
        'input_preprocessing':policy,'pillow':PILLOW_VERSION,'official_read':False}
    if not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected.items()):raise ValueError('paired-cache identity or complete coverage differs')
    global_root=Path(value.get('global_cache_root',''))
    if not global_root.is_absolute() or not global_root.resolve().is_relative_to(root) or str(global_root)==site['prepared_cache']:raise ValueError('explicit separate global cache required')
    index=value.get('index',{});index_path=Path(index.get('path',''))
    if not index_path.is_absolute() or not index_path.resolve().is_relative_to(root) or index_path.is_symlink() or hashlib.sha256(index_path.read_bytes()).hexdigest()!=index.get('sha256'):raise ValueError('paired-cache coverage index differs')
    if six:
        base_ref=value.get('base_pair_receipt',{});base_relative=Path(base_ref.get('path',''))
        if base_relative.is_absolute() or '..' in base_relative.parts or not base_relative.parts:raise ValueError('explicit original paired-cache reference required')
        base_path=root/base_relative
        if base_path.is_symlink() or hashlib.sha256(base_path.read_bytes()).hexdigest()!=base_ref.get('sha256'):raise ValueError('original paired-cache proof SHA differs')
        base=json.loads(base_path.read_text())
        source=Path(__file__).resolve().parents[3]
        if (base.get('format')!='global-native-inset-cache-pair-v1' or base.get('input_preprocessing')!=GLOBAL_NATIVE_INSET_RULE
                or any(base.get(k)!=value.get(k) for k in (*[k for k in expected if k not in ('format','input_preprocessing')],'global_cache_root','index'))
                or value.get('cache_core_AST_unchanged') is not True
                or value.get('source_files')!={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                    for name in ('runtime/bundle/retrain/data.py','runtime/bundle/retrain/preprocessing.py')}):
            raise ValueError('six-channel asset reuse or unchanged cache core proof differs')
    return value


def verify_native_cache_asset(science, site):
    """Bind a completed derived fit/dev cache without rewriting migration proof."""
    from .config import NATIVE_BRIGHT_CENTER_RULE
    if science.get('job',{}).get('input_preprocessing') != NATIVE_BRIGHT_CENTER_RULE:
        return None
    root = Path(site['work_root']).resolve(strict=True)
    ref = science.get('prepared_cache_receipt',{})
    relative = Path(ref.get('path',''))
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('explicit relative native cache receipt required')
    path = root / relative
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != ref.get('sha256'):
        raise ValueError('native cache receipt SHA differs')
    value = json.loads(path.read_text())
    expected = {'format':'native-bright-prepared-cache-v1','passed':True,'pod_uid':site['pod']['uid'],
        'cache_root':site['prepared_cache'],'data_root':site['data_root'],'metadata_root':site['metadata_root'],
        'manifest_sha256':hashlib.sha256((Path(site['metadata_root'])/'manifest.json').read_bytes()).hexdigest(),
        'fit_count':46256,'dev_count':5140,'row_count':51396,'verified_readonly_count':51396,
        'input_preprocessing':NATIVE_BRIGHT_CENTER_RULE,'prepared_size':512,'pillow':PILLOW_VERSION,
        'official_read':False}
    if not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('native cache completion, policy or identity differs')
    source = Path(__file__).resolve().parents[3]
    implementations = ('runtime/bundle/retrain/data.py','runtime/bundle/retrain/preprocessing.py')
    if value.get('source_files') != {n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in implementations}:
        raise ValueError('native cache implementation differs')
    index = value.get('index',{});index_path = Path(index.get('path',''))
    if (not index_path.is_absolute() or not index_path.resolve().is_relative_to(root)
            or index_path.is_symlink() or hashlib.sha256(index_path.read_bytes()).hexdigest()!=index.get('sha256')):
        raise ValueError('native cache coverage index differs')
    return value


def prepare_image(image, prepared_size, input_preprocessing=None):
    from .data import _crop_black_border
    if isinstance(prepared_size, bool) or not isinstance(prepared_size, int) or prepared_size < 1:
        raise ValueError('prepared_size must be a positive integer')
    image = _crop_black_border(ImageOps.exif_transpose(image).convert('RGB'))
    if input_preprocessing is not None:
        from .config import NATIVE_BRIGHT_CENTER_RULE
        if input_preprocessing != NATIVE_BRIGHT_CENTER_RULE:
            raise ValueError('native preparation policy differs')
        image = bright_center_crop(image,input_preprocessing)
    ratio = min(1.0, prepared_size / max(image.size))
    if ratio < 1:
        image = image.resize((max(1, round(image.width * ratio)),
                              max(1, round(image.height * ratio))), Image.Resampling.BILINEAR)
    return image


class PreparedImageCache:
    def __init__(self, data_root, cache_root, prepared_size, input_preprocessing=None):
        if isinstance(prepared_size, bool) or not isinstance(prepared_size, int) or prepared_size < 1:
            raise ValueError('prepared_size must be a positive integer')
        self.root = Path(data_root).resolve()
        self.prepared_size = prepared_size
        self.input_preprocessing = input_preprocessing
        mode = os.environ.get('TRAIN_IMAGE_CACHE_READONLY','0')
        if mode not in ('0','1'):
            raise ValueError('TRAIN_IMAGE_CACHE_READONLY must be 0 or 1')
        self.readonly = mode == '1'
        self.policy = {'version': PREPARATION_VERSION, 'prepared_size': prepared_size,
                       'pillow': PILLOW_VERSION, 'dataset_root': str(self.root)}
        if input_preprocessing is not None:
            from .config import NATIVE_BRIGHT_CENTER_RULE
            if input_preprocessing != NATIVE_BRIGHT_CENTER_RULE:
                raise ValueError('native prepared cache policy differs')
            self.policy['input_preprocessing'] = input_preprocessing
        namespace = hashlib.sha256(json.dumps(self.policy, sort_keys=True).encode()).hexdigest()
        self.directory = Path(cache_root) / namespace
        self.verifier = configured_cache(self.root) or VerificationCache(
            self.root, Path(cache_root).resolve() / '.verification')

    def read(self, relative, expected_sha):
        lexical_path(relative)
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha)):
            raise ValueError('a full sealed SHA256 is required for prepared images')
        fingerprint = source_fingerprint(self.root, relative)
        key = hashlib.sha256((relative + '\0' + expected_sha).encode()).hexdigest()
        directory = self.directory / key[:2]
        array_path, receipt_path = directory / (key + '.npy'), directory / (key + '.json')
        expected = {'policy': self.policy, 'relative_path': relative,
                    'source_sha256': expected_sha, 'source_file': fingerprint}
        try:
            if receipt_path.is_symlink() or array_path.is_symlink():
                raise ValueError('prepared cache entries cannot be symlinks')
            if receipt_path.stat().st_size > 16384:
                raise ValueError('oversized prepared receipt')
            receipt = json.loads(receipt_path.read_text())
            if not isinstance(receipt, dict):
                raise ValueError('prepared receipt must be an object')
            if {k: receipt.get(k) for k in expected} != expected:
                raise ValueError('prepared cache identity changed')
            if array_path.stat().st_size > self.prepared_size ** 2 * 3 + 16384:
                raise ValueError('oversized prepared image')
            raw = array_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt['array_sha256']:
                raise ValueError('prepared cache checksum changed')
            array = np.load(io.BytesIO(raw), allow_pickle=False)
            if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
                    or min(array.shape[:2]) < 1 or max(array.shape[:2]) > self.prepared_size):
                raise ValueError('invalid prepared image shape/dtype')
            if source_fingerprint(self.root, relative) != fingerprint:
                raise ValueError('source changed while reading prepared image')
            return Image.fromarray(array)
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            if self.readonly:
                raise ValueError('read-only prepared cache missing, corrupt or source identity changed: '+relative)
            # Every miss, damaged entry or changed file returns to sealed bytes.
            raw = self.verifier.read_verified(relative, expected_sha)
            if source_fingerprint(self.root, relative) != fingerprint:
                raise ValueError('source changed while preparing image')
            with Image.open(io.BytesIO(raw)) as original:
                image = prepare_image(original, self.prepared_size,self.input_preprocessing)
            array = np.asarray(image).copy()
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            encoded = buffer.getvalue()
            directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.' + key + '.', suffix='.tmp', dir=directory)
            try:
                os.fchmod(fd,0o644)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, array_path)
                _atomic_json(receipt_path, dict(expected,
                    array_sha256=hashlib.sha256(encoded).hexdigest()))
            finally:
                Path(temporary).unlink(missing_ok=True)
            return image

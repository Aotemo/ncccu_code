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
    from .config import BRIGHT_CENTER_RULE
    if policy != BRIGHT_CENTER_RULE or image.mode!='RGB':
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


def prepare_image(image, prepared_size):
    from .data import _crop_black_border
    if isinstance(prepared_size, bool) or not isinstance(prepared_size, int) or prepared_size < 1:
        raise ValueError('prepared_size must be a positive integer')
    image = _crop_black_border(ImageOps.exif_transpose(image).convert('RGB'))
    ratio = min(1.0, prepared_size / max(image.size))
    if ratio < 1:
        image = image.resize((max(1, round(image.width * ratio)),
                              max(1, round(image.height * ratio))), Image.Resampling.BILINEAR)
    return image


class PreparedImageCache:
    def __init__(self, data_root, cache_root, prepared_size):
        if isinstance(prepared_size, bool) or not isinstance(prepared_size, int) or prepared_size < 1:
            raise ValueError('prepared_size must be a positive integer')
        self.root = Path(data_root).resolve()
        self.prepared_size = prepared_size
        mode = os.environ.get('TRAIN_IMAGE_CACHE_READONLY','0')
        if mode not in ('0','1'):
            raise ValueError('TRAIN_IMAGE_CACHE_READONLY must be 0 or 1')
        self.readonly = mode == '1'
        self.policy = {'version': PREPARATION_VERSION, 'prepared_size': prepared_size,
                       'pillow': PILLOW_VERSION, 'dataset_root': str(self.root)}
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
                image = prepare_image(original, self.prepared_size)
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

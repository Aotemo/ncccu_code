"""Durable, on-access verification for trusted persistent dataset storage.

Only manifest metadata is checked at startup. Each sampled image is checked
before decoding, and successful SHA256 checks survive processes and containers.
Cache reuse assumes the filesystem reports honest inode/size/mtime/ctime;
this is not a defense against an actor who can forge filesystem metadata.
"""
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile


def _bytes_sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def lexical_path(relative, split=None):
    """Validate a sealed relative name without accessing the image filesystem."""
    if not isinstance(relative, str) or any(c in relative for c in ('\\', '\x00', ':')):
        raise ValueError(f'unsafe relative path: {relative!r}')
    path = PurePosixPath(relative)
    if (len(path.parts) != 2 or path.parts[0] not in ('train', 'valid', 'test')
            or any(part in ('.', '..') for part in path.parts)
            or path.as_posix() != relative or (split is not None and path.parts[0] != split)):
        raise ValueError(f'unsafe {split or "image"} relative path: {relative!r}')
    return path


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        os.fchmod(fd,0o644)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fingerprint(info):
    # A container remount can change st_dev even for the same persistent file.
    # The canonical dataset namespace and inode identify that file instead.
    return {'inode': info.st_ino, 'size': info.st_size,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def source_fingerprint(data_root, relative):
    """Safely inspect an original without reading its bytes on prepared hits."""
    path = lexical_path(relative)
    descriptors = []
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        root = os.open(data_root, flags)
        descriptors.append(root)
        split = os.open(path.parts[0], flags, dir_fd=root)
        descriptors.append(split)
        image = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                        dir_fd=split)
        descriptors.append(image)
        info = os.fstat(image)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('unsafe non-regular image path: ' + relative)
        return _fingerprint(info)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError('unsafe image path or symlink: ' + relative) from exc
        raise
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


class VerificationCache:
    def __init__(self, data_root, cache_root):
        directory = Path(cache_root)
        if not directory.is_absolute():
            raise ValueError('TRAIN_VERIFICATION_CACHE must be an absolute persistent directory')
        self.root = Path(data_root).resolve()
        root_key = hashlib.sha256(os.fsencode(self.root)).hexdigest()
        self.directory = directory / root_key

    def record_manifest(self, identity):
        _atomic_json(self.directory / 'manifest.json', {
            'schema_version': 1, 'mode': 'on-access', 'status': 'manifest-validated',
            'dataset_root': str(self.root), 'source_identity': identity,
            'all_images_verified': False,
        })

    def read_verified(self, relative, expected_sha):
        path = lexical_path(relative)
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha)):
            raise ValueError(f'invalid sealed SHA256: {relative}')
        key = hashlib.sha256(f'{relative}\0{expected_sha}'.encode()).hexdigest()
        receipt_path = self.directory / key[:2] / f'{key}.json'
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, UnicodeError):
            receipt = None
        descriptors = []
        try:
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            root_fd = os.open(self.root, directory_flags)
            descriptors.append(root_fd)
            split_fd = os.open(path.parts[0], directory_flags, dir_fd=root_fd)
            descriptors.append(split_fd)
            image_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                               dir_fd=split_fd)
            # fdopen owns image_fd even when reading fails.
            with os.fdopen(image_fd, 'rb') as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(f'unsafe non-regular image path: {relative}')
                raw = stream.read()
                after = os.fstat(stream.fileno())
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError(f'unsafe image path or symlink: {relative}') from exc
            raise
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
        fingerprint = _fingerprint(before)
        if fingerprint != _fingerprint(after) or len(raw) != before.st_size:
            raise ValueError(f'image changed while reading: {relative}')
        expected_receipt = {
            'schema_version': 1, 'status': 'verified', 'dataset_root': str(self.root),
            'relative_path': relative, 'sha256': expected_sha, 'file': fingerprint,
        }
        if receipt != expected_receipt:
            if _bytes_sha256(raw) != expected_sha:
                raise ValueError(f'image SHA256 identity mismatch: {relative}')
            _atomic_json(receipt_path, expected_receipt)
        return raw


def configured_cache(data_root):
    path = os.environ.get('TRAIN_VERIFICATION_CACHE')
    return VerificationCache(data_root, path) if path else None

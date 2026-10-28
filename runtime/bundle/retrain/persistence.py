"""Durable checkpoints for trusted, local artifacts produced by this tool.

Checkpoints contain optimizer and RNG objects and consequently require pickle.
The metadata format and SHA-256 checks reject accidental/legacy inputs and
corruption; they are not an authenticity signature. Do not load checkpoints or
metadata obtained from an untrusted producer.

One writer must hold RunLock for the entire training run. A checkpoint commit
publishes its immutable blob and sidecar before replacing index.json. Complete
blobs not referenced by the committed index can be left behind by a killed
process; a later successful save removes them. Incomplete temporary files are
ignored. The store never guesses from orphan files.

Before publishing its first blob, a new store durably commits an empty index.
An explicitly resumed worker may reinitialize only when is_empty() confirms
this marker. A missing index alongside existing blobs is an error, not proof
that a run never started.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import stat
import tempfile
import uuid
import warnings


_FORMAT = "retraining-container-checkpoint"
_INDEX_FORMAT = "retraining-container-checkpoint-index"
_FILENAME = re.compile(r"checkpoint-[0-9a-f]{32}\.pt\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CheckpointError(RuntimeError):
    """An artifact is corrupt, unsafe, or not a checkpoint from this tool."""


def _json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, obj):
    """Write canonical JSON through a same-directory, fsynced atomic replace."""
    path = Path(path)
    encoded = _json_bytes(obj) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(descriptor,0o644)  # new R8 account-readable metadata; old artifacts untouched
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return _hash_stream(stream)


def canonical_digest(obj):
    return hashlib.sha256(_json_bytes(obj)).hexdigest()


def capture_rng():
    """Capture CPU RNGs and the active single-card worker's CUDA RNG.

    CUDA is queried only when already initialized. The worker must select its
    device before calling this function. A single logical RNG is saved so a
    resumed worker can move to another visible device without touching peers.
    """
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state().cpu().clone() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    """Restore a captured state after model/optimizer construction has finished."""
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise CheckpointError("Cannot restore CUDA RNG on a CPU-only runtime")
        torch.cuda.set_rng_state(state["cuda"].cpu())


@contextmanager
def _regular_file(path):
    """Open without following the final symlink or blocking on a special file."""
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise CheckpointError(f"Cannot open regular checkpoint/metadata (symlink forbidden): {path.name}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CheckpointError(f"Expected regular checkpoint/metadata file: {path.name}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            yield stream
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_json(path):
    try:
        with _regular_file(path) as stream:
            return json.load(stream)
    except (ValueError, UnicodeError) as exc:
        raise CheckpointError(f"Invalid JSON metadata: {Path(path).name}") from exc


def _validate_record(record):
    if not isinstance(record, dict) or record.get("format") != _FORMAT or record.get("format_version") != 1:
        raise CheckpointError("Checkpoint metadata was not generated by this tool")
    filename = record.get("file")
    if not isinstance(filename, str) or not _FILENAME.fullmatch(filename):
        raise CheckpointError(f"Unsafe checkpoint filename in metadata: {filename!r}")
    digest = record.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise CheckpointError("Invalid checkpoint SHA-256 metadata")


@contextmanager
def _verified_checkpoint(path, expected=None):
    path = Path(path)
    record = _read_json(path.with_name(path.name + ".json"))
    _validate_record(record)
    if record["file"] != path.name:
        raise CheckpointError("Checkpoint filename does not match sidecar metadata")
    if expected is not None and record != expected:
        raise CheckpointError("Checkpoint sidecar metadata does not match committed index")
    with _regular_file(path) as stream:
        if _hash_stream(stream) != record["sha256"]:
            raise CheckpointError(f"Checkpoint hash mismatch: {path.name}")
        stream.seek(0)
        yield stream


def _validate_payload(payload):
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise CheckpointError("Checkpoint payload must be a dict with format_version=1")


def _load_file(path, map_location, expected=None):
    import torch

    with _verified_checkpoint(path, expected) as stream:
        try:
            # Full optimizer/RNG state needs pickle. This loader is exclusively
            # for trusted local tool artifacts, after the above metadata/hash checks.
            envelope = torch.load(stream, map_location=map_location, weights_only=False)
        except Exception as exc:
            raise CheckpointError(f"Cannot deserialize checkpoint: {Path(path).name}: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("format") != _FORMAT or envelope.get("format_version") != 1:
        raise CheckpointError("Checkpoint envelope was not generated by this tool")
    payload = envelope.get("payload")
    _validate_payload(payload)
    return payload


def load_file(path, map_location="cpu"):
    """Load a trusted local checkpoint with its original adjacent .pt.json file."""
    return _load_file(path, map_location)


class RunLock:
    """Nonblocking Unix advisory writer lock, released by the kernel on death.

    The empty lock file deliberately remains in place: unlinking a held lock
    would let another writer lock a different inode at the same pathname.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._descriptor = None

    def __enter__(self):
        if self._descriptor is not None:
            raise RuntimeError(f"Run lock already held: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError(f"Run lock is not a regular file: {self.path}")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RuntimeError(f"Run lock is in use by another writer: {self.path}") from exc
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, *exc):
        if self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


class CheckpointStore:
    """Transactional last/previous/best pointers to immutable checkpoint blobs."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def _index(self, allow_missing=False):
        path = self.directory / "index.json"
        if not path.exists() and not path.is_symlink():
            if allow_missing:
                if any(self.directory.glob("checkpoint-*.pt")):
                    raise CheckpointError("Checkpoint blobs exist but their committed index is missing")
                return {"format": _INDEX_FORMAT, "format_version": 1, "last": None, "previous": None, "best": None}
            raise FileNotFoundError(f"No committed checkpoint index: {path}")
        index = _read_json(path)
        if not isinstance(index, dict) or index.get("format") != _INDEX_FORMAT or index.get("format_version") != 1:
            raise CheckpointError("Invalid checkpoint index format")
        for key in ("last", "previous", "best"):
            if key not in index:
                raise CheckpointError(f"Checkpoint index is missing {key}")
            if index[key] is not None:
                _validate_record(index[key])
        protected=index.get("protected",{})
        if not isinstance(protected,dict):raise CheckpointError("Invalid protected checkpoint references")
        for key,record in protected.items():
            if not isinstance(key,str) or not re.fullmatch(r"[a-zA-Z0-9_-]+",key):raise CheckpointError("Invalid protected reference name")
            _validate_record(record)
        return index

    def is_empty(self):
        """Whether a valid committed index proves no payload was committed yet.

        Missing indexes raise FileNotFoundError; malformed indexes raise
        CheckpointError. Neither case authorizes reinitializing a resumed run.
        """
        index = self._index()
        return not index.get("protected") and all(index[kind] is None for kind in ("last", "previous", "best"))

    def _latest_valid_record(self, index):
        for kind in ("last", "previous"):
            record = index[kind]
            if record is None:
                continue
            try:
                with _verified_checkpoint(self.directory / record["file"], record):
                    return record
            except CheckpointError:
                continue
        return None

    def _prune(self, index):
        keep = {record["file"] for key in ("last", "previous", "best") if (record := index[key]) is not None}
        keep.update(record["file"] for record in index.get("protected",{}).values())
        for path in self.directory.iterdir():
            filename = path.name[:-5] if path.name.endswith(".pt.json") else path.name
            if _FILENAME.fullmatch(filename) and filename not in keep:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    warnings.warn(f"Committed checkpoint; obsolete file cleanup failed: {path.name}: {exc}", RuntimeWarning)
        _fsync_directory(self.directory)

    def save(self, payload, best=False, protect=()):
        import torch

        _validate_payload(payload)
        if any(not isinstance(k,str) or not re.fullmatch(r"[a-zA-Z0-9_-]+",k) for k in protect):raise ValueError("Invalid protected reference")
        self.directory.mkdir(parents=True, exist_ok=True)
        index = self._index(allow_missing=True)
        if not (self.directory / "index.json").exists():
            # This marker must be durable before even the first blob is created:
            # it distinguishes an interrupted first commit from a lost index.
            atomic_json(self.directory / "index.json", index)
        previous = self._latest_valid_record(index)
        destination = self.directory / f"checkpoint-{uuid.uuid4().hex}.pt"
        descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=self.directory)
        try:
            os.fchmod(descriptor,0o644)  # new immutable blobs survive container-root ownership
            with os.fdopen(descriptor, "wb") as stream:
                torch.save({"format": _FORMAT, "format_version": 1, "payload": payload}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            record = {"format": _FORMAT, "format_version": 1, "file": destination.name, "sha256": sha256_file(temporary)}
            os.replace(temporary, destination)
            _fsync_directory(self.directory)
            atomic_json(destination.with_name(destination.name + ".json"), record)
            committed = {"format": _INDEX_FORMAT, "format_version": 1, "last": record, "previous": previous, "best": record if best else index["best"]}
            protected={**index.get("protected",{}),**{k:record for k in protect}}
            if protected:committed['protected']=protected
            atomic_json(self.directory / "index.json", committed)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self._prune(committed)
        return destination

    def load(self, kind="last", map_location="cpu"):
        index = self._index()
        if kind not in ("last", "previous", "best") and kind not in index.get('protected',{}):
            raise ValueError("Checkpoint kind must be committed or protected")
        record = index[kind] if kind in ('last','previous','best') else index['protected'][kind]
        if record is None:
            raise FileNotFoundError(f"No committed {kind} checkpoint in {self.directory}")
        try:
            return _load_file(self.directory / record["file"], map_location, record)
        except CheckpointError as exc:
            previous = index["previous"]
            if kind != "last" or previous is None:
                raise
            restored = _load_file(self.directory / previous["file"], map_location, previous)
            atomic_json(self.directory / "recovery.json", {
                "failed_file": record["file"],
                "recovered_file": previous["file"],
                "reason": str(exc),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            warnings.warn(f"Checkpoint fallback from {record['file']} to {previous['file']}: {exc}", RuntimeWarning, stacklevel=2)
            return restored

    load_file = staticmethod(load_file)

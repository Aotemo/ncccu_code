"""Ephemeral N03-only FileStore on the shared Pod-local tmpfs."""
import hashlib
import os
from pathlib import Path
import time
import uuid


def new_path():
    base=Path('/dev/shm')
    mounts=[line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    matches=[parts for parts in mounts if len(parts)>=4 and parts[1]==str(base)]
    if (len(matches)!=1 or matches[0][2]!='tmpfs' or 'rw' not in matches[0][3].split(',')
            or base.is_symlink() or not os.access(base,os.W_OK|os.X_OK)):
        raise RuntimeError('N03 Pod-local writable tmpfs is unavailable')
    path=base/('n03-rendezvous-'+uuid.uuid4().hex)
    if path.exists() or path.is_symlink():raise FileExistsError('rendezvous path collision')
    return path


def finish(path,control):
    from .persistence import atomic_json
    path=Path(path);control=Path(control)
    if path.parent!=Path('/dev/shm') or not path.name.startswith('n03-rendezvous-'):
        raise ValueError('unexpected N03 rendezvous path')
    receipt={'path':str(path),'at':time.time(),'present_when_checked':path.exists()}
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.stat().st_size>1048576:
            raise ValueError('unexpected rendezvous type or size')
        contents=path.read_bytes();control.mkdir(parents=True,exist_ok=True)
        snapshot=control/'rendezvous-at-termination.bin'
        with snapshot.open('xb') as f:f.write(contents);f.flush();os.fsync(f.fileno())
        receipt.update(snapshot_path=str(snapshot),snapshot_sha256=hashlib.sha256(contents).hexdigest(),
                       snapshot_bytes=len(contents),removed_own_temporary_file=True)
        path.unlink()
    atomic_json(control/'rendezvous-terminal.json',receipt)

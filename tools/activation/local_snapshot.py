"""Prepare a Git-verified source snapshot on a Pod-local filesystem."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_regular(archive, destination):
    """Extract only unique regular files/directories without following links."""
    destination = Path(destination)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError('fresh regular extraction directory required')
    seen, count = set(), 0
    with tarfile.open(archive, 'r:*') as package:
        for member in package:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or not path.parts or any(part in ('', '.', '..') for part in path.parts)
                    or str(path) in seen or not (member.isdir() or member.isfile())):
                raise ValueError('unsafe or duplicate archive entry')
            seen.add(str(path))
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or not target.is_dir():
                    raise ValueError('archive directory is not regular')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if any(path.is_symlink() for path in [target.parent, *target.parent.parents]
                   if path != destination.parent):
                raise ValueError('archive path traverses a symlink')
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, 'wb') as stream, package.extractfile(member) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    stream.write(block)
                stream.flush(); os.fsync(stream.fileno())
            os.chmod(target, member.mode & 0o777)
            count += 1
    if not count:
        raise ValueError('source archive has no regular files')
    return count


def mount_type(path):
    """Read the most specific live mount for the local path."""
    path = Path(path).resolve(strict=True)
    choices = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, filesystem = left.split(), right.split()[0]
        name = fields[4]
        for escaped, char in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
            name = name.replace(escaped, char)
        mount = Path(name)
        if path.is_relative_to(mount):
            choices.append((len(mount.parts), filesystem, str(mount)))
    if not choices:
        raise ValueError('local source mount is unknown')
    _, filesystem, mount = max(choices)
    if filesystem.lower() in ('nfs', 'nfs4', 'cifs', 'smb3', 'fuse.sshfs'):
        raise ValueError('source mount is remote')
    return {'type': filesystem, 'mount': mount}


def verify_local_source(source, uid):
    source = Path(source)
    expected = Path('/tmp/technical-prep') / uid
    if (not re.fullmatch(r'[0-9a-f-]{36}', uid or '') or not source.is_absolute()
            or source.name != 'source-git' or source.is_symlink()
            or source.resolve(strict=True) != source or not source.is_relative_to(expected)):
        raise ValueError('source must be an explicit Pod-UID-bound local snapshot')
    return mount_type(source)


def verify_local_receipt(source, uid, commit, tree, receipt, files):
    """Require the archived whole-tree proof for this exact Pod-local snapshot."""
    verify_local_source(source, uid)
    if (not isinstance(receipt, dict)
            or receipt.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
            or receipt.get('source_root') != str(source) or receipt.get('pod_uid') != uid
            or receipt.get('commit') != commit or receipt.get('tree') != tree
            or not re.fullmatch(r'[0-9a-f]{40}', commit or '')
            or not re.fullmatch(r'[0-9a-f]{40}', tree or '')
            or not re.fullmatch(r'[0-9a-f]{64}', receipt.get('archive', {}).get('sha256', ''))
            or not isinstance(files, dict) or receipt.get('file_count') != len(files)
            or receipt.get('file_sha256_map_digest') != hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            or receipt.get('GPU_initialized') is not False):
        raise ValueError('local source full-tree receipt differs')


def own_published_files(metadata, helper_root):
    entries = {row['path']: row for row in metadata['tree'] if row.get('type') == 'blob'}
    for name in ('tools/activation/local_snapshot.py', 'tools/activation/freeze_deployment.py'):
        path = helper_root / Path(name).name
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        if path.is_symlink() or name not in entries or entries[name]['sha'] != blob or entries[name]['mode'] != mode:
            raise ValueError('bootstrap helper differs from the published tree: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'metadata-root', 'source-root', 'receipt', 'work-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('archive-sha256', 'commit', 'tree-sha', 'pod-uid'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(key) != '' for key in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or os.environ.get('NCCCU_POD_UID') != args.pod_uid):
        raise ValueError('live UID-bound hidden-device Pod CPU required')
    work = args.work_root.resolve(strict=True)
    if (args.archive.is_symlink() or not args.archive.resolve(strict=True).is_relative_to(work)
            or not args.metadata_root.resolve(strict=True).is_relative_to(work / 'technical-prep')
            or args.receipt.exists() or not args.receipt.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')
            or sha(args.archive) != args.archive_sha256):
        raise ValueError('archive or NFS evidence identity differs')
    ref = json.loads((args.metadata_root / 'REMOTE-REF.json').read_text())
    commit = json.loads((args.metadata_root / 'REMOTE-COMMIT.json').read_text())
    tree = json.loads((args.metadata_root / 'REMOTE-TREE.json').read_text())
    if (not re.fullmatch(r'[0-9a-f]{40}', args.commit) or not re.fullmatch(r'[0-9a-f]{40}', args.tree_sha)
            or ref.get('ref') != 'refs/heads/main' or ref.get('object', {}).get('sha') != args.commit
            or commit.get('sha') != args.commit or commit.get('tree', {}).get('sha') != args.tree_sha
            or tree.get('sha') != args.tree_sha or tree.get('truncated') is not False):
        raise ValueError('complete API-published commit and tree required')
    own_published_files(tree, Path(__file__).resolve().parent)
    source = args.source_root
    expected = Path('/tmp/technical-prep') / args.pod_uid
    if (not source.is_absolute() or source.name != 'source-git' or source.exists()
            or not source.is_relative_to(expected)):
        raise ValueError('fresh UID-bound Pod-local source path required')
    source.parent.mkdir(parents=True, exist_ok=True)
    if source.parent.resolve(strict=True) != source.parent:
        raise ValueError('local source parent traverses a symlink')
    mount_type(source.parent)
    source.mkdir()
    count = extract_regular(args.archive, source)
    mount = verify_local_source(source, args.pod_uid)
    helper = Path(__file__).resolve().parent / 'freeze_deployment.py'
    spec = importlib.util.spec_from_file_location('published_freeze_deployment', helper)
    freeze = importlib.util.module_from_spec(spec); spec.loader.exec_module(freeze)
    freeze.verify_commit(commit, args.commit, args.tree_sha)
    files = freeze.verify_tree(source, tree, args.tree_sha)
    if count != len(files):
        raise ValueError('archive and full Git tree counts differ')
    canonical = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    result = {'status': 'published_full_git_tree_verified_not_formal_deployment', 'at': time.time(),
              'commit': args.commit, 'tree': args.tree_sha, 'source_root': str(source),
              'pod_uid': args.pod_uid, 'local_mount': mount,
              'archive': {'path': str(args.archive), 'sha256': args.archive_sha256},
              'file_count': len(files), 'file_sha256_map_digest': hashlib.sha256(canonical).hexdigest(),
              'GPU_initialized': False, 'formal_CPU_or_activation_executed': False, 'official_read': False}
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'status': result['status'], 'source_root': str(source), 'files': len(files),
                      'receipt_sha256': sha(args.receipt)}), flush=True)


if __name__ == '__main__':
    main()

"""Observe same-length writes and metadata on the stage scratch filesystem only."""
import argparse
import inspect
import os
from pathlib import Path
import tempfile

from source_chain import cpu_context, file_identity, inherited_file, sha256, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    context = cpu_context()
    with tempfile.TemporaryDirectory(dir=args.scratch) as folder:
        path = Path(folder) / 'same-size.json'
        path.write_bytes(b'{}\n')
        original = dict(file_identity(path), exists=True, sha256=sha256(path))
        before = path.stat()
        with path.open('r+b') as stream:
            stream.write(b'[]\n')
        after_write = dict(file_identity(path), sha256=sha256(path), content=repr(path.read_bytes()))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        after_utime = dict(file_identity(path), sha256=sha256(path), content=repr(path.read_bytes()))
        observations = {}
        try:
            inherited_file(path, original, original['sha256'])
            observations['after_utime_rejected'] = False
        except ValueError as error:
            observations.update(after_utime_rejected=True, after_utime_error=str(error))
        with path.open('r+b') as stream:
            stream.flush()
            os.fsync(stream.fileno())
            descriptor_stat = stream.fileno()
            actual = os.fstat(descriptor_stat)
            observations['after_fsync_fstat'] = {k: getattr(actual, k) for k in
                                                  ('st_size', 'st_dev', 'st_ino', 'st_mtime_ns', 'st_ctime_ns')}
        after_fsync = dict(file_identity(path), sha256=sha256(path), content=repr(path.read_bytes()))
        try:
            inherited_file(path, original, original['sha256'])
            observations['after_fsync_rejected'] = False
        except ValueError as error:
            observations.update(after_fsync_rejected=True, after_fsync_error=str(error))
        result = {'passed': True, 'scope': 'scratch_filesystem_observation_only_not_source_acceptance',
                  'execution': context, 'original': original, 'after_write': after_write,
                  'after_utime': after_utime, 'requested_mtime_ns': before.st_mtime_ns + 1_000_000_000,
                  'after_fsync': after_fsync, 'observations': observations,
                  'checker_source': inspect.getsource(inherited_file),
                  'checker_file_sha256': sha256(inspect.getsourcefile(inherited_file)),
                  'diagnostic_source_sha256': sha256(__file__), 'weight_files_read': False}
    write_new_json(args.output, result)
    print({k: result[k] for k in ('original', 'after_write', 'after_utime', 'requested_mtime_ns',
                                 'after_fsync', 'observations', 'checker_source')}, flush=True)


if __name__ == '__main__':
    main()

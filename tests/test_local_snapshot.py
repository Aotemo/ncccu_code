"""Archive extraction must reject traversal, links and duplicate entries."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

source = Path(__file__).resolve().parents[1] / 'tools/activation/local_snapshot.py'
spec = importlib.util.spec_from_file_location('local_snapshot', source)
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


class LocalSnapshotTests(unittest.TestCase):
    def archive(self, root, members):
        path = root / 'source.tar'
        with tarfile.open(path, 'w') as stream:
            for name, kind, payload in members:
                entry = tarfile.TarInfo(name)
                if kind == 'file':
                    entry.mode = 0o755 if name.endswith('.sh') else 0o644
                    entry.size = len(payload)
                    stream.addfile(entry, io.BytesIO(payload))
                elif kind == 'link':
                    entry.type = tarfile.SYMTYPE; entry.linkname = payload
                    stream.addfile(entry)
                else:
                    entry.type = tarfile.DIRTYPE; stream.addfile(entry)
        return path

    def test_extracts_only_regular_files_and_preserves_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dest = root / 'out'; dest.mkdir()
            archive = self.archive(root, [('src', 'dir', None), ('src/run.sh', 'file', b'hello')])
            self.assertEqual(snapshot.extract_regular(archive, dest), 1)
            self.assertEqual((dest / 'src/run.sh').read_bytes(), b'hello')
            self.assertEqual((dest / 'src/run.sh').stat().st_mode & 0o777, 0o755)

    def test_rejects_traversal_absolute_symlinks_and_duplicate_files(self):
        for entries in ([('../escape', 'file', b'x')], [('/escape', 'file', b'x')],
                        [('src', 'link', '../escape')],
                        [('a', 'file', b'x'), ('a', 'file', b'y')]):
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); dest = root / 'out'; dest.mkdir()
                with self.assertRaises(ValueError):
                    snapshot.extract_regular(self.archive(root, entries), dest)


if __name__ == '__main__':
    unittest.main()

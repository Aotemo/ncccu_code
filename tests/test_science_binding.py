"""A local Git science file is bound only to an exact NFS-side byte copy."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_under_test', root / 'tools/activation/prepare.py')
prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)


class ScienceBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'local/science.json'; self.source.parent.mkdir()
        self.source.write_bytes(b'{"candidate":"identity"}\n')
        self.work = self.root / 'work'; (self.work / 'technical-prep').mkdir(parents=True)
        self.bound = self.work / 'technical-prep/science.json'
        self.bound.write_bytes(self.source.read_bytes())

    def test_accepts_exact_byte_copy_inside_technical_prep(self):
        self.assertEqual(prepare.science_binding_path(self.source, self.bound, self.work), self.bound)
        self.assertEqual(prepare.science_binding_path(self.source, None, self.work), self.source)

    def test_rejects_changed_external_or_symlinked_copy(self):
        self.bound.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            prepare.science_binding_path(self.source, self.bound, self.work)
        outside = self.root / 'outside.json'; outside.write_bytes(self.source.read_bytes())
        with self.assertRaises(ValueError):
            prepare.science_binding_path(self.source, outside, self.work)
        self.bound.unlink(); self.bound.symlink_to(outside)
        with self.assertRaises(ValueError):
            prepare.science_binding_path(self.source, self.bound, self.work)


if __name__ == '__main__':
    unittest.main()

"""Necessary byte-integrity and path-safety checks; execute on the Pod only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/migration'))
sys.path.insert(0, str(ROOT / 'tools/pretraining'))
import relocate_assets
from source_chain import cpu_context, sha256, write_new_json
SCRATCH = None


class AssetRelocationTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(relocate_assets.copy_verified_file), 'verified byte-copy feature is absent')
        self.folder = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'old' / 'FREEZE.json'
        self.source.parent.mkdir()
        self.source.write_bytes(b'{"canonical":"old","files":{}}\n')
        self.digest = sha256(self.source)
        self.target = self.root / 'new' / 'FREEZE.json'

    def test_copy_preserves_original_and_target_bytes_with_current_content_proof(self):
        result = relocate_assets.copy_verified_file(self.source, self.target, self.digest)
        self.assertEqual(self.source.read_bytes(), self.target.read_bytes())
        self.assertEqual(result['sha256'], self.digest)
        self.assertEqual(result['target_sha256'], self.digest)
        self.assertFalse(result['source_modified_by_tool'])
        self.assertNotEqual(self.source.stat().st_ino, self.target.stat().st_ino)

    def test_existing_same_length_corruption_and_wrong_expected_sha_are_rejected(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b'x' * self.source.stat().st_size)
        with self.assertRaises(ValueError):
            relocate_assets.copy_verified_file(self.source, self.target, self.digest)
        self.assertEqual(self.target.read_bytes(), b'x' * self.source.stat().st_size)
        with self.assertRaises(ValueError):
            relocate_assets.copy_verified_file(self.source, self.root / 'wrong', '0' * 64)
        self.assertFalse((self.root / 'wrong').exists())

    def test_source_and_target_symlink_routes_are_rejected(self):
        link = self.root / 'source-link'
        link.symlink_to(self.source)
        with self.assertRaises(ValueError):
            relocate_assets.copy_verified_file(link, self.target, self.digest)
        actual = self.root / 'outside'
        actual.mkdir()
        self.target.parent.symlink_to(actual, target_is_directory=True)
        with self.assertRaises(ValueError):
            relocate_assets.copy_verified_file(self.source, self.target, self.digest)
        self.assertFalse((actual / self.target.name).exists())


def main():
    global SCRATCH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    context = cpu_context()
    SCRATCH = args.scratch
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AssetRelocationTest))
    write_new_json(args.output, {'passed': result.wasSuccessful(), 'execution': context,
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'source_files': {str(path.relative_to(ROOT)): sha256(path) for path in
            (ROOT / 'tools/migration/relocate_assets.py', Path(__file__).resolve())}})
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()

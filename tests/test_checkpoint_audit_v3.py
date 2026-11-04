"""Narrow checks for the read-only, resumable checkpoint inventory."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


here = Path(__file__).resolve()
source = here.parent / 'checkpoint_audit_v3.py'
if not source.exists():
    source = here.parents[1] / 'tools/storage/checkpoint_audit_v3.py'
spec = importlib.util.spec_from_file_location('checkpoint_audit_v3', source)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CheckpointAuditTests(unittest.TestCase):
    def test_name_patterns_include_models_without_classifying_image_cache_arrays(self):
        for name in ('checkpoint-0001.pt', 'model.PTH', 'weights.safetensors',
                     'model.ckpt', 'pytorch_model.bin', 'model.npz', 'checkpoint.pth.tar'):
            self.assertTrue(audit.potential_weight(name, 'round/checkpoints'), name)
        self.assertTrue(audit.potential_weight('abcdef.npy', 'round/weights'))
        self.assertTrue(audit.potential_weight('array.npy', 'technical-prep/unknown-copy'))
        self.assertFalse(audit.potential_weight('abcdef.npy', 'prepared-images/00'))
        self.assertFalse(audit.potential_weight('figure.png', 'round/checkpoints'))
        self.assertFalse(audit.potential_weight('notes.tar.gz', 'round/notes'))

    def test_paths_use_directory_boundaries(self):
        self.assertTrue(audit.inside('/work/rounds/r1/model.pt', '/work/rounds/r1'))
        self.assertFalse(audit.inside('/work/rounds/r10/model.pt', '/work/rounds/r1'))

    def test_resumable_scan_does_not_follow_symlinks_or_change_files(self):
        with tempfile.TemporaryDirectory(prefix='audit-test-', dir=Path.cwd()) as tmp:
            root = Path(tmp) / 'root'
            (root / 'r1/checkpoints').mkdir(parents=True)
            (root / 'cache').mkdir()
            (root / 'r1/checkpoints/one.pt').write_bytes(b'one')
            (root / 'r1/checkpoints/two.ckpt').write_bytes(b'two')
            (root / 'cache/image.npy').write_bytes(b'image')
            outside = Path(tmp) / 'outside'
            outside.mkdir()
            (outside / 'hidden.pt').write_bytes(b'outside')
            os.symlink(outside, root / 'linked')
            state = audit.new_scan(root, excluded=[])
            for _ in range(100):
                state = audit.scan_chunk(state, max_entries=2, delay=0, seconds=30)
                state = json.loads(json.dumps(state))
                if state['complete']:
                    break
            self.assertTrue(state['complete'])
            names = {Path(row['path']).name for row in state['files']}
            self.assertEqual(names, {'one.pt', 'two.ckpt'})
            self.assertEqual((outside / 'hidden.pt').read_bytes(), b'outside')
            self.assertEqual((root / 'r1/checkpoints/one.pt').read_bytes(), b'one')
            self.assertEqual(len(state['symlinks']), 1)


if __name__ == '__main__':
    unittest.main()

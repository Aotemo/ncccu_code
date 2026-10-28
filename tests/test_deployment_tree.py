"""Freeze receipts must bind actual bytes and modes to the remote Git tree."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('freeze_deployment', ROOT / 'tools/activation/freeze_deployment.py')
freeze = importlib.util.module_from_spec(spec); spec.loader.exec_module(freeze)


class DeploymentTreeTest(unittest.TestCase):
    def test_commit_readback_must_name_the_exact_deployed_tree(self):
        commit = {'sha': 'b' * 40, 'tree': {'sha': 'a' * 40}}
        freeze.verify_commit(commit, 'b' * 40, 'a' * 40)
        for value in ({**commit, 'sha': 'c' * 40}, {**commit, 'tree': {'sha': 'd' * 40}}, {}):
            with self.subTest(commit=value), self.assertRaises(ValueError):
                freeze.verify_commit(value, 'b' * 40, 'a' * 40)

    def test_remote_blob_identity_and_complete_file_set_are_required(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory); path = root / 'independent.txt'; content = b'fixed remote content\n'
            path.write_bytes(content); path.chmod(0o644)
            blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            tree = {'sha': 'a' * 40, 'truncated': False,
                    'tree': [{'type': 'blob', 'path': 'independent.txt', 'mode': '100644', 'sha': blob}]}
            self.assertEqual(freeze.verify_tree(root, tree, 'a' * 40),
                             {'independent.txt': hashlib.sha256(content).hexdigest()})
            path.write_bytes(b'changed remote content\n')
            with self.assertRaises(ValueError):
                freeze.verify_tree(root, tree, 'a' * 40)
            path.write_bytes(content); (root / 'extra.txt').write_text('not in remote commit\n')
            with self.assertRaises(ValueError):
                freeze.verify_tree(root, tree, 'a' * 40)

    def test_truncation_tree_pin_executable_mode_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory); path = root / 'entry'; path.write_bytes(b'x'); path.chmod(0o644)
            tree = {'sha': 'a' * 40, 'truncated': False, 'tree': [{'type': 'blob', 'path': 'entry',
                'mode': '100755', 'sha': hashlib.sha1(b'blob 1\0x').hexdigest()}]}
            for candidate, pin in ((tree, 'b' * 40), ({**tree, 'truncated': True}, 'a' * 40), (tree, 'a' * 40)):
                with self.subTest(candidate=candidate, pin=pin), self.assertRaises(ValueError):
                    freeze.verify_tree(root, candidate, pin)
            path.unlink(); path.symlink_to(root / 'missing')
            with self.assertRaises(ValueError):
                freeze.verify_tree(root, tree, 'a' * 40)


if __name__ == '__main__':
    unittest.main()

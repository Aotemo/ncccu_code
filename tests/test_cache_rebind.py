"""Narrow prepared-cache migration checks; run through the verified Pod CLI only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/migration'))
sys.path.insert(0, str(ROOT / 'tools/pretraining'))
import cache_rebind
from source_chain import cpu_context, sha256, write_new_json

SCRATCH = None
LIVE_CONTEXT = None


def small_npy():
    # A genuine NPY v1 uint8 RGB array; no decode, preparation or model needed.
    header = b"{'descr': '|u1', 'fortran_order': False, 'shape': (1, 1, 3), }"
    header += b' ' * ((-(10 + len(header) + 1)) % 64) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + b'\x01\x02\x03'


class CacheRebindTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.old_data, self.new_data = self.root / 'old/data', self.root / 'new/data'
        self.old_cache, self.new_cache = self.root / 'old/cache', self.root / 'new/cache'
        for directory in (self.old_data, self.new_data, self.old_cache, self.new_cache):
            directory.mkdir(parents=True)
        self.old_policy = cache_rebind.policy(self.old_data)
        self.new_policy = cache_rebind.policy(self.new_data)
        self.relative = 'train/example.raw'
        self.raw = b'original sealed raw image bytes'
        self.array = small_npy()
        self.source_sha = hashlib.sha256(self.raw).hexdigest()
        self.array_sha = hashlib.sha256(self.array).hexdigest()
        self.key = cache_rebind.entry_key(self.relative, self.source_sha)
        self.old_raw, self.new_raw = self.old_data / self.relative, self.new_data / self.relative
        tail = Path(self.key[:2]) / (self.key + '.npy')
        self.old_array = self.old_cache / 'train/sealed' / cache_rebind.namespace(self.old_policy) / tail
        self.new_array = self.new_cache / 'train/sealed' / cache_rebind.namespace(self.new_policy) / tail
        for path, raw in ((self.old_raw, self.raw), (self.new_raw, self.raw),
                          (self.old_array, self.array), (self.new_array, self.array)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.old_receipt = self.old_array.with_suffix('.json')
        self.new_receipt = self.new_array.with_suffix('.json')
        self.original_receipt = {'policy': self.old_policy, 'relative_path': self.relative,
                                 'source_sha256': self.source_sha, 'array_sha256': self.array_sha,
                                 'source_file': cache_rebind.source_file(self.identity(self.old_raw))}
        self.old_receipt.write_bytes(cache_rebind.json_bytes(self.original_receipt))
        self.execution = copy.deepcopy(LIVE_CONTEXT) if LIVE_CONTEXT else {
            'namespace': 'fixture', 'pod': 'fixture-pod', 'uid': 'current-fixture-uid',
            'container': 'fixture-container', 'python': sys.executable,
            'cpu_allocation': {'path': str(self.root / 'allocation.json'), 'sha256': 'a' * 64,
                               'executor_id': 'fixture-B', 'threads': 1, 'channel_mask': [0]}}
        self.rows = [self.copy_row('raw', self.old_raw, self.new_raw),
                     self.copy_row('prepared-train', self.old_array, self.new_array)]
        self.manifest, self.copy_receipt = self.root / 'COPY.jsonl', self.root / 'COPY.json'
        self.output = self.root / 'REBOUND.json'
        self.output_serial = 0
        self.pin_copy()

    def identity(self, path):
        return cache_rebind.file_identity(path, path.stat())

    def copy_row(self, asset, source, target):
        digest, target_digest = sha256(source), sha256(target)
        self.assertEqual(digest, target_digest)
        return {'asset': asset, 'source_path': str(source), 'target_path': str(target),
                'sha256': digest, 'target_sha256': target_digest, 'size': source.stat().st_size,
                'source_identity': self.identity(source), 'target_identity': self.identity(target),
                'evidence': 'fresh_pod_sha256', 'source_modified_by_tool': False}

    def pin_copy(self, execution=None):
        self.manifest.write_bytes(b''.join(cache_rebind.json_bytes(row) for row in self.rows))
        self.copy_receipt.write_bytes(cache_rebind.json_bytes({
            'schema_version': 1, 'passed': True, 'execution': execution or self.execution,
            'manifest': {'path': str(self.manifest), 'sha256': sha256(self.manifest)}}))

    def rebind(self, **changes):
        self.output_serial += 1
        self.output = Path(changes.get('output', self.root / f'REBOUND-{self.output_serial}.json'))
        args = {'source_cache_root': self.old_cache, 'target_cache_root': self.new_cache,
                'source_data_root': self.old_data, 'target_data_root': self.new_data,
                'split': 'train', 'expected_count': 1,
                'copy_manifest': self.manifest, 'copy_manifest_sha256': sha256(self.manifest),
                'copy_receipt': self.copy_receipt, 'copy_receipt_sha256': sha256(self.copy_receipt),
                'output': self.output, 'execution': self.execution}
        args.update(changes)
        return cache_rebind.rebind_cache(**args)

    def test_byte_preserving_rebind_and_exact_existing_receipt_reuse(self):
        original = {p: p.read_bytes() for p in (self.old_raw, self.old_array, self.old_receipt)}
        actual_content_proof = cache_rebind.content_proof

        def no_old_blob_read(path, *args, **kwargs):
            self.assertNotIn(Path(path), (self.old_raw, self.old_array), 'old blobs must use pinned copy evidence')
            return actual_content_proof(path, *args, **kwargs)

        with mock.patch.object(cache_rebind, 'content_proof', side_effect=no_old_blob_read):
            report = self.rebind()
        receipt = json.loads(self.new_receipt.read_bytes())
        self.assertEqual(receipt, dict(self.original_receipt, policy=self.new_policy,
                                      source_file=cache_rebind.source_file(self.identity(self.new_raw))))
        self.assertNotEqual(report['source']['namespace'], report['target']['namespace'])
        self.assertEqual(report['counts'], {'entries': 1, 'raw_bytes': len(self.raw),
            'array_bytes': len(self.array), 'receipts_created': 1, 'receipts_reused': 0})
        self.assertEqual(report['byte_evidence']['old_raw_or_array_bytes_reread'], 0)
        self.assertEqual(report['entries_manifest']['sha256'], sha256(Path(report['entries_manifest']['path'])))
        self.assertEqual(self.new_array.read_bytes(), self.array)
        self.assertEqual(original, {p: p.read_bytes() for p in original})
        second = self.rebind(output=self.root / 'REUSED.json')
        self.assertEqual(second['counts']['receipts_created'], 0)
        self.assertEqual(second['counts']['receipts_reused'], 1)
        legacy = cache_rebind.policy('/public/home/acfwvv9ejn/SothisAI/dataset/Console/glaucoma_first/main/glaucoma_first')
        self.assertEqual(cache_rebind.namespace(legacy),
                         '19b2e1abff37369711cf59110dd52a0fb4894dd35037ee4b3eb758f8bd201a48')

    def test_same_length_target_raw_and_array_corruption_requires_content_sha(self):
        for path, original in ((self.new_raw, self.raw), (self.new_array, self.array)):
            with self.subTest(path=path.name):
                before = path.stat()
                changed = bytes([original[0] ^ 1]) + original[1:]
                path.write_bytes(changed)
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                self.assertEqual(path.stat().st_size, len(original))
                # Require the content failure even on a filesystem with frozen stat.
                with self.assertRaisesRegex(ValueError, 'current content SHA256 mismatch'):
                    self.rebind()
                self.assertFalse(self.new_receipt.exists())
                self.assertFalse(self.output.with_suffix('.entries.jsonl').exists())
                path.write_bytes(original)
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    def test_wrong_policy_key_and_target_namespace_are_rejected(self):
        for field, value, message in (
                ('policy', dict(self.old_policy, pillow='wrong'), 'policy/fields differ'),
                ('relative_path', 'train/other.raw', 'key/namespace differs')):
            with self.subTest(field=field):
                changed = dict(self.original_receipt, **{field: value})
                self.old_receipt.write_bytes(cache_rebind.json_bytes(changed))
                with self.assertRaisesRegex(ValueError, message):
                    self.rebind()
                self.assertFalse(self.new_receipt.exists())
        self.old_receipt.write_bytes(cache_rebind.json_bytes(self.original_receipt))
        wrong = self.new_cache / 'train/sealed' / ('0' * 64) / self.key[:2] / self.new_array.name
        self.rows[1]['target_path'] = str(wrong)
        self.rows[1]['target_identity'].update(path=str(wrong), resolved=str(wrong))
        self.pin_copy()
        with self.assertRaisesRegex(ValueError, 'target namespace/path differs'):
            self.rebind()

    def test_old_fingerprint_and_current_copy_receipt_binding_are_required(self):
        changed = copy.deepcopy(self.original_receipt)
        changed['source_file']['inode'] += 1
        self.old_receipt.write_bytes(cache_rebind.json_bytes(changed))
        with self.assertRaisesRegex(ValueError, 'old receipt source_file differs'):
            self.rebind()
        self.old_receipt.write_bytes(cache_rebind.json_bytes(self.original_receipt))
        for field in ('uid', 'cpu_allocation'):
            with self.subTest(field=field):
                wrong_context = copy.deepcopy(self.execution)
                if field == 'uid':
                    wrong_context['uid'] = 'historical-uid'
                else:
                    wrong_context['cpu_allocation']['sha256'] = 'b' * 64
                self.pin_copy(execution=wrong_context)
                with self.assertRaisesRegex(ValueError, 'different current Pod|CPU allocation differs'):
                    self.rebind()
                self.assertFalse(self.new_receipt.exists())

    def test_manifest_pin_missing_duplicate_and_failed_proofs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'manifest path/SHA'):
            self.rebind(copy_manifest_sha256='0' * 64)
        base = copy.deepcopy(self.rows)
        for variant in ('missing', 'duplicate', 'failed', 'wrong-raw-target', 'count'):
            with self.subTest(variant=variant):
                self.rows = copy.deepcopy(base)
                if variant == 'missing':
                    self.rows.pop(0)
                elif variant == 'duplicate':
                    self.rows.append(copy.deepcopy(self.rows[0]))
                elif variant == 'failed':
                    self.rows[0]['error'] = 'copy failed'
                elif variant == 'wrong-raw-target':
                    wrong = str(self.new_data / 'train/wrong.raw')
                    self.rows[0]['target_path'] = wrong
                    self.rows[0]['target_identity'].update(path=wrong, resolved=wrong)
                self.pin_copy()
                with self.assertRaises(ValueError):
                    self.rebind(**({'expected_count': 2} if variant == 'count' else {}))
                self.assertFalse(self.new_receipt.exists())

    def test_unsafe_relative_and_symlink_route_and_wrong_existing_receipt_are_rejected(self):
        changed = dict(self.original_receipt, relative_path='train/../outside.raw')
        self.old_receipt.write_bytes(cache_rebind.json_bytes(changed))
        with self.assertRaisesRegex(ValueError, 'unsafe relative path'):
            self.rebind()
        self.old_receipt.write_bytes(cache_rebind.json_bytes(self.original_receipt))
        wrong_receipt = b'{"policy":"wrong"}\n'
        self.new_receipt.write_bytes(wrong_receipt)
        with self.assertRaisesRegex(ValueError, 'existing target receipt differs'):
            self.rebind()
        self.assertEqual(self.new_receipt.read_bytes(), wrong_receipt)
        self.new_receipt.unlink()
        self.new_raw.unlink()
        self.new_raw.parent.rmdir()
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / self.new_raw.name).write_bytes(self.raw)
        self.new_raw.parent.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'unsafe or inaccessible directory route'):
            self.rebind()
        self.assertFalse(self.new_receipt.exists())


def main():
    global SCRATCH, LIVE_CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    LIVE_CONTEXT = cpu_context()
    SCRATCH = args.scratch
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CacheRebindTest))
    write_new_json(args.output, {'passed': result.wasSuccessful(), 'execution': LIVE_CONTEXT,
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'source_files': {str(path.relative_to(ROOT)): sha256(path) for path in
                         (ROOT / 'tools/migration/cache_rebind.py', Path(__file__).resolve())}})
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())

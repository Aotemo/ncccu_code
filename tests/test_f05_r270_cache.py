"""Cache revalidation rejects altered, missing or out-of-scope inputs."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'runtime/bundle'))
from retrain.preprocessing import PreparedImageCache

spec = importlib.util.spec_from_file_location('f05_r270_cache', root / 'tools/diagnostics/f05_r270_cache.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class R270CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        data, cache_root = base / 'data', base / 'cache'
        (data / 'train').mkdir(parents=True)
        rows = []
        for name in ('a.png', 'b.png'):
            path = data / 'train' / name
            Image.new('RGB', (7, 9), (50, 80, 90)).save(path)
            rows.append({'relative_path': 'train/' + name,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.manifest = {'rows': rows, 'fit_indices': [0], 'dev_indices': [1]}
        with mock.patch.dict(os.environ, {'TRAIN_IMAGE_CACHE_READONLY': '0'}):
            writable = PreparedImageCache(data, cache_root, 16)
            for row in rows:
                writable.read(row['relative_path'], row['sha256'])
        with mock.patch.dict(os.environ, {'TRAIN_IMAGE_CACHE_READONLY': '1'}):
            self.cache = PreparedImageCache(data, cache_root, 16)
        self.index = []
        for number, row in enumerate(rows):
            key = hashlib.sha256((row['relative_path'] + '\0' + row['sha256']).encode()).hexdigest()
            sidecar = self.cache.directory / key[:2] / (key + '.json')
            self.index.append({'manifest_index': number, 'key': key,
                               'receipt_sha256': hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                               'array_sha256': json.loads(sidecar.read_text())['array_sha256']})

    def test_complete_fit_dev_index_and_readonly_cache_pass(self):
        entries = checker.validate_index(self.manifest, self.index, expected_count=2)
        self.assertEqual(checker.verify_rows(self.manifest, entries, self.cache, workers=2), 2)

    def test_duplicate_missing_and_official_rows_are_rejected(self):
        for broken in (self.index[:1], [self.index[0], self.index[0]]):
            with self.subTest(index=broken), self.assertRaises(ValueError):
                checker.validate_index(self.manifest, broken, expected_count=2)
        self.manifest['rows'][1]['relative_path'] = 'valid/image.png'
        with self.assertRaises(ValueError):
            checker.validate_index(self.manifest, self.index, expected_count=2)

    def test_changed_cached_array_or_sidecar_is_rejected(self):
        entries = checker.validate_index(self.manifest, self.index, expected_count=2)
        entry = entries[0]
        path = self.cache.directory / entry['key'][:2] / (entry['key'] + '.npy')
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaises(ValueError):
            checker.verify_rows(self.manifest, entries, self.cache, workers=1)

    def test_writable_cache_or_changed_historical_receipt_is_rejected(self):
        entries = checker.validate_index(self.manifest, self.index, expected_count=2)
        altered = [dict(row) for row in entries]
        altered[0]['receipt_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            checker.verify_rows(self.manifest, altered, self.cache, workers=1)
        self.cache.readonly = False
        with self.assertRaises(ValueError):
            checker.verify_rows(self.manifest, entries, self.cache, workers=1)

    def test_changed_cache_core_ast_is_rejected(self):
        source = Path(self.temporary.name) / 'source'
        file = source / 'runtime/bundle/retrain/preprocessing.py'
        file.parent.mkdir(parents=True)
        file.write_text('def prepare_image(value):\n    return value\n')
        node = ast.parse(file.read_text()).body[0]
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        key = 'runtime/bundle/retrain/preprocessing.py:prepare_image'
        self.assertEqual(checker._core_matches(source, {key: digest}), 1)
        file.write_text('def prepare_image(value):\n    return value + 1\n')
        with self.assertRaises(ValueError):
            checker._core_matches(source, {key: digest})


if __name__ == '__main__':
    unittest.main()

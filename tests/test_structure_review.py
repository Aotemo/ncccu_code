"""Blinding and image identity checks, run only on the guarded compute Pod."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
spec = importlib.util.spec_from_file_location('structure_review', ROOT / 'tools/diagnostics/structure_review.py')
review = importlib.util.module_from_spec(spec) if spec and spec.loader and Path(spec.origin).exists() else None
if review:
    spec.loader.exec_module(review)


class StructureReviewTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(review, 'blinded structure packet implementation is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'data/train').mkdir(parents=True)
        self.rows = []
        for i in range(2):
            name = f'private-label-{i}-hard.png'
            path = self.root / 'data/train' / name
            image = Image.new('RGB', (680, 520), (45 + i * 50, 80, 100))
            image.save(path)
            self.rows.append({'name': name, 'label': i, 'relative_path': 'train/' + name,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    def packet(self):
        return review.prepare_packet(self.rows, [1, 0], self.root / 'data', self.root / 'packet')

    def annotations(self, packet):
        fields = review.REVIEW_FIELDS
        return [{'alias': row['alias'], **{field: 'unknown' for field in fields},
                 'notes': 'Cannot infer label from technical visibility.'} for row in packet['cases']]

    def test_packet_hides_identity_and_reproduces_actual_input(self):
        packet = self.packet()
        visible = self.root / 'packet/reviewer'
        text = (visible / 'PACKET.json').read_text()
        for secret in ('private-label', 'relative_path', 'hard', 'label', 'score', 'group'):
            self.assertNotIn(secret, text)
        self.assertEqual([row['alias'] for row in packet['cases']], ['B001', 'B002'])
        from retrain.preprocessing import prepare_image
        from retrain.data import ImageDataset
        dataset = ImageDataset(str(self.root / 'data'), [], 512, False, normalization='flair_rgb01')
        with Image.open(self.root / 'data' / self.rows[1]['relative_path']) as raw:
            actual = dataset._image_tensor(prepare_image(raw, 512), None)
        with Image.open(visible / packet['cases'][0]['prepared']['path']) as exported:
            pixels = np.asarray(exported).copy()
        expected = np.rint(actual.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        np.testing.assert_array_equal(pixels, expected)
        self.assertEqual(pixels.shape, (512, 512, 3))

    def test_wrong_original_sha_and_official_source_are_rejected(self):
        self.rows[0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.packet()
        self.rows[0]['relative_path'] = 'valid/unseen.png'
        with self.assertRaisesRegex(ValueError, 'train'):
            review.prepare_packet(self.rows, [0], self.root / 'data', self.root / 'second')

    def test_review_is_sealed_before_unblinding_and_must_be_complete(self):
        packet = self.packet()
        notes = self.annotations(packet)
        packet_root = self.root / 'packet'
        for invalid in (notes[:1], [notes[0], notes[0]]):
            with self.assertRaisesRegex(ValueError, 'complete|unique'):
                review.seal_review(packet_root, invalid, 'AI-assisted; not clinical expert')
            self.assertFalse((packet_root / 'BLIND-REVIEW.json').exists())
            self.assertFalse((packet_root / 'UNBLINDED-REVIEW.json').exists())
        result = review.seal_review(packet_root, notes, 'AI-assisted; not clinical expert')
        sealed = packet_root / 'BLIND-REVIEW.json'
        self.assertEqual(result['blind_review_sha256'], hashlib.sha256(sealed.read_bytes()).hexdigest())
        self.assertEqual(result['cases'][0]['name'], self.rows[1]['name'])
        self.assertEqual(result['cases'][0]['label_review'], 'uncertain')
        self.assertFalse(result['expert_review'])
        with self.assertRaises(FileExistsError):
            review.seal_review(packet_root, notes, 'same reviewer')

    def test_changed_review_image_cannot_be_unblinded(self):
        packet = self.packet()
        target = self.root / 'packet/reviewer' / packet['cases'][0]['prepared']['path']
        target.write_bytes(b'changed input')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            review.seal_review(self.root / 'packet', self.annotations(packet), 'AI-assisted')
        self.assertFalse((self.root / 'packet/BLIND-REVIEW.json').exists())


if __name__ == '__main__':
    torch.set_num_threads(int(__import__('os').environ['NCCCU_CPU_THREADS']))
    torch.set_num_interop_threads(1)
    unittest.main()

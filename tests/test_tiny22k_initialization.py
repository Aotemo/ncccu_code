"""Only the pinned Tiny22K conversion may enter the generic public loader."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain.models import validate_initialization

WEIGHT = '1ec1dd0b7045237eafe1c8f1becd58456cc811326c97ec416ae8b2e5ede860ed'
PROVENANCE = '40c14c8bd780d4feb88ae96d7e9def1ef78499daf93ae4412eff78502780233f'
META = {'model': 'convnext_tiny', 'format': 'torchvision-1000-state-dict-v1',
        'source_url': 'https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k/resolve/bc48a87f119bae1b06418f5340bfa97ef75d4609/pytorch_model.bin',
        'revision': 'bc48a87f119bae1b06418f5340bfa97ef75d4609',
        'training_data': 'ImageNet-22K pretraining -> ImageNet-1K fine-tuning',
        'original_sha256': '8cb65dc650a7fb25eaadbff3eb69559d02f83cdc5afdfb3785d1889cdc3e3518',
        'sha256': WEIGHT, 'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6,
        'license_file': '../original/LICENSE-apache-2.0.txt', 'redistribution_confirmed': True}


class Tiny22KInitializationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'converted').mkdir(); (self.root / 'original').mkdir()
        (self.root / 'original/LICENSE-apache-2.0.txt').write_text('fixture license text')
        self.weight = self.root / 'converted/weight.pt'; self.weight.write_bytes(b'fixture')
        self.provenance = self.root / 'converted/provenance.json'
        self.init = {'kind': 'imagenet', 'path': str(self.weight), 'provenance': str(self.provenance),
                     'sha256': WEIGHT, 'format': META['format']}

    def validate(self, meta, proof_sha=PROVENANCE):
        self.provenance.write_text(json.dumps(meta))
        # This unit check exercises policy only. Real hashes/strict tensor loading
        # are separately required on the public object in the integration gate.
        def digest(path):
            return WEIGHT if Path(path) == self.weight else proof_sha if Path(path) == self.provenance else 'a' * 64
        with patch('retrain.models._hash', side_effect=digest):
            return validate_initialization(self.init)

    def test_exact_pinned_source_is_accepted(self):
        self.assertEqual(self.validate(META)['final_norm_eps'], 1e-6)

    def test_other_revision_ln_model_or_source_is_rejected(self):
        for key, value in [('revision', '0' * 40), ('final_norm_eps', 1e-12),
                           ('internal_norm_eps', 1e-12), ('model', 'convnext_small'),
                           ('training_data', 'ImageNet-22K'), ('source_url', 'https://example.invalid/x')]:
            bad = copy.deepcopy(META); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(bad)

    def test_different_provenance_blob_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(META, proof_sha='0' * 64)


if __name__ == '__main__':
    unittest.main()

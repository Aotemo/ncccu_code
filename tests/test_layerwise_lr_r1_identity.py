"""Catch a new fresh identity rejected or routed outside the existing layerwise checks."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex


class LayerwiseR1IdentityTest(unittest.TestCase):
    def test_new_fresh_identity_preserves_science_and_existing_formal_route(self):
        old = json.loads((ROOT / 'configs/R130-SCIENCE.json').read_text())
        new = json.loads((ROOT / 'configs/R130R1-SCIENCE.json').read_text())
        for filename, order, candidate in (
            ('R130-SCIENCE.json', 40, 'r130-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010'),
            ('R130R1-SCIENCE.json', 41, 'r130r1-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010'),
        ):
            path = ROOT / 'configs' / filename
            declared = json.loads(path.read_text())
            site = {'schema_version': 1, 'scope': 'cpu_technical_prep',
                'round_id': 'identity-fixture', 'git_commit': '0' * 40,
                'work_root': str(ROOT), 'round_root': str(ROOT / 'rounds/identity-fixture'),
                'pod': {'namespace': 'fixture', 'name': 'fixture',
                    'uid': '00000000-0000-0000-0000-000000000001',
                    'container': 'fixture', 'python': sys.executable},
                'gpus': [{'ordinal': i, 'pci': f'0000:{i+1:02x}:00.0',
                          'vram_total_bytes': 64 * 1024**3} for i in range(8)],
                **{key: str(ROOT / relative) for key, relative in declared['paths'].items()},
                'science': {'binding_version': declared['candidate_profile'],
                    'candidate': declared['candidate'], 'configuration': {
                    'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
            with tempfile.TemporaryDirectory() as temporary:
                site_path = Path(temporary) / 'SITE_BINDING.json'
                site_path.write_text(json.dumps(site))
                with patch.dict(os.environ, NCCCU_SITE_BINDING=str(site_path),
                                NCCCU_SITE_BINDING_SHA256=hashlib.sha256(site_path.read_bytes()).hexdigest()):
                    try:
                        science = ex.science_configuration()
                        actual = ex.candidate_from_configuration(science)
                    except ValueError as error:
                        self.fail(f'accepted fresh identity {filename} rejected: {error}')
            self.assertEqual((actual['id'], actual['experiment_order']), (candidate, order))
        left, right = copy.deepcopy(old), copy.deepcopy(new)
        for value in (left, right):
            value.pop('candidate_profile')
            value.pop('comparison_note')
            value['candidate'].pop('id')
            value['candidate'].pop('experiment_order')
        self.assertEqual(left, right)
        self.assertIsNone(new['job']['initial_checkpoint'])
        self.assertEqual(new['job']['seed'], 20261029)
        spec = importlib.util.spec_from_file_location('r1_formal_candidate', ROOT / 'tools/validation/check_candidate.py')
        formal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(formal)
        self.assertTrue(formal.layerwise_profile(old['candidate_profile']))
        self.assertTrue(formal.layerwise_profile(new['candidate_profile']))


if __name__ == '__main__':
    unittest.main()

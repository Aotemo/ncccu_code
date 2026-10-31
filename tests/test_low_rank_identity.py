"""One adopted R131 identity check; algorithm toys and formal are separate evidence."""
import copy
import hashlib
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
from retrain.low_rank import LOW_RANK_RULE


class LowRankIdentityTest(unittest.TestCase):
    def test_R131_exact_identity_direct_R107_science_and_public_initial_pin(self):
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        current = json.loads((ROOT / 'configs/R131-SCIENCE.json').read_text())
        initial = json.loads((ROOT / 'records/preparation/r131-public-fresh/PUBLIC-FRESH-INITIAL.json').read_text())
        for filename, expected_order in (('R130R1-SCIENCE.json', 41), ('R131-SCIENCE.json', 42)):
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
            with tempfile.TemporaryDirectory() as folder:
                site_path = Path(folder) / 'SITE_BINDING.json'
                site_path.write_text(json.dumps(site))
                with patch.dict(os.environ, NCCCU_SITE_BINDING=str(site_path),
                                NCCCU_SITE_BINDING_SHA256=hashlib.sha256(site_path.read_bytes()).hexdigest()):
                    actual = ex.candidate_from_configuration(ex.science_configuration())
            self.assertEqual((actual['id'], actual['experiment_order']),
                             (declared['candidate']['id'], expected_order))
        self.assertEqual(current['candidate_profile'], ex.POINTWISE_LORA8_SMALL_BINDING)
        self.assertEqual(current['job']['low_rank_policy'], LOW_RANK_RULE)
        self.assertEqual(current['candidate']['low_rank_policy'], LOW_RANK_RULE)
        self.assertIsNone(current['job'].get('layerwise_lr_policy'))
        self.assertIsNone(current['job']['initial_checkpoint'])
        self.assertEqual(current['official_evaluation'], ex.OFFICIAL_GATE_0996)
        self.assertEqual(current['initial_model_tensor_sha256'], initial['new_489_tensor_sha256'])
        self.assertEqual(original['initial_model_tensor_sha256'], initial['common_345_tensor_sha256'])
        self.assertNotEqual(current['initial_model_tensor_sha256'], original['initial_model_tensor_sha256'])
        left, right = copy.deepcopy(original), copy.deepcopy(current)
        for value in (left, right):
            for key in ('candidate_profile', 'comparison_note', 'initial_model_tensor_sha256', 'official_evaluation'):
                value.pop(key, None)
            for key in ('id', 'experiment_order', 'low_rank_policy'):
                value['candidate'].pop(key, None)
            value['job'].pop('low_rank_policy', None)
        self.assertEqual(left, right)


if __name__ == '__main__':
    unittest.main()

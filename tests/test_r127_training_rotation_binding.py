"""R127's non-tensor rotation policy binds the frozen job to the dataset."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime' / 'bundle'))
from retrain import experiment91 as ex
from retrain.data import ImageDataset


class R127RotationBindingTest(unittest.TestCase):
    def test_science_job_candidate_and_dataset_policy(self):
        baseline = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        science = json.loads((ROOT / 'configs/R127-SCIENCE.json').read_text())
        self.assertEqual(science['candidate_profile'], ex.ROTATION_IDENTITY_SMALL_BINDING)
        self.assertEqual(science['candidate']['training_rotation'], 'identity-v1')
        self.assertEqual(science['job']['training_rotation'], 'identity-v1')
        self.assertEqual(science['job']['loss'], ex.PAIR_LOGISTIC_LOSS)
        self.assertEqual(science['official_evaluation'], ex.OFFICIAL_GATE_0996)
        self.assertEqual(ex.candidate_from_configuration(science), science['candidate'])

        remaining = copy.deepcopy(science)
        original = copy.deepcopy(baseline)
        for key in ('candidate_profile', 'comparison_note', 'official_evaluation'):
            remaining.pop(key, None)
            original.pop(key, None)
        for key in ('id', 'experiment_order', 'training_rotation'):
            remaining['candidate'].pop(key, None)
            original['candidate'].pop(key, None)
        remaining['job'].pop('training_rotation')
        self.assertEqual(remaining, original)

        site = {'gpus': list(range(8)), 'public_weight': '/tmp/r127-public-weight',
                'public_provenance': '/tmp/r127-public-provenance'}
        age = {'enabled': True}
        binding = {'version': ex.ROTATION_IDENTITY_SMALL_BINDING}
        job = copy.deepcopy(science['job'])
        job.update(physical_batch=8, age=age, experiment91=binding,
                   initialization={'kind': 'imagenet',
                                   'sha256': science['public_source']['sha256'],
                                   'format': science['public_source']['format'],
                                   'path': site['public_weight'],
                                   'provenance': site['public_provenance']})
        restored = json.loads(json.dumps(job))
        with patch.object(ex, 'science_configuration', return_value=science), \
                patch.object(ex, 'candidate', return_value=science['candidate']), \
                patch.object(ex, 'load_site', return_value=site), \
                patch.object(ex, 'binding', return_value=binding), \
                patch.object(ex, 'resolved_age_spec', return_value=age), \
                patch.object(ex, 'public_source', return_value=science['public_source']):
            self.assertEqual(ex.validate_job(restored), restored)
            omitted = copy.deepcopy(restored)
            omitted.pop('training_rotation')
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                ex.validate_job(omitted)

        with patch('retrain.data.configured_cache', return_value=None):
            training = ImageDataset(Path('.'), [], job['image_size'], training=True,
                                    training_scale=job['training_scale'],
                                    training_rotation=restored['training_rotation'])
            evaluation = ImageDataset(Path('.'), [], job['image_size'], training=False,
                                      training_rotation=restored['training_rotation'])
        self.assertEqual(training.training_rotation, 'identity-v1')
        self.assertEqual(evaluation.training_rotation, 'identity-v1')
        self.assertIsNone(training.training_photometric)


if __name__ == '__main__':
    unittest.main()

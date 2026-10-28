"""The explicit source must determine final LN; absent bindings never select R92."""
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
from retrain import experiment91 as experiment

TINY22 = {
    'id': 'r95-convnext-tiny-imagenet22k-to-1k-24k-earlystop-9010', 'experiment_order': 5,
    'model': 'convnext_tiny', 'public_revision': 'bc48a87f119bae1b06418f5340bfa97ef75d4609',
    'readout': 'native_avgpool_final_LN_dropout_linear_plus_age', 'features': 768,
    'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6, 'batch_norm_modules': 0,
    'batch_norm_buffers': 0, 'attention_pool': None,
    'weight_sha256': '1ec1dd0b7045237eafe1c8f1becd58456cc811326c97ec416ae8b2e5ede860ed',
    'provenance_sha256': '40c14c8bd780d4feb88ae96d7e9def1ef78499daf93ae4412eff78502780233f',
    'stochastic_depth': .1, 'head_initialization': 'torch.nn.Linear_native', 'head_lr_scope': 'last_linear',
}


class CandidateBindingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.configuration = root / 'configs/R95-SCIENCE.json'
        self.configuration.parent.mkdir()
        self.configuration.write_bytes((ROOT / 'configs/R95-SCIENCE.json').read_bytes())
        self.site = {'schema_version': 1, 'round_id': 'candidate-fixture', 'git_commit': 'b' * 40,
                     'work_root': str(root), 'round_root': str(root / 'rounds/fixture'),
                     'pod': {'namespace': 'ns', 'name': 'pod', 'container': 'container',
                             'uid': '11111111-2222-4333-8444-555555555555', 'python': sys.executable},
                     'gpus': [{'ordinal': n, 'pci': '0000:%02x:00.0' % (n + 1),
                               'vram_total_bytes': 68702699520} for n in range(8)],
                     'metadata_root': str(root / 'stage1'), 'public_weight': str(root / 'public/weight.pt'),
                     'public_provenance': str(root / 'public/provenance.json'),
                     'science': {'binding_version': 'single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-v1',
                                 'candidate': copy.deepcopy(TINY22),
                                 'configuration': {'path': str(self.configuration),
                                     'sha256': hashlib.sha256(self.configuration.read_bytes()).hexdigest()}}}
        self.path = root / 'SITE_BINDING.json'
        self.path.write_text(json.dumps(self.site))
        self.environment = patch.dict(os.environ, {'NCCCU_SITE_BINDING': str(self.path)}, clear=False)
        self.environment.start(); self.addCleanup(self.environment.stop)
        os.environ.pop('NCCCU_SITE_BINDING_SHA256', None)

    def test_candidate_uses_explicit_source_identity_and_ln(self):
        self.assertEqual(experiment.candidate(), TINY22)

    def test_R94_only_changes_backbone_LR_scientific_fields(self):
        old = json.loads((ROOT / 'configs/R93-SCIENCE.json').read_text())
        new = json.loads((ROOT / 'configs/R94-SCIENCE.json').read_text())
        self.assertEqual({key for key in old['job'] if old['job'][key] != new['job'][key]}, {'backbone_lr'})
        self.assertEqual(new['job']['backbone_lr'], old['job']['backbone_lr'] / 2)
        for key in ('paths', 'data_split', 'schedule', 'official_manifest_sha256'):
            self.assertEqual(old[key], new[key])

    def test_R94_configuration_controls_job_LR_and_rejects_old_R93_LR(self):
        science = experiment.science_configuration()
        job = copy.deepcopy(science['job'])
        job.update(age={'fixture': 'unchanged-age'}, experiment91={'fixture': 'binding'},
                   initialization={'kind': 'imagenet', 'sha256': experiment.WEIGHT_SHA,
                       'format': 'torchvision-1000-state-dict-v1', 'path': self.site['public_weight'],
                       'provenance': self.site['public_provenance']})
        with patch.object(experiment, 'metadata', return_value={'AGE.json': job['age']}), \
                patch.object(experiment, 'binding', return_value=job['experiment91']):
            self.assertEqual(experiment.validate_job(job)['backbone_lr'], 1e-4)
            wrong = copy.deepcopy(job); wrong['backbone_lr'] = 2e-4
            with self.assertRaises(ValueError):
                experiment.validate_job(wrong)

    def test_old_R93_profile_is_not_accepted_with_new_LR(self):
        bad = copy.deepcopy(self.site)
        bad['science']['binding_version'] = 'single-tiny22k-bc48a87f-fixed-9010-v1'
        self.path.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            experiment.candidate()
        self.path.write_text(json.dumps(self.site))
        value = json.loads(self.configuration.read_text())
        value['candidate_profile'] = bad['science']['binding_version']
        self.configuration.write_text(json.dumps(value))
        self.site['science']['configuration']['sha256'] = hashlib.sha256(self.configuration.read_bytes()).hexdigest()
        self.path.write_text(json.dumps(self.site))
        with self.assertRaises(ValueError):
            experiment.science_configuration()

    def test_same_length_configuration_change_invalidates_pin(self):
        original = self.configuration.read_text()
        self.configuration.write_text(original.replace('0.0001', '0.0002', 1))
        with self.assertRaises(ValueError):
            experiment.candidate()

    def test_parameter_group_LRs_follow_configured_job_and_age_contract(self):
        location = ROOT / 'tools/validation/check_candidate.py'
        spec = importlib.util.spec_from_file_location('candidate_LR_checker', location)
        checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
        job = experiment.science_configuration()['job']
        job['age'] = {'contract': {'lr': 1e-3}}
        groups = [{'role': role, 'lr': lr, 'initial_lr': lr}
                  for role, lr in [('backbone', 1e-4), ('head', 1e-3), ('age', 1e-3)]]
        self.assertTrue(checker.group_lrs_match(groups, job))
        for key in ('lr', 'initial_lr'):
            wrong = copy.deepcopy(groups); wrong[0][key] = 2e-4
            with self.subTest(key=key):
                self.assertFalse(checker.group_lrs_match(wrong, job))

    def test_absent_site_has_no_legacy_candidate_fallback(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            experiment.candidate()

    def test_wrong_source_ln_or_weight_is_rejected(self):
        for key, value in [('final_norm_eps', 1e-12), ('weight_sha256', '0' * 64),
                           ('provenance_sha256', '0' * 64), ('public_revision', '0' * 40)]:
            bad = copy.deepcopy(self.site); bad['science']['candidate'][key] = value
            self.path.write_text(json.dumps(bad))
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.candidate()

    def test_actual_normalization_accepts_source_ln_and_rejects_old_final_ln(self):
        value = {'candidate': TINY22, 'batch_norm_modules': 0, 'buffer_count': 0,
                 'buffer_sha256': hashlib.sha256(b'').hexdigest(),
                 'head_in_features': 768, 'head_out_features': 1, 'dropout': .2,
                 'layer_norms': [{'name': 'network.classifier.0', 'eps': 1e-6},
                                 {'name': 'network.features.0.1', 'eps': 1e-6}],
                 'stochastic_depth': [{'name': 'first', 'mode': 'row', 'p': 0.0},
                                      {'name': 'last', 'mode': 'row', 'p': .1}]}
        self.assertTrue(experiment.validate_normalization(value))
        bad = copy.deepcopy(value); bad['layer_norms'][0]['eps'] = 1e-12
        with self.assertRaises(ValueError):
            experiment.validate_normalization(bad)


if __name__ == '__main__':
    unittest.main()

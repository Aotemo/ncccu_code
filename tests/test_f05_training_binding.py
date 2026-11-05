"""Synthetic-only OOF provenance, fit-index binding and training handoff."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/bundle'), str(ROOT / 'tools/diagnostics'), str(ROOT / 'tests')]

import f05_oof_targets as targets
from test_f05_oof_targets import synthetic_fixture
from retrain import config, data, experiment91, pipeline
from retrain.persistence import canonical_digest


def write_json(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode('utf-8') + b'\n'
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def prepared_fixture(root):
    root = Path(root)
    source = synthetic_fixture(root)
    base = json.loads(Path(source['base']['path']).read_text())
    folds = json.loads(Path(source['folds']['path']).read_text())
    verified = [targets.verify_fold_input(source['base'], source['folds'], parent,
                                          work_root=root, allow_fixture=True) for parent in source['parents']]
    results, refs = [], []
    for number, fold in enumerate(verified):
        calibration, probabilities = targets.fit_fold_calibration(fold)
        reference = write_json(root / ('CALIBRATION-' + str(number) + '.json'), calibration)
        refs.append(reference)
        results.append({'verified': fold, 'calibration': calibration,
                        'calibration_sha256': reference['sha256'], 'probabilities': probabilities})
    clean, distilled, aggregate = targets.build_target_tables(base, folds, results,
        base_sha=source['base']['sha256'], folds_sha=source['folds']['sha256'])
    clean_ref = write_json(root / 'CLEAN-TARGETS-private.json', clean)
    distill_ref = write_json(root / 'DISTILL-TARGETS-private.json', distilled)
    summary = {**aggregate, 'status': 'completed_fit_only_oof_targets', 'fixture_only': True,
        'inputs': {'base': source['base'], 'folds': source['folds'], 'parents': source['parents']},
        'calibrations': refs, 'heldout_score_sha256': [value['heldout_score_sha256'] for value in verified],
        'clean_targets': clean_ref, 'distill_targets': distill_ref}
    return base, source, write_json(root / 'SUMMARY.json', summary)


class F05TrainingBindingTest(unittest.TestCase):
    def test_registered_science_pins_summary_seed_and_fresh_identity(self):
        baseline = json.loads((ROOT / 'configs/R120-SCIENCE.json').read_text())
        for mode, profile, seed, initial, round_id in (
            ('clean', config.F05_CLEAN_PROFILE, 20261201,
             '6e2e8f4b20c211d3ed446187ddc9ea4803e1d560ed1741b00381213431ae7001',
             'r268-small-f05clean-seed20261201-pairsigmoid-24k-p4-9010-20261105'),
            ('distill', config.F05_DISTILL_PROFILE, 20261202,
             'b14f7c368ab70e6591c25ea6d9ab4c67b563876bb10dcd2050d75127e7d0da2c',
             'r269-small-f05distill-seed20261202-pairsigmoid-24k-p4-9010-20261105'),
        ):
            value = copy.deepcopy(baseline)
            value['candidate_profile'] = profile
            value['job']['seed'] = seed
            value['initial_model_tensor_sha256'] = initial
            value['f05_targets'] = {'version': 'f05-oof-training-v1', 'mode': mode,
                'summary_relative_path': config.F05_TRAIN_SUMMARY_RELATIVE, 'summary_sha256': 'b' * 64}
            label, _, order = experiment91.PAIR_SIGMOID_SEED_REPLICAS[profile]
            value['candidate']['id'] = f'{label.lower()}-convnext-small-imagenet22k-to-1k-24k-p4-seed{seed}-scale085-pairsigmoid-9010'
            value['candidate']['experiment_order'] = order
            self.assertEqual(experiment91.candidate_from_configuration(value), value['candidate'])
            self.assertEqual(experiment91.validate_f05_training_science(value, {'round_id': round_id}), value['f05_targets'])
            if round_id.startswith('r268-'):
                alias = round_id.replace('r268-', 'r268-r1-', 1)
                self.assertEqual(experiment91.validate_f05_training_science(value, {'round_id': alias}), value['f05_targets'])
            for mutation in ('summary', 'mode', 'seed', 'initial', 'round', 'dropout', 'schedule', 'patience'):
                broken = copy.deepcopy(value); site = {'round_id': round_id}
                if mutation == 'summary': broken['f05_targets']['summary_relative_path'] += '/../bad'
                if mutation == 'mode': broken['f05_targets']['mode'] = 'unknown'
                if mutation == 'seed': broken['job']['seed'] += 1
                if mutation == 'initial': broken['initial_model_tensor_sha256'] = '0' * 64
                if mutation == 'round': site['round_id'] += '-other'
                if mutation == 'dropout': broken['job']['dropout'] = 0.
                if mutation == 'schedule': broken['schedule']['total_updates'] -= 800
                if mutation == 'patience': broken['early_stopping']['patience'] = 3
                with self.assertRaises(ValueError, msg=mutation):
                    experiment91.validate_f05_training_science(broken, site)

    def test_templates_keep_r120_science_and_bind_only_pending_summary(self):
        base = json.loads((ROOT / 'configs/R120-SCIENCE.json').read_text())
        for number, mode, seed in ((268, 'clean', 20261201), (269, 'distill', 20261202)):
            for pod in ('a', 'b'):
                path = ROOT / f'configs/templates/R{number}-{pod}-SCIENCE.json.in'
                value = json.loads(path.read_text())
                self.assertEqual(value['f05_targets']['summary_sha256'], '__OOF_SUMMARY_SHA256__')
                with self.assertRaises(ValueError):
                    experiment91.validate_f05_training_science(value)
                value['f05_targets']['summary_sha256'] = 'b' * 64  # synthetic only
                self.assertEqual(value['f05_targets']['mode'], mode)
                self.assertEqual(value['job']['seed'], seed)
                job = copy.deepcopy(value['job']); job['seed'] = base['job']['seed']
                self.assertEqual(job, base['job'])
                for field in ('public_source', 'structure', 'paths', 'schedule', 'early_stopping', 'data_split'):
                    self.assertEqual(value[field], base[field])
                expected_asset = json.loads((ROOT / f'configs/R271-{pod.upper()}-SCIENCE.json').read_text())['public_asset_receipt']
                self.assertEqual(value['public_asset_receipt'], expected_asset)
                self.assertEqual(experiment91.candidate_from_configuration(value), value['candidate'])
                self.assertEqual(experiment91.validate_f05_training_science(value), value['f05_targets'])

    def test_prepare_binds_reconstructed_targets_into_package(self):
        specification = importlib.util.spec_from_file_location('f05_prepare_test', ROOT / 'tools/activation/prepare.py')
        prepare = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(prepare)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / config.F05_TRAIN_SUMMARY_RELATIVE
            output.parent.mkdir(parents=True)
            base, source, summary_ref = prepared_fixture(output.parent)
            actual_verify = data.verify_f05_oof_for_training
            def verify_fixture(reference, work_root, *, verifier):
                return actual_verify(reference, work_root, verifier=verifier, allow_fixture=True,
                    fit_count=len(base['fit_indices']), dev_count=len(base['dev_indices']))
            site = {'work_root': str(root), 'round_root': str(root / 'formal')}
            package = root / 'package'
            with mock.patch.object(data, 'F05_BASE_MANIFEST_SHA', source['base']['sha256']), \
                 mock.patch.object(data, 'F05_THREE_FOLD_SHA', source['folds']['sha256']), \
                 mock.patch.object(experiment91, 'binding', return_value={'manifest_digest': canonical_digest(base)}), \
                 mock.patch.object(experiment91, 'validate_f05_training_science',
                    return_value={'mode': 'clean', 'summary_relative_path': config.F05_TRAIN_SUMMARY_RELATIVE,
                                  'summary_sha256': summary_ref['sha256']}), \
                 mock.patch.object(data, 'verify_f05_oof_for_training', side_effect=verify_fixture):
                binding = prepare.prepare_f05_targets(ROOT, site, {}, package)
            proof = package / 'evidence/F05-OOF-VERIFICATION.json'
            self.assertTrue(proof.is_file())
            self.assertEqual(binding['verification']['sha256'], hashlib.sha256(proof.read_bytes()).hexdigest())
            self.assertEqual(binding['summary'], summary_ref)
            self.assertEqual(binding['target']['path'], str(output.parent / 'CLEAN-TARGETS-private.json'))
            self.assertEqual(binding['verification']['path'], str(root / 'formal/evidence/F05-OOF-VERIFICATION.json'))

    def test_prepared_job_loads_through_real_config_and_science_validation(self):
        for number, mode in ((268, 'clean'), (269, 'distill')):
            science = json.loads((ROOT / f'configs/templates/R{number}-b-SCIENCE.json.in').read_text())
            science['f05_targets']['summary_sha256'] = 'b' * 64  # synthetic only
            seed = science['job']['seed']
            round_id = f'r{number}-small-f05{mode}-seed{seed}-pairsigmoid-24k-p4-9010-20261105'
            site = {'work_root': '/synthetic', 'round_root': '/synthetic/rounds/' + round_id,
                    'round_id': round_id, 'gpus': list(range(8)),
                    'public_weight': '/synthetic/weight.pt', 'public_provenance': '/synthetic/provenance.json'}
            age_contract = {'version': 'age-linear-v1', 'field': 'age', 'valid_min_exclusive': 0,
                'valid_max_inclusive': 120, 'features': ['standardized_age', 'missing_indicator'],
                'fit_count': 2, 'valid_fit_count': 2, 'missing_fit_count': 0,
                'median': 50., 'mean': 50., 'population_std': 10., 'lr': 1e-3,
                'weight_decay': 1e-4, 'parameter_count': 2, 'bias': False}
            age = {'enabled': True, 'contract': age_contract, 'contract_sha256': canonical_digest(age_contract)}
            binding = {'version': science['candidate_profile']}
            summary_path = '/synthetic/' + config.F05_TRAIN_SUMMARY_RELATIVE
            target_name = 'CLEAN-TARGETS-private.json' if mode == 'clean' else 'DISTILL-TARGETS-private.json'
            targets = {'version': 'f05-oof-training-v1', 'mode': mode,
                'summary': {'path': summary_path, 'sha256': 'b' * 64},
                'verification': {'path': site['round_root'] + '/evidence/F05-OOF-VERIFICATION.json', 'sha256': 'c' * 64},
                'target': {'path': str(Path(summary_path).parent / target_name), 'sha256': 'd' * 64}}
            job = copy.deepcopy(science['job'])
            job.update(age=age, experiment91=binding, f05_targets=targets,
                initialization={'kind': 'imagenet', 'format': science['public_source']['format'],
                    'path': site['public_weight'], 'provenance': site['public_provenance'],
                    'sha256': science['public_source']['sha256']})
            raw = {'schema_version': 1, 'mode': 'train', 'resume': False, 'run_name': 'synthetic',
                   'output_root': '/synthetic/output', 'data': {'split': science['data_split']}, 'jobs': [job]}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'TRAIN_CONFIG.json'; path.write_text(json.dumps(raw))
                with mock.patch.object(experiment91, 'science_configuration', return_value=science), \
                     mock.patch.object(experiment91, 'load_site', return_value=site), \
                     mock.patch.object(experiment91, 'candidate', return_value=science['candidate']), \
                     mock.patch.object(experiment91, 'binding', return_value=binding), \
                     mock.patch.object(experiment91, 'resolved_age_spec', return_value=age), \
                     mock.patch.object(experiment91, 'public_source', return_value=science['public_source']):
                    cfg = config.load_config(path, environ={})
                    self.assertEqual(config.job_contract(cfg['jobs'][0])['f05_targets'], targets)
                    broken = copy.deepcopy(job); broken['f05_targets']['summary']['sha256'] = 'e' * 64
                    raw['jobs'] = [broken]; path.write_text(json.dumps(raw))
                    with self.assertRaises(ValueError):
                        config.load_config(path, environ={})

    def test_legacy_pipeline_refuses_f05_before_loading_any_data(self):
        cfg = {'mode': 'benchmark', 'output_root': '/synthetic', 'run_name': 'test',
               'data': {'split': {'kind': 'holdout', 'include_valid': False}, 'root': '/synthetic',
                        'identity_path': '/synthetic/identity', 'expected_count': 1},
               'jobs': [{'f05_targets': {'mode': 'clean'}}]}
        plan = {'stages': {'benchmark': '/synthetic/config'}, 'outputs': {'benchmark': '/synthetic/test'}}
        with mock.patch.object(config, 'load_config', return_value=cfg), \
             mock.patch.object(data, 'load_source', side_effect=AssertionError('must not read data')):
            with self.assertRaisesRegex(ValueError, 'F05'):
                pipeline.stage_expectations(plan, 'synthetic')

    def test_single_process_entry_refuses_f05_before_inspection(self):
        import train
        cfg = {'jobs': [{'f05_targets': {'mode': 'distill'}}]}
        with mock.patch.object(train, 'environment_report', side_effect=AssertionError('must not inspect')):
            with self.assertRaisesRegex(ValueError, 'F05'):
                train.run(cfg)

    def test_training_policy_is_explicit_pinned_and_rejects_wrong_fields(self):
        self.assertTrue(hasattr(config, 'validate_f05_training_binding'))
        ref = {'path': '/synthetic/receipt.json', 'sha256': 'a' * 64}
        value = {'version': 'f05-oof-training-v1', 'mode': 'clean',
                 'summary': ref, 'verification': ref, 'target': ref}
        self.assertEqual(config.validate_f05_training_binding(value), value)
        for change in ({'mode': 'other'}, {'version': 'other'}, {'extra': 1},
                       {'target': {'path': '/synthetic/target', 'sha256': 'bad'}}):
            invalid = copy.deepcopy(value); invalid.update(change)
            with self.assertRaises(ValueError):
                config.validate_f05_training_binding(invalid)

    def test_oof_is_independently_reconstructed_then_both_modes_load(self):
        self.assertTrue(hasattr(data, 'verify_f05_oof_for_training'))
        self.assertTrue(hasattr(data, 'load_f05_training_targets'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, source, summary = prepared_fixture(root)
            with mock.patch.object(data, 'F05_BASE_MANIFEST_SHA', source['base']['sha256']), \
                 mock.patch.object(data, 'F05_THREE_FOLD_SHA', source['folds']['sha256']):
                proof = data.verify_f05_oof_for_training(summary, root, verifier=targets, allow_fixture=True,
                    fit_count=len(base['fit_indices']), dev_count=len(base['dev_indices']))
                proof_ref = write_json(root / 'F05-TARGET-VERIFICATION.json', proof)
                for mode in ('clean', 'distill'):
                    binding = {'version': 'f05-oof-training-v1', 'mode': mode,
                               'summary': summary, 'verification': proof_ref, 'target': proof['targets'][mode]}
                    values = data.load_f05_training_targets(binding, base, work_root=root, allow_fixture=True,
                        fit_count=len(base['fit_indices']), dev_count=len(base['dev_indices']))
                    self.assertEqual(set(values), set(base['fit_indices']))
                    self.assertTrue(all(x in (.25, 1.) for x in values.values()) if mode == 'clean'
                                    else all(0 <= x <= 1 for x in values.values()))

    def test_changed_target_with_rewritten_summary_does_not_fake_lineage(self):
        self.assertTrue(hasattr(data, 'verify_f05_oof_for_training'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, source, summary_ref = prepared_fixture(root)
            summary = json.loads(Path(summary_ref['path']).read_text())
            table = json.loads(Path(summary['distill_targets']['path']).read_text())
            table['rows'][0]['soft_target'] = 1. - table['rows'][0]['soft_target']
            summary['distill_targets'] = write_json(Path(summary['distill_targets']['path']), table)
            summary_ref = write_json(Path(summary_ref['path']), summary)
            with mock.patch.object(data, 'F05_BASE_MANIFEST_SHA', source['base']['sha256']), \
                 mock.patch.object(data, 'F05_THREE_FOLD_SHA', source['folds']['sha256']):
                with self.assertRaises(ValueError):
                    data.verify_f05_oof_for_training(summary_ref, root, verifier=targets, allow_fixture=True,
                        fit_count=len(base['fit_indices']), dev_count=len(base['dev_indices']))

    def test_rank_reader_rejects_wrong_sha_and_missing_fit(self):
        self.assertTrue(hasattr(data, 'load_f05_training_targets'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, source, summary = prepared_fixture(root)
            with mock.patch.object(data, 'F05_BASE_MANIFEST_SHA', source['base']['sha256']), \
                 mock.patch.object(data, 'F05_THREE_FOLD_SHA', source['folds']['sha256']):
                proof = data.verify_f05_oof_for_training(summary, root, verifier=targets, allow_fixture=True,
                    fit_count=len(base['fit_indices']), dev_count=len(base['dev_indices']))
                proof_ref = write_json(root / 'F05-TARGET-VERIFICATION.json', proof)
                binding = {'version': 'f05-oof-training-v1', 'mode': 'clean', 'summary': summary,
                           'verification': proof_ref, 'target': proof['targets']['clean']}
                broken = copy.deepcopy(binding); broken['target']['sha256'] = 'f' * 64
                with self.assertRaises(ValueError):
                    data.load_f05_training_targets(broken, base, work_root=root, allow_fixture=True,
                        fit_count=len(base['fit_indices']), dev_count=len(base['dev_indices']))
                broken_base = copy.deepcopy(base); broken_base['fit_indices'].pop()
                with self.assertRaises(ValueError):
                    data.load_f05_training_targets(binding, broken_base, work_root=root, allow_fixture=True,
                        fit_count=len(base['fit_indices']), dev_count=len(base['dev_indices']))


if __name__ == '__main__':
    unittest.main()

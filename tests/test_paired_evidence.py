"""Independent ranking oracles and fail-closed selected-evidence contracts."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/diagnostics/paired_evidence.py'


def pinned(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return pinned(path)


def save_csv(path, rows, fields=('name', 'label', 'logit')):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return pinned(path)


def fixture(directory, with_fit=False):
    rows = [{'name': f'image-{i}.jpg', 'label': i % 2,
             'relative_path': f'train/image-{i}.jpg', 'sha256': str(i + 1) * 64}
            for i in range(6)]
    manifest = save_json(directory / 'manifest.json',
                         {'rows': rows, 'dev_indices': [0, 1, 2, 3], 'fit_indices': [4, 5]})
    result = {'manifest': manifest, 'models': [],
              'comparisons': [['candidate', 'baseline']],
              'bootstrap': {'replicates': 128, 'seed': 20261103, 'confidence': .95}}
    for name, scores, auc, ap, error in (
            ('baseline', [0., 1., 2., 3.], .75, 5 / 6, 2),
            ('candidate', [0., 2., 1., 3.], 1., 1., 0)):
        prediction = directory / name / 'predictions/ema/dev-cycle-0004.csv'
        table = save_csv(prediction, [{'name': row['name'], 'label': row['label'], 'logit': score}
                                     for row, score in zip(rows[:4], scores)])
        precision = save_json(Path(str(prediction) + '.precision.json'),
                              {'variant': 'ema', 'view': 'identity', 'actual_tensor_sha256': 'a' * 64,
                               'state_identity': {'successful_updates': 3200}})
        selected = {'prediction_path': table['path'], 'prediction_sha256': table['sha256'],
                    'precision_path': precision['path'], 'precision_sha256': precision['sha256'],
                    'variant': 'ema', 'cycle': 4, 'successful_updates': 3200,
                    'tensor_sha256': 'a' * 64, 'auroc': auc, 'ap': ap, 'error_twice': error}
        freeze = save_json(directory / name / 'SINGLE-MODEL-FREEZE.json',
                           {'cpu_fixture_only': False, 'precision': 'fp32', 'view': 'identity',
                            'experiment91': {'manifest_sha256': manifest['sha256']},
                            'selected': selected})
        completion = save_json(directory / name / 'TRAIN_RESULT.json',
                               {'status': 'completed', 'successful_updates': 6400, 'skipped_updates': 0})
        model = {'id': name, 'freeze': freeze, 'completion': completion}
        if with_fit:
            fit = directory / name / 'predictions/ema/fit-selected.csv'
            fit_table = save_csv(fit, [{'name': row['name'], 'label': row['label'], 'logit': i}
                                     for i, row in enumerate(rows[4:])])
            fit_precision = save_json(Path(str(fit) + '.precision.json'),
                                      {'variant': 'ema', 'view': 'identity',
                                       'actual_tensor_sha256': 'a' * 64,
                                       'state_identity': {'successful_updates': 3200}})
            model['fit_prediction'] = {**fit_table, 'precision': fit_precision,
                                       'manifest_indices': [4, 5]}
        result['models'].append(model)
    return result


class PairedEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), 'paired diagnostic implementation is missing')
        spec = importlib.util.spec_from_file_location('paired_evidence_test_subject', SOURCE)
        self.diagnostic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.diagnostic)

    def test_weighted_auc_matches_independent_repeated_sample_oracle_with_ties(self):
        labels = [1, 0, 1, 0, 0]
        scores = [1., 0., 0., 1., 2.]
        weights = np.array([[1, 1, 1, 1, 1], [2, 0, 1, 3, 1], [0, 3, 4, 2, 0]])
        expected = []
        for counts in weights:
            positives = [s for y, s, c in zip(labels, scores, counts) if y for _ in range(c)]
            negatives = [s for y, s, c in zip(labels, scores, counts) if not y for _ in range(c)]
            expected.append(sum(float(p > n) + .5 * float(p == n)
                                for p in positives for n in negatives) / (len(positives) * len(negatives)))
        np.testing.assert_allclose(self.diagnostic.weighted_auc(labels, scores, weights),
                                   expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.diagnostic.weighted_auc(labels, scores, [[0, 1, 0, 1, 1]])

    def test_bootstrap_is_paired_reproducible_and_has_explicit_group_unit(self):
        labels, scores = [0, 1, 0, 1], [0., 2., 1., 3.]
        same = self.diagnostic.paired_bootstrap(labels, scores, scores, replicates=128, seed=73)
        self.assertEqual(same['interval'], [0., 0.])
        self.assertEqual(same['delta_auroc'], 0.)
        self.assertEqual(same['unit'], 'stratified_image')
        self.assertFalse(same['selection_bias_corrected'])
        kwargs = {'replicates': 128, 'seed': 79, 'groups': ['one', 'one', 'two', 'two']}
        a = self.diagnostic.paired_bootstrap(labels, [-s for s in scores], scores, **kwargs)
        b = self.diagnostic.paired_bootstrap(labels, [-s for s in scores], scores, **kwargs)
        self.assertEqual(a, b)
        self.assertEqual(a['interval'], [1., 1.])
        self.assertEqual(a['unit'], 'supplied_group')
        self.assertEqual(a['group_count'], 2)
        with self.assertRaises(ValueError):
            self.diagnostic.paired_bootstrap(labels, scores, scores, groups=['one'] * 4)

    def test_selected_manifest_precision_and_csv_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            inventory = fixture(Path(temporary))
            report = self.diagnostic.analyze(inventory)
            self.assertEqual(report['models']['baseline']['auroc'], .75)
            self.assertEqual(report['models']['candidate']['auroc'], 1.)
            self.assertEqual(report['comparisons'][0]['candidate'], 'candidate')
            self.assertEqual(report['comparisons'][0]['delta_auroc'], .25)
            changed = json.loads(json.dumps(inventory))
            changed['models'][0]['freeze']['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                self.diagnostic.analyze(changed)
            source = Path(inventory['models'][0]['freeze']['path'])
            freeze = json.loads(source.read_text())
            freeze['selected']['prediction_path'] = str(source.parent / 'official.csv')
            changed = json.loads(json.dumps(inventory))
            changed['models'][0]['freeze'] = save_json(source, freeze)
            with self.assertRaises(ValueError):
                self.diagnostic.analyze(changed)

    def test_failed_or_running_train_cannot_be_ranked_as_a_complete_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for status in ('failed', 'running'):
                inventory = fixture(directory / status)
                model = inventory['models'][0]
                path = Path(model['completion']['path'])
                model['completion'] = save_json(path, {'status': status, 'successful_updates': 6400,
                                                       'skipped_updates': 0})
                with self.subTest(status=status), self.assertRaises(ValueError):
                    self.diagnostic.analyze(inventory)

    def test_completed_receipt_accepts_existing_evidence_layout_and_rejects_other_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            inventory = fixture(Path(temporary))
            model = inventory['models'][0]
            original = Path(model['completion']['path'])
            receipt = json.loads(original.read_text())
            model['completion'] = save_json(original.parent / 'evidence/TRAIN_RESULT.json', receipt)
            report = self.diagnostic.analyze(inventory)
            self.assertEqual(report['models']['baseline']['completion']['status'], 'completed')
            model['completion'] = save_json(original.parent.parent / 'other/evidence/TRAIN_RESULT.json', receipt)
            with self.assertRaises(ValueError):
                self.diagnostic.analyze(inventory)

    def test_manifest_uses_existing_casefold_filename_contract_without_allowing_other_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            inventory = fixture(Path(temporary))
            path = Path(inventory['manifest']['path'])
            manifest = json.loads(path.read_text())
            for row in manifest['rows']:
                row['relative_path'] = 'train/' + row['name'].upper()
            inventory['manifest'] = save_json(path, manifest)
            for model in inventory['models']:
                freeze_path = Path(model['freeze']['path'])
                freeze = json.loads(freeze_path.read_text())
                freeze['experiment91']['manifest_sha256'] = inventory['manifest']['sha256']
                model['freeze'] = save_json(freeze_path, freeze)
            self.assertEqual(self.diagnostic.analyze(inventory)['count'], 4)
            manifest['rows'][0]['relative_path'] = 'valid/' + manifest['rows'][0]['name']
            inventory['manifest'] = save_json(path, manifest)
            with self.assertRaises(ValueError):
                self.diagnostic.analyze(inventory)

    def test_fit_and_structure_missing_are_unknown_not_zero_or_causal_diagnoses(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.diagnostic.analyze(fixture(Path(temporary)))
            fit = report['evidence']['fit_dev']['baseline']
            self.assertEqual(fit['status'], 'unknown')
            self.assertIsNone(fit['gap'])
            self.assertEqual(report['evidence']['structure']['status'], 'unknown')
            self.assertIsNone(report['evidence']['structure']['reviewed_count'])
            self.assertEqual(report['causal_diagnosis'], 'not_identified')
            self.assertFalse(report['official_read'])
            self.assertFalse(report['new_forward'])

    def test_fit_requires_same_selected_tensor_and_fit_only_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            inventory = fixture(Path(temporary), with_fit=True)
            report = self.diagnostic.analyze(inventory)
            fit = report['evidence']['fit_dev']['baseline']
            self.assertEqual(fit['status'], 'available')
            self.assertTrue(fit['complete_fit'])
            self.assertEqual(fit['gap'], .25)
            self.assertEqual(report['causal_diagnosis'], 'not_identified')
            wrong = json.loads(json.dumps(inventory))
            wrong['models'][0]['fit_prediction']['manifest_indices'] = [0, 1]
            with self.assertRaises(ValueError):
                self.diagnostic.analyze(wrong)
            model = inventory['models'][0]
            precision = Path(model['fit_prediction']['precision']['path'])
            value = json.loads(precision.read_text()); value['actual_tensor_sha256'] = 'b' * 64
            model['fit_prediction']['precision'] = save_json(precision, value)
            with self.assertRaises(ValueError):
                self.diagnostic.analyze(inventory)

    def test_explicit_structure_review_counts_observations_without_relabelling(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); inventory = fixture(directory)
            fields = ('name', 'original_sha256', 'prepared_sha256', 'original_disc_visible',
                      'prepared_disc_visible', 'label_review', 'reviewer')
            inventory['structure_review'] = save_csv(directory / 'review.csv', [
                dict(zip(fields, ['image-1.jpg', '2' * 64, 'a' * 64, 'yes', 'no', 'uncertain', 'fixture']))], fields)
            report = self.diagnostic.analyze(inventory)
            review = report['evidence']['structure']
            self.assertEqual(review['reviewed_count'], 1)
            self.assertEqual(review['potential_visibility_loss'], 1)
            self.assertEqual(review['label_disagreement_count'], 0)
            self.assertFalse(review['images_independently_checked'])
            self.assertEqual(report['models']['baseline']['auroc'], .75)


if __name__ == '__main__':
    unittest.main()

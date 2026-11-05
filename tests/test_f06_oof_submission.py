"""Synthetic checks for the F06 three-fold OOF decision and submission writer."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/diagnostics/f06_oof_submission.py'
spec = importlib.util.spec_from_file_location('f06_oof_submission', SOURCE)
f06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f06)


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def _write_csv(path, header, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.reader(stream))


def fixture(root, *, count=60, test_count=12, signal=6.0, seed=20261106):
    generator = np.random.default_rng(seed)
    names = ['official_' + str(index).zfill(5) + '.jpg' for index in range(count)]
    labels = (generator.random(count) < 0.35).astype(int)
    labels[:2] = [1, 0]
    fold = np.empty(count, dtype=int)
    for value in (0, 1):
        index = generator.permutation(np.flatnonzero(labels == value))
        fold[index] = np.arange(index.size) % 3
    for key in range(3):
        mask = fold == key
        if set(labels[mask].tolist()) != {0, 1}:
            raise AssertionError('fixture fold lacks both classes')
    test_names = ['test_' + str(index).zfill(5) + '.jpg' for index in range(test_count)]

    members = []
    for key in range(3):
        mask = fold == key
        fold_names = [name for name, value in zip(names, fold) if value == key]
        logits = (signal * labels[mask] - signal / 2.0
                  + 0.5 * generator.normal(size=int(mask.sum())))
        oof_path = root / ('oof-' + str(key) + '.csv')
        _write_csv(oof_path, ['name', 'label', 'logit', 'probability'],
                   [[name, int(label), repr(float(logit)), repr(float(_sigmoid(logit)))]
                    for name, label, logit in zip(fold_names, labels[mask], logits)])
        test_logits = generator.normal(size=test_count) + 0.2 * key
        test_path = root / ('test-' + str(key) + '.csv')
        _write_csv(test_path, ['Image_name', 'logit'],
                   [[name, repr(float(logit))] for name, logit in zip(test_names, test_logits)])
        members.append({'fold': key, 'score_column': 'logit',
                        'oof': {'path': str(oof_path), 'sha256': f06.sha(oof_path)},
                        'test': {'path': str(test_path), 'sha256': f06.sha(test_path)}})

    reference = root / 'reference.csv'
    _write_csv(reference, ['Image_name', 'Probability', 'Pred_Label'],
               [[name, repr(float(generator.random())), 0] for name in test_names])
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'format': 'f06-oof-members-v1', 'folds': members}))
    folds = root / 'FOLDS.json'
    folds.write_text(json.dumps([{'name': name, 'label': int(label), 'fold': int(value)}
                                 for name, label, value in zip(names, labels, fold)]))
    return {'folds': folds, 'manifest': manifest, 'reference': reference,
            'names': names, 'labels': labels,
            'fold': [str(int(value)) for value in fold], 'test_names': test_names,
            'members': members}


def _latency_report(root, passed, *, name='latency.json'):
    subsets = {}
    for size, members in (('1', ['0']), ('2', ['0', '1']), ('3', ['0', '1', '2'])):
        subsets[size] = {
            'members': members, 'pass': passed, 'conservative_total_ms': 60.0,
            'by_flops': {'gpu_ms_per_image': 20.0, 'total_ms_per_image': 60.0,
                         'with_safety_ms': 90.0, 'within_budget': passed},
            'by_dcu_scale': {'gpu_ms_per_image': 30.0, 'total_ms_per_image': 70.0,
                             'with_safety_ms': 105.0, 'within_budget': passed}}
    path = Path(root) / name
    path.write_text(json.dumps({'format': 'f06-latency-bench-v1', 'pass': passed,
                                'subset_estimates': subsets,
                                'estimates': {'assumptions': {'t4_effective_tflops': 12.0,
                                                              'safety_factor': 1.5,
                                                              'budget_ms': 100.0}}}))
    return path


class F06OofSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='f06-oof-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture(self.root)
        self.folds = f06.load_folds(self.data['folds'], allow_fixture=True)
        self.manifest = f06.load_manifest(self.data['manifest'], self.folds)

    def predictions(self):
        manifest = f06.load_manifest(self.data['manifest'], self.folds)
        fold_array = np.asarray(self.data['fold'], dtype=object)
        result = {}
        for key in self.folds['fold_ids']:
            mask = fold_array == key
            names = [name for name, value in zip(self.data['names'], self.data['fold']) if value == key]
            result[key] = f06.load_predictions(manifest['entries'][key]['oof'], names,
                                               labels=self.folds['labels'][mask])
        return result

    def update_sha(self, member_index, stream):
        member = self.data['members'][member_index]
        member[stream]['sha256'] = f06.sha(member[stream]['path'])
        self.data['manifest'].write_text(json.dumps({'format': 'f06-oof-members-v1',
                                                     'folds': self.data['members']}))

    def test_folds_accept_array_object_jsonl_and_reject_bad_rows(self):
        self.assertEqual(self.folds['count'], 60)
        self.assertEqual(self.folds['fold_ids'], ['0', '1', '2'])
        rows = [{'name': name, 'label': int(label), 'fold': int(value)}
                for name, label, value in zip(self.data['names'], self.data['labels'], self.data['fold'])]
        wrapped = self.root / 'wrapped.json'
        wrapped.write_text(json.dumps({'rows': rows, 'counts': {'0': 20, '1': 20, '2': 20}}))
        self.assertEqual(f06.load_folds(wrapped, allow_fixture=True)['count'], 60)
        lines = self.root / 'folds.jsonl'
        lines.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
        self.assertEqual(f06.load_folds(lines, allow_fixture=True)['count'], 60)

        duplicate = rows + [dict(rows[0])]
        path = self.root / 'duplicate.json'
        path.write_text(json.dumps(duplicate))
        with self.assertRaises(ValueError):
            f06.load_folds(path, allow_fixture=True)
        bad_label = [dict(row) for row in rows]
        bad_label[0]['label'] = 2
        path.write_text(json.dumps(bad_label))
        with self.assertRaises(ValueError):
            f06.load_folds(path, allow_fixture=True)
        two_folds = [dict(row, fold=int(row['fold']) % 2) for row in rows]
        path.write_text(json.dumps(two_folds))
        with self.assertRaises(ValueError):
            f06.load_folds(path, allow_fixture=True)
        path.write_text(json.dumps(rows))
        with self.assertRaises(ValueError):
            f06.load_folds(path, allow_fixture=False)

    def test_oof_is_z_scored_per_fold_and_concatenated_in_row_order(self):
        predictions = self.predictions()
        oof = f06.build_oof(self.folds, predictions)
        self.assertEqual(oof['z'].size, 60)
        fold_array = np.asarray(self.data['fold'], dtype=object)
        covered = 0
        for key in self.folds['fold_ids']:
            mask = fold_array == key
            covered += int(mask.sum())
            self.assertAlmostEqual(float(oof['z'][mask].mean()), 0.0, places=12)
            self.assertAlmostEqual(float(oof['z'][mask].std(ddof=0)), 1.0, places=12)
            self.assertGreater(oof['per_fold'][key]['metrics']['auroc'], 0.85)
            for field in ('auroc', 'ap', 'count', 'positive'):
                self.assertIn(field, oof['per_fold'][key]['metrics'])
        self.assertEqual(covered, 60)
        self.assertEqual(set(oof['moments']), {'0', '1', '2'})
        self.assertGreater(oof['metrics']['auroc'], 0.9)
        self.assertEqual(oof['metrics']['positive'], int(self.data['labels'].sum()))

    def test_fold_file_coverage_sha_labels_and_finiteness_are_enforced(self):
        member = self.data['members'][0]
        path = Path(member['oof']['path'])
        original = path.read_bytes()

        def tamper(transform):
            path.write_bytes(original)
            rows = _read_rows(path)
            _write_csv(path, rows[0], transform(rows[1:]))
            self.update_sha(0, 'oof')
            with self.assertRaises(ValueError):
                self.predictions()

        tamper(lambda rows: rows[:-1])
        tamper(lambda rows: rows + [['foreign.jpg', 0, '0.5', '0.62']])
        tamper(lambda rows: rows + [rows[0]])
        tamper(lambda rows: [[rows[0][0], str(1 - int(rows[0][1]))] + rows[0][2:]] + rows[1:])
        tamper(lambda rows: [rows[0][:2] + ['inf', rows[0][3]]] + rows[1:])
        tamper(lambda rows: [rows[0][:2] + ['nan', rows[0][3]]] + rows[1:])

        path.write_bytes(original)
        member['oof']['sha256'] = '0' * 64
        self.data['manifest'].write_text(json.dumps({'format': 'f06-oof-members-v1',
                                                     'folds': self.data['members']}))
        with self.assertRaises(ValueError):
            self.predictions()

        member['oof']['sha256'] = f06.sha(member['oof']['path'])
        self.data['manifest'].write_text(json.dumps({'format': 'f06-oof-members-v1',
                                                     'folds': self.data['members']}))
        self.assertEqual(sorted(self.predictions()), ['0', '1', '2'])

    def test_manifest_requires_three_matching_folds(self):
        broken = json.loads(self.data['manifest'].read_text())
        broken['format'] = 'other'
        path = self.root / 'broken.json'
        path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):
            f06.load_manifest(path, self.folds)
        broken = json.loads(self.data['manifest'].read_text())
        broken['folds'][0]['fold'] = 7
        path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):
            f06.load_manifest(path, self.folds)
        broken = json.loads(self.data['manifest'].read_text())
        broken['folds'] = broken['folds'][:2]
        path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):
            f06.load_manifest(path, self.folds)

    def test_decision_is_strictly_above_the_f04_official_auroc(self):
        threshold = f06.F04_OFFICIAL_AUROC
        self.assertEqual(f06.decide(threshold), 'F04')
        self.assertEqual(f06.decide(threshold - 1e-12), 'F04')
        self.assertEqual(f06.decide(threshold + 1e-12), 'F06')

    def test_reference_order_is_read_from_the_frozen_file(self):
        reference = f06.load_reference(self.data['reference'], allow_fixture=True)
        self.assertEqual(reference['names'], self.data['test_names'])
        rows = _read_rows(self.data['reference'])
        _write_csv(self.data['reference'], rows[0], list(reversed(rows[1:])))
        reversed_order = f06.load_reference(self.data['reference'], allow_fixture=True)
        self.assertEqual(reversed_order['names'], list(reversed(self.data['test_names'])))
        _write_csv(self.data['reference'], ['Image_name', 'Score', 'Pred_Label'], rows[1:])
        with self.assertRaises(ValueError):
            f06.load_reference(self.data['reference'], allow_fixture=True)
        _write_csv(self.data['reference'], rows[0], rows[1:] + [rows[1]])
        with self.assertRaises(ValueError):
            f06.load_reference(self.data['reference'], allow_fixture=True)

    def test_test_files_must_cover_the_reference_order(self):
        member = self.data['members'][0]
        path = Path(member['test']['path'])
        rows = _read_rows(path)
        _write_csv(path, rows[0], rows[1:-1])
        self.update_sha(0, 'test')
        with self.assertRaises(ValueError):
            f06.execute(self.data['folds'], self.data['manifest'], self.root / 'out',
                        self.data['reference'], allow_fixture=True)

    def test_cli_writes_f06_submission_with_reference_order_and_crlf(self):
        output = self.root / 'out'
        command = [sys.executable, '-B', str(SOURCE), '--folds', str(self.data['folds']),
                   '--manifest', str(self.data['manifest']), '--reference', str(self.data['reference']),
                   '--output', str(output), '--allow-fixture']
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        raw = (output / 'submit.csv').read_bytes()
        self.assertTrue(raw.endswith(b'\r\n'))
        self.assertEqual(raw.count(b'\n'), raw.count(b'\r\n'))
        rows = _read_rows(output / 'submit.csv')
        self.assertEqual(rows[0], ['Image_name', 'Probability', 'Pred_Label'])
        self.assertEqual([row[0] for row in rows[1:]], self.data['test_names'])
        probabilities = [float(row[1]) for row in rows[1:]]
        labels = [int(row[2]) for row in rows[1:]]
        self.assertTrue(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities))
        self.assertEqual(labels, [int(value >= 0.5) for value in probabilities])

        metadata = json.loads((output / 'METADATA.json').read_text())
        predictions = self.predictions()
        oof = f06.build_oof(self.folds, predictions)
        self.assertEqual(metadata['decision'], 'F06')
        self.assertGreater(oof['metrics']['auroc'], f06.F04_OFFICIAL_AUROC)
        self.assertEqual(metadata['status'], 'f06_submission_written')
        self.assertEqual(metadata['oof']['count'], 60)
        self.assertEqual(metadata['oof']['metrics'], oof['metrics'])
        self.assertEqual(set(metadata['oof']['per_fold']), {'0', '1', '2'})
        self.assertTrue(metadata['official_read'])
        self.assertEqual(metadata['official_read_scope'], 'oof_decision_only')
        self.assertFalse(metadata['TTA'])
        self.assertFalse(metadata['test_label_values_read'])
        self.assertGreater(metadata['platt']['slope'], 0)
        self.assertEqual(metadata['platt']['dev_count'], 60)
        self.assertEqual(metadata['submission']['sha256'], f06.sha(output / 'submit.csv'))
        self.assertEqual(metadata['submission']['rows'], 12)
        self.assertEqual(len(metadata['models']), 3)
        fold_array = np.asarray(self.data['fold'], dtype=object)
        for model in metadata['models']:
            self.assertEqual(model['z']['count'], int((fold_array == model['fold']).sum()))
            self.assertEqual(model['oof']['sha256'], f06.sha(model['oof']['path']))
            self.assertEqual(model['test']['sha256'], f06.sha(model['test']['path']))
            self.assertAlmostEqual(model['z']['mean'],
                                   predictions[model['fold']]['values'].mean(), places=12)
        self.assertEqual(metadata['folds']['sha256'], f06.sha(self.data['folds']))
        self.assertEqual(metadata['manifest']['sha256'], f06.sha(self.data['manifest']))
        self.assertEqual(metadata['reference_order']['sha256'], f06.sha(self.data['reference']))

    def test_oof_only_writes_no_submission(self):
        output = self.root / 'oof'
        metadata, written = f06.execute(self.data['folds'], self.data['manifest'], output,
                                        oof_only=True, allow_fixture=True)
        self.assertIsNone(written)
        self.assertEqual(metadata['status'], 'oof_only')
        self.assertFalse((output / 'submit.csv').exists())
        self.assertTrue((output / 'METADATA.json').exists())
        self.assertIsNone(metadata['submission'])

    def test_weak_oof_retains_f04_without_a_submission(self):
        root = self.root / 'weak'
        root.mkdir()
        data = fixture(root, signal=0.0, seed=7)
        output = root / 'out'
        command = [sys.executable, '-B', str(SOURCE), '--folds', str(data['folds']),
                   '--manifest', str(data['manifest']), '--reference', str(data['reference']),
                   '--output', str(output), '--allow-fixture']
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        metadata = json.loads((output / 'METADATA.json').read_text())
        self.assertEqual(metadata['decision'], 'F04')
        self.assertEqual(metadata['status'], 'f04_retained_no_submission')
        self.assertFalse((output / 'submit.csv').exists())
        self.assertIsNone(metadata['platt'])

    def test_latency_failure_marks_needs_reduction_and_lists_subsets(self):
        report = _latency_report(self.root, False)
        output = self.root / 'latency-fail'
        metadata, written = f06.execute(self.data['folds'], self.data['manifest'], output,
                                        self.data['reference'], allow_fixture=True,
                                        latency_report_path=report)
        self.assertIsNone(written)
        self.assertEqual(metadata['decision'], 'needs_reduction')
        self.assertEqual(metadata['status'], 'latency_needs_reduction')
        self.assertFalse((output / 'submit.csv').exists())
        self.assertFalse(metadata['latency']['pass'])
        self.assertEqual(set(metadata['latency']['subset_estimates']), {'1', '2', '3'})
        self.assertEqual(metadata['latency']['report']['sha256'], f06.sha(report))
        self.assertFalse(metadata['decision_rule']['latency_pass'])
        for key in ('1', '2', '3'):
            self.assertIn('members', metadata['latency']['subset_estimates'][key])

    def test_latency_pass_keeps_f06_submission(self):
        output = self.root / 'latency-pass'
        metadata, written = f06.execute(self.data['folds'], self.data['manifest'], output,
                                        self.data['reference'], allow_fixture=True,
                                        latency_report_path=_latency_report(self.root, True))
        self.assertEqual(metadata['decision'], 'F06')
        self.assertEqual(metadata['status'], 'f06_submission_written')
        self.assertTrue(metadata['latency']['pass'])
        self.assertIsNotNone(written)
        self.assertTrue((output / 'submit.csv').exists())

    def test_f04_decision_is_not_overridden_by_a_failing_latency_report(self):
        root = self.root / 'weak-latency'
        root.mkdir()
        data = fixture(root, signal=0.0, seed=7)
        report = _latency_report(root, False)
        metadata, written = f06.execute(data['folds'], data['manifest'], root / 'out',
                                        data['reference'], allow_fixture=True,
                                        latency_report_path=report)
        self.assertIsNone(written)
        self.assertEqual(metadata['decision'], 'F04')
        self.assertEqual(metadata['status'], 'f04_retained_no_submission')
        self.assertFalse(metadata['latency']['pass'])

    def test_latency_report_format_and_subsets_are_validated(self):
        path = self.root / 'bad-latency.json'
        path.write_text(json.dumps({'format': 'other', 'pass': False, 'subset_estimates': {}}))
        with self.assertRaises(ValueError):
            f06.load_latency_report(path)
        path.write_text(json.dumps({'format': 'f06-latency-bench-v1', 'pass': False,
                                    'subset_estimates': {'1': {}, '2': {}, '3': {}}}))
        with self.assertRaises(ValueError):
            f06.load_latency_report(path)
        path.write_text(json.dumps({'format': 'f06-latency-bench-v1', 'pass': 'no',
                                    'subset_estimates': {}}))
        with self.assertRaises(ValueError):
            f06.load_latency_report(path)

    def test_cli_latency_report_marks_needs_reduction(self):
        report = _latency_report(self.root, False)
        output = self.root / 'cli-latency'
        command = [sys.executable, '-B', str(SOURCE), '--folds', str(self.data['folds']),
                   '--manifest', str(self.data['manifest']), '--reference', str(self.data['reference']),
                   '--latency-report', str(report), '--output', str(output), '--allow-fixture']
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        metadata = json.loads((output / 'METADATA.json').read_text())
        self.assertEqual(metadata['decision'], 'needs_reduction')
        self.assertFalse((output / 'submit.csv').exists())
        self.assertEqual(set(metadata['latency']['subset_estimates']), {'1', '2', '3'})

    def test_cli_rejects_a_stale_output_directory(self):
        output = self.root / 'used'
        output.mkdir()
        command = [sys.executable, '-B', str(SOURCE), '--folds', str(self.data['folds']),
                   '--manifest', str(self.data['manifest']), '--output', str(output), '--oof-only',
                   '--allow-fixture']
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)


if __name__ == '__main__':
    unittest.main()

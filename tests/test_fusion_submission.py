"""Synthetic checks for the fused submission writer."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/diagnostics/fusion_submission.py'
spec = importlib.util.spec_from_file_location('fusion_submission', SOURCE)
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)


def write_csv(path, names, values, labels=None):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['name', 'label', 'probability'] if labels is not None else ['Image_name', 'Probability'])
        for index, (name, value) in enumerate(zip(names, values)):
            row = [name, labels[index], repr(float(value))] if labels is not None else [name, repr(float(value))]
            writer.writerow(row)


def fixture(root, count=40, test_count=12):
    generator = np.random.default_rng(20261105)
    labels = np.array([1 if index % 3 == 0 else 0 for index in range(count)])
    dev_names = ['train_' + str(index).zfill(5) + '.jpg' for index in range(count)]
    test_names = ['test_' + str(index).zfill(5) + '.jpg' for index in range(test_count)]
    members = []
    for number in range(3):
        dev = np.clip(0.5 * labels + 0.4 * generator.random(count) + 0.05 * number, 0.001, 0.999)
        test = np.clip(0.45 * generator.random(test_count) + 0.05 * number, 0.001, 0.999)
        dev_path, test_path = root / ('dev-' + str(number) + '.csv'), root / ('test-' + str(number) + '.csv')
        write_csv(dev_path, dev_names, dev, labels)
        write_csv(test_path, test_names, test)
        members.append({'id': 'm' + str(number), 'weight': 1.0,
                        'dev': {'path': str(dev_path), 'sha256': fusion.sha(dev_path)},
                        'test': {'path': str(test_path), 'sha256': fusion.sha(test_path)},
                        'dev_mean': float(dev.mean()), 'dev_std': float(dev.std(ddof=0))})
    manifest = {'format': 'f05-fusion-members-v1', 'official_read': False, 'members': members}
    path = root / 'members.json'
    path.write_text(json.dumps(manifest))
    return path, test_names, dev_names, labels


class FusionSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fusion-submission-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest, self.test_names, self.dev_names, self.labels = fixture(self.root)

    def load(self, *, allow_fixture=True):
        manifest = json.loads(self.manifest.read_text())
        return fusion.read_members(manifest, allow_fixture=allow_fixture)

    def test_members_bind_files_and_orders(self):
        members, names, labels = self.load()
        self.assertEqual(names, self.test_names)
        self.assertEqual(len(members), 3)
        self.assertTrue(np.array_equal(labels, self.labels))

    def test_wrong_sha_or_order_or_moment_is_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        manifest['members'][0]['test']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            fusion.read_members(manifest, allow_fixture=True)
        manifest = json.loads(self.manifest.read_text())
        manifest['members'][1]['dev_mean'] = 1.5
        with self.assertRaises(ValueError):
            fusion.read_members(manifest, allow_fixture=False)
        manifest = json.loads(self.manifest.read_text())
        other = self.root / 'other.csv'
        write_csv(other, list(reversed(self.test_names)), [0.5] * len(self.test_names))
        manifest['members'][2]['test'] = {'path': str(other), 'sha256': fusion.sha(other)}
        with self.assertRaises(ValueError):
            fusion.read_members(manifest, allow_fixture=True)

    def test_standardisation_and_weighted_fusion(self):
        members, names, labels = self.load()
        vectors = {member['id']: fusion.standardise(member['dev']['values'], member['dev_mean'], member['dev_std'])
                   for member in members}
        for values in vectors.values():
            self.assertAlmostEqual(float(values.mean()), 0.0, places=12)
            self.assertAlmostEqual(float(values.std(ddof=0)), 1.0, places=12)
        equal = fusion.fuse(vectors, [1.0, 1.0, 1.0])
        heavy = fusion.fuse(vectors, [1.0, 9.0, 1.0])
        self.assertFalse(np.allclose(equal, heavy))

    def test_platt_is_monotone_and_bounded(self):
        members, names, labels = self.load()
        result = fusion.build(members, names)
        parameters = result['platt']
        self.assertGreater(parameters['slope'], 0)
        grid = np.linspace(-3, 3, 41)
        mapped = fusion.apply_platt(grid, parameters)
        self.assertTrue(np.all(np.diff(mapped) >= 0))
        self.assertGreaterEqual(int(np.count_nonzero(np.diff(mapped) > 0)), 35)
        self.assertTrue(np.all((mapped >= 0) & (mapped <= 1)))
        for key in ('auroc', 'ap', 'count', 'positive'):
            self.assertIn(key, result['dev_metrics'])
        self.assertEqual(set(result['spearman_test_vs_member']), {'m0', 'm1', 'm2'})

    def test_cli_writes_valid_submission_and_metadata(self):
        output = self.root / 'out'
        command = [sys.executable, '-B', str(SOURCE), '--members', str(self.manifest),
                   '--output', str(output), '--allow-fixture']
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with (output / 'submit.csv').open(newline='') as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ['Image_name', 'Probability', 'Pred_Label'])
        self.assertEqual([row[0] for row in rows[1:]], self.test_names)
        values = [float(row[1]) for row in rows[1:]]
        labels = [int(row[2]) for row in rows[1:]]
        self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertEqual(labels, [int(value >= 0.5) for value in values])
        metadata = json.loads((output / 'METADATA.json').read_text())
        self.assertEqual(metadata['rows'], len(self.test_names))
        self.assertFalse(metadata['official_read'])
        self.assertFalse(metadata['TTA'])
        self.assertEqual(metadata['submission']['sha256'], fusion.sha(output / 'submit.csv'))
        self.assertEqual(metadata['platt']['format'], 'platt-one-dimensional-v1')

    def test_expected_names_mismatch_is_rejected(self):
        members, names, labels = self.load()
        with self.assertRaises(ValueError):
            fusion.build(members, names, expected_names=list(reversed(names)))

    def test_submission_checks_reject_bad_values(self):
        names = ['a', 'b']
        with self.assertRaises(ValueError):
            fusion.check_submission(names, [0.1, 1.2], [0, 1])
        with self.assertRaises(ValueError):
            fusion.check_submission(names, [0.1, 0.9], [0, 0])
        with self.assertRaises(ValueError):
            fusion.check_submission(['a', 'a'], [0.1, 0.9], [0, 1])


if __name__ == '__main__':
    unittest.main()

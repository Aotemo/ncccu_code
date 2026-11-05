"""Fit-only fold coverage, group isolation and age-statistics checks."""
import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'runtime/bundle'))
source = root / 'tools/diagnostics/f05_oof.py'
spec = importlib.util.spec_from_file_location('f05_oof', source)
oof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oof)


def fixture():
    rows = [{'name': 'row-' + str(i), 'sha256': ('%064x' % (i + 1)),
             'relative_path': 'train/row-' + str(i), 'label': i % 2,
             'group': 'group-' + str(i // 2), 'age': str(30 + i % 50),
             'original_size': [1000, 1000]} for i in range(120)]
    return {'rows': rows, 'fit_indices': list(range(100)), 'dev_indices': list(range(100, 120)),
            'audit_indices': [], 'source_identity': {'fixture': True},
            'split': {'kind': 'holdout', 'seed': 20261024}}


class OOFTests(unittest.TestCase):
    def test_all_fit_rows_are_held_out_once_with_groups_and_dev_isolated(self):
        base = fixture()
        folds = oof.build_fold_manifests(base, 20261121)
        self.assertEqual(len(folds), 5)
        self.assertEqual(folds, oof.build_fold_manifests(base, 20261121))
        heldout = []
        for fold in folds:
            train, dev, audit = map(set, (fold['fit_indices'], fold['dev_indices'], fold['audit_indices']))
            self.assertEqual(dev, set(base['dev_indices']))
            self.assertFalse(train & audit or train & dev or audit & dev)
            self.assertEqual(train | audit, set(base['fit_indices']))
            train_groups = {base['rows'][i]['group'] for i in train}
            self.assertFalse(train_groups & {base['rows'][i]['group'] for i in audit})
            self.assertEqual({base['rows'][i]['label'] for i in train}, {0, 1})
            heldout.extend(audit)
        self.assertEqual(sorted(heldout), base['fit_indices'])

    def test_age_is_computed_only_on_fold_training_rows(self):
        from retrain.age import make_spec
        fold = oof.build_fold_manifests(fixture(), 20261121)[0]
        age = make_spec(fold)
        self.assertEqual(age['contract']['fit_count'], len(fold['fit_indices']))
        self.assertNotEqual(age['contract']['fit_count'], 100)

    def test_rejects_duplicate_indices_and_base_group_leakage(self):
        base = fixture(); base['fit_indices'].append(base['fit_indices'][0])
        with self.assertRaises(ValueError):
            oof.build_fold_manifests(base, 20261121)
        base = fixture(); base['rows'][100]['group'] = base['rows'][0]['group']
        with self.assertRaises(ValueError):
            oof.build_fold_manifests(base, 20261121)


if __name__ == '__main__':
    unittest.main()

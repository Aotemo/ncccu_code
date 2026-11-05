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
        self.assertEqual(len(folds), 3)
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
        positives = [sum(base['rows'][i]['label'] for i in fold['audit_indices']) for fold in folds]
        self.assertLessEqual(max(positives) - min(positives), 1)
        self.assertTrue(all(fold['split']['folds'] == 3 for fold in folds))

    def test_age_is_computed_only_on_fold_training_rows(self):
        from retrain.age import make_spec
        fold = oof.build_fold_manifests(fixture(), 20261121)[0]
        age = make_spec(fold)
        self.assertEqual(age['contract']['fit_count'], len(fold['fit_indices']))
        self.assertNotEqual(age['contract']['fit_count'], 100)

    def test_training_manifest_excludes_audit_without_changing_sealed_fold(self):
        base = fixture()
        fold = oof.build_fold_manifests(base, 20261121)[0]
        held = list(fold['audit_indices'])
        train = oof.training_manifest(fold)
        self.assertEqual(train['audit_indices'], [])
        self.assertEqual(train['fit_indices'], fold['fit_indices'])
        self.assertEqual(train['dev_indices'], base['dev_indices'])
        self.assertFalse(set(held) & (set(train['fit_indices']) | set(train['dev_indices'])))
        self.assertEqual(fold['audit_indices'], held)
        self.assertEqual(train['counts']['audit'], {'rows': 0, 'positive': 0, 'groups': 0})

    def test_training_manifest_rejects_a_fold_with_overlapping_roles(self):
        fold = oof.build_fold_manifests(fixture(), 20261121)[0]
        fold['audit_indices'].append(fold['fit_indices'][0])
        with self.assertRaises(ValueError):
            oof.training_manifest(fold)

    def test_sealed_folds_bind_roles_age_and_dev_before_derivation(self):
        from retrain.age import make_spec
        from retrain.persistence import canonical_digest
        base = fixture(); folds = oof.build_fold_manifests(base, 20261121)
        rows = []
        for number, fold in enumerate(folds):
            rows.append({'fold': number, 'round': 'R' + str(265 + number), 'counts': fold['counts'],
                         'fit_indices': fold['fit_indices'], 'audit_indices': fold['audit_indices'],
                         'fit_indices_sha256': canonical_digest(fold['fit_indices']),
                         'audit_indices_sha256': canonical_digest(fold['audit_indices']),
                         'dev_indices_sha256': canonical_digest(fold['dev_indices']),
                         'manifest_canonical_sha256': canonical_digest(fold), 'age': make_spec(fold)})
        receipt = {'status': 'three_fold_fit_only_indices_sealed_no_training', 'seed': 20261121,
                   'folds': rows, 'GPU_initialized': False, 'training_started': False, 'official_read': False}
        self.assertTrue(oof.verify_sealed_folds(folds, receipt, 20261121))
        old = dict(receipt, status='five_fold_fit_only_indices_sealed_no_training')
        with self.assertRaises(ValueError):
            oof.verify_sealed_folds(folds, old, 20261121)
        changed = dict(receipt); changed['folds'] = [dict(row) for row in rows]
        changed['folds'][0]['audit_indices'] = rows[1]['audit_indices']
        with self.assertRaises(ValueError):
            oof.verify_sealed_folds(folds, changed, 20261121)
        changed['folds'][0] = dict(rows[0]); changed['folds'][0]['age'] = rows[1]['age']
        with self.assertRaises(ValueError):
            oof.verify_sealed_folds(folds, changed, 20261121)

    def test_rejects_duplicate_indices_and_base_group_leakage(self):
        base = fixture(); base['fit_indices'].append(base['fit_indices'][0])
        with self.assertRaises(ValueError):
            oof.build_fold_manifests(base, 20261121)
        base = fixture(); base['rows'][100]['group'] = base['rows'][0]['group']
        with self.assertRaises(ValueError):
            oof.build_fold_manifests(base, 20261121)


if __name__ == '__main__':
    unittest.main()

"""Five-fold bindings cannot change dev, source rows or held-out roles."""
import copy
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'runtime/bundle'), str(root / 'tools/diagnostics'), str(root / 'tests')]
import f05_oof
from retrain import age
from retrain.f05_folds import verify_roles
from retrain.persistence import canonical_digest
from test_f05_oof import fixture


class FoldBindingTests(unittest.TestCase):
    def fixture(self):
        base = fixture()
        fold = f05_oof.build_fold_manifests(base, 20261121)[0]
        sealed = {'fold':0, 'round':'R265', 'counts':fold['counts'],
                  'fit_indices':fold['fit_indices'], 'audit_indices':fold['audit_indices'],
                  'fit_indices_sha256':canonical_digest(fold['fit_indices']),
                  'audit_indices_sha256':canonical_digest(fold['audit_indices']),
                  'dev_indices_sha256':canonical_digest(fold['dev_indices']),
                  'manifest_canonical_sha256':canonical_digest(fold), 'age':age.make_spec(fold)}
        return base, f05_oof.training_manifest(fold), sealed

    def test_sealed_train_dev_and_fit_only_age_pass(self):
        base, train, sealed = self.fixture()
        result = verify_roles(base, train, sealed, fold_number=0, split_seed=20261121)
        self.assertEqual(result['fit_count'], len(train['fit_indices']))
        self.assertEqual(result['heldout_count'], len(sealed['audit_indices']))

    def test_dev_heldout_or_source_row_changes_are_rejected(self):
        base, train, sealed = self.fixture()
        cases = []
        changed = copy.deepcopy(train); changed['dev_indices'][0] = changed['fit_indices'][0]; cases.append(changed)
        changed = copy.deepcopy(train); changed['fit_indices'][0] = sealed['audit_indices'][0]; cases.append(changed)
        changed = copy.deepcopy(train); changed['rows'][0]['sha256'] = 'f' * 64; cases.append(changed)
        changed = copy.deepcopy(train); changed['audit_indices'] = [sealed['audit_indices'][0]]; cases.append(changed)
        for changed in cases:
            with self.subTest(changed=cases.index(changed)), self.assertRaises(ValueError):
                verify_roles(base, changed, sealed, fold_number=0, split_seed=20261121)

    def test_seal_or_fit_only_age_changes_are_rejected(self):
        base, train, sealed = self.fixture()
        for key, wrong in (('fold', 1), ('round', 'R266'), ('fit_indices_sha256', '0' * 64)):
            changed = copy.deepcopy(sealed); changed[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_roles(base, train, changed, fold_number=0, split_seed=20261121)
        changed = copy.deepcopy(sealed)
        changed['age']['contract']['mean'] += 1
        with self.assertRaises(ValueError):
            verify_roles(base, train, changed, fold_number=0, split_seed=20261121)
        with self.assertRaises(ValueError):
            verify_roles(base, train, sealed, fold_number=3, split_seed=20261121)


if __name__ == '__main__':
    unittest.main()

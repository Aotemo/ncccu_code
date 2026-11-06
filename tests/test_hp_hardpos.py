"""HP hard-positive rule, hardpos targets, OHEM loss and warm-start job contracts."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
import torch  # noqa: E402

from retrain.classification_loss import (classification_loss, ohem_pairwise_values,  # noqa: E402
    f05_pairwise_objective, validate_loss, PAIR_SIGMOID_OHEM_LOSS)
from retrain.config import (HP_BINDING, HP_SEED, OHEM_SEED, HP_WARM_START_FILE_SHA256,  # noqa: E402
    HP_WARM_START_TENSOR_SHA256, validate_f05_training_binding, validate_job)
from retrain.data import (F05_BASE_MANIFEST_SHA, F05_THREE_FOLD_SHA, f05_hard_positive_rule,  # noqa: E402
    validate_f05_target_table)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # another imported module already pinned the interop pool

SPEC = {'version': 'binary-pairwise-sigmoid-v1', 'kind': 'binary_pairwise_sigmoid',
        'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}


def _folds_fixture():
    rows = [{'label': 1 if index % 2 == 0 else 0, 'group': str(index)}
            for index in range(22)]
    base = {'rows': rows, 'fit_indices': list(range(20)), 'dev_indices': [20, 21]}
    roles = [list(range(0, 7)), list(range(7, 14)), list(range(14, 20))]
    folds = {'status': 'three_fold_fit_only_indices_sealed_no_training',
             'base_manifest_sha256': F05_BASE_MANIFEST_SHA, 'seed': 20261121,
             'official_read': False, 'training_started': False,
             'folds': [{'fold': number, 'round': 'R' + str(265 + number), 'audit_indices': held,
                        'fit_indices': [x for x in base['fit_indices'] if x not in held]}
                       for number, held in enumerate(roles)]}
    sources = [{'fold': number, 'selected_model_sha256': str(number + 1) * 64,
                'calibration_sha256': str(number + 4) * 64} for number in range(3)]
    return base, folds, sources, roles


def _hardpos_table(base, folds, sources, roles, probabilities):
    ordered = sorted(base['fit_indices'])
    labels = [base['rows'][index]['label'] for index in ordered]
    hard, stats = f05_hard_positive_rule(labels, [probabilities[index] for index in ordered])
    membership = {index: number for number, held in enumerate(roles) for index in held}
    table = {'format': 'f05-fit-oof-targets-v1', 'mode': 'hardpos',
             'base_manifest_sha256': F05_BASE_MANIFEST_SHA, 'folds_sha256': F05_THREE_FOLD_SHA,
             'fold_sources': copy.deepcopy(sources),
             'rows': [{'index': index, 'label': base['rows'][index]['label'],
                       'fold': membership[index],
                       'weight': 3. if position in hard else 1.}
                      for position, index in enumerate(ordered)]}
    return table, stats


class HardPositiveRuleTest(unittest.TestCase):
    def test_smaller_set_selection_tiebreak_and_pinned_stats(self):
        # 10 positives / 10 negatives: floor(.15*10)=1 lowest vs below-max p95.
        labels = [1] * 10 + [0] * 10
        probabilities = [.01, .02, .03, .04, .05, .06, .07, .08, .09, .10] + [.50] * 9 + [.11]
        hard, stats = f05_hard_positive_rule(labels, probabilities)
        self.assertEqual(stats['lowest_count'], 1)
        self.assertEqual(stats['negative_p95'], .50)
        self.assertEqual(stats['below_p95_count'], 10)
        self.assertEqual(stats['selection'], 'lowest15')
        self.assertEqual(hard, {0})
        # Set B smaller would need >6 positives; with 10 positives set A (size 1)
        # is always at most as large as set B, so A wins when B is larger.
        probabilities = [.90, .91, .92, .93, .94, .95, .96, .97, .98, .99] + [.50] * 9 + [.93]
        hard, stats = f05_hard_positive_rule(labels, probabilities)
        self.assertEqual(stats['negative_p95'], .93)
        self.assertEqual(stats['below_p95_count'], 3)
        self.assertEqual(stats['selection'], 'lowest15')
        self.assertEqual(hard, {0})
        # Equal sizes select set A; ties break by original order.
        probabilities = [.10] * 10 + [.20] * 10
        hard, stats = f05_hard_positive_rule(labels, probabilities)
        self.assertEqual(stats['negative_p95'], .20)
        self.assertEqual(stats['below_p95_count'], 10)
        self.assertEqual(stats['selection'], 'lowest15')
        self.assertEqual(hard, {0})
        for bad in (([1, 0], [.5, float('nan')]), ([1, 2], [.5, .5]), ([1], [.5])):
            with self.assertRaises(ValueError):
                f05_hard_positive_rule(*bad)

    def test_hardpos_target_table_validation(self):
        base, folds, sources, roles = _folds_fixture()
        probabilities = {index: (.01 * ((index // 2) + 1)) if index % 2 == 0 else .50
                         for index in range(20)}
        probabilities[1] = .11
        table, stats = _hardpos_table(base, folds, sources, roles, probabilities)
        # 10 positives (probabilities .01..10) / 10 negatives: floor(.15*10)=1 lowest
        # versus all positives strictly below the negative p95 .50 -> set A smaller.
        self.assertEqual(stats['lowest_count'], 1)
        self.assertEqual(stats['negative_p95'], .50)
        self.assertEqual(stats['below_p95_count'], 10)
        self.assertEqual(stats['selection'], 'lowest15')
        self.assertEqual(stats['hard_count'], 1)
        result, returned = validate_f05_target_table(
            base, folds, table, sources, probabilities, base_sha=F05_BASE_MANIFEST_SHA,
            folds_sha=F05_THREE_FOLD_SHA, fit_count=20, dev_count=2)
        self.assertEqual(result[0], 3.)
        self.assertEqual(sum(1 for value in result.values() if value == 3.), 1)
        self.assertEqual(returned, stats)
        wrong = copy.deepcopy(table)
        wrong['rows'][0]['weight'] = 1.
        with self.assertRaises(ValueError):
            validate_f05_target_table(base, folds, wrong, sources, probabilities,
                base_sha=F05_BASE_MANIFEST_SHA, folds_sha=F05_THREE_FOLD_SHA,
                fit_count=20, dev_count=2)
        wrong = copy.deepcopy(table)
        wrong['rows'][1]['weight'] = .25
        with self.assertRaises(ValueError):
            validate_f05_target_table(base, folds, wrong, sources, probabilities,
                base_sha=F05_BASE_MANIFEST_SHA, folds_sha=F05_THREE_FOLD_SHA,
                fit_count=20, dev_count=2)


class HardposObjectiveTest(unittest.TestCase):
    def test_pair_weight_product_and_value_domain(self):
        logits = torch.tensor([-1., .5, -2., 1.], requires_grad=True)
        labels = torch.tensor([0., 1., 0., 1.])
        weights = torch.tensor([1., 1., 3., 1.])
        pair = weights[labels == 1, None] * weights[labels == 0][None, :]
        differences = logits[labels == 0][None, :] - logits[labels == 1][:, None]
        expected = 2. * (torch.sigmoid(differences) * pair).sum() / pair.sum()
        actual = f05_pairwise_objective(logits, labels, weights, SPEC, mode='hardpos')
        self.assertTrue(torch.allclose(actual, expected))
        for values in (torch.tensor([.25, 1., 3., 1.]), torch.tensor([1., 1., 4., 1.]),
                       torch.tensor([1., 1., float('nan'), 1.])):
            with self.assertRaises(ValueError):
                f05_pairwise_objective(logits, labels, values, SPEC, mode='hardpos')


class OhemLossTest(unittest.TestCase):
    def test_full_fraction_matches_original_and_gradients_only_on_hardest(self):
        torch.manual_seed(7)
        logits = torch.randn(8)
        labels = torch.tensor([0.] * 4 + [1.] * 4)
        positives, negatives = logits[labels == 1], logits[labels == 0]
        differences = negatives[None, :] - positives[:, None]
        full = ohem_pairwise_values(differences, 1.0, 1.0)
        self.assertTrue(torch.equal(full, torch.sort(
            2. * torch.sigmoid(differences).reshape(-1), descending=True, stable=True).values))
        self.assertTrue(torch.allclose(full.mean(), classification_loss(logits, labels, SPEC)))
        # 2x2 pairs, floor(.25*4)=1: only the hardest pair keeps a gradient.
        logits = torch.tensor([-1., -2., 3., -4.], requires_grad=True)
        labels = torch.tensor([0., 0., 1., 1.])
        # positives 3, -4; negatives -1, -2 -> pair values:
        # (-1,3)=2*sigmoid(-4)=.036, (-1,-4)=2*sigmoid(3)=1.905,
        # (-2,3)=2*sigmoid(-5)=.013, (-2,-4)=2*sigmoid(2)=1.762 -> hardest is (-1,-4).
        ohem = classification_loss(logits, labels, PAIR_SIGMOID_OHEM_LOSS)
        ohem.backward()
        gradient = logits.grad.tolist()
        self.assertEqual(math.floor(4 * PAIR_SIGMOID_OHEM_LOSS['hard_fraction']), 1)
        self.assertAlmostEqual(gradient[1], 0., places=9)
        self.assertAlmostEqual(gradient[2], 0., places=9)
        self.assertNotEqual(gradient[0], 0.)
        self.assertNotEqual(gradient[3], 0.)
        self.assertTrue(validate_loss(PAIR_SIGMOID_OHEM_LOSS) is PAIR_SIGMOID_OHEM_LOSS)
        tampered = dict(PAIR_SIGMOID_OHEM_LOSS, hard_fraction=.5)
        with self.assertRaises(ValueError):
            validate_loss(tampered)


def _hp_job(mode):
    age = json.loads((ROOT / 'configs/HP-AGE-CONTRACT.json').read_text())
    job = {
        'name': 'convnext-small', 'model': 'convnext_small',
        'seed': HP_SEED if mode == 'hp' else OHEM_SEED,
        'training_scale': [.85, 1.0],
        'loss': dict(SPEC) if mode == 'hp' else dict(PAIR_SIGMOID_OHEM_LOSS),
        'image_size': 512, 'prepared_size': 512, 'physical_batch': 32, 'effective_batch': 64,
        'positive_per_batch': 8, 'precision': 'fp32', 'cycles': 15, 'warmup_cycles': 1,
        'min_cycles': 1, 'patience': 4, 'min_delta': 0.0, 'stop_after_cycles': 0,
        'backbone_lr': 1e-5, 'head_lr': 1e-4, 'weight_decay': 1e-4, 'dropout': .2,
        'stochastic_depth': .1, 'normalization': 'imagenet', 'final_norm_eps': 1e-6,
        'head_lr_scope': 'last_linear', 'mixup': None, 'attention_pool': None,
        'initial_checkpoint': None, 'frozen_bn': False, 'benchmark_warmup': 50,
        'benchmark_updates': 100, 'checkpoint_seconds': 300, 'checkpoint_updates': 0,
        'log_every': 25, 'prefetch_factor': 2, 'workers': 0, 'eval_workers': 0, 'split': {},
        'age': age,
        'initialization': {'kind': 'checkpoint_warm_start', 'format': 'exported-fp32-state-dict-v1',
                           'path': '/tmp/warmstart.pt', 'sha256': HP_WARM_START_FILE_SHA256,
                           'tensor_sha256': HP_WARM_START_TENSOR_SHA256, 'source_model': '/tmp/src.pt'},
        'hp': dict(HP_BINDING),
    }
    if mode == 'hp':
        job['f05_targets'] = {'version': 'f05-oof-training-v1', 'mode': 'hardpos',
                              'summary': {'path': '/tmp/HP-SUMMARY.json', 'sha256': 'a' * 64},
                              'verification': {'path': '/tmp/F05-OOF-VERIFICATION.json', 'sha256': 'b' * 64},
                              'target': {'path': '/tmp/HARDPOS-TARGETS-private.json', 'sha256': 'c' * 64}}
    return job


class WarmStartJobContractTest(unittest.TestCase):
    def test_hp_and_ohem_jobs_accepted_and_rejections_pinned(self):
        validate_job(copy.deepcopy(_hp_job('hp')))
        validate_job(copy.deepcopy(_hp_job('ohem')))
        wrong = _hp_job('hp')
        wrong['seed'] = OHEM_SEED
        with self.assertRaises(ValueError):
            validate_job(wrong)
        both = _hp_job('ohem')
        both['f05_targets'] = _hp_job('hp')['f05_targets']
        with self.assertRaises(ValueError):
            validate_job(both)
        neither = _hp_job('hp')
        neither['f05_targets'] = None
        with self.assertRaises(ValueError):
            validate_job(neither)
        bad_init = _hp_job('hp')
        bad_init['initialization']['tensor_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            validate_job(bad_init)
        stray = _hp_job('hp')
        stray['f06'] = {'version': 'f06-official-ft-v1', 'fold': 0, 'folds_sha256': 'd' * 64}
        with self.assertRaises(ValueError):
            validate_job(stray)

    def test_binding_accepts_hardpos_only_with_exact_schema(self):
        binding = _hp_job('hp')['f05_targets']
        self.assertEqual(validate_f05_training_binding(binding), binding)
        self.assertEqual(validate_f05_training_binding(dict(binding, mode='clean')), dict(binding, mode='clean'))
        with self.assertRaises(ValueError):
            validate_f05_training_binding(dict(binding, version='f05-oof-training-v2'))


if __name__ == '__main__':
    paths = ('runtime/bundle/retrain/classification_loss.py', 'runtime/bundle/retrain/config.py',
             'runtime/bundle/retrain/data.py', 'runtime/bundle/retrain/distributed_engine.py',
             'runtime/bundle/retrain/fresh.py', 'tests/test_hp_hardpos.py')
    print(json.dumps({'source_sha256': {name: __import__('hashlib').sha256((ROOT / name).read_bytes()).hexdigest()
                                        for name in paths}}), flush=True)
    unittest.main()

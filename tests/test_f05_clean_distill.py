"""F05 target-rule and real two-rank CPU WORLD gradient checks, using synthetic data."""
import json
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

# A two-rank CPU test must subdivide the one guarded executor allocation.
# Set these before NumPy/Torch imports and before multiprocessing children import
# this module, so each process cannot start a full allocation-sized BLAS pool.
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from retrain.classification_loss import (classification_loss, f05_confident_joint,
    f05_pairwise_objective, gather_f05_auxiliary)
from retrain.data import F05_BASE_MANIFEST_SHA, F05_THREE_FOLD_SHA, validate_f05_target_table
import retrain.distributed_engine as distributed_engine

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SPEC = {'version': 'binary-pairwise-sigmoid-v1', 'kind': 'binary_pairwise_sigmoid',
        'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}
INPUTS = [[(index - 31.5) / 16 for index in range(rank * 32, (rank + 1) * 32)]
          for rank in range(2)]
# G64/P8, with no local positives on rank 0, verifies true WORLD alignment.
LABELS = [[0.] * 32, [0.] * 24 + [1.] * 8]
AUXILIARY = {
    'clean': [[.25 if index % 7 == 0 else 1. for index in range(rank * 32, (rank + 1) * 32)]
              for rank in range(2)],
    'distill': [[(index + 1) / 65 for index in range(rank * 32, (rank + 1) * 32)]
                for rank in range(2)],
}


def _world_worker(rank, rendezvous, result_dir):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        result = {}
        for mode in ('clean', 'distill'):
            layer = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                layer.weight.fill_(.5)
            model = DistributedDataParallel(layer)
            images = torch.tensor(INPUTS[rank], dtype=torch.float32).reshape(-1, 1)
            labels = torch.tensor(LABELS[rank], dtype=torch.float32)
            auxiliary = torch.tensor(AUXILIARY[mode][rank], dtype=torch.float32)
            logits = model(images).reshape(-1)
            global_labels = gather_f05_auxiliary(labels)
            global_auxiliary = gather_f05_auxiliary(auxiliary)
            table = dict(enumerate(sum(AUXILIARY[mode], [])))
            job = {'loss': SPEC, 'f05_targets': {'mode': mode}}
            loss, routed_labels = distributed_engine.training_pairwise_loss(job, logits, labels,
                torch.arange(rank * 32, (rank + 1) * 32), table, return_global_labels=True)
            loss.backward()
            result[mode] = {'loss': loss.item(), 'gradient': layer.weight.grad.item(),
                            'labels': global_labels.tolist(), 'routed_labels': routed_labels.tolist(),
                            'auxiliary': global_auxiliary.tolist()}
        legacy_loss, legacy_labels = distributed_engine.training_pairwise_loss({'loss': SPEC},
            logits.detach(), labels, torch.arange(rank * 32, (rank + 1) * 32), return_global_labels=True)
        result['legacy'] = {'loss': legacy_loss.item(), 'labels': legacy_labels.tolist()}
        try:
            distributed_engine.training_pairwise_loss({'loss': SPEC, 'f05_targets': {'mode': 'clean'}},
                torch.tensor([.1, .2], requires_grad=True), torch.tensor([0., 1.]),
                torch.tensor([rank * 2, rank * 2 + 1]), {0: .25, 1: 1., 2: 1.})
        except ValueError:
            result['missing_target_rejected_on_all_ranks'] = True
        else:
            result['missing_target_rejected_on_all_ranks'] = False
        Path(result_dir, str(rank) + '.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class CleanDistillTest(unittest.TestCase):
    def test_target_table_rejects_leakage_and_wrong_sources(self):
        rows = [{'label': label, 'group': str(index)}
                for index, label in enumerate([0, 0, 1, 1, 0, 1])]
        base = {'rows': rows, 'fit_indices': [0, 1, 2, 3], 'dev_indices': [4, 5]}
        roles = [[0, 2], [1], [3]]
        folds = {'status': 'three_fold_fit_only_indices_sealed_no_training',
                 'base_manifest_sha256': F05_BASE_MANIFEST_SHA, 'seed': 20261121,
                 'official_read': False, 'training_started': False,
                 'folds': [{'fold': i, 'round': 'R' + str(265 + i), 'audit_indices': held,
                            'fit_indices': [x for x in base['fit_indices'] if x not in held]}
                           for i, held in enumerate(roles)]}
        sources = [{'fold': i, 'selected_model_sha256': str(i + 1) * 64,
                    'calibration_sha256': str(i + 4) * 64} for i in range(3)]
        targets = {'format': 'f05-fit-oof-targets-v1', 'mode': 'clean',
                   'base_manifest_sha256': F05_BASE_MANIFEST_SHA, 'folds_sha256': F05_THREE_FOLD_SHA,
                   'fold_sources': copy.deepcopy(sources),
                   'rows': [{'index': i, 'label': rows[i]['label'],
                             'fold': next(f for f, held in enumerate(roles) if i in held),
                             'weight': .25 if i in (0, 3) else 1.} for i in range(4)]}
        oof = {0: .99, 1: .03, 2: .98, 3: .01}
        validate = lambda value, probs=oof, sources_arg=sources: validate_f05_target_table(
            base, folds, value, sources_arg, probs, base_sha=F05_BASE_MANIFEST_SHA,
            folds_sha=F05_THREE_FOLD_SHA, fit_count=4, dev_count=2)
        actual, _ = validate(targets)
        self.assertEqual(actual, {0: .25, 1: 1., 2: 1., 3: .25})
        altered = copy.deepcopy(targets); altered['rows'][1]['index'] = 4
        duplicate = copy.deepcopy(targets); duplicate['rows'][1]['index'] = 0
        wrong_fold = copy.deepcopy(targets); wrong_fold['rows'][0]['fold'] = 1
        wrong_weight = copy.deepcopy(targets); wrong_weight['rows'][0]['weight'] = 1.
        for invalid in (altered, duplicate, wrong_fold, wrong_weight):
            with self.assertRaises(ValueError):
                validate(invalid)
        with self.assertRaises(ValueError):
            validate(targets, {0: .99, 1: .03, 2: .98})
        with self.assertRaises(ValueError):
            validate(targets, oof, [{**sources[0], 'calibration_sha256': 'f' * 64}, *sources[1:]])
        distilled = copy.deepcopy(targets); distilled['mode'] = 'distill'
        for row in distilled['rows']:
            row.pop('weight'); row['soft_target'] = oof[row['index']]
        self.assertEqual(validate(distilled)[0], oof)
        distilled['rows'][0]['soft_target'] = float('nan')
        with self.assertRaises(ValueError):
            validate(distilled)

    def test_fixed_confident_joint_thresholds_and_candidates(self):
        result = f05_confident_joint([0, 0, 1, 1], [.99, .03, .98, .01])
        self.assertEqual(result['thresholds'], [.95, .95])
        self.assertEqual(result['confident_joint'], [[1, 1], [1, 1]])
        self.assertEqual(result['suspicious_positions'], [0, 3])
        self.assertEqual(f05_confident_joint([0, 1], [np.float64(.1), np.float64(.9)])['thresholds'], [.95, .95])
        for labels, probs in (([0, 0], [.1, .2]), ([0, 1], [.1, float('nan')]),
                              ([0, 1], [.1, 1.1])):
            with self.assertRaises(ValueError):
                f05_confident_joint(labels, probs)

    def test_weighted_pairwise_and_distillation_equations(self):
        logits = torch.tensor([-1., .5, -2., 1.], requires_grad=True)
        labels = torch.tensor([0., 1., 0., 1.])
        ones = torch.ones(4)
        self.assertTrue(torch.allclose(f05_pairwise_objective(logits, labels, ones, SPEC, mode='clean'),
                                       classification_loss(logits, labels, SPEC)))
        weights = torch.tensor([.25, 1., 1., .25])
        pair = weights[labels == 1, None] * weights[labels == 0][None, :]
        diff = logits[labels == 0][None, :] - logits[labels == 1][:, None]
        expected = 2 * (torch.sigmoid(diff) * pair).sum() / pair.sum()
        self.assertTrue(torch.allclose(f05_pairwise_objective(logits, labels, weights, SPEC, mode='clean'), expected))
        soft = torch.tensor([.1, .8, .2, .7])
        expected = classification_loss(logits, labels, SPEC) + .25 * torch.nn.functional.binary_cross_entropy_with_logits(logits, soft)
        self.assertTrue(torch.allclose(f05_pairwise_objective(logits, labels, soft, SPEC, mode='distill'), expected))
        for mode, values in (('clean', torch.tensor([1., 1., .5, 1.])),
                             ('distill', torch.tensor([0., 1., float('nan'), .3]))):
            with self.assertRaises(ValueError):
                f05_pairwise_objective(logits, labels, values, SPEC, mode=mode)

    def test_two_rank_world_order_and_ddp_gradient_match_single_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous = str(Path(temporary) / 'gloo')
            mp.spawn(_world_worker, args=(rendezvous, temporary), nprocs=2, join=True)
            rank_results = [json.loads(Path(temporary, str(rank) + '.json').read_text()) for rank in range(2)]
            self.assertTrue(all(result['missing_target_rejected_on_all_ranks'] for result in rank_results))
            all_images = torch.tensor(sum(INPUTS, []), dtype=torch.float32)
            all_labels = torch.tensor(sum(LABELS, []), dtype=torch.float32)
            legacy_expected = classification_loss(all_images * .5, all_labels, SPEC)
            for result in rank_results:
                self.assertEqual(result['legacy']['labels'], all_labels.tolist())
                self.assertAlmostEqual(result['legacy']['loss'], legacy_expected.item(), places=6)
            for mode in ('clean', 'distill'):
                layer = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    layer.weight.fill_(.5)
                images = torch.tensor(sum(INPUTS, []), dtype=torch.float32).reshape(-1, 1)
                labels = torch.tensor(sum(LABELS, []), dtype=torch.float32)
                self.assertEqual((labels.numel(), int(labels.sum())), (64, 8))
                auxiliary = torch.tensor(sum(AUXILIARY[mode], []), dtype=torch.float32)
                expected = f05_pairwise_objective(layer(images).reshape(-1), labels, auxiliary, SPEC, mode=mode)
                expected.backward()
                for result in rank_results:
                    self.assertEqual(result[mode]['labels'], labels.tolist())
                    self.assertEqual(result[mode]['routed_labels'], labels.tolist())
                    self.assertEqual(result[mode]['auxiliary'], auxiliary.tolist())
                    self.assertAlmostEqual(result[mode]['loss'], expected.item(), places=6)
                    self.assertAlmostEqual(result[mode]['gradient'], layer.weight.grad.item(), places=6)


if __name__ == '__main__':
    paths = ('runtime/bundle/retrain/classification_loss.py', 'runtime/bundle/retrain/data.py',
             'runtime/bundle/retrain/distributed_engine.py', 'tests/test_f05_clean_distill.py')
    print(json.dumps({'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                        for name in paths}}), flush=True)
    unittest.main()

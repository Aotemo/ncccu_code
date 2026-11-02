"""Fixed R156 objective: independent 8x56 arithmetic and real CPU WORLD DDP."""
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain.classification_loss import classification_loss, gather_pairwise_inputs, validate_loss

HARD = {'version': 'binary-pairwise-logistic-positive-top2-mix05-v1',
        'kind': 'binary_pairwise_logistic_hard_positive', 'temperature': 1.0,
        'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
        'top_positive_count': 2, 'hard_positive_weight': .5,
        'tie_break': 'WORLD_positive_order_stable'}
LOGISTIC = {'version': 'binary-pairwise-logistic-v1', 'kind': 'binary_pairwise_logistic',
            'temperature': 1., 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}
SIGMOID = {**LOGISTIC, 'version': 'binary-pairwise-sigmoid-v1', 'kind': 'binary_pairwise_sigmoid'}


def _fixture(positive_values):
    return [((i % 9) - 4) * .21 + .002 * i for i in range(56)] + positive_values, [0.] * 56 + [1.] * 8


def _analytic(values, labels):
    """Scalar double arithmetic; stable Python ordering, no Torch loss/sort."""
    positive = [i for i, label in enumerate(labels) if label == 1]
    negative = [i for i, label in enumerate(labels) if label == 0]
    assert len(positive) == 8 and len(negative) == 56
    means = [sum(math.log1p(math.exp(values[n] - values[p])) for n in negative) / 56
             for p in positive]
    selected = sorted(range(8), key=lambda i: (-means[i], i))[:2]
    weights = [.5 / 8 + (.5 / 2 if i in selected else 0.) for i in range(8)]
    gradient = [0.] * 64
    for index, p in enumerate(positive):
        for n in negative:
            derivative = weights[index] / 56 / (1. + math.exp(values[p] - values[n]))
            gradient[p] -= derivative
            gradient[n] += derivative
    return sum(weight * value for weight, value in zip(weights, means)), gradient, selected


def _gloo_worker(rank, rendezvous, output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        network = torch.nn.Linear(1, 1)
        with torch.no_grad():
            network.weight.fill_(1.)
            network.bias.fill_(.125)
        ddp = DistributedDataParallel(network, broadcast_buffers=False)
        cases = [_fixture([-2.1, -.9, -.3, .1, .45, .8, 1.3, 2.]),
                 ([0.] * 64, [0.] * 56 + [1.] * 8),
                 _fixture([-2., -1., -1., 0., 1., 2., 3., 4.])]
        reports = []
        for values, labels in cases:
            network.zero_grad(set_to_none=True)
            inputs = torch.tensor(values).reshape(64, 1)
            targets = torch.tensor(labels)
            interval = slice(rank * 32, (rank + 1) * 32)
            local = ddp(inputs[interval]).reshape(-1)
            global_logits, global_targets = gather_pairwise_inputs(local, targets[interval])
            expected_logits = torch.tensor(values) + .125
            torch.testing.assert_close(global_logits, expected_logits, rtol=0, atol=0)
            torch.testing.assert_close(global_targets, targets, rtol=0, atol=0)
            expected, gradient, selected = _analytic(expected_logits.tolist(), labels)
            global_logits.retain_grad()
            loss = classification_loss(global_logits, global_targets, HARD)
            loss.backward()
            # Check each WORLD logit's local loss gradient before gather SUM.
            # This distinguishes tied indices even when Linear inputs coincide.
            torch.testing.assert_close(global_logits.grad, global_logits.new_tensor(gradient),
                                       rtol=3e-6, atol=2e-7)
            wanted_weight = sum(g * v for g, v in zip(gradient, values))
            torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(network.weight.grad, network.weight.new_tensor([[wanted_weight]]),
                                       rtol=2e-5, atol=5e-7)
            if network.bias.grad is None or abs(network.bias.grad.item()) > 3e-7:
                raise AssertionError('WORLD gather SUM and DDP mean must preserve zero shared-bias gradient')
            reports.append({'loss': float(loss.detach()), 'expected_positive_indices': selected,
                            'actual_global_logit_gradient_validated': True,
                            'weight_gradient': network.weight.grad.item(),
                            'bias_gradient': network.bias.grad.item()})
        if torch.cuda.is_initialized():
            raise AssertionError('CPU fixture initialized GPU')
        Path(output).write_text(json.dumps({'rank': rank, 'cases': reports, 'pairs': 448,
                                          'local_positives': int(targets[interval].sum()),
                                          'gpu_initialized': False}))
    finally:
        dist.destroy_process_group()


class PairwiseHardPositiveLossTest(unittest.TestCase):
    def test_actual_8x56_value_gradient_shift_and_fixed_contract(self):
        import torch
        values, labels = _fixture([-2.1, -.9, -.3, .1, .45, .8, 1.3, 2.])
        expected, expected_gradient, selected = _analytic(values, labels)
        self.assertEqual(selected, [0, 1])
        logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor(labels)
        self.assertIs(validate_loss(HARD), HARD)
        loss = classification_loss(logits, targets, HARD)
        gradient, = torch.autograd.grad(loss, logits)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), expected, delta=3e-7)
        torch.testing.assert_close(gradient, gradient.new_tensor(expected_gradient), rtol=3e-6, atol=2e-7)
        bias = torch.tensor(3.25, dtype=torch.float64, requires_grad=True)
        shifted = classification_loss(logits.detach() + bias, targets, HARD)
        bias_gradient, = torch.autograd.grad(shifted, bias)
        torch.testing.assert_close(shifted, loss.detach(), rtol=2e-6, atol=3e-7)
        self.assertLess(abs(bias_gradient.item()), 2e-7)
        for key, value in [('top_positive_count', 3), ('hard_positive_weight', .75),
                           ('tie_break', 'unstable'), ('version', 'binary-pairwise-logistic-v1')]:
            with self.assertRaises(ValueError):
                validate_loss({**HARD, key: value})
        with self.assertRaises(ValueError):
            classification_loss(logits[:60], targets[:60], HARD)

    def test_all_ties_boundary_ties_and_extremes(self):
        import torch
        cases = [([0.] * 64, [0.] * 56 + [1.] * 8),
                 _fixture([-2., -1., -1., 0., 1., 2., 3., 4.])]
        for values, labels in cases:
            expected, expected_gradient, selected = _analytic(values, labels)
            self.assertEqual(selected, [0, 1])
            logits = torch.tensor(values, requires_grad=True)
            loss = classification_loss(logits, torch.tensor(labels), HARD)
            gradient, = torch.autograd.grad(loss, logits)
            self.assertAlmostEqual(loss.item(), expected, delta=3e-7)
            torch.testing.assert_close(gradient, gradient.new_tensor(expected_gradient), rtol=3e-6, atol=2e-7)
        logits = torch.tensor([10000., -10000.] * 28 + [-10000., 10000.] * 4, requires_grad=True)
        loss = classification_loss(logits, torch.tensor([0.] * 56 + [1.] * 8), HARD)
        gradient, = torch.autograd.grad(loss, logits)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_original_BCE_logistic_sigmoid_values_and_gradients_unchanged(self):
        import torch
        from torch.nn import functional as F
        values, labels = _fixture([-2.1, -.9, -.3, .1, .45, .8, 1.3, 2.])
        for specification in [None, LOGISTIC, SIGMOID]:
            logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            targets = torch.tensor(labels, dtype=torch.float64)
            actual = classification_loss(logits, targets, specification)
            if specification is None:
                expected = F.binary_cross_entropy_with_logits(logits, targets)
            else:
                difference = logits.float()[:56][None, :] - logits.float()[56:, None]
                expected = (F.softplus(difference) if specification == LOGISTIC
                            else 2. * torch.sigmoid(difference)).mean()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
            expected_gradient, = torch.autograd.grad(expected, logits)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_WORLD_gather_DDP_gradients_and_ties_agree_on_every_rank(self):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                   GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='hard-positive-gloo-', dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            processes, logs = [], []
            deadline = time.monotonic() + 80
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as endpoint:
                    endpoint.bind(('127.0.0.1', 0))
                    rendezvous = f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
                for rank in range(2):
                    log = (root / f'{rank}.log').open('wb'); logs.append(log)
                    processes.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                        '--gloo-worker', str(rank), rendezvous, str(root / f'{rank}.json')],
                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                for process in processes:
                    self.assertEqual(process.wait(timeout=max(.1, deadline-time.monotonic())), 0)
                reports = [json.loads((root / f'{rank}.json').read_text()) for rank in range(2)]
                self.assertEqual([r['local_positives'] for r in reports], [0, 8])
                self.assertEqual(reports[0]['cases'], reports[1]['cases'])
                self.assertEqual([r['expected_positive_indices'] for r in reports[0]['cases']], [[0, 1]] * 3)
                self.assertTrue(all(r['actual_global_logit_gradient_validated'] for r in reports[0]['cases']))
                for report in reports:
                    self.assertEqual(report['pairs'], 448)
                    self.assertFalse(report['gpu_initialized'])
            except Exception as error:
                details = '\n'.join(f'{path.name}: {path.read_text(errors="replace")}' for path in root.iterdir())
                raise AssertionError(f'{error}\n{details}') from error
            finally:
                for process in processes:
                    if process.poll() is None: process.kill()
                    process.wait(timeout=5)
                for log in logs: log.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--gloo-worker':
        _gloo_worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        unittest.main()

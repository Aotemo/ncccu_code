"""Focused R120 sigmoid pairwise checks, including real WORLD gather + Gloo DDP gradients."""
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
from retrain.classification_loss import (PAIRWISE_LOSS_KINDS, classification_loss,
                                         gather_pairwise_inputs, validate_loss)

SIGMOID = {'version': 'binary-pairwise-sigmoid-v1', 'kind': 'binary_pairwise_sigmoid',
           'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}
LOGISTIC = {'version': 'binary-pairwise-logistic-v1', 'kind': 'binary_pairwise_logistic',
            'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}


def _gloo_worker(rank, rendezvous, output):
    started = time.monotonic()

    def phase(stage):
        now = time.monotonic()
        print(json.dumps({'rank': rank, 'pid': os.getpid(), 'stage': stage,
                          'time': time.time(), 'monotonic': now,
                          'elapsed_seconds': now - started}), flush=True)

    phase('worker_entry')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise AssertionError('CPU fixture must not initialize GPU')
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        index = torch.arange(64, dtype=torch.float32)
        inputs = torch.stack(((index - 31.5) / 16, (index.remainder(7) - 3) / 4,
                              ((3 * index).remainder(11) - 5) / 5), dim=1)
        labels = torch.cat((torch.zeros(56), torch.ones(8)))

        def model():
            linear = torch.nn.Linear(3, 1)
            with torch.no_grad():
                linear.weight.copy_(torch.tensor([[.3, -.2, .1]]))
                linear.bias.fill_(.15)
            return linear

        network = model()
        ddp = DistributedDataParallel(network, broadcast_buffers=False)
        interval = slice(rank * 32, (rank + 1) * 32)
        local_logits = ddp(inputs[interval]).reshape(-1)
        global_logits, global_targets = gather_pairwise_inputs(local_logits, labels[interval])
        torch.testing.assert_close(global_targets, labels, rtol=0, atol=0)
        if global_logits.dtype != torch.float32 or global_logits.numel() != 64:
            raise AssertionError('gather must retain all 64 FP32 logits')
        loss = classification_loss(global_logits, global_targets, SIGMOID)
        loss.backward()

        reference = model()
        full_logits = reference(inputs).reshape(-1)
        differences = full_logits[labels == 0][None, :] - full_logits[labels == 1][:, None]
        expected = (2. * torch.sigmoid(differences)).mean()
        expected.backward()
        if differences.numel() != 448 or reference.weight.grad.abs().max().item() < 1e-3:
            raise AssertionError('fixture must expose all 448 pairs and nonzero parameter gradients')
        torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(network.weight.grad, reference.weight.grad, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(network.bias.grad, reference.bias.grad, rtol=2e-5, atol=2e-7)
        if network.bias.grad is None or abs(network.bias.grad.item()) > 2e-7:
            raise AssertionError('common bias must have a present, near-zero gradient')
        if torch.cuda.is_initialized():
            raise AssertionError('CPU fixture initialized GPU')
        Path(output).write_text(json.dumps({'passed': True, 'rank': rank, 'global_pairs': 448,
            'local_positives': int(labels[interval].sum()), 'loss': float(loss.detach()),
            'weight_gradient': network.weight.grad.tolist(), 'bias_gradient': network.bias.grad.tolist(),
            'gpu_initialized': False}))
        phase('report_written')
    finally:
        dist.destroy_process_group()
    phase('worker_return')


class PairwiseSigmoidLossTest(unittest.TestCase):
    def test_scalar_value_gradient_symmetry_and_zero_slope(self):
        import torch
        values = [-1.7, .2, 1.3, -.6, .9, -2.2]
        labels = [1, 0, 1, 0, 0, 1]
        positives = [i for i, label in enumerate(labels) if label == 1]
        negatives = [i for i, label in enumerate(labels) if label == 0]
        pairs = len(positives) * len(negatives)
        expected_value, expected_gradient = 0., [0.] * len(values)
        for p in positives:
            for n in negatives:
                difference = values[n] - values[p]
                sigmoid = 1. / (1. + math.exp(-difference))
                expected_value += 2. * sigmoid / pairs
                derivative = 2. * sigmoid * (1. - sigmoid) / pairs
                expected_gradient[p] -= derivative
                expected_gradient[n] += derivative
        logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor(labels, dtype=torch.float64)
        self.assertEqual(PAIRWISE_LOSS_KINDS[:2], ('binary_pairwise_logistic', 'binary_pairwise_sigmoid'))
        self.assertIs(validate_loss(SIGMOID), SIGMOID)
        actual = classification_loss(logits, targets, SIGMOID)
        gradient, = torch.autograd.grad(actual, logits)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertAlmostEqual(actual.item(), expected_value, delta=2e-6)
        torch.testing.assert_close(gradient, gradient.new_tensor(expected_gradient), rtol=2e-6, atol=1e-7)
        mirrored = classification_loss(-logits.detach(), targets, SIGMOID)
        self.assertAlmostEqual(float(mirrored), 2. - float(actual.detach()), delta=2e-6)
        zero = torch.tensor([0., 0.], dtype=torch.float64, requires_grad=True)
        zero_loss = classification_loss(zero, torch.tensor([1., 0.]), SIGMOID)
        zero_gradient, = torch.autograd.grad(zero_loss, zero)
        self.assertAlmostEqual(float(zero_loss), 1.0, delta=1e-7)
        torch.testing.assert_close(zero_gradient, zero.new_tensor([-.5, .5]), rtol=0, atol=1e-7)

    def test_extreme_tail_shift_and_legacy_contracts(self):
        import torch
        from torch.nn import functional as F
        extreme = torch.tensor([-10000., 10000., 0., 10000., -10000., 0.], requires_grad=True)
        targets = torch.tensor([1., 1., 1., 0., 0., 0.])
        extreme_loss = classification_loss(extreme, targets, SIGMOID)
        extreme_gradient, = torch.autograd.grad(extreme_loss, extreme)
        self.assertTrue(bool(torch.isfinite(extreme_loss)))
        self.assertTrue(bool(torch.isfinite(extreme_gradient).all()))
        tail = torch.tensor([0., 10000.], requires_grad=True)
        tail_loss = classification_loss(tail, torch.tensor([1., 0.]), SIGMOID)
        tail_gradient, = torch.autograd.grad(tail_loss, tail)
        self.assertEqual(float(tail_loss), 2.0)
        self.assertTrue(bool((tail_gradient == 0).all()))
        values = torch.tensor([-1.7, .2, 1.3, -.6, .9, -2.2], dtype=torch.float64)
        labels = torch.tensor([1., 0., 1., 0., 0., 1.])
        bias = torch.tensor(3.25, dtype=torch.float64, requires_grad=True)
        shifted = classification_loss(values + bias, labels, SIGMOID)
        bias_gradient, = torch.autograd.grad(shifted, bias)
        torch.testing.assert_close(shifted, classification_loss(values, labels, SIGMOID), rtol=2e-6, atol=1e-7)
        self.assertLess(abs(float(bias_gradient)), 1e-7)
        bce = classification_loss(values, labels, None)
        torch.testing.assert_close(bce, F.binary_cross_entropy_with_logits(values, labels), rtol=0, atol=0)
        positives, negatives = values[labels == 1], values[labels == 0]
        legacy = F.softplus(negatives.float()[None, :] - positives.float()[:, None]).mean()
        torch.testing.assert_close(classification_loss(values, labels, LOGISTIC), legacy, rtol=2e-6, atol=1e-7)
        wrong = dict(SIGMOID, version='binary-pairwise-logistic-v1')
        with self.assertRaises(ValueError):
            validate_loss(wrong)
        with self.assertRaises(ValueError):
            validate_loss(dict(SIGMOID, temperature=2.0))

    def test_two_rank_gloo_ddp_matches_complete_64_sample_parameter_gradients(self):
        child_env = dict(os.environ)
        child_env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                         CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                         GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='sigmoid-pairwise-gloo-',
                                         dir=Path(__file__).resolve().parent) as scratch:
            scratch = Path(scratch)
            processes, logs = [], []
            deadline = time.monotonic() + 75

            def diagnostics():
                details = []
                for rank in range(2):
                    process = processes[rank] if rank < len(processes) else None
                    details.append(f'rank={rank} pid={None if process is None else process.pid} '
                                   f'exit_status={"not_started" if process is None else process.poll()}')
                    for suffix in ('log', 'json'):
                        path = scratch / f'rank-{rank}.{suffix}'
                        details.append(f'--- {path.name} ---\n'
                                       + (path.read_text(errors='replace') if path.exists() else '<missing>'))
                return '\n'.join(details)

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as endpoint:
                    endpoint.bind(('127.0.0.1', 0))
                    rendezvous = f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
                for rank in range(2):
                    log = (scratch / f'rank-{rank}.log').open('wb')
                    logs.append(log)
                    processes.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                        '--gloo-worker', str(rank), rendezvous, str(scratch / f'rank-{rank}.json')],
                        env=child_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                for rank, process in enumerate(processes):
                    try:
                        code = process.wait(timeout=max(.1, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        self.fail('Gloo DDP fixture exceeded its 75-second bound')
                    self.assertEqual(code, 0, f'rank {rank} exited with status {code}')
                reports = [json.loads((scratch / f'rank-{rank}.json').read_text()) for rank in range(2)]
                self.assertEqual([report['local_positives'] for report in reports], [0, 8])
                for report in reports:
                    self.assertTrue(report['passed'])
                    self.assertEqual(report['global_pairs'], 448)
                    self.assertTrue(math.isfinite(report['loss']))
                    self.assertFalse(report['gpu_initialized'])
            except Exception as error:
                raise AssertionError(f'{type(error).__name__}: {error}\n{diagnostics()}') from error
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                for log in logs:
                    log.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--gloo-worker':
        _gloo_worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        unittest.main()

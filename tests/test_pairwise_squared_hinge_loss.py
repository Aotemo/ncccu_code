"""Independent per-pair finite-margin arithmetic and real WORLD gradients."""
from datetime import timedelta
import json
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

SQUARED_HINGE = {'version': 'binary-pairwise-squared-hinge-margin1-quarter-v1',
    'kind': 'binary_pairwise_squared_hinge', 'temperature': 1.,
    'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}
LABELS = [0.] * 56 + [1.] * 8


def _cases():
    return [[-.75 + .125 * (i % 13) for i in range(56)] + [-1., -.5, 0., .5, 1., 1.5, 2., 2.5],
            [-.5 + .125 * (i % 5) for i in range(56)] + [2. + .25 * (i % 4) for i in range(8)],
            [0.] * 56 + [1.] * 8, [0.] * 64]


def _reference(values):
    value = 0.; gradient = [0.] * 64
    for positive in range(56, 64):
        for negative in range(56):
            residual = max(0., 1. + values[negative] - values[positive])
            value += residual * residual / (4. * 448.)
            slope = residual / (2. * 448.)
            gradient[negative] += slope; gradient[positive] -= slope
    return value, gradient


def _worker(rank, rendezvous, output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        network = torch.nn.Linear(1, 1)
        with torch.no_grad(): network.weight.fill_(1.); network.bias.fill_(.125)
        ddp = DistributedDataParallel(network, broadcast_buffers=False); reports = []
        for values in _cases():
            network.zero_grad(set_to_none=True)
            inputs = torch.tensor(values).reshape(64, 1); labels = torch.tensor(LABELS)
            part = slice(rank * 32, (rank + 1) * 32)
            local = ddp(inputs[part]).reshape(-1); local.retain_grad()
            logits, targets = gather_pairwise_inputs(local, labels[part]); logits.retain_grad()
            torch.testing.assert_close(logits, inputs.reshape(-1) + .125, rtol=0, atol=0)
            torch.testing.assert_close(targets, labels, rtol=0, atol=0)
            expected, gradient = _reference(logits.detach().tolist())
            loss = classification_loss(logits, targets, SQUARED_HINGE); loss.backward()
            torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=3e-6, atol=3e-7)
            wanted = logits.new_tensor(gradient)
            torch.testing.assert_close(logits.grad, wanted, rtol=3e-6, atol=3e-7)
            torch.testing.assert_close(local.grad, 2. * wanted[part], rtol=3e-6, atol=3e-7)
            weight = sum(g * v for g, v in zip(gradient, values))
            torch.testing.assert_close(network.weight.grad, network.weight.new_tensor([[weight]]), rtol=3e-6, atol=3e-7)
            if network.bias.grad is None or abs(network.bias.grad.item()) > 2e-7:
                raise AssertionError('WORLD/DDP shared bias differs')
            reports.append({'loss': float(loss.detach()), 'weight_gradient': network.weight.grad.item(),
                'bias_gradient': network.bias.grad.item(), 'actual_global_local_gradients_checked': True})
        if torch.cuda.is_initialized(): raise AssertionError('CPU fixture initialized GPU')
        Path(output).write_text(json.dumps({'rank': rank, 'pairs': 448,
            'local_positives': int(labels[part].sum()), 'cases': reports}))
    finally:
        dist.destroy_process_group()


class PairwiseSquaredHingeTest(unittest.TestCase):
    def test_8x56_independent_pair_values_gradients_and_margin_regimes(self):
        import torch
        from retrain.classification_loss import PAIRWISE_LOSS_KINDS
        self.assertIn(SQUARED_HINGE['kind'], PAIRWISE_LOSS_KINDS)
        self.assertIs(validate_loss(SQUARED_HINGE), SQUARED_HINGE)
        for values in _cases():
            logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            loss = classification_loss(logits, torch.tensor(LABELS), SQUARED_HINGE)
            gradient, = torch.autograd.grad(loss, logits)
            expected, wanted = _reference(values)
            self.assertEqual(loss.dtype, torch.float32)
            self.assertAlmostEqual(loss.item(), expected, delta=3e-7)
            torch.testing.assert_close(gradient, gradient.new_tensor(wanted), rtol=3e-6, atol=3e-7)
            bias = torch.tensor(3.25, dtype=torch.float64, requires_grad=True)
            shifted = classification_loss(logits.detach() + bias, torch.tensor(LABELS), SQUARED_HINGE)
            common, = torch.autograd.grad(shifted, bias)
            torch.testing.assert_close(shifted, loss.detach(), rtol=0, atol=0)
            self.assertLess(abs(common.item()), 2e-7)
        # Nonconstant classes with all pairs beyond margin also have zero gradient.
        for index in (1, 2): self.assertEqual(_reference(_cases()[index]), (0., [0.] * 64))
        self.assertAlmostEqual(_reference(_cases()[3])[0], .25)
        extreme = torch.tensor([10000., -10000.] * 32, requires_grad=True)
        loss = classification_loss(extreme, torch.tensor(LABELS), SQUARED_HINGE)
        gradient, = torch.autograd.grad(loss, extreme)
        self.assertTrue(bool(torch.isfinite(loss))); self.assertTrue(bool(torch.isfinite(gradient).all()))
        for field, value in (('temperature', 2.), ('version', 'other'), ('margin', 2.), ('scope', 'rank')):
            with self.assertRaises(ValueError): validate_loss({**SQUARED_HINGE, field: value})
        for labels in ([0.] * 64, [1.] * 64, LABELS[:-1] + [2.]):
            with self.assertRaises(ValueError): classification_loss(torch.zeros(64), torch.tensor(labels), SQUARED_HINGE)

    def test_legacy_none_logistic_sigmoid_values_and_gradients(self):
        import torch
        from torch.nn import functional as F
        for kind in (None, 'logistic', 'sigmoid'):
            x = torch.tensor(_cases()[0], requires_grad=True); labels = torch.tensor(LABELS)
            spec = None if kind is None else {'version': f'binary-pairwise-{kind}-v1',
                'kind': 'binary_pairwise_' + kind, 'temperature': 1., 'reduction': 'mean',
                'dtype': 'float32', 'scope': 'global_batch'}
            d = x[:56][None, :] - x[56:][:, None]
            expected = F.binary_cross_entropy_with_logits(x, labels) if kind is None else \
                F.softplus(d).mean() if kind == 'logistic' else 2. * torch.sigmoid(d).mean()
            actual = classification_loss(x, labels, spec)
            wanted, = torch.autograd.grad(expected, x, retain_graph=True)
            gradient, = torch.autograd.grad(actual, x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(gradient, wanted, rtol=0, atol=0)

    def test_WORLD_global_local_and_DDP_parameter_gradients(self):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='sqhinge-gloo-', dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory); processes = []; logs = []; deadline = time.monotonic() + 80
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as endpoint:
                    endpoint.bind(('127.0.0.1', 0)); rendezvous = f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
                for rank in range(2):
                    log = (root / f'{rank}.log').open('wb'); logs.append(log)
                    processes.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                        '--worker', str(rank), rendezvous, str(root / f'{rank}.json')],
                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                for process in processes: self.assertEqual(process.wait(timeout=max(.1, deadline - time.monotonic())), 0)
                reports = [json.loads((root / f'{rank}.json').read_text()) for rank in range(2)]
                self.assertEqual([r['local_positives'] for r in reports], [0, 8])
                self.assertEqual(reports[0]['cases'], reports[1]['cases'])
                self.assertTrue(all(r['pairs'] == 448 for r in reports))
            except Exception as error:
                detail = '\n'.join(p.name + ': ' + p.read_text(errors='replace') for p in root.iterdir())
                raise AssertionError(f'{error}\n{detail}') from error
            finally:
                for process in processes:
                    if process.poll() is None: process.kill()
                    process.wait(timeout=5)
                for log in logs: log.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker': _worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else: unittest.main()

"""Independent arithmetic and real WORLD/DDP checks for the fixed joint loss."""
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

MIX = {'version': 'binary-pairwise-logistic-bce-mix05-v1',
       'kind': 'binary_pairwise_logistic_bce', 'temperature': 1.,
       'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch',
       'pointwise_weight': .5}
LABELS = [0.] * 56 + [1.] * 8


def _softplus(value):
    return max(value, 0.) + math.log1p(math.exp(-abs(value)))


def _sigmoid(value):
    return 1. / (1. + math.exp(-value)) if value >= 0 else math.exp(value) / (1. + math.exp(value))


def _reference(values):
    loss = 0.; gradient = [0.] * 64
    for p in range(56, 64):
        for n in range(56):
            delta = values[n] - values[p]
            loss += .5 * _softplus(delta) / 448
            derivative = .5 * _sigmoid(delta) / 448
            gradient[n] += derivative; gradient[p] -= derivative
    for i, (value, label) in enumerate(zip(values, LABELS)):
        loss += .5 * (_softplus(value) - label * value) / 64
        gradient[i] += .5 * (_sigmoid(value) - label) / 64
    return loss, gradient


def _worker(rank, rendezvous, output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                self.age_weight = torch.nn.Parameter(torch.tensor(.2))
                with torch.no_grad():
                    self.linear.weight.copy_(torch.tensor([[.7, -.4]])); self.linear.bias.fill_(.125)
            def forward(self, x, age):
                return self.linear(x).reshape(-1) + self.age_weight * age
        model = Model(); ddp = DistributedDataParallel(model, broadcast_buffers=False)
        x = torch.tensor([[.03 * (i - 32), (i % 7 - 3) / 5.] for i in range(64)])
        ages = torch.tensor([(i % 11 - 5) / 7. for i in range(64)])
        labels = torch.tensor(LABELS); part = slice(rank * 32, (rank + 1) * 32)
        actual, targets = gather_pairwise_inputs(ddp(x[part], ages[part]), labels[part])
        expected_logits = .7 * x[:, 0] - .4 * x[:, 1] + .125 + .2 * ages
        torch.testing.assert_close(actual, expected_logits, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(targets, labels, rtol=0, atol=0)
        expected, wanted = _reference(actual.detach().tolist())
        actual.retain_grad(); loss = classification_loss(actual, targets, MIX); loss.backward()
        torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(actual.grad, actual.new_tensor(wanted), rtol=3e-6, atol=3e-7)
        wanted = x.new_tensor(wanted)
        torch.testing.assert_close(model.linear.weight.grad, (wanted[:, None] * x).sum(0)[None], rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(model.linear.bias.grad, wanted.sum().reshape(1), rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(model.age_weight.grad, (wanted * ages).sum(), rtol=3e-6, atol=3e-7)
        if torch.cuda.is_initialized():
            raise AssertionError('CPU fixture initialized GPU')
        Path(output).write_text(json.dumps({'rank': rank, 'local_positives': int(labels[part].sum()),
            'global_inputs': 64, 'pairs': 448, 'loss': loss.item(), 'weight_gradient': model.linear.weight.grad.tolist(),
            'bias_gradient': model.linear.bias.grad.item(), 'age_gradient': model.age_weight.grad.item(),
            'all_global_logit_gradients_checked': True}))
    finally:
        dist.destroy_process_group()


class PairwiseBceMixLossTest(unittest.TestCase):
    def test_independent_value_gradients_bias_shift_extremes_and_legacy(self):
        import torch
        self.assertIs(validate_loss(MIX), MIX)
        labels = torch.tensor(LABELS)
        for values in ([.03 * (i - 32) for i in range(64)], [0.] * 64,
                       [10000., -10000.] * 32):
            logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            expected, wanted = _reference(values)
            loss = classification_loss(logits, labels, MIX)
            gradient, = torch.autograd.grad(loss, logits)
            self.assertEqual(loss.dtype, torch.float32)
            torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=3e-6, atol=3e-7)
            torch.testing.assert_close(gradient, gradient.new_tensor(wanted), rtol=3e-6, atol=3e-7)
            self.assertTrue(bool(torch.isfinite(loss))); self.assertTrue(bool(torch.isfinite(gradient).all()))
        values = [.03 * (i - 32) for i in range(64)]
        bias = torch.tensor(2.25, requires_grad=True)
        shifted = classification_loss(torch.tensor(values) + bias, labels, MIX)
        actual_bias, = torch.autograd.grad(shifted, bias)
        expected, wanted = _reference([v + 2.25 for v in values])
        self.assertAlmostEqual(shifted.item(), expected, delta=5e-7)
        self.assertAlmostEqual(actual_bias.item(), sum(wanted), delta=3e-7)
        self.assertGreater(abs(shifted.item() - _reference(values)[0]), .1)
        # Closed-form old zero-logit objectives and gradients preserve every old branch.
        base = {'temperature': 1., 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}
        from retrain.classification_loss import PAIR_HARD_POSITIVE_LOSS, PAIR_AUC_MARGIN_LOSS
        old = [(None, math.log(2.), [.5/64]*56+[-.5/64]*8),
               ({**base, 'kind': 'binary_pairwise_logistic', 'version': 'binary-pairwise-logistic-v1'}, math.log(2.), [.5/56]*56+[-.5/8]*8),
               ({**base, 'kind': 'binary_pairwise_sigmoid', 'version': 'binary-pairwise-sigmoid-v1'}, 1., [.5/56]*56+[-.5/8]*8),
               (PAIR_HARD_POSITIVE_LOSS, math.log(2.), [.5/56]*56+[-.15625]*2+[-.03125]*6),
               (PAIR_AUC_MARGIN_LOSS, 1., [2./56]*56+[-2./8]*8)]
        for spec, expected, wanted in old:
            logits = torch.zeros(64, requires_grad=True)
            loss = classification_loss(logits, labels, spec); gradient, = torch.autograd.grad(loss, logits)
            self.assertAlmostEqual(loss.item(), expected, delta=2e-7)
            torch.testing.assert_close(gradient, gradient.new_tensor(wanted), rtol=3e-6, atol=3e-7)

    def test_WORLD_DDP_all_logit_classifier_bias_and_age_gradients(self):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='pair-bce-gloo-', dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory); processes = []; logs = []; deadline = time.monotonic() + 80
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as endpoint:
                    endpoint.bind(('127.0.0.1', 0)); rendezvous = f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
                for rank in range(2):
                    log = (root/f'{rank}.log').open('wb'); logs.append(log)
                    processes.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                        str(rank), rendezvous, str(root/f'{rank}.json')], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                for process in processes:
                    self.assertEqual(process.wait(timeout=max(.1, deadline-time.monotonic())), 0)
                reports = [json.loads((root/f'{rank}.json').read_text()) for rank in range(2)]
                self.assertEqual([r['local_positives'] for r in reports], [0, 8])
                for key in ('loss', 'weight_gradient', 'bias_gradient', 'age_gradient'):
                    self.assertEqual(reports[0][key], reports[1][key])
                self.assertTrue(all(r['all_global_logit_gradients_checked'] for r in reports))
            except Exception as error:
                detail = '\n'.join(p.name+': '+p.read_text(errors='replace') for p in root.iterdir())
                raise AssertionError(f'{error}\n{detail}') from error
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                for log in logs:
                    log.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        _worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        unittest.main()

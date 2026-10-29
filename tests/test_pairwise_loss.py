"""Three CPU checks, including real WORLD gather + Gloo DDP parameter gradients."""
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

PAIRWISE = {'version': 'binary-pairwise-logistic-v1', 'kind': 'binary_pairwise_logistic',
            'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}


def _gloo_worker(rank, rendezvous, output):
    started = time.monotonic()

    def phase(stage):
        now = time.monotonic()
        print(json.dumps({'rank': rank, 'pid': os.getpid(), 'stage': stage,
                          'time': time.time(), 'monotonic': now,
                          'elapsed_seconds': now - started}), flush=True)

    phase('worker_entry')
    phase('torch_import_before')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    phase('torch_import_after')

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise AssertionError('CPU fixture must not initialize GPU')
    phase('init_process_group_before')
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    phase('init_process_group_after')
    try:
        i = torch.arange(64, dtype=torch.float32)
        inputs = torch.stack(((i - 31.5) / 16, (i.remainder(7) - 3) / 4,
                              ((3 * i).remainder(11) - 5) / 5), dim=1)
        labels = torch.cat((torch.zeros(56), torch.ones(8)))

        def model():
            linear = torch.nn.Linear(3, 1)
            with torch.no_grad():
                linear.weight.copy_(torch.tensor([[.3, -.2, .1]]))
                linear.bias.fill_(.15)
            return linear

        phase('DDP_construction_before')
        network = model()
        ddp = DistributedDataParallel(network, broadcast_buffers=False)
        phase('DDP_construction_after')
        interval = slice(rank * 32, (rank + 1) * 32)
        phase('forward_before')
        local_logits = ddp(inputs[interval]).reshape(-1)
        phase('forward_after')
        phase('gather_before')
        global_logits, global_targets = gather_pairwise_inputs(local_logits, labels[interval])
        phase('gather_after')
        torch.testing.assert_close(global_targets, labels, rtol=0, atol=0)
        if global_logits.dtype != torch.float32 or global_logits.numel() != 64:
            raise AssertionError('gather must retain all 64 FP32 logits')
        loss = classification_loss(global_logits, global_targets, PAIRWISE)
        phase('backward_before')
        loss.backward()
        phase('backward_after')

        phase('reference_and_assertions_before')
        reference = model()
        full_logits = reference(inputs).reshape(-1)
        differences = full_logits[labels == 0][None, :] - full_logits[labels == 1][:, None]
        expected = torch.logaddexp(torch.zeros_like(differences), differences).mean()
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
        phase('reference_and_assertions_after')
        phase('report_before')
        Path(output).write_text(json.dumps({'passed': True, 'rank': rank, 'global_pairs': 448,
            'local_positives': int(labels[interval].sum()), 'loss': float(loss.detach()),
            'weight_gradient': network.weight.grad.tolist(), 'bias_gradient': network.bias.grad.tolist(),
            'gpu_initialized': False}))
        phase('report_written')
    finally:
        phase('destroy_process_group_before')
        dist.destroy_process_group()
        phase('destroy_process_group_after')
    phase('worker_return')


class PairwiseLossTest(unittest.TestCase):
    def test_value_gradient_common_shift_and_default_BCE(self):
        import torch
        from torch.nn import functional as F

        values = [-1.7, .2, 1.3, -.6, .9, -2.2]
        labels = [1, 0, 1, 0, 0, 1]
        positives = [i for i, label in enumerate(labels) if label == 1]
        negatives = [i for i, label in enumerate(labels) if label == 0]
        pairs = len(positives) * len(negatives)
        expected_value, expected_gradient = 0., [0.] * len(values)
        # Independent scalar logistic arithmetic, including its analytic gradient.
        for p in positives:
            for n in negatives:
                difference = values[n] - values[p]
                expected_value += math.log1p(math.exp(difference)) / pairs
                derivative = 1. / (1. + math.exp(-difference)) / pairs
                expected_gradient[p] -= derivative
                expected_gradient[n] += derivative
        logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor(labels, dtype=torch.float64)
        self.assertIs(validate_loss(PAIRWISE), PAIRWISE)
        actual = classification_loss(logits, targets, PAIRWISE)
        gradient, = torch.autograd.grad(actual, logits)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertAlmostEqual(actual.item(), expected_value, places=6)
        torch.testing.assert_close(gradient, gradient.new_tensor(expected_gradient), rtol=2e-6, atol=1e-7)
        bias = torch.tensor(3.25, dtype=torch.float64, requires_grad=True)
        shifted = classification_loss(logits.detach() + bias, targets, PAIRWISE)
        bias_gradient, = torch.autograd.grad(shifted, bias)
        torch.testing.assert_close(shifted, actual.detach(), rtol=2e-6, atol=1e-7)
        self.assertLess(abs(bias_gradient.item()), 1e-7)
        bce = classification_loss(logits, targets, None)
        torch.testing.assert_close(bce, F.binary_cross_entropy_with_logits(logits, targets), rtol=0, atol=0)

    def test_extreme_and_zero_logits_have_finite_loss_and_gradients(self):
        import torch
        logits = torch.tensor([-10000., 10000., 0., 10000., -10000., 0.], requires_grad=True)
        targets = torch.tensor([1., 1., 1., 0., 0., 0.])
        loss = classification_loss(logits, targets, PAIRWISE)
        gradient, = torch.autograd.grad(loss, logits)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(gradient).all().item())

    def test_two_rank_Gloo_DDP_matches_complete_64_sample_parameter_gradients(self):
        # The test file is deployed inside A's isolated CPU prep. Direct child
        # entry avoids re-running a caller's unittest runner during spawn/import.
        child_env = dict(os.environ)
        child_env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                         CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                         GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='pairwise-gloo-', dir=Path(__file__).resolve().parent) as scratch:
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
                # Select one loopback endpoint, then release it for rank 0's
                # TCP store. A collision fails this attempt; there is no retry.
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
                    self.assertFalse(report['gpu_initialized'])
            except Exception as error:
                # Preserve both ranks' flushed phases, reports and live exit
                # states in the failure before TemporaryDirectory cleans up.
                raise AssertionError(f'{type(error).__name__}: {error}\n{diagnostics()}') from error
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()  # Only this fixture's own bounded CPU children.
                    process.wait(timeout=5)
                for log in logs:
                    log.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--gloo-worker':
        _gloo_worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        unittest.main()

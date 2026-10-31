"""Two CPU checks for the raw-logit objective and its saved WORLD identity."""
from datetime import timedelta
import copy
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
from retrain import classification_loss as losses

PAIRWISE = {'version': 'binary-pairwise-logistic-v1', 'kind': 'binary_pairwise_logistic',
            'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}
POLICY = {'version': 'global-raw-logit-square01-v1', 'coefficient': .1,
          'dtype': 'float32', 'scope': 'global_batch', 'reduction': 'mean',
          'logits': 'final_image_plus_age_raw', 'schedule': 'constant_from_step1'}


def _job():
    from retrain.age import make_spec
    from retrain.config import default_job
    job = default_job('output-objective-fixture', 'convnext_small')
    job.update(precision='fp32', loss=copy.deepcopy(PAIRWISE), age=make_spec({
        'rows': [{'name': 'a', 'sha256': 'a' * 64, 'age': 40},
                 {'name': 'b', 'sha256': 'b' * 64, 'age': 60}], 'fit_indices': [0, 1]}))
    return job


def _gloo_worker(rank, rendezvous, checkpoint_dir, output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from retrain.config import job_contract
    from retrain.engine import _validate_payload
    from retrain.persistence import CheckpointStore

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = CheckpointStore(checkpoint_dir).load()
    job = payload['job']
    _validate_payload(payload, {'job': job_contract(job)}, 'synthetic-only', job, 'train')
    if job['output_regularization'] != POLICY or job['loss'] != PAIRWISE:
        raise AssertionError('saved objective identity changed')
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        i = torch.arange(64, dtype=torch.float32)
        inputs = torch.stack(((i - 31.5) / 16, (i.remainder(7) - 3) / 4,
                              ((3 * i).remainder(11) - 5) / 5), dim=1)
        ages = torch.stack(((i - 20) / 30, (i.remainder(3) == 0).float()), dim=1)
        labels = torch.cat((torch.zeros(56), torch.ones(8)))

        class SharedParameters(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([.3, -.2, .1]))
                self.bias = torch.nn.Parameter(torch.tensor(.15))
                self.age_weight = torch.nn.Parameter(torch.tensor([.05, -.075]))

            def forward(self, images, age):
                return images @ self.weight + self.bias + age @ self.age_weight

        network = SharedParameters()
        ddp = DistributedDataParallel(network, broadcast_buffers=False)
        interval = slice(rank * 32, (rank + 1) * 32)
        local_logits = ddp(inputs[interval], ages[interval])
        global_logits, global_labels = losses.gather_pairwise_inputs(local_logits, labels[interval])
        loss = losses.regularized_pairwise_loss(global_logits, global_labels,
                                                job['loss'], job['output_regularization'])
        loss.backward()

        # Independent global reference: explicit logaddexp pairs and scalar
        # squared terms, without the production objective or its regularizer.
        reference = SharedParameters()
        z = reference(inputs, ages)
        differences = z[labels == 0][None, :] - z[labels == 1][:, None]
        expected = torch.logaddexp(torch.zeros_like(differences), differences).mean()
        expected = expected + sum(value * value for value in z.unbind()) / 640
        expected.backward()
        torch.testing.assert_close(global_labels, labels, rtol=0, atol=0)
        if global_logits.dtype != torch.float32 or global_logits.numel() != 64 or differences.numel() != 448:
            raise AssertionError('the complete 64-logit/448-pair objective is required')
        torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-7)
        for name, parameter in network.named_parameters():
            target = dict(reference.named_parameters())[name]
            torch.testing.assert_close(parameter.grad, target.grad, rtol=2e-5, atol=2e-7)
            if not bool((target.grad.abs() > 1e-5).all()):
                raise AssertionError(f'{name} must expose a nonzero shared-parameter gradient')
        # The pairwise term cancels a shared shift; only the new raw-logit
        # term contributes 0.2 * mean(z) to the common bias derivative.
        torch.testing.assert_close(network.bias.grad, .2 * z.detach().mean(), rtol=2e-5, atol=2e-7)
        if torch.cuda.is_initialized():
            raise AssertionError('CPU objective fixture initialized GPU')
        Path(output).write_text(json.dumps({'passed': True, 'rank': rank,
            'local_positives': int(labels[interval].sum()), 'global_samples': 64,
            'global_pairs': 448, 'loss': float(loss.detach()),
            'gradients': {name: parameter.grad.tolist() for name, parameter in network.named_parameters()},
            'saved_policy': job['output_regularization'], 'gpu_initialized': False}))
    finally:
        dist.destroy_process_group()


class OutputRegularizationTest(unittest.TestCase):
    def _objective(self):
        objective = getattr(losses, 'regularized_pairwise_loss', None)
        self.assertTrue(callable(objective), 'production raw-logit total objective is missing')
        return objective

    def test_raw_logit_value_shared_bias_age_gradients_and_legacy_None_identity(self):
        objective = self._objective()
        import numpy as np
        import torch
        from retrain.config import job_contract, load_config
        from retrain.persistence import capture_rng, canonical_digest

        # Hand-derived at z=[-1,0,1,2], y=[1,0,1,0]: differences
        # [1,3,-1,1], raw square mean=1.5, and shared w/b/age chain rule.
        x, age = torch.tensor([-2., 0., 2., 4.]), torch.tensor([0., 1., 0., 1.])
        labels = torch.tensor([1., 0., 1., 0.])
        w, bias, age_weight = (torch.tensor(v, requires_grad=True) for v in (.5, 0., 0.))
        z = w * x + bias + age_weight * age
        before = capture_rng()
        actual = objective(z, labels, PAIRWISE, POLICY)
        gradients = torch.autograd.grad(actual, (w, bias, age_weight))
        after = capture_rng()
        s, t = 1 / (1 + math.exp(-1)), 1 / (1 + math.exp(-3))
        base = (2 * math.log1p(math.exp(1)) + math.log1p(math.exp(3))
                + math.log1p(math.exp(-1))) / 4
        self.assertEqual(actual.dtype, torch.float32)
        self.assertAlmostEqual(actual.item(), base + .15, places=6)
        for value, expected in zip(gradients, (1.5 * (s + t) + .1, .1, .35 + (s + t) / 4)):
            self.assertAlmostEqual(value.item(), expected, places=6)
        self.assertEqual(before['python'], after['python'])
        self.assertEqual(before['numpy'][0], after['numpy'][0])
        self.assertTrue(np.array_equal(before['numpy'][1], after['numpy'][1]))
        self.assertEqual(before['numpy'][2:], after['numpy'][2:])
        self.assertTrue(torch.equal(before['torch'], after['torch']))
        self.assertIsNone(after['cuda'])

        legacy_z = torch.tensor([-1., 0., 1., 2.], requires_grad=True)
        original = losses.classification_loss(legacy_z, labels, PAIRWISE)
        original_gradient, = torch.autograd.grad(original, legacy_z)
        before_None = capture_rng()
        unchanged = objective(legacy_z, labels, PAIRWISE, None)
        unchanged_gradient, = torch.autograd.grad(unchanged, legacy_z)
        after_None = capture_rng()
        torch.testing.assert_close(unchanged, original, rtol=0, atol=0)
        torch.testing.assert_close(unchanged_gradient, original_gradient, rtol=0, atol=0)
        self.assertAlmostEqual(unchanged.item(), base, places=6)
        u = (s + t) / 4
        torch.testing.assert_close(unchanged_gradient, torch.tensor([-u, .25, -.25, u]), rtol=2e-6, atol=1e-7)
        self.assertEqual(before_None['python'], after_None['python'])
        self.assertTrue(np.array_equal(before_None['numpy'][1], after_None['numpy'][1]))
        self.assertEqual(before_None['numpy'][2:], after_None['numpy'][2:])
        self.assertTrue(torch.equal(before_None['torch'], after_None['torch']))
        self.assertIsNone(after_None['cuda'])
        with tempfile.TemporaryDirectory(prefix='output-none-', dir=Path(__file__).parent) as scratch:
            path = Path(scratch) / 'config.json'
            path.write_text(json.dumps({'jobs': [_job()]}))
            old_job = load_config(path, environ={})['jobs'][0]
            path.write_text(json.dumps({'jobs': [{**_job(), 'output_regularization': None}]}))
            none_job = load_config(path, environ={})['jobs'][0]
            self.assertNotIn('output_regularization', none_job)
            self.assertNotIn('output_regularization', job_contract(none_job))
            self.assertEqual(canonical_digest(job_contract(none_job)), canonical_digest(job_contract(old_job)))

    def test_saved_total_objective_matches_global_shared_gradients_in_two_rank_Gloo(self):
        self._objective()
        import torch
        from retrain.config import job_contract, load_config, validate_job
        from retrain.engine import _validate_payload
        from retrain.persistence import CheckpointStore, canonical_digest

        child_env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                         CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                         GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='output-gloo-', dir=Path(__file__).parent) as scratch:
            scratch = Path(scratch)
            config_path = scratch / 'config.json'
            config_path.write_text(json.dumps({'jobs': [{**_job(), 'output_regularization': POLICY}]}))
            job = load_config(config_path, environ={})['jobs'][0]
            validate_job(job)
            identity = {'job': job_contract(job)}
            payload = {'format_version': 1, 'identity': identity, 'job': job,
                       'manifest_digest': 'synthetic-only', 'mode': 'train',
                       'model': {'fixture': torch.tensor([1.])}, 'successful_updates': 1}
            store = CheckpointStore(scratch / 'checkpoints')
            store.save(payload)
            loaded = store.load()
            _validate_payload(loaded, identity, 'synthetic-only', job, 'train')
            self.assertEqual(loaded['job']['loss'], PAIRWISE)
            self.assertEqual(loaded['job']['output_regularization'], POLICY)
            self.assertEqual(canonical_digest(loaded['identity']['job']), canonical_digest(job_contract(job)))
            processes, logs = [], []
            deadline = time.monotonic() + 75
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as endpoint:
                    endpoint.bind(('127.0.0.1', 0))
                    rendezvous = f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
                for rank in range(2):
                    log = (scratch / f'rank-{rank}.log').open('wb')
                    logs.append(log)
                    processes.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                        '--gloo-worker', str(rank), rendezvous, str(scratch / 'checkpoints'),
                        str(scratch / f'rank-{rank}.json')], env=child_env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                for rank, process in enumerate(processes):
                    code = process.wait(timeout=max(.1, deadline - time.monotonic()))
                    self.assertEqual(code, 0, f'rank {rank} exited with status {code}')
                reports = [json.loads((scratch / f'rank-{rank}.json').read_text()) for rank in range(2)]
                self.assertEqual([report['local_positives'] for report in reports], [0, 8])
                for report in reports:
                    self.assertTrue(report['passed'])
                    self.assertEqual(report['global_samples'], 64)
                    self.assertEqual(report['global_pairs'], 448)
                    self.assertEqual(report['saved_policy'], POLICY)
                    self.assertFalse(report['gpu_initialized'])
                print(json.dumps({'output_regularization_Gloo': reports}), flush=True)
            except Exception as error:
                details = '\n'.join(f'rank={rank} pid={process.pid} exit={process.poll()}\n'
                    + (scratch / f'rank-{rank}.log').read_text(errors='replace')
                    for rank, process in enumerate(processes))
                raise AssertionError(f'{type(error).__name__}: {error}\n{details}') from error
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()  # Only this test's own bounded CPU children.
                    process.wait(timeout=5)
                for log in logs:
                    log.close()
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--gloo-worker':
        _gloo_worker(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        unittest.main()

"""Narrow VICReg fixtures; parent A executes them under the guarded CPU channel."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'runtime/bundle'))

X1 = [[0.5, -1.0], [1.5, 0.0], [-0.5, 2.0], [0.0, 0.5]]
X2 = [[0.0, -0.5], [1.0, 0.5], [-1.0, 1.5], [0.5, 0.0]]


def manual_loss(a, b):
    n, d = len(a), len(a[0])
    invariance = sum((a[i][j] - b[i][j]) ** 2 for i in range(n) for j in range(d)) / (n * d)

    def variance_term(z):
        total = 0.0
        for j in range(d):
            mean = sum(z[i][j] for i in range(n)) / n
            var = sum((z[i][j] - mean) ** 2 for i in range(n)) / (n - 1)
            total += max(0.0, 1.0 - math.sqrt(var + 1e-4))
        return total / d

    def covariance_term(z):
        means = [sum(z[i][j] for i in range(n)) / n for j in range(d)]
        total = 0.0
        for j in range(d):
            for k in range(d):
                if j == k:
                    continue
                cov = sum((z[i][j] - means[j]) * (z[i][k] - means[k]) for i in range(n)) / (n - 1)
                total += cov * cov
        return total / d

    variance = 0.5 * (variance_term(a) + variance_term(b))
    covariance = covariance_term(a) + covariance_term(b)
    return 25.0 * invariance + 25.0 * variance + covariance


def _vicreg_child(rank, init_method, output):
    import torch
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from retrain.distributed import RankContext
    from retrain.vicreg import gather_embeddings, vicreg_loss
    context = RankContext(rank, 2, 'cpu', init_method=init_method, timeout_seconds=120)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(1234)
        model = nn.Linear(6, 4, bias=False)
        distributed = DDP(model)
        generator = torch.Generator().manual_seed(4321)
        view1 = torch.randn(8, 6, generator=generator)
        view2 = torch.randn(8, 6, generator=generator)
        shard = slice(rank * 4, (rank + 1) * 4)
        loss = vicreg_loss(gather_embeddings(distributed(view1[shard]), rank, 2),
                           gather_embeddings(distributed(view2[shard]), rank, 2))
        loss.backward()
        gradient = model.weight.grad.detach().clone()

        torch.manual_seed(1234)
        reference = nn.Linear(6, 4, bias=False)
        reference_loss = vicreg_loss(reference(view1), reference(view2))
        reference_loss.backward()

        gradient_difference = (gradient - reference.weight.grad).abs().max().item()
        loss_difference = abs(loss.item() - reference_loss.item())
        torch.optim.SGD(distributed.parameters(), lr=0.1).step()
        torch.optim.SGD(reference.parameters(), lr=0.1).step()
        parameter_difference = (model.weight.detach() - reference.weight.detach()).abs().max().item()
        result = {'rank': rank, 'world_size': 2, 'loss': loss.item(),
                  'gradient_difference': gradient_difference, 'loss_difference': loss_difference,
                  'parameter_difference': parameter_difference,
                  'gradient_finite': bool(torch.isfinite(gradient).all())}
    finally:
        context.close()
    Path(output).write_text(json.dumps(result))


if os.environ.get('NCCU_VICREG_CHILD_RANK') is not None:
    _vicreg_child(int(os.environ['NCCU_VICREG_CHILD_RANK']), os.environ['NCCU_VICREG_INIT'],
                  os.environ['NCCU_VICREG_OUT'])
    raise SystemExit(0)


class VicregTest(unittest.TestCase):
    def test_small_matrix_value_and_both_branch_gradients(self):
        import torch
        from retrain.vicreg import vicreg_loss
        loss = vicreg_loss(torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32))
        self.assertAlmostEqual(loss.item(), manual_loss(X1, X2), delta=1e-4)

        left = torch.tensor(X1, dtype=torch.float64, requires_grad=True)
        right = torch.tensor(X2, dtype=torch.float64, requires_grad=True)
        value = vicreg_loss(left, right)
        value.backward()
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(right.grad).all())
        self.assertGreater(left.grad.abs().max().item(), 0.0)
        self.assertGreater(right.grad.abs().max().item(), 0.0)
        for tensor, base, position in ((left, X1, (0, 0)), (left, X1, (2, 1)),
                                       (right, X2, (1, 0)), (right, X2, (3, 1))):
            step = 1e-6
            plus = torch.tensor(base, dtype=torch.float64)
            minus = torch.tensor(base, dtype=torch.float64)
            plus[position] += step
            minus[position] -= step
            with torch.no_grad():
                numeric = (vicreg_loss(plus, torch.tensor(X2 if tensor is left else X1, dtype=torch.float64)).item()
                           - vicreg_loss(minus, torch.tensor(X2 if tensor is left else X1, dtype=torch.float64)).item()) / (2 * step)
            self.assertAlmostEqual(tensor.grad[position].item(), numeric, delta=2e-4)

    def test_two_rank_ddp_global_loss_matches_monolithic_reference(self):
        directory = Path(tempfile.mkdtemp(prefix='vicreg-ddp-'))
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='',
                           ROCR_VISIBLE_DEVICES='', NCCU_VICREG_INIT=(directory / 'gloo-init').as_uri())
        processes = []
        try:
            for rank in (0, 1):
                child = environment.copy()
                child.update(NCCU_VICREG_CHILD_RANK=str(rank), NCCU_VICREG_OUT=str(directory / f'rank-{rank}.json'))
                processes.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve())],
                                                  env=child, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            results = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=240)
                self.assertEqual(process.returncode, 0, (stdout, stderr))
            for rank in (0, 1):
                results.append(json.loads((directory / f'rank-{rank}.json').read_text()))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            shutil.rmtree(directory, ignore_errors=True)
        for result in results:
            self.assertEqual(result['world_size'], 2)
            self.assertTrue(result['gradient_finite'])
            self.assertLess(result['loss_difference'], 1e-5)
            self.assertLess(result['gradient_difference'], 2e-5)
            self.assertLess(result['parameter_difference'], 5e-6)

    def test_single_rank_gather_is_noop_and_matches_formula(self):
        import torch
        from retrain.vicreg import gather_embeddings, vicreg_loss
        left = torch.tensor(X1, dtype=torch.float32)
        self.assertIs(gather_embeddings(left, 0, 1), left)
        self.assertAlmostEqual(vicreg_loss(left, torch.tensor(X2, dtype=torch.float32)).item(),
                               manual_loss(X1, X2), delta=1e-4)
        with self.assertRaises(ValueError):
            vicreg_loss(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 2.0]]))


if __name__ == '__main__':
    unittest.main()

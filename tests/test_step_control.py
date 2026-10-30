"""Two-rank Gloo fixtures for fixed-shape step-control collectives; parent A runs them in the Pod."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / 'runtime/bundle'
sys.path.insert(0, str(BUNDLE))


def _child(rank, init_method, output):
    from retrain.distributed import RankContext
    context = RankContext(rank, 2, 'cpu', init_method=init_method, timeout_seconds=120)
    result = {'rank': rank}
    try:
        result['any_mixed'] = context.any_flags(
            [False, True, True] if rank == 0 else [True, False, False])
        result['any_all_false'] = context.any_flags([False, False])
        result['failure_visible'] = context.any_flags([False, rank == 1])
        result['broadcast_true'] = context.broadcast_flag(True if rank == 0 else None)
        result['broadcast_false'] = context.broadcast_flag(False if rank == 0 else None)
        result['legacy_any'] = context.any(rank == 1)
        result['agreed_false'] = context.agreed(False)
        result['agreed_true'] = context.agreed(True)
        result['maximum'] = context.maximum(1.5 if rank == 0 else 2.5)
        result['values'] = context.values('v')
        result['gather'] = context.gather_indices([1, 2] if rank == 0 else [3, 4])
        try:
            context.agreed(rank == 0)
            result['agreed_disagree'] = 'no_error'
        except RuntimeError:
            result['agreed_disagree'] = 'RuntimeError'
    finally:
        context.close()
    Path(output).write_text(json.dumps(result))


if os.environ.get('NCCU_STEP_CHILD_RANK') is not None:
    _child(int(os.environ['NCCU_STEP_CHILD_RANK']), os.environ['NCCU_STEP_INIT'],
           os.environ['NCCU_STEP_OUT'])
    raise SystemExit(0)


class StepControlTest(unittest.TestCase):
    def test_two_rank_gloo_flags_and_legacy_semantics(self):
        directory = Path(tempfile.mkdtemp(prefix='step-control-'))
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           PYTHONDONTWRITEBYTECODE='1', NCCU_STEP_INIT=(directory / 'gloo-init').as_uri())
        processes = []
        try:
            for rank in (0, 1):
                child = environment.copy()
                child.update(NCCU_STEP_CHILD_RANK=str(rank), NCCU_STEP_OUT=str(directory / f'rank-{rank}.json'))
                processes.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve())],
                                                  env=child, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            outputs = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=180)
                self.assertEqual(process.returncode, 0, (stdout, stderr))
            for rank in (0, 1):
                outputs.append(json.loads((directory / f'rank-{rank}.json').read_text()))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            shutil.rmtree(directory, ignore_errors=True)
        for result in outputs:
            self.assertEqual(result['any_mixed'], [True, True, True])
            self.assertEqual(result['any_all_false'], [False, False])
            self.assertEqual(result['failure_visible'], [False, True])
            self.assertIs(result['broadcast_true'], True)
            self.assertIs(result['broadcast_false'], False)
            self.assertIs(result['legacy_any'], True)
            self.assertIs(result['agreed_false'], False)
            self.assertIs(result['agreed_true'], True)
            self.assertEqual(result['maximum'], 2.5)
            self.assertEqual(result['values'], ['v', 'v'])
            self.assertEqual(result['gather'], [[1, 2], [3, 4]])
            self.assertEqual(result['agreed_disagree'], 'RuntimeError')

    def test_world_one_flags_match_local_booleans(self):
        from retrain.distributed import RankContext
        context = RankContext(0, 1, 'cpu')
        values = [True, False, True]
        self.assertEqual(context.any_flags(values), values)
        self.assertEqual(values, [True, False, True])
        self.assertIs(context.broadcast_flag(True), True)
        self.assertIs(context.broadcast_flag(False), False)
        self.assertIs(context.any(False), False)
        self.assertIs(context.agreed(True), True)
        self.assertEqual(context.maximum(3.5), 3.5)
        self.assertEqual(context.gather_indices([4, 5]), [[4, 5]])
        context.close()


if __name__ == '__main__':
    unittest.main()

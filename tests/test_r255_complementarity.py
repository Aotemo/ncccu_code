"""Descriptive shared-pair repair and tie-aware correlations."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'runtime/bundle'), str(ROOT / 'tools/diagnostics')]
spec = importlib.util.spec_from_file_location('r255_complementarity', ROOT / 'tools/diagnostics/r255_complementarity.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_strict_shared_repair_keeps_ties_and_groups_separate():
    labels = [1, 1, 0, 0, 0]
    common = [0., 2., 1., 3., -1.]
    candidate = [3., 4., 1., 3., -1.]
    result = study.shared_repair(labels, [common] * 7, candidate, {0})
    assert result['fixed20'] == {'shared': 2, 'repaired': 1, 'ties': 1, 'still_wrong': 0}
    assert result['remaining_positive'] == {'shared': 1, 'repaired': 1, 'ties': 0, 'still_wrong': 0}
    assert result['all'] == {'shared': 3, 'repaired': 2, 'ties': 1, 'still_wrong': 0}


def test_correlation_uses_average_tied_ranks():
    value = study.correlations([1., 1., 2., 3.], [4., 4., 3., 2.])
    assert abs(value['Pearson'] + 1.) < 1e-12 and abs(value['Spearman'] + 1.) < 1e-12
    try:
        study.correlations([1., 1.], [2., 3.])
    except ValueError:
        pass
    else:
        raise AssertionError('undefined correlation accepted')


if __name__ == '__main__':
    test_strict_shared_repair_keeps_ties_and_groups_separate()
    test_correlation_uses_average_tied_ranks()
    print('R255 strict shared-pair repair and tie-aware correlations passed', flush=True)

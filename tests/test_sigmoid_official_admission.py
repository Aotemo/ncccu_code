"""CPU admission keeps both historical and final-ensemble-only sigmoid gates."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex


def test_sigmoid_official_policy_is_round_specific():
    spec = importlib.util.spec_from_file_location('candidate_admission', ROOT / 'tools/validation/check_candidate.py')
    admission = importlib.util.module_from_spec(spec); spec.loader.exec_module(admission)
    old = json.loads((ROOT / 'configs/R120-SCIENCE.json').read_text())
    new = json.loads((ROOT / 'configs/R254-A-SCIENCE.json').read_text())
    old_site = {'round_id': 'r120-historical'}
    new_site = {'round_id': 'r254-small-seed20261109-pairsigmoid-24k-p4-9010-20261104'}
    assert admission.sigmoid_formal_contract(old['job'], old, old_site)
    assert admission.sigmoid_formal_contract(new['job'], new, new_site)
    invalid = copy.deepcopy(new); invalid['official_evaluation'] = ex.OFFICIAL_GATE_0996
    assert not admission.sigmoid_formal_contract(invalid['job'], invalid, new_site)
    invalid = copy.deepcopy(old); invalid['official_evaluation'] = ex.OFFICIAL_FINAL_ENSEMBLE_ONLY
    assert not admission.sigmoid_formal_contract(invalid['job'], invalid, old_site)
    invalid = copy.deepcopy(new); invalid['job']['loss'] = ex.PAIR_LOGISTIC_LOSS
    assert not admission.sigmoid_formal_contract(invalid['job'], invalid, new_site)
    invalid = copy.deepcopy(new); invalid['job']['training_views'] = {'unexpected': True}
    assert not admission.sigmoid_formal_contract(invalid['job'], invalid, new_site)
    try:
        admission.sigmoid_formal_contract(old['job'], old, new_site)
    except ValueError:
        pass
    else:
        raise AssertionError('round/profile mismatch must not be accepted')


if __name__ == '__main__':
    test_sigmoid_official_policy_is_round_specific()
    print('historical and final-only sigmoid admission gates passed', flush=True)

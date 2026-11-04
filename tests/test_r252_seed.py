"""R252 changes only the registered R201 seed and fresh member identity."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(os.environ.get('NCCCU_TEST_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'runtime/bundle'))

def test_seed_identity_and_final_only_gate():
    from retrain import experiment91 as ex
    base=json.loads((ROOT/'configs/R201-SCIENCE.json').read_text())
    value=copy.deepcopy(base)
    value['candidate_profile']=ex.PAIR_SIGMOID_WIDE_SMALL_SEED20261107_BINDING
    value['job']['seed']=20261107
    value['candidate'].update(id='r252-convnext-small-imagenet22k-to-1k-24k-p4-seed20261107-scale085-pairsigmoid-wide2-peakgrad-9010',experiment_order=162)
    assert value['candidate_profile'] in ex.SUPPORTED_BINDINGS
    assert ex.candidate_from_configuration(value)==value['candidate']
    assert {k:v for k,v in value['job'].items() if k!='seed'}=={k:v for k,v in base['job'].items() if k!='seed'}
    invalid=copy.deepcopy(value);invalid['job']['seed']=20261029
    try:ex.candidate_from_configuration(invalid)
    except ValueError:pass
    else:raise AssertionError('R252 identity accepted historical seed')
    invalid=copy.deepcopy(value);invalid['job']['loss']=ex.PAIR_SIGMOID_LOSS
    try:ex.candidate_from_configuration(invalid)
    except ValueError:pass
    else:raise AssertionError('R252 identity accepted a different scientific loss')
    for label in ('R201','R202'):
        original=json.loads((ROOT/'configs'/(label+'-SCIENCE.json')).read_text())
        assert ex.candidate_from_configuration(original)==original['candidate']
    rid='r252-small-wide2-seed20261107-pairsigmoid-24k-p4-9010-20261104'
    assert ex.expected_official_policy({'round_id':rid},value)==ex.OFFICIAL_FINAL_ENSEMBLE_ONLY
    assert not ex.official_allowed_for_selected(ex.OFFICIAL_FINAL_ENSEMBLE_ONLY,{'auroc':1.0})

if __name__=='__main__':
    test_seed_identity_and_final_only_gate()
    print('R252 seed, R201/R202 invariants and final-only official gate passed',flush=True)

"""New-seed corner-view identity retains the original scientific recipe."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(os.environ.get('NCCCU_TEST_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'runtime/bundle'))


def test_new_quadrant_seed_and_original_profiles():
    from retrain import experiment91 as ex
    base=json.loads((ROOT/'configs/R172-SCIENCE.json').read_text())
    value=copy.deepcopy(base)
    value['candidate_profile']=ex.QUADRANT_MIL_SMALL_SEED20261106_BINDING
    value['job']['seed']=20261106
    value['candidate'].update(id='r251-convnext-small-imagenet22k-to-1k-24k-p4-seed20261106-scale085-pairlog-corner288max05-9010',
                              experiment_order=161)
    assert value['candidate_profile'] in ex.SUPPORTED_BINDINGS
    assert ex.candidate_from_configuration(value)==value['candidate']
    assert {k:v for k,v in value['job'].items() if k!='seed'}=={k:v for k,v in base['job'].items() if k!='seed'}
    invalid=copy.deepcopy(value);invalid['job']['seed']=20261029
    try:ex.candidate_from_configuration(invalid)
    except ValueError:pass
    else:raise AssertionError('new-seed profile accepted the historical seed')
    for label in ('R172','R173'):
        original=json.loads((ROOT/'configs'/(label+'-SCIENCE.json')).read_text())
        assert ex.candidate_from_configuration(original)==original['candidate']


def test_new_members_defer_official_evaluation():
    from retrain import experiment91 as ex
    for round_id,profile in (
        ('r250-small-wide2-seed29-pairsigmoid-24k-p4-9010-20261104',ex.PAIR_SIGMOID_WIDE_SMALL_BINDING),
        ('r251-small-quadrant-seed20261106-pairlog-24k-p4-9010-20261104',ex.QUADRANT_MIL_SMALL_SEED20261106_BINDING)):
        policy=ex.expected_official_policy({'round_id':round_id},{'candidate_profile':profile})
        assert policy==ex.OFFICIAL_FINAL_ENSEMBLE_ONLY
        assert ex.official_allowed_for_selected(policy,{'auroc':1.0}) is False
        try:ex.expected_official_policy({'round_id':round_id},{'candidate_profile':'wrong'})
        except ValueError:pass
        else:raise AssertionError('wrong profile was accepted')


if __name__=='__main__':
    test_new_quadrant_seed_and_original_profiles()
    test_new_members_defer_official_evaluation()
    print('new quadrant seed, historical recipes and final-only gate passed',flush=True)

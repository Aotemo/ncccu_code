"""The two fresh ensemble members cannot consume the final official evaluation."""
import os
import json
from pathlib import Path
import sys

ROOT=Path(os.environ.get('NCCCU_TEST_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'runtime/bundle'))


def test_final_only_policy_is_bound_to_the_exact_round_and_profile():
    from retrain import experiment91 as ex
    assert ex.ENSEMBLE_MEMBER_ROUNDS['r248-large22k-seed29-pairlog-storage32-24k-p4-9010-20261104']==ex.LARGE_BINDING
    assert ex.ENSEMBLE_MEMBER_ROUNDS['r249-small-gamma-seed29-pairsigmoid-24k-p4-9010-20261104']==ex.PAIR_SIGMOID_GAMMA_SMALL_BINDING
    for round_id,profile in ex.ENSEMBLE_MEMBER_ROUNDS.items():
        policy=ex.expected_official_policy({'round_id':round_id},{'candidate_profile':profile})
        assert policy==ex.OFFICIAL_FINAL_ENSEMBLE_ONLY
        assert ex.official_allowed_for_selected(policy,{'auroc':1.0}) is False
        try:ex.expected_official_policy({'round_id':round_id},{'candidate_profile':'wrong'})
        except ValueError:pass
        else:raise AssertionError('wrong member profile was accepted')
    assert ex.expected_official_policy({'round_id':'historical'},{})==ex.OFFICIAL_GATE_0996
    assert not ex.official_allowed_for_selected(ex.OFFICIAL_GATE_0996,{'auroc':.995})
    assert ex.official_allowed_for_selected(ex.OFFICIAL_GATE_0996,{'auroc':.996})
    for bad in (None,{}, {'auroc':float('nan')}, {'auroc':True}):
        try:ex.official_allowed_for_selected(ex.OFFICIAL_FINAL_ENSEMBLE_ONLY,bad)
        except ValueError:pass
        else:raise AssertionError('invalid selected AUROC was accepted')


def test_member_science_keeps_the_original_recipe_and_final_only_gate():
    from retrain import experiment91 as ex
    ignored={'public_asset_receipt','official_evaluation','comparison_note'}
    for label,base,round_id in (
        ('R246','R172','r246-small-quadrant-seed29-pairlog-24k-p4-9010-20261104'),
        ('R247','R182','r247-large-dinov3-seed29-pairlog-storage32-24k-p4-9010-20261104'),
        ('R248-A','R180','r248-large22k-seed29-pairlog-storage32-24k-p4-9010-20261104'),
        ('R248-B','R180','r248-large22k-seed29-pairlog-storage32-24k-p4-9010-20261104'),
        ('R249-A','R197','r249-small-gamma-seed29-pairsigmoid-24k-p4-9010-20261104'),
        ('R249-B','R197','r249-small-gamma-seed29-pairsigmoid-24k-p4-9010-20261104'),
        ('R250-A','R201','r250-small-wide2-seed29-pairsigmoid-24k-p4-9010-20261104'),
        ('R250-B','R201','r250-small-wide2-seed29-pairsigmoid-24k-p4-9010-20261104')):
        original=json.loads((ROOT/'configs'/(base+'-SCIENCE.json')).read_text())
        member=json.loads((ROOT/'configs'/(label+'-SCIENCE.json')).read_text())
        assert {k:v for k,v in original.items()if k not in ignored}=={k:v for k,v in member.items()if k not in ignored}
        assert ex.candidate_from_configuration(member)==member['candidate']
        assert member['official_evaluation']==ex.expected_official_policy({'round_id':round_id},member)


def test_new_seed_view_configs_keep_original_recipe():
    from retrain import experiment91 as ex
    import copy
    original=json.loads((ROOT/'configs/R172-SCIENCE.json').read_text())
    ignored={'public_asset_receipt','official_evaluation','comparison_note'}
    tensors=[]
    for pod in ('A','B'):
        member=json.loads((ROOT/'configs'/('R251-'+pod+'-SCIENCE.json')).read_text())
        assert member['candidate_profile']==ex.QUADRANT_MIL_SMALL_SEED20261106_BINDING
        assert member['job']['seed']==20261106 and ex.candidate_from_configuration(member)==member['candidate']
        assert member['official_evaluation']==ex.expected_official_policy(
            {'round_id':'r251-small-quadrant-seed20261106-pairlog-24k-p4-9010-20261104'},member)
        tensors.append(member['initial_model_tensor_sha256'])
        normalized=copy.deepcopy(member)
        normalized['candidate_profile']=original['candidate_profile'];normalized['job']['seed']=original['job']['seed']
        normalized['candidate']['id']=original['candidate']['id']
        normalized['candidate']['experiment_order']=original['candidate']['experiment_order']
        normalized['initial_model_tensor_sha256']=original['initial_model_tensor_sha256']
        assert {k:v for k,v in normalized.items()if k not in ignored}=={k:v for k,v in original.items()if k not in ignored}
    assert tensors[0]==tensors[1] and tensors[0]!=original['initial_model_tensor_sha256']


if __name__=='__main__':
    test_final_only_policy_is_bound_to_the_exact_round_and_profile()
    test_member_science_keeps_the_original_recipe_and_final_only_gate()
    test_new_seed_view_configs_keep_original_recipe()
    print('final ensemble-only official gate and historical gate passed',flush=True)

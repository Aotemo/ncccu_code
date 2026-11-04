"""The two fresh ensemble members cannot consume the final official evaluation."""
import os
import json
from pathlib import Path
import sys

ROOT=Path(os.environ.get('NCCCU_TEST_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'runtime/bundle'))


def test_final_only_policy_is_bound_to_the_exact_round_and_profile():
    from retrain import experiment91 as ex
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
        ('R247','R182','r247-large-dinov3-seed29-pairlog-storage32-24k-p4-9010-20261104')):
        original=json.loads((ROOT/'configs'/(base+'-SCIENCE.json')).read_text())
        member=json.loads((ROOT/'configs'/(label+'-SCIENCE.json')).read_text())
        assert {k:v for k,v in original.items()if k not in ignored}=={k:v for k,v in member.items()if k not in ignored}
        assert ex.candidate_from_configuration(member)==member['candidate']
        assert member['official_evaluation']==ex.expected_official_policy({'round_id':round_id},member)


if __name__=='__main__':
    test_final_only_policy_is_bound_to_the_exact_round_and_profile()
    test_member_science_keeps_the_original_recipe_and_final_only_gate()
    print('final ensemble-only official gate and historical gate passed',flush=True)

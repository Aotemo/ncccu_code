"""Only new seed identities; reuse unchanged production model and loss paths."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(os.environ.get('NCCCU_TEST_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'runtime/bundle'))


def test_two_new_replicas_keep_the_original_R120_job():
    from retrain import experiment91 as ex
    definitions=getattr(ex,'PAIR_SIGMOID_SEED_REPLICAS',None)
    assert definitions is not None,'two R120 seed replica identities are missing'
    base=json.loads((ROOT/'configs/R120-SCIENCE.json').read_text())
    for label,seed,order in [('R244',20261104,159),('R245',20261105,160)]:
        binding=next(key for key,value in definitions.items()if value==(label,seed,order))
        value=copy.deepcopy(base);value['candidate_profile']=binding;value['job']['seed']=seed
        value['candidate'].update(id=f'{label.lower()}-convnext-small-imagenet22k-to-1k-24k-p4-seed{seed}-scale085-pairsigmoid-9010',experiment_order=order)
        assert ex.candidate_from_configuration(value)==value['candidate']
        assert {k:v for k,v in value['job'].items()if k!='seed'}=={k:v for k,v in base['job'].items()if k!='seed'}
        assert binding in ex.SUPPORTED_BINDINGS


def test_original_two_sigmoid_identities_remain_exact():
    from retrain import experiment91 as ex
    for label in ('R120','R121'):
        value=json.loads((ROOT/'configs'/(label+'-SCIENCE.json')).read_text())
        assert ex.candidate_from_configuration(value)==value['candidate']


if __name__=='__main__':
    test_two_new_replicas_keep_the_original_R120_job()
    test_original_two_sigmoid_identities_remain_exact()
    print('two seed-registration and original-profile targets passed',flush=True)

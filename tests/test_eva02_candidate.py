"""Fixed public source, original training recipe, and conditional seed identity."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import experiment91 as ex,config,age

class EVA02CandidateTest(unittest.TestCase):
    def test_R220_same_small12k_seed27_and_R221_independent_recipe(self):
        synthetic=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]})
        for label,prior,order in(('R220','R219',133),('R221','R107',134)):
            science=json.loads((ROOT/'configs'/(label+'-SCIENCE.json')).read_text())
            baseline=json.loads((ROOT/'configs'/(prior+'-SCIENCE.json')).read_text())
            expected=copy.deepcopy(baseline['job'])
            if label=='R220':expected['seed']=20261027
            else:expected.update(name='eva02-base',model='eva02_base',normalization='clip')
            self.assertEqual(science['job'],expected)
            for key in('data_split','schedule','early_stopping'):self.assertEqual(science[key],baseline[key])
            self.assertEqual(science['official_evaluation'],ex.OFFICIAL_GATE_0996)
            self.assertEqual(science['candidate']['experiment_order'],order)
            self.assertEqual(ex.candidate_from_configuration(science),science['candidate'])
            if label=='R220':
                for key in('public_source','public_asset_receipt','paths','structure'):self.assertEqual(science[key],baseline[key])
            else:
                from retrain.eva02_model import WEIGHT_SHA,REVISION,VARIANT,LIBRARY,FORMAT
                source=science['public_source']
                self.assertEqual((source['model'],source['format'],source['revision'],source['sha256'],source['original_sha256']),
                    (VARIANT,FORMAT,REVISION,WEIGHT_SHA,WEIGHT_SHA))
                self.assertEqual(source['native_library'],LIBRARY)
            binding={'version':science['candidate_profile']}
            site={'gpus':[{}for _ in range(8)],'public_weight':'/synthetic/public.bin','public_provenance':'/synthetic/provenance.json'}
            site['science']={'candidate':science['candidate'],'binding_version':science['candidate_profile']}
            with patch.object(ex,'science_configuration',return_value=science),patch.object(ex,'binding',return_value=binding),patch.object(ex,'metadata',return_value={'AGE.json':synthetic}),patch.object(ex,'load_site',return_value=site):
                value=config.default_job(label,science['job']['model'],'random');value.update(science['job'])
                value.update(age=ex.resolved_age_spec(),experiment91=binding,initialization={'kind':'imagenet',
                    'format':science['public_source']['format'],'sha256':science['public_source']['sha256'],
                    'path':'/synthetic/public.bin','provenance':'/synthetic/provenance.json'})
                config.validate_job(value);self.assertEqual(ex.validate_job(value),value)
                for key,new in(('seed',20261029 if label=='R220'else 20261027),('backbone_lr',1e-5),('normalization','unit'),('nadam_policy',config.NADAM_RULE)):
                    wrong=copy.deepcopy(value);wrong[key]=new
                    with self.assertRaises(ValueError):ex.validate_job(wrong)
            for section,key,new in(('public_source','revision','0'*40),('public_source','original_sha256','0'*64),('candidate','readout','wrong')):
                wrong=copy.deepcopy(science);wrong[section][key]=new
                with self.assertRaises(ValueError):ex.candidate_from_configuration(wrong)

if __name__=='__main__':unittest.main()

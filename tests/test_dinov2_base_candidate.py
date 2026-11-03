"""Bind the independent original Base source and conditional EVA02 seed27."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import config,experiment91 as ex,age

class DINOv2BaseCandidateTest(unittest.TestCase):
    def test_R222_only_same_EVA02_seed27_after_R221R2(self):
        self.assertTrue(hasattr(ex,'EVA02_SEED27_BINDING'))
        original=json.loads((ROOT/'configs/R221R2-SCIENCE.json').read_text())
        value=json.loads((ROOT/'configs/R222-SCIENCE.json').read_text())
        self.assertEqual(value['candidate_profile'],ex.EVA02_SEED27_BINDING)
        self.assertEqual(value['candidate']['experiment_order'],137)
        expected=copy.deepcopy(original['job']);expected['seed']=20261027
        self.assertEqual(value['job'],expected)
        for key in('public_source','public_asset_receipt','paths','structure','data_split','schedule','early_stopping','official_evaluation'):
            self.assertEqual(value[key],original[key])
        self.assertEqual(ex.candidate_from_configuration(value),value['candidate'])

    def test_R223_original_Base_full512_original_optimization_binding(self):
        self.assertTrue(hasattr(ex,'DINOV2_BASE_BINDING'))
        science=json.loads((ROOT/'configs/R223-SCIENCE.json').read_text())
        original=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
        expected=copy.deepcopy(original['job']);expected.update(name='dinov2-base',model='dinov2_vitb14')
        self.assertEqual(science['job'],expected)
        self.assertEqual(science['candidate_profile'],ex.DINOV2_BASE_BINDING)
        self.assertEqual(science['candidate']['experiment_order'],138)
        for key in('data_split','schedule','early_stopping'):self.assertEqual(science[key],original[key])
        self.assertEqual(science['official_evaluation'],ex.OFFICIAL_GATE_0996)
        self.assertEqual(ex.candidate_from_configuration(science),science['candidate'])
        self.assertEqual(science['structure']['features'],1536)
        self.assertEqual(science['structure']['frozen_parameter_names'],['backbone.mask_token'])
        self.assertEqual(science['structure']['backbone_parameters'],86580480)
        synthetic=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]})
        binding={'version':science['candidate_profile']};site={'gpus':[{}for _ in range(8)],
            'public_weight':'/synthetic/original.pth','public_provenance':'/synthetic/PROVENANCE.json',
            'science':{'candidate':science['candidate'],'binding_version':science['candidate_profile']}}
        with patch.object(ex,'science_configuration',return_value=science),patch.object(ex,'binding',return_value=binding),patch.object(ex,'metadata',return_value={'AGE.json':synthetic}),patch.object(ex,'load_site',return_value=site):
            value=config.default_job('base-test','dinov2_vitb14');value.update(science['job'])
            value.update(age=ex.resolved_age_spec(),experiment91=binding,initialization={'kind':'self_supervised',
                'format':science['public_source']['format'],'sha256':science['public_source']['sha256'],
                'path':site['public_weight'],'provenance':site['public_provenance']})
            config.validate_job(value);self.assertEqual(ex.validate_job(value),value)
            for key,new in(('image_size',504),('seed',20261027),('normalization','clip'),('backbone_lr',1e-5),('head_seed',20261027),('nadam_policy',config.NADAM_RULE)):
                wrong=copy.deepcopy(value);wrong[key]=new
                with self.assertRaises(ValueError):ex.validate_job(wrong)
        for section,key,new in(('public_source','original_sha256','0'*64),('public_source','revision','0'*40),('candidate','readout','CLS_only'),('candidate','features',768)):
            wrong=copy.deepcopy(science);wrong[section][key]=new
            with self.assertRaises(ValueError):ex.candidate_from_configuration(wrong)

if __name__=='__main__':unittest.main()

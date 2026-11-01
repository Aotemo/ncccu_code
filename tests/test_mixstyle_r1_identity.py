"""Only the independent R134R1 identity; the fixed scientific strategy is unchanged."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import experiment91 as ex
from retrain.mixstyle import MIXSTYLE_RULE


class MixStyleR1IdentityTest(unittest.TestCase):
    def test_old_R13445_and_new_R134R1_46_keep_identical_science(self):
        old=json.loads((ROOT/'configs/R134-SCIENCE.json').read_text())
        new=json.loads((ROOT/'configs/R134R1-SCIENCE.json').read_text())
        for filename,order in (('R134-SCIENCE.json',45),('R134R1-SCIENCE.json',46)):
            path=ROOT/'configs'/filename;value=json.loads(path.read_text())
            site={'schema_version':1,'scope':'cpu_technical_prep','round_id':'identity-fixture',
                'git_commit':'0'*40,'work_root':str(ROOT),'round_root':str(ROOT/'rounds/identity-fixture'),
                'pod':{'namespace':'fixture','name':'fixture','uid':'00000000-0000-0000-0000-000000000001',
                       'container':'fixture','python':sys.executable},
                'gpus':[{'ordinal':i,'pci':f'0000:{i+1:02x}:00.0','vram_total_bytes':64*1024**3} for i in range(8)],
                **{key:str(ROOT/relative) for key,relative in value['paths'].items()},
                'science':{'binding_version':value['candidate_profile'],'candidate':value['candidate'],
                    'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
            with tempfile.TemporaryDirectory() as directory:
                binding=Path(directory)/'SITE_BINDING.json';binding.write_text(json.dumps(site))
                with patch.dict(os.environ,NCCCU_SITE_BINDING=str(binding),
                                NCCCU_SITE_BINDING_SHA256=hashlib.sha256(binding.read_bytes()).hexdigest()):
                    try:actual=ex.candidate_from_configuration(ex.science_configuration())
                    except ValueError as error:self.fail(f'adopted {filename} identity missing: {error}')
            self.assertEqual((actual['id'],actual['experiment_order']),(value['candidate']['id'],order))
            self.assertEqual(actual['mixstyle_policy'],MIXSTYLE_RULE)
        self.assertEqual(new['candidate_profile'],ex.MIXSTYLE_SMALL_R1_BINDING)
        self.assertEqual(new['job']['mixstyle_policy'],new['candidate']['mixstyle_policy'])
        self.assertEqual(new['job']['mixstyle_policy'],MIXSTYLE_RULE)
        self.assertEqual(new['official_evaluation'],ex.OFFICIAL_GATE_0996)
        self.assertEqual(new['initial_model_tensor_sha256'],old['initial_model_tensor_sha256'])
        self.assertIsNone(new['job']['initial_checkpoint'])
        self.assertNotIn('final_downsample_stride',new['job'])
        for key in ('tri3_downsample_policy','output_regularization','low_rank_policy','layerwise_lr_policy','mixup'):
            self.assertIsNone(new['job'].get(key))
        left,right=copy.deepcopy(old),copy.deepcopy(new)
        for value in (left,right):
            for key in ('candidate_profile','comparison_note'):value.pop(key,None)
            for key in ('id','experiment_order'):value['candidate'].pop(key,None)
        self.assertEqual(left,right)

"""One positive R134 registration check; fixed MixStyle has prior CPU evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex
from retrain.mixstyle import MIXSTYLE_RULE


class MixStyleIdentityTest(unittest.TestCase):
    def test_R134_exact_identity_direct_R107_science_and_old_R133_unchanged(self):
        original = json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
        current = json.loads((ROOT/'configs/R134-SCIENCE.json').read_text())
        for filename, order in (('R133-SCIENCE.json',44),('R134-SCIENCE.json',45)):
            path = ROOT/'configs'/filename
            declared = json.loads(path.read_text())
            site = {'schema_version':1,'scope':'cpu_technical_prep',
                'round_id':'identity-fixture','git_commit':'0'*40,
                'work_root':str(ROOT),'round_root':str(ROOT/'rounds/identity-fixture'),
                'pod':{'namespace':'fixture','name':'fixture',
                       'uid':'00000000-0000-0000-0000-000000000001',
                       'container':'fixture','python':sys.executable},
                'gpus':[{'ordinal':i,'pci':f'0000:{i+1:02x}:00.0','vram_total_bytes':64*1024**3} for i in range(8)],
                **{key:str(ROOT/relative) for key,relative in declared['paths'].items()},
                'science':{'binding_version':declared['candidate_profile'],
                    'candidate':declared['candidate'],'configuration':{
                        'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
            with tempfile.TemporaryDirectory() as folder:
                site_path = Path(folder)/'SITE_BINDING.json'
                site_path.write_text(json.dumps(site))
                with patch.dict(os.environ,NCCCU_SITE_BINDING=str(site_path),
                                NCCCU_SITE_BINDING_SHA256=hashlib.sha256(site_path.read_bytes()).hexdigest()):
                    try:
                        actual = ex.candidate_from_configuration(ex.science_configuration())
                    except ValueError as error:
                        self.fail(f'adopted {filename} identity is not connected: {error}')
            self.assertEqual((actual['id'],actual['experiment_order']),
                             (declared['candidate']['id'],order))
            self.assertEqual(actual.get('mixstyle_policy'),MIXSTYLE_RULE if order==45 else None)
        self.assertEqual(current['candidate_profile'],ex.MIXSTYLE_SMALL_BINDING)
        self.assertEqual(current['candidate']['mixstyle_policy'],MIXSTYLE_RULE)
        self.assertEqual(current['job']['mixstyle_policy'],MIXSTYLE_RULE)
        self.assertEqual(current['official_evaluation'],ex.OFFICIAL_GATE_0996)
        self.assertEqual(current['initial_model_tensor_sha256'],original['initial_model_tensor_sha256'])
        for key in ('tri3_downsample_policy','output_regularization','low_rank_policy','layerwise_lr_policy','mixup'):
            self.assertIsNone(current['job'].get(key))
        self.assertNotIn('final_downsample_stride',current['job'])
        self.assertIsNone(current['job']['initial_checkpoint'])
        left,right = copy.deepcopy(original),copy.deepcopy(current)
        for value in (left,right):
            for key in ('candidate_profile','comparison_note','official_evaluation'):
                value.pop(key,None)
            for key in ('id','experiment_order','mixstyle_policy'):
                value['candidate'].pop(key,None)
            value['job'].pop('mixstyle_policy',None)
        self.assertEqual(left,right)


if __name__=='__main__':
    unittest.main()

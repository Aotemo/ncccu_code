"""The explicit source must determine final LN; absent bindings never select R92."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as experiment

CONVNEXT22 = json.loads((ROOT / "configs/R114-SCIENCE.json").read_text())["candidate"]
STEM_XY = {'version':'convnext-small-rgbxy512-v1','channels':['R','G','B','X','Y'],
           'canvas_size':512,'coordinates':'2*k/(size-1)-1',
           'position':'after_augmentation_and_normalization','xy_initialization':'zeros'}


class CandidateBindingTest(unittest.TestCase):
    @staticmethod
    def stem_xy_fixture():
        import torch
        from retrain.config import default_job
        frames=[]
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features=torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(3,96,4,stride=4)))
                self.avgpool=torch.nn.AdaptiveAvgPool2d(1)
                self.classifier=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(96,1000))
            def forward(self,images):
                self.observed_images=images
                return self.classifier(self.features(images)[:,:,:1,:1])
        def native(**_kwargs):
            network=Network();stem=network.features[0][0]
            frames.append((id(stem.weight),stem.weight.detach().clone(),stem.bias.detach().clone()))
            return network
        job=default_job('rgbxy-small-fixture','convnext_small','random')
        job.update(image_size=512,prepared_size=512,precision='fp32',dropout=0.,age=None,seed=20261029)
        return job,native,frames

    def test_R135_zero_extension_coordinates_gradient_and_None_rng(self):
        import torch
        import numpy as np
        from retrain import models
        from retrain.persistence import capture_rng,restore_rng
        from retrain.engine import _optimizer
        job,native,frames=self.stem_xy_fixture()
        before=capture_rng()
        with patch('torchvision.models.convnext_small',side_effect=native):
            plain=models.create_model(job,initialize=False);plain_rng=capture_rng()
            restore_rng(before);old_None=models.create_model({**job,'stem_xy_policy':None},initialize=False)
            none_rng=capture_rng();restore_rng(before)
            model=models.create_model({**job,'stem_xy_policy':STEM_XY},initialize=False)
        stem=model.network.features[0][0]
        self.assertEqual(tuple(stem.weight.shape),(96,5,4,4))
        self.assertEqual(id(stem.weight),frames[-1][0])
        torch.testing.assert_close(stem.weight[:,:3],frames[-1][1],rtol=0,atol=0)
        torch.testing.assert_close(stem.bias,frames[-1][2],rtol=0,atol=0)
        self.assertEqual(torch.count_nonzero(stem.weight[:,3:]).item(),0)
        self.assertEqual(dict(model.named_buffers()),{})
        self.assertEqual(set(model.state_dict()),set(plain.state_dict()))
        for name,value in plain.state_dict().items():
            torch.testing.assert_close(old_None.state_dict()[name],value,rtol=0,atol=0)
            if name!='network.features.0.0.weight':
                torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        for state in (none_rng,capture_rng()):
            self.assertEqual(state['python'],plain_rng['python'])
            np.testing.assert_equal(state['numpy'],plain_rng['numpy'])
            torch.testing.assert_close(state['torch'],plain_rng['torch'],rtol=0,atol=0)
        with torch.no_grad():
            model.network.classifier[-1].weight.fill_(1);model.network.classifier[-1].bias.zero_();stem.bias.zero_()
        image=torch.zeros(1,3,512,512,requires_grad=True);model.train();model(image).sum().backward()
        actual=model.network.observed_images
        self.assertEqual(tuple(actual.shape),(1,5,512,512))
        self.assertEqual((actual[0,3,0,0].item(),actual[0,3,0,-1].item(),
                          actual[0,4,0,0].item(),actual[0,4,-1,0].item()),(-1.,1.,-1.,1.))
        torch.testing.assert_close(actual[0,3,:,256],torch.full((512,),1/511),rtol=0,atol=1e-7)
        oracle=torch.tensor([-1.,-509/511,-507/511,-505/511])
        torch.testing.assert_close(stem.weight.grad[:,3],oracle.reshape(1,1,4).expand(96,4,4),rtol=0,atol=1e-7)
        torch.testing.assert_close(stem.weight.grad[:,4],oracle.reshape(1,4,1).expand(96,4,4),rtol=0,atol=1e-7)
        model.eval();model(image.detach())
        torch.testing.assert_close(model.network.observed_images,actual.detach(),rtol=0,atol=0)
        optimizer=_optimizer(model,job)
        membership=[g for g in optimizer.param_groups if any(p is stem.weight for p in g['params'])]
        self.assertEqual(len(membership),1);self.assertEqual(membership[0]['role'],'backbone')
        self.assertEqual(membership[0]['weight_decay'],job['weight_decay']);self.assertFalse(optimizer.state)

    def test_R135_science_identity_and_default_factory_strict_roundtrip(self):
        import torch
        from retrain import models,config
        from retrain.persistence import CheckpointStore,canonical_digest
        from retrain.inference import _load_model
        new=json.loads((ROOT/'configs/R135-SCIENCE.json').read_text())
        self.configuration.write_text(json.dumps(new));self.site['science'].update(binding_version=new['candidate_profile'],candidate=new['candidate'])
        self.site['science']['configuration']['sha256']=hashlib.sha256(self.configuration.read_bytes()).hexdigest()
        self.path.write_text(json.dumps(self.site))
        try:actual=experiment.science_configuration();candidate=experiment.candidate_from_configuration(actual)
        except ValueError as exc:self.fail('R135 binding unavailable: '+str(exc))
        self.assertEqual(candidate,new['candidate']);self.assertEqual(candidate['experiment_order'],47)
        self.assertEqual(new['job']['stem_xy_policy'],STEM_XY);self.assertEqual(candidate['stem_xy_policy'],STEM_XY)
        self.assertEqual(new['official_evaluation'],experiment.OFFICIAL_GATE_0996)
        old=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text());left,right=copy.deepcopy(old),copy.deepcopy(new)
        for value in (left,right):
            for key in ('candidate_profile','initial_model_tensor_sha256','comparison_note','official_evaluation'):value.pop(key,None)
            for key in ('id','experiment_order','stem_xy_policy'):value['candidate'].pop(key,None)
            value['job'].pop('stem_xy_policy',None)
        self.assertEqual(left,right)
        job,native,_=self.stem_xy_fixture();original=config.job_contract(job)
        self.assertEqual(config.job_contract({**job,'stem_xy_policy':None}),original)
        path=self.configuration.parent/'runtime.json';path.write_text(json.dumps({'jobs':[{**job,'stem_xy_policy':STEM_XY}]}))
        job=config.load_config(path,environ={})['jobs'][0]
        self.assertEqual(config.job_contract(job)['stem_xy_policy'],STEM_XY)
        with patch('torchvision.models.convnext_small',side_effect=native):
            model=models.create_model(job,initialize=False).eval()
            with torch.no_grad():model.network.features[0][0].weight[:,3:].fill_(.125)
            payload={'format_version':1,'mode':'train','job':job,'model':model.state_dict(),'age':None,
                     'identity':{'job':config.job_contract(job)},'manifest_digest':'synthetic-only','successful_updates':0}
            checkpoint=CheckpointStore(self.configuration.parent/'checkpoints').save(payload)
            rebuilt,reloaded,_,_,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(reloaded['stem_xy_policy'],STEM_XY)
            self.assertEqual(rebuilt.stem_xy_policy,STEM_XY)
            self.assertEqual(tuple(rebuilt.network.features[0][0].weight.shape),(96,5,4,4))
            for name,value in model.state_dict().items():torch.testing.assert_close(rebuilt.state_dict()[name],value,rtol=0,atol=0)
            image=torch.zeros(1,3,512,512)
            torch.testing.assert_close(rebuilt(image),model(image),rtol=0,atol=0)
            self.assertEqual(dict(rebuilt.named_buffers()),{})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.configuration = root / 'configs/R114-SCIENCE.json'
        self.configuration.parent.mkdir()
        self.configuration.write_bytes((ROOT / 'configs/R114-SCIENCE.json').read_bytes())
        self.site = {'schema_version': 1, 'round_id': 'candidate-fixture', 'git_commit': 'b' * 40,
                     'work_root': str(root), 'round_root': str(root / 'rounds/fixture'),
                     'pod': {'namespace': 'ns', 'name': 'pod', 'container': 'container',
                             'uid': '11111111-2222-4333-8444-555555555555', 'python': sys.executable},
                     'gpus': [{'ordinal': n, 'pci': '0000:%02x:00.0' % (n + 1),
                               'vram_total_bytes': 68702699520} for n in range(8)],
                     'metadata_root': str(root / 'stage1'), 'public_weight': str(root / 'public/weight.pt'),
                     'public_provenance': str(root / 'public/provenance.json'),
                     'science': {'binding_version': 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-laststride1-v1',
                                 'candidate': copy.deepcopy(CONVNEXT22),
                                 'configuration': {'path': str(self.configuration),
                                     'sha256': hashlib.sha256(self.configuration.read_bytes()).hexdigest()}}}
        self.path = root / 'SITE_BINDING.json'
        self.path.write_text(json.dumps(self.site))
        self.environment = patch.dict(os.environ, {'NCCCU_SITE_BINDING': str(self.path)}, clear=False)
        self.environment.start(); self.addCleanup(self.environment.stop)
        os.environ.pop('NCCCU_SITE_BINDING_SHA256', None)

    def test_candidate_uses_explicit_source_identity_and_ln(self):
        self.assertEqual(experiment.candidate(), CONVNEXT22)
        self.assertEqual(experiment.candidate()['stochastic_depth'], experiment.science_configuration()['job']['stochastic_depth'])

    def test_resolved_age_preserves_frozen_contract_and_legacy_default(self):
        frozen = {'enabled': True, 'contract': {'fit_only': True}, 'contract_sha256': 'frozen'}
        with patch.object(experiment, 'metadata', return_value={'AGE.json': frozen}), \
                patch.object(experiment, 'science_configuration', return_value={'age_enabled': False}):
            disabled = experiment.resolved_age_spec()
        with patch.object(experiment, 'metadata', return_value={'AGE.json': frozen}), \
                patch.object(experiment, 'science_configuration', return_value={}):
            legacy = experiment.resolved_age_spec()
        self.assertEqual(disabled, {**frozen, 'enabled': False})
        self.assertEqual(legacy, frozen)
        self.assertIsNot(disabled['contract'], frozen['contract'])
        self.assertTrue(frozen['enabled'])

    def test_R124_content_pool_science_job_and_age_binding(self):
        from retrain.attention_pool import ATTENTION_POOL_RULE
        from retrain.config import job_contract
        old = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        new = json.loads((ROOT / 'configs/R124-SCIENCE.json').read_text())
        self.assertEqual({key for key in old['job'] if old['job'][key] != new['job'][key]},
                         {'attention_pool'})
        self.assertEqual(new['job']['attention_pool'], ATTENTION_POOL_RULE)
        self.assertEqual(new['candidate']['attention_pool'], ATTENTION_POOL_RULE)
        self.assertEqual(new['job']['loss'], old['job']['loss'])
        self.assertNotIn('age_enabled', new)
        frozen_age = {'enabled': True, 'contract': {'fit_only': True},
                      'contract_sha256': 'frozen'}
        with patch.object(experiment, 'science_configuration', return_value=new), \
                patch.object(experiment, 'metadata', return_value={'AGE.json': frozen_age}), \
                patch.object(experiment, 'binding', return_value={'fixture': 'binding'}), \
                patch.object(experiment, 'candidate', return_value=new['candidate']):
            self.assertEqual(experiment.candidate_from_configuration(new), new['candidate'])
            age = experiment.resolved_age_spec()
            self.assertEqual(age, frozen_age)
            self.assertIsNot(age, frozen_age)
            job = copy.deepcopy(new['job'])
            job.update(age=age, experiment91={'fixture': 'binding'},
                       initialization={'kind': 'imagenet', 'sha256': new['public_source']['sha256'],
                                       'format': new['public_source']['format'],
                                       'path': self.site['public_weight'],
                                       'provenance': self.site['public_provenance']})
            self.assertEqual(experiment.validate_job(job), job)
            self.assertEqual(job_contract(job)['attention_pool'], ATTENTION_POOL_RULE)

    def test_R94_only_changes_backbone_LR_scientific_fields(self):
        old = json.loads((ROOT / 'configs/R93-SCIENCE.json').read_text())
        new = json.loads((ROOT / 'configs/R94-SCIENCE.json').read_text())
        self.assertEqual({key for key in old['job'] if old['job'][key] != new['job'][key]}, {'backbone_lr'})
        self.assertEqual(new['job']['backbone_lr'], old['job']['backbone_lr'] / 2)
        for key in ('paths', 'data_split', 'schedule', 'official_manifest_sha256'):
            self.assertEqual(old[key], new[key])

    def test_configuration_controls_job_LR_and_rejects_old_LR(self):
        science = experiment.science_configuration()
        job = copy.deepcopy(science['job'])
        job.update(age={'fixture': 'unchanged-age'}, experiment91={'fixture': 'binding'},
                   initialization={'kind': 'imagenet', 'sha256': experiment.public_source()["sha256"],
                       'format': 'torchvision-1000-state-dict-v1', 'path': self.site['public_weight'],
                       'provenance': self.site['public_provenance']})
        with patch.object(experiment, 'metadata', return_value={'AGE.json': job['age']}), \
                patch.object(experiment, 'binding', return_value=job['experiment91']):
            self.assertEqual(experiment.validate_job(job)['backbone_lr'], science['job']['backbone_lr'])
            self.assertEqual(experiment.validate_job(job)['seed'], science['job']['seed'])
            self.assertEqual(experiment.validate_job(job)['stochastic_depth'], science['job']['stochastic_depth'])
            self.assertEqual(experiment.validate_job(job)['positive_per_batch'], science['job']['positive_per_batch'])
            self.assertEqual(experiment.validate_job(job).get('training_scale'), science['job'].get('training_scale'))
            self.assertEqual(experiment.validate_job(job).get('training_views'), science['job'].get('training_views'))
            self.assertEqual(experiment.validate_job(job).get('freeze_features_before'), science['job'].get('freeze_features_before'))
            self.assertEqual(experiment.validate_job(job).get('final_downsample_stride'), science['job']['final_downsample_stride'])
            wrong = copy.deepcopy(job); wrong['backbone_lr'] = 2e-4
            with self.assertRaises(ValueError):
                experiment.validate_job(wrong)

    def test_old_R93_profile_is_not_accepted_with_new_LR(self):
        bad = copy.deepcopy(self.site)
        bad['science']['binding_version'] = 'single-tiny22k-bc48a87f-fixed-9010-v1'
        self.path.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            experiment.candidate()
        self.path.write_text(json.dumps(self.site))
        value = json.loads(self.configuration.read_text())
        value['candidate_profile'] = bad['science']['binding_version']
        self.configuration.write_text(json.dumps(value))
        self.site['science']['configuration']['sha256'] = hashlib.sha256(self.configuration.read_bytes()).hexdigest()
        self.path.write_text(json.dumps(self.site))
        with self.assertRaises(ValueError):
            experiment.science_configuration()

    def test_same_length_configuration_change_invalidates_pin(self):
        original = self.configuration.read_text()
        self.configuration.write_text(original.replace('0.0001', '0.0002', 1))
        with self.assertRaises(ValueError):
            experiment.candidate()

    def test_parameter_group_LRs_follow_configured_job_and_age_contract(self):
        location = ROOT / 'tools/validation/check_candidate.py'
        spec = importlib.util.spec_from_file_location('candidate_LR_checker', location)
        checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
        job = experiment.science_configuration()['job']
        job['age'] = {'enabled': True, 'contract': {'lr': 1e-3}}
        groups = [{'role': role, 'lr': lr, 'initial_lr': lr}
                  for role, lr in [('backbone', job['backbone_lr']), ('head', job['head_lr']), ('age', 1e-3)]]
        self.assertTrue(checker.group_lrs_match(groups, job))
        for key in ('lr', 'initial_lr'):
            wrong = copy.deepcopy(groups); wrong[0][key] = 2e-4
            with self.subTest(key=key):
                self.assertFalse(checker.group_lrs_match(wrong, job))

    def test_configuration_roundtrips_loss_and_preserves_omitted_BCE_identity(self):
        from retrain.config import load_config, job_contract
        raw = {'jobs': [{'name': 'loss-fixture', 'model': 'convnext_small', 'precision': 'fp32'}]}
        configuration = self.configuration.parent / 'training.json'
        configuration.write_text(json.dumps(raw))
        legacy = load_config(configuration, environ={})['jobs'][0]
        self.assertIsNone(legacy.get('loss'))
        self.assertNotIn('loss', job_contract(legacy))
        loss = experiment.science_configuration()['job'].get('loss')
        if loss is None:
            return
        raw['jobs'][0]['loss'] = loss
        configuration.write_text(json.dumps(raw))
        current = load_config(configuration, environ={})['jobs'][0]
        self.assertEqual(current['loss'], loss)
        self.assertEqual(job_contract(current)['loss'], loss)

    def test_absent_site_has_no_legacy_candidate_fallback(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            experiment.candidate()

    def test_R114_stride_field_is_the_only_R107_learning_change_and_optional(self):
        from retrain.config import load_config, job_contract
        old = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        new = experiment.science_configuration()
        self.assertEqual({k: v for k, v in new['job'].items() if k != 'final_downsample_stride'}, old['job'])
        self.assertEqual(new['job']['final_downsample_stride'], 1)
        self.assertNotIn('freeze_features_before', new['job'])
        for key in ('public_source', 'public_asset_receipt', 'structure', 'paths', 'data_split',
                    'schedule', 'early_stopping', 'initial_model_tensor_sha256'):
            self.assertEqual(new[key], old[key])
        raw = {'jobs': [{'name': 'stride-fixture', 'model': 'convnext_small', 'precision': 'fp32'}]}
        path = self.configuration.parent / 'stride-training.json'
        path.write_text(json.dumps(raw))
        legacy = load_config(path, environ={})['jobs'][0]
        self.assertNotIn('final_downsample_stride', job_contract(legacy))
        raw['jobs'][0]['final_downsample_stride'] = 1
        path.write_text(json.dumps(raw))
        current = load_config(path, environ={})['jobs'][0]
        self.assertEqual(job_contract(current)['final_downsample_stride'], 1)

    def test_training_scale_roundtrip_preserves_legacy_job_identity(self):
        from retrain.config import load_config, job_contract
        raw = {'jobs': [{'name': 'scale-fixture', 'model': 'convnext_small', 'precision': 'fp32'}]}
        path = self.configuration.parent / 'scale-training.json'
        path.write_text(json.dumps(raw))
        legacy = load_config(path, environ={})['jobs'][0]
        self.assertNotIn('training_scale', legacy)
        self.assertNotIn('training_scale', job_contract(legacy))
        raw['jobs'][0]['training_scale'] = json.loads((ROOT / 'configs/R104-SCIENCE.json').read_text())['job']['training_scale']
        path.write_text(json.dumps(raw))
        current = load_config(path, environ={})['jobs'][0]
        self.assertEqual(current['training_scale'], [.85, 1.0])
        self.assertEqual(job_contract(current)['training_scale'], [.85, 1.0])

    def test_training_views_roundtrip_preserves_legacy_job_identity(self):
        from retrain.config import load_config, job_contract
        raw = {'jobs': [{'name': 'views-fixture', 'model': 'convnext_small', 'precision': 'fp32'}]}
        path = self.configuration.parent / 'views-training.json'
        path.write_text(json.dumps(raw))
        legacy = load_config(path, environ={})['jobs'][0]
        self.assertNotIn('training_views', legacy)
        self.assertNotIn('training_views', job_contract(legacy))
        for views in (1, 2):
            raw['jobs'][0]['training_views'] = views
            path.write_text(json.dumps(raw))
            current = load_config(path, environ={})['jobs'][0]
            self.assertEqual(current['training_views'], views)
            self.assertEqual(job_contract(current)['training_views'], views)

    def test_wrong_source_ln_or_weight_is_rejected(self):
        for key, value in [('final_norm_eps', 1e-12), ('weight_sha256', '0' * 64),
                           ('provenance_sha256', '0' * 64), ('public_revision', '0' * 40)]:
            bad = copy.deepcopy(self.site); bad['science']['candidate'][key] = value
            self.path.write_text(json.dumps(bad))
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.candidate()

    def test_actual_normalization_accepts_source_ln_and_rejects_old_final_ln(self):
        if CONVNEXT22['model'] == 'resnet50_imagenet':
            science = experiment.science_configuration()
            digest = science['initial_buffer_sha256'] or 'a' * 64
            value = {'candidate': CONVNEXT22, 'batch_norm_modules': 53, 'buffer_count': 159,
                'buffer_sha256': digest, 'head_in_features': 2048, 'head_out_features': 1,
                'dropout': science['job']['dropout'], 'layer_norms': [], 'stochastic_depth': [],
                'batch_norms': [{'name': 'backbone.bn'+str(i), 'eps': 1e-5,
                    'training': False, 'track_running_stats': True} for i in range(53)]}
            self.assertTrue(experiment.validate_normalization(value))
            bad = copy.deepcopy(value);bad['batch_norms'][0]['training'] = True
            with self.assertRaises(ValueError):experiment.validate_normalization(bad)
            return
        contract = experiment.structure_contract()
        count = contract['stochastic_depth_modules']
        maximum = experiment.science_configuration()['job']['stochastic_depth']
        value = {'candidate': CONVNEXT22, 'batch_norm_modules': 0, 'buffer_count': 0,
                 'buffer_sha256': hashlib.sha256(b'').hexdigest(),
                 'head_in_features': 768, 'head_out_features': 1, 'dropout': experiment.science_configuration()['job']['dropout'],
                 'layer_norms': [{'name': 'network.classifier.0', 'eps': 1e-6}]
                     + [{'name': 'fixture.' + str(i), 'eps': 1e-6} for i in range(contract['layer_norms'] - 1)],
                 'stochastic_depth': [{'name': str(i), 'mode': 'row', 'p': maximum * i / (count - 1)}
                                      for i in range(count)]}
        if 'final_downsample_stride' in experiment.science_configuration()['job']:
            stride = experiment.science_configuration()['job']['final_downsample_stride']
            value['final_downsample'] = {'name': 'network.features.6.1', 'stride': [stride, stride],
                'kernel_size': [2, 2], 'padding': [0, 0], 'dilation': [1, 1]}
        self.assertTrue(experiment.validate_normalization(value))
        bad = copy.deepcopy(value); bad['layer_norms'][0]['eps'] = 1e-12
        with self.assertRaises(ValueError):
            experiment.validate_normalization(bad)


if __name__ == '__main__':
    unittest.main()

"""Two CPU integration boundaries for the existing public DINOv2 factory."""
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


class DINOCurrentTest(unittest.TestCase):
    def job(self):
        from retrain import age,config
        asset=Path(os.environ['NCCCU_DINOV2_PUBLIC_ASSET']).resolve(strict=True)
        job=config.default_job('dino-public-cpu','dinov2_vits14','self_supervised')
        job.update(image_size=504,prepared_size=512,precision='fp32',normalization='imagenet',
            seed=20261029,training_scale=[.85,1.],head_lr_scope='last_linear',backbone_lr=1e-4,
            head_lr=1e-3,weight_decay=1e-4,dropout=.2,stochastic_depth=.1,
            cycles=30,min_cycles=1,patience=4,min_delta=0.,stop_after_cycles=0,
            workers=0,eval_workers=0,prefetch_factor=2,
            loss={'version':'binary-pairwise-logistic-v1','kind':'binary_pairwise_logistic',
                  'temperature':1.,'reduction':'mean','dtype':'float32','scope':'global_batch'},
            initialization={'kind':'self_supervised','format':'dinov2-official-backbone-state-dict-v1',
                'path':str(asset/'dinov2_vits14_pretrain.pth'),
                'provenance':str(asset/'PUBLIC_ASSET_PROVENANCE.json'),
                'sha256':'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'})
        job['age']=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+10*i)}
                                         for i in range(4)],'fit_indices':[0,1,2,3]})
        return job

    def test_original_public_code_strict_factory_504_and_unique_empty_Adam(self):
        import torch
        from retrain import dinov2_model,models
        from retrain.engine import _optimizer,_seed
        from retrain.ema import tensor_digest
        self.assertTrue(hasattr(dinov2_model,'public_code_root'),
                        'current public-code asset binding is missing')
        job=self.job();_seed(job['seed'],torch.device('cpu'))
        with patch.dict(os.environ,XFORMERS_DISABLED='1'):
            model=models.create_model(job,initialize=True)
        source=torch.load(job['initialization']['path'],map_location='cpu',weights_only=True)
        self.assertEqual(set(source),set(model.backbone.state_dict()))
        for name,tensor in source.items():torch.testing.assert_close(model.backbone.state_dict()[name],tensor,rtol=0,atol=0)
        self.assertEqual(dinov2_model.public_code_root(job),Path(job['initialization']['provenance']).parent/'dinov2-source')
        proof=model.public_initialization_receipt
        self.assertTrue(proof['all_public_tensors_exact']);self.assertTrue(proof['all_public_tensors_FP32_finite'])
        self.assertEqual(proof['public_tensor_count'],len(source));self.assertTrue(proof['backbone_only'])
        self.assertEqual(proof['public_tensor_sha256'],tensor_digest(source))
        self.assertEqual(proof['initialization']['kind'],'self_supervised')
        self.assertEqual(proof['initialization']['weights_sha256'],job['initialization']['sha256'])
        self.assertEqual(model.classifier[-1].in_features,768)
        generator=torch.Generator(device='cpu').manual_seed(20261027)
        torch.testing.assert_close(model.classifier[-1].weight,torch.randn((1,768),generator=generator)*.01,rtol=0,atol=0)
        self.assertEqual([n for n,p in model.named_parameters() if not p.requires_grad],['backbone.mask_token'])
        self.assertEqual(dict(model.named_buffers()),{})
        optimizer=_optimizer(model,job)
        members=[p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(members),len({id(p) for p in members}))
        self.assertEqual({id(p) for p in members},{id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({g['role']:g['initial_lr'] for g in optimizer.param_groups},
                         {'backbone':1e-4,'head':1e-3,'age':1e-3})
        reference=json.loads((ROOT/'configs/R141-SCIENCE.json').read_text())
        self.assertEqual(tensor_digest(model.state_dict()),reference['initial_model_tensor_sha256'])
        model.eval()
        with torch.inference_mode():result=model(torch.zeros(1,3,504,504),torch.zeros(1,2))
        self.assertEqual(tuple(result.shape),(1,));self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.cuda.is_initialized())
        print(json.dumps({'DINO_native_cpu':{'tensor_sha256':tensor_digest(model.state_dict()),
            'public_tensors':len(source),'trainable_tensors':len(members),'logit':result.item(),'groups':5}}),flush=True)

    def test_current_science_eight_rank_configuration_variant_load_and_geometry(self):
        import torch
        from retrain import config,experiment91 as ex,models,single91
        from retrain.ema import EMA,RULE,tensor_digest
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        self.assertTrue(hasattr(ex,'DINOV2_PAIRLOG_BINDING'),'current DINOv2 science binding is missing')
        path=ROOT/'configs/R141-SCIENCE.json';declared=json.loads(path.read_text())
        job=self.job();job.update(declared['job']);job['name']='dino-public-cpu'
        site={'schema_version':1,'scope':'cpu_technical_prep','round_id':'dino-identity-fixture',
            'git_commit':'0'*40,'work_root':str(Path(job['initialization']['path']).parents[2]),
            'round_root':str(Path(job['initialization']['path']).parents[2]/'rounds/dino-identity-fixture'),
            'pod':{'namespace':'fixture','name':'fixture','uid':'00000000-0000-0000-0000-000000000001',
                   'container':'fixture','python':sys.executable},
            'gpus':[{'ordinal':i,'pci':f'0000:{i+1:02x}:00.0','vram_total_bytes':64*1024**3} for i in range(8)],
            **{k:str(ROOT/v) for k,v in declared['paths'].items()},
            'public_weight':job['initialization']['path'],'public_provenance':job['initialization']['provenance'],
            'science':{'binding_version':declared['candidate_profile'],'candidate':declared['candidate'],
                       'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,XFORMERS_DISABLED='1'):
            root=Path(directory);p=root/'SITE.json';p.write_text(json.dumps(site))
            binding={'version':declared['candidate_profile']};original_age=copy.deepcopy(job['age'])
            with patch.dict(os.environ,NCCCU_SITE_BINDING=str(p),NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()), \
                    patch.object(ex,'metadata',return_value={'AGE.json':original_age}),patch.object(ex,'binding',return_value=binding):
                self.assertEqual(ex.candidate_from_configuration(ex.science_configuration()),declared['candidate'])
                job.update(age=ex.resolved_age_spec(),experiment91=binding)
                cfg=root/'TRAIN.json'
                cfg.write_text(json.dumps({'mode':'train','data':{'split':declared['data_split']},
                    'execution':{'devices':['cuda:'+str(i) for i in range(8)],'evaluation_physical_batch':32},'jobs':[job]}))
                resolved=config.load_config(cfg,environ={})['jobs'][0]
                self.assertEqual(ex.validate_job(resolved),resolved)
                model=models.create_model(resolved,initialize=False)
                description=ex.describe_model(model)
                self.assertEqual(description['dino']['patch_size'],14)
                self.assertEqual(len(description['layer_norms']),25)
                self.assertEqual(single91.input_geometry(),{'image_size':504,'prepared_size':512})
                old=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
                with patch.object(ex,'science_configuration',return_value=old):
                    self.assertEqual(single91.input_geometry(),{'image_size':512,'prepared_size':512})
            # A CPU-only synthetic checkpoint exercises the real variant loader;
            # the adopted formal package supplies the complete runtime identity.
            fixture=copy.deepcopy(resolved);fixture.pop('experiment91')
            ema=EMA(model,RULE)
            with torch.no_grad():ema.model.classifier[-1].bias.fill_(.25)
            payload={'format_version':1,'mode':'train','job':fixture,'age':fixture['age'],'identity':{},'manifest_digest':'a'*64,
                'model':model.state_dict(),'ema':ema.state_dict(),'successful_updates':0}
            checkpoint=CheckpointStore(root/'checkpoint').save(payload)
            load=torch.load
            def checkpoint_only(value,*args,**kwargs):
                if isinstance(value,(str,Path)) and Path(value)==Path(fixture['initialization']['path']):
                    self.fail('initialize=False must not reopen the public weights')
                return load(value,*args,**kwargs)
            with patch('torch.load',side_effect=checkpoint_only):
                for variant,expected in (('online',model),('ema',ema.model)):
                    restored,saved,_,origin,_=_load_model(checkpoint,device='cpu',variant=variant)
                    self.assertEqual(saved,fixture);self.assertEqual(origin['variant'],variant)
                    self.assertEqual(tensor_digest(restored.state_dict()),tensor_digest(expected.state_dict()))
                    self.assertEqual(restored.classifier[-1].bias.item(),0. if variant=='online' else .25)
            self.assertFalse(torch.cuda.is_initialized())

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
    def test_R153_to_R166_science_consumes_geometry_model_loss_and_fresh_identity(self):
        from retrain import age,config,single91
        original_age=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]})
        for label,order,size,seed,model in [('R153',65,768,20261027,'convnext_small'),('R154',66,512,20261029,'swin_v2_small'),
                                         ('R155',67,512,20261027,'swin_v2_small'),('R156',68,512,20261029,'convnext_small'),
                                         ('R157',69,512,20261027,'convnext_small'),('R158',70,512,20261029,'convnext_small'),
                                         ('R159',71,512,20261027,'convnext_small'),('R160',72,512,20261029,'convnext_small'),
                                         ('R161',73,512,20261027,'convnext_small'),('R162',74,512,20261029,'convnext_small'),
                                         ('R163',75,512,20261027,'convnext_small'),('R164',76,512,20261029,'deit3_small'),
                                         ('R165',77,512,20261027,'deit3_small'),('R166',78,512,20261029,'deit3_small')]:
            path=ROOT/'configs'/(label+'-SCIENCE.json');science=json.loads(path.read_text())
            self.assertEqual((science['candidate']['experiment_order'],science['job']['seed'],science['job']['model']),(order,seed,model))
            site=copy.deepcopy(self.site);site.update(work_root=str(ROOT),round_root=str(ROOT/'rounds'/(label+'-fixture')),**{k:str(ROOT/v) for k,v in science['paths'].items()})
            site['science']={'binding_version':science['candidate_profile'],'candidate':science['candidate'],'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
            self.path.write_text(json.dumps(site));binding={'version':science['candidate_profile']}
            with patch.dict(os.environ,NCCCU_SITE_BINDING_SHA256=hashlib.sha256(self.path.read_bytes()).hexdigest()),patch.object(experiment,'binding',return_value=binding),patch.object(experiment,'metadata',return_value={'AGE.json':original_age}):
                self.assertEqual(experiment.science_configuration(),science)
                self.assertEqual(experiment.candidate(),science['candidate'])
                job=copy.deepcopy(science['job']);job.update(age=experiment.resolved_age_spec(),experiment91=binding,initialization={'kind':'imagenet','format':science['public_source']['format'],'sha256':science['public_source']['sha256'],'path':site['public_weight'],'provenance':site['public_provenance']})
                cfg=self.path.parent/(label+'-TRAIN.json');cfg.write_text(json.dumps({'mode':'train','data':{'split':science['data_split']},'execution':{'devices':['cuda:'+str(i) for i in range(8)],'evaluation_physical_batch':32},'jobs':[job]}))
                resolved=config.load_config(cfg,environ={})['jobs'][0]
                self.assertEqual(experiment.validate_job(resolved),resolved)
                if label=='R166':
                    missing=copy.deepcopy(resolved);missing.pop('deit_pool_policy')
                    with self.assertRaises(ValueError):experiment.validate_job(missing)
                    wrong=copy.deepcopy(resolved);wrong['deit_pool_policy']['exclude_prefix_tokens']=0
                    with self.assertRaises(ValueError):experiment.validate_job(wrong)
                self.assertEqual(single91.input_geometry(),{'image_size':size,'prepared_size':size})
                saved=config.job_contract(resolved)
                self.assertEqual((saved['model'],saved['seed'],saved['image_size'],saved['prepared_size']),(model,seed,size,size))
                self.assertEqual(saved['loss'],science['job']['loss'])
            if label in ('R153','R155','R157','R159','R161','R163','R165'):
                baseline=json.loads((ROOT/'configs'/({'R153':'R152','R155':'R154','R157':'R156','R159':'R158','R161':'R160','R163':'R162','R165':'R164'}[label]+'-SCIENCE.json')).read_text())
                science['job']['seed']=baseline['job']['seed']
                for value in (science,baseline):
                    for key in ('candidate_profile','comparison_note','initial_model_tensor_sha256'):value.pop(key)
                    for key in ('id','experiment_order'):value['candidate'].pop(key)
                self.assertEqual(science,baseline)
            if label=='R166':
                baseline=json.loads((ROOT/'configs/R164-SCIENCE.json').read_text())
                self.assertEqual(science['job']['deit_pool_policy'],config.DEIT3_MEANPATCH_RULE)
                self.assertEqual(science['candidate']['deit_pool_policy'],config.DEIT3_MEANPATCH_RULE)
                self.assertEqual({k:v for k,v in science['job'].items()if k!='deit_pool_policy'},baseline['job'])
                self.assertEqual(science['candidate']['readout'],'native_final_LN_meanpatch_dropout_linear_plus_age')
                for key in ('public_source','public_asset_receipt','paths','structure','data_split','schedule','early_stopping','initial_model_tensor_sha256'):
                    self.assertEqual(science[key],baseline[key])
            if label=='R164':
                baseline=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
                self.assertEqual({k:v for k,v in science['job'].items()if k!='model'},
                                 {k:v for k,v in baseline['job'].items()if k!='model'})
                for key in ('data_split','schedule','early_stopping'):
                    self.assertEqual(science[key],baseline[key])
                self.assertEqual(science['job']['loss'],experiment.PAIR_LOGISTIC_LOSS)
                self.assertEqual(science['public_source']['revision'],'8d59b2810f277c19d0cdd1b1e691d93c24aab6db')
                self.assertEqual(science['public_source']['sha256'],'88140b2a114ec4db0e3588af052162b87fcdd1bb92d5e85dd93f7d2adc1fb526')
                self.assertEqual((science['candidate']['features'],science['candidate']['readout']),
                                 (384,'native_LN_CLS_dropout_linear_plus_age'))
            if label=='R156':
                self.assertEqual(science['job']['loss'],experiment.PAIR_HARD_POSITIVE_LOSS)
                baseline=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
                self.assertEqual(science['initial_model_tensor_sha256'],baseline['initial_model_tensor_sha256'])
                self.assertEqual(science['public_source'],baseline['public_source'])
                self.assertEqual(science['paths'],baseline['paths'])
                self.assertEqual({k:v for k,v in science['job'].items() if k!='loss'},
                                 {k:v for k,v in baseline['job'].items() if k!='loss'})
            if label=='R158':
                baseline=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
                self.assertEqual(science['job'],baseline['job'])
                for key in ('data_split','schedule','early_stopping','structure'):
                    self.assertEqual(science[key],baseline[key])
                self.assertEqual(science['public_source']['revision'],'14a837353b6ffeced1c635dc5c8c799356e316f4')
            if label=='R160':
                baseline=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
                self.assertEqual(science['job']['loss'],experiment.PAIR_AUC_MARGIN_LOSS)
                self.assertEqual(science['initial_model_tensor_sha256'],baseline['initial_model_tensor_sha256'])
                self.assertEqual(science['public_source'],baseline['public_source'])
                self.assertEqual({k:v for k,v in science['job'].items()if k!='loss'},
                                 {k:v for k,v in baseline['job'].items()if k!='loss'})
            if label=='R162':
                baseline=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
                self.assertEqual(science['job']['cosine_head_policy'],config.COSINE_HEAD_RULE)
                self.assertEqual(science['candidate']['cosine_head_policy'],config.COSINE_HEAD_RULE)
                self.assertEqual(science['initial_model_tensor_sha256'],baseline['initial_model_tensor_sha256'])
                self.assertEqual({k:v for k,v in science['job'].items()if k!='cosine_head_policy'},baseline['job'])
                for key in ('public_source','paths','data_split','structure','schedule','early_stopping'):
                    self.assertEqual(science[key],baseline[key])

    def test_R154_swin_v2_static_buffers_EMA_and_strict_cold_reload(self):
        import torch
        from retrain import age,models,ema,single91
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        job=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
        job.update(model='swin_v2_small',age=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]}))
        torch.manual_seed(job['seed']);model=models.create_model(job,initialize=False).eval()
        buffers={n:t.clone() for n,t in model.named_buffers()}
        self.assertEqual(len(buffers),48)
        self.assertEqual(sum(n.endswith('.relative_coords_table') for n in buffers),24)
        self.assertEqual(sum(n.endswith('.relative_position_index') for n in buffers),24)
        with patch.object(experiment,'candidate',return_value={'model':'swin_v2_small','batch_norm_buffers':0,'static_buffer_count':48}):
            self.assertEqual(single91.immutable_buffer_count(),48)
            self.assertEqual(set(single91.immutable_buffers(model.state_dict())),set(buffers))
        self.assertEqual(len(dict(model.named_parameters())),402)
        optimizer=_optimizer(model,job);self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        rule={'version':'ema-warmcopy1600-decay0999-nobn-v1','warmcopy_updates':1600,'decay':.999,
              'dtype':'torch.float32','floating_buffers':'same_as_parameters_except_static_swin',
              'integer_buffers':'copy','batch_norm_buffers':0,'static_swin_buffers':'swin_v2_small_window8',
              'static_swin_buffer_count':48}
        with patch.object(experiment,'ema_rule',return_value=rule):
            averaged=ema.EMA(model,rule)
            for update in (1600,1601,24000):
                averaged.successful=update-1
                averaged.update(model,update)
                ema.validate_state(averaged.state_dict(),model.state_dict(),update,rule)
                for name,tensor in averaged.model.named_buffers():self.assertTrue(torch.equal(tensor,buffers[name]),name)
        image=torch.zeros(1,3,512,512);features=torch.zeros(1,2)
        with torch.inference_mode():expected=model(image,features)
        self.assertEqual(tuple(expected.shape),(1,));self.assertTrue(bool(torch.isfinite(expected).all()))
        payload={'format_version':1,'mode':'train','job':job,'model':model.state_dict(),'age':job['age'],
                 'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
        with tempfile.TemporaryDirectory(dir=self.path.parent) as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved_job,_,_,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved_job['model'],'swin_v2_small')
            with torch.inference_mode():actual=cold(image,features)
        self.assertTrue(torch.equal(expected,actual));self.assertFalse(torch.cuda.is_initialized())

    def test_R151_R152_fixed_science_saved_geometry_and_derived_cache(self):
        from retrain import age,config,preprocessing,single91
        self.assertTrue(hasattr(experiment,'GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING'))
        self.assertTrue(hasattr(experiment,'GLOBAL_768_SMALL_BINDING'))
        for label,base in [('R151','R150'),('R152','R107')]:
            left=json.loads((ROOT/'configs'/f'{base}-SCIENCE.json').read_text());path=ROOT/'configs'/f'{label}-SCIENCE.json';right=json.loads(path.read_text())
            self.assertEqual(right['candidate']['experiment_order'],63 if label=='R151' else 64)
            self.assertEqual(right['official_evaluation'],experiment.OFFICIAL_GATE_0996)
            self.assertEqual(experiment.candidate_from_configuration(right),right['candidate'])
            if label=='R151':
                self.assertEqual(right['candidate_profile'],experiment.GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING)
                self.assertEqual(right['job']['seed'],20261027)
                public=json.loads((ROOT/'records/preparation/global768-20261101/R151-PUBLIC-FRESH-INITIAL.json').read_text())
                self.assertEqual(right['initial_model_tensor_sha256'],public['model_tensor_sha256'])
                self.assertEqual(public['shared_rgb_tensor_sha256'],json.loads((ROOT/'configs/R108-SCIENCE.json').read_text())['initial_model_tensor_sha256'])
            else:
                self.assertEqual(right['candidate_profile'],experiment.GLOBAL_768_SMALL_BINDING)
                self.assertEqual((right['job']['image_size'],right['job']['prepared_size']),(768,768))
                self.assertIsNone(right['job'].get('input_preprocessing'))
                self.assertEqual(right['initial_model_tensor_sha256'],left['initial_model_tensor_sha256'])
            site=copy.deepcopy(self.site);site.update(work_root=str(ROOT),round_root=str(ROOT/'rounds'/f'{label}-fixture'),**{k:str(ROOT/v) for k,v in right['paths'].items()})
            site['science']={'binding_version':right['candidate_profile'],'candidate':right['candidate'],'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
            binding={'version':right['candidate_profile']};age_spec=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]})
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'SITE.json';p.write_text(json.dumps(site))
                with patch.dict(os.environ,NCCCU_SITE_BINDING=str(p),NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()),patch.object(experiment,'binding',return_value=binding),patch.object(experiment,'metadata',return_value={'AGE.json':age_spec}):
                    self.assertEqual(experiment.science_configuration(),right)
                    job=copy.deepcopy(right['job']);job.update(age=experiment.resolved_age_spec(),experiment91=binding,initialization={'kind':'imagenet','format':right['public_source']['format'],'sha256':right['public_source']['sha256'],'path':site['public_weight'],'provenance':site['public_provenance']})
                    cfg=Path(directory)/'TRAIN.json';cfg.write_text(json.dumps({'mode':'train','data':{'split':right['data_split']},'execution':{'devices':['cuda:'+str(i) for i in range(8)],'evaluation_physical_batch':32},'jobs':[job]}))
                    resolved=config.load_config(cfg,environ={})['jobs'][0];self.assertEqual(experiment.validate_job(resolved),resolved)
                    self.assertEqual({k:config.job_contract(resolved)[k] for k in ('image_size','prepared_size')},single91.input_geometry())
            for value in (left,right):
                for key in ('candidate_profile','comparison_note','official_evaluation','initial_model_tensor_sha256'):value.pop(key,None)
                for key in ('id','experiment_order'):value['candidate'].pop(key)
            if label=='R151':right['job']['seed']=left['job']['seed'];right['composite_cache_receipt']=left['composite_cache_receipt']
            else:
                right['job'].update(image_size=512,prepared_size=512);right['paths']['prepared_cache']=left['paths']['prepared_cache'];right.pop('prepared_cache_receipt')
            self.assertEqual(left,right)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'metadata').mkdir();(root/'metadata/manifest.json').write_text('{}\n');index=root/'INDEX.jsonl';index.write_text('{}\n')
            site={'work_root':str(root),'data_root':str(root/'data'),'metadata_root':str(root/'metadata'),'prepared_cache':str(root/'cache'),'pod':{'uid':'fixture'}}
            value={'format':'global-prepared-cache-v1','passed':True,'pod_uid':'fixture','cache_root':site['prepared_cache'],'data_root':site['data_root'],'metadata_root':site['metadata_root'],
                'manifest_sha256':hashlib.sha256((root/'metadata/manifest.json').read_bytes()).hexdigest(),'fit_count':46256,'dev_count':5140,'row_count':51396,'verified_readonly_count':51396,
                'input_preprocessing':None,'prepared_size':768,'pillow':'12.3.0','official_read':False,'source_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('runtime/bundle/retrain/data.py','runtime/bundle/retrain/preprocessing.py')},'index':{'path':str(index),'sha256':hashlib.sha256(index.read_bytes()).hexdigest()}}
            p=root/'CACHE.json';p.write_text(json.dumps(value));science={'job':{'prepared_size':768},'prepared_cache_receipt':{'path':'CACHE.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
            self.assertEqual(preprocessing.verify_native_cache_asset(science,site),value)

    def test_R152_native768_factory_strict_reload_and_inference_geometry(self):
        import torch
        from retrain import age,models
        from retrain.engine import _optimizer
        from retrain.inference import _runtime
        job=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
        job['age']=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]})
        torch.manual_seed(job['seed']);baseline=models.create_model(job,initialize=False);rng=torch.get_rng_state().clone()
        job.update(image_size=768,prepared_size=768)
        torch.manual_seed(job['seed']);model=models.create_model(job,initialize=False)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(len(dict(model.named_parameters())),345)
        for name,tensor in baseline.state_dict().items():self.assertTrue(torch.equal(tensor,model.state_dict()[name]),name)
        optimizer=_optimizer(model,job);self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        model.eval();shapes=[];hook=model.network.features.register_forward_hook(lambda _,args,result:shapes.append(tuple(result.shape)))
        image=torch.zeros(1,3,768,768);features=torch.zeros(1,2)
        try:
            with torch.no_grad():output=model(image,features)
        finally:hook.remove()
        self.assertEqual(shapes,[(1,768,24,24)]);self.assertTrue(bool(torch.isfinite(output).all()))
        cold=models.create_model(job,initialize=False);loaded=cold.load_state_dict(model.state_dict(),strict=True)
        self.assertFalse(loaded.missing_keys);self.assertFalse(loaded.unexpected_keys);cold.eval()
        with torch.no_grad():torch.testing.assert_close(output,cold(image,features),rtol=0,atol=0)
        runtime=_runtime(job,{});self.assertEqual((runtime['image_size'],runtime['prepared_size']),(768,768));self.assertFalse(torch.cuda.is_initialized())

    def test_R150_reducer_priming_consumes_six_channels_and_restores_state(self):
        import random
        from types import SimpleNamespace
        import numpy as np
        import torch
        from retrain import config,models
        from retrain.distributed_engine import prime_reducer
        for channels,policy in ((3,None),(6,config.GLOBAL_NATIVE_SIX_RULE)):
            with self.subTest(channels=channels):
                job=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
                job.update(age=None,physical_batch=1,image_size=32)
                if policy is not None:job['input_preprocessing']=policy
                model=models.create_model(job,initialize=False);model.eval()
                before={name:value.detach().clone() for name,value in model.state_dict().items()}
                rng=torch.get_rng_state().clone();python_rng=random.getstate();numpy_rng=np.random.get_state()
                seen=[]
                handle=model.register_forward_pre_hook(lambda _,args:seen.append(tuple(args[0].shape)))
                try:prime_reducer(model,model,job,SimpleNamespace(world_size=8,rank=0,device=torch.device('cpu')))
                finally:handle.remove()
                self.assertEqual(seen,[(1,channels,32,32)]*2)
                self.assertFalse(model.training)
                self.assertTrue(all(torch.equal(value,before[name]) for name,value in model.state_dict().items()))
                self.assertTrue(all(value.grad is None for value in model.parameters()))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(python_rng,random.getstate())
                current=np.random.get_state();self.assertEqual(numpy_rng[0],current[0]);np.testing.assert_array_equal(numpy_rng[1],current[1]);self.assertEqual(numpy_rng[2:],current[2:])

    def test_R149_R150_fixed_science_saved_job_and_derived_asset(self):
        from retrain import age,config,preprocessing
        self.assertTrue(hasattr(experiment,'NO_MODEL_STOCH_SMALL_SEED27_BINDING'),'conditional no-model-stochasticity binding is missing')
        self.assertTrue(hasattr(experiment,'GLOBAL_NATIVE_SIX_SMALL_BINDING'),'six-channel binding is missing')
        for label,base in [('R149','R148'),('R150','R107')]:
            left=json.loads((ROOT/'configs'/f'{base}-SCIENCE.json').read_text());path=ROOT/'configs'/f'{label}-SCIENCE.json';right=json.loads(path.read_text())
            self.assertEqual(right['candidate']['experiment_order'],61 if label=='R149' else 62)
            self.assertEqual(right['official_evaluation'],experiment.OFFICIAL_GATE_0996)
            self.assertEqual(experiment.candidate_from_configuration(right),right['candidate'])
            if label=='R149':
                self.assertEqual(right['candidate_profile'],experiment.NO_MODEL_STOCH_SMALL_SEED27_BINDING)
                self.assertEqual(right['job']['seed'],20261027)
                self.assertEqual(right['initial_model_tensor_sha256'],json.loads((ROOT/'configs/R108-SCIENCE.json').read_text())['initial_model_tensor_sha256'])
            else:
                self.assertEqual(right['candidate_profile'],experiment.GLOBAL_NATIVE_SIX_SMALL_BINDING)
                self.assertEqual(right['job']['input_preprocessing'],config.GLOBAL_NATIVE_SIX_RULE)
                public=json.loads((ROOT/'records/preparation/native6-20261101/PUBLIC-FRESH-INITIAL.json').read_text())
                self.assertEqual(right['initial_model_tensor_sha256'],public['model_tensor_sha256'])
                self.assertEqual(public['shared_rgb_tensor_sha256'],left['initial_model_tensor_sha256'])
                self.assertEqual((public['trainable_tensors'],public['added_zero_elements']),(345,4608))
            site=copy.deepcopy(self.site);site.update(work_root=str(ROOT),round_root=str(ROOT/'rounds'/f'{label}-fixture'),**{k:str(ROOT/v) for k,v in right['paths'].items()})
            site['science']={'binding_version':right['candidate_profile'],'candidate':right['candidate'],'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
            binding={'version':right['candidate_profile']};age_spec=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]})
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'SITE.json';p.write_text(json.dumps(site))
                with patch.dict(os.environ,NCCCU_SITE_BINDING=str(p),NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()),patch.object(experiment,'binding',return_value=binding),patch.object(experiment,'metadata',return_value={'AGE.json':age_spec}):
                    self.assertEqual(experiment.science_configuration(),right)
                    job=copy.deepcopy(right['job']);job.update(age=experiment.resolved_age_spec(),experiment91=binding,initialization={'kind':'imagenet','format':right['public_source']['format'],'sha256':right['public_source']['sha256'],'path':site['public_weight'],'provenance':site['public_provenance']})
                    cfg=Path(directory)/'TRAIN.json';cfg.write_text(json.dumps({'mode':'train','data':{'split':right['data_split']},'execution':{'devices':['cuda:'+str(i) for i in range(8)],'evaluation_physical_batch':32},'jobs':[job]}))
                    resolved=config.load_config(cfg,environ={})['jobs'][0];self.assertEqual(experiment.validate_job(resolved),resolved)
                    self.assertEqual(config.job_contract(resolved).get('input_preprocessing'),right['job'].get('input_preprocessing'))
            for value in (left,right):
                for key in ('candidate_profile','comparison_note','official_evaluation','initial_model_tensor_sha256'):value.pop(key,None)
                for key in ('id','experiment_order'):value['candidate'].pop(key)
            if label=='R149':right['job']['seed']=left['job']['seed']
            else:
                right['candidate'].pop('input_preprocessing');right['job'].pop('input_preprocessing');right.pop('composite_cache_receipt');right['paths']['prepared_cache']=left['paths']['prepared_cache']
            self.assertEqual(right,left)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'metadata').mkdir();(root/'metadata/manifest.json').write_text('{}\n');index=root/'INDEX.jsonl';index.write_text('{}\n')
            site={'work_root':str(root),'data_root':str(root/'data'),'metadata_root':str(root/'metadata'),'prepared_cache':str(root/'native'),'pod':{'uid':'fixture'}}
            base={'format':'global-native-inset-cache-pair-v1','passed':True,'pod_uid':'fixture','data_root':site['data_root'],'metadata_root':site['metadata_root'],'native_cache_root':site['prepared_cache'],'global_cache_root':str(root/'global'),
                'manifest_sha256':hashlib.sha256((root/'metadata/manifest.json').read_bytes()).hexdigest(),'fit_count':46256,'dev_count':5140,'row_count':51396,'verified_readonly_count_each':51396,'input_preprocessing':config.GLOBAL_NATIVE_INSET_RULE,'pillow':'12.3.0','official_read':False,'index':{'path':str(index),'sha256':hashlib.sha256(index.read_bytes()).hexdigest()}}
            p=root/'BASE.json';p.write_text(json.dumps(base));value=copy.deepcopy(base)
            value.update(format='global-native-six-cache-pair-v1',input_preprocessing=config.GLOBAL_NATIVE_SIX_RULE,base_pair_receipt={'path':'BASE.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()},cache_core_AST_unchanged=True,
                source_files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('runtime/bundle/retrain/data.py','runtime/bundle/retrain/preprocessing.py')})
            p=root/'SIX.json';p.write_text(json.dumps(value));spec={'job':{'input_preprocessing':config.GLOBAL_NATIVE_SIX_RULE},'composite_cache_receipt':{'path':'SIX.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
            self.assertEqual(preprocessing.verify_composite_cache_asset(spec,site)['global_cache_root'],str(root/'global'))

    def test_R150_existing_factory_rgb_preservation_rng_gradient_and_strict_reload(self):
        from retrain import config
        self.assertTrue(hasattr(config,'GLOBAL_NATIVE_SIX_RULE'),'six-channel policy is missing')
        import torch
        from retrain import age,models
        from retrain.engine import _optimizer
        job=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
        job['age']=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]})
        torch.manual_seed(job['seed']);baseline=models.create_model(job,initialize=False);rng=torch.get_rng_state().clone()
        job['input_preprocessing']=config.GLOBAL_NATIVE_SIX_RULE
        torch.manual_seed(job['seed']);model=models.create_model(job,initialize=False)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        old=baseline.state_dict();current=model.state_dict();self.assertEqual(old.keys(),current.keys());self.assertEqual(len(dict(model.named_parameters())),345)
        for name,tensor in old.items():self.assertTrue(torch.equal(tensor,current[name][:,:3] if name=='network.features.0.0.weight' else current[name]),name)
        stem=model.network.features[0][0];self.assertEqual(tuple(stem.weight.shape),(96,6,4,4));self.assertFalse(bool(torch.count_nonzero(stem.weight[:,3:])))
        self.assertEqual(sum(p.numel() for p in model.parameters())-sum(p.numel() for p in baseline.parameters()),4608)
        images=torch.linspace(-1,1,2*3*32*32).reshape(2,3,32,32);paired=torch.cat((images,images.flip(-1)),1);features=torch.tensor([[.25,1.],[-.25,1.]])
        baseline.eval();model.eval()
        with torch.no_grad():torch.testing.assert_close(model(paired,features),baseline(images,features),rtol=2e-5,atol=2e-6)
        optimizer=_optimizer(model,job);self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        model.train();model(paired,features).sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertTrue(bool(torch.count_nonzero(stem.weight.grad[:,3:])));optimizer.step()
        self.assertTrue(bool(torch.count_nonzero(stem.weight[:,3:])))
        self.assertEqual(tuple(optimizer.state[stem.weight]['exp_avg'].shape),(96,6,4,4))
        cold=models.create_model(job,initialize=False);loaded=cold.load_state_dict(model.state_dict(),strict=True)
        self.assertFalse(loaded.missing_keys);self.assertFalse(loaded.unexpected_keys)
        self.assertTrue(all(torch.equal(tensor,cold.state_dict()[name]) for name,tensor in model.state_dict().items()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_R147_R148_fixed_science_and_saved_job_contract(self):
        from retrain import age,config
        self.assertTrue(hasattr(experiment,'GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING'),'conditional inset binding is missing')
        self.assertTrue(hasattr(experiment,'NO_MODEL_STOCH_SMALL_BINDING'),'fixed zero stochastic regularization binding is missing')
        for label,base in [('R147','R146'),('R148','R107')]:
            left=json.loads((ROOT/'configs'/f'{base}-SCIENCE.json').read_text())
            path=ROOT/'configs'/f'{label}-SCIENCE.json';right=json.loads(path.read_text())
            self.assertEqual(right['candidate']['experiment_order'],59 if label=='R147' else 60)
            self.assertEqual(right['official_evaluation'],experiment.OFFICIAL_GATE_0996)
            self.assertEqual(experiment.candidate_from_configuration(right),right['candidate'])
            if label=='R147':
                self.assertEqual(right['candidate_profile'],experiment.GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING)
                self.assertEqual(right['job']['seed'],20261027)
                self.assertEqual(right['composite_cache_receipt'],left['composite_cache_receipt'])
                self.assertEqual(right['initial_model_tensor_sha256'],json.loads((ROOT/'configs/R108-SCIENCE.json').read_text())['initial_model_tensor_sha256'])
            else:
                self.assertEqual(right['candidate_profile'],experiment.NO_MODEL_STOCH_SMALL_BINDING)
                self.assertEqual((right['job']['dropout'],right['job']['stochastic_depth'],right['candidate']['head_dropout']),(0.,0.,0.))
                self.assertEqual(right['initial_model_tensor_sha256'],left['initial_model_tensor_sha256'])
            site=copy.deepcopy(self.site)
            site.update(work_root=str(ROOT),round_root=str(ROOT/'rounds'/f'{label}-fixture'),**{k:str(ROOT/v) for k,v in right['paths'].items()})
            site['science']={'binding_version':right['candidate_profile'],'candidate':right['candidate'],'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
            binding={'version':right['candidate_profile']};age_spec=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]})
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'SITE.json';p.write_text(json.dumps(site))
                with patch.dict(os.environ,NCCCU_SITE_BINDING=str(p),NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()),patch.object(experiment,'binding',return_value=binding),patch.object(experiment,'metadata',return_value={'AGE.json':age_spec}):
                    self.assertEqual(experiment.science_configuration(),right)
                    job=copy.deepcopy(right['job']);job.update(age=experiment.resolved_age_spec(),experiment91=binding,initialization={'kind':'imagenet','format':right['public_source']['format'],'sha256':right['public_source']['sha256'],'path':site['public_weight'],'provenance':site['public_provenance']})
                    cfg=Path(directory)/'TRAIN.json';cfg.write_text(json.dumps({'mode':'train','data':{'split':right['data_split']},'execution':{'devices':['cuda:'+str(i) for i in range(8)],'evaluation_physical_batch':32},'jobs':[job]}))
                    resolved=config.load_config(cfg,environ={})['jobs'][0]
                    self.assertEqual(experiment.validate_job(resolved),resolved)
                    saved=config.job_contract(resolved)
                    self.assertEqual((saved['dropout'],saved['stochastic_depth']),(right['job']['dropout'],right['job']['stochastic_depth']))
            for value in (left,right):
                for key in ('candidate_profile','comparison_note','official_evaluation'):value.pop(key,None)
                for key in ('id','experiment_order'):value['candidate'].pop(key)
            if label=='R147':right['job']['seed']=left['job']['seed'];right['initial_model_tensor_sha256']=left['initial_model_tensor_sha256']
            else:
                right['job']['dropout']=left['job']['dropout'];right['job']['stochastic_depth']=left['job']['stochastic_depth']
                right['candidate']['stochastic_depth']=left['candidate']['stochastic_depth'];right['candidate'].pop('head_dropout')
            self.assertEqual(right,left)

    def test_R148_existing_factory_zero_stochasticity_and_finite_gradient(self):
        self.assertTrue(hasattr(experiment,'NO_MODEL_STOCH_SMALL_BINDING'),'fixed zero stochastic regularization binding is missing')
        import torch
        from torchvision.ops import StochasticDepth
        from retrain import age,models
        spec=json.loads((ROOT/'configs/R148-SCIENCE.json').read_text())
        job=copy.deepcopy(spec['job']);job.pop('experiment91',None)
        job['age']=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]})
        torch.manual_seed(job['seed']);model=models.create_model(job,initialize=False)
        depth=[m for m in model.modules() if isinstance(m,StochasticDepth)]
        dropout=[m for m in model.modules() if isinstance(m,torch.nn.Dropout)]
        self.assertEqual(len(depth),36);self.assertTrue(all(m.p==0 for m in depth))
        self.assertEqual(len(dropout),1);self.assertEqual(dropout[0].p,0.)
        self.assertEqual(sum(p.requires_grad for p in model.parameters()),345)
        images=torch.linspace(-1,1,2*3*32*32).reshape(2,3,32,32)
        features=torch.tensor([[.25,1.],[-.25,1.]])
        rng=torch.get_rng_state().clone();model.train()
        with torch.no_grad():training=model(images,features)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));model.eval()
        with torch.no_grad():evaluation=model(images,features)
        self.assertTrue(torch.equal(training,evaluation));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        model.train();model(images,features).sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(torch.cuda.is_initialized())

    def test_R146_global_native_inset_identity_and_asset_binding(self):
        from retrain import config,preprocessing
        self.assertTrue(hasattr(experiment,'GLOBAL_NATIVE_INSET_SMALL_BINDING'),'global/native inset binding is missing')
        self.assertTrue(hasattr(preprocessing,'verify_composite_cache_asset'),'paired cache verifier is missing')
        spec=copy.deepcopy(json.loads((ROOT/'configs/R144-SCIENCE.json').read_text()))
        spec['candidate_profile']=experiment.GLOBAL_NATIVE_INSET_SMALL_BINDING
        spec['candidate'].update(id='r146-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-inset256-9010',experiment_order=58,input_preprocessing=config.GLOBAL_NATIVE_INSET_RULE)
        spec['job']['input_preprocessing']=config.GLOBAL_NATIVE_INSET_RULE
        spec.pop('prepared_cache_receipt');spec['composite_cache_receipt']={'path':'fixture/HANDOFF.json','sha256':'a'*64}
        self.assertEqual(experiment.candidate_from_configuration(spec),spec['candidate'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'metadata').mkdir();(root/'metadata/manifest.json').write_text('{}\n');index=root/'INDEX.jsonl';index.write_text('{"fixture":true}\n')
            site={'work_root':str(root),'data_root':str(root/'data'),'metadata_root':str(root/'metadata'),'prepared_cache':str(root/'native'),'pod':{'uid':'fixture'}}
            handoff={'format':'global-native-inset-cache-pair-v1','passed':True,'pod_uid':'fixture','data_root':site['data_root'],'metadata_root':site['metadata_root'],'native_cache_root':site['prepared_cache'],'global_cache_root':str(root/'global'),
                'manifest_sha256':hashlib.sha256((root/'metadata/manifest.json').read_bytes()).hexdigest(),'fit_count':46256,'dev_count':5140,'row_count':51396,'verified_readonly_count_each':51396,'input_preprocessing':config.GLOBAL_NATIVE_INSET_RULE,'pillow':'12.3.0','official_read':False,
                'index':{'path':str(index),'sha256':hashlib.sha256(index.read_bytes()).hexdigest()}}
            path=root/'fixture/HANDOFF.json';path.parent.mkdir();path.write_text(json.dumps(handoff));spec['composite_cache_receipt']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(preprocessing.verify_composite_cache_asset(spec,site)['global_cache_root'],str(root/'global'))
            handoff['verified_readonly_count_each']=51395;path.write_text(json.dumps(handoff));spec['composite_cache_receipt']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):preprocessing.verify_composite_cache_asset(spec,site)

    def test_R145_native_seed27_identity_and_unchanged_cache_contract(self):
        from retrain import age,config
        self.assertTrue(hasattr(experiment,'NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING'),'conditional native seed27 binding is missing')
        left=json.loads((ROOT/'configs/R144-SCIENCE.json').read_text())
        path=ROOT/'configs/R145-SCIENCE.json';right=json.loads(path.read_text())
        self.assertEqual(right['candidate_profile'],experiment.NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING)
        self.assertEqual(right['candidate']['experiment_order'],57);self.assertEqual(right['job']['seed'],20261027)
        self.assertEqual(right['initial_model_tensor_sha256'],json.loads((ROOT/'configs/R108-SCIENCE.json').read_text())['initial_model_tensor_sha256'])
        self.assertEqual(right['prepared_cache_receipt'],left['prepared_cache_receipt'])
        site=copy.deepcopy(self.site)
        site.update(work_root=str(ROOT),round_root=str(ROOT/'rounds/native-seed27-fixture'),**{k:str(ROOT/v) for k,v in right['paths'].items()})
        site['science']={'binding_version':right['candidate_profile'],'candidate':right['candidate'],
            'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
        binding={'version':right['candidate_profile']};age_spec=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],'fit_indices':[0,1,2,3]})
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'SITE.json';p.write_text(json.dumps(site))
            with patch.dict(os.environ,NCCCU_SITE_BINDING=str(p),NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()),patch.object(experiment,'binding',return_value=binding),patch.object(experiment,'metadata',return_value={'AGE.json':age_spec}):
                self.assertEqual(experiment.candidate_from_configuration(experiment.science_configuration()),right['candidate'])
                job=copy.deepcopy(right['job']);job.update(age=experiment.resolved_age_spec(),experiment91=binding,initialization={'kind':'imagenet','format':right['public_source']['format'],'sha256':right['public_source']['sha256'],'path':site['public_weight'],'provenance':site['public_provenance']})
                cfg=Path(directory)/'TRAIN.json';cfg.write_text(json.dumps({'mode':'train','data':{'split':right['data_split']},'execution':{'devices':['cuda:'+str(i) for i in range(8)],'evaluation_physical_batch':32},'jobs':[job]}))
                resolved=config.load_config(cfg,environ={})['jobs'][0]
                self.assertEqual(experiment.validate_job(resolved),resolved)
                self.assertEqual(config.job_contract(resolved)['input_preprocessing'],config.NATIVE_BRIGHT_CENTER_RULE)
        for value in (left,right):
            for key in ('candidate_profile','comparison_note','initial_model_tensor_sha256'):value.pop(key,None)
            for key in ('id','experiment_order'):value['candidate'].pop(key)
        right['job']['seed']=left['job']['seed'];self.assertEqual(right,left)

    def test_R144_native_science_and_explicit_derived_cache_binding(self):
        from retrain import config,preprocessing
        self.assertTrue(hasattr(experiment,'NATIVE_BRIGHT_CENTER_SMALL_BINDING'),'native bright-center binding is missing')
        self.assertTrue(hasattr(preprocessing,'verify_native_cache_asset'),'native cache asset verifier is missing')
        spec=copy.deepcopy(json.loads((ROOT/'configs/R143-SCIENCE.json').read_text()))
        spec['candidate_profile']=experiment.NATIVE_BRIGHT_CENTER_SMALL_BINDING
        spec['candidate'].update(id='r144-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-native-brightcenter-half-9010',experiment_order=56,input_preprocessing=config.NATIVE_BRIGHT_CENTER_RULE)
        spec['job']['input_preprocessing']=config.NATIVE_BRIGHT_CENTER_RULE
        self.assertEqual(experiment.candidate_from_configuration(spec),spec['candidate'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'metadata').mkdir()
            manifest=root/'metadata/manifest.json';manifest.write_text('{}\n')
            index=root/'CACHE-INDEX.jsonl';index.write_text('{"fixture":true}\n')
            site={'work_root':str(root),'data_root':str(root/'data'),'metadata_root':str(root/'metadata'),
                  'prepared_cache':str(root/'cache'),'pod':{'uid':'fixture-uid'}}
            impl=('runtime/bundle/retrain/data.py','runtime/bundle/retrain/preprocessing.py')
            receipt={'format':'native-bright-prepared-cache-v1','passed':True,'pod_uid':'fixture-uid',
                'cache_root':site['prepared_cache'],'data_root':site['data_root'],'metadata_root':site['metadata_root'],
                'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'fit_count':46256,'dev_count':5140,'row_count':51396,'verified_readonly_count':51396,
                'input_preprocessing':config.NATIVE_BRIGHT_CENTER_RULE,'prepared_size':512,'pillow':'12.3.0',
                'official_read':False,'index':{'path':str(index),'sha256':hashlib.sha256(index.read_bytes()).hexdigest()},
                'source_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in impl}}
            path=root/'HANDOFF.json';path.write_text(json.dumps(receipt))
            spec['prepared_cache_receipt']={'path':'HANDOFF.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            verified=preprocessing.verify_native_cache_asset(spec,site)
            self.assertEqual(verified['row_count'],51396)
            declared_path=root/'SCIENCE.json';declared_path.write_text(json.dumps(spec))
            declared_site={**site,'science':{'configuration':{'path':str(declared_path),'sha256':hashlib.sha256(declared_path.read_bytes()).hexdigest()}}}
            with patch.object(experiment,'load_site',return_value=declared_site):
                self.assertEqual(experiment.science_configuration()['job']['input_preprocessing'],config.NATIVE_BRIGHT_CENTER_RULE)
            for field,value in [('verified_readonly_count',51395),('official_read',True),('cache_root',str(root/'other'))]:
                changed=copy.deepcopy(receipt);changed[field]=value;path.write_text(json.dumps(changed))
                spec['prepared_cache_receipt']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(field=field),self.assertRaises(ValueError):preprocessing.verify_native_cache_asset(spec,site)

    def test_R142_conditional_DINO27_and_R143_fixed_bright_crop_science(self):
        from retrain import age,config
        self.assertTrue(hasattr(experiment,'DINOV2_PAIRLOG_SEED27_BINDING'),'conditional DINO seed27 binding is missing')
        self.assertTrue(hasattr(experiment,'BRIGHT_CENTER_SMALL_BINDING'),'fixed bright-center binding is missing')
        original_age=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+10*i)} for i in range(4)],'fit_indices':[0,1,2,3]})
        for filename,baseline,order in (('R142-SCIENCE.json','R141-SCIENCE.json',54),('R143-SCIENCE.json','R107-SCIENCE.json',55)):
            path=ROOT/'configs'/filename;declared=json.loads(path.read_text())
            site={'schema_version':1,'scope':'cpu_technical_prep','round_id':'next-identity-fixture','git_commit':'0'*40,
                'work_root':str(ROOT),'round_root':str(ROOT/'rounds/next-identity-fixture'),
                'pod':{'namespace':'fixture','name':'fixture','uid':'00000000-0000-0000-0000-000000000001','container':'fixture','python':sys.executable},
                'gpus':[{'ordinal':i,'pci':f'0000:{i+1:02x}:00.0','vram_total_bytes':64*1024**3} for i in range(8)],
                **{k:str(ROOT/v) for k,v in declared['paths'].items()},
                'science':{'binding_version':declared['candidate_profile'],'candidate':declared['candidate'],
                    'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'SITE.json';p.write_text(json.dumps(site));binding={'version':declared['candidate_profile']}
                with patch.dict(os.environ,NCCCU_SITE_BINDING=str(p),NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()), \
                        patch.object(experiment,'metadata',return_value={'AGE.json':original_age}),patch.object(experiment,'binding',return_value=binding):
                    self.assertEqual(experiment.candidate_from_configuration(experiment.science_configuration()),declared['candidate'])
                    job=copy.deepcopy(declared['job']);job.update(age=experiment.resolved_age_spec(),experiment91=binding,
                        initialization={'kind':declared['public_source'].get('kind','imagenet'),
                            'sha256':declared['public_source']['sha256'],'format':declared['public_source']['format'],
                            'path':site['public_weight'],'provenance':site['public_provenance']})
                    cfg=Path(directory)/'TRAIN.json';cfg.write_text(json.dumps({'mode':'train','data':{'split':declared['data_split']},
                        'execution':{'devices':['cuda:'+str(i) for i in range(8)],'evaluation_physical_batch':32},'jobs':[job]}))
                    resolved=config.load_config(cfg,environ={})['jobs'][0]
                    self.assertEqual(experiment.validate_job(resolved),resolved)
                    self.assertEqual(config.job_contract(resolved).get('input_preprocessing'),declared['job'].get('input_preprocessing'))
            left=json.loads((ROOT/'configs'/baseline).read_text());right=copy.deepcopy(declared)
            for value in (left,right):
                for key in ('candidate_profile','comparison_note','official_evaluation'):value.pop(key,None)
                for key in ('id','experiment_order'):value['candidate'].pop(key)
            if order==54:
                self.assertEqual(right['job']['seed'],20261027);right['job']['seed']=left['job']['seed']
                self.assertEqual(declared['job']['head_seed'],20261027)
                self.assertEqual(declared['initial_model_tensor_sha256'],left['initial_model_tensor_sha256'])
            else:
                self.assertEqual(right['candidate'].pop('input_preprocessing'),config.BRIGHT_CENTER_RULE)
                self.assertEqual(right['job'].pop('input_preprocessing'),config.BRIGHT_CENTER_RULE)
            self.assertEqual(left,right)

    def test_R140_conditional_seed27_identity_and_exact_R139_recipe(self):
        self.assertTrue(hasattr(experiment, 'LOCAL_CONTRAST_SMALL_SEED27_BINDING'),
                        'conditional local contrast seed27 binding is missing')
        left=json.loads((ROOT/'configs/R139-SCIENCE.json').read_text())
        path=ROOT/'configs/R140-SCIENCE.json';right=json.loads(path.read_text())
        self.assertEqual(right['candidate_profile'],experiment.LOCAL_CONTRAST_SMALL_SEED27_BINDING)
        self.assertEqual(right['candidate']['experiment_order'],52)
        self.assertEqual(right['job']['seed'],20261027)
        self.assertEqual(right['initial_model_tensor_sha256'],
                         json.loads((ROOT/'configs/R108-SCIENCE.json').read_text())['initial_model_tensor_sha256'])
        site=copy.deepcopy(self.site)
        site.update(work_root=str(ROOT),round_root=str(ROOT/'rounds/local-contrast-seed27-fixture'),
                    **{k:str(ROOT/v) for k,v in right['paths'].items()})
        site['science']={'binding_version':right['candidate_profile'],'candidate':right['candidate'],
                        'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
        self.path.write_text(json.dumps(site))
        self.assertEqual(experiment.candidate_from_configuration(experiment.science_configuration()),right['candidate'])
        self.assertEqual(experiment.candidate(),right['candidate'])
        for value in (left,right):
            for key in ('candidate_profile','comparison_note','initial_model_tensor_sha256'):value.pop(key)
            for key in ('id','experiment_order'):value['candidate'].pop(key)
            value['job'].pop('seed')
        self.assertEqual(left,right)

    def test_R138_Lion27_and_R139_local_contrast_fixed_science(self):
        from retrain import age, config
        from retrain.persistence import canonical_digest
        self.assertTrue(hasattr(experiment, 'LOCAL_CONTRAST_SMALL_BINDING'), 'local contrast science binding is missing')
        self.assertTrue(hasattr(experiment, 'LION_SMALL_SEED27_BINDING'), 'conditional Lion seed27 binding is missing')
        original_age = age.make_spec({'rows': [{'name': str(i), 'sha256': str(i) * 64, 'age': str(30 + 10 * i)}
                                              for i in range(4)], 'fit_indices': [0, 1, 2, 3]})
        for filename, baseline, order in (('R138-SCIENCE.json', 'R137-SCIENCE.json', 50),
                                          ('R139-SCIENCE.json', 'R107-SCIENCE.json', 51)):
            path = ROOT / 'configs' / filename
            declared = json.loads(path.read_text())
            site = {'schema_version': 1, 'scope': 'cpu_technical_prep', 'round_id': 'identity-fixture', 'git_commit': '0' * 40,
                'work_root': str(ROOT), 'round_root': str(ROOT / 'rounds/identity-fixture'),
                'pod': {'namespace': 'fixture', 'name': 'fixture', 'uid': '00000000-0000-0000-0000-000000000001',
                        'container': 'fixture', 'python': sys.executable},
                'gpus': [{'ordinal': i, 'pci': f'0000:{i+1:02x}:00.0', 'vram_total_bytes': 64 * 1024 ** 3} for i in range(8)],
                **{key: str(ROOT / value) for key, value in declared['paths'].items()},
                'science': {'binding_version': declared['candidate_profile'], 'candidate': declared['candidate'],
                    'configuration': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
            with tempfile.TemporaryDirectory() as folder:
                p = Path(folder) / 'SITE_BINDING.json'; p.write_text(json.dumps(site))
                with patch.dict(os.environ, NCCCU_SITE_BINDING=str(p), NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()), \
                        patch.object(experiment, 'metadata', return_value={'AGE.json': original_age}):
                    actual = experiment.candidate_from_configuration(experiment.science_configuration())
                    self.assertEqual(actual, declared['candidate']); self.assertEqual(actual['experiment_order'], order)
                    resolved_age = experiment.resolved_age_spec(); age.validate_spec(resolved_age)
                    binding = {'version': declared['candidate_profile']}
                    job = copy.deepcopy(declared['job'])
                    job.update(age=resolved_age, experiment91=binding, physical_batch=8,
                        initialization={'kind': 'imagenet', 'sha256': declared['public_source']['sha256'],
                            'format': declared['public_source']['format'], 'path': site['public_weight'],
                            'provenance': site['public_provenance']})
                    with patch.object(experiment, 'binding', return_value=binding):
                        self.assertEqual(experiment.validate_job(json.loads(json.dumps(job))), job)
                    self.assertEqual(config.job_contract(job), json.loads(json.dumps(config.job_contract(job))))
            left, right = json.loads((ROOT / 'configs' / baseline).read_text()), copy.deepcopy(declared)
            for value in (left, right):
                for key in ('candidate_profile', 'comparison_note', 'official_evaluation'): value.pop(key, None)
                for key in ('id', 'experiment_order'): value['candidate'].pop(key)
            if order == 50:
                self.assertEqual(declared['job']['seed'], 20261027)
                seed27 = json.loads((ROOT / 'configs/R108-SCIENCE.json').read_text())
                self.assertEqual(right['initial_model_tensor_sha256'], seed27['initial_model_tensor_sha256'])
                right['initial_model_tensor_sha256'] = left['initial_model_tensor_sha256']
                right['job']['seed'] = left['job']['seed']
                self.assertEqual(resolved_age['contract']['version'], 'age-linear-lion10x-v1')
            else:
                self.assertEqual(declared['job']['input_preprocessing'], config.LOCAL_CONTRAST_RULE)
                self.assertEqual(right['candidate'].pop('input_preprocessing'), config.LOCAL_CONTRAST_RULE)
                right['job'].pop('input_preprocessing')
                self.assertNotIn('optimizer_policy', declared['job'])
                self.assertEqual(resolved_age, original_age)
            self.assertEqual(left, right)
        legacy = copy.deepcopy(job); legacy.pop('input_preprocessing')
        explicit_none = copy.deepcopy(legacy); explicit_none['input_preprocessing'] = None
        self.assertEqual(config.job_contract(legacy), config.job_contract(explicit_none))

    def test_lion_candidate_identity(self):
        from retrain import age
        from retrain.persistence import canonical_digest
        policy={'version':'lion10x-v1','betas':[.9,.99],'learning_rate_scale':.1,
            'nonzero_weight_decay_scale':10.,'precision':'fp32','state':'exp_avg_only',
            'update':'decoupled_decay_then_sign_then_momentum'}
        source=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+10*i)} for i in range(4)],'fit_indices':[0,1,2,3]})
        for filename,order in (('R136-SCIENCE.json',48),('R137-SCIENCE.json',49)):
            path=ROOT/'configs'/filename;declared=json.loads(path.read_text())
            site={'schema_version':1,'scope':'cpu_technical_prep','round_id':'identity-fixture','git_commit':'0'*40,
                'work_root':str(ROOT),'round_root':str(ROOT/'rounds/identity-fixture'),
                'pod':{'namespace':'fixture','name':'fixture','uid':'00000000-0000-0000-0000-000000000001','container':'fixture','python':sys.executable},
                'gpus':[{'ordinal':i,'pci':f'0000:{i+1:02x}:00.0','vram_total_bytes':64*1024**3} for i in range(8)],
                **{k:str(ROOT/v) for k,v in declared['paths'].items()},
                'science':{'binding_version':declared['candidate_profile'],'candidate':declared['candidate'],
                    'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'SITE_BINDING.json';p.write_text(json.dumps(site))
                with patch.dict(os.environ,NCCCU_SITE_BINDING=str(p),NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()):
                    try:actual=experiment.candidate_from_configuration(experiment.science_configuration())
                    except ValueError as error:self.fail(f'adopted {filename} identity is not connected: {error}')
                    self.assertEqual((actual['id'],actual['experiment_order']),(declared['candidate']['id'],order))
                    self.assertEqual(actual.get('optimizer_policy'),policy if order==49 else None)
                    with patch.object(experiment,'metadata',return_value={'AGE.json':source}):
                        resolved=experiment.resolved_age_spec()
                    age.validate_spec(resolved)
                    if order==49:
                        c=copy.deepcopy(resolved['contract'])
                        self.assertEqual((c['version'],c['lr'],c['weight_decay']),('age-linear-lion10x-v1',1e-4,1e-3))
                        self.assertEqual(c.pop('source_contract_sha256'),source['contract_sha256'])
                        c.update(version='age-linear-v1',lr=1e-3,weight_decay=1e-4)
                        self.assertEqual(c,source['contract']);self.assertEqual(canonical_digest(c),source['contract_sha256'])
                        self.assertNotEqual(resolved['contract_sha256'],source['contract_sha256'])
                    else:self.assertEqual(resolved,source)
        current=json.loads((ROOT/'configs/R137-SCIENCE.json').read_text())
        self.assertEqual(current['candidate_profile'],experiment.LION_SMALL_BINDING)
        self.assertEqual(current['job']['optimizer_policy'],policy);self.assertEqual(current['candidate']['optimizer_policy'],policy)
        self.assertEqual(current['official_evaluation'],experiment.OFFICIAL_GATE_0996)
        original=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
        self.assertEqual(current['initial_model_tensor_sha256'],original['initial_model_tensor_sha256'])
        self.assertEqual((current['job']['backbone_lr'],current['job']['head_lr'],current['job']['weight_decay']),(1e-5,1e-4,1e-3))
        left,right=copy.deepcopy(original),copy.deepcopy(current)
        for value in (left,right):
            for k in ('candidate_profile','comparison_note','official_evaluation'):value.pop(k,None)
            for k in ('id','experiment_order','optimizer_policy'):value['candidate'].pop(k,None)
            value['job'].pop('optimizer_policy',None)
        right['job'].update(backbone_lr=1e-4,head_lr=1e-3,weight_decay=1e-4)
        self.assertEqual(left,right)

    def test_R136_fixed_raw_GC_identity_and_R107_science(self):
        policy={'version':'raw-convlinear-weight-gc-v1','targets':'Conv2d_and_Linear_direct_weight','dtype':'float32',
            'axes':'all_non_output_dimensions','position':'after_ddp_mean_mirror_merge_unscale_before_global_clip','schedule':'each_real_update'}
        for filename,order in (('R135-SCIENCE.json',47),('R136-SCIENCE.json',48)):
            path=ROOT/'configs'/filename;declared=json.loads(path.read_text())
            site={'schema_version':1,'scope':'cpu_technical_prep','round_id':'identity-fixture','git_commit':'0'*40,
                'work_root':str(ROOT),'round_root':str(ROOT/'rounds/identity-fixture'),
                'pod':{'namespace':'fixture','name':'fixture','uid':'00000000-0000-0000-0000-000000000001','container':'fixture','python':sys.executable},
                'gpus':[{'ordinal':i,'pci':f'0000:{i+1:02x}:00.0','vram_total_bytes':64*1024**3} for i in range(8)],
                **{k:str(ROOT/v) for k,v in declared['paths'].items()},
                'science':{'binding_version':declared['candidate_profile'],'candidate':declared['candidate'],
                    'configuration':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'SITE_BINDING.json';p.write_text(json.dumps(site))
                with patch.dict(os.environ,NCCCU_SITE_BINDING=str(p),NCCCU_SITE_BINDING_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()):
                    try:actual=experiment.candidate_from_configuration(experiment.science_configuration())
                    except ValueError as e:self.fail(f'adopted {filename} identity is not connected: {e}')
            self.assertEqual((actual['id'],actual['experiment_order']),(declared['candidate']['id'],order))
            self.assertEqual(actual.get('gradient_centralization_policy'),policy if order==48 else None)
        current=json.loads((ROOT/'configs/R136-SCIENCE.json').read_text())
        self.assertEqual(current['candidate_profile'],experiment.RAW_GC_SMALL_BINDING)
        self.assertEqual(current['candidate']['gradient_centralization_policy'],policy)
        self.assertEqual(current['job']['gradient_centralization_policy'],policy)
        self.assertEqual(current['official_evaluation'],experiment.OFFICIAL_GATE_0996)
        original=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
        self.assertEqual(current['initial_model_tensor_sha256'],original['initial_model_tensor_sha256'])
        left,right=copy.deepcopy(original),copy.deepcopy(current)
        for value in (left,right):
            for k in ('candidate_profile','comparison_note','official_evaluation'):value.pop(k,None)
            for k in ('id','experiment_order','gradient_centralization_policy'):value['candidate'].pop(k,None)
            value['job'].pop('gradient_centralization_policy',None)
        self.assertEqual(left,right)

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

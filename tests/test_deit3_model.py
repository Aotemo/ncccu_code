"""Native DeiT3 geometry, original-factory oracle and real P8/cold reconstruction."""
import copy,json,os
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import torch
from torch import nn
import timm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain.models import _create_deit3_binary

def job():
    from retrain import age
    value=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
    value.update(model='deit3_small',initialization={'kind':'random'},
        age=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]}))
    return value

class DeiT3Test(unittest.TestCase):
    def test_original_factory_reference_parameters_rng_and_512_forward(self):
        torch.manual_seed(51)
        reference=timm.create_model('deit3_small_patch16_384.fb_in22k_ft_in1k',pretrained=False,drop_path_rate=.1,img_size=384)
        reference.set_input_size(img_size=512)
        reference.head=nn.Sequential(nn.Dropout(.2),nn.Linear(384,1))
        expected_rng=torch.random.get_rng_state().clone()
        torch.manual_seed(51)
        model=_create_deit3_binary(job(),initialize=False)
        self.assertTrue(torch.equal(expected_rng,torch.random.get_rng_state()))
        for name,tensor in model.network.state_dict().items():
            other=name.replace('classifier.','head.')
            self.assertTrue(torch.equal(tensor,reference.state_dict()[other]),name)
        self.assertEqual(len(dict(model.named_parameters())),177)
        self.assertFalse(dict(model.named_buffers()))
        self.assertEqual(tuple(model.network.pos_embed.shape),(1,1024,384))
        self.assertTrue(model.network.no_embed_class)
        images=torch.randn(1,3,512,512);ages=torch.zeros(1,2)
        model.eval();reference.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(images,ages),reference(images).flatten(),rtol=0,atol=0)

    def test_actual_P8_backward_optimizer_and_saved_cold_graph(self):
        from retrain import models,config,ema
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        value=job();torch.manual_seed(53)
        model=_create_deit3_binary(value,initialize=False)
        optimizer=_optimizer(model,value)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        shadow=ema.EMA(model,ema.RULE)
        self.assertEqual(models.describe_deit3(shadow.model),models.describe_deit3(model))
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        images=torch.randn(8,3,512,512);ages=torch.stack((torch.linspace(-1,1,8),torch.ones(8)),dim=1)
        saved={}
        def pack(t):
            storage=t.untyped_storage();saved[storage.data_ptr()]=storage.nbytes()
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            logits=model(images,ages)
            (logits*torch.linspace(-1,1,8)).sum().backward()
        self.assertEqual(tuple(logits.shape),(8,))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
        model.eval()
        with torch.inference_mode():
            expected=model(images[:1],ages[:1])
            image_only=model.network(images[:1]).flatten()
            torch.testing.assert_close(expected,image_only+(ages[:1]*model.age_weight).sum(1),rtol=0,atol=0)
        payload={'format_version':1,'mode':'train','job':value,'model':model.state_dict(),'age':value['age'],
                 'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved_job,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved_job,value)
            self.assertEqual(origin['deit3'],models.describe_deit3(model))
            with torch.inference_mode():actual=cold(images[:1],ages[:1])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        print(json.dumps({'scope':'CPU_synthetic_P8_512_saved_storage_not_GPU_peak',
            'unique_saved_CPU_storage_bytes':sum(saved.values()),'parameter_elements':sum(p.numel()for p in model.parameters()),
            'GPU_peak_or_latency_measured':False}),flush=True)
        self.assertFalse(torch.cuda.is_initialized())

class DeiT3MeanPatchTest(unittest.TestCase):
    def test_native_meanpatch_oracle_rng_age_and_saved_cold_graph(self):
        from retrain import config,models
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        from retrain.config import DEIT3_MEANPATCH_RULE
        value=job();value['deit_pool_policy']=copy.deepcopy(DEIT3_MEANPATCH_RULE)
        config.validate_job(value)
        wrong=copy.deepcopy(value);wrong['model']='convnext_small'
        with self.assertRaises(ValueError):config.validate_job(wrong)
        absent=job();explicit=job();explicit['deit_pool_policy']=None
        self.assertEqual(config.job_contract(absent),config.job_contract(explicit))
        torch.manual_seed(67)
        reference=timm.create_model('deit3_small_patch16_384.fb_in22k_ft_in1k',pretrained=False,drop_path_rate=.1,img_size=384)
        reference.set_input_size(img_size=512)
        reference.head=nn.Sequential(nn.Dropout(.2),nn.Linear(384,1));reference.global_pool='avg'
        expected_rng=torch.random.get_rng_state().clone()
        torch.manual_seed(67);model=models.create_model(value,initialize=False)
        self.assertTrue(torch.equal(expected_rng,torch.random.get_rng_state()))
        for name,tensor in model.network.state_dict().items():
            self.assertTrue(torch.equal(tensor,reference.state_dict()[name.replace('classifier.','head.')]),name)
        torch.manual_seed(67);original=models.create_model(absent,initialize=False)
        self.assertEqual(set(original.state_dict()),set(model.state_dict()))
        self.assertTrue(all(torch.equal(original.state_dict()[n],t)for n,t in model.state_dict().items()))
        del original
        model.eval();reference.eval();images=torch.randn(1,3,512,512);ages=torch.tensor([[.3,1.]])
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        with torch.inference_mode():
            features=model.network.forward_features(images)
            manual=model.network.classifier(features[:,1:].mean(1)).flatten()+(ages*model.age_weight).sum(1)
            expected=model(images,ages)
            torch.testing.assert_close(expected,manual,rtol=0,atol=0)
            torch.testing.assert_close(expected,reference(images).flatten()+(ages*model.age_weight).sum(1),rtol=0,atol=0)
        self.assertIsInstance(model.network.fc_norm,nn.Identity)
        payload={'format_version':1,'mode':'train','job':value,'model':model.state_dict(),'age':value['age'],
            'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved_job,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved_job['deit_pool_policy'],DEIT3_MEANPATCH_RULE)
            self.assertEqual(origin['deit3']['pool_policy'],DEIT3_MEANPATCH_RULE)
            self.assertEqual(origin['deit3']['global_pool'],'avg')
            with torch.inference_mode():actual=cold(images,ages)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_P8_meanpatch_all_gradients_empty_optimizer_and_EMA(self):
        from retrain import config,models,ema
        from retrain.engine import _optimizer
        value=job();value['deit_pool_policy']=copy.deepcopy(config.DEIT3_MEANPATCH_RULE)
        torch.manual_seed(71);model=models.create_model(value,initialize=False)
        optimizer=_optimizer(model,value)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual(len(dict(model.named_parameters())),177)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        shadow=ema.EMA(model,ema.RULE)
        self.assertEqual(models.describe_deit3(shadow.model),models.describe_deit3(model))
        images=torch.randn(8,3,512,512);ages=torch.stack((torch.linspace(-1,1,8),torch.ones(8)),dim=1)
        logits=model(images,ages);(logits*torch.linspace(-1,1,8)).sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

class ViTDINO3Test(unittest.TestCase):
    def _job(self):
        value=job();value.update(model='vit_dino3_small',final_norm_eps=1e-5)
        return value

    def test_native_factory_rng_prefix_mean_buffer_and_age(self):
        from retrain import models
        value=self._job();torch.manual_seed(83)
        reference=timm.create_model('vit_small_patch16_dinov3.lvd1689m',pretrained=False,
            img_size=512,num_classes=0,global_pool='avg',drop_path_rate=.1,drop_rate=0.)
        head=nn.Sequential(nn.Dropout(.2),nn.Linear(384,1))
        expected_rng=torch.random.get_rng_state().clone()
        torch.manual_seed(83);model=models._create_vit_dino3_binary(value,initialize=False)
        self.assertTrue(torch.equal(expected_rng,torch.random.get_rng_state()))
        for name,tensor in reference.state_dict().items():
            self.assertTrue(torch.equal(tensor,model.network.state_dict()[name]),name)
        for name,tensor in head.state_dict().items():
            self.assertTrue(torch.equal(tensor,model.classifier.state_dict()[name]),name)
        self.assertEqual(len(dict(model.named_parameters())),165)
        self.assertEqual(set(dict(model.named_buffers())),{'network.rope.periods'})
        self.assertNotIn('network.rope.periods',model.state_dict())
        self.assertTrue(torch.equal(model.network.rope.periods,reference.rope.periods))
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        model.eval();reference.eval();head.eval()
        images=torch.randn(1,3,512,512);ages=torch.tensor([[.3,1.]])
        with torch.inference_mode():
            features=reference.forward_features(images)
            manual=head(features[:,5:].mean(1)).flatten()+(ages*model.age_weight).sum(1)
            torch.testing.assert_close(model(images,ages),manual,rtol=0,atol=0)
        description=models.describe_vit_dino3(model)
        self.assertEqual(description['num_prefix_tokens'],5)
        self.assertEqual(description['global_pool'],'avg')
        self.assertEqual(description['norm_eps'],1e-5)
        self.assertFalse(torch.cuda.is_initialized())

    def test_P8_finite_gradients_optimizer_EMA_and_real_cold_graph(self):
        from retrain import config,ema,models
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        value=self._job();config.validate_job(value)
        torch.manual_seed(89);model=models.create_model(value,initialize=False)
        optimizer=_optimizer(model,value)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        shadow=ema.EMA(model,ema.RULE)
        self.assertEqual(models.describe_vit_dino3(model),models.describe_vit_dino3(shadow.model));del shadow
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        periods=model.network.rope.periods.clone()
        images=torch.randn(8,3,512,512);ages=torch.stack((torch.linspace(-1,1,8),torch.ones(8)),dim=1)
        storage={};calls=[]
        def pack(tensor):
            s=tensor.untyped_storage();storage[s.data_ptr()]=s.nbytes();return tensor
        hook=model.network.register_forward_hook(lambda module,args,output:calls.append(list(args[0].shape)))
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
            logits=model(images,ages);(logits*torch.linspace(-1,1,8)).sum().backward()
        hook.remove();self.assertEqual(calls,[[8,3,512,512]])
        self.assertEqual(len(dict(model.named_parameters())),165)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
        self.assertTrue(torch.equal(periods,model.network.rope.periods));model.eval()
        with torch.inference_mode():expected=model(images[:1],ages[:1])
        payload={'format_version':1,'mode':'train','job':value,'model':model.state_dict(),'age':value['age'],
                 'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved,value)
            self.assertEqual(origin['vit_dino3'],models.describe_vit_dino3(model))
            self.assertTrue(torch.equal(cold.network.rope.periods,periods))
            with torch.inference_mode():actual=cold(images[:1],ages[:1])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        print(json.dumps({'scope':'native_ViTS_CPU_P8_saved_storage_not_DCU_peak',
            'saved_storage_bytes':sum(storage.values()),'parameter_tensors':165,
            'nonpersistent_periods_cold_exact':True,'GPU_initialized':False}),flush=True)
        self.assertFalse(torch.cuda.is_initialized())

VIT_CLS_RULE={'version':'vit-dinov3-final-ln-cls-v1','pool':'token','token_index':0,
    'prefix_tokens_retained_in_attention':5,'position':'after_native_final_LN_before_original_dropout_linear'}


class ViTDINO3CLSTest(unittest.TestCase):
    def _job(self):
        value=ViTDINO3Test._job(self);value['vit_dino3_pool_policy']=copy.deepcopy(VIT_CLS_RULE)
        return value

    def test_native_CLS_oracle_default_None_RNG_age_and_config(self):
        from retrain import age,config,models
        value=self._job();torch.manual_seed(113)
        reference=timm.create_model('vit_small_patch16_dinov3.lvd1689m',pretrained=False,
            img_size=512,num_classes=0,global_pool='avg',drop_path_rate=.1,drop_rate=0.)
        head=nn.Sequential(nn.Dropout(.2),nn.Linear(384,1));rng=torch.random.get_rng_state().clone()
        reference.global_pool='token';reference.eval();head.eval()
        torch.manual_seed(113);model=models._create_vit_dino3_binary(value,initialize=False).eval()
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        for name,tensor in reference.state_dict().items():self.assertTrue(torch.equal(tensor,model.network.state_dict()[name]),name)
        for name,tensor in head.state_dict().items():self.assertTrue(torch.equal(tensor,model.classifier.state_dict()[name]),name)
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        images=torch.randn(1,3,512,512);ages=torch.tensor([[.3,1.]])
        with torch.inference_mode():
            features=reference.forward_features(images)
            manual=head(features[:,0]).flatten()+(ages*model.age_weight).sum(1)
            torch.testing.assert_close(reference.forward_head(features),features[:,0],rtol=0,atol=0)
            with patch('retrain.age.residual',wraps=age.residual)as residual:
                torch.testing.assert_close(model(images,ages),manual,rtol=0,atol=0)
                self.assertEqual(residual.call_count,1)
        self.assertEqual(models.describe_vit_dino3(model)['vit_dino3_pool_policy'],VIT_CLS_RULE)
        baseline=ViTDINO3Test._job(self);explicit={**baseline,'vit_dino3_pool_policy':None}
        torch.manual_seed(113);legacy=models._create_vit_dino3_binary(baseline,initialize=False);old_rng=torch.random.get_rng_state().clone()
        torch.manual_seed(113);none=models._create_vit_dino3_binary(explicit,initialize=False)
        self.assertTrue(torch.equal(old_rng,torch.random.get_rng_state()))
        for name,tensor in legacy.state_dict().items():self.assertTrue(torch.equal(tensor,none.state_dict()[name]),name)
        self.assertEqual(models.describe_vit_dino3(legacy),models.describe_vit_dino3(none))
        self.assertEqual(legacy.network.global_pool,'avg')
        self.assertNotIn('vit_dino3_pool_policy',models.describe_vit_dino3(legacy))
        self.assertEqual(config.job_contract(baseline),config.job_contract(explicit))
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            path=Path(directory)/'config.json'
            for original in(value,explicit,baseline):
                item={**original,'physical_batch':8}
                path.write_text(json.dumps({'jobs':[item],'execution':{'devices':['cuda:'+str(i)for i in range(8)],'per_rank_microbatch':8}}))
                loaded=config.load_config(path,environ={})['jobs'][0]
                self.assertEqual(loaded.get('vit_dino3_pool_policy'),original.get('vit_dino3_pool_policy'))
                if original.get('vit_dino3_pool_policy')is None:self.assertNotIn('vit_dino3_pool_policy',loaded)
        self.assertFalse(torch.cuda.is_initialized())

    def test_CLS_P8_all_gradients_real_noBN_EMA_and_cold_graph(self):
        from retrain import config,ema,models
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        value=self._job();config.validate_job(value);torch.manual_seed(127)
        model=models.create_model(value,initialize=False);optimizer=_optimizer(model,value)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        images=torch.randn(8,3,512,512);ages=torch.stack((torch.linspace(-1,1,8),torch.ones(8)),dim=1)
        calls=[];hook=model.network.register_forward_hook(lambda module,args,out:calls.append(list(args[0].shape)))
        logits=model(images,ages);(logits*torch.linspace(-1,1,8)).sum().backward();hook.remove()
        self.assertEqual(calls,[[8,3,512,512]])
        self.assertEqual(len(dict(model.named_parameters())),165)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
        rule={**ema.RULE,'version':'ema-warmcopy1600-decay0999-nobn-v1','warmcopy_updates':1600,'decay':.999,'batch_norm_buffers':0}
        with patch('retrain.experiment91.ema_rule',return_value=rule),patch('retrain.experiment91.candidate',return_value={'model':value['model']}):
            shadow=ema.EMA(model,rule);shadow.update(model,1)
            self.assertEqual(ema.tensor_digest(shadow.model.state_dict()),ema.tensor_digest(model.state_dict()))
            self.assertEqual(models.describe_vit_dino3(shadow.model),models.describe_vit_dino3(model));del shadow
        self.assertFalse(optimizer.state);model.eval()
        with torch.inference_mode():expected=model(images[:1],ages[:1])
        payload={'format_version':1,'mode':'train','job':value,'model':model.state_dict(),'age':value['age'],
            'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved['vit_dino3_pool_policy'],VIT_CLS_RULE)
            self.assertEqual(origin['vit_dino3'],models.describe_vit_dino3(model))
            self.assertEqual(cold.network.global_pool,'token')
            self.assertTrue(torch.equal(cold.network.rope.periods,model.network.rope.periods))
            with torch.inference_mode():actual=cold(images[:1],ages[:1])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())


class ViTDINO3BaseCLSTest(unittest.TestCase):
    def _job(self):
        value=ViTDINO3CLSTest._job(self);value['model']='vit_dino3_base'
        return value

    def test_base_native_CLS_oracle_RNG_age_and_config(self):
        from retrain import age,config,models
        value=self._job();torch.manual_seed(191)
        reference=timm.create_model('vit_base_patch16_dinov3.lvd1689m',pretrained=False,
            img_size=512,num_classes=0,global_pool='avg',drop_path_rate=.1,drop_rate=0.)
        head=nn.Sequential(nn.Dropout(.2),nn.Linear(768,1));rng=torch.random.get_rng_state().clone()
        reference.global_pool='token';reference.eval();head.eval()
        torch.manual_seed(191);model=models._create_vit_dino3_binary(value,initialize=False).eval()
        self.assertEqual(model.network.num_features,768)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        for name,tensor in reference.state_dict().items():self.assertTrue(torch.equal(tensor,model.network.state_dict()[name]),name)
        for name,tensor in head.state_dict().items():self.assertTrue(torch.equal(tensor,model.classifier.state_dict()[name]),name)
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        images=torch.randn(1,3,512,512);ages=torch.tensor([[.3,1.]])
        with torch.inference_mode():
            features=reference.forward_features(images)
            manual=head(features[:,0]).flatten()+(ages*model.age_weight).sum(1)
            torch.testing.assert_close(reference.forward_head(features),features[:,0],rtol=0,atol=0)
            with patch('retrain.age.residual',wraps=age.residual)as residual:
                torch.testing.assert_close(model(images,ages),manual,rtol=0,atol=0)
                self.assertEqual(residual.call_count,1)
        description=models.describe_vit_dino3(model)
        self.assertEqual(description['features'],768)
        self.assertEqual(description['vit_dino3_pool_policy'],VIT_CLS_RULE)
        config.validate_job(value)
        self.assertFalse(torch.cuda.is_initialized())

    def test_base_CLS_P8_EMA_and_cold_graph(self):
        storage={}
        def pack(tensor):
            value=tensor.untyped_storage();storage[value.data_ptr()]=value.nbytes();return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
            ViTDINO3CLSTest.test_CLS_P8_all_gradients_real_noBN_EMA_and_cold_graph(self)
        print(json.dumps({'scope':'native_ViTB_CPU_P8_saved_storage_not_DCU_peak',
            'saved_storage_bytes':sum(storage.values()),'parameter_tensors':165,'GPU_initialized':False}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']));torch.set_num_interop_threads(1)
    unittest.main()

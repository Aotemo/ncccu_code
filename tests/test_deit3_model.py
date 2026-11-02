"""Native DeiT3 geometry, original-factory oracle and real P8/cold reconstruction."""
import copy,json,os
from pathlib import Path
import sys,tempfile,unittest
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

if __name__=='__main__':
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']));torch.set_num_interop_threads(1)
    unittest.main()

"""Native Large graph/RNG, actual CPU P8 gradients and checkpoint cold load."""
import copy,json,os
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))
CPU_EVIDENCE={}
DINO3_FIXTURE=None


class ConvnextLargeTest(unittest.TestCase):
    def test_large_dinov3_actual_initializer_head_rng_and_cold_graph(self):
        if DINO3_FIXTURE is None:self.skipTest('requires guarded Pod accepted public source/fresh fixture')
        import torchvision
        from retrain import age,models
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        job=copy.deepcopy(DINO3_FIXTURE['job']);job.pop('experiment91',None)
        with patch('retrain.experiment91.public_source',return_value=DINO3_FIXTURE['public_source']):
            torch.manual_seed(job['seed']);model=models.create_model(job,initialize=True)
        actual_rng=torch.random.get_rng_state().clone()
        torch.manual_seed(job['seed']);oracle=torchvision.models.convnext_large(weights=None,stochastic_depth_prob=.1)
        expected_head=torch.nn.Linear(1536,1)
        self.assertTrue(torch.equal(actual_rng,torch.random.get_rng_state()))
        for name,value in expected_head.state_dict().items():
            self.assertTrue(torch.equal(model.network.classifier[-1].state_dict()[name],value))
        del oracle,expected_head
        optimizer=_optimizer(model,job)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual(len(dict(model.named_parameters())),345)
        self.assertFalse(dict(model.named_buffers()))
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        self.assertTrue(torch.equal(model.age_weight,torch.zeros(2)))
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        model.eval();images=torch.randn(2,3,64,64);ages=torch.tensor([[.5,1.],[-.25,1.]])
        with patch('retrain.age.residual',wraps=age.residual)as residual:
            with torch.inference_mode():before=model(images,ages)
            self.assertEqual(residual.call_count,1)
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            payload={'format_version':1,'mode':'train','job':job,'model':model.state_dict(),'age':job['age'],
                'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved,_,_,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved['initialization']['format'],models.LARGE_DINO3_FORMAT)
            with torch.inference_mode():after=cold(images,ages)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())
        CPU_EVIDENCE.update(actual_public_headless_initializer=True,native_discarded1000_head_RNG_preserved=True,
            binary_head_rng_exact=True,cold_graph_exact=True,age_residual_calls=1,parameter_tensors=345,
            optimizer_groups=5,optimizer_state_entries=0,synthetic_input_shape=[2,3,64,64],
            no_real_images=True,GPU_initialized=False,
            unrelated_native512_P8_graph_evidence='R180 preparation; constructor prefix and forward suffix checked unchanged separately')

    def test_large_dinov3_format_local_kind_and_small_compatibility(self):
        from retrain import config,models
        self.assertEqual(models.LARGE_DINO3_FORMAT,'torchvision-convnext-large-dinov3-backbone-state-dict-v1')
        for name,format_name in(('convnext_small',models.SMALL_DINO3_FORMAT),('convnext_large',models.LARGE_DINO3_FORMAT)):
            job=config.default_job('format-fixture',name,'random')
            job['normalization']='imagenet'
            job['initialization']={'kind':'self_supervised','format':format_name,'sha256':'a'*64,
                'path':str(ROOT/'not-loaded-weight.pt'),'provenance':str(ROOT/'not-loaded-provenance.json'),
                'expected_backbone_state_keys':342}
            config.validate_job(job)
            self.assertEqual(config.job_contract(job)['initialization']['format'],format_name)
            wrong=copy.deepcopy(job);wrong['initialization']['expected_backbone_state_keys']=344
            with self.assertRaises(ValueError):config.validate_job(wrong)

    def test_native_factory_rng_P8_age_optimizer_ema_and_cold_load(self):
        import torchvision
        from retrain import age,config,ema,models
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        self.assertIn('convnext_large',config.MODELS)
        job=config.default_job('large-fixture','convnext_large','random')
        job.update(copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job']))
        job['model']='convnext_large';job['initialization']={'kind':'random'}
        job['age']=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]})
        config.validate_job(job)
        torch.manual_seed(29)
        oracle=torchvision.models.convnext_large(weights=None,stochastic_depth_prob=.1)
        old=oracle.classifier[-1];head=torch.nn.Linear(old.in_features,1)
        oracle.classifier=torch.nn.Sequential(*list(oracle.classifier.children())[:-1],torch.nn.Dropout(.2),head)
        oracle_rng=torch.random.get_rng_state().clone()
        torch.manual_seed(29);model=models.create_model(job,initialize=False).eval()
        self.assertTrue(torch.equal(oracle_rng,torch.random.get_rng_state()))
        for name,value in oracle.state_dict().items():
            self.assertTrue(torch.equal(value,model.network.state_dict()[name]),name)
        self.assertTrue(torch.equal(model.age_weight,torch.zeros(2)))
        del oracle,old,head
        self.assertEqual(len(dict(model.named_parameters())),345);self.assertFalse(dict(model.named_buffers()))
        self.assertEqual(model.network.classifier[-1].in_features,1536)
        self.assertEqual(model.network.features[0][0].out_channels,192)
        optimizer=_optimizer(model,job)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        shadow=ema.EMA(model,ema.RULE)
        self.assertEqual(set(shadow.model.state_dict()),set(model.state_dict()));del shadow
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        images=torch.randn(2,3,512,512);ages=torch.tensor([[.5,1.],[-.25,1.]])
        with patch('retrain.age.residual',wraps=age.residual)as residual:
            with torch.inference_mode():before=model(images,ages)
            self.assertEqual(residual.call_count,1)
        model.train();calls=[];storages={}
        hook=model.network.register_forward_hook(lambda module,args,out:calls.append(list(args[0].shape)))
        def save(tensor):
            storage=tensor.untyped_storage();storages[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(save,lambda tensor:tensor):
            outputs=model(images.repeat(4,1,1,1),ages.repeat(4,1))
            (outputs*torch.arange(1,9,dtype=torch.float32)).mean().backward()
        hook.remove();self.assertEqual(calls,[[8,3,512,512]])
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
        self.assertFalse(optimizer.state);model.eval()
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            payload={'format_version':1,'mode':'train','job':job,'model':model.state_dict(),'age':job['age'],
                     'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved,_,_,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved['model'],'convnext_large')
            with torch.inference_mode():after=cold(images,ages)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())
        CPU_EVIDENCE.update(native_random_initializer_fixture=True,actual_public_source_and_fresh_checked_separately=True,
            native_factory_and_head_age_rng_exact=True,P8_network_shapes=calls,finite_gradient_tensors=345,
            P8_saved_unique_storage_bytes=sum(storages.values()),storage_stat_is_not_DCU_peak_or_upper_bound=True,
            optimizer_steps=0,cold_graph_exact=True,age_residual_calls=1,GPU_initialized=False)


if __name__=='__main__':
    torch.set_num_threads(max(1,int(os.environ['NCCCU_CPU_THREADS'])-1));torch.set_num_interop_threads(1)
    torch.autograd.set_multithreading_enabled(False)
    result=unittest.main(exit=False).result
    print(json.dumps(CPU_EVIDENCE,indent=2),flush=True)
    raise SystemExit(0 if result.wasSuccessful()else 1)

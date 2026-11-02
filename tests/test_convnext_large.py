"""Native Large graph/RNG, actual CPU P8 gradients and checkpoint cold load."""
import copy,json,os
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))
CPU_EVIDENCE={}


class ConvnextLargeTest(unittest.TestCase):
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

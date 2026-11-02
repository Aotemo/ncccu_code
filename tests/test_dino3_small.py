"""Complete headless public loading with the unchanged binary Small graph."""
import copy,json,os
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import models
CPU_EVIDENCE={}


class Dino3SmallTest(unittest.TestCase):
    def test_fixed_self_supervised_config_and_fresh_identity(self):
        from retrain import config,fresh,precision
        public_format=getattr(models,'SMALL_DINO3_FORMAT',None)
        self.assertIsNotNone(public_format)
        job=config.default_job('small-ssl','convnext_small','imagenet')
        job.update(copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job']))
        job['initialization']={'kind':'self_supervised','format':public_format,'sha256':'0'*64,
            'path':'/synthetic/public.pt','provenance':'/synthetic/provenance.json','expected_backbone_state_keys':342}
        config.validate_job(job)
        for key,value in(('model','convnext_base'),('normalization','unit')):
            invalid=copy.deepcopy(job);invalid[key]=value
            with self.assertRaises(ValueError):config.validate_job(invalid)
        topology={'stop_after_updates':24000}
        initialization={'kind':'self_supervised','weights_sha256':'0'*64}
        spec={'format':'glaucoma-fresh-init-v1','purpose':'train','plan_version':'synthetic-source-contract',
              'plan_sha256':'1'*64,'source_digest':'2'*64,'job':config.job_contract(job),
              'initialization':initialization,'amp_initial_scale':1.,'precision_protocol':copy.deepcopy(precision.RULE),
              'schedule':{'steps_per_cycle':800,'total_updates':24000,'warmup_updates':1600,'stop_after_updates':24000}}
        identity={'fresh':spec,'phase_parent_sha256':None,'source_digest':spec['source_digest'],
                  'job':config.job_contract(job),'distributed':topology,'initialization':initialization}
        fresh.validate_fresh(spec,identity,job,topology,800,mode='train')
        invalid_spec,invalid_identity=copy.deepcopy((spec,identity))
        invalid_spec['initialization']['kind']='imagenet'
        invalid_identity['initialization']['kind']='imagenet'
        with self.assertRaises(ValueError):fresh.validate_fresh(invalid_spec,invalid_identity,job,topology,800,mode='train')

    def test_complete_backbone_native_rng_P8_and_checkpoint_cold_graph(self):
        from retrain import age,ema
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        public_format=getattr(models,'SMALL_DINO3_FORMAT',None)
        self.assertIsNotNone(public_format,'fixed Small headless public loader is not implemented')
        job=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
        job['age']=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]})
        job['initialization']={'kind':'random'}
        torch.manual_seed(29);reference=models.create_model(job,initialize=False).eval()
        reference_rng=torch.random.get_rng_state().clone()
        public={n:t.clone()for n,t in reference.network.state_dict().items()if not n.startswith('classifier.3.')}
        self.assertEqual(len(public),342)
        public['features.0.0.weight'].mul_(.75)
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            path=Path(directory)/'synthetic-public-backbone.pt';torch.save(public,path)
            job['initialization']={'kind':'self_supervised','format':public_format,'path':str(path),
                'sha256':'0'*64,'provenance':str(Path(directory)/'synthetic-provenance.json'),
                'expected_backbone_state_keys':342}
            init_job={**job,'experiment91':{'synthetic_loader_test':True}}
            proof={'kind':'self_supervised','model':'convnext_small','format':public_format,
                   'provenance_sha256':'synthetic-only','final_norm_eps':1e-6,'backbone_only':True}
            torch.manual_seed(29)
            with patch.object(models,'validate_initialization',return_value=proof), \
                    patch('retrain.experiment91.public_source',return_value={'provenance_sha256':'synthetic-only'}), \
                    patch('retrain.experiment91.validate_job'), \
                    patch('retrain.experiment91.describe_model',return_value={'scope':'synthetic-loader-fixture'}):
                model=models.create_model(init_job,initialize=True).eval()
            self.assertTrue(torch.equal(reference_rng,torch.random.get_rng_state()))
            for name in('weight','bias'):
                self.assertTrue(torch.equal(getattr(model.network.classifier[-1],name),getattr(reference.network.classifier[-1],name)))
            self.assertTrue(torch.equal(model.age_weight,reference.age_weight))
            for name,tensor in public.items():self.assertTrue(torch.equal(model.network.state_dict()[name],tensor))
            self.assertFalse(torch.equal(model.network.features[0][0].weight,reference.network.features[0][0].weight))
            receipt=model.public_initialization_receipt
            self.assertEqual(receipt['public_tensor_count'],342);self.assertIsNone(receipt['full_head_shape'])
            self.assertIs(receipt['backbone_only'],True)
            self.assertTrue(receipt['all_public_tensors_exact']and receipt['all_public_tensors_FP32_finite'])
            self.assertFalse(receipt['strict_missing_keys']or receipt['strict_unexpected_keys'])
            self.assertEqual(len(dict(model.named_parameters())),345);self.assertFalse(dict(model.named_buffers()))
            optimizer=_optimizer(model,job)
            self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
            self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
            shadow=ema.EMA(model,ema.RULE)
            self.assertEqual(set(shadow.model.state_dict()),set(model.state_dict()))
            with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
            images=torch.randn(2,3,512,512);ages=torch.tensor([[.5,1.],[-.25,1.]])
            with patch('retrain.age.residual',wraps=age.residual)as residual:
                with torch.inference_mode():before=model(images,ages)
                self.assertEqual(residual.call_count,1)
            model.train();calls=[]
            hook=model.network.register_forward_hook(lambda module,args,out:calls.append(list(args[0].shape)))
            outputs=model(images.repeat(4,1,1,1),ages.repeat(4,1))
            (outputs*torch.arange(1,9,dtype=torch.float32)).mean().backward();hook.remove()
            self.assertEqual(calls,[[8,3,512,512]])
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
            self.assertFalse(optimizer.state);model.eval()
            payload={'format_version':1,'mode':'train','job':job,'model':model.state_dict(),'age':job['age'],
                     'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved,_,_,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved['initialization']['format'],public_format)
            with torch.inference_mode():after=cold(images,ages)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())
        CPU_EVIDENCE.update(public_tensor_count=342,synthetic_public_values=True,actual_public_source_verified_separately=True,
            binary_head_and_age_RNG_exact=True,P8_network_shapes=calls,finite_gradient_tensors=345,
            optimizer_steps=0,cold_graph_exact=True,age_residual_calls=1,GPU_initialized=False)


if __name__=='__main__':
    torch.set_num_threads(max(1,int(os.environ['NCCCU_CPU_THREADS'])-1));torch.set_num_interop_threads(1)
    torch.autograd.set_multithreading_enabled(False)
    result=unittest.main(exit=False).result
    print(json.dumps(CPU_EVIDENCE,indent=2),flush=True)
    raise SystemExit(0 if result.wasSuccessful()else 1)

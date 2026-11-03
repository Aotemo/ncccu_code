"""Native fixed-BN graph, finite P8 gradients, and persistent cold reconstruction."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import timm,torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import config,models

def job():
    from retrain.age import make_spec
    value=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
    value.update(name='efficientnetv2-rw-s',model='efficientnetv2_rw_s',frozen_bn=True,
        initialization={'kind':'random'},age=make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}
            for i in range(4)],'fit_indices':[0,1,2,3]}))
    return value

class EfficientNetTest(unittest.TestCase):
    def test_WORLD8_keeps_native_fixed_BN_without_SyncBN_or_reference_merge(self):
        from retrain.distributed import RankContext,ReferenceBatchNorm
        value=job();model=models.create_model(value,initialize=False)
        context=object.__new__(RankContext)
        context.world_size=8;context.rank=0;context.device=torch.device('cpu');context.group=None
        before={n:t.clone()for n,t in model.named_buffers()}
        with patch('torch.nn.SyncBatchNorm.convert_sync_batchnorm',side_effect=AssertionError('fixed BN must remain native')):
            self.assertIs(context.prepare_model(model),model)
        mirror=ReferenceBatchNorm(model,context)
        self.assertEqual(mirror.layers,[]);mirror.capture();mirror.merge()
        self.assertEqual(mirror.before,[])
        self.assertTrue(all(torch.equal(before[n],t)for n,t in model.named_buffers()))
        self.assertTrue(all(not m.training for m in model.modules()if isinstance(m,nn.modules.batchnorm._BatchNorm)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_factory_RNG_frozen_BN_and_single_age_readout(self):
        self.assertIn('efficientnetv2_rw_s',config.MODELS)
        from retrain.age import residual
        value=job();torch.manual_seed(77)
        reference=timm.create_model('efficientnetv2_rw_s.ra2_in1k',pretrained=False,drop_rate=.2,drop_path_rate=.1)
        reference.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(1792,1));reference.drop_rate=0.
        expected_rng=torch.random.get_rng_state().clone();torch.manual_seed(77)
        model=models.create_model(value,initialize=False)
        self.assertTrue(torch.equal(expected_rng,torch.random.get_rng_state()))
        for n,t in reference.state_dict().items():self.assertTrue(torch.equal(t,model.network.state_dict()[n]),n)
        model.train(True)
        layers=[m for m in model.modules()if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        self.assertEqual(len(layers),112)
        self.assertTrue(all(not m.training and m.weight.requires_grad and m.bias.requires_grad for m in layers))
        self.assertTrue(model.network.classifier[0].training);self.assertEqual(model.network.drop_rate,0.)
        self.assertEqual(len(dict(model.named_parameters())),459);self.assertEqual(len(dict(model.named_buffers())),336)
        reference.eval();model.eval();images=torch.randn(1,3,512,512);ages=torch.tensor([[.7,1.]])
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        with torch.inference_mode():
            expected=reference(images).flatten()+(ages*model.age_weight).sum(1)
            with patch('retrain.age.residual',wraps=residual)as call:actual=model(images,ages)
        self.assertEqual(call.call_count,1);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        with self.assertRaises(ValueError):config.validate_job({**value,'frozen_bn':False})
        self.assertFalse(torch.cuda.is_initialized())

    def test_P8_gradients_frozen_buffers_EMA_legacy159_and_real_cold_load(self):
        self.assertIn('efficientnetv2_rw_s',config.MODELS)
        from retrain import ema
        from retrain.efficientnet_model import EMA_RULE,describe
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        value=job();config.validate_job(value);torch.manual_seed(79)
        model=models.create_model(value,initialize=False);optimizer=_optimizer(model,value)
        self.assertIs(type(optimizer),torch.optim.AdamW);self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        self.assertEqual({id(p)for g in optimizer.param_groups if g['role']=='head'for p in g['params']},
                         {id(p)for p in model.network.classifier[-1].parameters()})
        before={n:t.clone()for n,t in model.named_buffers()};shadow=ema.EMA(model,EMA_RULE)
        self.assertEqual(describe(model),describe(shadow.model))
        images=torch.randn(8,3,512,512);ages=torch.stack((torch.linspace(-1,1,8),torch.ones(8)),dim=1)
        logits=model(images,ages);F.softplus(logits[1::2][None,:]-logits[::2][:,None]).mean().backward()
        self.assertEqual(tuple(logits.shape),(8,))
        self.assertTrue(all(p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
        optimizer.step();shadow.update(model,1)
        self.assertTrue(all(torch.equal(before[n],t)for n,t in model.named_buffers()))
        self.assertTrue(all(torch.equal(before[n],t)for n,t in shadow.model.named_buffers()))
        shadow.successful=1600
        with torch.no_grad():model.network.classifier[-1].bias.add_(.1)
        prior=shadow.model.network.classifier[-1].bias.clone();shadow.update(model,1601)
        torch.testing.assert_close(shadow.model.network.classifier[-1].bias,
            prior*.999+model.network.classifier[-1].bias*.001,rtol=1e-6,atol=1e-7)
        self.assertTrue(all(torch.equal(before[n],t)for n,t in shadow.model.named_buffers()))
        ema.validate_state(shadow.state_dict(),model.state_dict(),1601,EMA_RULE)
        legacy={f'b{i}.{s}':torch.zeros((),dtype=torch.int64)if s=='num_batches_tracked'else torch.zeros(1)
            for i in range(53)for s in ('running_mean','running_var','num_batches_tracked')}
        state={'rule':ema.FROZEN_BN_RULE,'model':legacy,'update_count':0,'copy_count':0,'blend_count':0,'last_update_success':0}
        ema.validate_state(state,legacy,0,ema.FROZEN_BN_RULE)
        short=dict(legacy);short.pop(next(iter(short)))
        with self.assertRaises(ValueError):ema.validate_state({**state,'model':short},short,0,ema.FROZEN_BN_RULE)
        model.eval()
        with torch.inference_mode():expected=model(images[:1],ages[:1])
        payload={'format_version':1,'mode':'train','job':value,'model':model.state_dict(),'age':value['age'],
            'identity':{},'manifest_digest':'synthetic-only','successful_updates':1,'attempted_updates':1,'cpu_fixture_only':True}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved_job,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved_job,value);self.assertEqual(origin['efficientnetv2'],describe(model))
            with torch.inference_mode():actual=cold(images[:1],ages[:1])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    torch.set_num_threads(min(5,int(os.environ['NCCCU_CPU_THREADS'])));torch.set_num_interop_threads(1);unittest.main()

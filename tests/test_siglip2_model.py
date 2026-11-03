"""Native public MAP graph, P8 gradients, and saved graph reconstruction."""
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
    value.update(name='siglip2-vitb16-512',model='siglip2_vitb16',normalization='unit',
        initialization={'kind':'random'},age=make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}
            for i in range(4)],'fit_indices':[0,1,2,3]}))
    return value


class SigLIP2Test(unittest.TestCase):
    def test_native_MAP_factory_RNG_full_state_and_single_age_oracle(self):
        self.assertIn('siglip2_vitb16',config.MODELS)
        from retrain.age import residual
        from retrain.siglip2_model import describe
        value=job();torch.manual_seed(81)
        native=timm.create_model('vit_base_patch16_siglip_512.v2_webli',pretrained=False,
            drop_rate=0.,drop_path_rate=.1)
        classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(768,1))
        expected_rng=torch.random.get_rng_state().clone();torch.manual_seed(81)
        model=models.create_model(value,initialize=False)
        self.assertTrue(torch.equal(expected_rng,torch.random.get_rng_state()))
        for name,tensor in native.state_dict().items():
            self.assertTrue(torch.equal(tensor,model.network.state_dict()[name]),name)
        for name,tensor in classifier.state_dict().items():
            self.assertTrue(torch.equal(tensor,model.classifier.state_dict()[name]),name)
        self.assertEqual(len(dict(model.named_parameters())),165)
        self.assertEqual(sum(p.numel()for p in model.parameters()),93520899)
        self.assertEqual(len(dict(model.named_buffers())),0)
        self.assertEqual(describe(model)['public_pool'],'map')
        self.assertEqual(model.network.num_prefix_tokens,0)
        self.assertEqual(tuple(model.network.attn_pool.latent.shape),(1,1,768))
        model.eval();native.eval();classifier.eval()
        images=torch.randn(1,3,512,512);ages=torch.tensor([[.7,1.]])
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        with torch.inference_mode():
            features=native.forward_features(images)
            independent=native.attn_pool(features)
            torch.testing.assert_close(native(images),independent,rtol=0,atol=0)
            expected=classifier(independent).flatten()+(ages*model.age_weight).sum(1)
            with patch('retrain.age.residual',wraps=residual)as call:actual=model(images,ages)
        self.assertEqual(call.call_count,1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        config.validate_job(value)
        with self.assertRaises(ValueError):models.create_model({**value,'normalization':'imagenet'},initialize=False)
        self.assertFalse(torch.cuda.is_initialized())

    def test_P8_all_gradients_optimizer_EMA_and_real_checkpoint_cold_exact(self):
        self.assertIn('siglip2_vitb16',config.MODELS)
        from retrain import ema
        from retrain.siglip2_model import describe
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        value=job();config.validate_job(value);torch.manual_seed(83)
        model=models.create_model(value,initialize=False);optimizer=_optimizer(model,value)
        self.assertIs(type(optimizer),torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},
                         {id(p)for p in model.parameters()})
        self.assertEqual({id(p)for g in optimizer.param_groups if g['role']=='head'for p in g['params']},
                         {id(p)for p in model.classifier[-1].parameters()})
        shadow=ema.EMA(model,ema.R8_RULE);self.assertEqual(describe(model),describe(shadow.model))
        images=torch.randn(8,3,512,512);ages=torch.stack((torch.linspace(-1,1,8),torch.ones(8)),dim=1)
        logits=model(images,ages);F.softplus(logits[1::2][None,:]-logits[::2][:,None]).mean().backward()
        self.assertEqual(tuple(logits.shape),(8,))
        self.assertTrue(all(p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in model.parameters()))
        optimizer.step();shadow.update(model,1)
        self.assertTrue(all(torch.equal(t,shadow.model.state_dict()[n])for n,t in model.state_dict().items()))
        model.eval()
        with torch.inference_mode():expected=model(images[:1],ages[:1])
        payload={'format_version':1,'mode':'train','job':value,'model':model.state_dict(),'age':value['age'],
            'identity':{},'manifest_digest':'synthetic-only','successful_updates':1,'attempted_updates':1,'cpu_fixture_only':True}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved_job,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved_job,value);self.assertEqual(origin['siglip2'],describe(model))
            with torch.inference_mode():actual=cold(images[:1],ages[:1])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    torch.set_num_threads(min(5,int(os.environ['NCCCU_CPU_THREADS'])));torch.set_num_interop_threads(1);unittest.main()

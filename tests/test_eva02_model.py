"""Native EVA02 graph, explicit 512 padding, and production cold reconstruction."""
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
    value=config.default_job('eva02-fixture','eva02_base','random')
    value.update(copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job']))
    value.update(name='eva02-base',model='eva02_base',normalization='clip',
        initialization={'kind':'random'},age=make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}
            for i in range(4)],'fit_indices':[0,1,2,3]}))
    return value

class EVA02Test(unittest.TestCase):
    def test_clip_input_and_fixed_config(self):
        from PIL import Image
        from retrain.data import ImageDataset
        value=job();config.validate_job(value)
        with patch('retrain.data.configured_cache',return_value=None):
            dataset=ImageDataset('.',[],512,False,normalization='clip')
        image=Image.new('RGB',(512,512),(255,255,255))
        tensor=dataset._image_tensor(image,None)
        expected=(torch.ones(3)-torch.tensor([.48145466,.4578275,.40821073]))/torch.tensor([.26862954,.26130258,.27577711])
        torch.testing.assert_close(tensor,expected[:,None,None].expand_as(tensor),rtol=0,atol=0)
        for name in ('imagenet','unit'):
            with self.assertRaises(ValueError):config.validate_job({**value,'normalization':name})

    def test_native_pool_position_padding_RNG_and_age_once(self):
        self.assertIn('eva02_base',config.MODELS)
        from retrain.eva02_model import VARIANT,describe
        from retrain.age import residual
        torch.manual_seed(81)
        native=timm.create_model(VARIANT,pretrained=False,drop_rate=0.,drop_path_rate=.1)
        prefix=native.pos_embed[:,:1].clone();native.set_input_size(518)
        native.patch_embed.strict_img_size=False;native.patch_embed.dynamic_img_pad=True
        native.head=nn.Identity();classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(768,1))
        rng=torch.random.get_rng_state().clone();torch.manual_seed(81)
        model=models.create_model(job(),initialize=False)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        for name,tensor in native.state_dict().items():self.assertTrue(torch.equal(tensor,model.network.state_dict()[name]),name)
        self.assertTrue(torch.equal(prefix,model.network.pos_embed[:,:1]))
        for name,tensor in classifier.state_dict().items():self.assertTrue(torch.equal(tensor,model.classifier.state_dict()[name]),name)
        self.assertEqual(describe(model)['patch_grid'],[37,37])
        self.assertEqual(tuple(model.network.rope.pos_embed.shape),(1369,128))
        self.assertNotIn('network.rope.pos_embed',model.state_dict())
        model.eval();native.eval();classifier.eval();images=torch.randn(1,3,512,512);ages=torch.tensor([[.7,1.]])
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        with torch.inference_mode():
            patch_tokens=native.patch_embed(images)
            explicit=native.patch_embed.proj(F.pad(images,(0,6,0,6))).flatten(2).transpose(1,2)
            torch.testing.assert_close(patch_tokens,explicit,rtol=0,atol=0)
            features=native.forward_features(images)
            independent=native.fc_norm(features[:,1:].mean(1))
            torch.testing.assert_close(native(images),independent,rtol=0,atol=0)
            expected=classifier(independent).flatten()+(ages*model.age_weight).sum(1)
            with patch('retrain.age.residual',wraps=residual)as call:actual=model(images,ages)
        self.assertEqual(call.call_count,1);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_P8_gradients_optimizer_EMA_and_checkpoint_cold_exact(self):
        self.assertIn('eva02_base',config.MODELS)
        from retrain import ema
        from retrain.eva02_model import describe
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        value=job();config.validate_job(value);torch.manual_seed(83)
        model=models.create_model(value,initialize=False);optimizer=_optimizer(model,value)
        self.assertIs(type(optimizer),torch.optim.AdamW);self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        self.assertEqual({id(p)for g in optimizer.param_groups if g['role']=='head'for p in g['params']},{id(p)for p in model.classifier[-1].parameters()})
        shadow=ema.EMA(model,ema.R8_RULE);self.assertEqual(describe(model),describe(shadow.model))
        buffer_before=model.network.rope.pos_embed.clone()
        images=torch.randn(8,3,512,512);ages=torch.stack((torch.linspace(-1,1,8),torch.ones(8)),dim=1)
        logits=model(images,ages);F.softplus(logits[1::2][None,:]-logits[::2][:,None]).mean().backward()
        self.assertEqual(tuple(logits.shape),(8,))
        self.assertTrue(all(p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
        self.assertTrue(torch.equal(buffer_before,model.network.rope.pos_embed))
        optimizer.step();shadow.update(model,1)
        self.assertTrue(all(torch.equal(t,shadow.model.state_dict()[n])for n,t in model.state_dict().items()))
        model.eval()
        with torch.inference_mode():expected=model(images[:1],ages[:1])
        payload={'format_version':1,'mode':'train','job':value,'model':model.state_dict(),'age':value['age'],
            'identity':{},'manifest_digest':'synthetic-only','successful_updates':1,'attempted_updates':1,'cpu_fixture_only':True}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved_job,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved_job,value);self.assertEqual(origin['eva02'],describe(model))
            self.assertTrue(torch.equal(buffer_before,cold.network.rope.pos_embed))
            with torch.inference_mode():actual=cold(images[:1],ages[:1])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    torch.set_num_threads(min(5,int(os.environ['NCCCU_CPU_THREADS'])));torch.set_num_interop_threads(1);unittest.main()

"""Original natural-image Base/14, full512 pad6 and production EMA/cold graph."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import config,models,ema,experiment91 as ex

def job():
    from retrain.age import make_spec
    asset=Path(os.environ['NCCCU_DINOV2B_PUBLIC_ASSET']);meta=json.loads((asset/'PROVENANCE.json').read_text())
    value=config.default_job('dinov2-base-fixture','dinov2_vitb14','random')
    value.update(copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job']))
    value.update(name='dinov2-base',model='dinov2_vitb14',normalization='imagenet',
        initialization={'kind':'self_supervised','format':'dinov2-official-vitb14-512-pad6-state-dict-v1',
            'path':str(asset/'original/dinov2_vitb14_pretrain.pth'),'sha256':meta['sha256'],
            'provenance':str(asset/'PROVENANCE.json')},
        age=make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]}))
    return value

class DINOv2BaseTest(unittest.TestCase):
    def test_old_Small_factory_fixed_head_and_RNG_are_preserved(self):
        from retrain.dinov2_model import DINOv2Binary,_official_factory
        value=job();meta=json.loads(Path(value['initialization']['provenance']).read_text())
        value['model']='dinov2_vits14'
        value['initialization']={'provenance':meta['code_asset_provenance']['path']}
        factory=_official_factory(value)
        from dinov2.hub.backbones import dinov2_vits14
        self.assertIs(factory,dinov2_vits14)
        torch.manual_seed(73);backbone=factory(pretrained=False,drop_path_rate=.1)
        before=torch.random.get_rng_state().clone();model=DINOv2Binary(backbone,value['age'])
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        generator=torch.Generator(device='cpu').manual_seed(20261027)
        expected=torch.randn((1,768),generator=generator,dtype=torch.float32)*.01
        torch.testing.assert_close(model.classifier[-1].weight,expected,rtol=0,atol=0)
        self.assertTrue(bool((model.classifier[-1].bias==0).all()))
        self.assertEqual([n for n,p in model.named_parameters()if not p.requires_grad],['backbone.mask_token'])
        old=json.loads((ROOT/'configs/R141-SCIENCE.json').read_text())
        with patch.object(ex,'candidate',return_value=old['candidate']):
            proof=ex.describe_model(model)
        self.assertEqual(proof['head_in_features'],768)
        self.assertEqual(proof['dino']['embed_dim'],384)
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_public_factory_strict_native_head_and_RNG(self):
        self.assertIn('dinov2_vitb14',config.MODELS)
        from retrain.engine import _optimizer
        value=job();config.validate_job(value);torch.manual_seed(57)
        model=models.create_model(value,initialize=True);actual_rng=torch.random.get_rng_state().clone()
        state=torch.load(value['initialization']['path'],map_location='cpu',weights_only=True)
        self.assertEqual(len(state),175);self.assertEqual(set(state),set(model.backbone.state_dict()))
        for name,tensor in state.items():torch.testing.assert_close(model.backbone.state_dict()[name],tensor,rtol=0,atol=0)
        from dinov2.hub.backbones import dinov2_vitb14
        torch.manual_seed(57);native=dinov2_vitb14(pretrained=False,drop_path_rate=.1)
        native.load_state_dict(state,strict=True)
        head=nn.Sequential(nn.Dropout(.2),nn.Linear(1536,1))
        self.assertTrue(torch.equal(actual_rng,torch.random.get_rng_state()))
        for name,tensor in head.state_dict().items():self.assertTrue(torch.equal(tensor,model.classifier.state_dict()[name]),name)
        proof=model.public_initialization_receipt
        self.assertTrue(proof['all_public_tensors_exact']);self.assertTrue(proof['all_public_tensors_FP32_finite'])
        self.assertTrue(proof['backbone_only']);self.assertEqual(proof['public_tensor_count'],175)
        self.assertEqual(proof['public_tensor_sha256'],ema.tensor_digest(state))
        self.assertEqual([n for n,p in model.named_parameters()if not p.requires_grad],['backbone.mask_token'])
        self.assertEqual(len(dict(model.named_parameters())),178);self.assertEqual(dict(model.named_buffers()),{})
        optimizer=_optimizer(model,value);self.assertIs(type(optimizer),torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()if p.requires_grad})
        self.assertFalse(torch.cuda.is_initialized())

    def test_full512_pad6_CLS_mean_and_age_once(self):
        self.assertIn('dinov2_vitb14',config.MODELS)
        from retrain.age import residual
        model=models.create_model(job(),initialize=False).eval()
        images=torch.randn(1,3,512,512);ages=torch.tensor([[.7,1.]])
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            padded=F.pad(images,(0,6,0,6));features=model.backbone.forward_features(padded)
            self.assertEqual(tuple(features['x_norm_clstoken'].shape),(1,768))
            self.assertEqual(tuple(features['x_norm_patchtokens'].shape),(1,1369,768))
            independent=torch.cat((features['x_norm_clstoken'],features['x_norm_patchtokens'].mean(1)),dim=1)
            expected=model.classifier(independent).flatten()+(ages*model.age_weight).sum(1)
            with patch('retrain.age.residual',wraps=residual)as call:actual=model(images,ages)
        self.assertEqual(call.call_count,1);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(tuple(model.backbone.pos_embed.shape),(1,1370,768))
        self.assertFalse(torch.cuda.is_initialized())

    def test_P8_gradients_production_EMA_and_real_checkpoint_cold(self):
        self.assertIn('dinov2_vitb14',config.MODELS)
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        value=job();model=models.create_model(value,initialize=False);optimizer=_optimizer(model,value)
        rule=json.loads((ROOT/'runtime/EMA-TINY.json').read_text())
        with patch.object(ex,'ema_rule',return_value=rule):
            shadow=ema.EMA(model,rule);frozen=model.backbone.mask_token.clone()
            images=torch.randn(8,3,512,512);ages=torch.stack((torch.linspace(-1,1,8),torch.ones(8)),dim=1)
            logits=model(images,ages);F.softplus(logits[1::2][None,:]-logits[::2][:,None]).mean().backward()
            active=[p for p in model.parameters()if p.requires_grad]
            self.assertEqual(len(active),177)
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in active))
            self.assertIsNone(model.backbone.mask_token.grad)
            optimizer.step();shadow.update(model,1)
            self.assertTrue(torch.equal(frozen,model.backbone.mask_token))
            for name,tensor in model.state_dict().items():self.assertTrue(torch.equal(tensor,shadow.model.state_dict()[name]),name)
            synthetic=shadow.state_dict();synthetic.update(update_count=1600,copy_count=1600,blend_count=0,last_update_success=1600)
            blended=ema.EMA(model,rule,state=synthetic,successful=1600)
            old=blended.model.classifier[-1].bias.clone()
            with torch.no_grad():model.classifier[-1].bias.add_(.1)
            expected=old.mul(.999).add(model.classifier[-1].bias,alpha=1-.999)
            blended.update(model,1601)
            torch.testing.assert_close(blended.model.classifier[-1].bias,expected,rtol=0,atol=0)
            self.assertTrue(torch.equal(frozen,blended.model.backbone.mask_token))
        model.eval()
        with torch.inference_mode():expected=model(images[:1],ages[:1])
        payload={'format_version':1,'mode':'train','job':value,'age':value['age'],'model':model.state_dict(),
            'identity':{},'manifest_digest':'synthetic-only','successful_updates':1,'attempted_updates':1,'cpu_fixture_only':True}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload);load=torch.load
            def checkpoint_only(path,*args,**kwargs):
                if isinstance(path,(str,Path))and Path(path)==Path(value['initialization']['path']):self.fail('cold graph reopened public weights')
                return load(path,*args,**kwargs)
            with patch('torch.load',side_effect=checkpoint_only):cold,saved,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved,value);self.assertEqual(ema.tensor_digest(cold.state_dict()),ema.tensor_digest(model.state_dict()))
            self.assertEqual(tuple(cold.backbone.pos_embed.shape),(1,1370,768))
            with torch.inference_mode():actual=cold(images[:1],ages[:1])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    os.environ['XFORMERS_DISABLED']='1'
    torch.set_num_threads(min(5,int(os.environ['NCCCU_CPU_THREADS'])));torch.set_num_interop_threads(1);unittest.main()

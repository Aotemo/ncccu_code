"""Independent scalar reference for the registered longer averaging window."""
import copy,io,os,sys,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import ema,experiment91 as ex

class Scalar(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor([1.,-1.]))

class LongEMATest(unittest.TestCase):
    def test_exact_rule_warmcopy_reference_blends_and_state_roundtrip(self):
        rule=getattr(ex,'LONG_EMA_RULE',None)
        self.assertIsNotNone(rule)
        self.assertEqual(rule,{'version':'ema-warmcopy1600-decay09999-nobn-v1','warmcopy_updates':1600,
            'decay':.9999,'dtype':'torch.float32','floating_buffers':'same_as_parameters',
            'integer_buffers':'copy','batch_norm_buffers':0})
        model=Scalar();rng=torch.random.get_rng_state().clone()
        with patch.object(ex,'ema_rule',return_value=rule):
            shadow=ema.EMA(model,rule)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            for step in range(1,1601):
                with torch.no_grad():model.weight.copy_(torch.tensor([step/1600,-step/1600]))
                shadow.update(model,step)
            self.assertTrue(torch.equal(shadow.model.weight,model.weight))
            self.assertEqual((shadow.state_dict()['copy_count'],shadow.state_dict()['blend_count']),(1600,0))
            expected=[1.,-1.]
            for step,values in((1601,[3.,2.]),(1602,[-2.,5.])):
                with torch.no_grad():model.weight.copy_(torch.tensor(values))
                expected=[.9999*a+.0001*b for a,b in zip(expected,values)]
                shadow.update(model,step)
                torch.testing.assert_close(shadow.model.weight,torch.tensor(expected),rtol=2e-7,atol=2e-7)
            saved=io.BytesIO();torch.save(shadow.state_dict(),saved);saved.seek(0)
            loaded=torch.load(saved,map_location='cpu',weights_only=True)
            cold=ema.EMA(model,rule,state=loaded,successful=1602)
            with torch.no_grad():model.weight.copy_(torch.tensor([4.,-3.]))
            shadow.update(model,1603);cold.update(model,1603)
            self.assertTrue(torch.equal(shadow.model.weight,cold.model.weight))
            self.assertEqual((cold.state_dict()['copy_count'],cold.state_dict()['blend_count']),(1600,3))
            ema.validate_state(cold.state_dict(),model.state_dict(),1603,rule)
        self.assertEqual(ema.R8_RULE['decay'],.999)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);unittest.main()

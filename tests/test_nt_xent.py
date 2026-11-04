"""Independent instance-contrastive values, gradients, pair identities and science."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))


class NTXentTest(unittest.TestCase):
    def test_independent_scalar_oracle_and_both_view_gradient_check(self):
        import torch
        from retrain.vicreg import nt_xent_loss
        a=torch.tensor([[1.,2.,-.5],[.3,-1.,2.],[2.,.5,1.]],dtype=torch.float64,requires_grad=True)
        b=torch.tensor([[1.2,1.6,-.2],[.1,-1.2,1.8],[1.7,.3,1.3]],dtype=torch.float64,requires_grad=True)
        rows=a.detach().tolist()+b.detach().tolist();normalized=[]
        for row in rows:
            length=math.sqrt(sum(v*v for v in row));normalized.append([v/length for v in row])
        terms=[]
        for i,row in enumerate(normalized):
            scores={j:sum(x*y for x,y in zip(row,other))/.1 for j,other in enumerate(normalized)if j!=i}
            peak=max(scores.values());normalizer=peak+math.log(sum(math.exp(s-peak)for s in scores.values()))
            terms.append(normalizer-scores[(i+3)%6])
        loss,stats=nt_xent_loss(a,b,temperature=.1)
        self.assertAlmostEqual(float(loss.detach()),sum(terms)/6,places=12)
        self.assertEqual(stats['contrastive_logits_shape'],[6,6])
        self.assertTrue(torch.autograd.gradcheck(lambda x,y:nt_xent_loss(x,y,temperature=.1)[0],(a,b)))
        gradients=torch.autograd.grad(loss,(a,b))
        self.assertTrue(all(torch.isfinite(g).all()and g.abs().sum()>0 for g in gradients))

    def test_global64_true_pairs_self_exclusion_permutation_and_row_scaling(self):
        import torch
        from retrain.vicreg import nt_xent_loss
        rng=torch.Generator().manual_seed(20261104)
        a=torch.randn(64,256,generator=rng);b=a+.05*torch.randn(64,256,generator=rng)
        loss,stats=nt_xent_loss(a,b,temperature=.1)
        wrong,_=nt_xent_loss(a,b.roll(1,0),temperature=.1)
        self.assertLess(float(loss),float(wrong))
        self.assertEqual(stats['contrastive_logits_shape'],[128,128])
        self.assertEqual(stats['contrastive_anchor_count'],128)
        self.assertEqual(stats['contrastive_negative_views_per_anchor'],126)
        self.assertEqual(stats['contrastive_positive_targets_per_anchor'],1)
        self.assertIs(stats['contrastive_self_excluded'],True)
        self.assertEqual(stats['contrastive_temperature'],.1)
        permutation=torch.randperm(64,generator=rng)
        reordered,_=nt_xent_loss(a[permutation],b[permutation],temperature=.1)
        scaled,_=nt_xent_loss(a*torch.linspace(.5,2,64)[:,None],b*3,temperature=.1)
        torch.testing.assert_close(loss,reordered,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(loss,scaled,rtol=2e-5,atol=2e-6)
        equal,_=nt_xent_loss(torch.ones_like(a),torch.ones_like(b),temperature=.1)
        self.assertAlmostEqual(float(equal),math.log(127),places=5)

    def test_recipe_preserves_original_R115_rng_and_complete_R107_supervision(self):
        from retrain import experiment91 as ex
        self.assertTrue(hasattr(ex,'FIT_NTXENT_SMALL_BINDING'),'instance-contrastive candidates missing')
        from retrain.ssl_stage import NTXENT_RECIPE,ssl_rng_derivation,validate_recipe,ssl_phase_completed
        original=json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())
        previous=json.loads((ROOT/'configs/R115-SCIENCE.json').read_text())['ssl_recipe']
        expected=dict(previous,version='fit-ntxent-1600-v1',loss='symmetric_nt_xent',temperature=.1,
                      learning_rng_domain=previous['version'],sampling_rng_domain=previous['version'])
        self.assertEqual(NTXENT_RECIPE,expected)
        for label,seed,order,binding in [('R234',20261029,149,ex.FIT_NTXENT_SMALL_BINDING),
                                       ('R235',20261027,150,ex.FIT_NTXENT_SMALL_SEED27_BINDING)]:
            value=json.loads((ROOT/'configs'/(label+'-SCIENCE.json')).read_text())
            job=copy.deepcopy(original['job']);job['seed']=seed
            self.assertEqual(value['job'],job);self.assertEqual(value['candidate_profile'],binding)
            self.assertEqual(value['candidate']['experiment_order'],order)
            self.assertEqual(value['ssl_recipe'],NTXENT_RECIPE)
            self.assertEqual(ex.candidate_from_configuration(value),value['candidate'])
            for key in ('public_source','paths','data_split','schedule','early_stopping','structure'):
                self.assertEqual(value[key],original[key])
            derived=ssl_rng_derivation(NTXENT_RECIPE,seed,8);prior=ssl_rng_derivation(previous,seed,8)
            for key in ('learning_domain','sampling_domain','projector_seed','model_seeds','sampler_seed','loader_seeds'):
                self.assertEqual(derived[key],prior[key])
            manifest={'fit_indices':list(range(128)),'dev_indices':[128],'audit_indices':[]}
            self.assertEqual(validate_recipe(NTXENT_RECIPE,job,manifest,8),NTXENT_RECIPE)
            with self.assertRaises(ValueError):validate_recipe(dict(NTXENT_RECIPE,temperature=.2),job,manifest,8)
            record=dict(status='completed',recipe=NTXENT_RECIPE,successful_updates=1600,attempted_updates=1600,skipped_updates=0,dev_evaluations=0)
            self.assertTrue(ssl_phase_completed(record,NTXENT_RECIPE))
        # R107's supervision is compared above. Its historical profile belongs
        # to its own frozen runtime; the current runtime rejects it unchanged.
        for label in ('R115','R230','R231','R232','R233'):
            value=json.loads((ROOT/'configs'/(label+'-SCIENCE.json')).read_text())
            self.assertEqual(ex.candidate_from_configuration(value),value['candidate'])
            if value.get('ssl_recipe') is not None:
                validate_recipe(value['ssl_recipe'],value['job'],manifest,8)


if __name__=='__main__':
    import torch
    torch.set_num_threads(max(1,int(os.environ['NCCCU_CPU_THREADS'])//2));torch.set_num_interop_threads(1)
    unittest.main()

"""Native nonpersistent RoPE survives actual EMA updates and selected cold reload."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))


class NonpersistentBufferTest(unittest.TestCase):
    science_file='R184-SCIENCE.json'
    def test_native_rope_real_EMA_copy_blend_freeze_and_cold_reload(self):
        from retrain import age,ema,experiment91 as ex,models,single91
        from retrain.persistence import CheckpointStore,canonical_digest,sha256_file
        science=json.loads((ROOT/'configs'/self.science_file).read_text())
        job=copy.deepcopy(science['job']);job.update(initialization={'kind':'random'},
            age=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]}))
        warm=science['schedule']['warmup_updates']
        rule={'version':f'ema-warmcopy{warm}-decay0999-nobn-v1','warmcopy_updates':warm,
            'decay':.999,'dtype':'torch.float32','floating_buffers':'same_as_parameters','integer_buffers':'copy','batch_norm_buffers':0}
        # Only source/data admission and selection are fixtures; graph, EMA, blob,
        # freeze validation and the inference factory execute their real functions.
        with patch.object(ex,'candidate',return_value=science['candidate']),patch.object(ex,'science_configuration',return_value=science), \
                patch.object(ex,'ema_rule',return_value=rule),patch.object(ex,'binding',return_value={'fixture':'buffer-path-only'}), \
                patch.object(ex,'validate_checkpoint'),patch.object(ex,'validate_runtime'):
            torch.manual_seed(184);model=models.create_model(job,initialize=False)
            normalization=ex.describe_model(model);periods=model.network.rope.periods.clone()
            self.assertEqual(normalization['buffer_count'],1)
            self.assertNotIn('network.rope.periods',model.state_dict())
            shadow=ema.EMA(model,rule);resume=shadow.state_dict();resume.update(update_count=warm-1,copy_count=warm-1,last_update_success=warm-1)
            shadow=ema.EMA(model,rule,state=resume,successful=warm-1)
            with torch.no_grad():model.classifier[-1].bias.fill_(2.)
            shadow.update(model,warm)
            torch.testing.assert_close(shadow.model.classifier[-1].bias,torch.tensor([2.]),rtol=0,atol=0)
            with torch.no_grad():model.classifier[-1].bias.fill_(6.)
            shadow.update(model,warm+1)
            torch.testing.assert_close(shadow.model.classifier[-1].bias,torch.tensor([2.*.999+6.*.001]),rtol=0,atol=3e-7)
            resumed=ema.EMA(model,rule,state=shadow.state_dict(),successful=warm+1)
            with torch.no_grad():model.classifier[-1].bias.fill_(9.)
            shadow.update(model,warm+2);resumed.update(model,warm+2)
            self.assertEqual(ema.tensor_digest(shadow.model.state_dict()),ema.tensor_digest(resumed.model.state_dict()))
            self.assertTrue(torch.equal(periods,shadow.model.network.rope.periods))
            with torch.no_grad():model.network.rope.periods[0].add_(1.)
            with self.assertRaises(ValueError):shadow.update(model,warm+3)
            with torch.no_grad():model.network.rope.periods.copy_(periods)
            bundle=ROOT/'runtime/bundle'
            sources={str(p.relative_to(bundle)):sha256_file(p)for p in sorted([*bundle.glob('*.py'),*(bundle/'retrain').glob('*.py')])}
            identity={'source_digest':canonical_digest(sources),'initialization':job['initialization'],'round_id':'CPU-only-nonpersistent-buffer-fixture'}
            payload={'format_version':1,'mode':'train','cpu_fixture_only':True,'job':job,'model':model.state_dict(),'age':job['age'],
                'identity':identity,'manifest_digest':'synthetic-only','successful_updates':warm+2,'fresh':{},'candidate_normalization':normalization,'ema':shadow.state_dict()}
            selected={'variant':'online','cycle':1,'successful_updates':warm+2,'tensor_sha256':ema.tensor_digest(payload['model']),'auroc':.5,'ap':.1,'error_twice':1}
            with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
                folder=Path(directory);(folder/'complete.json').write_text(json.dumps({'cpu_fixture_only':True}))
                store=CheckpointStore(folder/'checkpoints');store.save(payload,protect=('best_online',))
                with patch.object(ex,'validate_training_completion',return_value=1),patch.object(single91,'collect_dev_entries',return_value=[]), \
                        patch.object(single91,'select_checkpoint',return_value=selected):
                    destination=folder/'freeze.json'
                    frozen=single91.freeze_training(folder,destination,technical_fixture=True)
                    self.assertEqual(frozen['buffer_sha256'],ema.tensor_digest({}))
                    self.assertEqual(frozen['candidate_normalization'],normalization)
                    cold,_,origin,_=single91.load_frozen_model(destination,device='cpu',allow_fixture=True)
                    self.assertEqual(ema.tensor_digest(cold.state_dict()),selected['tensor_sha256'])
                    self.assertTrue(torch.equal(cold.network.rope.periods,periods))
                    self.assertEqual(ex.describe_model(cold),normalization)
                    single91.validate_frozen_checkpoint(frozen,store.load('last'),origin)
            self.assertFalse(torch.cuda.is_initialized())


class ViTBaseNonpersistentBufferTest(NonpersistentBufferTest):
    science_file='R191-SCIENCE.json'


if __name__=='__main__':
    torch.set_num_threads(max(1,min(5,int(os.environ['NCCCU_CPU_THREADS'])-1)));torch.set_num_interop_threads(1)
    unittest.main()

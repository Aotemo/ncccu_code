"""Only a complete selected internal AUROC can admit the one official readout."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex
from retrain import single91


class OfficialGateTest(unittest.TestCase):
    def test_freeze_normalization_is_bound_to_frozen_science(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);checkpoint=root/'fixture.bin';checkpoint.write_bytes(b'CPU-metadata-fixture')
            prediction=root/'dev.csv';prediction.write_text('synthetic metadata only\n')
            precision=root/'dev.precision.json';precision.write_text('{}\n')
            bundle=ROOT/'runtime/bundle'
            sources={str(p.relative_to(bundle)):single91.sha256_file(p)
                for p in sorted([*bundle.glob('*.py'),*(bundle/'retrain').glob('*.py')])}
            binding={'version':'normalization-test'}
            candidate={'model':'siglip2_vitb16','experiment_order':128,'batch_norm_buffers':0}
            entries=[{'cycle':1,'variant':v,'successful_updates':800,'error_twice':e,'ap':.8,'auroc':.98,
                'prediction_path':str(prediction),'prediction_sha256':single91.sha256_file(prediction),
                'precision_path':str(precision),'precision_sha256':single91.sha256_file(precision),
                'tensor_sha256':'1'*64}for v,e in(('online',22),('ema',20))]
            for normalization in('unit','imagenet'):
                science={'job':{'normalization':normalization,'image_size':512,'prepared_size':512},
                    'official_evaluation':ex.OFFICIAL_GATE_0996}
                value={'format':'single-model-9010-official-freeze-v1','cpu_fixture_only':True,
                    'experiment91':binding,'experiment_order':128,'precision':'fp32','view':'identity',
                    'geometry':{'image_size':512,'prepared_size':512},'initialization':{},
                    'identity':{'initialization':{},'source_digest':single91.canonical_digest(sources)},
                    'official_evaluations_allowed':0,'official_metadata_used_for_selection':False,
                    'selected':entries[1],'dev_entries':entries,'training_completion':{},'variant':'ema',
                    'tensor_sha256':'1'*64,'normalization':normalization,'fresh':{},'job':science['job'],
                    'candidate_normalization':{'buffer_sha256':single91.tensor_digest({})},
                    'buffer_sha256':single91.tensor_digest({}),'source_files':sources,
                    'source_digest':single91.canonical_digest(sources),'checkpoint':str(checkpoint),
                    'blob_sha256':single91.sha256_file(checkpoint)}
                path=root/'freeze.json';path.write_text(json.dumps(value))
                with patch.object(ex,'binding',return_value=binding),patch.object(ex,'candidate',return_value=candidate), \
                        patch.object(ex,'science_configuration',return_value=science), \
                        patch.object(ex,'schedule',return_value={'steps_per_cycle':800,'total_updates':800}), \
                        patch.object(ex,'validate_training_completion',return_value=1), \
                        patch.object(ex,'validate_runtime',return_value=True), \
                        patch.object(ex,'validate_normalization',return_value=True):
                    self.assertEqual(single91.validate_freeze(path,allow_fixture=True),value)
                    value['normalization']='imagenet'if normalization=='unit'else'unit'
                    path.write_text(json.dumps(value))
                    with self.assertRaisesRegex(ValueError,'unique single-model freeze differs'):
                        single91.validate_freeze(path,allow_fixture=True)

    def test_selected_internal_auroc_boundary(self):
        policy = dict(ex.OFFICIAL_GATE_0996)
        self.assertFalse(ex.official_allowed_for_selected(policy, {'auroc': 0.995999999}))
        self.assertTrue(ex.official_allowed_for_selected(policy, {'auroc': 0.996}))
        self.assertTrue(ex.official_allowed_for_selected(policy, {'auroc': 1.0}))
        self.assertTrue(ex.official_allowed_for_selected(None, {'auroc': 0.5}))
        for value in ({}, {'auroc': float('nan')}, {'auroc': True}):
            with self.assertRaises(ValueError):
                ex.official_allowed_for_selected(policy, value)

    def test_below_threshold_freeze_cannot_reserve_official(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'official'
            with patch.object(single91, 'validate_freeze', return_value={'official_evaluations_allowed': 0}):
                with self.assertRaisesRegex(ValueError, 'deferred by frozen policy'):
                    single91.reserve_official(Path(directory) / 'freeze.json', output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

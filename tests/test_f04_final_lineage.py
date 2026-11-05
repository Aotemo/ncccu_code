"""CPU tests for a final F04 screening after both training rounds stop."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('f04_gpu_full_dev', ROOT / 'tools/diagnostics/f04_gpu_full_dev.py')
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def safety(name, observed, auroc):
    return {'phase': 'terminal', 'formal_root': '/work/rounds/' + name,
            'observed_at': observed, 'first_success_receipt_read': True,
            'done': {'present': True, 'status': 'completed'}, 'safety': {'safe': True},
            'signals_sent': False, 'model_forward_executed': False, 'official_read': False,
            'selected': {'auroc': auroc},
            'identity': {'owner_expected': {'pid': 100, 'start_ticks': '123'},
                         'ranks_expected': [{'pid': i, 'start_ticks': str(i)} for i in range(201, 209)]}}


class FinalLineageTest(unittest.TestCase):
    def test_final_lineage_options_are_paired_and_forwarded(self):
        self.assertTrue(hasattr(gpu, 'final_lineage_options'), 'final lineage options gate must exist')
        self.assertEqual(gpu.final_lineage_options(None, None), {})
        self.assertEqual(gpu.final_lineage_options(Path('/sealed/lineage.json'), 'a' * 64),
                         {'final-lineage': '/sealed/lineage.json', 'final-lineage-sha256': 'a' * 64})
        for path, digest in [(None, 'a' * 64), (Path('/sealed/lineage.json'), None),
                             (Path('relative'), 'a' * 64), (Path('/sealed/lineage.json'), 'bad')]:
            with self.subTest(path=path, digest=digest), self.assertRaises(ValueError):
                gpu.final_lineage_options(path, digest)

    def test_final_census_follows_both_safe_receipts_and_current_device_proof(self):
        self.assertTrue(hasattr(gpu, 'validate_final_screening'), 'final safety lineage gate must exist')
        now = time.time()
        a = safety('round-a', now - 400, .989)
        b = safety('round-b', now - 300, .987)
        live = safety('round-a', now, .989)
        inventory = {'status': 'read_only_F03_candidate_discovery_no_fusion', 'official_read': False,
                     'at': now - 250, 'members': [
                         {'round_id': 'round-a', 'selected': {'auroc': .989}},
                         {'round_id': 'round-b', 'selected': {'auroc': .987}}]}
        preflight = {'status': 'F04_fp16_size_members_sealed_before_blend', 'official_read': False,
                     'at': now - 240, 'inventory_sha256': 'i', 'old_inventory_sha256': 'old',
                     'fp16': {'members': ['a', 'b', 'c', 'd', 'e'], 'bytes': 495},
                     'eligible_order': ['round-a'], 'excluded': []}
        inputs = {'inventory': {'sha256': 'old'}, 'weight_bytes': 495,
                  'members': [{'round_id': x} for x in ['a', 'b', 'c', 'd', 'e']]}
        gpu.validate_final_screening(preflight, inventory, 'i', [a, b], live, inputs, now)
        invalid = [
            (dict(inventory, at=now - 350), [a, b], live),
            (dict(inventory, members=inventory['members'][:1]), [a, b], live),
            (inventory, [a], live),
            (inventory, [a, b], dict(live, selected={'auroc': .988})),
            (inventory, [a, b], dict(live, identity={'owner_expected': {'pid': 999}, 'ranks_expected': []})),
            (inventory, [dict(a, identity={}), b], dict(live, identity={})),
            (inventory, [a, b], dict(live, observed_at=now - 121)),
        ]
        for current, old, current_safe in invalid:
            with self.subTest(current=current, old=old, live=current_safe), self.assertRaises(ValueError):
                gpu.validate_final_screening(preflight, current, 'i', old, current_safe, inputs, now)
        with self.assertRaises(ValueError):
            gpu.validate_final_screening(dict(preflight, eligible_order=[]), inventory, 'i', [a, b], live, inputs, now)

    def test_lineage_reads_exact_ordered_pinned_rounds_and_rejects_tampering(self):
        self.assertTrue(hasattr(gpu, 'load_final_lineage'), 'pinned final safety lineage loader must exist')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); now = time.time()
            a, b = safety('round-a', now, .989), safety('round-b', now, .987)
            refs = []
            for name, value in [('a', a), ('b', b)]:
                path = root / (name + '.json'); path.write_text(json.dumps(value))
                refs.append({'path': str(path), 'sha256': gpu.sha(path)})
            lineage = root / 'LINEAGE.json'
            lineage.write_text(json.dumps({'status': 'F04_final_two_training_safeties_sealed',
                                           'official_read': False, 'safeties': refs}))
            self.assertEqual(gpu.load_final_lineage(lineage, gpu.sha(lineage), ['round-a', 'round-b']), [a, b])
            with self.assertRaises(ValueError):
                gpu.load_final_lineage(lineage, gpu.sha(lineage), ['round-b', 'round-a'])
            (root / 'a.json').write_text('{}')
            with self.assertRaises(ValueError):
                gpu.load_final_lineage(lineage, gpu.sha(lineage), ['round-a', 'round-b'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

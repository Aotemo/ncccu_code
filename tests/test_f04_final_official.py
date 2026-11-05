"""CPU fixtures for one frozen F04 ensemble and its single official reservation."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tools/diagnostics/f04_final_official.py'
if MODULE.is_file():
    spec = importlib.util.spec_from_file_location('f04_final_official', MODULE)
    final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(final)
else:
    final = None


def frozen_fixture():
    members = []
    for index in range(5):
        members.append({'round_id': 'member-' + str(index),
                        'model': {'path': '/models/' + str(index), 'sha256': str(index) * 64,
                                  'bytes': 99_000_000, 'fp16_tensor_sha256': str(index + 5) * 64},
                        'metadata': {'path': '/metadata/' + str(index), 'sha256': 'a' * 64},
                        'normalization': {'mean': float(index), 'std': float(index + 1)}})
    return {'format': 'f04-final-frozen-v1', 'status': 'F04_ensemble_final_frozen',
            'official_evaluations_allowed': 1, 'official_metadata_used_for_selection': False,
            'TTA': False, 'weight_bytes': 495_000_000, 'members': members,
            'official': {'manifest_path': '/data/official-manifest.json', 'manifest_sha256': 'c' * 64,
                         'data_root': '/data'},
            'full_dev': {'count': 5140, 'positive': 167, 'fp32_auroc': .9902,
                         'fp16_auroc': .9901, 'fp16_ap': .84,
                         'official_read': False, 'summary_sha256': 'b' * 64}}


class FinalOfficialTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(final, 'F04 final official adapter must exist before official evaluation')

    def test_freeze_rejects_over_budget_invalid_normalization_and_unreviewed_drop(self):
        value = frozen_fixture()
        final.validate_final_freeze(value)
        for change in ('size', 'duplicates', 'nan', 'drop', 'tta', 'official_selection'):
            altered = json.loads(json.dumps(value))
            if change == 'size':
                altered['members'][0]['model']['bytes'] = 105_000_001
                altered['weight_bytes'] = 501_000_001
            elif change == 'duplicates':
                altered['members'][1]['round_id'] = altered['members'][0]['round_id']
            elif change == 'nan':
                altered['members'][1]['normalization']['std'] = math.nan
            elif change == 'drop':
                altered['full_dev']['fp16_auroc'] = .9896
            elif change == 'tta':
                altered['TTA'] = True
            else:
                altered['official_metadata_used_for_selection'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                final.validate_final_freeze(altered)

    def test_only_canonical_frozen_directory_can_claim_and_claim_is_once_only(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            parent = work / 'final/F04-20261104'
            parent.mkdir(parents=True)
            freeze = parent / 'FREEZE.json'
            freeze.write_text(json.dumps(frozen_fixture()))
            digest = final.sha(freeze)
            marker = final.reserve_official_once(freeze, digest, work, parent / 'results')
            self.assertEqual(marker, parent / 'OFFICIAL-ONCE.json')
            with self.assertRaises(FileExistsError):
                final.reserve_official_once(freeze, digest, work, parent / 'other-output')
            copied = work / 'copy.json'
            copied.write_bytes(freeze.read_bytes())
            with self.assertRaises(ValueError):
                final.reserve_official_once(copied, digest, work, parent / 'results')

    def test_official_ensemble_uses_frozen_dev_moments_without_refitting(self):
        value = frozen_fixture()
        vectors = {item['round_id']: [item['normalization']['mean'],
                   item['normalization']['mean'] + 2 * item['normalization']['std']]
                   for item in value['members']}
        self.assertEqual(final.ensemble_with_frozen_moments(value, vectors), [0.0, 2.0])
        vectors.pop('member-0')
        with self.assertRaises(ValueError):
            final.ensemble_with_frozen_moments(value, vectors)

    def test_official_read_cannot_start_before_the_bound_reservation(self):
        self.assertTrue(hasattr(final, 'read_official_inputs_once'), 'official reader gate must exist')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); parent = work / 'final/F04-20261104'; parent.mkdir(parents=True)
            freeze = parent / 'FREEZE.json'
            calls = []
            def reader(root, split):
                calls.append((root, split))
                return records, 'b' * 64
            rows = [{'name': f'image-{i}', 'label': int(i < 191), 'age': '50',
                     'relative_path': f'valid/image-{i}', 'sha256': 'a' * 64} for i in range(5849)]
            records = [{k: row[k] for k in ('name', 'label', 'age')} for row in rows]
            manifest = parent / 'manifest.json'
            manifest.write_text(json.dumps({'training_eligible': False, 'rows': rows,
                                            'source_identity': {'valid_csv_sha256': 'b' * 64}}))
            value = frozen_fixture()
            value['official'] = {'manifest_path': str(manifest), 'manifest_sha256': final.sha(manifest),
                                 'data_root': str(work)}
            freeze.write_text(json.dumps(value))
            digest = final.sha(freeze); output = parent / 'results'
            with self.assertRaisesRegex(ValueError, 'reservation'):
                final.read_official_inputs_once(freeze, digest, work, output, manifest, final.sha(manifest), work, reader)
            self.assertEqual(calls, [])
            final.reserve_official_once(freeze, digest, work, output)
            result = final.read_official_inputs_once(freeze, digest, work, output, manifest, final.sha(manifest), work, reader)
            self.assertEqual(len(result['rows']), 5849)
            self.assertEqual(result['csv_sha256'], 'b' * 64)
            self.assertEqual(calls, [(work, 'valid')])
            with self.assertRaises(FileExistsError):
                final.read_official_inputs_once(freeze, digest, work, output, manifest, final.sha(manifest), work, reader)
            self.assertEqual(calls, [(work, 'valid')])
            with self.assertRaises(ValueError):
                final.read_official_inputs_once(freeze, digest, work, parent / 'other', manifest, final.sha(manifest), work, reader)

    def test_final_freeze_is_built_only_from_the_verified_final_dev_and_preserves_members(self):
        self.assertTrue(hasattr(final, 'freeze_from_verified_dev'), 'verified-dev freeze builder must exist')
        base = frozen_fixture(); members = base['members']
        for item in members:
            item.update(selected={'variant': 'ema', 'cycle': 2, 'successful_updates': 1600,
                                  'tensor_sha256': 'd' * 64},
                        freeze={'path': '/historical/freeze.json', 'sha256': 'e' * 64},
                        bundle='/historical/bundle', data_root='/data')
        ids = [item['round_id'] for item in members]
        inputs = {'status': 'F04_full_dev_gpu_inputs_sealed_no_GPU', 'members': members,
                  'weight_bytes': base['weight_bytes'], 'official_read': False, 'TTA': False}
        preflight = {'status': 'F04_fp16_size_members_sealed_before_blend',
                     'fp16': {'members': ids, 'bytes': base['weight_bytes']}, 'official_read': False}
        norms = {item['round_id']: item['normalization'] for item in members}
        summary = {'status': 'F04_full_internal_dev_fp16_storage_evaluated',
                   'inputs_sha256': '1' * 64, 'current_preflight_sha256': '2' * 64,
                   'final_lineage_sha256': '3' * 64, 'weight_bytes': base['weight_bytes'],
                   'dev_count': 5140, 'dev_positive': 167, 'TTA': False, 'official_read': False,
                   'training_executed': False, 'supervisor': {'status': 'completed'},
                   'released_ranks': [{'pid': 100 + i, 'start_ticks': str(100 + i), 'inactive': True}
                                      for i in range(8)], 'device_handles_after': [],
                   'resources_after': {'cards': [{'busy_percent': 0, 'vram_used_bytes': 0} for _ in range(8)]},
                   'comparison': {'fp32': {'metrics': {'auroc': .9902, 'ap': .83}},
                                  'fp16': {'metrics': {'auroc': .9901, 'ap': .84},
                                           'normalization': norms}}}
        identity = {'inputs_sha256': '1' * 64, 'preflight_sha256': '2' * 64,
                    'lineage_sha256': '3' * 64, 'summary_sha256': '4' * 64}
        runtime = {'code_commit': 'a' * 40, 'code_tree': 'b' * 40,
                   'gpu_owner_sha256': 'c' * 64, 'official_adapter_sha256': 'd' * 64,
                   'python_version': '3.12.7', 'torch_version': '2.10.0', 'pillow_version': '12.3.0'}
        frozen = final.freeze_from_verified_dev(inputs, preflight, summary, identity, base['official'], runtime)
        self.assertEqual([row['round_id'] for row in frozen['members']], ids)
        self.assertEqual(frozen['members'][0]['normalization'], norms[ids[0]])
        self.assertEqual(frozen['members'][0]['selected']['tensor_sha256'], 'd' * 64)
        self.assertEqual(frozen['full_dev']['summary_sha256'], '4' * 64)
        for altered in (dict(summary, final_lineage_sha256='0' * 64),
                        dict(summary, released_ranks=[]), dict(summary, official_read=True),
                        dict(summary, comparison={**summary['comparison'],
                             'fp16': {'metrics': {'auroc': .9890, 'ap': .84}, 'normalization': norms}})):
            with self.assertRaises(ValueError):
                final.freeze_from_verified_dev(inputs, preflight, altered, identity, base['official'], runtime)
        with self.assertRaises(ValueError):
            final.freeze_from_verified_dev(inputs, dict(preflight, fp16={'members': ids[::-1], 'bytes': base['weight_bytes']}),
                                           summary, identity, base['official'], runtime)

    def test_official_population_must_match_sealed_csv_and_full_class_counts(self):
        rows = [{'name': f'image-{i}', 'label': int(i < 191), 'age': '50',
                 'relative_path': f'valid/image-{i}', 'sha256': 'a' * 64} for i in range(5849)]
        records = [{k: row[k] for k in ('name', 'label', 'age')} for row in rows]
        manifest = {'training_eligible': False, 'rows': rows,
                    'source_identity': {'valid_csv_sha256': 'b' * 64}}
        self.assertEqual(len(final.validate_official_population(manifest, records, 'b' * 64)), 5849)
        changed = dict(manifest, rows=rows[:-1])
        with self.assertRaises(ValueError):
            final.validate_official_population(changed, records, 'b' * 64)
        changed = dict(manifest, training_eligible=True)
        with self.assertRaises(ValueError):
            final.validate_official_population(changed, records, 'b' * 64)
        altered = list(records); altered[0] = dict(records[0], label=0)
        with self.assertRaises(ValueError):
            final.validate_official_population(manifest, altered, 'b' * 64)


if __name__ == '__main__':
    unittest.main(verbosity=2)

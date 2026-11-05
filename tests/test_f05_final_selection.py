"""Synthetic F05 candidate ordering, strict export and fusion checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/diagnostics'), str(ROOT / 'runtime/bundle')]
from f04_size_ensemble import export_state_files, compare_predictions

SOURCE = ROOT / 'tools/diagnostics/f05_final_selection.py'
if SOURCE.is_file():
    spec = importlib.util.spec_from_file_location('f05_final_selection', SOURCE)
    final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(final)
else:
    final = None


def file_ref(path):
    raw = Path(path).read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    return file_ref(path)


def example_export(root, variant):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)
            self.age_weight = torch.nn.Parameter(torch.tensor([.1, -.2]))
            self.register_buffer('index', torch.tensor([1, 2], dtype=torch.int64))
    root = Path(root)
    model = Toy().eval()
    exported = export_state_files(model, root)
    checkpoint = root / 'checkpoint.pt'
    torch.save({'toy_only': True}, checkpoint)
    checkpoint_ref = file_ref(checkpoint)
    sidecar = write_json(root / 'checkpoint.pt.json', {'sha256': checkpoint_ref['sha256']})
    selected = {'auroc': .990, 'variant': variant, 'tensor_sha256': 'b' * 64,
                'successful_updates': 800}
    public_source = {'sha256': 'a' * 64, 'training_data': 'general'}
    freeze = write_json(root / 'freeze.json', {'cpu_fixture_only': True,
        'selected': selected, 'blob_sha256': checkpoint_ref['sha256']})
    science = write_json(root / 'science.json', {'public_source': public_source,
        'candidate': {'weight_sha256': public_source['sha256']}, 'job': {'initial_checkpoint': None}})
    candidate = {'round_id': 'synthetic-round', 'status': 'prediction_verified_weight_pending',
                 'selected': selected, 'public_source': public_source,
                 'selected_blob_sha256': checkpoint_ref['sha256'],
                 'checkpoint_candidates': [{**checkpoint_ref, 'sidecar': sidecar}],
                 'spec': {'freeze': freeze}, 'science': science}
    receipt = {'status': 'F04_complete_inference_state_exported_and_strictly_reloaded',
               'inventory': {'sha256': 'e' * 64}, 'round_id': candidate['round_id'],
               'selected': candidate['selected'], 'selected_blob_sha256': candidate['selected_blob_sha256'],
               'freeze': candidate['spec']['freeze'], 'science': candidate['science'],
               'checkpoint': candidate['checkpoint_candidates'][0], 'exports': exported,
               'model_constructed': True, 'forward_calls': 0, 'GPU_initialized': False, 'official_read': False}
    return candidate, write_json(root / 'EXPORT.json', receipt)


def example_soup(root):
    """Fabricate a soup, its source manifests and full small dev rank packets."""
    root = Path(root)
    bundle, cache = root / 'bundle', root / 'cache'
    bundle.mkdir(); cache.mkdir()
    (bundle / 'fake.py').write_text('# fixture only\n')
    freeze = write_json(root / 'soup-freeze.json', {'source_files': {'fake.py': file_ref(bundle / 'fake.py')['sha256']}})
    site = write_json(root / 'soup-site.json', {'work_root': str(root), 'data_root': str(root / 'data'),
                                               'prepared_cache': str(cache)})
    parents = []
    for name in ('r-a', 'r-b'):
        directory = root / 'rounds' / name
        directory.mkdir(parents=True)
        science = write_json(directory / 'SCIENCE.json', {'public_source': {'sha256': 'a' * 64,
            'training_data': 'general'}, 'candidate': {'weight_sha256': 'a' * 64},
            'job': {'initial_checkpoint': None}})
        parents.append({'round_id': name, 'science_sha256': science['sha256']})
    exports = {}
    for precision in ('fp32', 'fp16'):
        directory = root / precision
        directory.mkdir()
        model = directory / 'model.pt'
        torch.save({'w': torch.tensor([1., 2.], dtype=torch.float32 if precision == 'fp32' else torch.float16)}, model)
        model_ref = {**file_ref(model), 'bytes': model.stat().st_size, 'tensor_sha256': 'b' * 64}
        metadata = write_json(directory / 'model.json', {'model_sha256': model_ref['sha256'],
            'checkpoint': {'variant': 'ema', 'tensor_sha256': 'b' * 64}})
        exports[precision] = {'model': model_ref, 'metadata': metadata}
    common = {'freeze': freeze, 'site': site, 'bundle': str(bundle), 'data_root': str(root / 'data'),
              'cache': str(cache)}
    standalone = {'status': 'uniform_soup_standalone_strict_synthetic_passed', 'base_variant_for_loader': 'ema',
                  'state_sha256': exports['fp32']['model']['sha256'], 'exports': exports,
                  'synthetic_forward_only': True, 'official_read': False, 'dev_read': False,
                  'GPU_initialized': False, 'TTA': False, **common}
    standalone_ref = write_json(root / 'STANDALONE.json', standalone)
    soup = {'status': 'uniform_soup_created_no_dev_evaluation', 'method': 'uniform_all_compatible_selected_states_v1',
            'group': 'r120', 'parents': parents, 'standalone': standalone_ref,
            'average_state_sha256': standalone['state_sha256'], 'official_read': False, 'dev_read': False,
            'GPU_initialized': False, 'TTA': False}
    member = {'round_id': 'r120', 'soup': write_json(root / 'SOUP.json', soup), 'receipt': standalone_ref,
              'exports': exports, 'selected': {'variant': 'ema'}, **common}
    rows = [{'name': 'row-' + str(i), 'label': label} for i, label in enumerate((1, 0, 1, 0))]
    manifest = write_json(root / 'manifest.json', {'rows': rows, 'dev_indices': [0, 1, 2, 3]})
    plan = [{'device_rank': rank, 'member': 'r120', 'rank': rank, 'world': 8} for rank in range(8)]
    inputs = {'fixture_only': True, 'members': [member], 'plan': plan, 'manifest': manifest,
              'batch_size': 8, 'official_read': False, 'TTA': False}
    inputs_ref = write_json(root / 'inputs.json', inputs)
    output = root / 'evaluation'; output.mkdir()
    (output / 'proofs').mkdir(); (output / 'packets').mkdir()
    worker = write_json(output / 'WORKER-SPEC.json', {'inputs': inputs_ref, 'soup_mode': True})
    proofs = [write_json(output / 'proofs' / ('rank-' + str(rank) + '.json'), {'rank': rank, 'fixture_only': True})
              for rank in range(8)]
    write_json(output / 'RELEASE.json', {'status': 'all_rank_device_proofs_passed',
        'spec_sha256': worker['sha256'], 'proof_sha256': {str(i): item['sha256'] for i, item in enumerate(proofs)}})
    values = {'fp32': [3., 0., 2., -1.], 'fp16': [3.01, 0., 1.99, -1.]}
    packet_sha = {}
    for precision in ('fp32', 'fp16'):
        for rank in range(8):
            records = []
            if rank == 0:
                for index, value in enumerate(values[precision]):
                    records.append({'index': index, 'rank': rank, 'batch': 0, 'offset': index,
                                    'name': rows[index]['name'], 'label': rows[index]['label'],
                                    'logit': value, 'probability': 0.5})
            packet = {'device_rank': rank, 'member': 'r120', 'rank': rank, 'error': None,
                      'storage_precision': precision, 'model_sha256': exports[precision]['model']['sha256'],
                      'input_sha256': inputs_ref['sha256'], 'proof_sha256': proofs[rank]['sha256'],
                      'TTA': False, 'official_read': False, 'records': records}
            packet_ref = write_json(output / 'packets' / ('rank-' + str(rank) + '-' + precision + '.json'), packet)
            packet_sha[str(rank) + '-' + precision] = packet_ref['sha256']
    comparison = compare_predictions([r['label'] for r in rows], values['fp32'], values['fp16'])
    summary = {'status': 'F05_uniform_soups_full_internal_dev_evaluated', 'fixture_only': True,
               'inputs_sha256': inputs_ref['sha256'], 'TTA': False, 'official_read': False,
               'training_executed': False, 'comparison': {'soups': {'r120': comparison}, 'packet_sha256': packet_sha}}
    return {'id': 'soup-r120'}, {'inputs': inputs_ref, 'summary': write_json(output / 'SUMMARY.json', summary)}


class F05FinalSelectionTests(unittest.TestCase):
    def test_selection_module_exists(self):
        self.assertIsNotNone(final, 'F05 final candidate selector not implemented')

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_byte_boundary_ties_and_first_over_limit_stops(self):
        candidates = [
            {'id': 'later', 'order': 2, 'auroc': .989, 'fp16_bytes': 1, 'fp32_bytes': 1},
            {'id': 'second', 'order': 1, 'auroc': .990, 'fp16_bytes': 200_000_000, 'fp32_bytes': 3},
            {'id': 'first', 'order': 0, 'auroc': .990, 'fp16_bytes': 300_000_000, 'fp32_bytes': 4},
        ]
        result = final.select_by_fixed_budget(candidates)
        self.assertEqual(result, {'members': ['first', 'second'], 'bytes': 500_000_000,
                                  'stopped_before': 'later'})
        candidates[1]['fp16_bytes'] += 1
        result = final.select_by_fixed_budget(candidates)
        self.assertEqual(result, {'members': ['first'], 'bytes': 300_000_000, 'stopped_before': 'second'})
        boundary = final.select_by_fixed_budget([dict(candidates[0], auroc=.988)])
        self.assertEqual(boundary['members'], ['later'])
        with self.assertRaises(ValueError):
            final.select_by_fixed_budget([dict(candidates[0], auroc=.987999)])
        with self.assertRaises(ValueError):
            final.select_by_fixed_budget([dict(candidates[0], auroc=float('nan'))])

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_original_and_soup_discovery_is_fixed_and_complete(self):
        rounds = {'status': 'read_only_F03_candidate_discovery_no_fusion', 'official_read': False,
                  'members': [{'round_id': 'r1'}, {'round_id': 'r2'}]}
        soups = [{'id': 'soup-r120'}, {'id': 'soup-r134r1'}, {'id': 'soup-r201'}]
        order = ['r1', 'r2', 'soup-r120', 'soup-r134r1', 'soup-r201']
        result = final.build_candidate_inventory(rounds, soups, order)
        self.assertEqual([row['id'] for row in result], order)
        with self.assertRaises(ValueError):
            final.build_candidate_inventory(rounds, soups, order[:-1])
        with self.assertRaises(ValueError):
            final.build_candidate_inventory(rounds, soups + [soups[0]], order)

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_soup_parent_and_actual_dev_packets_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            row, evidence = example_soup(directory)
            candidate = final.verify_soup_export(row, evidence, approved={'a' * 64: 'general'}, allow_fixture=True)
            self.assertEqual(candidate['id'], 'soup-r120')
            self.assertEqual(candidate['auroc'], 1.)
            packet = Path(evidence['summary']['path']).parent / 'packets/rank-0-fp32.json'
            packet.write_text('{}')
            with self.assertRaises(ValueError):
                final.verify_soup_export(row, evidence, approved={'a' * 64: 'general'}, allow_fixture=True)

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_actual_strict_export_online_ema_and_missing_weight(self):
        for variant in ('online', 'ema'):
            with tempfile.TemporaryDirectory() as directory:
                row, receipt = example_export(directory, variant)
                actual = final.verify_round_export(row, receipt, inventory_sha='e' * 64,
                                                   approved={'a' * 64: 'general'}, allow_fixture=True)
                self.assertEqual(actual['variant'], variant)
                self.assertGreater(actual['fp16_bytes'], 0)
                wrong = copy.deepcopy(row)
                wrong['selected']['variant'] = 'ema' if variant == 'online' else 'online'
                with self.assertRaises(ValueError):
                    final.verify_round_export(wrong, receipt, inventory_sha='e' * 64,
                                              approved={'a' * 64: 'general'}, allow_fixture=True)
                wrong_source = copy.deepcopy(row)
                wrong_source['public_source']['training_data'] = 'medical'
                with self.assertRaises(ValueError):
                    final.verify_round_export(wrong_source, receipt, inventory_sha='e' * 64,
                                              approved={'a' * 64: 'general'}, allow_fixture=True)
                record = json.loads(Path(receipt['path']).read_text())
                Path(record['exports']['fp16']['path']).unlink()
                with self.assertRaises(ValueError):
                    final.verify_round_export(row, receipt, inventory_sha='e' * 64,
                                              approved={'a' * 64: 'general'}, allow_fixture=True)

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_missing_evidence_stays_unknown_and_cannot_freeze_a_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            row, receipt = example_export(directory, 'ema')
            missing = copy.deepcopy(row)
            missing['round_id'] = 'another-round'
            candidates = [{'id': row['round_id'], 'kind': 'round', 'order': 0, 'discovery': row},
                          {'id': missing['round_id'], 'kind': 'round', 'order': 1, 'discovery': missing}]
            screen = final.screen_candidates(candidates, {row['round_id']: receipt},
                inventory_sha='e' * 64, soup_catalog_sha='d' * 64,
                approved={'a' * 64: 'general'}, allow_fixture=True)
            self.assertEqual([x['id'] for x in screen['verified']], [row['round_id']])
            self.assertEqual([x['id'] for x in screen['unknown']], [missing['round_id']])
            with self.assertRaises(ValueError):
                final.freeze_selection(screen)
            missing['status'] = 'selected_checkpoint_missing'
            missing['checkpoint_candidates'] = []
            screen = final.screen_candidates(candidates, {row['round_id']: receipt},
                inventory_sha='e' * 64, soup_catalog_sha='d' * 64,
                approved={'a' * 64: 'general'}, allow_fixture=True)
            self.assertFalse(screen['unknown'])
            self.assertEqual(screen['excluded'][0]['reason'], 'selected_checkpoint_missing')
            for status in ('unknown_or_excluded', 'selected_checkpoint_ambiguous'):
                missing['status'] = status
                missing['selected']['auroc'] = .987
                screen = final.screen_candidates(candidates, {row['round_id']: receipt},
                    inventory_sha='e' * 64, soup_catalog_sha='d' * 64,
                    approved={'a' * 64: 'general'}, allow_fixture=True)
                self.assertEqual([item['id'] for item in screen['unknown']], [missing['round_id']])
            missing['status'] = 'selected_checkpoint_missing'
            missing['checkpoint_candidates'] = [{'path': '/contradiction'}]
            screen = final.screen_candidates(candidates, {row['round_id']: receipt},
                inventory_sha='e' * 64, soup_catalog_sha='d' * 64,
                approved={'a' * 64: 'general'}, allow_fixture=True)
            self.assertEqual(screen['unknown'][0]['reason'], 'missing_checkpoint_inventory_inconsistent')
            missing['checkpoint_candidates'] = []
            missing['selected']['tensor_sha256'] = None
            screen = final.screen_candidates(candidates, {row['round_id']: receipt},
                inventory_sha='e' * 64, soup_catalog_sha='d' * 64,
                approved={'a' * 64: 'general'}, allow_fixture=True)
            self.assertEqual(screen['unknown'][0]['reason'], 'missing_checkpoint_inventory_inconsistent')

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_task_fit_adaptation_needs_its_own_matching_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, receipt = example_export(root, 'ema')
            science_path = Path(row['science']['path'])
            science = json.loads(science_path.read_text()); science['ssl_recipe'] = {'fixture': True}
            row['science'] = write_json(science_path, science)
            exported = json.loads(Path(receipt['path']).read_text()); exported['science'] = row['science']
            receipt = write_json(Path(receipt['path']), exported)
            with self.assertRaises(ValueError):
                final.verify_round_export(row, receipt, inventory_sha='e' * 64,
                                          approved={'a' * 64: 'general'}, allow_fixture=True)
            stage = write_json(root / 'SSL-RESULT.json', {'fixture_only': True})
            audit = write_json(root / 'AUDIT.json', {'status': 'historic_fit_ssl_provenance_audit', 'rounds': [{
                'round': 'SYNTHETIC', 'status': 'evidence_supports_competition_fit_only',
                'fit_indices_only': True, 'dev_evaluations': 0, 'manifest_sha256': final.BASE_MANIFEST_SHA,
                'public_weight_sha256': 'a' * 64, 'receipt_sha256': stage['sha256']}]})
            result = final.verify_round_export(row, receipt, inventory_sha='e' * 64,
                approved={'a' * 64: 'general'}, adaptation_audit=audit, adaptation_receipt=stage, allow_fixture=True)
            self.assertEqual(result['id'], row['round_id'])

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_prior_export_inventory_must_match_the_current_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, receipt = example_export(root, 'ema')
            old = write_json(root / 'old-inventory.json', {'status': 'read_only_F03_candidate_discovery_no_fusion',
                                                           'official_read': False, 'members': [copy.deepcopy(row)]})
            export = json.loads(Path(receipt['path']).read_text()); export['inventory']['sha256'] = old['sha256']
            receipt = write_json(Path(receipt['path']), export)
            candidates = [{'id': row['round_id'], 'kind': 'round', 'order': 0, 'discovery': row}]
            evidence = {row['round_id']: {'export': receipt, 'original_inventory': old}}
            screen = final.screen_candidates(candidates, evidence, inventory_sha='f' * 64,
                                             soup_catalog_sha='d' * 64, approved={'a' * 64: 'general'}, allow_fixture=True)
            self.assertEqual([item['id'] for item in screen['verified']], [row['round_id']])
            altered = copy.deepcopy(row); altered['selected']['variant'] = 'online'
            candidates[0]['discovery'] = altered
            screen = final.screen_candidates(candidates, evidence, inventory_sha='f' * 64,
                                             soup_catalog_sha='d' * 64, approved={'a' * 64: 'general'}, allow_fixture=True)
            self.assertEqual([item['id'] for item in screen['unknown']], [row['round_id']])

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_sealed_policy_binds_candidate_order_and_terminal_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, receipt = example_export(directory, 'online')
            candidates = [{'id': row['round_id'], 'kind': 'round', 'order': 0, 'discovery': row}]
            screen = final.screen_candidates(candidates, {row['round_id']: receipt},
                inventory_sha='e' * 64, soup_catalog_sha='d' * 64,
                approved={'a' * 64: 'general'}, allow_fixture=True)
            directions = {}
            for direction in ('soup', 'oof', 'clean', 'distill', 'resolution_768'):
                directions[direction] = write_json(root / (direction + '.json'),
                    {'direction': direction, 'status': 'completed', 'official_read': False})
            policy = {'format': 'f05-final-selection-policy-v1', 'status': 'registered_before_new_f05_fusion',
                      'candidate_order': [row['round_id']], 'inventory_sha256': 'e' * 64,
                      'soup_catalog_sha256': 'd' * 64,
                      'threshold': .988, 'byte_limit': 500_000_000,
                      'ordering': 'selected_auroc_desc_then_sealed_order',
                      'overflow': 'stop_before_first_over_limit', 'normalization': 'dev_float64_zscore_ddof0',
                      'weights': 'equal', 'TTA': False, 'official_read': False,
                      'fusion_scores_read': False, 'direction_terminals': directions}
            policy_ref = write_json(root / 'POLICY.json', policy)
            fixed = final.freeze_selection(screen, policy_ref=policy_ref)
            self.assertEqual(fixed['fp16']['members'], [row['round_id']])
            policy['candidate_order'] = ['other']
            wrong = write_json(root / 'wrong-policy.json', policy)
            with self.assertRaises(ValueError):
                final.freeze_selection(screen, policy_ref=wrong)

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_equal_z_internal_dev_and_rejects_zero_variance(self):
        result = final.fuse_internal_dev([1, 0, 1, 0], {'a': [3, 1, 2, 0], 'b': [2, 0, 1, -1]}, ['a', 'b'])
        self.assertEqual(result['metrics']['auroc'], 1.)
        with self.assertRaises(ValueError):
            final.fuse_internal_dev([1, 0, 1, 0], {'a': [1, 1, 1, 1]}, ['a'])

if __name__ == '__main__':
    paths = ('tools/diagnostics/f05_final_selection.py', 'tests/test_f05_final_selection.py',
             'tools/diagnostics/f04_size_ensemble.py', 'tools/diagnostics/f04_bounded_fusion.py',
             'tools/diagnostics/ensemble_dev.py', 'tools/diagnostics/f05_soup_dev.py',
             'tools/diagnostics/f04_gpu_full_dev.py')
    present = {name: file_ref(ROOT / name)['sha256'] for name in paths if (ROOT / name).is_file()}
    print(json.dumps({'source_sha256': present}), flush=True)
    unittest.main()

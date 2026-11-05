"""Synthetic F05 candidate ordering, strict export and fusion checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/diagnostics'), str(ROOT / 'runtime/bundle'), str(ROOT / 'tests')]
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


def example_soup(root, manifest=None):
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
    if manifest is None:
        rows = [{'name': 'row-' + str(i), 'label': label} for i, label in enumerate((1, 0, 1, 0))]
        manifest = write_json(root / 'manifest.json', {'rows': rows, 'dev_indices': [0, 1, 2, 3]})
    else:
        population = json.loads(Path(manifest['path']).read_text())
        if population['dev_indices'] != [0, 1, 2, 3]:
            raise ValueError('small fixture expects its dev prefix')
        rows = [population['rows'][i] for i in population['dev_indices']]
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


def example_selected_dev(root):
    """Fake selected, soup, original fp32 references and new fp16 shards."""
    root = Path(root)
    rows = [{'name': 'dev-' + str(i), 'label': label, 'relative_path': 'train/dev-' + str(i)}
            for i, label in enumerate((1, 0, 1, 0, 0))]
    manifest = write_json(root / 'manifest.json', {'fixture_only': True, 'rows': rows,
                                                    'dev_indices': [0, 1, 2, 3], 'fit_indices': [4]})
    round_root = root / 'rounds' / 'a'
    prediction_dir = round_root / 'predictions' / 'online'; prediction_dir.mkdir(parents=True)
    prediction_path = prediction_dir / 'dev-cycle-0001.csv'
    prediction_path.write_text('name,label,logit\n' + ''.join(
        rows[i]['name'] + ',' + str(rows[i]['label']) + ',' + str(value) + '\n'
        for i, value in enumerate((3., 0., 2., -1.))))
    prediction = file_ref(prediction_path)
    precision = write_json(Path(str(prediction_path) + '.precision.json'),
        {'variant': 'online', 'view': 'identity', 'actual_tensor_sha256': 'c' * 64,
         'state_identity': {'successful_updates': 800}})
    selected = {'variant': 'online', 'cycle': 1, 'successful_updates': 800, 'tensor_sha256': 'c' * 64,
                'prediction_path': prediction['path'], 'prediction_sha256': prediction['sha256'],
                'precision_path': precision['path'], 'precision_sha256': precision['sha256'],
                'auroc': 1., 'ap': 1., 'error_twice': 0}
    freeze = write_json(round_root / 'SINGLE-MODEL-FREEZE.json',
        {'cpu_fixture_only': False, 'precision': 'fp32', 'view': 'identity', 'selected': selected,
         'blob_sha256': 'd' * 64, 'experiment91': {'manifest_sha256': manifest['sha256']}})
    completion = write_json(round_root / 'TRAIN_RESULT.json', {'status': 'completed', 'successful_updates': 800})
    row = {'round_id': 'a', 'selected': selected, 'spec': {'freeze': freeze, 'completion': completion}}

    soup_root = root / 'soup'; soup_root.mkdir()
    soup_row, soup_evidence = example_soup(soup_root, manifest)
    soup_inputs = json.loads(Path(soup_evidence['inputs']['path']).read_text())
    soup_member = soup_inputs['members'][0]
    model_path = root / 'a-fp16.pt'; model_path.write_bytes(b'fixture-a')
    model_ref = {**file_ref(model_path), 'bytes': model_path.stat().st_size,
                 'fp16_tensor_sha256': 'a' * 64}
    metadata = write_json(root / 'a-model.json', {'model_sha256': model_ref['sha256'],
        'checkpoint': {'variant': 'online', 'tensor_sha256': 'a' * 64}})
    members = [
        {'id': 'a', 'round_id': 'a', 'model': model_ref, 'metadata': metadata, 'selected': {'variant': 'online'}},
        {'id': 'soup-r120', 'round_id': 'r120', 'model': soup_member['exports']['fp16']['model'],
         'metadata': soup_member['exports']['fp16']['metadata'], 'selected': {'variant': 'ema'}},
    ]
    verified = [
        {'id': 'a', 'auroc': 1., 'variant': 'online', 'tensor_sha256': 'c' * 64, 'successful_updates': 800,
         'selected_blob_sha256': 'd' * 64, 'fp16_sha256': model_ref['sha256'], 'fp16_bytes': model_ref['bytes']},
        {'id': 'soup-r120', 'auroc': 1., 'variant': 'ema',
         'fp16_sha256': members[1]['model']['sha256'], 'fp16_bytes': members[1]['model']['bytes'],
         'soup_receipt_sha256': soup_member['soup']['sha256'], 'dev_summary_sha256': soup_evidence['summary']['sha256']},
    ]
    inventory = write_json(root / 'inventory.json', {'status': 'read_only_F03_candidate_discovery_no_fusion',
        'official_read': False, 'manifest': manifest, 'members': [row]})
    catalog = write_json(root / 'catalog.json', {'format': 'f05-soup-candidates-v1',
        'official_read': False, 'soups': [soup_row]})
    evidence = write_json(root / 'evidence.json', {'format': 'f05-candidate-evidence-v1',
        'official_read': False, 'entries': {'soup-r120': soup_evidence}})
    screen = write_json(root / 'screen.json', {'inventory': inventory, 'soup_catalog': catalog,
                                                'evidence_manifest': evidence})
    preflight = write_json(root / 'PREFLIGHT.json', {'fixture_only': True,
        'status': 'F05_candidates_fixed_before_fusion', 'official_read': False, 'TTA': False,
        'inventory_sha256': inventory['sha256'], 'soup_catalog_sha256': catalog['sha256'],
        'screen_sha256': screen['sha256'],
        'fp16': {'members': ['a', 'soup-r120'], 'bytes': sum(item['fp16_bytes'] for item in verified),
                 'stopped_before': None}, 'verified': verified})
    plan = [{'device_rank': rank, 'member': 'a' if rank < 4 else 'r120',
             'rank': rank if rank < 4 else rank - 4, 'world': 4} for rank in range(8)]
    inputs = {'format': 'f05-selected-dev-inputs-v1', 'status': 'selected_fp16_dev_inputs_sealed',
              'fixture_only': True, 'preflight_sha256': preflight['sha256'], 'manifest': manifest,
              'members': members, 'plan': plan, 'batch_size': 8, 'view': 'identity', 'model_mode': 'eval',
              'dev_count': 4, 'dev_positive': 2, 'TTA': False, 'official_read': False}
    inputs_ref = write_json(root / 'INPUTS.json', inputs)
    spec = write_json(root / 'WORKER-SPEC.json', {'fixture_only': True, 'inputs': inputs_ref})
    proofs = [write_json(root / ('proof-' + str(rank) + '.json'), {'fixture_only': True, 'rank': rank})
              for rank in range(8)]
    release = write_json(root / 'RELEASE.json', {'status': 'all_rank_device_proofs_passed',
        'spec_sha256': spec['sha256'], 'proof_sha256': {str(i): ref['sha256'] for i, ref in enumerate(proofs)}})
    vectors = {'a': [3., 0., 2., -1.], 'r120': [2., 0., 1., -1.]}
    packets = []
    for physical, assignment in enumerate(plan):
        records = []
        if assignment['rank'] == 0:
            for index, value in enumerate(vectors[assignment['member']]):
                records.append({'index': index, 'rank': 0, 'batch': 0, 'offset': index,
                                'name': rows[index]['name'], 'label': rows[index]['label'],
                                'logit': value, 'probability': .5})
        packets.append(write_json(root / ('packet-' + str(physical) + '.json'), {
            'device_rank': physical, 'member': assignment['member'], 'rank': assignment['rank'],
            'error': None, 'input_sha256': inputs_ref['sha256'], 'proof_sha256': proofs[physical]['sha256'],
            'TTA': False, 'official_read': False, 'records': records}))
    seal = {'format': 'f05-selected-dev-packets-v1', 'status': 'selected_fp16_dev_completed',
            'fixture_only': True, 'preflight_sha256': preflight['sha256'], 'manifest_sha256': manifest['sha256'],
            'inputs': inputs_ref, 'worker_spec': spec, 'release': release, 'proofs': proofs, 'packets': packets,
            'forward_counts': {'a': 4, 'soup-r120': 4}, 'storage': 'fp16', 'arithmetic': 'fp32',
            'TTA': False, 'official_read': False}
    return preflight, manifest, screen, write_json(root / 'PACKETS-SEAL.json', seal)


def example_fold_candidate(root):
    """Build synthetic but fully bound three-fold metadata and one selected model."""
    import f05_oof
    import f05_oof_assets
    from test_f05_oof import fixture as base_fixture
    from retrain.age import make_spec
    from retrain.persistence import canonical_digest
    from retrain.f05_folds import F05_FOLD_PROFILES

    root = Path(root)
    base = base_fixture()
    base_dir = root / 'base-meta'; base_dir.mkdir()
    base_ref = write_json(base_dir / 'manifest.json', base)
    contract = write_json(base_dir / 'CONTRACT.json', {'source_identity': base['source_identity']})
    age = write_json(base_dir / 'AGE.json', {})
    schedule_path = base_dir / 'SCHEDULE.json'; schedule_path.write_bytes(b'{"steps_per_cycle":800}\n')
    schedule = file_ref(schedule_path)
    base_files = {'manifest.json': base_ref, 'CONTRACT.json': contract, 'AGE.json': age, 'SCHEDULE.json': schedule}
    base_freeze = write_json(base_dir / 'FREEZE.json', {'files': {
        name: {'path': name, 'bytes': Path(ref['path']).stat().st_size, 'sha256': ref['sha256']}
        for name, ref in base_files.items()}})
    folds = f05_oof.build_fold_manifests(base, 20261121)
    sealed_rows = []
    for number, fold in enumerate(folds):
        sealed_rows.append({'fold': number, 'round': 'R' + str(265 + number), 'counts': fold['counts'],
            'fit_indices': fold['fit_indices'], 'audit_indices': fold['audit_indices'],
            'fit_indices_sha256': canonical_digest(fold['fit_indices']),
            'audit_indices_sha256': canonical_digest(fold['audit_indices']),
            'dev_indices_sha256': canonical_digest(fold['dev_indices']),
            'manifest_canonical_sha256': canonical_digest(fold), 'age': make_spec(fold)})
    prep = root / 'technical-prep'; prep.mkdir()
    sealed_ref = write_json(prep / 'folds.json', {'status': 'three_fold_fit_only_indices_sealed_no_training',
        'seed': 20261121, 'base_manifest_sha256': base_ref['sha256'], 'folds': sealed_rows,
        'GPU_initialized': False, 'training_started': False, 'official_read': False})
    assets = f05_oof_assets.construct_assets(folds[0], {'source_identity': base['source_identity']},
        schedule_path.read_bytes(), base_manifest_sha256=base_ref['sha256'],
        sealed_folds_sha256=sealed_ref['sha256'], base_freeze_sha256=base_freeze['sha256'])
    fold_dir = prep / 'fold-meta'; fold_dir.mkdir()
    for name, raw in assets.items():
        (fold_dir / name).write_bytes(raw)
    train_ref, derived_ref = file_ref(fold_dir / 'manifest.json'), file_ref(fold_dir / 'FREEZE.json')
    round_id = 'r265-small-f05fold0-seed20261029-pairsigmoid-24k-p4-9010-20261105'
    round_root = root / 'rounds' / round_id; round_root.mkdir(parents=True)
    row, receipt = example_export(round_root, 'ema')
    row['round_id'] = round_id
    science_value = {'fixture_only': True, 'candidate_profile': F05_FOLD_PROFILES[0],
        'public_source': row['public_source'], 'candidate': {'weight_sha256': 'a' * 64},
        'job': {'initial_checkpoint': None}, 'paths': {'metadata_root': 'technical-prep/fold-meta'},
        'f05_fold': {'version': 'fit-only-three-fold-v1', 'fold': 0, 'folds': 3, 'split_seed': 20261121,
            'base_metadata_root': 'base-meta', 'base_freeze_sha256': base_freeze['sha256'],
            'base_manifest_sha256': base_ref['sha256'], 'sealed_folds_path': 'technical-prep/folds.json',
            'sealed_folds_sha256': sealed_ref['sha256'], 'derived_freeze_sha256': derived_ref['sha256']}}
    science = write_json(round_root / 'SCIENCE.json', science_value)
    site_science = write_json(prep / 'site-SCIENCE.json', science_value)
    site = write_json(round_root / 'SITE_BINDING.json', {'fixture_only': True, 'round_id': round_id,
        'work_root': str(root), 'metadata_root': str(fold_dir), 'science': {'configuration': site_science}})
    runtime = write_json(round_root / 'RUNTIME_BINDING.json', {'fixture_only': True,
                                                                'site_binding_sha256': site['sha256']})
    freeze_value = {'cpu_fixture_only': True, 'selected': row['selected'],
        'blob_sha256': row['selected_blob_sha256'], 'experiment91': {'manifest_sha256': train_ref['sha256']}}
    freeze = write_json(round_root / 'SINGLE-MODEL-FREEZE.json', freeze_value)
    row['spec']['freeze'], row['science'] = freeze, science
    row['fold_binding'] = {'fold': 0, 'site': site, 'runtime_binding': runtime, 'training_manifest': train_ref}
    export = json.loads(Path(receipt['path']).read_text())
    export.update(round_id=round_id, freeze=freeze, science=science)
    receipt = write_json(Path(receipt['path']), export)
    constants = {'ORIGINAL_FREEZE_SHA': base_freeze['sha256'], 'ORIGINAL_MANIFEST_SHA': base_ref['sha256'],
                 'THREE_FOLD_SEAL_SHA': sealed_ref['sha256']}
    return row, receipt, base_ref, constants


class F05FinalSelectionTests(unittest.TestCase):
    def test_selection_module_exists(self):
        self.assertIsNotNone(final, 'F05 final candidate selector not implemented')

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_threefold_members_must_be_explicitly_present_before_screening(self):
        inventory = {'members': [{'round_id': 'r265-synthetic'}, {'round_id': 'r266-synthetic'}]}
        with self.assertRaises(ValueError):
            final.require_fold_candidate_census(inventory)
        inventory['members'].append({'round_id': 'r267-synthetic'})
        self.assertIsNone(final.require_fold_candidate_census(inventory))

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_bound_threefold_candidate_is_screened_and_unknown_is_not_excluded(self):
        from retrain import f05_folds
        with tempfile.TemporaryDirectory() as directory:
            row, receipt, base, constants = example_fold_candidate(directory)
            candidate = [{'id': row['round_id'], 'kind': 'round', 'order': 0, 'discovery': row}]
            with mock.patch.multiple(f05_folds, **constants):
                population = final.validate_candidate_population(row, base, work_root=directory, allow_fixture=True)
                self.assertEqual(population['kind'], 'threefold')
                screen = final.screen_candidates(candidate, {row['round_id']: receipt}, inventory_sha='e' * 64,
                    soup_catalog_sha='d' * 64, approved={'a' * 64: 'general'}, base_ref=base,
                    work_root=directory, allow_fixture=True)
                self.assertEqual([item['id'] for item in screen['verified']], [row['round_id']])
                self.assertEqual(screen['unknown'], [])

                train_ref = row['fold_binding']['training_manifest']
                train = json.loads(Path(train_ref['path']).read_text())
                train['dev_indices'] = list(reversed(train['dev_indices']))
                row['fold_binding']['training_manifest'] = write_json(Path(train_ref['path']), train)
                freeze_ref = row['spec']['freeze']
                freeze = json.loads(Path(freeze_ref['path']).read_text())
                freeze['experiment91']['manifest_sha256'] = row['fold_binding']['training_manifest']['sha256']
                row['spec']['freeze'] = write_json(Path(freeze_ref['path']), freeze)
                bad = final.screen_candidates(candidate, {row['round_id']: receipt}, inventory_sha='e' * 64,
                    soup_catalog_sha='d' * 64, approved={'a' * 64: 'general'}, base_ref=base,
                    work_root=directory, allow_fixture=True)
                self.assertEqual([item['id'] for item in bad['unknown']], [row['round_id']])
                self.assertEqual(bad['excluded'], [])

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_fold_rejects_modified_derived_bytes_and_site_science(self):
        from retrain import f05_folds
        for target in ('training_manifest', 'derived_freeze', 'site_science', 'missing_binding'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                row, receipt, base, constants = example_fold_candidate(directory)
                if target == 'training_manifest':
                    Path(row['fold_binding']['training_manifest']['path']).write_text('{}')
                elif target == 'derived_freeze':
                    Path(row['fold_binding']['training_manifest']['path']).with_name('FREEZE.json').write_text('{}')
                elif target == 'site_science':
                    site = json.loads(Path(row['fold_binding']['site']['path']).read_text())
                    Path(site['science']['configuration']['path']).write_text('{}')
                else:
                    del row['fold_binding']
                candidates = [{'id': row['round_id'], 'kind': 'round', 'order': 0, 'discovery': row}]
                with mock.patch.multiple(f05_folds, **constants):
                    screen = final.screen_candidates(candidates, {row['round_id']: receipt},
                        inventory_sha='e' * 64, soup_catalog_sha='d' * 64, approved={'a' * 64: 'general'},
                        base_ref=base, work_root=directory, allow_fixture=True)
                self.assertEqual([item['id'] for item in screen['unknown']], [row['round_id']])
                self.assertEqual(screen['excluded'], [])

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_fold_fp32_reference_binds_derived_manifest_and_original_dev_order(self):
        from retrain import f05_folds
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, _, base_ref, constants = example_fold_candidate(root)
            base = json.loads(Path(base_ref['path']).read_text())
            round_root = Path(row['spec']['freeze']['path']).parent
            prediction_dir = round_root / 'predictions/ema'; prediction_dir.mkdir(parents=True)
            prediction_path = prediction_dir / 'dev-cycle-0001.csv'
            prediction_path.write_text('name,label,logit\n' + ''.join(
                base['rows'][i]['name'] + ',' + str(base['rows'][i]['label']) + ','
                + str(2. if base['rows'][i]['label'] else -2.) + '\n' for i in base['dev_indices']))
            prediction = file_ref(prediction_path)
            precision = write_json(Path(str(prediction_path) + '.precision.json'),
                {'variant': 'ema', 'view': 'identity', 'actual_tensor_sha256': row['selected']['tensor_sha256'],
                 'state_identity': {'successful_updates': row['selected']['successful_updates']}})
            row['selected'].update(cycle=1, auroc=1., ap=1., error_twice=0,
                prediction_path=prediction['path'], prediction_sha256=prediction['sha256'],
                precision_path=precision['path'], precision_sha256=precision['sha256'])
            freeze_ref = row['spec']['freeze']
            freeze = json.loads(Path(freeze_ref['path']).read_text())
            freeze.update(cpu_fixture_only=False, precision='fp32', view='identity', selected=row['selected'])
            row['spec']['freeze'] = write_json(Path(freeze_ref['path']), freeze)
            row['spec']['completion'] = write_json(round_root / 'TRAIN_RESULT.json',
                                                   {'status': 'completed', 'successful_updates': 800})
            inventory = write_json(root / 'inventory.json', {'status': 'read_only_F03_candidate_discovery_no_fusion',
                'official_read': False, 'manifest': base_ref, 'members': [row]})
            catalog = write_json(root / 'catalog.json', {'format': 'f05-soup-candidates-v1',
                                                          'official_read': False, 'soups': []})
            evidence = write_json(root / 'evidence.json', {'format': 'f05-candidate-evidence-v1',
                                                            'official_read': False, 'entries': {}})
            screen = write_json(root / 'screen.json', {'inventory': inventory, 'soup_catalog': catalog,
                                                        'evidence_manifest': evidence})
            asset = {'id': row['round_id'], 'variant': 'ema', 'successful_updates': 800, 'auroc': 1.,
                     'tensor_sha256': row['selected']['tensor_sha256'],
                     'selected_blob_sha256': row['selected_blob_sha256']}
            preflight = {'inventory_sha256': inventory['sha256'], 'soup_catalog_sha256': catalog['sha256'],
                         'fp16': {'members': [row['round_id']]}, 'verified': [asset]}
            with mock.patch.multiple(f05_folds, **constants):
                vectors, sources = final._selected_fp32_references(preflight, screen, base_ref, work=root,
                    allow_fixture=True, approved={'a' * 64: 'general'})
            self.assertEqual(len(vectors[row['round_id']]), len(base['dev_indices']))
            self.assertEqual(sources[row['round_id']]['kind'], 'original_selected_fp32_prediction')

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_other_manifest_mismatch_has_reason_but_unknown_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = write_json(root / 'base.json', {'rows': [], 'dev_indices': []})
            round_root = root / 'synthetic-round'; round_root.mkdir()
            row, _ = example_export(round_root, 'online')
            freeze_ref = row['spec']['freeze']
            freeze = json.loads(Path(freeze_ref['path']).read_text())
            freeze['experiment91'] = {'manifest_sha256': 'd' * 64}
            row['spec']['freeze'] = write_json(round_root / 'SINGLE-MODEL-FREEZE.json', freeze)
            row['spec']['completion'] = write_json(round_root / 'TRAIN_RESULT.json',
                                                    {'status': 'completed', 'successful_updates': 800})
            row['status'] = 'manifest_mismatch_verified'
            candidates = [{'id': row['round_id'], 'kind': 'round', 'order': 0, 'discovery': row}]
            kwargs = dict(inventory_sha='e' * 64, soup_catalog_sha='d' * 64,
                          approved={'a' * 64: 'general'}, base_ref=base, work_root=directory, allow_fixture=True,
                          _fixture_registered_manifest_shas={'1' * 64, '2' * 64, '3' * 64})
            screen = final.screen_candidates(candidates, {}, **kwargs)
            self.assertEqual(screen['excluded'][0]['reason'], 'freeze_declared_training_manifest_mismatch')
            no_registry = {key: value for key, value in kwargs.items() if key != '_fixture_registered_manifest_shas'}
            screen = final.screen_candidates(candidates, {}, **no_registry)
            self.assertEqual([item['id'] for item in screen['unknown']], [row['round_id']])
            self.assertEqual(screen['excluded'], [])
            freeze['experiment91']['manifest_sha256'] = '1' * 64
            row['spec']['freeze'] = write_json(round_root / 'SINGLE-MODEL-FREEZE.json', freeze)
            screen = final.screen_candidates(candidates, {}, **kwargs)
            self.assertEqual([item['id'] for item in screen['unknown']], [row['round_id']])
            self.assertEqual(screen['excluded'], [])
            Path(row['spec']['freeze']['path']).write_text('{}')
            screen = final.screen_candidates(candidates, {}, **kwargs)
            self.assertEqual([item['id'] for item in screen['unknown']], [row['round_id']])
            self.assertEqual(screen['excluded'], [])
            freeze['experiment91']['manifest_sha256'] = 'not-a-sha'
            row['spec']['freeze'] = write_json(round_root / 'SINGLE-MODEL-FREEZE.json', freeze)
            screen = final.screen_candidates(candidates, {}, **kwargs)
            self.assertEqual([item['id'] for item in screen['unknown']], [row['round_id']])
            self.assertEqual(screen['excluded'], [])
            for status in ('unknown_or_excluded', 'unknown', 'discovery_failed'):
                row['status'] = status
                screen = final.screen_candidates(candidates, {}, **kwargs)
                self.assertEqual([item['id'] for item in screen['unknown']], [row['round_id']])
                self.assertEqual(screen['excluded'], [])

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_mismatch_exclusion_requires_all_three_verified_fold_manifest_shas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = write_json(root / 'base.json', {'rows': [], 'dev_indices': []})
            round_root = root / 'synthetic-round'; round_root.mkdir()
            row, _ = example_export(round_root, 'ema')
            freeze = json.loads(Path(row['spec']['freeze']['path']).read_text())
            freeze['experiment91'] = {'manifest_sha256': 'd' * 64}
            row['spec']['freeze'] = write_json(round_root / 'SINGLE-MODEL-FREEZE.json', freeze)
            row['spec']['completion'] = write_json(round_root / 'TRAIN_RESULT.json', {'status': 'completed'})
            row['status'] = 'manifest_mismatch_verified'
            fold_rows = [{'round_id': 'r' + str(265 + i) + '-fixture', 'status': 'prediction_verified_weight_pending'}
                         for i in range(3)]
            all_rows = fold_rows + [row]
            candidates = [{'id': item['round_id'], 'kind': 'round', 'order': i, 'discovery': item}
                          for i, item in enumerate(all_rows)]
            real_validator = final.validate_candidate_population
            def synthetic_fold_or_actual(candidate, *args, **kwargs):
                if candidate in fold_rows:
                    number = fold_rows.index(candidate)
                    return {'kind': 'threefold', 'fold': number, 'training_manifest_sha256': str(number + 1) * 64}
                return real_validator(candidate, *args, **kwargs)
            kwargs = dict(inventory_sha='e' * 64, soup_catalog_sha='d' * 64,
                          approved={'a' * 64: 'general'}, base_ref=base, work_root=directory, allow_fixture=True)
            with mock.patch.object(final, 'validate_candidate_population', side_effect=synthetic_fold_or_actual):
                screen = final.screen_candidates(candidates, {}, **kwargs)
                self.assertIn({'id': row['round_id'], 'reason': 'freeze_declared_training_manifest_mismatch'},
                              screen['excluded'])
                fold_rows[1]['status'] = 'unknown_or_excluded'
                screen = final.screen_candidates(candidates, {}, **kwargs)
                self.assertNotIn(row['round_id'], [item['id'] for item in screen['excluded']])
                self.assertIn(row['round_id'], [item['id'] for item in screen['unknown']])

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
            fixed = final.freeze_selection(screen, policy_ref=policy_ref, allow_fixture=True)
            self.assertEqual(fixed['fp16']['members'], [row['round_id']])
            policy['candidate_order'] = ['other']
            wrong = write_json(root / 'wrong-policy.json', policy)
            with self.assertRaises(ValueError):
                final.freeze_selection(screen, policy_ref=wrong, allow_fixture=True)

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_equal_z_internal_dev_and_rejects_zero_variance(self):
        result = final.fuse_internal_dev([1, 0, 1, 0], {'a': [3, 1, 2, 0], 'b': [2, 0, 1, -1]}, ['a', 'b'])
        self.assertEqual(result['metrics']['auroc'], 1.)
        with self.assertRaises(ValueError):
            final.fuse_internal_dev([1, 0, 1, 0], {'a': [1, 1, 1, 1]}, ['a'])

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_fusion_requires_frozen_models_and_complete_pinned_rank_packets(self):
        self.assertTrue(hasattr(final, 'fuse_from_selected_packets'), 'verified F05 dev packet consumer missing')
        with tempfile.TemporaryDirectory() as directory:
            preflight, manifest, screen, seal = example_selected_dev(directory)
            result = final.fuse_from_selected_packets(preflight, manifest, [seal], work_root=directory,
                screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})
            self.assertEqual(result['members'], ['a', 'soup-r120'])
            self.assertEqual(result['metrics']['auroc'], 1.)
            self.assertEqual(result['fp32_reference']['metrics']['auroc'], 1.)
            self.assertEqual(set(result['member_precision_comparison']), {'a', 'soup-r120'})
            packet_ref = json.loads(Path(seal['path']).read_text())['packets'][0]
            Path(packet_ref['path']).write_text('{}')
            with self.assertRaises(ValueError):
                final.fuse_from_selected_packets(preflight, manifest, [seal], work_root=directory,
                    screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_fusion_rejects_wrong_model_metadata_and_missing_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            preflight, manifest, screen, seal = example_selected_dev(directory)
            seal_value = json.loads(Path(seal['path']).read_text())
            inputs_ref = seal_value['inputs']
            inputs = json.loads(Path(inputs_ref['path']).read_text())
            model = Path(inputs['members'][0]['model']['path'])
            model.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                final.fuse_from_selected_packets(preflight, manifest, [seal], work_root=directory,
                    screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})
        with tempfile.TemporaryDirectory() as directory:
            preflight, manifest, screen, seal = example_selected_dev(directory)
            seal_value = json.loads(Path(seal['path']).read_text())
            seal_value['packets'].pop()
            seal = write_json(Path(seal['path']), seal_value)
            with self.assertRaises(ValueError):
                final.fuse_from_selected_packets(preflight, manifest, [seal], work_root=directory,
                    screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})
        with tempfile.TemporaryDirectory() as directory:
            preflight, manifest, screen, seal = example_selected_dev(directory)
            inputs_ref = json.loads(Path(seal['path']).read_text())['inputs']
            inputs = json.loads(Path(inputs_ref['path']).read_text())
            metadata = Path(inputs['members'][0]['metadata']['path'])
            metadata.write_text('{}')
            with self.assertRaises(ValueError):
                final.fuse_from_selected_packets(preflight, manifest, [seal], work_root=directory,
                    screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_fusion_rejects_tta_official_and_duplicate_member(self):
        for field in ('TTA', 'official_read'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                preflight, manifest, screen, seal = example_selected_dev(directory)
                seal_value = json.loads(Path(seal['path']).read_text())
                seal_value[field] = True
                seal = write_json(Path(seal['path']), seal_value)
                with self.assertRaises(ValueError):
                    final.fuse_from_selected_packets(preflight, manifest, [seal], work_root=directory,
                        screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})
        with tempfile.TemporaryDirectory() as directory:
            preflight, manifest, screen, seal = example_selected_dev(directory)
            with self.assertRaises(ValueError):
                final.fuse_from_selected_packets(preflight, manifest, [seal, seal], work_root=directory,
                    screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})

    @unittest.skipIf(final is None, 'selector not implemented')
    def test_fusion_rejects_tampered_original_and_soup_fp32_references(self):
        with tempfile.TemporaryDirectory() as directory:
            preflight, manifest, screen, seal = example_selected_dev(directory)
            freeze_path = Path(directory) / 'rounds/a/SINGLE-MODEL-FREEZE.json'
            prediction = json.loads(freeze_path.read_text())['selected']['prediction_path']
            Path(prediction).write_text('name,label,logit\n')
            with self.assertRaises(ValueError):
                final.fuse_from_selected_packets(preflight, manifest, [seal], work_root=directory,
                    screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})
        with tempfile.TemporaryDirectory() as directory:
            preflight, manifest, screen, seal = example_selected_dev(directory)
            packet = Path(directory) / 'soup/evaluation/packets/rank-0-fp32.json'
            packet.write_text('{}')
            with self.assertRaises(ValueError):
                final.fuse_from_selected_packets(preflight, manifest, [seal], work_root=directory,
                    screen_ref=screen, allow_fixture=True, approved={'a' * 64: 'general'})


if __name__ == '__main__':
    paths = ('tools/diagnostics/f05_final_selection.py', 'tests/test_f05_final_selection.py',
             'tools/diagnostics/f04_size_ensemble.py', 'tools/diagnostics/f04_bounded_fusion.py',
             'tools/diagnostics/ensemble_dev.py', 'tools/diagnostics/f05_soup_dev.py',
             'tools/diagnostics/f04_gpu_full_dev.py')
    present = {name: file_ref(ROOT / name)['sha256'] for name in paths if (ROOT / name).is_file()}
    print(json.dumps({'source_sha256': present}), flush=True)
    unittest.main()

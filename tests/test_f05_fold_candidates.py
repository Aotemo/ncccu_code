"""Synthetic F03 augmentation, sealed-fold discovery and failure semantics."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'tools/diagnostics'), str(ROOT / 'runtime/bundle')]
import paired_evidence
import f05_final_selection as final
from retrain import f05_folds
from test_f05_final_selection import example_fold_candidate, file_ref, write_json

SOURCE = ROOT / 'tools/diagnostics/f05_fold_candidates.py'
if SOURCE.is_file():
    spec = importlib.util.spec_from_file_location('f05_fold_candidates', SOURCE)
    discovery = importlib.util.module_from_spec(spec); spec.loader.exec_module(discovery)
else:
    discovery = None


def prepare_completed_fold(root, row, base_ref):
    """Create only small synthetic dev predictions and an opaque toy checkpoint."""
    root = Path(root)
    round_root = Path(row['spec']['freeze']['path']).parent
    base = json.loads(Path(base_ref['path']).read_text())
    pred_dir = round_root / 'predictions/ema'; pred_dir.mkdir(parents=True, exist_ok=True)
    pred = pred_dir / 'dev-cycle-0001.csv'
    pred.write_text('name,label,logit\n' + ''.join(
        f"{base['rows'][i]['name']},{base['rows'][i]['label']},{2. if base['rows'][i]['label'] else -2.}\n"
        for i in base['dev_indices']))
    prediction = file_ref(pred)
    precision = write_json(Path(str(pred) + '.precision.json'),
        {'variant': 'ema', 'view': 'identity', 'actual_tensor_sha256': row['selected']['tensor_sha256'],
         'state_identity': {'successful_updates': 800}})
    selected = {**row['selected'], 'cycle': 1, 'variant': 'ema', 'auroc': 1., 'ap': 1., 'error_twice': 0,
                'prediction_path': prediction['path'], 'prediction_sha256': prediction['sha256'],
                'precision_path': precision['path'], 'precision_sha256': precision['sha256']}
    checkpoint_dir = round_root / 'outputs/job/run/checkpoints'; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / 'checkpoint-0001.pt'
    checkpoint.write_bytes(b'synthetic-checkpoint-only')
    blob_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(Path(str(checkpoint) + '.json'), {'sha256': blob_sha})
    freeze = json.loads(Path(row['spec']['freeze']['path']).read_text())
    freeze.update(cpu_fixture_only=False, precision='fp32', view='identity', selected=selected, blob_sha256=blob_sha)
    write_json(round_root / 'SINGLE-MODEL-FREEZE.json', freeze)
    (round_root / 'control').mkdir(exist_ok=True)
    write_json(round_root / 'control/done.json', {'status': 'completed'})
    (round_root / 'evidence').mkdir(exist_ok=True)
    write_json(round_root / 'evidence/TRAIN_RESULT.json', {'status': 'completed', 'successful_updates': 800})
    return checkpoint


def add_synthetic_fold(root, first, number, base_ref, constants):
    import f05_oof
    import f05_oof_assets
    from retrain.f05_folds import F05_FOLD_PROFILES
    root = Path(root)
    base = json.loads(Path(base_ref['path']).read_text())
    fold = f05_oof.build_fold_manifests(base, 20261121)[number]
    assets = f05_oof_assets.construct_assets(fold, {'source_identity': base['source_identity']},
        (root / 'base-meta/SCHEDULE.json').read_bytes(), base_manifest_sha256=base_ref['sha256'],
        sealed_folds_sha256=constants['THREE_FOLD_SEAL_SHA'], base_freeze_sha256=constants['ORIGINAL_FREEZE_SHA'])
    folder = root / 'technical-prep' / ('fold-meta-' + str(number)); folder.mkdir()
    for name, raw in assets.items(): (folder / name).write_bytes(raw)
    round_id = f'r{265 + number}-small-f05fold{number}-seed20261029-pairsigmoid-24k-p4-9010-20261105'
    round_root = root / 'rounds' / round_id; round_root.mkdir()
    science = json.loads(Path(first['science']['path']).read_text())
    science['candidate_profile'] = F05_FOLD_PROFILES[number]
    science['paths']['metadata_root'] = str(folder.relative_to(root))
    science['f05_fold']['fold'] = number
    science['f05_fold']['derived_freeze_sha256'] = file_ref(folder / 'FREEZE.json')['sha256']
    science_ref = write_json(round_root / 'SCIENCE.json', science)
    site_science = write_json(root / 'technical-prep' / ('site-SCIENCE-' + str(number) + '.json'), science)
    site = {'fixture_only': True, 'round_id': round_id, 'work_root': str(root),
            'metadata_root': str(folder), 'science': {'configuration': site_science}}
    site_ref = write_json(round_root / 'SITE_BINDING.json', site)
    write_json(round_root / 'RUNTIME_BINDING.json', {'fixture_only': True, 'site_binding_sha256': site_ref['sha256']})
    row = copy.deepcopy(first)
    row['round_id'] = round_id; row['science'] = science_ref
    row['spec']['freeze'] = write_json(round_root / 'SINGLE-MODEL-FREEZE.json',
        {'cpu_fixture_only': True, 'selected': row['selected'], 'experiment91': {'manifest_sha256': file_ref(folder / 'manifest.json')['sha256']}})
    prepare_completed_fold(root, row, base_ref)
    return row


def fixture(root, all_folds=False):
    root = Path(root)
    row, _, base_ref, constants = example_fold_candidate(root)
    checkpoint = prepare_completed_fold(root, row, base_ref)
    if all_folds:
        for number in (1, 2): add_synthetic_fold(root, row, number, base_ref, constants)
    unknown = {'round_id': 'r200-existing-unknown', 'status': 'unknown_or_excluded', 'reason': 'existing_problem'}
    f03 = {'status': 'read_only_F03_candidate_discovery_no_fusion', 'fixture_only': True,
           'official_read': False, 'forward_calls': 0, 'GPU_initialized': False,
           'manifest': base_ref, 'fixed_positive_indices': [], 'members': [unknown]}
    f03_ref = write_json(root / 'F03-CANDIDATES.json', f03)
    return row, base_ref, constants, checkpoint, unknown, f03_ref


class F05FoldCandidatesTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(discovery, 'F05 fold discovery helper not implemented')
        return discovery

    def test_fold_binds_derived_manifest_before_selected_and_preserves_unknown(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as directory:
            row, base, constants, checkpoint, unknown, f03_ref = fixture(directory)
            with mock.patch.multiple(f05_folds, **constants), \
                 mock.patch.object(paired_evidence, '_selected', wraps=paired_evidence._selected) as selected:
                result = module.discover(f03_ref, work_root=directory, allow_fixture=True)
            fold = next(item for item in result['members'] if item['round_id'] == row['round_id'])
            self.assertEqual(fold['status'], 'prediction_verified_weight_pending')
            self.assertEqual(fold['fold_binding']['fold'], 0)
            self.assertEqual(fold['fold_binding']['training_manifest']['sha256'], row['fold_binding']['training_manifest']['sha256'])
            self.assertEqual(selected.call_count, 1)
            self.assertEqual(selected.call_args.args[1]['manifest_sha256'], row['fold_binding']['training_manifest']['sha256'])
            self.assertIn(unknown, result['members'])
            for number in (266, 267):
                found = [r for r in result['members'] if r['round_id'].startswith(f'r{number}-')]
                self.assertEqual(len(found), 1); self.assertEqual(found[0]['status'], 'unknown_or_excluded')
            self.assertEqual(result['artifact_kind'], 'f05-augmented-candidate-discovery-v1')
            self.assertTrue(result['augmented']); self.assertEqual(result['source_f03_inventory'], f03_ref)
            self.assertEqual(result['manifest'], base)
            sealed = write_json(Path(directory) / 'AUGMENTED.json', result)
            with mock.patch.multiple(f05_folds, **constants):
                module.verify_augmented(sealed, work_root=directory, allow_fixture=True)
                altered = copy.deepcopy(result); altered['members'].remove(unknown)
                bad = write_json(Path(directory) / 'ALTERED.json', altered)
                with self.assertRaises(ValueError): module.verify_augmented(bad, work_root=directory, allow_fixture=True)
            Path(f03_ref['path']).write_text('{}')
            with self.assertRaises(ValueError): module.verify_augmented(sealed, work_root=directory, allow_fixture=True)

    def test_bad_fold_binding_does_not_call_selected_or_become_excluded(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as directory:
            row, _, constants, _, _, f03_ref = fixture(directory)
            Path(row['fold_binding']['runtime_binding']['path']).unlink()
            with mock.patch.multiple(f05_folds, **constants), \
                 mock.patch.object(paired_evidence, '_selected', wraps=paired_evidence._selected) as selected:
                result = module.discover(f03_ref, work_root=directory, allow_fixture=True)
            fold = next(item for item in result['members'] if item['round_id'] == row['round_id'])
            self.assertEqual(fold['status'], 'unknown_or_excluded'); self.assertEqual(selected.call_count, 0)

    def test_original_f03_fold_unknown_is_preserved_without_duplicate_or_promotion(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as directory:
            row, _, constants, _, _, f03_ref = fixture(directory)
            original = json.loads(Path(f03_ref['path']).read_text())
            source_unknown = {'round_id': row['round_id'], 'status': 'unknown_or_excluded',
                              'reason': 'original_F03_could_not_read_freeze'}
            original['members'].append(source_unknown)
            f03_ref = write_json(Path(f03_ref['path']), original)
            with mock.patch.multiple(f05_folds, **constants), \
                 mock.patch.object(paired_evidence, '_selected', wraps=paired_evidence._selected) as selected:
                result = module.discover(f03_ref, work_root=directory, allow_fixture=True)
            self.assertEqual([r for r in result['members'] if r['round_id'] == row['round_id']], [source_unknown])
            self.assertEqual(selected.call_count, 0)
            sealed = write_json(Path(directory) / 'AUGMENTED.json', result)
            module.verify_augmented(sealed, work_root=directory, allow_fixture=True)

    def test_missing_and_ambiguous_selected_files_are_distinct_from_unknown(self):
        module = self.require_module()
        for mode in ('missing', 'sidecar_missing', 'ambiguous'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                row, _, constants, checkpoint, _, f03_ref = fixture(directory)
                sidecar = Path(str(checkpoint) + '.json')
                if mode == 'missing': checkpoint.unlink(); sidecar.unlink()
                elif mode == 'sidecar_missing': sidecar.unlink()
                else:
                    second = checkpoint.with_name('checkpoint-0002.pt'); shutil.copyfile(checkpoint, second)
                    shutil.copyfile(sidecar, Path(str(second) + '.json'))
                with mock.patch.multiple(f05_folds, **constants):
                    result = module.discover(f03_ref, work_root=directory, allow_fixture=True)
                fold = next(item for item in result['members'] if item['round_id'] == row['round_id'])
                expected = {'missing': 'selected_checkpoint_missing', 'sidecar_missing': 'unknown_or_excluded',
                            'ambiguous': 'selected_checkpoint_ambiguous'}[mode]
                self.assertEqual(fold['status'], expected)

    def test_other_manifest_is_excluded_only_after_all_registered_folds_verify(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, constants, _, _, f03_ref = fixture(root, all_folds=True)
            other = root / 'rounds/r123-unregistered'; (other / 'control').mkdir(parents=True)
            (other / 'evidence').mkdir()
            write_json(other / 'control/done.json', {'status': 'completed'})
            write_json(other / 'evidence/TRAIN_RESULT.json', {'status': 'completed'})
            write_json(other / 'SINGLE-MODEL-FREEZE.json', {'experiment91': {'manifest_sha256': 'f' * 64}})
            with mock.patch.multiple(f05_folds, **constants):
                result = module.discover(f03_ref, work_root=root, allow_fixture=True)
            rows = [r for r in result['members'] if r['round_id'] == 'r123-unregistered']
            self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['status'], 'manifest_mismatch_verified')
            self.assertEqual(rows[0]['reason'], 'freeze_declared_training_manifest_mismatch')
            self.assertTrue(all(sum(r['round_id'].startswith(f'r{number}-') for r in result['members']) == 1
                                for number in (265, 266, 267)))
            final.require_fold_candidate_census(result)
            candidates = final.build_candidate_inventory(result, [], [row['round_id'] for row in result['members']])
            with mock.patch.multiple(f05_folds, **constants):
                screen = final.screen_candidates(candidates, {}, inventory_sha='a' * 64, soup_catalog_sha='b' * 64,
                    approved={'a' * 64: 'general'}, base_ref=result['manifest'], work_root=root, allow_fixture=True)
            self.assertIn({'id': 'r123-unregistered', 'reason': 'freeze_declared_training_manifest_mismatch'},
                          screen['excluded'])
            (root / 'rounds/r267-small-f05fold2-seed20261029-pairsigmoid-24k-p4-9010-20261105/RUNTIME_BINDING.json').unlink()
            with mock.patch.multiple(f05_folds, **constants):
                result = module.discover(f03_ref, work_root=root, allow_fixture=True)
            other_row = next(r for r in result['members'] if r['round_id'] == 'r123-unregistered')
            self.assertEqual(other_row['status'], 'unknown_or_excluded')

    def test_verifier_rejects_dropping_newly_discovered_unknown(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, constants, _, _, f03_ref = fixture(root)
            partial = root / 'rounds/r124-partial'; partial.mkdir()
            write_json(partial / 'SINGLE-MODEL-FREEZE.json', {'incomplete': True})
            with mock.patch.multiple(f05_folds, **constants):
                result = module.discover(f03_ref, work_root=root, allow_fixture=True)
                unknown = next(row for row in result['members'] if row['round_id'] == 'r124-partial')
                self.assertEqual(unknown['status'], 'unknown_or_excluded')
                result['members'].remove(unknown)
                sealed = write_json(root / 'ALTERED.json', result)
                with self.assertRaises(ValueError):
                    module.verify_augmented(sealed, work_root=root, allow_fixture=True)


if __name__ == '__main__': unittest.main()

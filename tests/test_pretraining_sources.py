"""Narrow rejection checks using verified public metadata, compute Pod only."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/pretraining'))
from source_chain import (configure_torch_threads, cpu_context, file_identity, inherited_file, read_pinned_receipt,
                          sha256, validate_metadata, write_new_json)
from strict_load import check_tensor, gelu_semantics, target_key

FIXTURE = None
SCRATCH = None
LIVE_CONTEXT = None


class PretrainingSourcesTest(unittest.TestCase):
    def setUp(self):
        independent = ('test_non_tiny_blocks_and_unknown_head_parameters_are_rejected',
                       'test_value_dtype_shape_and_nonfinite_corruption_are_rejected',
                       'test_actual_custom_gelu_matches_and_tanh_or_unknown_classes_are_rejected')
        if FIXTURE is None and self._testMethodName not in independent:
            self.skipTest('requires CLI with a pinned source receipt and explicit Pod scratch directory')

    def test_verified_source_metadata(self):
        validate_metadata(*copy.deepcopy(FIXTURE))

    def test_revision_latest_url_and_foreign_blob_are_rejected(self):
        mutations = [(0, 'revision', 'main'), (0, 'source_url', FIXTURE[0]['source_url'].replace('/bc48a87f119bae1b06418f5340bfa97ef75d4609/', '/main/')),
                     (2, 'sha', '0' * 40), (4, 'actual_sha256', '0' * 64)]
        for index, field, value in mutations:
            fixture = copy.deepcopy(FIXTURE)
            fixture[index][field] = value
            with self.subTest(field=field, index=index), self.assertRaises(ValueError):
                validate_metadata(*fixture)

    def test_wrong_norm_head_features_and_preprocessing_are_rejected(self):
        for index, field, value in ((0, 'final_norm_eps', 1e-12), (0, 'internal_norm_eps', 1e-12),
                                    (3, 'num_classes', 1001), (3, 'num_features', 1024)):
            fixture = copy.deepcopy(FIXTURE)
            fixture[index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_metadata(*fixture)
        fixture = copy.deepcopy(FIXTURE)
        fixture[3]['pretrained_cfg']['mean'] = [0., 0., 0.]
        with self.assertRaises(ValueError):
            validate_metadata(*fixture)

    def test_incomplete_public_recipe_or_model_card_is_rejected(self):
        fixture = copy.deepcopy(FIXTURE)
        fixture[0]['training_data'] = 'ImageNet-1K'
        with self.assertRaises(ValueError):
            validate_metadata(*fixture)
        fixture = copy.deepcopy(FIXTURE)
        fixture[5] = 'license: apache-2.0\n'
        with self.assertRaises(ValueError):
            validate_metadata(*fixture)

    def test_prior_hash_is_not_accepted_for_changed_file(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            path = Path(folder) / 'receipt.json'
            path.write_text('{}\n')
            digest = sha256(path)
            record = dict(file_identity(path), exists=True, sha256=digest)
            inherited_file(path, record, digest)
            path.write_text('{"changed":true}\n')
            with self.assertRaises(ValueError):
                inherited_file(path, record, digest)
            with self.assertRaises(ValueError):
                read_pinned_receipt(path, digest)

    def test_non_tiny_blocks_and_unknown_head_parameters_are_rejected(self):
        for name in ('stages.0.blocks.3.gamma', 'stages.2.blocks.9.norm.weight',
                     'stages.0.downsample.0.weight', 'head.fc.running_mean',
                     'stages.4.blocks.0.conv_dw.weight', 'teacher.head.fc.weight'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                target_key(name)

    def test_value_dtype_shape_and_nonfinite_corruption_are_rejected(self):
        import torch
        source = torch.tensor([.25, .5], dtype=torch.float32)
        converted = source.reshape(2, 1, 1).clone()
        check_tensor(source, converted, converted.shape, gamma=True)
        for invalid in (converted + .25, converted.to(torch.float16), converted.reshape(2),
                        torch.full_like(converted, float('nan')), torch.full_like(converted, float('inf'))):
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype), self.assertRaises(ValueError):
                check_tensor(source, invalid, converted.shape, gamma=True)
        with self.assertRaises(ValueError):
            check_tensor(source.reshape(2, 1), converted, converted.shape, gamma=True)

    def test_actual_custom_gelu_matches_and_tanh_or_unknown_classes_are_rejected(self):
        import torch
        from timm.layers.activations import GELU, GELUTanh
        self.assertEqual(gelu_semantics(GELU())['verified_approximate'], 'none')
        self.assertEqual(gelu_semantics(torch.nn.GELU(approximate='none'))['verified_approximate'], 'none')
        class UnknownGELU(torch.nn.GELU):
            pass
        for module in (torch.nn.GELU(approximate='tanh'), GELUTanh(), UnknownGELU()):
            with self.subTest(module=repr(module)), self.assertRaises(ValueError):
                gelu_semantics(module)


class SmallMappingTest(unittest.TestCase):
    def test_large_full_mapping_and_existing_base_defaults(self):
        import torch,timm,torchvision
        import base_source,base_strict
        self.assertIn('large224',base_source.SOURCE_PROFILES)
        profile=base_source.SOURCE_PROFILES['large224']
        self.assertEqual((profile['model'],profile['features']),('convnext_large',1536))
        self.assertEqual(profile['revision'],'dc2e53fad264aa1fb4e5940e0d1c9b79ac6e0854')
        self.assertEqual(base_source.download.__defaults__,(None,'base224'))
        self.assertEqual((base_source.MODEL,base_source.REPO),('convnext_base','timm/convnext_base.fb_in22k_ft_in1k'))
        source=timm.create_model('convnext_large.fb_in22k_ft_in1k',pretrained=False,drop_path_rate=.1)
        target=torchvision.models.convnext_large(weights=None,stochastic_depth_prob=.1)
        state=source.state_dict();converted={}
        for key,value in state.items():
            name,gamma=target_key(key,(3,3,27,3))
            converted[name]=value.reshape(-1,1,1).contiguous()if gamma else value
        rows=base_strict.base_mapping(state,converted,target.state_dict())
        self.assertEqual(len(rows),344);self.assertEqual(sum(r['gamma_reshape']for r in rows),36)
        self.assertEqual(tuple(converted['classifier.2.weight'].shape),(1000,1536))
        loaded=target.load_state_dict(converted,strict=True)
        self.assertFalse(loaded.missing_keys or loaded.unexpected_keys)
        self.assertFalse(torch.cuda.is_initialized())

    def test_dino3_complete_headless_mapping_and_original_source_defaults(self):
        import torch
        import timm
        import torchvision
        import small_source
        import small_strict
        self.assertIn('smalldino3', small_source.SOURCE_PROFILES)
        profile = small_source.SOURCE_PROFILES['smalldino3']
        self.assertEqual(profile['revision'], 'c497e64d5d483b3ad42eac93e2ef9d1799f9547c')
        self.assertEqual(profile['num_classes'], 0)
        self.assertEqual(small_source.download.__defaults__, (None, 'small224'))
        self.assertEqual(small_source.REVISION, 'fe6b358aa409414cdb522ae3d43a2c9922f7cc1b')
        source = timm.create_model('convnext_small.dinov3_lvd1689m', pretrained=False)
        target = torchvision.models.convnext_small(weights=None)
        target.classifier[-1] = torch.nn.Identity()
        state = source.state_dict()
        converted = {}
        for key, value in state.items():
            name, gamma = target_key(key, (3, 3, 27, 3))
            converted[name] = value.reshape(-1, 1, 1).contiguous() if gamma else value
        rows = small_strict.small_mapping(state, converted, target.state_dict(), public_head=False)
        self.assertEqual(len(rows), 342)
        self.assertTrue(all(row['role'] == 'backbone' for row in rows))
        self.assertEqual(sum(row['gamma_reshape'] for row in rows), 36)
        self.assertIn('classifier.0.weight', converted)
        self.assertFalse(any(key.startswith('classifier.2.') for key in converted))
        loaded = target.load_state_dict(converted, strict=True)
        self.assertFalse(loaded.missing_keys or loaded.unexpected_keys)
        with self.assertRaises(ValueError):
            small_strict.small_mapping(state, converted, target.state_dict())
        incomplete = dict(converted)
        del incomplete['classifier.0.weight']
        with self.assertRaises(ValueError):
            small_strict.small_mapping(state, incomplete, target.state_dict(), public_head=False)
        self.assertFalse(torch.cuda.is_initialized())

    def test_small_extra_blocks_map_without_changing_tiny_depth(self):
        depths = (3, 3, 27, 3)
        self.assertEqual(target_key('stages.2.blocks.26.mlp.fc1.weight', depths),
                         ('features.5.26.block.3.weight', False))
        self.assertEqual(target_key('stages.2.blocks.26.gamma', depths),
                         ('features.5.26.layer_scale', True))
        with self.assertRaises(ValueError):
            target_key('stages.2.blocks.27.norm.weight', depths)
        with self.assertRaises(ValueError):
            target_key('stages.2.blocks.9.norm.weight')


class ReusedIdentityTest(unittest.TestCase):
    def setUp(self):
        if SCRATCH is None:
            self.skipTest('requires the Pod CLI and an explicit scratch directory')

    def test_same_length_inplace_change_cannot_inherit_old_sha(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            path = Path(folder) / 'receipt.json'
            path.write_bytes(b'{}\n')
            record = dict(file_identity(path), exists=True, sha256=sha256(path))
            old = path.stat()
            with path.open('r+b') as stream:
                stream.write(b'[]\n')
            os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
            self.assertEqual(path.stat().st_size, record['bytes'])
            self.assertEqual(path.stat().st_ino, record['inode'])
            self.assertNotEqual(sha256(path), record['sha256'])
            with self.assertRaisesRegex(ValueError, 'mtime_ns|ctime_ns|content SHA'):
                inherited_file(path, record, record['sha256'])

    def test_either_stale_timestamp_rejects_sha_reuse(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            path = Path(folder) / 'proof.txt'
            path.write_bytes(b'AB')
            record = dict(file_identity(path), exists=True, sha256=sha256(path))
            for key in ('mtime_ns', 'ctime_ns'):
                stale = dict(record, **{key: record[key] - 1})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    inherited_file(path, stale, record['sha256'])

    def test_missing_timestamp_cannot_be_backfilled_from_current_stat(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            path = Path(folder) / 'proof.txt'
            path.write_bytes(b'AB')
            record = dict(file_identity(path), exists=True, sha256=sha256(path))
            for key in ('mtime_ns', 'ctime_ns'):
                incomplete = dict(record)
                del incomplete[key]
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'incomplete|missing'):
                    inherited_file(path, incomplete, record['sha256'])


class StageBindingTest(unittest.TestCase):
    def setUp(self):
        if SCRATCH is None or LIVE_CONTEXT is None:
            self.skipTest('requires the verified Pod CLI context')
        self.folder = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.context = copy.deepcopy(LIVE_CONTEXT)
        historical = {k: self.context[k] for k in ('namespace', 'pod', 'uid', 'container', 'python')}
        historical.update(uid='historical-test-uid', pod='historical-test-pod')
        files = {}
        for name in ('source-manifest.json', 'model-api.json', 'config.json', 'README.md',
                     'LICENSE-apache-2.0.txt', 'pytorch_model.bin', 'download-receipt.json',
                     'NOTICE-upstream-MIT.txt', 'converted', 'provenance', 'historical_converter'):
            path = root / name
            path.write_bytes(b'AB')
            files[name] = dict(file_identity(path), sha256=sha256(path))
        snapshot = root / 'runtime'
        snapshot.mkdir()
        (snapshot / 'models.py').write_text('# fixed test source\n')
        self.chain = {'version': 'tiny22k-source-chain-v1', 'passed': True, 'execution': historical,
                      'files': {k: v for k, v in files.items() if k not in ('provenance', 'historical_converter')},
                      'provenance': files['provenance'], 'historical_converter': files['historical_converter'],
                      'runtime_snapshot': {'root': str(snapshot),
                                           'git_commit': 'cb939c8e5078a9c74a26158a78c4aa033009042a',
                                           'files': {'models.py': sha256(snapshot / 'models.py')}}}
        source = root / 'source-receipt.json'
        source.write_text(json.dumps(self.chain))
        self.reference = {'path': str(source), 'sha256': sha256(source)}
        self.identities = files

    def binding_fixture(self):
        allocation = self.context['cpu_allocation']
        return {'schema_version': 1, 'stage': 'pretraining-strict',
                'source_receipt': dict(self.reference),
                'execution': {k: self.context[k] for k in ('namespace', 'pod', 'uid', 'container', 'python')},
                'cpu_allocation': {'path': allocation['path'], 'sha256': allocation['sha256'],
                                   'executor_id': allocation['executor_id'], 'threads': allocation['threads'],
                                   'mask': allocation['channel_mask']},
                'assets': [{'source_key': key,
                            'current_identity': {k: value[k] for k in
                                                 ('path', 'resolved', 'bytes', 'device', 'inode', 'mtime_ns', 'ctime_ns')},
                            'sha256': value['sha256'], 'evidence': {'kind': 'fresh_pod_sha256',
                                                                 'source_receipt_sha256': self.reference['sha256']}}
                           for key, value in self.identities.items()],
                'runtime_snapshot': copy.deepcopy(self.chain['runtime_snapshot'])}

    def test_cross_uid_accepts_explicit_pinned_binding_without_mutating_history(self):
        from stage_binding import build_stage_binding, validate_execution_binding
        # A new Pod may assign a different device number to the same public bytes.
        self.chain['files']['converted']['device'] += 1
        Path(self.reference['path']).write_text(json.dumps(self.chain))
        self.reference['sha256'] = sha256(self.reference['path'])
        historical = copy.deepcopy(self.chain)
        value = build_stage_binding(self.chain, self.reference, self.context)
        validate_execution_binding(self.chain, self.reference, self.context, value)
        self.assertEqual(value['source_receipt'], self.reference)
        self.assertEqual(value['execution']['uid'], self.context['uid'])
        self.assertEqual(len(value['assets']), 11)
        current = next(row for row in value['assets'] if row['source_key'] == 'converted')
        self.assertEqual(current['current_identity']['device'], Path(self.identities['converted']['path']).stat().st_dev)
        self.assertEqual(current['evidence']['kind'], 'fresh_pod_sha256')
        self.assertEqual(self.chain, historical)

    def test_cross_uid_without_stage_binding_is_rejected(self):
        from stage_binding import validate_execution_binding
        with self.assertRaisesRegex(ValueError, 'stage binding'):
            validate_execution_binding(self.chain, self.reference, self.context)

    def test_same_uid_path_keeps_all_five_execution_checks(self):
        from stage_binding import validate_execution_binding
        chain = copy.deepcopy(self.chain)
        chain['execution'] = {k: self.context[k] for k in ('namespace', 'pod', 'uid', 'container', 'python')}
        validate_execution_binding(chain, self.reference, self.context)
        chain['execution']['python'] = '/other/interpreter'
        with self.assertRaisesRegex(ValueError, 'stage binding|execution'):
            validate_execution_binding(chain, self.reference, self.context)

    def test_wrong_new_uid_source_pin_or_cpu_grant_is_rejected(self):
        from stage_binding import validate_execution_binding
        mutations = [('execution', 'uid', 'unverified-uid'),
                     ('source_receipt', 'sha256', '0' * 64),
                     ('source_receipt', 'path', '/other/source.json'),
                     ('cpu_allocation', 'sha256', '0' * 64),
                     ('cpu_allocation', 'executor_id', 'another-executor'),
                     ('cpu_allocation', 'threads', self.context['cpu_allocation']['threads'] + 1),
                     ('cpu_allocation', 'mask', [])]
        for section, key, value in mutations:
            binding = self.binding_fixture()
            binding[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_execution_binding(self.chain, self.reference, self.context, binding)

    def test_missing_duplicate_or_changed_asset_and_evidence_are_rejected(self):
        from stage_binding import validate_execution_binding
        for mutation in ('missing', 'duplicate', 'timestamp', 'evidence', 'sha', 'metadata_only', 'content'):
            binding = self.binding_fixture()
            if mutation == 'missing':
                binding['assets'].pop()
            elif mutation == 'duplicate':
                binding['assets'].append(copy.deepcopy(binding['assets'][0]))
            elif mutation == 'timestamp':
                binding['assets'][0]['current_identity']['ctime_ns'] -= 1
            elif mutation == 'evidence':
                binding['assets'][0]['evidence']['source_receipt_sha256'] = '0' * 64
            elif mutation == 'sha':
                binding['assets'][0]['sha256'] = '0' * 64
            elif mutation == 'metadata_only':
                binding['assets'][0]['evidence']['kind'] = 'unchanged_shared_object'
            else:
                path = Path(binding['assets'][0]['current_identity']['path'])
                with path.open('r+b') as stream:
                    stream.write(b'CD')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_execution_binding(self.chain, self.reference, self.context, binding)

    def test_missing_historical_timestamp_rejects_builder_and_consumer(self):
        from stage_binding import build_stage_binding, validate_execution_binding
        binding = self.binding_fixture()
        del self.chain['files']['pytorch_model.bin']['mtime_ns']
        with self.assertRaisesRegex(ValueError, 'incomplete|missing'):
            build_stage_binding(self.chain, self.reference, self.context)
        with self.assertRaisesRegex(ValueError, 'incomplete|missing'):
            validate_execution_binding(self.chain, self.reference, self.context, binding)

    def test_changed_fixed_runtime_snapshot_is_rejected(self):
        from stage_binding import validate_execution_binding
        binding = self.binding_fixture()
        code = Path(self.chain['runtime_snapshot']['root']) / 'models.py'
        code.write_text('# changed test source\n')
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            validate_execution_binding(self.chain, self.reference, self.context, binding)


def main():
    global FIXTURE, SCRATCH, LIVE_CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'source-receipt-sha', 'scratch', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--review-checks-only', action='store_true',
                        help='Run only timestamp and stage-binding review regressions, without the old eight tests.')
    parser.add_argument('--small-checks-only', action='store_true',
                        help='Run only the new Small mapping boundary check; no historical tests or model load.')
    args = parser.parse_args()
    context = cpu_context()
    chain = read_pinned_receipt(args.source_receipt, args.source_receipt_sha)
    SCRATCH = args.scratch
    LIVE_CONTEXT = context
    if args.small_checks_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SmallMappingTest)
    elif args.review_checks_only:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                                   for cls in (ReusedIdentityTest, StageBindingTest))
    else:
        configure_torch_threads(context)
        assets = chain['files']
        read = lambda name: json.loads(Path(assets[name]['path']).read_text())
        FIXTURE = [json.loads(Path(chain['provenance']['path']).read_text()), read('source-manifest.json'),
                   read('model-api.json'), read('config.json'), read('download-receipt.json'),
                   Path(assets['README.md']['path']).read_text()]
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PretrainingSourcesTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_new_json(args.output, {'passed': result.wasSuccessful(), 'execution': context,
                                'scope': ('small_mapping_only' if args.small_checks_only else
                                          'review_regressions_only' if args.review_checks_only else 'original_source_checks'),
                                'tests_run': result.testsRun, 'failures': len(result.failures),
                                'errors': len(result.errors), 'source_receipt_sha256': args.source_receipt_sha,
                                'source_files': {name: sha256(Path(__file__).resolve().parents[1] / name)
                                                 for name in ('tools/pretraining/source_chain.py',
                                                              'tools/pretraining/strict_load.py',
                                                              'tools/pretraining/stage_binding.py',
                                                              'tests/test_pretraining_sources.py')}})
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()

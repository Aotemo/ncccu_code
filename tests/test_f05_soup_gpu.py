"""CPU-only synthetic isolation checks for the explicit F05 soup dev mode."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/bundle'), str(ROOT / 'tools/diagnostics')]
import f04_gpu_full_dev as gpu
import f05_soup_dev as soup


class SoupDevTest(unittest.TestCase):
    def inputs(self):
        names = soup.GROUPS
        policy = {'version': 'f05-uniform-soup-dev-20261105-v1', 'groups': names,
                  'precisions': ['fp32', 'fp16'], 'base_manifest_sha256': 'a' * 64,
                  'soup_receipt_sha256': {name: 'b' * 64 for name in names},
                  'standalone_receipt_sha256': {name: 'c' * 64 for name in names},
                  'ranks_per_group': [3, 3, 2], 'batch_size': 8, 'dev_count': 5140,
                  'dev_positive': 167, 'official_read': False, 'tta': False}
        members = [{'round_id': name, 'soup': {'sha256': 'b' * 64},
                    'receipt': {'sha256': 'c' * 64}, 'exports': {'fp32': {}, 'fp16': {}}} for name in names]
        inputs = {'status': 'F05_uniform_soup_dev_inputs_sealed_no_GPU', 'members': members,
                  'manifest': {'sha256': 'a' * 64}, 'plan': soup.build_plan(names),
                  'batch_size': 8, 'dev_count': 5140, 'dev_positive': 167,
                  'TTA': False, 'official_read': False}
        return inputs, policy

    def test_rank_plan_and_unsafe_or_missing_inputs(self):
        inputs, policy = self.inputs()
        self.assertEqual([x['device_rank'] for x in inputs['plan']], list(range(8)))
        self.assertEqual([x['world'] for x in inputs['plan']], [3, 3, 3, 3, 3, 3, 2, 2])
        soup.validate_inputs(inputs, policy)
        for changed in (dict(inputs, TTA=True), dict(inputs, official_read=True),
                        dict(inputs, members=inputs['members'][:-1]), dict(inputs, plan=inputs['plan'][:-1])):
            with self.assertRaises(ValueError):
                soup.validate_inputs(changed, policy)
        for names in (['r120', 'r201', 'r134r1'], ['r120'] * 3, ['r120']):
            with self.assertRaises(ValueError):
                soup.build_plan(names)

    def test_preview_uses_all_fixed_replacements_and_actual_byte_sizes(self):
        labels = [0, 1, 0, 1]
        base = {name: [float(i), i + 3., i + 1., i + 4.] for i, name in enumerate('abcde')}
        soups = {name: {'fp32': [-2., 1., -1., 2.], 'fp16': [-2., 1., -1., 2.]}
                 for name in soup.GROUPS}
        result = soup.metrics_and_preview(labels, soups, base, {x: 10 for x in 'abcde'},
                                          {x: 12 for x in soup.GROUPS})
        self.assertEqual(len(result['replacement_preview_fp32']), 15)
        self.assertEqual({x['fp16_bytes'] for x in result['replacement_preview_fp32']}, {52})
        self.assertTrue(all(not row['selection'] for row in result['replacement_preview_fp32']))
        self.assertEqual(result['replacement_preview_fp16']['status'], 'unknown')
        self.assertEqual(result['soups']['r120']['fp32']['auroc'], 1.)
        bad = copy.deepcopy(soups)
        bad['r120']['fp16'][0] = float('nan')
        with self.assertRaises(ValueError):
            soup.metrics_and_preview(labels, bad, base, {x: 10 for x in 'abcde'},
                                     {x: 12 for x in soup.GROUPS})

    def test_receipts_bind_each_export_and_historical_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def put(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return {'path': str(path), 'sha256': gpu.sha(path)}
            bundle = root / 'bundle'; bundle.mkdir()
            code = bundle / 'model.py'; code.write_text('historical model')
            freeze = put(root / 'freeze.json', {'source_files': {'model.py': gpu.sha(code)}})
            site = put(root / 'site.json', {'data_root': '/data', 'prepared_cache': '/cache'})
            exports = {}
            for precision in soup.PRECISIONS:
                folder = root / precision; folder.mkdir()
                model_path = folder / 'model.pt'; model_path.write_bytes(precision.encode())
                model = {'path': str(model_path), 'sha256': gpu.sha(model_path),
                         'bytes': model_path.stat().st_size, 'tensor_sha256': 'a' * 64}
                metadata = put(folder / 'model.json', {'model_sha256': model['sha256'],
                               'checkpoint': {'tensor_sha256': 'a' * 64, 'variant': 'ema'}})
                exports[precision] = {'model': model, 'metadata': metadata}
            common = {'freeze': freeze, 'site': site, 'bundle': str(bundle),
                      'data_root': '/data', 'cache': '/cache'}
            standalone = put(root / 'standalone.json', {'status': 'uniform_soup_standalone_strict_synthetic_passed',
                **common, 'exports': exports, 'state_sha256': 'd' * 64, 'base_variant_for_loader': 'ema',
                'synthetic_forward_only': True, 'official_read': False, 'dev_read': False,
                'GPU_initialized': False, 'TTA': False})
            provenance = put(root / 'soup.json', {'status': 'uniform_soup_created_no_dev_evaluation',
                'group': 'r120', 'method': 'uniform_all_compatible_selected_states_v1',
                'standalone': standalone, 'average_state_sha256': 'd' * 64,
                'official_read': False, 'dev_read': False, 'GPU_initialized': False, 'TTA': False})
            member = {'round_id': 'r120', 'soup': provenance, 'receipt': standalone,
                      'exports': exports, 'selected': {'variant': 'ema'}, **common}
            soup.verify_assets(gpu, member)
            Path(exports['fp16']['model']['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                soup.verify_assets(gpu, member)


if __name__ == '__main__':
    paths = ('tools/diagnostics/f04_gpu_full_dev.py', 'tools/diagnostics/f05_soup_dev.py',
             'tests/test_f05_soup_gpu.py')
    print(json.dumps({'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                        for name in paths}}), flush=True)
    unittest.main()

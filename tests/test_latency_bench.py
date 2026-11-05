"""Synthetic checks for the F06 latency bench and T4 subset estimates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/diagnostics/latency_bench.py'
spec = importlib.util.spec_from_file_location('latency_bench', SOURCE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def job(image_size=4):
    return {'image_size': image_size, 'normalization': 'imagenet', 'prepared_size': 0,
            'age': None, 'model': 'convnext_small'}


def tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 1))


class FakeDataset:
    def __init__(self, count=8, age=False):
        self.count, self.age = count, age

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if index >= self.count:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(index)
        image = torch.rand(3, 4, 4, generator=generator)
        if self.age:
            return image, 0.0, index, torch.zeros(2)
        return image, 0.0, index


def write_manifest(root, entries):
    path = Path(root) / 'members.json'
    path.write_text(json.dumps({'format': 'f06-latency-members-v1', 'members': entries}))
    return path


def member_entry(root, identifier, *, content=b'weights', image_size=4):
    path = Path(root) / (identifier + '.bin')
    path.write_bytes(content)
    return {'id': identifier, 'model': {'path': str(path), 'sha256': bench.sha(path)}}, job(image_size)


def latency_report(root, passed, *, name='latency.json'):
    path = Path(root) / name
    subsets = {}
    for size, members in (('1', ['m0']), ('2', ['m0', 'm1']), ('3', ['m0', 'm1', 'm2'])):
        subsets[size] = {'members': members, 'pass': passed, 'conservative_total_ms': 50.0,
                         'by_flops': {'gpu_ms_per_image': 10.0, 'total_ms_per_image': 50.0,
                                      'with_safety_ms': 75.0, 'within_budget': passed},
                         'by_dcu_scale': {'gpu_ms_per_image': 20.0, 'total_ms_per_image': 55.0,
                                          'with_safety_ms': 82.5, 'within_budget': passed}}
    path.write_text(json.dumps({'format': 'f06-latency-bench-v1', 'pass': passed,
                                'subset_estimates': subsets,
                                'estimates': {'assumptions': {'t4_effective_tflops': 12.0,
                                                              'safety_factor': 1.5, 'budget_ms': 100.0}}}))
    return path


class LatencyBenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='f06-latency-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_members_bind_weights_sha_and_share_preprocessing(self):
        built = {}

        def loader(path, variant):
            identifier = Path(path).stem
            built[identifier] = tiny_model()
            return {'model': built[identifier], 'job': job()}

        first, _ = member_entry(self.root, 'm0')
        second, _ = member_entry(self.root, 'm1')
        path = write_manifest(self.root, [first, second])
        loaded = bench.load_members(path, loader=loader)
        self.assertEqual([row['id'] for row in loaded['members']], ['m0', 'm1'])
        self.assertEqual(loaded['sha256'], bench.sha(path))

        broken = dict(first, model={'path': first['model']['path'], 'sha256': '0' * 64})
        with self.assertRaises(ValueError):
            bench.load_members(write_manifest(self.root, [broken]), loader=loader)

        duplicate = [first, dict(second, id='m0')]
        with self.assertRaises(ValueError):
            bench.load_members(write_manifest(self.root, duplicate), loader=loader)

        calls = {'count': 0}

        def mismatched(path, variant):
            calls['count'] += 1
            return {'model': tiny_model(), 'job': job(image_size=64 if calls['count'] > 1 else 4)}

        with self.assertRaises(ValueError):
            bench.load_members(write_manifest(self.root, [first, second]), loader=mismatched)

    def test_measure_collects_batch_records_and_summarize_takes_medians(self):
        members = [{'id': 'm0', 'model': tiny_model(), 'job': job()},
                   {'id': 'm1', 'model': tiny_model(), 'job': job()}]
        records = bench.measure(members, FakeDataset(), list(range(8)), device=torch.device('cpu'),
                                precision='fp32', batch_size=4, warmup=1, repetitions=5)
        self.assertEqual(len(records), 10)
        for record in records:
            self.assertEqual(record['batch'], 4)
            for key in ('cpu', 'h2d', 'fuse', 'total', ('member', 'm0'), ('member', 'm1')):
                self.assertGreaterEqual(record[key], 0.0)
        summary = bench.summarize(records, ['m0', 'm1'])
        self.assertEqual(summary['batches'], 10)
        for key in ('cpu_preprocess_ms_per_image', 'h2d_ms_per_image', 'fuse_ms_per_image',
                    'total_ms_per_image'):
            self.assertGreater(summary[key], 0.0)
        self.assertGreater(summary['members']['m0'], 0.0)
        with self.assertRaises(ValueError):
            bench.measure(members, FakeDataset(), list(range(8)), device=torch.device('cpu'),
                          precision='fp32', batch_size=4, warmup=1, repetitions=4)

    def test_summarize_divides_by_batch_and_uses_median(self):
        records = [
            {'cpu': 0.010, 'h2d': 0.002, 'fuse': 0.001, 'total': 0.020,
             ('member', 'a'): 0.007, 'batch': 2},
            {'cpu': 0.030, 'h2d': 0.004, 'fuse': 0.003, 'total': 0.040,
             ('member', 'a'): 0.011, 'batch': 2},
            {'cpu': 0.020, 'h2d': 0.003, 'fuse': 0.002, 'total': 0.030,
             ('member', 'a'): 0.009, 'batch': 2},
        ]
        summary = bench.summarize(records, ['a'])
        self.assertAlmostEqual(summary['cpu_preprocess_ms_per_image'], 10.0, places=9)
        self.assertAlmostEqual(summary['total_ms_per_image'], 15.0, places=9)
        self.assertAlmostEqual(summary['members']['a'], 4.5, places=9)

    def test_t4_estimates_follow_the_declared_assumptions(self):
        members = [{'id': 'a', 'flops_per_image': 1e11, 'gpu_ms_per_image': 5.0},
                   {'id': 'b', 'flops_per_image': 2e11, 'gpu_ms_per_image': 9.0}]
        result = bench.estimate(members, 10.0, t4_tflops=12.0, dcu_t4_ratio=1.0)
        self.assertAlmostEqual(result['subset_estimates']['1']['by_flops']['gpu_ms_per_image'],
                               1e11 / 12e12 * 1000, places=9)
        self.assertEqual(result['subset_estimates']['1']['members'], ['a'])
        self.assertAlmostEqual(result['subset_estimates']['2']['by_dcu_scale']['gpu_ms_per_image'],
                               14.0, places=9)
        self.assertAlmostEqual(result['subset_estimates']['2']['by_dcu_scale']['total_ms_per_image'],
                               24.0, places=9)
        self.assertTrue(result['pass'])
        self.assertEqual(result['assumptions']['t4_effective_tflops'], 12.0)
        self.assertEqual(result['assumptions']['safety_factor'], 1.5)

        slow = bench.estimate(members, 60.0, t4_tflops=12.0, dcu_t4_ratio=1.0)
        self.assertFalse(slow['pass'])
        self.assertGreater(slow['subset_estimates']['2']['conservative_total_ms'], 60.0)

        unknown = bench.estimate([{'id': 'a', 'flops_per_image': None, 'gpu_ms_per_image': 5.0}],
                                 10.0, t4_tflops=12.0, dcu_t4_ratio=2.0)
        self.assertIsNone(unknown['subset_estimates']['1']['by_flops']['gpu_ms_per_image'])
        self.assertAlmostEqual(unknown['subset_estimates']['1']['by_dcu_scale']['gpu_ms_per_image'],
                               10.0, places=9)
        self.assertTrue(unknown['pass'])
        with self.assertRaises(ValueError):
            bench.estimate(members, 10.0, t4_tflops=0, dcu_t4_ratio=1.0)
        with self.assertRaises(ValueError):
            bench.estimate(members, -1.0, t4_tflops=12.0, dcu_t4_ratio=1.0)

    def test_flop_counter_reports_per_image_flops(self):
        member = {'id': 'a', 'model': tiny_model(), 'job': job()}
        images = torch.rand(4, 3, 4, 4)
        value, method = bench.count_flops(member, images, None, torch.device('cpu'), 'fp32', 4)
        self.assertEqual(method, 'torch_flop_counter')
        self.assertAlmostEqual(value, 2 * 48, places=6)

    def test_bench_end_to_end_on_tiny_cpu_images(self):
        from PIL import Image
        images = self.root / 'images'
        images.mkdir()
        for index in range(3):
            Image.new('RGB', (48 + index, 40), (index * 40, 10, 10)).save(
                images / ('img_' + str(index) + '.png'))
        weight = self.root / 'tiny.bin'
        weight.write_bytes(b'tiny')
        manifest = write_manifest(self.root, [
            {'id': 'm0', 'model': {'path': str(weight), 'sha256': bench.sha(weight)}}])
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 1))
        report = bench.run_bench(
            members_path=manifest, images_path=images, device='cpu', precision='fp32',
            batch_size=2, warmup=1, repetitions=5,
            loader=lambda path, variant: {'model': model, 'job': job(image_size=32)})
        self.assertEqual(report['format'], 'f06-latency-bench-v1')
        self.assertEqual(report['images']['count'], 3)
        self.assertEqual(report['protocol']['repetitions'], 5)
        self.assertGreater(report['measurements']['total_ms_per_image'], 0.0)
        self.assertEqual(set(report['subset_estimates']), {'1'})
        self.assertIsInstance(report['pass'], bool)
        self.assertFalse(report['official_read'])
        self.assertFalse(report['TTA'])
        self.assertEqual(report['measurements']['members'][0]['flops_method'], 'torch_flop_counter')
        self.assertGreater(report['measurements']['members'][0]['flops_per_image'], 0.0)


if __name__ == '__main__':
    unittest.main()

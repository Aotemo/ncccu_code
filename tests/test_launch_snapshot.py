"""Fresh Pod launch snapshots must match the admitted Pod and time window."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'runtime/bundle'))
spec = importlib.util.spec_from_file_location('launch_snapshot', root / 'tools/activation/validate_launch_pod.py')
validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)


class LaunchSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.site = {'schema_version': 1, 'round_id': 'r263-r1-fixture', 'git_commit': 'a' * 40,
                     'work_root': str(root), 'round_root': str(root / 'rounds/r263-r1-fixture'),
                     'pod': {'namespace': 'ns', 'name': 'pod', 'uid': '11111111-2222-4333-8444-555555555555',
                             'container': 'work', 'python': sys.executable},
                     'gpus': [{'ordinal': n, 'pci': '0000:%02x:00.0' % (n + 1),
                               'vram_total_bytes': 68702699520} for n in range(8)],
                     'metadata_root': str(root / 'assets/stage1'),
                     'public_weight': str(root / 'assets/public/weights'),
                     'public_provenance': str(root / 'assets/public/identity'),
                     'science': {'candidate': {'id': 'fixture'}}}
        self.pod = {'metadata': {'uid': self.site['pod']['uid'], 'name': 'pod', 'namespace': 'ns'},
                    'spec': {'containers': [{'name': 'work'}]},
                    'status': {'phase': 'Running', 'conditions': [{'type': 'Ready', 'status': 'True'}]}}

    def check(self, pod=None, observed=1000, now=1001):
        return validator.validate_snapshot(pod or self.pod, self.site, observed_at=observed, now=now,
                                           hostname='pod', python=sys.executable)

    def test_valid_fresh_snapshot_matches_exact_pod(self):
        self.assertTrue(self.check())

    def test_changed_uid_or_unready_or_wrong_container_is_rejected(self):
        for section, key, value in [('metadata', 'uid', 'wrong'),
                                    ('status', 'phase', 'Pending'),
                                    ('spec', 'containers', [{'name': 'other'}])]:
            bad = copy.deepcopy(self.pod); bad[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(bad)
        bad = copy.deepcopy(self.pod); bad['status']['conditions'][0]['status'] = 'False'
        with self.assertRaises(ValueError): self.check(bad)

    def test_stale_future_and_deleting_snapshots_are_rejected(self):
        for observed, now in ((1000, 1121), (1002, 1001)):
            with self.assertRaises(ValueError): self.check(observed=observed, now=now)
        bad = copy.deepcopy(self.pod); bad['metadata']['deletionTimestamp'] = '2026-11-05T00:00:00Z'
        with self.assertRaises(ValueError): self.check(bad)

    def test_snapshot_must_follow_activation_and_leave_launch_time(self):
        self.assertTrue(validator.validate_request_time(1001, 1000, 1002))
        for observed, request, now in ((999, 1000, 1001), (1000, 1000, 1046), (1002, 1000, 1001)):
            with self.assertRaises(ValueError):
                validator.validate_request_time(observed, request, now)


if __name__ == '__main__':
    unittest.main()

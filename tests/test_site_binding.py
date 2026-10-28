"""Bad Pod identities, escaped paths and swapped device ranks must fail closed."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


class SiteBindingTest(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('retrain.site_binding')
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.site = {
            'schema_version': 1, 'round_id': 'test-round',
            'git_commit': 'a' * 40, 'work_root': str(root),
            'round_root': str(root / 'rounds/test-round'),
            'pod': {'namespace': 'test-namespace', 'name': 'test-pod',
                    'uid': '11111111-2222-4333-8444-555555555555',
                    'container': 'test-container', 'python': sys.executable},
            'gpus': [{'ordinal': n, 'pci': '0000:%02x:00.0' % (n + 1),
                      'vram_total_bytes': 68702699520} for n in range(8)],
            'metadata_root': str(root / 'assets/stage1'),
            'public_weight': str(root / 'assets/public/converted.pt'),
            'public_provenance': str(root / 'assets/public/provenance.json'),
            'science': {'candidate': {'id': 'cpu-structural-fixture'}},
        }

    def test_site_has_no_machine_defaults_and_accepts_explicit_eight_card_binding(self):
        value = self.api.validate_site(self.site)
        self.assertEqual(value['pod']['name'], 'test-pod')
        self.assertEqual(self.api.expected_pcis(value),
                         ['0000:01:00.0', '0000:02:00.0', '0000:03:00.0', '0000:04:00.0',
                          '0000:05:00.0', '0000:06:00.0', '0000:07:00.0', '0000:08:00.0'])

    def test_duplicate_device_relative_path_and_root_escape_are_rejected(self):
        wrongs = []
        bad = copy.deepcopy(self.site); bad['gpus'][7]['pci'] = bad['gpus'][0]['pci']; wrongs.append(bad)
        bad = copy.deepcopy(self.site); bad['gpus'][7]['ordinal'] = 0; wrongs.append(bad)
        bad = copy.deepcopy(self.site); bad['gpus'] = bad['gpus'][:4]; wrongs.append(bad)
        bad = copy.deepcopy(self.site); bad['public_weight'] = '/outside/weight.pt'; wrongs.append(bad)
        bad = copy.deepcopy(self.site); bad['round_root'] = str(Path(self.temp.name) / '../outside'); wrongs.append(bad)
        bad = copy.deepcopy(self.site); bad['pod']['python'] = 'python'; wrongs.append(bad)
        bad = copy.deepcopy(self.site); bad['git_commit'] = ''; wrongs.append(bad)
        for bad in wrongs:
            with self.subTest(binding=bad), self.assertRaises(ValueError):
                self.api.validate_site(bad)

    def test_site_file_digest_is_checked_before_use(self):
        path = Path(self.temp.name) / 'SITE_BINDING.json'
        path.write_text(json.dumps(self.site))
        with self.assertRaises(ValueError):
            self.api.load_site(path, expected_sha256='0' * 64)

    def test_live_pod_requires_exact_uid_ready_interpreter_and_fresh_observation(self):
        args = {'pod_uid': self.site['pod']['uid'], 'ready': True, 'observed_at': 1000,
                'hostname': 'test-pod', 'python': sys.executable, 'now': 1001}
        self.api.verify_live(self.site, **args)
        for key, value in [('pod_uid', 'wrong'), ('ready', 'True'), ('hostname', 'old-pod'),
                           ('python', '/other/python'), ('observed_at', 0), ('observed_at', 1002)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.api.verify_live(self.site, **{**args, key: value})

    def test_rank_coverage_and_ordered_pci_mapping_are_both_required(self):
        proofs = [{'rank': n, 'world_size': 8, 'device': 'cuda:' + str(n),
                   'physical_pci': card['pci'], 'total_memory_bytes': 68702699520}
                  for n, card in enumerate(self.site['gpus'])]
        self.api.verify_rank_devices(proofs, self.site)
        wrongs = [proofs[:7]]
        bad = copy.deepcopy(proofs); bad[1]['rank'] = 0; wrongs.append(bad)
        bad = copy.deepcopy(proofs); bad[0]['physical_pci'], bad[1]['physical_pci'] = bad[1]['physical_pci'], bad[0]['physical_pci']; wrongs.append(bad)
        bad = copy.deepcopy(proofs); bad[7]['world_size'] = 4; wrongs.append(bad)
        bad = copy.deepcopy(proofs); bad[7]['device'] = 'cuda:0'; wrongs.append(bad)
        for bad in wrongs:
            with self.subTest(proofs=bad), self.assertRaises(ValueError):
                self.api.verify_rank_devices(bad, self.site)


if __name__ == '__main__':
    unittest.main()

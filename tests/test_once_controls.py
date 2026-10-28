"""One-shot markers stay attached to a run, and component gates name their process."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'runtime/bundle')]


class OnceControlsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_dispatch_refuses_any_existing_reservation_without_spawning(self):
        spec = importlib.util.spec_from_file_location('launch_under_test', ROOT / 'runtime/launch_once.py')
        launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)
        control = self.folder / 'control'; control.mkdir()
        launch.ensure_new_dispatch(self.folder)
        for name in ('dispatch.json', 'controller-start.json', 'pod-admission.json'):
            marker = control / name; marker.write_text('{}')
            with self.subTest(marker=name), self.assertRaises(FileExistsError):
                launch.ensure_new_dispatch(self.folder)
            marker.unlink()

    def test_official_reservation_cannot_move_to_another_output_or_copied_freeze(self):
        from retrain.single91 import reserve_official
        checkpoint = self.folder / 'run/checkpoints/selected.pt'
        checkpoint.parent.mkdir(parents=True)
        value = {'checkpoint': str(checkpoint), 'variant': 'ema',
                 'blob_sha256': '1' * 64, 'tensor_sha256': '2' * 64}
        freeze = self.folder / 'freeze.json'; freeze.write_text(json.dumps(value))
        copied = self.folder / 'copied-freeze.json'; copied.write_text(freeze.read_text())
        with patch('retrain.single91.validate_freeze', return_value=value):
            reserve_official(freeze, self.folder / 'output1')
            with self.assertRaises(FileExistsError):
                reserve_official(copied, self.folder / 'output2')
        reservation = json.loads((self.folder / 'run/OFFICIAL-ONCE.json').read_text())
        self.assertEqual(reservation['output_root'], str(self.folder / 'output1'))

    def test_selector_requires_all_sixteen_dev_readouts_and_uses_fixed_ties(self):
        from retrain.single91 import select_checkpoint
        entries = [{'cycle': cycle, 'variant': variant, 'successful_updates': cycle * 800,
                    'error_twice': 100, 'ap': .9}
                   for cycle in range(1, 9) for variant in ('online', 'ema')]
        with patch('retrain.single91.ex.schedule', return_value={'steps_per_cycle': 800}):
            self.assertEqual(select_checkpoint(entries)['variant'], 'ema')
            self.assertEqual(select_checkpoint(entries)['successful_updates'], 800)
            entries[-1].update(error_twice=99, ap=.1)
            self.assertIs(select_checkpoint(entries), entries[-1])
            with self.assertRaises(ValueError):
                select_checkpoint(entries[:-1])

    def test_wrong_process_gate_never_executes_the_component(self):
        gate = self.folder / 'gate.json'
        gate.write_text(json.dumps({'token': 'fixture-token', 'pid': -1, 'start_ticks': '0'}))
        marker = self.folder / 'component-ran.txt'
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'runtime/component_gate.py'),
                                 '--gate', str(gate), '--token', 'fixture-token', '--',
                                 sys.executable, '-B', '-c',
                                 'from pathlib import Path; Path(' + repr(str(marker)) + ').write_text("ran")'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()

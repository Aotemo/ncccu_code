"""Two synthetic checks for the one-shot round observer; A runs them on the Pod."""
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'observer'))
import observe_round_once as observer


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def write_stat(proc, pid, state, start_ticks):
    directory = Path(proc) / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = [state] + ['0'] * 18 + [str(start_ticks)] + ['0'] * 3
    (directory / 'stat').write_text(str(pid) + ' (proc) ' + ' '.join(fields) + '\n')


def write_card(drm, pci, busy=0, ordinal=0):
    device = Path(drm).parent / pci
    device.mkdir(parents=True, exist_ok=True)
    (device / 'gpu_busy_percent').write_text(str(busy) + '\n')
    card = Path(drm) / ('card' + str(ordinal))
    card.mkdir(parents=True, exist_ok=True)
    (card / 'device').symlink_to(device, target_is_directory=True)


def build_round(root, status, done_status=None, world=1, final=True):
    formal = Path(root)
    write_json(formal / 'TRAIN_CONFIG.json', {'run_name': 'run-x', 'jobs': [{'name': 'job-x'}]})
    folder = formal / 'outputs' / 'run-x' / 'job-x'
    write_json(folder / 'status.json', status)
    proof = {'phase': 'warmup', 'active_roles': ['head', 'age'],
             'role_state': {'backbone': {'state_entries': 0}}}
    lines = [{'event': 'update', 'at': 1, 'successful_updates': 1600,
              'attempted_updates': 1600, 'cycle': 2, 'loss': .4}]
    for update in (1, 1600, 1601):
        lines.append({'event': 'head_warmup_phase_proof', 'successful_updates': update,
                      'at': update, 'rank_proofs': [dict(proof, world_finite=True) for _ in range(8)]})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'metrics.jsonl').write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
    if done_status is not None:
        write_json(formal / 'control' / 'done.json',
                   {'status': done_status, 'phase': 'done', 'error': None, 'live_owned_ranks': []})
    if final:
        write_json(formal / 'SINGLE-MODEL-FREEZE.json',
                   {'status': 'completed', 'selected': {'auroc': .991, 'ap': .777,
                                                        'variant': 'ema', 'cycle': 4,
                                                        'successful_updates': 3200}})
        write_json(formal / 'evidence/TRAIN_RESULT.json',
                   {'status': 'completed', 'phase': 'done', 'successful_updates': 6400,
                    'cycles': 8, 'early_stop': {'bad_cycles': 0}, 'error': None,
                    'component': {'exit_code': 0, 'signals': []}, 'token': 'SECRET-TOKEN'})
        write_json(formal / 'evidence/ROUND_RESULT.json',
                   {'status': 'completed', 'phase': 'done', 'successful_updates': 6400, 'cycles': 8,
                    'official_status': 'not_run_by_policy', 'official_reason': 'threshold_not_met',
                    'official_evaluated': False, 'official_metrics': None,
                    'selected_internal_dev_auroc': .5})
        write_json(formal / 'official/OFFICIAL-RESULT.json',
                   {'status': 'not_run_by_policy', 'metrics': None,
                    'target_auroc_at_least_0_999': False})
    write_json(formal / 'SITE_BINDING.json',
               {'work_root': str(formal), 'pod': {'uid': 'uid-test'},
                'gpus': [{'ordinal': index, 'pci': '0000:0a:00.' + str(index)}
                         for index in range(world)],
                'cpu_execution': {'owner_mask': []}})
    (formal / 'control').mkdir(parents=True, exist_ok=True)
    (formal / 'control' / '.instance-owner-uid-test.lock').write_text('')
    return formal


class ObserveRoundOnceTest(unittest.TestCase):
    def test_completion_pending_only_reads_the_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            formal = Path(directory) / 'round'
            (formal / 'control').mkdir(parents=True)
            (formal / 'TRAIN_CONFIG.json').write_text('{not json')
            (formal / 'metrics.jsonl').write_text('{not json')
            receipt = Path(directory) / 'first.json'
            receipt.write_text('{not json')
            output = Path(directory) / 'out.json'
            with patch.object(Path, 'read_text',
                              side_effect=AssertionError('pending must read no file')):
                result = observer.observe(
                    formal, receipt, output, 'pending-tag', 'completion',
                    proc_root=Path(directory) / 'no-proc-file',
                    drm_root=Path(directory) / 'no-drm-file',
                    locks_path=Path(directory) / 'no-locks-file',
                    clock=lambda: 1234.0)
            expected = {
                'tool': 'observe_round_once', 'tag': 'pending-tag', 'mode': 'completion',
                'observed_at': 1234.0,
                'observed_at_beijing': datetime.fromtimestamp(
                    1234.0, ZoneInfo('Asia/Shanghai')).isoformat(),
                'formal_root': str(formal), 'signals_sent': False,
                'model_forward_executed': False, 'official_read': False,
                'done': {'present': False, 'status': None}, 'phase': 'pending',
                'first_success_receipt_read': False}
            self.assertEqual(result, expected)
            self.assertEqual(json.loads(output.read_text()), expected)
            for absent in ('safety', 'metrics', 'progress', 'identity', 'terminal', 'selected'):
                self.assertNotIn(absent, result)

    def test_running_and_terminal_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = build_round(root / 'terminal', {
                'status': 'completed', 'phase': 'done', 'successful_updates': 6400,
                'attempted_updates': 6400, 'skipped_updates': 0, 'evaluated_cycle': 8,
                'early_stop': {'bad_cycles': 0}, 'stop_reason': 'completed'}, 'completed',
                world=2)
            proc = root / 'proc'
            write_stat(proc, 100, 'Z', 111)
            drm = root / 'drm'
            write_card(drm, '0000:0a:00.0')
            write_card(drm, '0000:0a:00.1', ordinal=1)
            locks = root / 'locks'
            locks.write_text('')
            receipt = root / 'first.json'
            write_json(receipt, {'owner': {'pid': 100, 'start_ticks': 111},
                                 'rank_processes': [{'pid': 200 + index, 'start_ticks': 300 + index}
                                                    for index in range(2)]})
            output = root / 'terminal.json'
            result = observer.observe(formal, receipt, output, 'terminal-tag', 'completion',
                                      proc_root=proc, drm_root=drm, locks_path=locks,
                                      clock=lambda: 2000.0)
            self.assertEqual(result['phase'], 'terminal')
            self.assertTrue(result['safety']['safe'])
            self.assertEqual(result['safety']['identity']['owner']['state'], 'Z')
            self.assertTrue(all(status == 'missing' for status in
                                result['safety']['identity']['rank_status']))
            self.assertEqual(result['safety']['lock']['holders'], [])
            self.assertEqual(result['safety']['device_scan_denied'], 0)
            self.assertEqual(result['selected']['auroc'], .991)
            self.assertTrue(result['official_read'])
            self.assertNotIn('SECRET-TOKEN', output.read_text())
            self.assertNotIn('reaped', output.read_text())

            running_root = build_round(root / 'running', {
                'status': 'running', 'phase': 'train', 'successful_updates': 5,
                'attempted_updates': 5, 'skipped_updates': 0, 'evaluated_cycle': 0,
                'early_stop': {'bad_cycles': 0},
                'best_metrics': {'auroc': .97, 'ap': .71, 'variant': 'online',
                                 'cycle': 2, 'successful_updates': 1600, 'error_twice': 123},
                'ema_best_metrics': {'auroc': .98, 'ap': .72, 'variant': 'ema',
                                     'cycle': 3, 'successful_updates': 2400, 'error_twice': 100},
                'lineage_best': {'variant': 'ema', 'cycle': 3, 'successful_updates': 2400,
                                 'metrics': {'auroc': .981, 'ap': .73}},
                'selection': {'variant': 'online', 'cycle': 1, 'successful_updates': 800,
                              'auroc': .96, 'ap': .69}}, world=2, final=False)
            running_proc = root / 'running-proc'
            write_stat(running_proc, 100, 'R', 111)
            for index in range(2):
                write_stat(running_proc, 200 + index, 'R', 300 + index)
            running_receipt = root / 'running-first.json'
            write_json(running_receipt, {'owner': {'pid': 100, 'start_ticks': 111},
                                         'rank_processes': [{'pid': 200 + index,
                                                             'start_ticks': 300 + index}
                                                            for index in range(2)]})
            running = observer.observe(
                running_root, running_receipt, root / 'running.json', 'running-tag', 'progress',
                proc_root=running_proc, drm_root=root / 'absent-drm',
                locks_path=root / 'absent-locks', clock=lambda: 2100.0)
            self.assertEqual(running['phase'], 'running')
            self.assertTrue(running['identity']['owner_match'])
            self.assertTrue(all(running['identity']['rank_matches']))
            self.assertIsNone(running['safety'])
            self.assertIsNone(running['selected'])
            self.assertEqual(running['status_best']['best_metrics']['auroc'], .97)
            self.assertEqual(running['status_best']['ema_best_metrics']['auroc'], .98)
            self.assertEqual(running['status_best']['lineage_best']['metrics']['auroc'], .981)
            self.assertEqual(running['status_best']['selection']['auroc'], .96)
            self.assertEqual(set(running['head_warmup_phase_proofs']), {'1', '1600', '1601'})

            write_json(running_receipt, {'owner': {'pid': 100, 'start_ticks': 111},
                                         'rank_processes': [{'pid': 200 + index,
                                                             'start_ticks': 999 if index == 0 else 300 + index}
                                                            for index in range(2)]})
            mismatched = observer.observe(
                running_root, running_receipt, root / 'mismatch.json', 'mismatch-tag', 'progress',
                proc_root=running_proc, drm_root=root / 'absent-drm',
                locks_path=root / 'absent-locks', clock=lambda: 2200.0)
            self.assertFalse(mismatched['identity']['rank_matches'][0])
            self.assertIsNone(mismatched['safety'])

            unknown_receipt = root / 'unknown.json'
            write_json(unknown_receipt, {})
            unknown = observer.observe(
                formal, unknown_receipt, root / 'unknown-out.json', 'unknown-tag', 'progress',
                proc_root=proc, drm_root=drm, locks_path=locks, clock=lambda: 2300.0)
            self.assertIsNone(unknown['safety']['safe'])

            with patch.object(observer, 'scan_device_users', return_value=([], 1)):
                denied = observer.observe(
                    formal, receipt, root / 'denied.json', 'denied-tag', 'progress',
                    proc_root=proc, drm_root=drm, locks_path=locks, clock=lambda: 2400.0)
            self.assertIsNone(denied['safety']['safe'])
            self.assertEqual(denied['safety']['device_scan_denied'], 1)

            short_receipt = root / 'short.json'
            write_json(short_receipt, {'owner': {'pid': 100, 'start_ticks': 111},
                                       'rank_processes': [{'pid': 200, 'start_ticks': 300}]})
            short = observer.observe(
                formal, short_receipt, root / 'short-out.json', 'short-tag', 'completion',
                proc_root=proc, drm_root=drm, locks_path=locks, clock=lambda: 2500.0)
            self.assertIsNone(short['safety']['safe'])
            self.assertEqual(short['safety']['reason'], 'rank_binding_incomplete')

            empty_root = build_round(root / 'empty-gpu', {
                'status': 'completed', 'phase': 'done', 'successful_updates': 6400,
                'attempted_updates': 6400, 'skipped_updates': 0, 'evaluated_cycle': 8,
                'early_stop': {'bad_cycles': 0}}, 'completed', world=0)
            empty = observer.observe(
                empty_root, receipt, root / 'empty-out.json', 'empty-tag', 'completion',
                proc_root=proc, drm_root=drm, locks_path=locks, clock=lambda: 2600.0)
            self.assertIsNone(empty['safety']['safe'])
            self.assertEqual(empty['safety']['reason'], 'gpu_binding_unknown')


if __name__ == '__main__':
    unittest.main()

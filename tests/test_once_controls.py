"""One-shot markers stay attached to a run, and component gates name their process."""
import importlib.util
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import threading
from types import SimpleNamespace
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
        with patch('retrain.single91.ex.schedule', return_value={'steps_per_cycle': 800, 'total_updates': 6400}):
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

    def test_fit_census_parent_waits_with_lock_after_coordinator_exit(self):
        # An early coordinator return must not release the parent lock over a live rank.
        import shutil
        control = self.folder / 'control'; control.mkdir()
        (self.folder / 'evidence').mkdir()
        ranks = self.folder / 'ranks'; ranks.mkdir()
        shutil.copyfile(ROOT / 'runtime/component_gate.py', self.folder / 'component_gate.py')
        # Only the external GPU binding is replaced; the gate, flock and signals are real.
        bindings = self.folder / 'bundle/retrain'; bindings.mkdir(parents=True)
        (bindings / '__init__.py').write_text('')
        (bindings / 'site_binding.py').write_text('def load_site():return {}\ndef bind_cpu_role(*_a):pass\n')
        rank_code = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'
        coordinator = self.folder / 'coordinator.py'
        coordinator.write_text('import json,subprocess,sys,time\nfrom pathlib import Path\n'
            'p=subprocess.Popen([sys.executable,"-c",'+repr(rank_code)+'],start_new_session=True)\n'
            'time.sleep(.1)\n'
            's=Path("/proc")/str(p.pid)\n'
            'r={"pid":p.pid,"start_ticks":(s/"stat").read_text().rsplit(")",1)[1].split()[19],'
            '"argv":[v.decode() for v in (s/"cmdline").read_bytes().split(b"\\0") if v]}\n'
            'Path('+repr(str(ranks / 'rank-0-launch.json'))+').write_text(json.dumps(r))\n'
            'p=subprocess.Popen([sys.executable,"-c",'+repr(rank_code)+'],start_new_session=True)\n'
            'time.sleep(.1)\ns=Path("/proc")/str(p.pid)\n'
            'r={"pid":p.pid,"start_ticks":(s/"stat").read_text().rsplit(")",1)[1].split()[19],'
            '"argv":[v.decode() for v in (s/"cmdline").read_bytes().split(b"\\0") if v]}\n'
            'Path('+repr(str(self.folder / 'unregistered.json'))+').write_text(json.dumps(r))\n'
            'time.sleep(30)\n')
        lock = self.folder / 'instance.lock'; lock.touch()
        runner = self.folder / 'parent.py'
        runner.write_text('import fcntl,sys,time,types\nfrom pathlib import Path\n'
            'sys.path[:0]='+repr([str(ROOT / 'runtime'), str(ROOT / 'runtime/bundle')])+'\n'
            'import run_training as m\n'
            'm.ROOT=Path('+repr(str(self.folder))+');m.CONTROL=m.ROOT/"control";m.EVIDENCE=m.ROOT/"evidence"\n'
            'sys.modules["run_round"]=types.SimpleNamespace(runtime_safety_snapshot=lambda:{},budget=lambda *_a:{})\n'
            'import retrain.execution_policy as ep\nep.audit_cpu_tree=lambda *_a:{}\nm.load_site=lambda:{}\n'
            'f=open('+repr(str(lock))+',"r");fcntl.flock(f,fcntl.LOCK_EX)\n'
            'try:\n'
            ' try:m.run_component("fit-census",[sys.executable,"-B",'+repr(str(coordinator))+'],time.time()+5,'+repr(str(ranks))+',dict(__import__("os").environ),soft_deadline_at=time.time()+1)\n'
            ' except RuntimeError:pass\n'
            'finally:fcntl.flock(f,fcntl.LOCK_UN);f.close()\n')
        parent = subprocess.Popen([sys.executable, '-B', str(runner)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rank = None
        try:
            until = time.monotonic() + 6
            while time.monotonic() < until and not (ranks / 'rank-0-launch.json').exists() and parent.poll() is None:
                time.sleep(.02)
            self.assertTrue((ranks / 'rank-0-launch.json').exists(),
                (self.folder / 'evidence/fit-census-component.log').read_text() if parent.poll() is not None else 'rank not launched')
            rank = json.loads((ranks / 'rank-0-launch.json').read_text())
            time.sleep(.25)
            with lock.open() as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out, err = parent.communicate(timeout=6)
            self.assertEqual(parent.returncode, 0, (out, err))
            import run_training
            self.assertFalse(run_training.same(rank))
            loose_live=run_training.same(json.loads((self.folder / 'unregistered.json').read_text()))
            terminal = json.loads((control / 'fit-census-component-terminal.json').read_text())
            self.assertEqual(terminal['live_owned_ranks'], [])
            self.assertNotEqual(terminal['status'], 'completed')
            with lock.open() as contender:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if parent.poll() is None:
                parent.kill(); parent.wait()
            if rank is not None:
                import run_training
                if run_training.same(rank):
                    os.killpg(rank['pid'], signal.SIGKILL)
            if (self.folder / 'unregistered.json').exists():
                import run_training
                loose=json.loads((self.folder / 'unregistered.json').read_text())
                if run_training.same(loose):os.killpg(loose['pid'],signal.SIGKILL)
        # The actual diagnostic main must not catch its final deadline into unbounded finally I/O.
        hard = self.folder / 'hard'; hard.mkdir(); (hard/'evidence').mkdir(); (hard/'control').mkdir()
        runner.write_text('import sys,time,types\nfrom pathlib import Path\n'
            'sys.path[:0]='+repr([str(ROOT / 'runtime'), str(ROOT / 'runtime/bundle')])+'\n'
            'import run_training as m,launch_once\nm.ROOT=Path('+repr(str(hard))+');m.CONTROL=m.ROOT/"control";m.EVIDENCE=m.ROOT/"evidence"\n'
            'born=int(Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19])/__import__("os").sysconf("SC_CLK_TCK")\n'
            'm.time.clock_gettime=lambda _:born+1797\n'
            'launch_once.load_fit_context=lambda *_a:time.sleep(20)\n'
            'raise SystemExit(m.run_fit_census(types.SimpleNamespace(fit_context=None,fit_context_sha256=None,token="fixture")))\n')
        before=time.monotonic()
        result=subprocess.run([sys.executable,'-B',str(runner)],timeout=5,capture_output=True)
        self.assertEqual((loose_live,result.returncode),(False,124),result.stderr)
        self.assertLess(time.monotonic()-before,5)
        self.assertIsNone(json.loads((hard/'evidence/FIT-CENSUS-RESULT.json').read_text())['metrics'])

    def test_fit_census_selected_full_order_transient_job_and_no_csv(self):
        # Wrong selection, job mutation, CSV output, and partial metrics change the result.
        import torch
        from retrain import inference, parallel_evaluation, distributed, execution_policy, site_binding
        from retrain.persistence import canonical_digest
        spec = importlib.util.spec_from_file_location('evaluate_under_test', ROOT / 'runtime/bundle/parallel_evaluate.py')
        evaluate = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluate)
        import launch_once
        manifest = {'rows': [{'name': str(i), 'label': int(i == 0), 'sha256': 'a' * 64} for i in range(4)],
                    'fit_indices': [2, 0, 3, 1]}
        manifest_path = self.folder / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
        stored_job = {'model': 'fixture', 'image_size': 512, 'prepared_size': 512, 'precision': 'fp32',
                      'physical_batch': 8, 'workers': 1, 'eval_workers': 1,
                      'runtime_deadline_at': 2, 'age': {'enabled': True, 'contract_sha256': 'a' * 64}}
        before = copy.deepcopy(stored_job)
        selected = {'variant': 'ema', 'cycle': 4, 'successful_updates': 3200,
                    'tensor_sha256': '1' * 64, 'blob_sha256': '2' * 64,
                    'checkpoint': str(self.folder / 'selected.pt'), 'dev_auroc': .75}
        context = {'selected': selected, '_frozen': {'job': before}, 'manifest': {'path': str(manifest_path)},
                   'counts': {'count': 4, 'positive_count': 1, 'negative_count': 3}}
        origin = {'variant': 'ema', 'tensor_sha256': '1' * 64, 'sha256': '2' * 64,
                  'manifest_digest': canonical_digest(manifest), 'successful_updates': 3200}
        ctx = SimpleNamespace(rank=0, world_size=1, device=torch.device('cpu'), primary=True,
                              values=lambda x: [x], close=lambda: None)
        output = self.folder / 'aggregate.json'
        context['output']=str(output)
        value = {'mode': 'fit-census', 'context': str(self.folder / 'context.json'), 'context_sha256': '3' * 64,
                 'source': selected['checkpoint'], 'variant': 'ema', 'source_sha256': '2' * 64,
                 'split': 'fit',
                 'devices': ['cpu'], 'init_method': 'file:///unused-cpu-fixture', 'deadline_at': time.time()+30, 'output': str(output),
                 'manifest': str(manifest_path), 'data_root': str(self.folder), 'precision': 'fp32'}
        value['coordinator']={'pid':os.getppid(),'start_ticks':(Path('/proc')/str(os.getppid())/'stat').read_text().rsplit(')',1)[1].split()[19]}
        value['owner']={'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}
        path = self.folder / 'worker-spec.json'; path.write_text(json.dumps(value))
        def consume(model, rows, indices, data_root, job, device, rank_ctx, output_csv=None, **kwargs):
            self.assertEqual(indices, [2, 0, 3, 1]); self.assertIsNone(output_csv)
            self.assertEqual(kwargs['variant'], 'ema'); self.assertEqual(kwargs.get('view'), 'identity')
            self.assertEqual((job['physical_batch'], job['workers'], job['eval_workers']), (32, 0, 0))
            self.assertEqual(job['runtime_deadline_at'], value['deadline_at'])
            self.assertEqual(job['age'], before['age']); self.assertIsNot(job['age'], stored_job['age'])
            self.assertEqual(stored_job, before)
            return {'count': 4, 'positive_count': 1, 'negative_count': 3, 'auroc': 1/3, 'error_twice': 4}
        args = SimpleNamespace(worker_spec=str(path), rank=0, device='cpu')
        supervised={};stop=threading.Event()
        def confirm():
            import run_training
            while not stop.is_set():
                run_training.own_ranks(self.folder/'processes',owned=supervised)
                time.sleep(.01)
        supervisor=threading.Thread(target=confirm,daemon=True);supervisor.start()
        self.addCleanup(lambda:(stop.set(),supervisor.join(timeout=2),[os.close(row['fd']) for row in supervised.values()]))
        with patch.object(launch_once, 'load_fit_context', return_value=context, create=True), \
             patch.object(execution_policy, 'bind_worker_cpu'), patch.object(execution_policy, 'rank_proof', return_value={}), \
             patch.object(site_binding, 'load_site', return_value={}), patch.object(site_binding, 'verify_rank_devices'), \
             patch.object(distributed, 'RankContext', return_value=ctx), \
             patch.object(inference, '_load_model', return_value=(torch.nn.Identity(), stored_job, {}, origin, ctx.device)), \
             patch.object(parallel_evaluation, 'evaluate', side_effect=consume), \
             patch.object(torch, 'set_num_interop_threads'):
            try:
                code = evaluate.worker(args)
            except Exception as exc:
                self.fail('fit-only worker did not reach the existing consumer: '+repr(exc))
            self.assertEqual(code, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report['metrics']['count'], 4)
            self.assertEqual(report['metrics']['error_twice'], 4)
            self.assertEqual(stored_job, before)
            output.unlink()
            with patch.object(parallel_evaluation, 'evaluate', return_value={'count': 3, 'positive_count': 1,
                              'negative_count': 2, 'auroc': .5, 'error_twice': 2}):
                with self.assertRaises(ValueError):
                    evaluate.worker(args)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

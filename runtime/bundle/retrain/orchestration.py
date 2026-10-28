"""Bounded local worker scheduling with process-group signal forwarding."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def supervise(tasks, devices, log_dir, grace_seconds=90, heartbeat_seconds=60, stop_event=None,
              slot_affinities=None):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stop = stop_event or threading.Event()
    active, results, pending = {}, {}, list(tasks)
    failed, deadline = False, None
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda *_: stop.set())

    def relay(proc, name, output):
        with output.open('a', encoding='utf-8') as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(f'[{name}] {line}', end='', flush=True)

    def terminate(proc, sig):
        try:
            # The training parent owns its DataLoader children. Let it finish
            # the current update and save before destroying that process group.
            if sig == signal.SIGKILL:
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
        except ProcessLookupError:
            pass

    def start(slot, task):
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
        argv = list(task['argv']) + ['--device', devices[slot]]
        if slot_affinities and slot_affinities[slot]:
            argv += ['--cpu-affinity', ','.join(map(str, slot_affinities[slot]))]
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, start_new_session=True)
        thread = threading.Thread(target=relay, args=(proc, task['name'], log_dir/(task['name']+'.log')), daemon=True)
        thread.start()
        active[slot] = (task, proc, thread)

    last_heartbeat = 0
    try:
        while pending or active:
            # Observe all exits before filling a free slot, so a failed sibling prevents queue expansion.
            for slot, (task, proc, thread) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                thread.join(timeout=3)
                proc.stdout.close()
                results[task['name']] = code
                if code == 0:
                    try:
                        marker = json.loads((Path(task['run_dir'])/'complete.json').read_text())
                        if marker.get('status') != 'completed':
                            raise ValueError('completion marker not completed')
                        if task.get('mode') and marker.get('mode') != task['mode']:
                            raise ValueError('completion marker has wrong mode')
                    except (OSError, ValueError):
                        failed = True
                elif not stop.is_set():
                    failed = True
                del active[slot]
            if failed or stop.is_set():
                pending.clear()
                if deadline is None:
                    deadline = time.monotonic() + grace_seconds
                    for _, proc, _ in active.values():
                        terminate(proc, signal.SIGTERM)
                elif time.monotonic() >= deadline:
                    for _, proc, _ in active.values():
                        terminate(proc, signal.SIGKILL)
            else:
                for slot in range(len(devices)):
                    if slot not in active and pending:
                        start(slot, pending.pop(0))
            now = time.monotonic()
            if now-last_heartbeat >= heartbeat_seconds:
                states = {}
                for task, _, _ in active.values():
                    try:
                        states[task['name']] = json.loads((Path(task['run_dir'])/'status.json').read_text())
                    except (OSError, ValueError):
                        states[task['name']] = {'status': 'starting'}
                print(json.dumps({'event': 'heartbeat', 'active': states, 'queued': len(pending)}, ensure_ascii=False), flush=True)
                last_heartbeat = now
            if active:
                time.sleep(.1)
    finally:
        for _, proc, _ in active.values():
            terminate(proc, signal.SIGTERM)
        for _, proc, thread in active.values():
            try:
                proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                terminate(proc, signal.SIGKILL)
                proc.wait()
            thread.join(timeout=3)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return {'status': 'failed' if failed else 'interrupted' if stop.is_set() else 'completed',
            'exit_codes': results}

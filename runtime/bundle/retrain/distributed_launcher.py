"""All-or-nothing rank supervision, with checkpoint-safe user interruption."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from .persistence import atomic_json


def supervise_ranks(tasks,control,grace_seconds=150,stop_event=None,env=None):
    control=Path(control);control.mkdir(parents=True,exist_ok=True)
    stop=stop_event or threading.Event()
    previous={};active={};codes={};failed=False;deadline=None
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,lambda *_:stop.set())
    try:
        if not stop.is_set():
            for task in tasks:
                log=(control/(task['name']+'.log')).open('ab')
                proc=subprocess.Popen(task['argv'],stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,env=env,start_new_session=True)
                log.close();active[task['name']]=proc
                stat=Path('/proc')/str(proc.pid)/'stat'
                receipt={'pid':proc.pid,'argv':task['argv'],'host':socket.gethostname(),'at':time.time(),
                    'start_ticks':stat.read_text().split()[21] if stat.exists() else None}
                atomic_json(control/(task['name']+'-launch.json'),receipt)
        while active:
            for name,proc in list(active.items()):
                code=proc.poll()
                if code is None:continue
                codes[name]=code
                if code!=0 and not stop.is_set():failed=True
                del active[name]
            if failed or stop.is_set():
                if deadline is None:
                    deadline=time.monotonic()+grace_seconds
                    for proc in active.values():
                        try:proc.send_signal(signal.SIGTERM)
                        except ProcessLookupError:pass
                elif time.monotonic()>=deadline:
                    for proc in active.values():
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
            if active:time.sleep(.1)
        return {'status':'failed' if failed else 'interrupted' if stop.is_set() else 'completed',
                'exit_codes':codes,'expected_ranks':len(tasks)}
    finally:
        for proc in active.values():
            try:proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:pass
        for proc in active.values():
            try:proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for sig,handler in previous.items():signal.signal(sig,handler)

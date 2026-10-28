#!/usr/bin/env python3
"""Wait for an exact owner receipt before replacing this process with work."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gate',required=True)
    parser.add_argument('--token',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command and args.command[0]=='--' else args.command
    if not command:raise ValueError('component command missing')
    gate=Path(args.gate);end=time.monotonic()+60
    while not gate.exists():
        if gate.is_symlink():raise ValueError('gate symlink forbidden')
        if time.monotonic()>=end:raise TimeoutError('component owner release timed out')
        time.sleep(.05)
    if gate.is_symlink():
        raise ValueError('component owner release identity differs')
    record=json.loads(gate.read_text())
    ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
    if (record.get('token')!=args.token or type(record.get('pid')) is not int
            or record['pid']!=os.getpid() or str(record.get('start_ticks'))!=ticks):
        raise ValueError('component token/PID/start identity differs')
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'bundle'))
    from retrain.site_binding import load_site, bind_cpu_role
    bind_cpu_role(load_site(), 'control')
    os.execvpe(command[0],command,os.environ.copy())


if __name__=='__main__':main()

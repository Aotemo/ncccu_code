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
    if gate.is_symlink() or json.loads(gate.read_text()).get('token')!=args.token:
        raise ValueError('component owner release identity differs')
    os.execvpe(command[0],command,os.environ.copy())


if __name__=='__main__':main()

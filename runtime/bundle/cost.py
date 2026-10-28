#!/usr/bin/env python3
"""Single-image timing entry sharing the exact frozen prediction implementation."""
import json
import sys

# predict sets the same thread limits before importing the shared implementation.
from predict import main

if __name__ == '__main__':
    try:
        raise SystemExit(main(['latency', *sys.argv[1:]]))
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise SystemExit(1)

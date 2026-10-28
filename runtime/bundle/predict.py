#!/usr/bin/env python3
"""Standalone deployment entry; all paths may be used from any current directory."""
import os
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import json
import sys

from retrain.inference import main

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise SystemExit(1)

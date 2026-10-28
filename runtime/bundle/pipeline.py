#!/usr/bin/env python3
"""Run the sealed benchmark -> full cycle -> fresh formal pipeline."""
import os
for _name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name] = '1'
import argparse
from retrain.pipeline import run_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    raise SystemExit(run_pipeline(parser.parse_args().plan))

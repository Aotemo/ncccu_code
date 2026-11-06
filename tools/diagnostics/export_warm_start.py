"""Export a selected EMA checkpoint as the registered fp32 warm-start state dict.

The export pins the expected tensor digest and update count so a mismatched
checkpoint cannot silently become another round's initialization.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-tensor-sha256', required=True)
    parser.add_argument('--expected-updates', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    import torch
    from retrain.ema import tensor_digest
    from retrain.persistence import load_file

    checkpoint = Path(args.checkpoint)
    payload = load_file(checkpoint, map_location='cpu')
    if payload.get('successful_updates') != args.expected_updates:
        raise SystemExit('source checkpoint update count differs')
    ema = payload.get('ema') or {}
    state = ema.get('model')
    if (not isinstance(state, dict) or ema.get('update_count') != args.expected_updates
            or tensor_digest(state) != args.expected_tensor_sha256):
        raise SystemExit('registered selected EMA tensor identity differs')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'warmstart.pt'
    if target.exists():
        raise SystemExit('warm-start export already exists')
    torch.save({name: tensor.detach().cpu().clone() for name, tensor in state.items()}, target)
    receipt = {'status': 'warm_start_exported', 'format': 'exported-fp32-state-dict-v1',
               'path': str(target), 'sha256': sha(target), 'tensor_sha256': tensor_digest(state),
               'tensors': len(state), 'variant': 'ema', 'updates': args.expected_updates,
               'source_model': str(checkpoint), 'source_model_sha256': sha(checkpoint),
               'GPU_initialized': False, 'official_read': False}
    (output / 'WARMSTART.json').write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()

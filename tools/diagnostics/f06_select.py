"""Select the unique F06 fold checkpoint by held-out official AUROC and export fp16 weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'
sys.path.insert(0, str(BUNDLE))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False).encode() + b'\n'
    with path.open('xb') as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    import torch
    from retrain.ema import tensor_digest
    from retrain.persistence import CheckpointStore
    store = CheckpointStore(Path(args.model_dir) / 'checkpoints')
    index = json.loads((store.directory / 'index.json').read_text())
    variants = []
    online = store.load('best', map_location='cpu')
    variants.append(('online', online, online['best_metrics']))
    if 'best_ema' in index.get('protected', {}):
        ema = store.load('best_ema', map_location='cpu')
        variants.append(('ema', ema, ema['ema_best_metrics']))
    if any(payload['job'].get('f06', {}).get('fold') != args.fold for _, payload, _ in variants):
        raise SystemExit('selected checkpoint fold identity differs')
    if len({payload['manifest_digest'] for _, payload, _ in variants}) != 1:
        raise SystemExit('selected checkpoints disagree on the manifest')
    def key(item):
        variant, payload, metrics = item
        return (metrics['auroc'], metrics['ap'], -metrics['successful_updates'], 1 if variant == 'ema' else 0)
    variant, payload, metrics = max(variants, key=key)
    state = payload['model'] if variant == 'online' else payload['ema']['model']
    meta = {name: tensor.half().clone() for name, tensor in state.items()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / 'model.pt'
    with model_path.open('xb') as stream:
        torch.save(meta, stream)
    job = payload['job']
    runtime = {'threshold': .5, 'precision': 'fp32', 'image_size': job['image_size'],
               'normalization': job['normalization'], 'prepared_size': job['prepared_size'],
               'preprocessing': ('rgb-conservative-border-prepared-letterbox-v2' if job.get('prepared_size')
                                 else 'rgb-conservative-border-letterbox-v1')}
    ema_meta = {key: value for key, value in payload['ema'].items() if key != 'model'} if payload.get('ema') else None
    metadata = {'format': 'glaucoma-frozen-inference-v1', 'model_sha256': sha(model_path),
                'job': job, 'age': job.get('age'), 'inference': runtime,
                'checkpoint': {'variant': variant, 'tensor_sha256': tensor_digest(meta),
                               'storage': 'float16', 'successful_updates': metrics['successful_updates'],
                               'cycle': metrics['cycle'], 'auroc': metrics['auroc'], 'ap': metrics['ap'],
                               'ema': ema_meta},
                'model_bytes': model_path.stat().st_size,
                'selection': 'maximum_heldout_official_auroc_then_ap_then_earlier_updates'}
    metadata_sha = write_new(output / 'model.json', metadata)
    summary = {'status': 'f06_fold_model_selected_and_exported', 'fold': args.fold,
               'model_dir': str(args.model_dir), 'export': str(output), 'variant': variant,
               'auroc': metrics['auroc'], 'ap': metrics['ap'],
               'successful_updates': metrics['successful_updates'], 'cycle': metrics['cycle'],
               'tensor_sha256': metadata['checkpoint']['tensor_sha256'],
               'model_sha256': metadata['model_sha256'], 'model_bytes': metadata['model_bytes'],
               'model_json_sha256': metadata_sha,
               'candidates': [{'variant': v, 'auroc': m['auroc'], 'ap': m['ap'],
                               'successful_updates': m['successful_updates']}
                              for v, _, m in variants]}
    print(json.dumps(summary | {'summary_sha256': write_new(args.summary, summary)}, indent=1))


if __name__ == '__main__':
    raise SystemExit(main())

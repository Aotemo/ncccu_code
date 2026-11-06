"""F07: export one F05 candidate round's selected checkpoint as a standalone fp16 model.

Runs in a fresh interpreter per member (historical site binding). CPU only.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def export_metadata(job, model_sha256, tensor_sha256, *, variant='ema'):
    import copy
    import re
    if (not isinstance(job, dict) or not job.get('model') or variant not in ('ema', 'online')
            or not all(re.fullmatch(r'[0-9a-f]{64}', value or '')
                       for value in (model_sha256, tensor_sha256))):
        raise ValueError('complete model and state identities are required')
    return {'format': 'glaucoma-frozen-inference-v1', 'model_sha256': model_sha256,
            'job': copy.deepcopy(job), 'age': copy.deepcopy(job.get('age')),
            'checkpoint': {'variant': variant, 'tensor_sha256': tensor_sha256},
            'inference': {'threshold': .5, 'precision': 'fp32'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze
    if sha(freeze_path) is None:
        raise ValueError('freeze unreadable')
    freeze = json.loads(freeze_path.read_text())
    root = freeze_path.parent
    bundle = root / 'bundle'
    for relative, expected in freeze['source_files'].items():
        if sha(bundle / relative) != expected:
            raise ValueError('frozen source differs: ' + relative)
    site_path = root / 'SITE_BINDING.json'
    runtime = json.loads((root / 'RUNTIME_BINDING.json').read_text())
    if sha(site_path) != runtime['site_binding_sha256']:
        raise ValueError('historical frozen site differs')
    if 'retrain' in sys.modules:
        raise ValueError('each historical model requires a fresh interpreter')
    os.environ.update(NCCCU_SITE_BINDING=str(site_path), NCCCU_SITE_BINDING_SHA256=sha(site_path))
    sys.path.insert(0, str(bundle))
    import torch
    threads = os.environ.get('NCCCU_CPU_THREADS')
    torch.set_num_threads(max(1, int(threads) - 2) if threads else 4)
    torch.set_num_interop_threads(1)
    from retrain.inference import _load_model
    from retrain.age import forward as age_forward
    from retrain.ema import tensor_digest
    variant = freeze['variant']
    if variant not in ('ema', 'online'):
        raise ValueError('historical variant differs')
    model, job, metadata, origin, device = _load_model(freeze['checkpoint'], device='cpu', variant=variant)
    if origin['sha256'] != freeze['blob_sha256'] or tensor_digest(model.state_dict()) != freeze['tensor_sha256']:
        raise ValueError('selected checkpoint identity differs')
    updates = freeze.get('selected', {}).get('successful_updates')
    if updates is not None and origin.get('successful_updates') != updates:
        raise ValueError('selected successful updates differ')
    output = args.output.resolve()
    if output.exists():
        raise ValueError('fresh output required: ' + str(output))
    (output / 'fp16').mkdir(parents=True)
    state = model.state_dict()
    values = {key: (value.to(torch.float16) if value.is_floating_point() else value.clone())
              for key, value in state.items()}
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError('fp16 export contains nonfinite tensors')
    model_path = output / 'fp16' / 'model.pt'
    with model_path.open('xb') as stream:
        torch.save(values, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tensor_sha = tensor_digest(values)
    metadata_json = export_metadata(job, sha(model_path), tensor_sha, variant=variant)
    with (output / 'fp16' / 'model.json').open('x') as stream:
        json.dump(metadata_json, stream, indent=2, allow_nan=False)
        stream.write('\n')
    loaded, loaded_job, _, loaded_origin, loaded_device = _load_model(output / 'fp16', device='cpu', variant=variant)
    if (loaded_job != job or str(loaded_device) != 'cpu' or loaded_origin['model_sha256'] != sha(model_path)
            or loaded_origin['tensor_sha256'] != tensor_sha
            or tensor_digest(loaded.state_dict()) != tensor_digest(values)):
        raise ValueError('strict fp16 standalone reload differs')
    size = int(job['image_size'])
    images = torch.zeros((1, 3, size, size), dtype=torch.float32)
    age = torch.zeros((1, 2), dtype=torch.float32) if (job.get('age') or {}).get('enabled') else None
    loaded.eval()
    model.eval()
    with torch.inference_mode():
        expected = age_forward(model, images, age, job).reshape(-1).float()
        actual = age_forward(loaded, images, age, job).reshape(-1).float()
    if expected.shape != (1,) or not bool(torch.isfinite(expected).all()) or not torch.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        raise ValueError('synthetic fp16 inference differs')
    receipt = {'status': 'f07_round_standalone_fp16_strict_synthetic_passed', 'round_id': freeze.get('round_id'),
               'freeze': {'path': str(freeze_path), 'sha256': sha(freeze_path)},
               'blob_sha256': freeze['blob_sha256'], 'variant': variant,
               'fp32_tensor_sha256': freeze['tensor_sha256'], 'fp16_tensor_sha256': tensor_sha,
               'model': {'path': str(model_path), 'sha256': sha(model_path),
                         'bytes': model_path.stat().st_size},
               'metadata': {'path': str(output / 'fp16' / 'model.json'), 'sha256': sha(output / 'fp16' / 'model.json')},
               'GPU_initialized': False, 'dev_read': False, 'official_read': False, 'TTA': False}
    with (output / 'STANDALONE.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'round': freeze.get('round_id'),
                      'model_sha256': receipt['model']['sha256'], 'bytes': receipt['model']['bytes']}), flush=True)


if __name__ == '__main__':
    main()

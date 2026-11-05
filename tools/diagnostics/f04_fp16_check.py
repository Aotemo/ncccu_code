"""Compare fp32 and fp16 forward logits for the four chosen members on 15 cost images."""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'
sys.path.insert(0, str(BUNDLE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', nargs=4, required=True, help='id=export_dir')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from retrain import inference
    from retrain.data import ImageDataset
    files = []
    for directory in args.images:
        items = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        files.extend(items)
    files = files[:15]
    if len(files) < 15:
        raise SystemExit('15 images required')
    device = torch.device(args.device)
    result = {'images': [p.name for p in files], 'members': {}, 'device': str(device)}
    with tempfile.TemporaryDirectory(prefix='f04-fp16-') as work:
        root = Path(work) / 'data'
        (root / 'test').mkdir(parents=True)
        rows = []
        for path in files:
            target = root / 'test' / path.name
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
            rows.append({'name': path.name, 'relative_path': 'test/' + path.name, 'label': 0, 'age': ''})
        for spec in args.members:
            identifier, export = spec.split('=', 1)
            declared = json.loads((Path(export) / 'model.json').read_text())
            variant = (declared.get('checkpoint') or {}).get('variant')
            model, job, metadata, origin, _ = inference._load_model(export, 'cpu', variant=variant)
            model = model.to(device).eval()
            dataset = ImageDataset(root, rows, int(job['image_size']), False,
                                   job.get('normalization', 'imagenet'),
                                   prepared_size=int(job.get('prepared_size', 0)), age=job.get('age'),
                                   input_preprocessing=job.get('input_preprocessing'))
            batch = [dataset[index] for index in range(len(rows))]
            images = torch.stack([item[0] for item in batch])
            ages = torch.stack([item[3] for item in batch]) if len(batch[0]) == 4 else None
            logits = {}
            for precision in ('fp32', 'fp16'):
                with torch.inference_mode(), inference._autocast(device, precision):
                    value = model(images.to(device),
                                  ages.to(device, dtype=torch.float32) if ages is not None else None)
                value = value.reshape(-1).float()
                if value.numel() != len(rows) or not torch.isfinite(value).all():
                    raise SystemExit('invalid logits: ' + identifier + ' ' + precision)
                logits[precision] = value.cpu()
            difference = (logits['fp32'] - logits['fp16']).abs()
            result['members'][identifier] = {
                'export': export, 'checkpoint_variant': (metadata.get('checkpoint') or {}).get('variant'),
                'max_abs_logit_difference': float(difference.max()),
                'mean_abs_logit_difference': float(difference.mean()),
                'logit_range': [float(logits['fp32'].min()), float(logits['fp32'].max())],
            }
            del model
            torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    raise SystemExit(main())

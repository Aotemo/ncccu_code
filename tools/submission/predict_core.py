"""F09 submission inference core: one process, serial batch=1, fp16.

Loads every member through the main deployed bundle (weights from the
latency-exports whose job carries no stale binding) with each member's own
site binding, and runs the frozen preprocessing (JPEG draft hint optional) +
equal-weight z fusion + Platt. Used by predict.py, cost.py and the latency
measurements alike.
"""
import hashlib
import json
import math
import os
from pathlib import Path

REQUIRED_JOB_KEYS = ('model', 'image_size', 'prepared_size', 'normalization', 'age')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class MemberSet:
    def __init__(self, config, device):
        import torch
        sys_path = config.get('bundle')
        if sys_path:
            import sys
            sys.path.insert(0, sys_path)
        from retrain import inference
        from retrain.age import enabled
        from retrain.data import ImageDataset
        self.torch = torch
        self.inference = inference
        self.enabled = enabled
        self.ImageDataset = ImageDataset
        self.device = device
        self.decode = config.get('decode', 'draft')
        self.draft_divisor = int(config.get('draft_divisor', 4))
        if self.draft_divisor < 1:
            raise ValueError('draft_divisor must be positive')
        self.platt = config['platt']
        self.members = []
        jobs = None
        for entry in config['members']:
            os.environ['NCCCU_SITE_BINDING'] = entry['site']
            os.environ['NCCCU_SITE_BINDING_SHA256'] = entry['site_sha256']
            model, job, metadata, origin, selected_device = inference._load_model(
                entry['export'], device=device, variant=entry.get('variant', 'ema'))
            if selected_device.type != device.type:
                raise ValueError('member not on the pinned device: ' + entry['id'])
            model.eval()
            comparable = {key: job.get(key) for key in REQUIRED_JOB_KEYS}
            if jobs is None:
                jobs = comparable
            elif comparable != jobs:
                raise ValueError('member preprocessing jobs differ: ' + entry['id'])
            self.members.append({'id': entry['id'], 'model': model, 'job': job,
                                 'z': entry['z'], 'model_sha256': entry.get('model_sha256')})
        self.job = jobs
        if self.torch.cuda.is_initialized() is False:
            pass

    def dataset(self, root, rows):
        from PIL import Image, ImageOps
        from retrain.data import _crop_black_border
        from retrain.preprocessing import prepare_image
        torch = self.torch
        parent = self

        divisor = self.draft_divisor

        class _DirectDataset(self.ImageDataset):
            def _read_image(self, row):
                source = Path(self.root) / row['relative_path']
                with Image.open(source) as opened:
                    if parent.decode == 'draft':
                        opened.draft('RGB', (max(1, opened.width // divisor),
                                             max(1, opened.height // divisor)))
                    if self.prepared_size:
                        return prepare_image(opened, self.prepared_size, self.native_preprocessing)
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                return _crop_black_border(image)

        job = self.job
        return _DirectDataset(root, rows, int(job['image_size']), False,
                              job.get('normalization', 'imagenet'),
                              prepared_size=int(job.get('prepared_size', 0) or 0),
                              age=job.get('age'))

    def predict_index(self, item):
        """item = dataset[index]; returns (probability, stage_seconds dict)."""
        torch = self.torch
        import time
        t0 = time.perf_counter()
        images = item[0].unsqueeze(0).to(self.device)
        age_on = self.enabled(self.job)
        ages = (item[3].unsqueeze(0).to(self.device, dtype=torch.float32) if age_on else None)
        t1 = time.perf_counter()
        logits = []
        per_member = []
        for member in self.members:
            member_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16):
                out = (member['model'](images, ages) if ages is not None else member['model'](images))
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize(self.device)
            value = float(out.reshape(-1).float()[0])
            if not math.isfinite(value):
                raise FloatingPointError('nonfinite member logit')
            logits.append(value)
            per_member.append(time.perf_counter() - member_start)
        t2 = time.perf_counter()
        fused = sum((value - member['z']['mean']) / member['z']['std']
                    for value, member in zip(logits, self.members)) / len(self.members)
        probability = 1.0 / (1.0 + math.exp(-(self.platt['a'] * fused + self.platt['b'])))
        t3 = time.perf_counter()
        stages = {'h2d': t1 - t0, 'forward': t2 - t1, 'post': t3 - t2,
                  'per_member_ms': [round(value * 1000, 3) for value in per_member]}
        return probability, stages


def rows_from_directory(pictures_dir, ages_csv=None):
    directory = Path(pictures_dir)
    names = sorted(path.name for path in directory.iterdir()
                   if path.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    ages = {}
    if ages_csv:
        import csv as _csv
        with Path(ages_csv).open(newline='') as stream:
            for record in _csv.DictReader(stream):
                ages[record['Image_name'].casefold()] = (record['Image_name'],
                                                         str(record.get('Age', '') or ''))
    rows = []
    for name in names:
        if ages:
            hit = ages.get(name.casefold())
            if hit is None:
                raise ValueError('age missing for timing picture: ' + name)
            rows.append({'name': hit[0], 'relative_path': name, 'label': 0, 'age': hit[1]})
        else:
            rows.append({'name': name, 'relative_path': name, 'label': 0, 'age': ''})
    return rows

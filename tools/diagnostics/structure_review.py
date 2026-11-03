"""Create an identity-hidden original/512 packet and seal notes before unblinding.

Only run in the guarded Pod CPU preparation pool. Pictures, aliases, names and
case-level annotations stay in exp-v3. This tool never judges a diagnosis,
changes a label, loads a model, or supplies review coordinates to training.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))

REVIEW_FIELDS = ('original_disc_visible', 'prepared_disc_visible',
                 'original_rim_resolvable', 'prepared_rim_resolvable',
                 'original_adjacent_resolvable', 'prepared_adjacent_resolvable',
                 'detail_loss', 'label_interpretability')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare_packet(rows, indices, data_root, output, *, cache_root=None):
    """Render caller's presealed indices, in caller's blinded order; no reselection."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from retrain.data import ImageDataset
    from retrain.preprocessing import prepare_image, PreparedImageCache
    from retrain.verification import lexical_path
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)):
        raise ValueError('unique presealed row indices required')
    for i in indices:
        lexical_path(rows[i]['relative_path'], 'train')
    if torch.cuda.is_initialized():
        raise ValueError('structure review requires uninitialized GPU')
    root = Path(data_root).resolve(strict=True)
    output = Path(output)
    output.mkdir(exist_ok=False)
    visible = output / 'reviewer'
    visible.mkdir()
    rgb_dataset = ImageDataset(str(root), [], 512, False, normalization='flair_rgb01')
    normalized_dataset = ImageDataset(str(root), [], 512, False, normalization='imagenet')
    cached = None
    if cache_root is not None:
        if os.environ.get('TRAIN_IMAGE_CACHE_READONLY') != '1':
            raise ValueError('existing prepared cache must be read-only')
        cached = PreparedImageCache(root, cache_root, 512)
    cases, private, panels = [], [], []
    for ordinal, index in enumerate(indices, 1):
        row = rows[index]
        path = root / row['relative_path']
        if path.is_symlink() or not path.resolve().is_relative_to(root / 'train'):
            raise ValueError('original train path aliases outside sealed source')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('original SHA differs from sealed train row')
        with Image.open(io.BytesIO(raw)) as opened:
            original = ImageOps.exif_transpose(opened).convert('RGB')
        prepared = prepare_image(original, 512)
        cache_checked = False
        if cached is not None:
            actual = cached.read(row['relative_path'], row['sha256'])
            if actual.size != prepared.size or actual.tobytes() != prepared.tobytes():
                raise ValueError('actual prepared cache and production preparation differ')
            prepared = actual
            cache_checked = True
        tensor = rgb_dataset._image_tensor(prepared, None)
        view = Image.fromarray(np.rint(tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        normalized = normalized_dataset._image_tensor(prepared, None).contiguous()
        alias = f'B{ordinal:03d}'
        folder = visible / alias
        folder.mkdir()
        original.save(folder / 'original.png')
        view.save(folder / 'prepared.png')
        case = {'alias': alias, 'original': {'path': alias + '/original.png',
                'sha256': sha(folder / 'original.png'), 'size': list(original.size)},
                'prepared': {'path': alias + '/prepared.png',
                'sha256': sha(folder / 'prepared.png'), 'size': [512, 512]},
                'normalized_tensor_sha256': hashlib.sha256(normalized.numpy().tobytes()).hexdigest()}
        cases.append(case)
        private.append({'alias': alias, 'index': index, 'name': row['name'], 'label': row['label'],
                        'original_sha256': row['sha256'], 'prepared_sha256': case['prepared']['sha256'],
                        'actual_readonly_cache_checked': cache_checked})
        panel = Image.new('RGB', (1280, 548), (28, 28, 28))
        panel.paste(ImageOps.pad(original, (768, 512), method=Image.Resampling.BILINEAR), (0, 32))
        panel.paste(view, (768, 32))
        ImageDraw.Draw(panel).text((8, 8), alias + ' / original overview | exact 512 input', fill='white')
        panels.append(panel)
    for start in range(0, len(panels), 2):
        sheet = Image.new('RGB', (1280, 548 * len(panels[start:start + 2])), (28, 28, 28))
        for offset, panel in enumerate(panels[start:start + 2]):
            sheet.paste(panel, (0, 548 * offset))
        sheet.save(visible / f'overview-{start // 2 + 1:02d}.png')
    packet = {'format': 'blinded-original-and-input512-v1', 'cases': cases,
              'display': 'Original overview is scaled for display; original.png preserves native decoded pixels. '
                         'prepared.png is the exact 512 RGB evaluation canvas. No enhancement or anatomy crop.'}
    write_new(visible / 'PACKET.json', packet)
    write_new(output / 'PRIVATE-MAPPING.json', private)
    write_new(output / 'PROVENANCE.json', {
        'packet_sha256': sha(visible / 'PACKET.json'), 'private_mapping_sha256': sha(output / 'PRIVATE-MAPPING.json'),
        'reviewer_artifacts': {str(p.relative_to(visible)): sha(p) for p in sorted(visible.rglob('*')) if p.is_file()},
        'production_sources_sha256': {name: sha(ROOT / name) for name in
            ('runtime/bundle/retrain/data.py', 'runtime/bundle/retrain/preprocessing.py')},
        'review_source_sha256': sha(__file__), 'actual_readonly_cache_checked_count': sum(
            item['actual_readonly_cache_checked'] for item in private),
        'official_read': False, 'model_loaded': False, 'new_forward': False, 'gpu_initialized': False})
    return packet


def seal_review(packet_root, annotations, reviewer):
    """Require every alias once, verify displayed bytes, durably seal then join names."""
    root = Path(packet_root)
    visible = root / 'reviewer'
    provenance = json.loads((root / 'PROVENANCE.json').read_text())
    for relative, digest in provenance['reviewer_artifacts'].items():
        if sha(visible / relative) != digest:
            raise ValueError('displayed reviewer artifact SHA differs')
    if sha(root / 'PRIVATE-MAPPING.json') != provenance['private_mapping_sha256']:
        raise ValueError('private mapping SHA differs')
    packet = json.loads((visible / 'PACKET.json').read_text())
    aliases = [row['alias'] for row in packet['cases']]
    if (not isinstance(annotations, list) or len(annotations) != len(aliases)
            or any(not isinstance(row, dict) for row in annotations)
            or len({row.get('alias') for row in annotations}) != len(aliases)
            or {row.get('alias') for row in annotations} != set(aliases)):
        raise ValueError('complete unique alias review required before unblinding')
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('explicit reviewer identity and qualification required')
    for row in annotations:
        if (set(row) != {'alias', *REVIEW_FIELDS, 'notes'}
                or any(row[k] not in ('yes', 'no', 'partial', 'unknown') for k in REVIEW_FIELDS)
                or not isinstance(row['notes'], str) or not row['notes'].strip()):
            raise ValueError('explicit structure observations and uncertainty notes required')
    write_new(root / 'BLIND-REVIEW.json', {'reviewer': reviewer, 'expert_review': False,
        'original_label_standard_supplied': False, 'packet_sha256': provenance['packet_sha256'],
        'annotations': annotations, 'limits': 'AI-assisted technical review; cannot establish clinical label correctness.'})
    sealed_sha = sha(root / 'BLIND-REVIEW.json')
    # First parse of hidden identity mapping occurs only after notes are sealed.
    private = json.loads((root / 'PRIVATE-MAPPING.json').read_text())
    notes = {row['alias']: row for row in annotations}
    result = {'blind_review_sha256': sealed_sha, 'reviewer': reviewer, 'expert_review': False,
              'cases': [{**row, **notes[row['alias']], 'label_review': 'uncertain'} for row in private],
              'label_standard_available': False, 'labels_changed': False, 'training_inputs_changed': False}
    write_new(root / 'UNBLINDED-REVIEW.json', result)
    return result

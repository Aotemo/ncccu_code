"""Blind fit-only technical localization gate; never load a model or dev rows.

Run in an identity-checked, hidden-device Pod CPU allocation. All case-level
artifacts stay in the supplied exp-v3 directory. The reviewer is not an expert.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime/bundle'))

OLD_MEMBERS_SHA = '3602cf733c5542d7a5e5c8b0b469fb205212431456598d45ec9c8b82d59dbf90'
MANIFEST_SHA = '7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def choose_fit(fit, old, count, seed):
    if (len(set(fit)) != len(fit) or len(set(old)) != len(old)
            or any(type(i) is not int for i in [*fit, *old]) or not set(old) <= set(fit)):
        raise ValueError('unique previously sampled fit identities required')
    eligible = sorted(set(fit) - set(old))
    if type(count) is not int or count <= 0 or count > len(eligible):
        raise ValueError('invalid blind sample count')
    return random.Random(seed).sample(eligible, count)


def consensus_box(size, disc, bright):
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('positive image size required')
    if disc.get('status') != 'located' or disc.get('center_xy') is None:
        return {'status': 'unknown', 'reason': 'disc_locator_unknown', 'box': None}
    dx, dy = disc['center_xy']
    bx = (bright[0] + bright[2]) / 2
    by = (bright[1] + bright[3]) / 2
    if not all(math.isfinite(v) for v in (dx, dy, bx, by)):
        return {'status': 'unknown', 'reason': 'nonfinite_center', 'box': None}
    if math.hypot(dx - bx, dy - by) > .15 * min(width, height):
        return {'status': 'unknown', 'reason': 'centers_disagree', 'box': None}
    side = round(.5 * min(width, height))
    left = max(0, min(width - side, round((dx + bx) / 2 - side / 2)))
    top = max(0, min(height - side, round((dy + by) / 2 - side / 2)))
    return {'status': 'located', 'box': [left, top, left + side, top + side]}


def box_coverage(window, reference):
    if window is None or reference is None:
        return False, False
    l, t, r, b = window
    x0, y0, x1, y1 = reference
    full = l <= x0 and t <= y0 and r >= x1 and b >= y1
    mx, my = (x1 - x0) / 2, (y1 - y0) / 2
    margin = l <= x0 - mx and t <= y0 - my and r >= x1 + mx and b >= y1 + my
    return full, margin


def validate_annotations(rows, aliases):
    if (not isinstance(rows, list) or len(rows) != len(aliases)
            or any(not isinstance(row, dict) or set(row) != {'alias', 'box'} for row in rows)
            or len({row['alias'] for row in rows}) != len(rows)
            or {row['alias'] for row in rows} != set(aliases)):
        raise ValueError('all blinded aliases exactly once required')
    for row in rows:
        box = row['box']
        if box is None:
            continue
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) is not int or not 0 <= v <= 512 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('a 512-pixel rough box or explicit unknown is required')


def require_context(root):
    root = Path(root)
    allowed = Path('/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep')
    if (not root.resolve().is_relative_to(allowed) or os.getpriority(os.PRIO_PROCESS, 0) != 19
            or any(os.environ.get(k) != '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))):
        raise ValueError('isolated hidden-device Pod CPU preparation required')
    return root


def prepare(root, site_path, site_sha, old_path):
    import io
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from retrain.data import ImageDataset
    from retrain.preprocessing import PreparedImageCache, prepare_image
    from retrain.verification import lexical_path
    root = require_context(root)
    if sha(site_path) != site_sha or sha(old_path) != OLD_MEMBERS_SHA:
        raise ValueError('site or old holdout identity differs')
    site, old = json.loads(Path(site_path).read_text()), json.loads(Path(old_path).read_text())
    if site['pod']['uid'] != os.environ['NCCCU_POD_UID']:
        raise ValueError('site UID differs')
    manifest_path = Path(site['metadata_root']) / 'manifest.json'
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError('sealed fit manifest differs')
    manifest = json.loads(manifest_path.read_text())
    fit = manifest['fit_indices']
    if len(fit) != 46256 or len(old['selected_global_indices']) != 96:
        raise ValueError('fit or excluded sample count differs')
    selected = choose_fit(fit, old['selected_global_indices'], 64, 20261112)
    root.mkdir(exist_ok=False)
    private = {'manifest_sha256': MANIFEST_SHA, 'excluded_sha256': OLD_MEMBERS_SHA,
               'random_seed': 20261112, 'selected_global_indices': selected,
               'selection_uses_labels_age_scores': False, 'official_read': False}
    write_new(root / 'PRIVATE-MEMBERS.json', private)
    visible = root / 'reviewer'
    visible.mkdir()
    os.environ['TRAIN_IMAGE_CACHE_READONLY'] = '1'
    torch.set_num_threads(min(12, int(os.environ['NCCCU_CPU_THREADS'])))
    torch.set_num_interop_threads(1)
    data_root = Path(site['data_root']).resolve(strict=True)
    cache = PreparedImageCache(data_root, Path(site['prepared_cache']) / 'train/sealed', 512)
    dataset = ImageDataset(str(data_root), [], 512, False, normalization='flair_rgb01')
    cases, private_rows = [], []
    started = time.monotonic()
    for ordinal, index in enumerate(selected, 1):
        delay = (ordinal - 1) * .25 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
        row = manifest['rows'][index]
        relative, expected = row['relative_path'], row['sha256']
        lexical_path(relative, 'train')
        alias = f'B{ordinal:03d}'
        image, source, error = None, 'unknown', None
        try:
            image = cache.read(relative, expected)
            source = 'sealed_readonly_512_cache'
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            try:
                original_path = data_root / relative
                if original_path.is_symlink() or not original_path.resolve().is_relative_to(data_root / 'train'):
                    raise ValueError('source path differs')
                raw = original_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError('source SHA differs')
                with Image.open(io.BytesIO(raw)) as opened:
                    image = prepare_image(ImageOps.exif_transpose(opened).convert('RGB'), 512)
                source = 'sha_verified_original'
            except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                error = type(exc).__name__
        if image is None:
            view = Image.new('RGB', (512, 512), (15, 15, 15))
            ImageDraw.Draw(view).text((16, 240), 'UNKNOWN: source image unavailable', fill='white')
        else:
            tensor = dataset._image_tensor(image, None)
            view = Image.fromarray(np.rint(tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        image_path = visible / (alias + '.png')
        view.save(image_path)
        cases.append({'alias': alias, 'image': image_path.name, 'sha256': sha(image_path),
                      'available': image is not None})
        private_rows.append({'alias': alias, 'global_index': index, 'source': source, 'error': error})
    sheets = []
    for start in range(0, 64, 4):
        canvas = Image.new('RGB', (1024, 1072), (25, 25, 25))
        for offset, case in enumerate(cases[start:start + 4]):
            x, y = (offset % 2) * 512, (offset // 2) * 536
            with Image.open(visible / case['image']) as im:
                canvas.paste(im.convert('RGB'), (x, y + 24))
            ImageDraw.Draw(canvas).text((x + 8, y + 6), case['alias'] + ' / exact input512', fill='white')
        path = visible / f'location-{start // 4 + 1:02d}.png'
        canvas.save(path)
        sheets.append({'path': path.name, 'sha256': sha(path)})
    packet = {'format': 'fit-blind-consensus-512-v1', 'cases': cases, 'sheets': sheets,
              'no_algorithm_predictions': True, 'no_labels_age_or_scores': True}
    write_new(visible / 'PACKET.json', packet)
    write_new(root / 'PRIVATE-MAPPING.json', {'rows': private_rows})
    result = {'status': 'blind_packet_ready_unreviewed', 'count': 64,
              'available': sum(x['available'] for x in cases), 'sheets': len(sheets),
              'private_members_sha256': sha(root / 'PRIVATE-MEMBERS.json'),
              'packet_sha256': sha(visible / 'PACKET.json'), 'source_sha256': sha(__file__),
              'official_read': False, 'model_loaded': False, 'forward_calls': 0, 'GPU_initialized': False}
    write_new(root / 'PACKET-READY.json', result)
    print(json.dumps({**result, 'receipt_sha256': sha(root / 'PACKET-READY.json')}), flush=True)


def seal(root, annotations_path):
    root = require_context(root)
    packet = json.loads((root / 'reviewer/PACKET.json').read_text())
    rows = json.loads(Path(annotations_path).read_text())
    validate_annotations(rows, [x['alias'] for x in packet['cases']])
    for case in packet['cases']:
        if sha(root / 'reviewer' / case['image']) != case['sha256']:
            raise ValueError('blinded source image changed')
    if not packet['no_algorithm_predictions'] or not packet['no_labels_age_or_scores']:
        raise ValueError('packet not blinded')
    result = {'reviewer': 'AI technical rough geometry; non-expert', 'expert_review': False,
              'packet_sha256': sha(root / 'reviewer/PACKET.json'), 'annotations': rows,
              'uses_model_predictions_labels_or_age': False,
              'limits': 'technical rough boxes, not clinical review or segmentation'}
    write_new(root / 'BLIND-ANNOTATIONS.json', result)
    print(json.dumps({'status': 'all_blind_annotations_sealed', 'count': len(rows),
                      'unknown': sum(x['box'] is None for x in rows),
                      'seal_sha256': sha(root / 'BLIND-ANNOTATIONS.json')}), flush=True)


def evaluate(root, expected_seal):
    import numpy as np
    from PIL import Image
    from retrain.preprocessing import locate_optic_disc, bright_center_box
    from retrain.config import NATIVE_BRIGHT_CENTER_RULE
    root = require_context(root)
    sealed_path = root / 'BLIND-ANNOTATIONS.json'
    if sha(sealed_path) != expected_seal:
        raise ValueError('pre-prediction blind seal differs')
    sealed = json.loads(sealed_path.read_text())
    packet_path = root / 'reviewer/PACKET.json'
    packet = json.loads(packet_path.read_text())
    if sealed['packet_sha256'] != sha(packet_path) or sealed['uses_model_predictions_labels_or_age'] is not False:
        raise ValueError('blind packet or seal differs')
    annotations = sealed['annotations']
    validate_annotations(annotations, [x['alias'] for x in packet['cases']])
    by_alias = {row['alias']: row['box'] for row in annotations}
    rows = []
    for case in packet['cases']:
        path = root / 'reviewer' / case['image']
        if sha(path) != case['sha256']:
            raise ValueError('blind image changed')
        if not case['available']:
            predicted = {'status': 'unknown', 'reason': 'image_unavailable', 'box': None}
            disc = {'center_xy': None}
        else:
            with Image.open(path) as image:
                image = image.convert('RGB')
                disc = locate_optic_disc(image)
                predicted = consensus_box(image.size, disc, bright_center_box(image, NATIVE_BRIGHT_CENTER_RULE))
        reference = by_alias[case['alias']]
        full, margin = box_coverage(predicted['box'], reference)
        center_error = None
        if predicted['box'] is not None and reference is not None:
            window = predicted['box']
            center_error = math.hypot((window[0] + window[2] - reference[0] - reference[2]) / 2,
                                      (window[1] + window[3] - reference[1] - reference[3]) / 2)
        rows.append({'alias': case['alias'], 'prediction': predicted, 'reference_box': reference,
                     'complete': full, 'complete_with_margin': margin, 'center_error_px': center_error})
    write_new(root / 'PRIVATE-RESULT.json', {'rows': rows, 'blind_seal_sha256': expected_seal})
    errors = [x['center_error_px'] for x in rows if x['center_error_px'] is not None]
    count = sum(x['complete_with_margin'] for x in rows)
    result = {'status': 'completed_blind_fit_geometry_gate', 'count': len(rows), 'required': 63,
              'complete': sum(x['complete'] for x in rows), 'complete_with_margin': count,
              'algorithm_unknown_or_fallback': sum(x['prediction']['status'] == 'unknown' for x in rows),
              'review_unknown': sum(x['reference_box'] is None for x in rows),
              'center_error_px_median': float(np.median(errors)) if errors else None,
              'passed': count >= 63, 'blind_seal_sha256': expected_seal,
              'private_result_sha256': sha(root / 'PRIVATE-RESULT.json'),
              'source_sha256': sha(__file__), 'official_read': False, 'model_loaded': False,
              'forward_calls': 0, 'GPU_initialized': False,
              'limitation': 'AI technical rough boxes, not expert segmentation; native mapping unvalidated'}
    write_new(root / 'SUMMARY.json', result)
    print(json.dumps({**result, 'summary_sha256': sha(root / 'SUMMARY.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'seal', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--site', type=Path)
    parser.add_argument('--site-sha')
    parser.add_argument('--old-members', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--seal-sha')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.root, args.site, args.site_sha, args.old_members)
    elif args.stage == 'seal':
        seal(args.root, args.annotations)
    else:
        evaluate(args.root, args.seal_sha)


if __name__ == '__main__':
    main()

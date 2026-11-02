"""Attribute existing selected internal-dev rankings; never run a model or official data.

Run only through the existing Pod CPU guard. Private row identities, image sheets
and prediction vectors belong to the supplied exp-v3 technical-prep directory.
"""
import argparse
from bisect import bisect_left, bisect_right
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys

ANCHORS = ('R107', 'R108', 'R120', 'R134R1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pair_contributions(labels, scores):
    """Twice each row's misordered-pair mass; each class sum is the total."""
    if (len(labels) != len(scores) or not labels
            or any(type(label) is not int or label not in (0, 1) for label in labels)
            or set(labels) != {0, 1} or any(not math.isfinite(s) for s in scores)):
        raise ValueError('finite binary scores with both classes required')
    groups = {label: sorted(s for y, s in zip(labels, scores) if y == label)
              for label in (0, 1)}
    result = []
    for label, score in zip(labels, scores):
        other = groups[1 - label]
        left, right = bisect_left(other, score), bisect_right(other, score)
        result.append(2 * (len(other) - right if label else left) + right - left)
    return result


def read_predictions(path, expected, variant):
    path = Path(path)
    if (variant not in ('ema', 'online') or path.parent.name != variant
            or path.parent.parent.name != 'predictions'
            or re.fullmatch(r'dev-cycle-\d{4}\.csv', path.name) is None):
        raise ValueError('only selected internal-dev prediction paths allowed')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected):
        raise ValueError('dev prediction count differs')
    scores = []
    for row, source in zip(rows, expected):
        if row['name'] != source['name'] or int(row['label']) != source['label']:
            raise ValueError('dev prediction identity/order/label differs')
        score = float(row['logit'])
        if not math.isfinite(score):
            raise ValueError('nonfinite prediction')
        scores.append(score)
    return scores


def stats(args):
    inventory = json.loads(args.inventory.read_text())
    manifest_path = Path(inventory['site']['metadata_root']) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha(manifest_path)
    rows = [manifest['rows'][i] for i in manifest['dev_indices']]
    labels = [r['label'] for r in rows]
    counts = {label: labels.count(label) for label in (0, 1)}
    if counts != {0: 4973, 1: 167} or any(Path(r['relative_path']).parts[0] != 'train' for r in rows):
        raise ValueError('fixed internal-dev population differs')
    from retrain.n05_metrics import metrics
    vectors, contributions, models, unknown = {}, {}, {}, {}
    for name, candidates in inventory['rounds'].items():
        try:
            if len(candidates) != 1:
                raise ValueError('unique complete freeze required')
            freeze_path = Path(candidates[0]['freeze'])
            freeze = json.loads(freeze_path.read_text())
            selected = freeze['selected']
            if (freeze['cpu_fixture_only'] is not False or freeze['precision'] != 'fp32'
                    or freeze['view'] != 'identity'
                    or freeze['experiment91']['manifest_sha256'] != manifest_sha):
                raise ValueError('freeze/manifest identity differs')
            path = Path(selected['prediction_path'])
            scores = read_predictions(path, rows, selected['variant'])
            if sha(path) != selected['prediction_sha256']:
                raise ValueError('selected prediction SHA differs')
            precision_path = Path(str(path) + '.precision.json')
            if (str(precision_path) != selected['precision_path']
                    or sha(precision_path) != selected['precision_sha256']):
                raise ValueError('selected precision SHA/path differs')
            precision = json.loads(precision_path.read_text())
            if (precision.get('variant') != selected['variant']
                    or precision.get('view') != 'identity'
                    or precision.get('actual_tensor_sha256') != selected['tensor_sha256']
                    or precision.get('state_identity', {}).get('successful_updates') != selected['successful_updates']):
                raise ValueError('selected precision state differs')
            actual = metrics([{'label': label, 'logit': score} for label, score in zip(labels, scores)])
            if any(abs(actual[key] - selected[key]) > 1e-12 for key in actual):
                raise ValueError('recomputed selected metrics differ')
            mass = pair_contributions(labels, scores)
            if any(sum(m for m, y in zip(mass, labels) if y == label) != actual['error_twice'] for label in (0, 1)):
                raise ValueError('per-class error attribution does not conserve total')
            vectors[name], contributions[name] = scores, mass
            top = sorted((i for i, y in enumerate(labels) if y == 1), key=lambda i: (-mass[i], i))[:20]
            models[name] = {**actual, 'variant': selected['variant'], 'cycle': selected['cycle'],
                            'successful_updates': selected['successful_updates'],
                            'top20_positive_error_fraction': sum(mass[i] for i in top) / actual['error_twice'],
                            'freeze_path': str(freeze_path), 'freeze_sha256': sha(freeze_path),
                            'prediction_sha256': sha(path), 'precision_sha256': sha(precision_path)}
        except (OSError, ValueError, KeyError, TypeError) as error:
            unknown[name] = str(error)
    if any(name not in vectors for name in ANCHORS):
        write_new(args.output / 'UNKNOWN.json', {'unknown': unknown, 'verified_models': models})
        raise ValueError('one or more predeclared anchors unavailable; no substitute selected')
    top20, average = {}, {}
    groups, group_stats = {}, {}
    for label in (1, 0):
        indices = [i for i, y in enumerate(labels) if y == label]
        top20[label] = {name: set(sorted(indices, key=lambda i: (-contributions[name][i], i))[:20])
                        for name in ANCHORS}
        average[label] = {i: sum(contributions[name][i] / (2 * counts[1 - label]) for name in ANCHORS) / len(ANCHORS)
                          for i in indices}
        ordered = sorted(indices, key=lambda i: (-average[label][i], i))
        persistent = [i for i in ordered if sum(i in top20[label][name] for name in ANCHORS) >= 3]
        excluded = set(ordered[:20]) | set().union(*top20[label].values())
        controls = random.Random(20261102 + label).sample([i for i in indices if i not in excluded], 8)
        prefix = 'positive' if label else 'negative'
        groups[prefix + '_hard'] = persistent[:8]
        groups[prefix + '_random'] = controls
        fixed = ordered[:20]
        group_stats[prefix] = {'persistent_top20_at_least_3_of_4': len(persistent),
            'top20_intersection_all_4': len(set.intersection(*top20[label].values())),
            'fixed20_mass_by_model': {name: sum(mass[i] for i in fixed) / 2 for name, mass in contributions.items()},
            'fixed20_error_fraction_by_model': {name: sum(mass[i] for i in fixed) / models[name]['error_twice']
                                              for name, mass in contributions.items()}}
    summary = {'manifest_sha256': manifest_sha, 'count': len(rows), 'positives': counts[1],
        'negatives': counts[0], 'anchors': list(ANCHORS), 'models': models, 'unknown': unknown,
        'groups': group_stats, 'image_selection_counts': {name: len(values) for name, values in groups.items()},
        'official_read': False, 'model_loaded': False, 'new_forward': False,
        'classification_threshold_used': False,
        'source_sha256': sha(__file__), 'metrics_source_sha256': sha(args.source_root / 'runtime/bundle/retrain/n05_metrics.py')}
    write_new(args.output / 'PRIVATE-SELECTION.json', {'rows': rows, 'groups': groups,
        'scores': vectors, 'twice_pair_contributions': contributions, 'manifest_sha256': manifest_sha})
    write_new(args.output / 'SUMMARY.json', summary)
    print(json.dumps(summary, indent=2))


def image_features(image):
    import numpy as np
    rgb = np.asarray(image, dtype=np.float64)
    mask = rgb.max(axis=2) > 12
    if not mask.any():
        raise ValueError('selected view has no technical content pixels')
    luminance = rgb @ np.array([.299, .587, .114])
    green = rgb[:, :, 1]
    laplacian = (green[1:-1, :-2] + green[1:-1, 2:] + green[:-2, 1:-1]
                 + green[2:, 1:-1] - 4 * green[1:-1, 1:-1])
    inner = mask[1:-1, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:] & mask[:-2, 1:-1] & mask[2:, 1:-1]
    if not inner.any():
        raise ValueError('no interior content pixels for sharpness proxy')
    return {'black_fraction': float(1 - mask.mean()),
            'content_luminance_mean': float(luminance[mask].mean()),
            'content_luminance_std': float(luminance[mask].std()),
            'green_laplacian_variance': float(laplacian[inner].var()),
            'content_red_green_ratio': float(rgb[:, :, 0][mask].mean() / max(green[mask].mean(), 1e-12)),
            'content_bright_fraction': float((luminance[mask] > 230).mean())}


def images(args):
    import os
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from retrain.data import ImageDataset
    from retrain.preprocessing import prepare_image
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']))
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError('GPU must remain uninitialized')
    inventory = json.loads(args.inventory.read_text())
    data_root = Path(inventory['site']['data_root']).resolve()
    private_path = args.output / 'PRIVATE-SELECTION.json'
    private = json.loads(private_path.read_text())
    rows = private['rows']
    dataset = ImageDataset(str(data_root), [], 512, False,
                           normalization='flair_rgb01', prepared_size=0)
    details, summaries, artifacts = {}, {}, {}
    for group, indices in private['groups'].items():
        sheet = Image.new('RGB', (1024, max(1, math.ceil(len(indices) / 2)) * 286), (35, 35, 35))
        draw = ImageDraw.Draw(sheet)
        entries = []
        for order, index in enumerate(indices):
            row = rows[index]
            relative = Path(row['relative_path'])
            path = (data_root / relative).resolve()
            if (len(relative.parts) != 2 or relative.parts[0] != 'train'
                    or not path.is_relative_to(data_root / 'train') or path.is_symlink()):
                raise ValueError('image must be a sealed train-source dev row')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('selected original image SHA differs')
            with Image.open(io.BytesIO(raw)) as opened:
                original = ImageOps.exif_transpose(opened).convert('RGB')
                prepared = prepare_image(original, 512)
            tensor = dataset._image_tensor(prepared, None)
            view = Image.fromarray(np.rint(tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            feature = image_features(view)
            feature.update(original_width=original.width, original_height=original.height,
                           original_aspect_ratio=original.width / original.height,
                           reduction_to_512=max(original.size) / 512)
            alias = f'{group}-{order + 1:02d}'
            entries.append({'alias': alias, 'dev_index': index, **feature})
            x, y = (order % 2) * 512, (order // 2) * 286
            sheet.paste(ImageOps.pad(original, (248, 248), color=(0, 0, 0)), (x + 4, y + 30))
            sheet.paste(view.resize((248, 248), Image.Resampling.NEAREST), (x + 258, y + 30))
            draw.text((x + 6, y + 4), f'{alias} / original | actual 512 view', fill='white')
            if order < 2 and group.endswith('_hard'):
                detail = Image.new('RGB', (1536, 768), (0, 0, 0))
                detail.paste(ImageOps.pad(original, (768, 768)), (0, 0))
                detail.paste(view.resize((768, 768), Image.Resampling.NEAREST), (768, 0))
                detail_path = args.output / f'{alias}-detail.png'
                detail.save(detail_path)
                artifacts[detail_path.name] = sha(detail_path)
        sheet_path = args.output / f'{group}.png'
        sheet.save(sheet_path)
        artifacts[sheet_path.name] = sha(sheet_path)
        details[group] = entries
        summaries[group] = {'count': len(entries), 'features': {key: {
            'mean': float(np.mean([item[key] for item in entries])),
            'median': float(np.median([item[key] for item in entries]))}
            for key in entries[0] if key not in ('alias', 'dev_index')} if entries else {}}
    write_new(args.output / 'PRIVATE-IMAGE-FEATURES.json', details)
    result = {'groups': summaries, 'selection_sha256': sha(private_path), 'artifacts_sha256': artifacts,
        'source_sha256': sha(__file__), 'production_sources_sha256': {name: sha(args.source_root / name)
            for name in ('runtime/bundle/retrain/data.py', 'runtime/bundle/retrain/preprocessing.py')},
        'originals_sha_verified': sum(len(items) for items in details.values()),
        'official_read': False, 'model_loaded': False, 'new_forward': False,
        'gpu_initialized': torch.cuda.is_initialized(), 'read_only_originals': True,
        'technical_content_mask': 'RGB max > 12; not anatomy', 'image_size': 512}
    write_new(args.output / 'IMAGE-SUMMARY.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('stats', 'images'))
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root / 'runtime/bundle'))
    {'stats': stats, 'images': images}[args.stage](args)


if __name__ == '__main__':
    main()

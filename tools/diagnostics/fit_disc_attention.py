"""A2 fit-only blinded Grad-CAM geometry gate; never read dev or official rows.

Prepare and seal precede both compare and evaluate. All case-level artifacts
remain in an independent technical-prep directory in the verified Pod.
"""
import argparse
import json
import math
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'tools/pretraining'))
from tools.diagnostics import fit_disc_consensus as blind
from tools.diagnostics.f04_gpu_full_dev import bind_member_runtime, read_pinned
from source_chain import cpu_context, configure_torch_threads

FREEZE_SHA = 'cf5ce24aeb854876d5143dd24ae045e50fabe12412700ca8a3c4a6329eec5ad3'
MODEL_SHA = 'e0bc2c52dcc62adcb8ab4f4fe47c8126b31a0939732deca3985d5757c8d99e83'
METADATA_SHA = '86d4293ce357b8d109e008c4cd75f1d617937387cd435c3222f9ae52caa76861'


def locate(cam, rect):
    """Apply the preregistered rule to a 512-square, interpolated ReLU map."""
    import numpy as np
    from scipy import ndimage

    cam = np.asarray(cam, dtype=np.float64)
    if cam.shape != (512, 512):
        raise ValueError('512-square Grad-CAM required')
    if (not isinstance(rect, list) or len(rect) != 4
            or any(type(v) is not int for v in rect)
            or not 0 <= rect[0] < rect[2] <= 512
            or not 0 <= rect[1] < rect[3] <= 512):
        raise ValueError('prepared image rectangle required')
    x0, y0, x1, y1 = rect
    area = cam[y0:y1, x0:x1]
    if not np.isfinite(area).all():
        return {'status': 'unknown', 'reason': 'nonfinite_map', 'box': None}
    lo, hi = float(area.min()), float(area.max())
    if hi - lo <= 1e-12:
        return {'status': 'unknown', 'reason': 'flat_map', 'box': None}
    norm = np.zeros((512, 512), dtype=np.float64)
    norm[y0:y1, x0:x1] = (area - lo) / (hi - lo)
    total = float(norm.sum())
    if not math.isfinite(total) or total <= 0:
        return {'status': 'unknown', 'reason': 'no_positive_mass', 'box': None}
    labels, count = ndimage.label(norm >= .85, structure=np.ones((3, 3), dtype=np.uint8))
    components = []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    for component_id, bounds in enumerate(ndimage.find_objects(labels), 1):
        if bounds is None or sizes[component_id] < 16:
            continue
        ys, xs = np.nonzero(labels[bounds] == component_id)
        ys += bounds[0].start
        xs += bounds[1].start
        values = norm[ys, xs]
        mass = float(values.sum())
        # np.nonzero scans top-to-bottom, left-to-right; ties use first pixel.
        components.append({'peak': float(values.max()), 'first': (int(ys[0]), int(xs[0])),
                           'center': (float((xs * values).sum() / mass),
                                      float((ys * values).sum() / mass))})
    if not components:
        return {'status': 'unknown', 'reason': 'no_qualifying_component', 'box': None}
    components.sort(key=lambda c: (-c['peak'], *c['first']))
    cx, cy = components[0]['center']
    radius = .15 * 512
    if any(math.hypot(other['center'][0] - cx, other['center'][1] - cy) > radius
           for other in components[1:]):
        return {'status': 'unknown', 'reason': 'distant_component', 'box': None}
    yy, xx = np.ogrid[:512, :512]
    fraction = float(norm[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2].sum()) / total
    if fraction < .30:
        return {'status': 'unknown', 'reason': 'diffuse_mass', 'box': None}
    left = max(0, min(256, round(cx - 128)))
    top = max(0, min(256, round(cy - 128)))
    return {'status': 'located', 'center_xy': [cx, cy], 'mass_fraction': fraction,
            'box': [left, top, left + 256, top + 256]}


def grad_cam(model, image, job):
    import torch
    import torch.nn.functional as functional
    from retrain.age import forward

    if image.shape != (1, 3, 512, 512) or image.dtype != torch.float32:
        raise ValueError('one fp32 image with production 512 geometry required')
    captured = []
    handle = model.network.features[7].register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        with torch.enable_grad():
            logits = forward(model, image, torch.zeros((1, 2), dtype=torch.float32), job).reshape(-1)
            if logits.shape != (1,) or len(captured) != 1 or captured[0].ndim != 4:
                raise ValueError('single logit and fixed feature output required')
            gradient, = torch.autograd.grad(logits[0], captured[0])
            weights = gradient.mean(dim=(-2, -1), keepdim=True)
            cam = (weights * captured[0]).sum(dim=1, keepdim=True).relu()
            return functional.interpolate(cam, size=(512, 512), mode='bilinear',
                                          align_corners=False)[0, 0].detach().cpu().numpy()
    finally:
        handle.remove()


def verify_packet(root, expected_ready):
    root = blind.require_context(root)
    ready = read_pinned(root / 'PACKET-READY.json', expected_ready)
    members = read_pinned(root / 'PRIVATE-MEMBERS.json', ready['private_members_sha256'])
    mapping = read_pinned(root / 'PRIVATE-MAPPING.json', ready['private_mapping_sha256'])
    packet = read_pinned(root / 'reviewer/PACKET.json', ready['packet_sha256'])
    if (ready.get('status') != 'blind_packet_ready_unreviewed' or ready.get('count') != 64
            or members.get('random_seed') != 20261124
            or members.get('excluded_sha256') != blind.OLD_MEMBERS_SHA
            or members.get('excluded_consensus_sha256') != blind.CONSENSUS_MEMBERS_SHA
            or members.get('manifest_sha256') != blind.MANIFEST_SHA
            or members.get('selection_uses_labels_age_scores') is not False
            or len(members.get('selected_global_indices', [])) != 64
            or len(set(members['selected_global_indices'])) != 64
            or len(packet.get('cases', [])) != 64 or len(mapping.get('rows', [])) != 64
            or not packet.get('no_algorithm_predictions') or not packet.get('no_labels_age_or_scores')):
        raise ValueError('A2 blind packet identity differs')
    aliases = [case['alias'] for case in packet['cases']]
    if (len(set(aliases)) != 64 or [row['alias'] for row in mapping['rows']] != aliases
            or [row['global_index'] for row in mapping['rows']] != members['selected_global_indices']):
        raise ValueError('blind image mapping differs')
    for case in packet['cases']:
        path = root / 'reviewer' / case['image']
        if path.is_symlink() or blind.sha(path) != case['sha256']:
            raise ValueError('blind image changed')
    return root, ready, packet, mapping


def evaluate(args, context):
    import numpy as np
    import torch
    from PIL import Image

    root, _, packet, mapping = verify_packet(args.root, args.ready_sha)
    sealed = read_pinned(root / 'BLIND-ANNOTATIONS.json', args.seal_sha)
    baseline = read_pinned(root / 'SUMMARY.json', args.baseline_sha)
    if (sealed.get('packet_sha256') != blind.sha(root / 'reviewer/PACKET.json')
            or sealed.get('uses_model_predictions_labels_or_age') is not False
            or baseline.get('blind_seal_sha256') != args.seal_sha
            or baseline.get('status') != 'completed_blind_fit_geometry_gate'):
        raise ValueError('prior blind seal or comparison identity differs')
    blind.validate_annotations(sealed['annotations'], [case['alias'] for case in packet['cases']])
    freeze = read_pinned(SOURCE / 'records/analysis/20261104-f04/F04-FINAL-FREEZE.json', FREEZE_SHA)
    members = [item for item in freeze['members'] if item['model']['sha256'] == MODEL_SHA]
    if len(members) != 1 or members[0]['metadata']['sha256'] != METADATA_SHA:
        raise ValueError('F04 R120 frozen member identity differs')
    member = members[0]
    model_path, metadata_path = Path(member['model']['path']), Path(member['metadata']['path'])
    if (model_path.is_symlink() or metadata_path.is_symlink() or model_path.name != 'model.pt'
            or metadata_path.name != 'model.json' or model_path.parent != metadata_path.parent
            or blind.sha(model_path) != MODEL_SHA or blind.sha(metadata_path) != METADATA_SHA):
        raise ValueError('frozen weight or metadata bytes differ')
    historical_freeze = read_pinned(member['freeze']['path'], member['freeze']['sha256'])
    for relative, digest in historical_freeze['source_files'].items():
        if blind.sha(Path(member['bundle']) / relative) != digest:
            raise ValueError('historical R120 runtime changed')
    bind_member_runtime(member)
    sys.path.insert(0, str(Path(member['bundle']).resolve(strict=True)))
    from retrain.inference import _load_model
    from retrain.ema import tensor_digest
    model, job, _, origin, device = _load_model(model_path.parent, device='cpu', variant='ema')
    if (str(device) != 'cpu' or origin.get('model_sha256') != MODEL_SHA
            or origin.get('tensor_sha256') != member['model']['fp16_tensor_sha256']
            or job.get('model') != 'convnext_small' or job.get('image_size') != 512
            or job.get('prepared_size') != 512 or job.get('input_preprocessing') is not None
            or job.get('normalization', 'imagenet') != 'imagenet'
            or not (job.get('age') or {}).get('enabled')
            or tensor_digest(model.state_dict()) != member['model']['restored_fp32_tensor_sha256']
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError('CPU fp32 R120 model contract differs')
    model.eval()
    refs = {row['alias']: row['box'] for row in sealed['annotations']}
    mapped = {row['alias']: row for row in mapping['rows']}
    mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
    rows, forwards = [], 0
    for case in packet['cases']:
        alias = case['alias']
        if not case['available']:
            prediction = {'status': 'unknown', 'reason': 'image_unavailable', 'box': None}
        else:
            with Image.open(root / 'reviewer' / case['image']) as image:
                pixels = np.asarray(image.convert('RGB')).copy()
            if pixels.shape != (512, 512, 3):
                raise ValueError('blinded image geometry differs')
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float().div_(255)
            cam = grad_cam(model, (tensor - mean) / std, job)
            forwards += 1
            prediction = locate(cam, mapped[alias]['valid_rect'])
        reference = refs[alias]
        full, margin = blind.box_coverage(prediction['box'], reference)
        error = None
        if prediction['box'] is not None and reference is not None:
            window = prediction['box']
            error = math.hypot((window[0] + window[2] - reference[0] - reference[2]) / 2,
                               (window[1] + window[3] - reference[1] - reference[3]) / 2)
        rows.append({'alias': alias, 'prediction': prediction, 'reference_box': reference,
                     'complete': full, 'complete_with_margin': margin, 'center_error_px': error})
    if torch.cuda.is_initialized():
        raise ValueError('GPU was initialized')
    blind.write_new(root / 'A2-PRIVATE-RESULT.json', {'rows': rows, 'blind_seal_sha256': args.seal_sha})
    errors = [row['center_error_px'] for row in rows if row['center_error_px'] is not None]
    count = sum(row['complete_with_margin'] for row in rows)
    result = {'status': 'completed_blind_fit_gradcam_geometry_gate', 'count': len(rows), 'required': 63,
              'complete': sum(row['complete'] for row in rows), 'complete_with_margin': count,
              'algorithm_unknown_or_fallback': sum(row['prediction']['status'] == 'unknown' for row in rows),
              'review_unknown': sum(row['reference_box'] is None for row in rows),
              'center_error_px_median': float(np.median(errors)) if errors else None, 'passed': count >= 63,
              'baseline_summary_sha256': args.baseline_sha, 'baseline': baseline,
              'blind_seal_sha256': args.seal_sha, 'packet_ready_sha256': args.ready_sha,
              'private_result_sha256': blind.sha(root / 'A2-PRIVATE-RESULT.json'),
              'source_sha256': blind.sha(__file__), 'model_sha256': MODEL_SHA, 'metadata_sha256': METADATA_SHA,
              'execution': context, 'fit_only': True, 'dev_read': False, 'official_read': False,
              'forward_calls': forwards, 'GPU_initialized': False,
              'limitation': 'non-expert rough boxes; fit-trained model; native coordinate mapping unvalidated'}
    blind.write_new(root / 'A2-SUMMARY.json', result)
    print(json.dumps({**result, 'summary_sha256': blind.sha(root / 'A2-SUMMARY.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'seal', 'compare', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--site', type=Path)
    parser.add_argument('--site-sha')
    parser.add_argument('--old-members', type=Path)
    parser.add_argument('--consensus-members', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--ready-sha')
    parser.add_argument('--seal-sha')
    parser.add_argument('--baseline-sha')
    args = parser.parse_args()
    context = cpu_context()
    if args.stage == 'prepare':
        if not all((args.site, args.site_sha, args.old_members, args.consensus_members)):
            parser.error('prepare requires site identity and both old member lists')
        blind.prepare(args.root, args.site, args.site_sha, args.old_members, consensus_path=args.consensus_members)
    elif args.stage == 'seal':
        if not args.ready_sha or not args.annotations:
            parser.error('seal requires packet-ready SHA and annotations')
        verify_packet(args.root, args.ready_sha)
        blind.seal(args.root, args.annotations)
    elif args.stage == 'compare':
        if not args.ready_sha or not args.seal_sha:
            parser.error('compare requires packet-ready and blind-seal SHA')
        verify_packet(args.root, args.ready_sha)
        blind.evaluate(args.root, args.seal_sha)
    else:
        if not args.ready_sha or not args.seal_sha or not args.baseline_sha:
            parser.error('evaluate requires packet-ready, blind-seal and baseline SHA')
        configure_torch_threads(context)
        evaluate(args, context)


if __name__ == '__main__':
    main()

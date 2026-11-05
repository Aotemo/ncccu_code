"""Input binding and reduction for the F04 supervisor's explicit F05 soup mode."""
import json
import math
from pathlib import Path
import sys
import time

GROUPS = ['r120', 'r134r1', 'r201']
PRECISIONS = ['fp32', 'fp16']


def build_plan(groups):
    if list(groups) != GROUPS:
        raise ValueError('the three preregistered soups must appear in order')
    plan = []
    for group, world in zip(GROUPS, (3, 3, 2)):
        for rank in range(world):
            plan.append({'device_rank': len(plan), 'member': group, 'rank': rank, 'world': world})
    return plan


def validate_inputs(inputs, policy):
    members = inputs.get('members', [])
    if (policy.get('version') != 'f05-uniform-soup-dev-20261105-v1'
            or policy.get('groups') != GROUPS or policy.get('precisions') != PRECISIONS
            or policy.get('ranks_per_group') != [3, 3, 2] or policy.get('batch_size') != 8
            or policy.get('dev_count') != 5140 or policy.get('dev_positive') != 167
            or policy.get('preview') != 'all_three_soups_replace_each_of_five_f04_members_once_per_precision'
            or policy.get('preview_selection') is not False
            or policy.get('normalization') != 'dev_population_zscore_equal_weight'
            or policy.get('max_total_fp16_weight_bytes') != 500000000
            or policy.get('official_read') is not False or policy.get('tta') is not False
            or inputs.get('status') != 'F05_uniform_soup_dev_inputs_sealed_no_GPU'
            or [item.get('round_id') for item in members] != GROUPS
            or inputs.get('plan') != build_plan(GROUPS)
            or inputs.get('manifest', {}).get('sha256') != policy.get('base_manifest_sha256')
            or inputs.get('batch_size') != 8 or inputs.get('dev_count') != 5140
            or inputs.get('dev_positive') != 167 or inputs.get('TTA') is not False
            or inputs.get('official_read') is not False):
        raise ValueError('fixed soup dev input or policy differs')
    for member in members:
        name = member['round_id']
        if (member.get('soup', {}).get('sha256') != policy['soup_receipt_sha256'][name]
                or member.get('receipt', {}).get('sha256') != policy['standalone_receipt_sha256'][name]
                or set(member.get('exports', {})) != set(PRECISIONS)):
            raise ValueError('fixed soup receipt or precision identity differs')
    return inputs


def verify_assets(gpu, member):
    soup = gpu.read_pinned(member['soup']['path'], member['soup']['sha256'])
    standalone = gpu.read_pinned(member['receipt']['path'], member['receipt']['sha256'])
    if (soup.get('status') != 'uniform_soup_created_no_dev_evaluation'
            or soup.get('method') != 'uniform_all_compatible_selected_states_v1'
            or soup.get('group') != member['round_id']
            or soup.get('standalone') != member['receipt']
            or soup.get('average_state_sha256') != standalone.get('state_sha256')
            or standalone.get('status') != 'uniform_soup_standalone_strict_synthetic_passed'
            or standalone.get('exports') != member['exports']
            or standalone.get('synthetic_forward_only') is not True
            or any(soup.get(k) is not False or standalone.get(k) is not False
                   for k in ('official_read', 'dev_read', 'GPU_initialized', 'TTA'))
            or any(standalone.get(k) != member.get(k)
                   for k in ('freeze', 'site', 'bundle', 'data_root', 'cache'))
            or member.get('selected', {}).get('variant') != standalone.get('base_variant_for_loader')):
        raise ValueError('soup lineage or standalone receipt differs')
    for precision in PRECISIONS:
        export = member['exports'][precision]
        model, metadata = export['model'], export['metadata']
        model_path, metadata_path = Path(model['path']), Path(metadata['path'])
        if (model_path.is_symlink() or metadata_path.is_symlink()
                or model_path.name != 'model.pt' or metadata_path.name != 'model.json'
                or model_path.parent != metadata_path.parent or model_path.stat().st_size != model['bytes']
                or gpu.sha(model_path) != model['sha256'] or gpu.sha(metadata_path) != metadata['sha256']):
            raise ValueError('soup export bytes differ')
        contents = json.loads(metadata_path.read_text())
        if (contents.get('model_sha256') != model['sha256']
                or contents.get('checkpoint', {}).get('tensor_sha256') != model['tensor_sha256']
                or contents.get('checkpoint', {}).get('variant') != member['selected']['variant']):
            raise ValueError('soup inference metadata differs')
    freeze = gpu.read_pinned(member['freeze']['path'], member['freeze']['sha256'])
    for relative, digest in freeze['source_files'].items():
        if gpu.sha(Path(member['bundle']) / relative) != digest:
            raise ValueError('soup historical runtime changed')
    site = gpu.read_pinned(member['site']['path'], member['site']['sha256'])
    if site.get('data_root') != member['data_root'] or site.get('prepared_cache') != member['cache']:
        raise ValueError('soup historical data/cache identity differs')
    return site


def prepare(gpu, args):
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('published source differs')
    sys.path[:0] = [str(source / 'tools/pretraining'), str(source / 'tools/diagnostics')]
    from source_chain import cpu_context
    from f04_export_member import verify_published
    from f04_bounded_fusion import verify_blob
    context = cpu_context()
    files = verify_published(source, args.remote_metadata, args.git_commit)
    for name in ('tools/diagnostics/f04_gpu_full_dev.py', 'tools/diagnostics/f05_soup_dev.py',
                 'configs/F05-SOUP-DEV-20261105.json'):
        files[name] = verify_blob(source, args.remote_metadata, name)
    policy = json.loads((source / 'configs/F05-SOUP-DEV-20261105.json').read_text())
    frozen = gpu.read_pinned(source / 'records/analysis/20261104-f04/F04-FINAL-FREEZE.json',
                             policy['f04_freeze_sha256'])
    if frozen.get('full_dev', {}).get('summary_sha256') != policy.get('f04_internal_fp16_summary_sha256'):
        raise ValueError('frozen F04 internal dev summary identity differs')
    original = gpu.read_pinned(args.f04_inputs, frozen['identity']['inputs_sha256'])
    if (original.get('status') != 'F04_full_dev_gpu_inputs_sealed_no_GPU'
            or original.get('official_read') is not False or original.get('TTA') is not False
            or original.get('manifest', {}).get('sha256') != policy['base_manifest_sha256']
            or [m['round_id'] for m in original['members']] != [m['round_id'] for m in frozen['members']]
            or any(any(left.get(key) != right.get(key) for key in
                       ('model', 'metadata', 'selected', 'freeze', 'bundle', 'site', 'data_root', 'cache'))
                   for left, right in zip(original['members'], frozen['members']))
            or original['weight_bytes'] != frozen['weight_bytes']):
        raise ValueError('original frozen internal F04 inputs differ')
    manifest = gpu.read_pinned(original['manifest']['path'], original['manifest']['sha256'])
    rows, indices = manifest['rows'], manifest['dev_indices']
    if (len(indices) != 5140 or len(set(indices)) != 5140
            or sum(rows[index]['label'] == 1 for index in indices) != 167
            or any(not rows[index]['relative_path'].startswith('train/') for index in indices)):
        raise ValueError('complete internal dev population required')
    receipts = {}
    for path in args.soup_receipt:
        path = path.resolve(strict=True)
        soup = json.loads(path.read_text())
        group = soup.get('group')
        if group not in GROUPS or group in receipts or gpu.sha(path) != policy['soup_receipt_sha256'][group]:
            raise ValueError('fixed unique soup receipt required')
        record = gpu.read_pinned(soup['standalone']['path'], policy['standalone_receipt_sha256'][group])
        member = {'round_id': group, 'soup': {'path': str(path), 'sha256': gpu.sha(path)},
                  'receipt': soup['standalone'], 'exports': record['exports'],
                  'selected': {'variant': record['base_variant_for_loader']},
                  **{key: record[key] for key in ('freeze', 'site', 'bundle', 'data_root', 'cache')}}
        receipts[group] = member
    if set(receipts) != set(GROUPS):
        raise ValueError('three fixed soup receipts required')
    roots = set()
    for member in receipts.values():
        roots.add(str(Path(verify_assets(gpu, member)['work_root']).resolve(strict=True)))
    if len(roots) != 1:
        raise ValueError('one work root is required')
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(Path(next(iter(roots))) / 'technical-prep'):
        raise ValueError('fresh technical-prep output required')
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = {'status': 'F05_uniform_soup_dev_inputs_sealed_no_GPU', 'at': time.time(),
              'git_commit': args.git_commit, 'git_files': files, 'source_root': str(source),
              'manifest': original['manifest'], 'members': [receipts[name] for name in GROUPS],
              'plan': build_plan(GROUPS), 'f04_inputs': {'path': str(args.f04_inputs),
              'sha256': frozen['identity']['inputs_sha256']}, 'f04_freeze_sha256': policy['f04_freeze_sha256'],
              'weight_bytes': sum(receipts[name]['exports']['fp16']['model']['bytes'] for name in GROUPS),
              'batch_size': 8, 'dev_count': 5140, 'dev_positive': 167, 'arithmetic': 'fp32',
              'TTA': False, 'official_read': False, 'GPU_initialized': False, 'execution': context}
    validate_inputs(inputs, policy)
    gpu.write_new(output, inputs)
    print(json.dumps({'status': inputs['status'], 'receipt_sha256': gpu.sha(output)}), flush=True)


def metrics_and_preview(labels, soup_vectors, original_vectors, original_sizes, soup_sizes):
    """Internal-dev-only fixed replacement preview; never choose members."""
    import numpy as np
    from f04_size_ensemble import compare_predictions, equal_zscore
    from retrain.n05_metrics import metrics
    if set(soup_vectors) != set(GROUPS) or set(soup_sizes) != set(GROUPS):
        raise ValueError('all three fixed soup vectors and sizes are required')
    names = list(original_sizes)
    if len(names) != 5 or len(set(names)) != 5 or set(original_vectors) != set(names):
        raise ValueError('five complete frozen F04 member vectors required')
    if any(type(value) is not int or value <= 0 for value in [*original_sizes.values(), *soup_sizes.values()]):
        raise ValueError('positive actual fp16 byte counts required')
    individual, previews = {}, []
    for group in GROUPS:
        precision_vectors = soup_vectors[group]
        if set(precision_vectors) != set(PRECISIONS):
            raise ValueError('both precision vectors required')
        comparison = compare_predictions(labels, precision_vectors['fp32'], precision_vectors['fp16'])
        left = np.asarray(precision_vectors['fp32'], dtype=np.float64)
        right = np.asarray(precision_vectors['fp16'], dtype=np.float64)
        if left.std() <= 0 or right.std() <= 0:
            raise ValueError('nonconstant precision vectors required')
        comparison['max_abs_z_score_difference'] = float(np.max(np.abs(
            (left - left.mean()) / left.std() - (right - right.mean()) / right.std())))
        individual[group] = comparison
        for replaced in names:
            vectors = {name: original_vectors[name] for name in names if name != replaced}
            vectors[group] = precision_vectors['fp32']
            order = [name for name in names if name != replaced] + [group]
            result = equal_zscore(labels, vectors, order)
            previews.append({'soup': group, 'replaced': replaced, 'precision': 'fp32',
                             'metrics': result['metrics'], 'fp16_bytes': sum(original_sizes.values())
                             - original_sizes[replaced] + soup_sizes[group], 'selection': False})
    return {'soups': individual, 'replacement_preview_fp32': previews,
            'replacement_preview_fp16': {'status': 'unknown',
                'reason': 'original F04 fp16 per-rank packet identity has not been independently verified'},
            'selection': False, 'official_read': False, 'TTA': False}


def aggregate(gpu, source, inputs, output, inputs_sha):
    """Check all 16 shards before reducing; sample-level vectors stay on Pod."""
    sys.path[:0] = [str(source / 'runtime/bundle'), str(source / 'tools/diagnostics')]
    import paired_evidence
    reference = paired_evidence._reference(inputs['manifest'])
    manifest = gpu.read_pinned(inputs['manifest']['path'], inputs['manifest']['sha256'])
    rows, indices = manifest['rows'], manifest['dev_indices']
    if len(indices) != 5140 or len(reference['dev']) != 5140:
        raise ValueError('complete sealed internal dev required')
    release = json.loads((output / 'RELEASE.json').read_text())
    members = {member['round_id']: member for member in inputs['members']}
    vectors = {group: {} for group in GROUPS}
    packet_shas = {}
    for precision in PRECISIONS:
        packets = []
        for device_rank, expected in enumerate(inputs['plan']):
            path = output / 'packets' / ('rank-' + str(device_rank) + '-' + precision + '.json')
            packet = json.loads(path.read_text())
            member = members[expected['member']]
            if (packet.get('device_rank') != device_rank or packet.get('member') != expected['member']
                    or packet.get('rank') != expected['rank'] or packet.get('error') is not None
                    or packet.get('storage_precision') != precision
                    or packet.get('model_sha256') != member['exports'][precision]['model']['sha256']
                    or packet.get('input_sha256') != inputs_sha or packet.get('TTA') is not False
                    or packet.get('official_read') is not False
                    or packet.get('proof_sha256') != release.get('proof_sha256', {}).get(str(device_rank))):
                raise ValueError('soup worker shard identity differs')
            packets.append(packet)
            packet_shas[f'{device_rank}-{precision}'] = gpu.sha(path)
        for group, world in zip(GROUPS, (3, 3, 2)):
            subset = [packet for packet in packets if packet['member'] == group]
            predictions = gpu.canonical_member(rows, indices, subset, world)
            vectors[group][precision] = [float(row['logit']) for row in predictions]
    original = gpu.read_pinned(inputs['f04_inputs']['path'], inputs['f04_inputs']['sha256'])
    baseline, sizes = {}, {}
    for member in original['members']:
        name = member['round_id']
        selected = member['selected']
        scores, details, _ = paired_evidence._selected(
            {'freeze': member['freeze'], 'completion': selected['completion']}, reference)
        if any(details[key] != selected[key] for key in ('variant', 'cycle', 'successful_updates', 'tensor_sha256')):
            raise ValueError('frozen F04 selected member changed')
        baseline[name] = scores
        sizes[name] = member['model']['bytes']
    soup_sizes = {name: member['exports']['fp16']['model']['bytes'] for name, member in members.items()}
    result = metrics_and_preview([row['label'] for row in reference['dev']], vectors, baseline, sizes, soup_sizes)
    result['packet_sha256'] = packet_shas
    result['dev_count'] = len(indices)
    result['f04_inputs_sha256'] = inputs['f04_inputs']['sha256']
    return result

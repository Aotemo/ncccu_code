"""F05 internal-dev candidate screening primitives; no official or GPU entry.

The final discovery order, candidate identities and source evidence must be
sealed after all F05 directions finish and before a new F05 blend is read.
Use the existing F03 discovery, F04 per-member strict export and eight-rank
dev prediction machinery. These helpers do not create an evaluation platform.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

from ensemble_dev import f03_candidate_gate
from f04_bounded_fusion import require_same_candidate
from f04_size_ensemble import equal_zscore, rank_and_cap

SOURCE = Path(__file__).resolve().parents[2]
BYTE_LIMIT = 500_000_000
THRESHOLD = .988
BASE_MANIFEST_SHA = '7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377'
FIT_ADAPTATION_AUDIT_SHA = '5bd9a3ae3e84b35131cfe8e8364aa0c001793937310803d5d129f66b5fcdd9e7'


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _pinned(reference):
    if (not isinstance(reference, dict) or not isinstance(reference.get('path'), str)
            or not _is_sha(reference.get('sha256'))):
        raise ValueError('explicit pinned receipt required')
    path = Path(reference['path'])
    if path.is_symlink() or not path.is_file() or _sha(path) != reference['sha256']:
        raise ValueError('pinned receipt missing or changed')
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('pinned receipt is not valid JSON') from error
    if not isinstance(value, dict):
        raise ValueError('pinned receipt must contain an object')
    return value


def _forbidden_id(value):
    return bool(re.search(r'(^|[-_/])n06($|[-_/])|flair', value.lower()))


def build_candidate_inventory(round_inventory, soups, order):
    """Bind every discovered round and every declared soup to a fixed order."""
    if (not isinstance(round_inventory, dict)
            or round_inventory.get('status') != 'read_only_F03_candidate_discovery_no_fusion'
            or round_inventory.get('official_read') is not False
            or not isinstance(round_inventory.get('members'), list)
            or not isinstance(soups, list) or not isinstance(order, list)):
        raise ValueError('sealed F03 round discovery, soup list and order required')
    by_id = {}
    for row in round_inventory['members']:
        name = row.get('round_id') if isinstance(row, dict) else None
        if not isinstance(name, str) or not name or name in by_id:
            raise ValueError('unique discovered round IDs required')
        by_id[name] = {'id': name, 'kind': 'round', 'discovery': row}
    for row in soups:
        name = row.get('id') if isinstance(row, dict) else None
        if not isinstance(name, str) or not name.startswith('soup-') or name in by_id:
            raise ValueError('unique explicit soup IDs required')
        by_id[name] = {'id': name, 'kind': 'soup', 'discovery': row}
    if (not order or any(not isinstance(name, str) or not name for name in order)
            or len(set(order)) != len(order) or set(order) != set(by_id)):
        raise ValueError('sealed candidate order must cover the entire discovery exactly once')
    return [{**by_id[name], 'order': index} for index, name in enumerate(order)]


def verify_round_export(row, receipt_ref, *, inventory_sha, approved, adaptation_audit=None,
                        adaptation_receipt=None, allow_fixture=False):
    """Recheck existing F04 strict CPU export and all selected/source bytes.

    The exporter reconstructs the historical model in its own process and
    strictly reloads each serialized inference state; this consumer verifies
    that receipt and bytes instead of reconstructing it a second time.
    """
    try:
        if (not isinstance(row, dict) or _forbidden_id(row['round_id'])
                or f03_candidate_gate(row, THRESHOLD, approved)[0] != 'eligible'
                or len(row.get('checkpoint_candidates', [])) != 1
                or row['selected'].get('variant') not in ('online', 'ema')
                or not _is_sha(row['selected'].get('tensor_sha256')) or not _is_sha(inventory_sha)):
            raise ValueError('eligible unique non-medical selected candidate required')
        exported = _pinned(receipt_ref)
        checkpoint = row['checkpoint_candidates'][0]
        if (exported.get('status') != 'F04_complete_inference_state_exported_and_strictly_reloaded'
                or exported.get('round_id') != row['round_id'] or exported.get('selected') != row['selected']
                or exported.get('inventory', {}).get('sha256') != inventory_sha
                or exported.get('selected_blob_sha256') != row['selected_blob_sha256']
                or exported.get('freeze') != row['spec']['freeze'] or exported.get('science') != row['science']
                or exported.get('checkpoint') != checkpoint or exported.get('model_constructed') is not True
                or exported.get('forward_calls') != 0 or exported.get('GPU_initialized') is not False
                or exported.get('official_read') is not False
                or exported.get('exports', {}).get('strict_reload') is not True
                or exported.get('exports', {}).get('nonfloating_buffers_preserved') is not True):
            raise ValueError('strict F04 export does not bind the selected member')
        freeze, science = _pinned(row['spec']['freeze']), _pinned(row['science'])
        if (freeze.get('cpu_fixture_only') is not allow_fixture
                or freeze.get('blob_sha256') != row['selected_blob_sha256']
                or any(freeze.get('selected', {}).get(key) != row['selected'].get(key)
                       for key in ('variant', 'tensor_sha256', 'successful_updates', 'auroc'))
                or science.get('public_source') != row['public_source']
                or science.get('candidate', {}).get('weight_sha256') != row['public_source']['sha256']
                or science.get('job', {}).get('initial_checkpoint') is not None):
            raise ValueError('frozen selected identity or public pretraining source differs')
        if science.get('ssl_recipe') is not None:
            if (not adaptation_audit or not adaptation_receipt
                    or (not allow_fixture and adaptation_audit.get('sha256') != FIT_ADAPTATION_AUDIT_SHA)):
                raise ValueError('task-fit adaptation provenance is not independently pinned')
            audit = _pinned(adaptation_audit)
            _pinned(adaptation_receipt)
            round_name = row['round_id'].split('-')[0].upper()
            matching = [item for item in audit.get('rounds', []) if item.get('round') == round_name]
            if (audit.get('status') != 'historic_fit_ssl_provenance_audit' or len(matching) != 1
                    or matching[0].get('status') != 'evidence_supports_competition_fit_only'
                    or matching[0].get('fit_indices_only') is not True
                    or matching[0].get('dev_evaluations') != 0
                    or matching[0].get('manifest_sha256') != BASE_MANIFEST_SHA
                    or matching[0].get('public_weight_sha256') != row['public_source']['sha256']
                    or matching[0].get('receipt_sha256') != adaptation_receipt.get('sha256')):
                raise ValueError('task-fit adaptation source or role evidence differs')
            if not allow_fixture:
                round_root = Path(row['spec']['freeze']['path']).parent
                stage_path = Path(adaptation_receipt['path'])
                site_path = round_root / 'SITE_BINDING.json'
                if (not stage_path.resolve().is_relative_to(round_root.resolve()) or site_path.is_symlink()
                        or not site_path.is_file() or _sha(site_path) != matching[0].get('site_sha256')
                        or any(freeze.get('source_files', {}).get(name) != digest
                               for name, digest in matching[0].get('source_file_sha256', {}).items())):
                    raise ValueError('adaptation receipt or frozen source belongs to another round')
        if not allow_fixture:
            files = freeze.get('source_files')
            bundle = Path(row['spec']['freeze']['path']).parent / 'bundle'
            if not isinstance(files, dict) or not files or bundle.is_symlink() or not bundle.is_dir():
                raise ValueError('frozen member runtime source is unavailable')
            for name, digest in files.items():
                if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('invalid frozen source path')
                path = bundle / name
                if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(bundle.resolve())
                        or _sha(path) != digest):
                    raise ValueError('frozen member runtime source differs')
        checkpoint_path = Path(checkpoint['path'])
        sidecar = _pinned(checkpoint['sidecar'])
        if (checkpoint_path.is_symlink() or not checkpoint_path.is_file()
                or _sha(checkpoint_path) != row['selected_blob_sha256']
                or sidecar.get('sha256') != row['selected_blob_sha256']):
            raise ValueError('selected checkpoint or sidecar is missing or changed')
        sizes = {}
        for precision in ('fp16', 'fp32'):
            weight = exported['exports'][precision]
            path = Path(weight['path'])
            if (path.is_symlink() or not path.is_file() or type(weight['bytes']) is not int
                    or weight['bytes'] <= 0 or path.stat().st_size != weight['bytes']
                    or not _is_sha(weight.get('sha256'))
                    or _sha(path) != weight['sha256']):
                raise ValueError('complete strictly reloaded exported weight is missing or changed')
            sizes[precision + '_bytes'] = weight['bytes']
        return {'id': row['round_id'], 'auroc': row['selected']['auroc'],
                'variant': row['selected']['variant'], 'tensor_sha256': row['selected']['tensor_sha256'],
                'selected_blob_sha256': row['selected_blob_sha256'], 'export_receipt_sha256': receipt_ref['sha256'],
                'fp16_sha256': exported['exports']['fp16']['sha256'], **sizes}
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as error:
        raise ValueError('complete independently pinned selected/export evidence required') from error


def verify_soup_export(discovery, evidence, *, approved, allow_fixture=False):
    """Recheck a pre-existing uniform soup and its actual internal-dev packets."""
    import f04_gpu_full_dev as gpu
    import f05_soup_dev as soup
    from f04_size_ensemble import compare_predictions
    try:
        group = discovery['id'].removeprefix('soup-')
        if group not in soup.GROUPS:
            raise ValueError('registered uniform soup identity required')
        inputs, report = _pinned(evidence['inputs']), _pinned(evidence['summary'])
        if (report.get('status') != 'F05_uniform_soups_full_internal_dev_evaluated'
                or report.get('inputs_sha256') != evidence['inputs']['sha256']
                or report.get('TTA') is not False or report.get('official_read') is not False
                or report.get('training_executed') is not False
                or inputs.get('official_read') is not False or inputs.get('TTA') is not False
                or inputs.get('batch_size') != 8):
            raise ValueError('completed internal-dev soup inference evidence required')
        if allow_fixture:
            if inputs.get('fixture_only') is not True or report.get('fixture_only') is not True:
                raise ValueError('synthetic-only soup inputs required')
        else:
            from pathlib import Path as _Path
            policy = json.loads((_Path(__file__).resolve().parents[2] / 'configs/F05-SOUP-DEV-20261105.json').read_text())
            soup.validate_inputs(inputs, policy)
            if report.get('dev_count') != 5140 or report.get('dev_positive') != 167:
                raise ValueError('complete original dev soup evaluation required')
        matches = [item for item in inputs.get('members', []) if item.get('round_id') == group]
        if len(matches) != 1:
            raise ValueError('one sealed soup member is required')
        member = matches[0]
        site = soup.verify_assets(gpu, member)
        lineage = _pinned(member['soup'])
        parents = lineage.get('parents')
        if not isinstance(parents, list) or len(parents) < 2:
            raise ValueError('at least two actual soup parents are required')
        work = Path(site['work_root'])
        for parent in parents:
            name = parent.get('round_id')
            if not isinstance(name, str) or _forbidden_id(name) or Path(name).name != name:
                raise ValueError('medical or invalid soup parent identity')
            science_path = work / 'rounds' / name / 'SCIENCE.json'
            science = _pinned({'path': str(science_path), 'sha256': parent['science_sha256']})
            source = science.get('public_source') or {}
            if (approved.get(source.get('sha256')) != source.get('training_data')
                    or science.get('candidate', {}).get('weight_sha256') != source.get('sha256')
                    or science.get('job', {}).get('initial_checkpoint') is not None):
                raise ValueError('soup parent public initialization differs')
        manifest = _pinned(inputs['manifest'])
        rows, indices = manifest['rows'], manifest['dev_indices']
        if (not indices or len(set(indices)) != len(indices)
                or any(type(index) is not int or not 0 <= index < len(rows) for index in indices)
                or (not allow_fixture and (len(indices) != 5140 or sum(rows[i]['label'] == 1 for i in indices) != 167))):
            raise ValueError('complete distinct internal dev manifest required')
        labels = [rows[index]['label'] for index in indices]
        output_root = Path(evidence['summary']['path']).parent
        release_path = output_root / 'RELEASE.json'
        if release_path.is_symlink() or not release_path.is_file():
            raise ValueError('supervised soup release is missing')
        release = json.loads(release_path.read_text())
        worker = _pinned({'path': str(output_root / 'WORKER-SPEC.json'), 'sha256': release['spec_sha256']})
        if (release.get('status') != 'all_rank_device_proofs_passed'
                or worker.get('inputs') != evidence['inputs'] or worker.get('soup_mode') is not True):
            raise ValueError('supervised soup release or worker input differs')
        vectors = {}
        for precision in ('fp32', 'fp16'):
            packets = []
            for physical, assignment in enumerate(inputs['plan']):
                if assignment['member'] != group:
                    continue
                path = output_root / 'packets' / ('rank-' + str(physical) + '-' + precision + '.json')
                key = str(physical) + '-' + precision
                expected_sha = report['comparison']['packet_sha256'][key]
                packet = _pinned({'path': str(path), 'sha256': expected_sha})
                proof_sha = release.get('proof_sha256', {}).get(str(physical))
                proof = _pinned({'path': str(output_root / 'proofs' / ('rank-' + str(physical) + '.json')),
                                 'sha256': proof_sha})
                if (packet.get('device_rank') != physical or packet.get('member') != group
                        or packet.get('rank') != assignment['rank'] or packet.get('error') is not None
                        or packet.get('storage_precision') != precision
                        or packet.get('model_sha256') != member['exports'][precision]['model']['sha256']
                        or packet.get('input_sha256') != evidence['inputs']['sha256']
                        or packet.get('proof_sha256') != proof_sha or proof.get('rank') != physical
                        or packet.get('TTA') is not False or packet.get('official_read') is not False):
                    raise ValueError('soup rank packet identity differs')
                packets.append(packet)
            worlds = {assignment['world'] for assignment in inputs['plan'] if assignment['member'] == group}
            if len(worlds) != 1:
                raise ValueError('one fixed soup shard world is required')
            prediction = gpu.canonical_member(rows, indices, packets, worlds.pop())
            vectors[precision] = [float(item['logit']) for item in prediction]
        measured = compare_predictions(labels, vectors['fp32'], vectors['fp16'])
        saved = report['comparison']['soups'][group]
        for precision in ('fp32', 'fp16'):
            if any(not math.isclose(float(measured[precision][key]), float(saved[precision][key]),
                                    rel_tol=0, abs_tol=1e-12)
                   for key in ('auroc', 'ap', 'error_twice')):
                raise ValueError('soup full-dev summary differs from the actual rank packets')
        auroc = measured['fp32']['auroc']
        if not math.isfinite(auroc) or not 0 <= auroc <= 1:
            raise ValueError('finite internal-dev soup AUROC required')
        return {'id': discovery['id'], 'auroc': auroc, 'variant': member['selected']['variant'],
                'fp16_bytes': member['exports']['fp16']['model']['bytes'],
                'fp32_bytes': member['exports']['fp32']['model']['bytes'],
                'fp16_sha256': member['exports']['fp16']['model']['sha256'],
                'soup_receipt_sha256': member['soup']['sha256'], 'dev_summary_sha256': evidence['summary']['sha256']}
    except (KeyError, TypeError, IndexError, OSError, json.JSONDecodeError) as error:
        raise ValueError('complete pinned soup, parent and internal-dev evidence required') from error


def screen_candidates(candidates, evidence, *, inventory_sha, soup_catalog_sha, approved, allow_fixture=False):
    """Report unknown evidence explicitly; never treat it as a loadable member."""
    if (not isinstance(candidates, list) or not isinstance(evidence, dict)
            or not _is_sha(inventory_sha) or not _is_sha(soup_catalog_sha)
            or any(item.get('order') != i for i, item in enumerate(candidates))
            or len({item.get('id') for item in candidates}) != len(candidates)
            or not set(evidence) <= {item.get('id') for item in candidates}):
        raise ValueError('a complete unique ordered candidate inventory is required')
    verified, unknown, excluded = [], [], []
    for item in candidates:
        name, kind, row = item['id'], item['kind'], item['discovery']
        if ((kind == 'round' and row.get('round_id') != name)
                or (kind == 'soup' and row.get('id') != name)):
            raise ValueError('candidate inventory and discovery IDs differ')
        if _forbidden_id(name):
            excluded.append({'id': name, 'reason': 'medical_pretraining_forbidden'}); continue
        if kind == 'round':
            status = row.get('status')
            if status in ('unknown_or_excluded', 'selected_checkpoint_ambiguous'):
                unknown.append({'id': name, 'reason': status}); continue
            if status == 'selected_checkpoint_missing':
                selected = row.get('selected')
                spec = row.get('spec')
                if not isinstance(selected, dict) or not isinstance(spec, dict):
                    unknown.append({'id': name, 'reason': 'missing_checkpoint_inventory_inconsistent'}); continue
                value = selected.get('auroc')
                if (row.get('checkpoint_candidates') != [] or type(value) not in (int, float)
                        or not math.isfinite(value) or not 0 <= value <= 1
                        or selected.get('variant') not in ('online', 'ema')
                        or not _is_sha(selected.get('tensor_sha256'))
                        or not _is_sha(row.get('selected_blob_sha256'))
                        or not isinstance(spec.get('freeze'), dict)
                        or not isinstance(row.get('science'), dict)):
                    unknown.append({'id': name, 'reason': 'missing_checkpoint_inventory_inconsistent'}); continue
                excluded.append({'id': name, 'reason': 'selected_checkpoint_missing'}); continue
            if status != 'prediction_verified_weight_pending':
                unknown.append({'id': name, 'reason': 'unrecognized_or_incomplete_discovery'}); continue
            if not isinstance(row.get('selected'), dict):
                unknown.append({'id': name, 'reason': 'incomplete_candidate_discovery'}); continue
            selected_auroc = row['selected'].get('auroc')
            if (type(selected_auroc) in (int, float) and math.isfinite(selected_auroc)
                    and 0 <= selected_auroc < THRESHOLD):
                excluded.append({'id': name, 'reason': 'below_fixed_selected_auroc_threshold'}); continue
            try:
                state, reason = f03_candidate_gate(row, THRESHOLD, approved)
            except (KeyError, TypeError, AttributeError):
                unknown.append({'id': name, 'reason': 'incomplete_candidate_discovery'}); continue
            if state == 'excluded':
                excluded.append({'id': name, 'reason': reason}); continue
            if state == 'pending':
                unknown.append({'id': name, 'reason': reason}); continue
        elif kind != 'soup':
            raise ValueError('unregistered candidate kind')
        if name not in evidence:
            unknown.append({'id': name, 'reason': 'strict_load_or_dev_evidence_missing'}); continue
        try:
            details = evidence[name]
            export_ref = details.get('export') if kind == 'round' and isinstance(details, dict) and 'export' in details else details
            audit_ref = details.get('adaptation_audit') if kind == 'round' and isinstance(details, dict) else None
            adaptation_ref = details.get('adaptation_receipt') if kind == 'round' and isinstance(details, dict) else None
            original_inventory = details.get('original_inventory') if kind == 'round' and isinstance(details, dict) else None
            export_inventory_sha = inventory_sha
            if original_inventory is not None:
                old = _pinned(original_inventory)
                matches = [entry for entry in old.get('members', []) if entry.get('round_id') == name]
                if (old.get('status') != 'read_only_F03_candidate_discovery_no_fusion'
                        or old.get('official_read') is not False or len(matches) != 1
                        or (not allow_fixture and old.get('manifest', {}).get('sha256') != BASE_MANIFEST_SHA)):
                    raise ValueError('original export inventory is incomplete or ambiguous')
                require_same_candidate(matches[0], row)
                export_inventory_sha = original_inventory['sha256']
            value = (verify_round_export(row, export_ref, inventory_sha=export_inventory_sha,
                                         approved=approved, adaptation_audit=audit_ref,
                                         adaptation_receipt=adaptation_ref, allow_fixture=allow_fixture)
                     if kind == 'round' else
                     verify_soup_export(row, evidence[name], approved=approved, allow_fixture=allow_fixture))
            if value['auroc'] < THRESHOLD:
                excluded.append({'id': name, 'reason': 'below_fixed_selected_auroc_threshold'}); continue
            verified.append({**value, 'order': item['order']})
        except ValueError as error:
            unknown.append({'id': name, 'reason': str(error)})
    return {'verified': verified, 'unknown': unknown, 'excluded': excluded,
            'candidate_order': [item['id'] for item in candidates],
            'inventory_sha256': inventory_sha, 'soup_catalog_sha256': soup_catalog_sha,
            'official_read': False, 'TTA': False}


def freeze_selection(screen, *, policy_ref=None):
    """Fix members from verified candidates before reading any F05 fusion."""
    if (screen.get('official_read') is not False or screen.get('TTA') is not False
            or screen.get('unknown') or not screen.get('verified') or policy_ref is None):
        raise ValueError('all candidate evidence must be resolved before F05 selection')
    policy = _pinned(policy_ref)
    if (policy.get('format') != 'f05-final-selection-policy-v1'
            or policy.get('status') != 'registered_before_new_f05_fusion'
            or policy.get('candidate_order') != screen.get('candidate_order')
            or policy.get('inventory_sha256') != screen.get('inventory_sha256')
            or policy.get('soup_catalog_sha256') != screen.get('soup_catalog_sha256')
            or policy.get('threshold') != THRESHOLD or policy.get('byte_limit') != BYTE_LIMIT
            or policy.get('ordering') != 'selected_auroc_desc_then_sealed_order'
            or policy.get('overflow') != 'stop_before_first_over_limit'
            or policy.get('normalization') != 'dev_float64_zscore_ddof0' or policy.get('weights') != 'equal'
            or policy.get('TTA') is not False or policy.get('official_read') is not False
            or policy.get('fusion_scores_read') is not False):
        raise ValueError('published F05 candidate order and fixed selection policy required')
    directions = policy.get('direction_terminals')
    expected = {'soup', 'oof', 'clean', 'distill', 'resolution_768'}
    if not isinstance(directions, dict) or set(directions) != expected:
        raise ValueError('all predeclared F05 directions require terminal evidence')
    for name, reference in directions.items():
        terminal = _pinned(reference)
        if (terminal.get('direction') != name or terminal.get('status') not in ('completed', 'closed_no_candidate')
                or terminal.get('official_read') is not False):
            raise ValueError('a required F05 direction is not confirmed terminal')
    chosen = select_by_fixed_budget(screen['verified'])
    if not chosen['members']:
        raise ValueError('no eligible member fits the fixed fp16 size budget')
    return {'status': 'F05_candidates_fixed_before_fusion', 'inventory_sha256': screen['inventory_sha256'],
            'soup_catalog_sha256': screen['soup_catalog_sha256'],
            'policy_sha256': policy_ref['sha256'],
            'eligible_order': [item['id'] for item in sorted(screen['verified'], key=lambda x: (-x['auroc'], x['order']))],
            'fp16': chosen, 'verified': screen['verified'], 'excluded': screen['excluded'],
            'threshold': THRESHOLD, 'byte_limit': BYTE_LIMIT, 'official_read': False, 'TTA': False}


def select_by_fixed_budget(verified_candidates):
    """F04 order, inclusive .988 threshold, exact fp16 bytes and stop-on-overflow."""
    rows = list(verified_candidates)
    return rank_and_cap(rows, 'fp16', BYTE_LIMIT)


def fuse_internal_dev(labels, vectors, member_order):
    """The F04 float64 population-z equal blend; caller must bind its inputs."""
    return equal_zscore(labels, vectors, member_order)


def _write_new(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def _published_source(metadata, source_ref, commit, uid):
    """Reuse the published F04 tree checks and verify this selector's blobs."""
    from f04_export_member import verify_published
    from f04_bounded_fusion import verify_blob
    metadata = Path(metadata)
    proof = _pinned(source_ref)
    tree = json.loads((metadata / 'REMOTE-TREE.json').read_text())
    if (not re.fullmatch('[0-9a-f]{40}', commit or '')
            or proof.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
            or proof.get('source_root') != str(SOURCE) or proof.get('pod_uid') != uid
            or proof.get('commit') != commit or proof.get('tree') != tree.get('sha')
            or proof.get('GPU_initialized') is not False or proof.get('official_read') is not False):
        raise ValueError('published Pod-local selector source proof differs')
    files = verify_published(SOURCE, metadata, commit)
    for name in ('tools/diagnostics/f05_final_selection.py', 'tools/diagnostics/f05_soup_dev.py',
                 'tools/diagnostics/f04_gpu_full_dev.py', 'tools/diagnostics/f04_bounded_fusion.py',
                 'configs/F05-SOUP-DEV-20261105.json'):
        files[name] = verify_blob(SOURCE, metadata, name)
    return files


def _inside_work(reference, work):
    path = Path(reference.get('path', ''))
    if not path.is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(work):
        raise ValueError('source evidence must stay inside the verified work root')


def _check_references(value, work):
    if isinstance(value, dict):
        if 'path' in value:
            _inside_work(value, work)
        for item in value.values():
            _check_references(item, work)
    elif isinstance(value, list):
        for item in value:
            _check_references(item, work)


def main():
    """Seal candidate screening only; full-dev fusion is a later owner stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'soup-catalog', 'policy', 'evidence', 'source-proof'):
        parser.add_argument('--' + name, nargs=2, metavar=('PATH', 'SHA256'), required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-metadata', type=Path, required=True)
    parser.add_argument('--git-commit', required=True)
    args = parser.parse_args()
    from source_chain import cpu_context
    context = cpu_context()
    work = args.work_root.resolve(strict=True)
    refs = {name: dict(zip(('path', 'sha256'), getattr(args, name)))
            for name in ('inventory', 'soup_catalog', 'policy', 'evidence', 'source_proof')}
    for reference in refs.values():
        _inside_work(reference, work)
    files = _published_source(args.remote_metadata, refs['source_proof'], args.git_commit, context['uid'])
    inventory, catalog, policy, evidence = (_pinned(refs[name])
                                             for name in ('inventory', 'soup_catalog', 'policy', 'evidence'))
    for value in (inventory, catalog, policy, evidence):
        _check_references(value, work)
    if (inventory.get('manifest', {}).get('sha256') != BASE_MANIFEST_SHA
            or catalog.get('format') != 'f05-soup-candidates-v1'
            or catalog.get('official_read') is not False
            or evidence.get('format') != 'f05-candidate-evidence-v1'
            or evidence.get('official_read') is not False or not isinstance(evidence.get('entries'), dict)):
        raise ValueError('fixed internal-dev inventory, soup catalog and private evidence required')
    candidates = build_candidate_inventory(inventory, catalog['soups'], policy['candidate_order'])
    output = args.output
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')):
        raise ValueError('fresh independent technical-prep selection output required')
    approved = json.loads((SOURCE / 'configs/F03-THRESHOLD-EQUAL.json').read_text())['approved_general_public_weight_sha256']
    screen = screen_candidates(candidates, evidence['entries'], inventory_sha=refs['inventory']['sha256'],
                               soup_catalog_sha=refs['soup_catalog']['sha256'], approved=approved)
    output.mkdir()
    screen.update(execution=context, git_commit=args.git_commit, source_proof=refs['source_proof'],
                  source_files=files, evidence_manifest=refs['evidence'])
    screen_ref = _write_new(output / 'SCREEN.json', screen)
    fixed = freeze_selection(screen, policy_ref=refs['policy'])
    fixed.update(screen_sha256=screen_ref['sha256'], source_proof=refs['source_proof'], git_commit=args.git_commit)
    fixed_ref = _write_new(output / 'PREFLIGHT.json', fixed)
    print(json.dumps({'status': fixed['status'], 'preflight_sha256': fixed_ref['sha256'],
                      'screen_sha256': screen_ref['sha256'], 'members': fixed['fp16']['members'],
                      'fp16_bytes': fixed['fp16']['bytes']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(SOURCE / 'tools/pretraining'))
    main()

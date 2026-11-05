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


def require_fold_candidate_census(round_inventory):
    """Do not freeze a census that silently omits the registered fold rounds."""
    rows = round_inventory.get('members') if isinstance(round_inventory, dict) else None
    if not isinstance(rows, list):
        raise ValueError('complete round discovery required')
    names = [row.get('round_id', '').lower() for row in rows
             if isinstance(row, dict) and isinstance(row.get('round_id'), str)]
    if any(not any(name == 'r' + str(number) or name.startswith('r' + str(number) + '-')
                   for name in names) for number in (265, 266, 267)):
        raise ValueError('three-fold candidate discovery is incomplete or has been silently omitted')


def validate_candidate_population(row, base_ref, *, work_root, allow_fixture=False,
                                  _verified_base=None, registered_manifest_shas=None):
    """Admit original dev or an independently bound registered training fold."""
    from retrain.persistence import canonical_digest
    try:
        work = Path(work_root).resolve(strict=True)
        _inside_work(base_ref, work)
        base = _pinned(base_ref) if _verified_base is None else _verified_base
        freeze_ref = row['spec']['freeze']
        _inside_work(freeze_ref, work)
        freeze = _pinned(freeze_ref)
        name = row['round_id']
        manifest_sha = freeze['experiment91']['manifest_sha256']
        if (not isinstance(name, str) or not name or not _is_sha(manifest_sha)
                or (not allow_fixture and (base_ref['sha256'] != BASE_MANIFEST_SHA
                    or len(base['dev_indices']) != 5140
                    or sum(base['rows'][i]['label'] for i in base['dev_indices']) != 167))):
            raise ValueError('valid original internal-dev population and frozen manifest are required')
        registered_fold = next((number for number in range(3)
                                if name.lower().startswith('r' + str(265 + number) + '-')), None)
        binding = row.get('fold_binding')
        if manifest_sha == base_ref['sha256'] and registered_fold is None and binding is None:
            return {'kind': 'original', 'training_manifest_sha256': manifest_sha}
        if registered_fold is None and binding is None:
            round_root = Path(freeze_ref['path']).parent
            completion_ref = row['spec']['completion']
            _inside_work(completion_ref, work)
            completion = _pinned(completion_ref)
            if (not isinstance(registered_manifest_shas, (set, frozenset)) or len(registered_manifest_shas) != 3
                    or not all(_is_sha(value) for value in registered_manifest_shas)
                    or Path(freeze_ref['path']).name != 'SINGLE-MODEL-FREEZE.json'
                    or (not allow_fixture and round_root.name != name)
                    or Path(completion_ref['path']).name != 'TRAIN_RESULT.json'
                    or Path(completion_ref['path']).parent not in (round_root, round_root / 'evidence')
                    or completion.get('status') != 'completed'):
                raise ValueError('completed round freeze or registered fold manifest identities are unknown')
            if manifest_sha in registered_manifest_shas:
                raise ValueError('registered three-fold manifest used without its matching fold binding')
            return {'kind': 'excluded', 'reason': 'freeze_declared_training_manifest_mismatch'}
        if (registered_fold is None or not isinstance(binding, dict) or type(binding.get('fold')) is not int
                or binding['fold'] != registered_fold
                or manifest_sha == base_ref['sha256']):
            raise ValueError('registered fold or derived training identity is missing or inconsistent')
        for reference in (binding['site'], binding['runtime_binding'], binding['training_manifest'], row['science']):
            _inside_work(reference, work)
        site, runtime, train, science = (_pinned(reference) for reference in
            (binding['site'], binding['runtime_binding'], binding['training_manifest'], row['science']))
        site_science_ref = site['science']['configuration']
        _inside_work(site_science_ref, work)
        site_science = _pinned(site_science_ref)
        round_root = Path(freeze_ref['path']).parent
        if (site.get('round_id') != name or runtime.get('site_binding_sha256') != binding['site']['sha256']
                or Path(site['work_root']).resolve(strict=True) != work
                or Path(binding['site']['path']) != round_root / 'SITE_BINDING.json'
                or Path(binding['runtime_binding']['path']) != round_root / 'RUNTIME_BINDING.json'
                or Path(row['science']['path']) != round_root / 'SCIENCE.json'
                or Path(binding['training_manifest']['path']) != Path(site['metadata_root']) / 'manifest.json'
                or binding['training_manifest']['sha256'] != manifest_sha
                or site_science_ref['sha256'] != row['science']['sha256'] or site_science != science
                or train.get('dev_indices') != base.get('dev_indices')
                or any(train['rows'][index] != base['rows'][index] for index in base['dev_indices'])):
            raise ValueError('fold site, derived manifest, original dev or SCIENCE identity differs')
        if allow_fixture and any(item.get('fixture_only') is not True for item in (site, runtime, science)):
            raise ValueError('synthetic fold mode requires explicit fixture identities')
        from retrain.f05_folds import verify_bound_fold
        roles = verify_bound_fold(site, science)
        derived_ref = {'path': str(Path(site['metadata_root']) / 'FREEZE.json'),
                       'sha256': science['f05_fold']['derived_freeze_sha256']}
        _inside_work(derived_ref, work)
        derived = _pinned(derived_ref)
        if (roles.get('fold_number') != registered_fold
                or roles.get('dev_indices_sha256') != canonical_digest(base['dev_indices'])
                or derived.get('files', {}).get('manifest.json', {}).get('sha256') != manifest_sha):
            raise ValueError('registered fold roles or derived manifest bytes differ')
        return {'kind': 'threefold', 'fold': registered_fold, 'training_manifest_sha256': manifest_sha}
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as error:
        raise ValueError('complete independently bound population evidence required') from error


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
                'successful_updates': row['selected']['successful_updates'],
                'selected_blob_sha256': row['selected_blob_sha256'], 'export_receipt_sha256': receipt_ref['sha256'],
                'fp16_sha256': exported['exports']['fp16']['sha256'], **sizes}
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as error:
        raise ValueError('complete independently pinned selected/export evidence required') from error


def verify_soup_export(discovery, evidence, *, approved, allow_fixture=False, include_vectors=False):
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
        if not allow_fixture:
            from retrain.site_binding import verify_rank_devices
            source_root = Path(inputs.get('source_root', ''))
            source_worker = source_root / 'tools/diagnostics/f04_gpu_full_dev.py'
            parent = worker.get('parent')
            proof_map = release.get('proof_sha256')
            site_ref = {'path': worker['site_binding'], 'sha256': worker['site_binding_sha256']}
            _inside_work(site_ref, work)
            if (not source_root.is_absolute() or worker.get('source_root') != str(source_root)
                    or source_worker.is_symlink() or not source_worker.is_file()
                    or _sha(source_worker) != worker.get('worker_source_sha256')
                    or _sha(source_worker) != inputs.get('git_files', {}).get('tools/diagnostics/f04_gpu_full_dev.py')
                    or not isinstance(parent, dict) or type(parent.get('pid')) is not int or parent['pid'] <= 0
                    or not str(parent.get('start_ticks', '')).isdecimal() or release.get('parent') != parent
                    or not isinstance(proof_map, dict) or set(proof_map) != {str(i) for i in range(8)}):
                raise ValueError('published soup worker or supervised parent differs')
            proofs = [_pinned({'path': str(output_root / 'proofs' / ('rank-' + str(rank) + '.json')),
                               'sha256': proof_map[str(rank)]}) for rank in range(8)]
            if (any(proof.get('parent') != parent or type(proof.get('pid')) is not int or proof['pid'] <= 0
                    or not str(proof.get('start_ticks', '')).isdecimal() for proof in proofs)
                    or len({proof['pid'] for proof in proofs}) != 8):
                raise ValueError('soup device proofs do not bind eight supervised ranks')
            verify_rank_devices(proofs, _pinned(site_ref))
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
        details = {'id': discovery['id'], 'auroc': auroc, 'variant': member['selected']['variant'],
                   'fp16_bytes': member['exports']['fp16']['model']['bytes'],
                   'fp32_bytes': member['exports']['fp32']['model']['bytes'],
                   'fp16_sha256': member['exports']['fp16']['model']['sha256'],
                   'soup_receipt_sha256': member['soup']['sha256'], 'dev_summary_sha256': evidence['summary']['sha256']}
        return (details, vectors, inputs['manifest']) if include_vectors else details
    except (KeyError, TypeError, IndexError, OSError, json.JSONDecodeError) as error:
        raise ValueError('complete pinned soup, parent and internal-dev evidence required') from error


def screen_candidates(candidates, evidence, *, inventory_sha, soup_catalog_sha, approved,
                      base_ref=None, work_root=None, allow_fixture=False, _fixture_registered_manifest_shas=None):
    """Report unknown evidence explicitly; never treat it as a loadable member."""
    if (not isinstance(candidates, list) or not isinstance(evidence, dict)
            or not _is_sha(inventory_sha) or not _is_sha(soup_catalog_sha)
            or any(item.get('order') != i for i, item in enumerate(candidates))
            or len({item.get('id') for item in candidates}) != len(candidates)
            or not set(evidence) <= {item.get('id') for item in candidates}
            or (base_ref is None) != (work_root is None)
            or (not allow_fixture and base_ref is None)):
        raise ValueError('a complete unique ordered candidate inventory is required')
    if base_ref is not None:
        _inside_work(base_ref, Path(work_root).resolve(strict=True))
    verified_base = _pinned(base_ref) if base_ref is not None else None
    fold_cache, registered_manifest_shas = {}, None
    if base_ref is not None:
        registered = []
        for number in range(3):
            prefix = 'r' + str(265 + number) + '-'
            matches = [item for item in candidates if item.get('kind') == 'round'
                       and isinstance(item.get('id'), str) and item['id'].lower().startswith(prefix)]
            if len(matches) != 1 or matches[0]['discovery'].get('status') not in (
                    'prediction_verified_weight_pending', 'selected_checkpoint_missing'):
                break
            item = matches[0]
            try:
                population = validate_candidate_population(item['discovery'], base_ref, work_root=work_root,
                                                           allow_fixture=allow_fixture, _verified_base=verified_base)
                if population.get('kind') != 'threefold' or population.get('fold') != number:
                    raise ValueError('registered fold population differs')
                fold_cache[item['id']] = population
                registered.append(population['training_manifest_sha256'])
            except ValueError as error:
                fold_cache[item['id']] = str(error)
                break
        if len(registered) == 3 and len(set(registered)) == 3:
            registered_manifest_shas = set(registered)
        if _fixture_registered_manifest_shas is not None:
            if not allow_fixture or len(set(_fixture_registered_manifest_shas)) != 3:
                raise ValueError('registered manifest override is synthetic-only')
            registered_manifest_shas = set(_fixture_registered_manifest_shas)
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
            if status not in ('prediction_verified_weight_pending', 'selected_checkpoint_missing',
                              'manifest_mismatch_verified'):
                unknown.append({'id': name, 'reason': 'unrecognized_discovery_status: ' + str(status)}); continue
            if base_ref is not None:
                try:
                    population = fold_cache.get(name)
                    if isinstance(population, str):
                        raise ValueError(population)
                    if population is None:
                        population = validate_candidate_population(row, base_ref, work_root=work_root,
                            allow_fixture=allow_fixture, _verified_base=verified_base,
                            registered_manifest_shas=registered_manifest_shas)
                except ValueError as error:
                    unknown.append({'id': name, 'reason': 'population_identity_unknown: ' + str(error)}); continue
                if population['kind'] == 'excluded':
                    excluded.append({'id': name, 'reason': population['reason']}); continue
                if status == 'manifest_mismatch_verified':
                    unknown.append({'id': name, 'reason': 'manifest_mismatch_disagrees_with_registered_population'}); continue
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
                if matches[0].get('fold_binding') != row.get('fold_binding'):
                    raise ValueError('fold binding changed since the sealed export inventory')
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


def freeze_selection(screen, *, policy_ref=None, allow_fixture=False):
    """Fix members from verified candidates before reading any F05 fusion."""
    if (screen.get('official_read') is not False or screen.get('TTA') is not False
            or screen.get('unknown') or not screen.get('verified') or policy_ref is None):
        raise ValueError('all candidate evidence must be resolved before F05 selection')
    if not allow_fixture:
        require_fold_candidate_census({'members': [{'round_id': name}
                                                  for name in screen.get('candidate_order', [])]})
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


def _selected_fp32_references(preflight, screen_ref, manifest_ref, *, work, allow_fixture, approved):
    """Reuse independently checked historical selected and supervised soup scores."""
    import paired_evidence
    _inside_work(screen_ref, work)
    screen = _pinned(screen_ref)
    refs = [screen['inventory'], screen['soup_catalog'], screen['evidence_manifest']]
    for reference in refs:
        _inside_work(reference, work)
    inventory, catalog, evidence = (_pinned(reference) for reference in refs)
    if (screen['inventory']['sha256'] != preflight.get('inventory_sha256')
            or screen['soup_catalog']['sha256'] != preflight.get('soup_catalog_sha256')
            or inventory.get('status') != 'read_only_F03_candidate_discovery_no_fusion'
            or inventory.get('official_read') is not False or inventory.get('manifest') != manifest_ref
            or catalog.get('format') != 'f05-soup-candidates-v1' or catalog.get('official_read') is not False
            or evidence.get('format') != 'f05-candidate-evidence-v1' or evidence.get('official_read') is not False):
        raise ValueError('sealed candidate, manifest and reference-evidence identities differ')
    reference = paired_evidence._reference(manifest_ref)
    rows = inventory['members']
    soups = catalog['soups']
    selected = preflight['fp16']['members']
    assets = {row['id']: row for row in preflight['verified']}
    vectors, sources = {}, {}
    for name in selected:
        asset = assets[name]
        if name.startswith('soup-'):
            matches = [item for item in soups if item.get('id') == name]
            details_ref = evidence.get('entries', {}).get(name)
            if len(matches) != 1 or not isinstance(details_ref, dict):
                raise ValueError('unique supervised soup reference evidence required')
            _check_references(details_ref, work)
            details, values, soup_manifest = verify_soup_export(matches[0], details_ref, approved=approved,
                                                                 allow_fixture=allow_fixture, include_vectors=True)
            if (soup_manifest != manifest_ref or any(details.get(key) != asset.get(key) for key in
                    ('id', 'auroc', 'variant', 'fp16_sha256', 'soup_receipt_sha256', 'dev_summary_sha256'))):
                raise ValueError('soup reference differs from its fixed candidate or original dev')
            vectors[name] = values['fp32']
            sources[name] = {'kind': 'supervised_soup_fp32_packets',
                             'inputs_sha256': details_ref['inputs']['sha256'],
                             'summary_sha256': details_ref['summary']['sha256']}
        else:
            matches = [item for item in rows if item.get('round_id') == name]
            if len(matches) != 1:
                raise ValueError('unique original selected member required')
            row = matches[0]
            spec = {'freeze': row['spec']['freeze'], 'completion': row['spec']['completion']}
            for ref in spec.values():
                _inside_work(ref, work)
            frozen = _pinned(spec['freeze'])
            original = frozen['selected']
            _inside_work({'path': original['prediction_path']}, work)
            _inside_work({'path': original['precision_path']}, work)
            population = validate_candidate_population(row, manifest_ref, work_root=work,
                                                       allow_fixture=allow_fixture)
            if population['kind'] not in ('original', 'threefold'):
                raise ValueError('selected fp32 source is outside the registered population')
            member_reference = {**reference, 'manifest_sha256': population['training_manifest_sha256']}
            scores, identity, _ = paired_evidence._selected(spec, member_reference)
            if (any(identity.get(key) != row['selected'].get(key) for key in
                    ('variant', 'cycle', 'successful_updates', 'tensor_sha256', 'auroc'))
                    or any(identity.get(key) != asset.get(key) for key in
                           ('variant', 'successful_updates', 'tensor_sha256', 'auroc'))
                    or frozen.get('blob_sha256') != asset.get('selected_blob_sha256')):
                raise ValueError('fp32 prediction does not match the fixed selected member')
            vectors[name] = scores
            sources[name] = {'kind': 'original_selected_fp32_prediction', 'freeze_sha256': spec['freeze']['sha256'],
                             'prediction_sha256': identity['prediction']['sha256'],
                             'precision_sha256': identity['precision']['sha256']}
    return vectors, sources


def _fuse_from_selected_packets(preflight_ref, manifest_ref, seal_refs, *, work_root,
                                policy_ref=None, screen_ref=None, producer_commit=None,
                                producer_worker_sha=None, allow_fixture=False, approved=None):
    """Reconstruct selected fp16 dev logits from pinned supervised packets."""
    from retrain.parallel_evaluation import canonical_predictions
    work = Path(work_root).resolve(strict=True)
    for reference in (preflight_ref, manifest_ref, *seal_refs):
        _inside_work(reference, work)
    preflight, manifest = _pinned(preflight_ref), _pinned(manifest_ref)
    if (preflight.get('status') != 'F05_candidates_fixed_before_fusion'
            or preflight.get('official_read') is not False or preflight.get('TTA') is not False
            or not isinstance(preflight.get('fp16'), dict) or not isinstance(preflight.get('verified'), list)
            or (allow_fixture and (preflight.get('fixture_only') is not True or manifest.get('fixture_only') is not True))
            or (not allow_fixture and (preflight.get('fixture_only') is True or manifest_ref['sha256'] != BASE_MANIFEST_SHA))):
        raise ValueError('a sealed final selection and original internal dev are required')
    if screen_ref is None or preflight.get('screen_sha256') != screen_ref.get('sha256'):
        raise ValueError('pinned screening receipt required for original fp32 references')
    if not allow_fixture:
        if (policy_ref is None or not re.fullmatch('[0-9a-f]{40}', producer_commit or '')
                or not _is_sha(producer_worker_sha)):
            raise ValueError('verified published producer, policy and screening receipt required')
        _inside_work(policy_ref, work); _inside_work(screen_ref, work)
        if (preflight.get('policy_sha256') != policy_ref.get('sha256')
                or preflight.get('screen_sha256') != screen_ref.get('sha256')):
            raise ValueError('preflight policy or screening identity differs')
        expected = freeze_selection(_pinned(screen_ref), policy_ref=policy_ref)
        if any(preflight.get(key) != value for key, value in expected.items()):
            raise ValueError('preflight does not reproduce the fixed screening rule')
    order = preflight['fp16'].get('members')
    all_verified = preflight['verified']
    if (not isinstance(order, list) or not order or len(set(order)) != len(order)
            or len({row.get('id') for row in all_verified}) != len(all_verified)):
        raise ValueError('distinct fixed members and verified assets required')
    assets = {row['id']: row for row in all_verified}
    if (not set(order) <= set(assets)
            or any(not _is_sha(assets[name].get('fp16_sha256')) or type(assets[name].get('fp16_bytes')) is not int
                   or assets[name]['fp16_bytes'] <= 0 or assets[name].get('variant') not in ('online', 'ema')
                   for name in order)):
        raise ValueError('complete fixed fp16 member identities required')
    weight_bytes = sum(assets[name]['fp16_bytes'] for name in order)
    if preflight['fp16'].get('bytes') != weight_bytes or weight_bytes > BYTE_LIMIT:
        raise ValueError('selected actual weight total differs from fixed budget')
    rows, indices = manifest['rows'], manifest['dev_indices']
    if (not isinstance(rows, list) or not isinstance(indices, list) or not indices
            or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
            or any(type(rows[i].get('label')) is not int or rows[i]['label'] not in (0, 1) for i in indices)
            or set(rows[i]['label'] for i in indices) != {0, 1}
            or (not allow_fixture and (len(indices) != 5140 or sum(rows[i]['label'] for i in indices) != 167
                or any(not rows[i]['relative_path'].startswith('train/') for i in indices)))):
        raise ValueError('complete internal-dev population required')
    if not isinstance(seal_refs, list) or not seal_refs or len({ref['sha256'] for ref in seal_refs}) != len(seal_refs):
        raise ValueError('distinct completed selected-dev packet seals required')
    vectors, used = {}, []
    for seal_ref in seal_refs:
        seal = _pinned(seal_ref)
        _check_references(seal, work)
        if (seal.get('format') != 'f05-selected-dev-packets-v1' or seal.get('status') != 'selected_fp16_dev_completed'
                or seal.get('preflight_sha256') != preflight_ref['sha256']
                or seal.get('manifest_sha256') != manifest_ref['sha256']
                or seal.get('storage') != 'fp16' or seal.get('arithmetic') != 'fp32'
                or seal.get('TTA') is not False or seal.get('official_read') is not False
                or (allow_fixture and seal.get('fixture_only') is not True)
                or (not allow_fixture and seal.get('fixture_only') is True)):
            raise ValueError('fixed selected fp16 internal-dev packet seal required')
        inputs, spec, release = (_pinned(seal[key]) for key in ('inputs', 'worker_spec', 'release'))
        if (inputs.get('format') != 'f05-selected-dev-inputs-v1'
                or inputs.get('status') != 'selected_fp16_dev_inputs_sealed'
                or inputs.get('preflight_sha256') != preflight_ref['sha256'] or inputs.get('manifest') != manifest_ref
                or inputs.get('batch_size') != 8 or inputs.get('view') != 'identity' or inputs.get('model_mode') != 'eval'
                or inputs.get('dev_count') != len(indices)
                or inputs.get('dev_positive') != sum(rows[i]['label'] for i in indices)
                or inputs.get('TTA') is not False or inputs.get('official_read') is not False
                or spec.get('official') is not None or spec.get('soup_mode', False) is not False
                or spec.get('inputs') != seal['inputs'] or release.get('spec_sha256') != seal['worker_spec']['sha256']
                or release.get('status') != 'all_rank_device_proofs_passed'
                or not isinstance(release.get('proof_sha256'), dict)
                or set(release['proof_sha256']) != {str(i) for i in range(8)}):
            raise ValueError('input plan and supervised release do not bind the frozen selection')
        if allow_fixture and (inputs.get('fixture_only') is not True or spec.get('fixture_only') is not True):
            raise ValueError('fixture mode requires explicit synthetic producer inputs')
        group_rows, plan = inputs.get('members'), inputs.get('plan')
        if (not isinstance(group_rows, list) or not group_rows or not isinstance(plan, list) or len(plan) != 8
                or [entry.get('device_rank') for entry in plan] != list(range(8))
                or len({member.get('id') for member in group_rows}) != len(group_rows)
                or len({member.get('round_id') for member in group_rows}) != len(group_rows)
                or any(not isinstance(member.get('round_id'), str) or not member['round_id']
                       for member in group_rows)):
            raise ValueError('eight device assignments and distinct input members required')
        groups = {member['round_id']: member for member in group_rows}
        group_ids = {member['id'] for member in group_rows}
        if (not group_ids <= set(order) or group_ids & set(vectors)
                or {entry.get('member') for entry in plan} != set(groups)):
            raise ValueError('producer includes a duplicate or unselected member')
        for member in group_rows:
            name = member['id']
            model = member.get('model') or {}
            model_path = Path(model.get('path', ''))
            _inside_work(model, work)
            _inside_work(member['metadata'], work)
            metadata = _pinned(member['metadata'])
            tensor_sha = model.get('fp16_tensor_sha256', model.get('tensor_sha256'))
            expected_round = name.removeprefix('soup-') if name.startswith('soup-') else name
            metadata_path = Path(member['metadata']['path'])
            if (member['round_id'] != expected_round
                    or (not allow_fixture and (model_path.name != 'model.pt' or metadata_path.name != 'model.json'
                        or model_path.parent != metadata_path.parent))
                    or model.get('sha256') != assets[name]['fp16_sha256']
                    or model.get('bytes') != assets[name]['fp16_bytes']
                    or member.get('selected', {}).get('variant') != assets[name]['variant']
                    or not _is_sha(tensor_sha) or metadata.get('model_sha256') != model['sha256']
                    or metadata.get('checkpoint', {}).get('tensor_sha256') != tensor_sha
                    or metadata.get('checkpoint', {}).get('variant') != assets[name]['variant']
                    or model_path.is_symlink() or not model_path.is_file()
                    or model_path.stat().st_size != model['bytes'] or _sha(model_path) != model['sha256']):
                raise ValueError('actual fp16 model bytes or selected variant differ from preflight')
        if (not isinstance(seal.get('proofs'), list) or len(seal['proofs']) != 8
                or not isinstance(seal.get('packets'), list) or len(seal['packets']) != 8):
            raise ValueError('eight device proofs and rank packets required')
        proofs, packets = [], []
        for physical, (proof_ref, packet_ref, assignment) in enumerate(zip(seal['proofs'], seal['packets'], plan)):
            proof, packet = _pinned(proof_ref), _pinned(packet_ref)
            if (proof.get('rank') != physical or release.get('proof_sha256', {}).get(str(physical)) != proof_ref['sha256']
                    or packet.get('device_rank') != physical or packet.get('member') != assignment.get('member')
                    or packet.get('rank') != assignment.get('rank') or packet.get('error') is not None
                    or packet.get('input_sha256') != seal['inputs']['sha256']
                    or packet.get('proof_sha256') != proof_ref['sha256']
                    or packet.get('TTA') is not False or packet.get('official_read') is not False):
                raise ValueError('selected-dev supervised rank packet differs')
            if packet.get('model_sha256') not in (None, groups[assignment['member']]['model']['sha256']):
                raise ValueError('rank packet selected model differs')
            proofs.append(proof); packets.append(packet)
        if not allow_fixture:
            from retrain.site_binding import verify_rank_devices
            source_proof = _pinned(seal['source_tree_proof'])
            source_root = Path(source_proof.get('source_root', ''))
            worker = source_root / 'tools/diagnostics/f04_gpu_full_dev.py'
            site_ref = {'path': spec['site_binding'], 'sha256': spec['site_binding_sha256']}
            _inside_work(site_ref, work)
            site = _pinned(site_ref)
            if (source_proof.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
                    or source_proof.get('GPU_initialized') is not False or source_proof.get('official_read') is not False
                    or source_proof.get('commit') != producer_commit
                    or source_proof.get('pod_uid') != site.get('pod', {}).get('uid')
                    or not source_root.is_absolute() or spec.get('source_root') != str(source_root)
                    or worker.is_symlink() or not worker.is_file()
                    or _sha(worker) != producer_worker_sha or spec.get('worker_source_sha256') != producer_worker_sha
                    or release.get('parent') != spec.get('parent')
                    or any(proof.get('parent') != spec.get('parent') for proof in proofs)):
                raise ValueError('published producer source or supervised parent differs')
            verify_rank_devices(proofs, site)
        counts = seal.get('forward_counts')
        if not isinstance(counts, dict) or set(counts) != set(group_ids) or any(counts[name] != len(indices) for name in group_ids):
            raise ValueError('one fp16 forward per selected member and dev row is required')
        for name in groups:
            group_plan = [entry for entry in plan if entry['member'] == name]
            worlds = {entry.get('world') for entry in group_plan}
            if len(worlds) != 1:
                raise ValueError('one fixed shard world is required for each member')
            world = worlds.pop()
            if type(world) is not int or not 1 <= world <= 8 or len(group_plan) != world:
                raise ValueError('incomplete member shard assignments')
            group_packets = [packet for packet in packets if packet['member'] == name]
            values = canonical_predictions(rows, indices, group_packets, 8, world)
            vectors[groups[name]['id']] = [float(item['logit']) for item in values]
        used.append(seal_ref['sha256'])
    if set(vectors) != set(order):
        raise ValueError('selected members lack complete fp16 internal-dev packets')
    if approved is None:
        approved = json.loads((SOURCE / 'configs/F03-THRESHOLD-EQUAL.json').read_text())['approved_general_public_weight_sha256']
    fp32, fp32_sources = _selected_fp32_references(preflight, screen_ref, manifest_ref, work=work,
                                                    allow_fixture=allow_fixture, approved=approved)
    labels = [rows[i]['label'] for i in indices]
    result = fuse_internal_dev(labels, vectors, order)
    baseline = fuse_internal_dev(labels, fp32, order)
    import numpy as np
    from f04_size_ensemble import compare_predictions
    comparisons = {}
    for name in order:
        old = np.asarray(fp32[name], dtype=np.float64)
        new = np.asarray(vectors[name], dtype=np.float64)
        old_moments, new_moments = baseline['normalization'][name], result['normalization'][name]
        old_z = (old - old_moments['mean']) / old_moments['std']
        new_z = (new - new_moments['mean']) / new_moments['std']
        comparisons[name] = {**compare_predictions(labels, old, new),
                             'max_abs_z_score_difference': float(np.max(np.abs(old_z - new_z)))}
    return {**result, 'status': 'F05_fixed_fp16_internal_dev_equal_z',
            'preflight_sha256': preflight_ref['sha256'], 'manifest_sha256': manifest_ref['sha256'],
            'packet_seal_sha256': used, 'fp16_bytes': weight_bytes, 'dev_count': len(indices),
            'fp32_reference': {'sources': fp32_sources, 'metrics': baseline['metrics'],
                               'normalization': baseline['normalization']},
            'member_precision_comparison': comparisons,
            'ensemble_max_abs_z_score_difference': float(np.max(np.abs(baseline['scores'] - result['scores']))),
            'TTA': False, 'official_read': False}


def fuse_from_selected_packets(preflight_ref, manifest_ref, seal_refs, *, work_root,
                               policy_ref=None, screen_ref=None, producer_commit=None,
                               producer_worker_sha=None, allow_fixture=False, approved=None):
    try:
        return _fuse_from_selected_packets(preflight_ref, manifest_ref, seal_refs,
                                           work_root=work_root, policy_ref=policy_ref, screen_ref=screen_ref,
                                           producer_commit=producer_commit, producer_worker_sha=producer_worker_sha,
                                           allow_fixture=allow_fixture, approved=approved)
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as error:
        raise ValueError('complete pinned selected-dev packet evidence is required') from error


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
                 'tools/diagnostics/paired_evidence.py',
                 'runtime/bundle/retrain/parallel_evaluation.py', 'runtime/bundle/retrain/n05_metrics.py',
                 'runtime/bundle/retrain/site_binding.py', 'runtime/bundle/retrain/f05_folds.py',
                 'runtime/bundle/retrain/persistence.py', 'runtime/bundle/retrain/age.py',
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
    require_fold_candidate_census(inventory)
    candidates = build_candidate_inventory(inventory, catalog['soups'], policy['candidate_order'])
    output = args.output
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')):
        raise ValueError('fresh independent technical-prep selection output required')
    approved = json.loads((SOURCE / 'configs/F03-THRESHOLD-EQUAL.json').read_text())['approved_general_public_weight_sha256']
    screen = screen_candidates(candidates, evidence['entries'], inventory_sha=refs['inventory']['sha256'],
                               soup_catalog_sha=refs['soup_catalog']['sha256'], approved=approved,
                               base_ref=inventory['manifest'], work_root=work)
    output.mkdir()
    screen.update(execution=context, git_commit=args.git_commit, source_proof=refs['source_proof'],
                  source_files=files, inventory=refs['inventory'], soup_catalog=refs['soup_catalog'],
                  evidence_manifest=refs['evidence'])
    screen_ref = _write_new(output / 'SCREEN.json', screen)
    fixed = freeze_selection(screen, policy_ref=refs['policy'])
    fixed.update(screen_sha256=screen_ref['sha256'], source_proof=refs['source_proof'], git_commit=args.git_commit)
    fixed_ref = _write_new(output / 'PREFLIGHT.json', fixed)
    print(json.dumps({'status': fixed['status'], 'preflight_sha256': fixed_ref['sha256'],
                      'screen_sha256': screen_ref['sha256'], 'members': fixed['fp16']['members'],
                      'fp16_bytes': fixed['fp16']['bytes']}, allow_nan=False), flush=True)


def fuse_main():
    """Reduce already-sealed internal-dev shards; never launch an inference worker."""
    parser = argparse.ArgumentParser(description=fuse_main.__doc__)
    for name in ('preflight', 'screen', 'policy', 'manifest', 'source-proof'):
        parser.add_argument('--' + name, nargs=2, metavar=('PATH', 'SHA256'), required=True)
    parser.add_argument('--packet-seal', nargs=2, metavar=('PATH', 'SHA256'), action='append', required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-metadata', type=Path, required=True)
    parser.add_argument('--git-commit', required=True)
    args = parser.parse_args()
    from source_chain import cpu_context
    context = cpu_context()
    work = args.work_root.resolve(strict=True)
    refs = {name: dict(zip(('path', 'sha256'), getattr(args, name)))
            for name in ('preflight', 'screen', 'policy', 'manifest', 'source_proof')}
    packet_refs = [dict(zip(('path', 'sha256'), value)) for value in args.packet_seal]
    for reference in (*refs.values(), *packet_refs):
        _inside_work(reference, work)
    _check_references(_pinned(refs['policy']), work)
    files = _published_source(args.remote_metadata, refs['source_proof'], args.git_commit, context['uid'])
    output = args.output
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')):
        raise ValueError('fresh independent technical-prep fusion output required')
    result = fuse_from_selected_packets(refs['preflight'], refs['manifest'], packet_refs, work_root=work,
                                        policy_ref=refs['policy'], screen_ref=refs['screen'],
                                        producer_commit=args.git_commit,
                                        producer_worker_sha=files['tools/diagnostics/f04_gpu_full_dev.py'])
    result.pop('scores')  # Per-sample results do not belong in this receipt.
    result.update(execution=context, git_commit=args.git_commit, source_proof=refs['source_proof'],
                  source_files=files, policy_sha256=refs['policy']['sha256'],
                  screen_sha256=refs['screen']['sha256'])
    output.mkdir()
    receipt = _write_new(output / 'SUMMARY.json', result)
    print(json.dumps({'status': result['status'], 'summary_sha256': receipt['sha256'],
                      'members': result['members'], 'fp16_bytes': result['fp16_bytes'],
                      'metrics': result['metrics']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    import sys
    sys.path[:0] = [str(SOURCE / 'runtime/bundle'), str(SOURCE / 'tools/pretraining')]
    if len(sys.argv) > 1 and sys.argv[1] == 'fuse':
        del sys.argv[1]
        fuse_main()
    else:
        main()

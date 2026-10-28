"""Join the four pinned, successful migration receipts without rereading blobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretraining'))
from source_chain import cpu_context, file_identity, read_pinned_receipt, require, sha256, write_new_json

IDENTITY = ('namespace', 'pod', 'uid', 'container', 'python')
GRANT = ('path', 'sha256', 'executor_id', 'threads', 'channel_mask')


def current_receipt(path, expected_sha, context):
    value = read_pinned_receipt(path, expected_sha)
    observed = value.get('execution', {})
    require(value.get('passed') is True and value.get('schema_version') == 1
            and all(observed.get(k) == context[k] for k in IDENTITY)
            and all(observed.get('cpu_allocation', {}).get(k) == context['cpu_allocation'][k] for k in GRANT),
            'successful current execution/allocation receipt required: ' + path)
    return value


def pinned_manifest(reference):
    file_identity(reference['path'])
    require(sha256(reference['path']) == reference['sha256'], 'migration manifest SHA differs')
    return dict(reference)


def run(args):
    context = cpu_context()
    references = {name: {'path': getattr(args, name), 'sha256': getattr(args, name + '_sha256')}
                  for name in ('small', 'bulk', 'train', 'valid')}
    values = {name: current_receipt(ref['path'], ref['sha256'], context) for name, ref in references.items()}
    small, bulk = values['small'], values['bulk']
    require(small['phase'] == 'small' and bulk['phase'] == 'bulk'
            and small['paths'] == bulk['paths'] and small['cache'] == bulk['cache']
            and small['source_modified_by_tool'] is False and bulk['source_modified_by_tool'] is False
            and small['historical_FREEZE_bytes_preserved'] is True, 'copy stage contract differs')
    root = Path(args.work_root)
    require(root.is_absolute() and small['paths']['work_root'] == str(root)
            and Path(args.output).is_relative_to(root / 'technical-prep'), 'explicit migration work-root/output differs')
    paths = small['paths']
    require(all(Path(value).is_absolute() and Path(value).is_relative_to(root) for value in paths.values()),
            'migration target escaped work root')
    manifests = {name: pinned_manifest(values[name]['manifest']) for name in ('small', 'bulk')}
    # Bind the copied CSV digest already computed during bulk; no CSV labels or
    # raw bytes are read here. Runtime load_source uses this exact relative name.
    train_csv = str(Path(paths['dataset_root']) / 'train.csv')
    csv_rows = []
    with Path(manifests['bulk']['path']).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['target_path'] == train_csv:
                csv_rows.append(row)
    require(len(csv_rows) == 1 and csv_rows[0]['sha256'] == csv_rows[0]['target_sha256'] == args.train_csv_sha256,
            'copied train CSV content proof differs from the sealed source SHA')
    require(small['groups']['fixed_9010_stage1_18_objects']['files'] == 19,
            'original 18 frozen files plus FREEZE copy incomplete')
    cache = {}
    for split in ('train', 'valid'):
        value = values[split]
        proof = value['copy_proof']
        require(value['stage'] == 'prepared-cache-rebind' and value['split'] == split
                and value['expected_count'] == value['counts']['entries'] == bulk['cache']['expected_entries'][split]
                and value['target']['cache_root'] == paths['prepared_cache_root']
                and value['target']['policy'] == bulk['cache']['new_policy']
                and value['target']['namespace'] == bulk['cache']['new_namespace']
                and proof['receipt'] == references['bulk']
                and all(proof['manifest'][key] == manifests['bulk'][key] for key in ('path', 'sha256')),
                'split prepared cache proof differs: ' + split)
        evidence = value['byte_evidence']
        require(evidence['target_raw_and_arrays'] == 'fresh_pod_sha256'
                and evidence['source_raw_and_arrays'] == 'pinned-copy-content-proof'
                and evidence['old_raw_or_array_bytes_reread'] == 0
                and evidence['preparation_changed'] is False and evidence['old_receipts_modified'] is False
                and evidence['labels_or_manifest_semantics_read'] is False,
                'prepared bytes/preparation evidence differs')
        entries = pinned_manifest(value['entries_manifest'])
        cache[split] = {'cache_root': str(Path(paths['prepared_cache_root']) / split / 'sealed'),
                        'entries': value['counts']['entries'],
                        'namespace': value['target']['namespace'], 'counts': value['counts'],
                        'entries_manifest': entries, 'receipt': references[split]}
    require(bulk['groups']['prepared512_pillow123']['files'] == sum(row['counts']['entries'] for row in cache.values()),
            'copied array count differs from rebound entry count')
    for key in ('train_identity', 'public_pretrained', 'public_metadata', 'official_manifest'):
        file_identity(paths[key])
    allocation = read_pinned_receipt(context['cpu_allocation']['path'], context['cpu_allocation']['sha256'])
    require(Path(args.guard).is_absolute() and sha256(args.guard) == allocation['guard_sha256'],
            'current CPU guard differs from the pinned allocation')
    result = {'schema_version': 1, 'passed': True, 'phase': 'complete',
        'scope': 'dependency_relocation_and_prepared_cache_binding_only_not_training_admission',
        'at': datetime.now(timezone.utc).isoformat(), 'execution': context, 'paths': paths,
        'inputs': {'small_copy': references['small'], 'bulk_copy': references['bulk'],
                   'train_cache': references['train'], 'valid_cache': references['valid']},
        'copy': {'files': small['files'] + bulk['files'],
            'bytes': small['bytes'] + bulk['bytes'], 'manifests': manifests,
            'groups': {'small': small['groups'], 'bulk': bulk['groups']}},
        'cache': {'policy': bulk['cache']['new_policy'], 'namespace': bulk['cache']['new_namespace'], **cache},
        'training_csv': {'path': train_csv, 'sha256': args.train_csv_sha256,
                          'copied_content_matches_sealed_sha': True, 'labels_parsed_by_finalizer': False},
        'source_modified_by_tool': False, 'historical_FREEZE_bytes_preserved': True,
        'canonical_source_identity_rewritten': False, 'official_manifest_semantics_read': False,
        'preparation_changed': False, 'old_clock_or_builder_state_inherited': False,
        'retained_models_used_as_training_dependencies': False, 'GPU_work': False,
        'verification': {'raw_or_weight_bytes_rehashed_by_finalizer': False,
                         'pinned_migration_manifests_rehashed': True,
                         'complete_copy_and_cache_content_evidence_in_inputs': True},
        'guard': {'path': args.guard, 'sha256': allocation['guard_sha256']},
        'free_bytes_after': shutil.disk_usage(root).free,
        'implementation_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                   ('relocate_assets.py', 'cache_rebind.py', 'finalize_relocation.py')},
        'pending': ['A production/runtime integration and fixed-source GitHub synchronization',
                    'A GPU admission after synchronized frozen deployment']}
    write_new_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('small', 'bulk', 'train', 'valid', 'work-root', 'guard', 'train-csv-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
        if name in ('small', 'bulk', 'train', 'valid'):
            parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if not Path(args.output).exists():
            write_new_json(args.output, {'schema_version': 1, 'passed': False,
                'phase': 'migration-finalization', 'execution': cpu_context(),
                'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()

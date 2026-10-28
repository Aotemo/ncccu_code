"""Bind current, content-verified source objects to a new verified CPU stage.

The filesystem can retain identical stat across same-length writes. This format
requires fresh_pod_sha256 evidence for the eleven selected assets, without
replaying source semantics or modifying historical receipts.
"""
import argparse
import copy
from pathlib import Path
import re

from source_chain import (FILE_IDENTITY_FIELDS, cpu_context, file_identity,
                          read_pinned_receipt, require, require_complete_identity, sha256,
                          verify_current_content, write_new_json)


EXECUTION_FIELDS = ('namespace', 'pod', 'uid', 'container', 'python')
REFERENCE_COMMIT = 'cb939c8e5078a9c74a26158a78c4aa033009042a'
PUBLIC_FILE_KEYS = {'source-manifest.json', 'model-api.json', 'config.json', 'README.md',
                    'LICENSE-apache-2.0.txt', 'pytorch_model.bin', 'download-receipt.json',
                    'NOTICE-upstream-MIT.txt', 'converted'}


def execution_identity(context):
    require(all(isinstance(context.get(key), str) and context[key] for key in EXECUTION_FIELDS),
            'complete verified execution identity required')
    return {key: context[key] for key in EXECUTION_FIELDS}


def allocation_identity(context):
    value = context['cpu_allocation']
    return {key: value[key] for key in ('path', 'sha256', 'executor_id', 'threads')} | {'mask': value['channel_mask']}


def validate_allocation_identity(value, expected):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256', 'executor_id', 'threads', 'mask'},
            'stage CPU allocation fields differ')
    mask = value['mask']
    require(type(value['threads']) is int and value['threads'] > 0
            and isinstance(mask, list) and bool(mask)
            and all(type(cpu) is int and cpu >= 0 for cpu in mask)
            and len(mask) == len(set(mask)) == value['threads']
            and value == expected, 'stage binding current CPU allocation differs')


def historical_assets(chain):
    require(chain.get('version') == 'tiny22k-source-chain-v1' and chain.get('passed') is True,
            'passed pinned historical source chain required')
    require(set(chain.get('files', {})) == PUBLIC_FILE_KEYS, 'historical source chain must cover all 11 assets')
    return {'provenance': chain['provenance'], 'historical_converter': chain['historical_converter'], **chain['files']}


def validate_reference(reference):
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'}
            and isinstance(reference['path'], str) and Path(reference['path']).is_absolute()
            and isinstance(reference['sha256'], str) and re.fullmatch('[0-9a-f]{64}', reference['sha256']),
            'explicit absolute source receipt path and exact SHA required')


def verify_snapshot(snapshot):
    require(snapshot.get('git_commit') == REFERENCE_COMMIT and bool(snapshot.get('files')),
            'fixed reference snapshot identity differs')
    root = Path(snapshot['root'])
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'absolute fixed snapshot root required')
    for name, expected_sha in snapshot['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'snapshot relative path escapes root')
        path = root / relative
        file_identity(path)
        require(path.resolve().is_relative_to(root.resolve()) and sha256(path) == expected_sha,
                'fixed runtime snapshot changed: ' + name)


def verify_asset_content(record, path=None):
    # Keep complete historical identity separate from independent current proof.
    require_complete_identity(record)
    return verify_current_content(path or record['path'], record['sha256'])


def build_stage_binding(chain, reference, context):
    validate_reference(reference)
    entries, bytes_hashed = [], 0
    for key, record in historical_assets(chain).items():
        current = verify_asset_content(record)
        bytes_hashed += current['bytes_hashed']
        entries.append({'source_key': key, 'current_identity': {field: current[field] for field in FILE_IDENTITY_FIELDS},
                        'sha256': current['sha256'], 'evidence': {'kind': 'fresh_pod_sha256',
                                     'source_receipt_sha256': reference['sha256']}})
    verify_snapshot(chain['runtime_snapshot'])
    return {'schema_version': 1, 'stage': 'pretraining-strict', 'passed': True,
            'scope': 'source_asset_stage_binding_only_not_model_acceptance',
            'source_receipt': dict(reference), 'execution': execution_identity(context),
            'cpu_allocation': allocation_identity(context), 'assets': entries,
            'runtime_snapshot': copy.deepcopy(chain['runtime_snapshot']),
            'content_verification': {'kind': 'fresh_pod_sha256', 'asset_count': len(entries),
                                     'bytes_hashed': bytes_hashed, 'weight_blob_hashes_recomputed': True}}


def validate_execution_binding(chain, reference, context, stage=None):
    validate_reference(reference)
    assets = historical_assets(chain)
    execution = execution_identity(context)
    if stage is None:
        require(execution_identity(chain['execution']) == execution,
                'historical execution differs; explicit pinned stage binding required')
        current_assets = {key: verify_asset_content(record) for key, record in assets.items()}
        verify_snapshot(chain['runtime_snapshot'])
        return current_assets
    require(type(stage.get('schema_version')) is int and stage['schema_version'] == 1
            and stage.get('stage') == 'pretraining-strict'
            and stage.get('passed', True) is True, 'stage binding schema/phase differs')
    require(stage.get('source_receipt') == reference, 'stage binding historical source receipt pin differs')
    require(stage.get('execution') == execution, 'stage binding current execution identity differs')
    validate_allocation_identity(stage.get('cpu_allocation'), allocation_identity(context))
    rows = stage.get('assets')
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), 'stage asset list required')
    keys = [row.get('source_key') for row in rows]
    require(all(isinstance(key, str) for key in keys) and len(keys) == len(set(keys))
            and set(keys) == set(assets), 'stage binding has missing, unknown or duplicate assets')
    current_assets = {}
    for row in rows:
        record = assets[row['source_key']]
        require_complete_identity(record)
        current = row.get('current_identity')
        require_complete_identity(current)
        require(set(current) == set(FILE_IDENTITY_FIELDS), 'stage asset identity fields differ')
        require(row.get('sha256') == record['sha256'] and row.get('evidence') ==
                {'kind': 'fresh_pod_sha256', 'source_receipt_sha256': reference['sha256']},
                'stage asset SHA or historical evidence differs')
        require(current == file_identity(current['path']), 'stage asset current identity changed')
        proof = verify_asset_content(record, current['path'])
        require(current == {key: proof[key] for key in FILE_IDENTITY_FIELDS}, 'stage asset current identity changed during verification')
        current_assets[row['source_key']] = proof
    require(stage.get('runtime_snapshot') == chain['runtime_snapshot'], 'stage reference snapshot binding differs')
    verify_snapshot(stage['runtime_snapshot'])
    return current_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'source-receipt-sha', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    context = cpu_context()
    reference = {'path': args.source_receipt, 'sha256': args.source_receipt_sha}
    try:
        chain = read_pinned_receipt(reference['path'], reference['sha256'])
        value = build_stage_binding(chain, reference, context)
    except Exception as error:
        write_new_json(args.output, {'passed': False, 'execution': context, 'source_receipt': reference,
                                     'scope': 'source_asset_stage_binding_only_not_model_acceptance',
                                     'error_type': type(error).__name__, 'error': str(error)})
        raise
    write_new_json(args.output, value)


if __name__ == '__main__':
    main()

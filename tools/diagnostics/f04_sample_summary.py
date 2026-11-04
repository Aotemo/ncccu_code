"""Summarize the preregistered first-64 paired F04 logits without exporting rows."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time


def validate_chunk_plan(chunks, round_id, inventory_sha, export_sha, git_commit):
    ordered = sorted(chunks, key=lambda row: row['offset'])
    next_offset = 0
    for row in ordered:
        visibility = row.get('execution', {}).get('gpu_visibility', {})
        if (row.get('status') != 'F04_paired_internal_dev_chunk_completed_no_fusion'
                or row.get('round_id') != round_id or row.get('inventory_sha256') != inventory_sha
                or row.get('export_sha256') != export_sha or row.get('git_commit') != git_commit
                or row.get('TTA') is not False or row.get('official_read') is not False
                or row.get('GPU_initialized') is not False or row.get('execution', {}).get('scope') != 'compute_pod_cpu'
                or any(visibility.get(name) != '' for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
                or type(row.get('count')) is not int or row['count'] < 1 or row['offset'] != next_offset):
            raise ValueError('chunk provenance, device or fixed-prefix coverage differs')
        next_offset += row['count']
    if next_offset != 64:
        raise ValueError('fixed first 64 dev examples must appear exactly once')
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'remote-metadata', 'inventory', 'preflight', 'result', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('git-commit', 'inventory-sha256', 'old-inventory-sha256', 'preflight-sha256',
                 'result-sha256', 'round-id', 'chunk-git-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--chunk', action='append', nargs=2, required=True, metavar=('DIRECTORY', 'RECEIPT_SHA256'))
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('published source root differs')
    sys.path[:0] = [str(source / 'runtime/bundle'), str(source / 'tools/diagnostics'),
                    str(source / 'tools/pretraining')]
    from source_chain import cpu_context
    from f04_export_member import verify_published
    from f04_bounded_fusion import compare_fixed_sample, sha, verify_blob, write
    import paired_evidence as pe
    context = cpu_context()
    git_files = verify_published(source, args.remote_metadata, args.git_commit)
    for name in ('tools/diagnostics/f04_bounded_fusion.py', 'tools/diagnostics/f04_sample_summary.py',
                 'tests/test_f04_sample_summary.py'):
        git_files[name] = verify_blob(source, args.remote_metadata, name)
    if (sha(args.inventory) != args.inventory_sha256 or sha(args.preflight) != args.preflight_sha256
            or sha(args.result) != args.result_sha256):
        raise ValueError('sealed input SHA differs')
    inventory = json.loads(args.inventory.read_text())
    preflight = json.loads(args.preflight.read_text())
    result = json.loads(args.result.read_text())
    if (result.get('status') != 'F04_fp32_dev_predictions_with_fp16_storage_budget'
            or result.get('preflight_sha256') != args.preflight_sha256
            or result.get('inventory_sha256') != args.inventory_sha256
            or preflight.get('inventory_sha256') != args.inventory_sha256
            or preflight.get('old_inventory_sha256') != args.old_inventory_sha256
            or args.round_id not in result['fp16_size_fp32_predictions']['members']):
        raise ValueError('member or preregistered main-result identity differs')
    export_sha = preflight['exports'][args.round_id]['sha256']
    reference = pe._reference(inventory['manifest'])
    work = Path(inventory['manifest']['path']).parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(work / 'technical-prep') or output.exists():
        raise ValueError('fresh technical-prep output required')
    chunks = []
    for directory, expected_sha in args.chunk:
        path = Path(directory) / 'CHUNK.json'
        if sha(path) != expected_sha:
            raise ValueError('paired chunk receipt SHA differs')
        row = json.loads(path.read_text())
        row['_receipt'] = {'path': str(path), 'sha256': expected_sha}
        chunks.append(row)
    ordered = validate_chunk_plan(chunks, args.round_id, args.old_inventory_sha256, export_sha, args.chunk_git_commit)
    actual, refs = [], []
    for row in ordered:
        prediction = row['prediction']; path = Path(prediction['path']).resolve(strict=True)
        if not path.is_relative_to(work / 'technical-prep') or sha(path) != prediction['sha256']:
            raise ValueError('private paired logits differ')
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['name', 'label', 'fp32', 'fp16']:
                raise ValueError('paired prediction columns differ')
            items = list(reader)
        if len(items) != row['count']:
            raise ValueError('paired prediction count differs')
        for item, wanted in zip(items, reference['dev'][row['offset']:row['offset'] + row['count']]):
            if item['name'] != wanted['name'] or item['label'] != str(wanted['label']):
                raise ValueError('paired prediction is not the fixed manifest prefix')
            actual.append((float(item['fp32']), float(item['fp16'])))
        refs.append(row['_receipt'])
    scale = result['fp16_size_fp32_predictions']['normalization'][args.round_id]['std']
    stats = compare_fixed_sample([row[0] for row in actual], [row[1] for row in actual], scale)
    summary = {'status': 'F04_fixed_first64_weight_storage_comparison', 'at': time.time(),
               'round_id': args.round_id, 'main_result_sha256': args.result_sha256,
               'inventory_sha256': args.inventory_sha256, 'chunks': refs, 'comparison': stats,
               'git_commit': args.git_commit, 'git_files': git_files, 'execution': context,
               'TTA': False, 'official_read': False, 'GPU_initialized': False,
               'limitations': ['First 64 examples only; full-dev fp16 forward remains required before final freeze.']}
    write(output, summary)
    print(json.dumps({'status': summary['status'], 'round_id': args.round_id,
                      'comparison': stats, 'receipt_sha256': sha(output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()

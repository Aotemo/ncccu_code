"""F09: derive a construction export for a model soup.

The soup export keeps the training-side `experiment91` pointer in its job, which
the deployed inference bundle cannot validate. The F04 latency exports already
use the established convention of dropping that pointer for construction only:
the weight file stays byte-identical (same SHA-256), and only the metadata job
pointer is cleared. Nothing else is rewritten.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.source / 'model.json').read_text())
    job = metadata.get('job') or {}
    if job.get('experiment91') is None:
        raise SystemExit('source job has no experiment91 pointer; nothing to derive')
    original_weight_sha = sha256(args.source / 'model.pt')
    if original_weight_sha != metadata['model_sha256']:
        raise SystemExit('source weight SHA mismatch')

    args.output.mkdir(parents=True, exist_ok=True)
    weight_target = args.output / 'model.pt'
    if weight_target.exists():
        weight_target.unlink()
    try:
        weight_target.hardlink_to(args.source / 'model.pt')
    except OSError:
        shutil.copyfile(args.source / 'model.pt', weight_target)
    derived_sha = sha256(weight_target)
    if derived_sha != original_weight_sha:
        raise SystemExit('derived weight SHA differs from source')

    derived = dict(metadata)
    derived_job = dict(job)
    derived_job['experiment91'] = None
    derived['job'] = derived_job
    derived['model_sha256'] = derived_sha
    (args.output / 'model.json').write_text(json.dumps(derived, indent=1) + '\n')
    receipt = {
        'schema_version': 1,
        'kind': 'construction_export_without_experiment91',
        'source': str(args.source),
        'source_model_sha256': original_weight_sha,
        'derived_model_sha256': derived_sha,
        'weights_identical': original_weight_sha == derived_sha,
        'cleared_fields': ['job.experiment91'],
        'reason': 'deployed inference bundle cannot validate the training-side round pointer',
    }
    (args.output / 'STANDALONE.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({'output': str(args.output), 'model_sha256': derived_sha,
                      'weights_identical': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()

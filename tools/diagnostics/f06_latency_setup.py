"""Build benchmark-only member exports (weights unchanged) and the latency manifest.

Only the historical 9:1 experiment91 pointer is removed from the inference job so
that the frozen loader can construct the identical architecture without a live
site binding. Weights, age, preprocessing and inference runtime are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True,
                        help='id=variant=/path/to/export')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    members = []
    for spec in args.source:
        identifier, variant, source = spec.split('=', 2)
        source = Path(source)
        metadata = json.loads((source / 'model.json').read_text())
        job = metadata['job']
        stripped = 'experiment91' in job
        job.pop('experiment91', None)
        target = root / identifier
        target.mkdir(exist_ok=False)
        weights = source / 'model.pt'
        link = target / 'model.pt'
        try:
            os.link(weights, link)
        except OSError:
            shutil.copy2(weights, link)
        metadata['job'] = job
        if stripped:
            metadata['benchmark_note'] = 'experiment91 pointer removed for construction only'
        (target / 'model.json').write_text(json.dumps(metadata, indent=1, sort_keys=True) + '\n')
        members.append({'id': identifier, 'variant': variant,
                        'model': {'path': str(target), 'sha256': sha(link)}})
    write_new(args.manifest, json.dumps({'format': 'f06-latency-members-v1', 'members': members},
                                        indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())

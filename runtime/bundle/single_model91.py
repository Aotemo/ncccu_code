"""Single-model controller. Formal owner must hold the original instance lock."""
import argparse
import json
from pathlib import Path
from retrain.single91 import freeze_training, run_official_once
from retrain.persistence import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--training-folder', required=True)
    freeze.add_argument('--output', required=True)
    official = sub.add_parser('official-once')
    official.add_argument('--freeze', required=True)
    official.add_argument('--output-root', required=True)
    official.add_argument('--data-root', required=True)
    official.add_argument('--eval-manifest', required=True)
    official.add_argument('--eval-manifest-sha256', required=True)
    official.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.operation == 'freeze':
        value = freeze_training(args.training_folder, args.output)
        print(json.dumps({'freeze_path': str(Path(args.output).resolve()), 'sha256': sha256_file(args.output),
                          'variant': value['variant'], 'selected': value['selected']}), flush=True)
    else:
        print(json.dumps(run_official_once(args.freeze, args.output_root, data_root=args.data_root,
            eval_manifest=args.eval_manifest, eval_manifest_sha256=args.eval_manifest_sha256, device=args.device)), flush=True)


if __name__ == '__main__':
    main()

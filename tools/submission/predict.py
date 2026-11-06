"""F09 submission predict entry: python predict.py <config.json> <pictures_dir> <output.csv>.

Serial batch=1 end-to-end inference producing the same predictions cost.py
verifies against submission.csv. Precision: fp16.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_core import MemberSet, rows_from_directory  # noqa: E402


def run(config_path, pictures_dir, output_csv, device='cuda'):
    config = json.loads(Path(config_path).read_text())
    import torch
    device = torch.device(device, 0)
    torch.cuda.set_device(device)
    members = MemberSet(config, device)
    rows = rows_from_directory(pictures_dir, config.get('ages_csv'))
    dataset = members.dataset(Path(pictures_dir), rows)
    started = time.perf_counter()
    with output_csv and __import__('pathlib').Path(output_csv).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Image_name', 'Probability', 'Pred_Label'])
        for index, row in enumerate(rows):
            probability, _ = members.predict_index(dataset[index])
            writer.writerow([row['name'], repr(float(probability)), int(probability >= 0.5)])
        stream.flush()
    return {'rows': len(rows), 'seconds': time.perf_counter() - started,
            'decode': config.get('decode', 'draft'), 'precision': 'fp16'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('pictures_dir')
    parser.add_argument('output_csv')
    args = parser.parse_args()
    print(run(args.config, args.pictures_dir, args.output_csv))


if __name__ == '__main__':
    main()

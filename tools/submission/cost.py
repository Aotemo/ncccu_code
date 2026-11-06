"""F09 cost.py: competition rule article 7 timing script.

Usage: python cost.py <submission.csv> <pictures_100> <pictures_5>

batch=1 serial; per-image wall time from reading the image file to writing the
final prediction; 5 warm-up images (not scored); all 100 samples repeated 3
rounds (per-image mean of 3 rounds, then mean of 100); predictions are compared
with the final submission.csv. Precision: fp16 (declared).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_core import MemberSet, rows_from_directory  # noqa: E402


def read_submission(path):
    table = {}
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            table[row['Image_name']] = row['Probability']
    return table


def score_round(members, dataset, rows, output_handle, expected):
    per_image = []
    mismatched = 0
    for index, row in enumerate(rows):
        started = time.perf_counter()
        source = Path(dataset.root) / row['relative_path']
        data = source.read_bytes()
        import io
        from PIL import Image
        # decode+preprocess through the same dataset path (bytes are re-opened
        # by the dataset; keep the read inside the timed window)
        del data
        item = dataset[index]
        probability, _ = members.predict_index(item)
        label = int(probability >= 0.5)
        output_handle.write(f"{row['name']},{probability:.6f},{label}\n")
        output_handle.flush()
        elapsed = time.perf_counter() - started
        per_image.append(elapsed)
        if row['name'] in expected:
            delta = abs(float(expected[row['name']]) - probability)
            if delta > 1e-6:
                mismatched += 1
    return per_image, mismatched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission_csv')
    parser.add_argument('pictures_100')
    parser.add_argument('pictures_5')
    parser.add_argument('--config', default=None,
                        help='inference config JSON; default: config.json next to this script')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-sample', action='store_true',
                        help='functional validation only: accept fewer than 100 timed images')
    args = parser.parse_args()
    config_path = args.config or str(Path(__file__).resolve().parent / 'config.json')
    config = json.loads(Path(config_path).read_text())
    import torch
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    members = MemberSet(config, device)

    expected = read_submission(args.submission_csv)
    rows_100 = rows_from_directory(args.pictures_100, config.get('ages_csv'))
    rows_5 = rows_from_directory(args.pictures_5, config.get('ages_csv'))
    if (len(rows_100) != 100 or len(rows_5) != 5) and not args.allow_sample:
        raise SystemExit(f'expected 100 timed and 5 warm-up images, found {len(rows_100)}/{len(rows_5)}')
    dataset = members.dataset(Path(args.pictures_100), rows_100)
    warmup = members.dataset(Path(args.pictures_5), rows_5)

    precision = config.get('precision', 'fp16')
    print(f'[COST] precision={precision} members={len(config["members"])} decode={config.get("decode")}', flush=True)
    import tempfile
    scratch = Path(tempfile.mkdtemp(prefix='ncccu-cost-'))
    print(f'[COST] local scratch {scratch}', flush=True)
    with (scratch / 'warmup.log').open('w') as handle:
        for index in range(len(rows_5)):
            members.predict_index(warmup[index])
            handle.write(f'{rows_5[index]["name"]}\n')

    rounds = []
    mismatched_total = 0
    with (scratch / 'round_predictions.csv').open('w') as handle:
        for round_index in range(3):
            started = time.perf_counter()
            per_image, mismatched = score_round(members, dataset, rows_100, handle, expected)
            mismatched_total += mismatched
            rounds.append(per_image)
            print(f'[ROUND] round={round_index + 1} mean_ms={1000 * sum(per_image) / len(per_image):.2f} '
                  f'wall_s={time.perf_counter() - started:.1f}', flush=True)
    per_image_mean = [sum(values[index] for values in rounds) / len(rounds) for index in range(len(rows_100))]
    mean_ms = 1000 * sum(per_image_mean) / len(per_image_mean)
    verdict = '未超过上限' if mean_ms <= 100.0 else '超过上限'
    print(f'[CONCLUSION] 结论：100 张样本、batch=1 串行、连续 3 轮；单张端到端墙钟时间先按每张 3 轮取均值，'
          f'再对 100 张取平均，最终平均耗时 {mean_ms:.1f} ms，{verdict}；预测结果与 submission.csv 比对 100 张，'
          f'不一致 {mismatched_total} 张；推理精度 {precision}。', flush=True)


if __name__ == '__main__':
    main()

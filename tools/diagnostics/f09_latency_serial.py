"""F09: serial batch=1 end-to-end latency measurement on DCU, fp16.

Measures read->decode->preprocess->per-member forward->fuse->write for several
member configurations under the full and draft decode paths, with 5 warm-up
images and 3 timed rounds over the 100 timing pictures (rule article 7
protocol, run on DCU; T4 conversion reported separately). Measured FLOPs via
torch FlopCounterMode provide the effective-TFLOPS denominator.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'submission'))
from predict_core import MemberSet, rows_from_directory  # noqa: E402


def measure_flops(members, dataset, index):
    """Measured FLOPs (torch FlopCounterMode) for one member at its input size."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    member = members.members[0]
    job = member['job']
    size = int(job['image_size'])
    model_cpu, job_cpu, metadata, origin, device = members.inference._load_model(
        members.members[0]['export'], device='cpu', variant='ema')
    age_on = members.enabled(job_cpu)
    model_cpu.eval()
    images = torch.zeros((1, 3, size, size))
    ages = torch.zeros((1, 2)) if age_on else None
    counter = FlopCounterMode(display=False)
    with counter:
        with torch.inference_mode():
            _ = (model_cpu(images, ages) if ages is not None else model_cpu(images))
    return counter.get_total_flops()


def measure(members, dataset, rows, rounds, output_handle):
    per_stage = {'read_decode_preprocess': [], 'h2d': [], 'forward': [], 'post': [], 'write': []}
    per_member_forward = None
    per_image_total = []
    for _ in range(rounds):
        for index in range(len(rows)):
            started = time.perf_counter()
            item = dataset[index]
            t1 = time.perf_counter()
            probability, stages = members.predict_index(item)
            t2 = time.perf_counter()
            output_handle.write(f"{rows[index]['name']},{probability:.6f},{int(probability >= 0.5)}\n")
            output_handle.flush()
            t3 = time.perf_counter()
            per_stage['read_decode_preprocess'].append(t1 - started)
            per_stage['h2d'].append(stages['h2d'])
            per_stage['forward'].append(stages['forward'])
            per_stage['post'].append(stages['post'])
            per_stage['write'].append(t3 - t2)
            per_image_total.append(t3 - started)
            if per_member_forward is None:
                per_member_forward = stages['per_member_ms']
            else:
                per_member_forward = [a + b for a, b in zip(per_member_forward, stages['per_member_ms'])]
    count = len(per_image_total)
    per_member_forward = [value / count for value in per_member_forward]
    return ( {key: 1000 * sum(values) / len(values) for key, values in per_stage.items()},
             1000 * sum(per_image_total) / count, per_member_forward )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members-config', type=Path, required=True)
    parser.add_argument('--configs', required=True,
                        help="'|'-separated member id groups, e.g. 'R120|R120+R134R1'")
    parser.add_argument('--pictures-100', type=Path, required=True)
    parser.add_argument('--pictures-5', type=Path, required=True)
    parser.add_argument('--decodes', nargs='+', choices=('full', 'draft'), default=('full', 'draft'))
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--skip-flops', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.members_config.read_text())
    import torch
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    groups = [group.split('+') for group in args.configs.split('|')]
    rows_100 = rows_from_directory(args.pictures_100, source.get('ages_csv'))
    if len(rows_100) < 1:
        raise SystemExit('no timing pictures found')
    warmup_rows = rows_from_directory(args.pictures_5, source.get('ages_csv'))
    if len(warmup_rows) != 5:
        raise SystemExit('expected 5 warm-up pictures')
    report = {'protocol': {'rounds': args.rounds, 'warmup_images': 5, 'batch': 1,
                           'precision': 'fp16', 'pictures_100': str(args.pictures_100),
                           'timed_images_actual': len(rows_100),
                           'gpu': 'DCU (pod A)', 't4_ratio': 0.53, 'safety': 1.3},
              'configs': {}}
    all_ids = sorted({mid for group in groups for mid in group})
    measured_flops = None
    if not args.skip_flops:
        probe_config = dict(source)
        probe_config['decode'] = args.decodes[0]
        probe_config['members'] = [entry for entry in source['members'] if entry['id'] == all_ids[0]]
        probe = MemberSet(probe_config, device)
        try:
            measured_flops = measure_flops(probe, None, 0)
            print(f'[FLOPS] measured {measured_flops / 1e9:.2f} GFLOPs/image/member', flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc()
            measured_flops = 8.7 * (512 / 224) ** 2 * 1e9
            print(f'[FLOPS] counter failed ({str(error)[:120]}); fallback analytic '
                  f'{measured_flops / 1e9:.2f} GFLOPs/image/member', flush=True)
        del probe
        torch.cuda.empty_cache()
    for decode in args.decodes:
        for group in groups:
            config = dict(source)
            config['decode'] = decode
            config['members'] = [entry for entry in source['members'] if entry['id'] in group]
            members = MemberSet(config, device)
            warm_dataset = members.dataset(args.pictures_5, warmup_rows)
            for index in range(len(warmup_rows)):
                members.predict_index(warm_dataset[index])
            key = '+'.join(group) + f'@{decode}'
            print('[RUN] ' + key, flush=True)
            dataset = members.dataset(args.pictures_100, rows_100)
            with Path(args.out.with_suffix('.runs.txt')).open('a') as handle:
                per_stage, total_ms, per_member = measure(members, dataset, rows_100,
                                                          args.rounds, handle)
            forward_ms = per_stage['forward']
            entry = {'members': group, 'decode': decode,
                     'stage_ms': {k: round(v, 3) for k, v in per_stage.items()},
                     'per_member_forward_ms': [round(v, 3) for v in per_member],
                     'total_ms': round(total_ms, 3),
                     't4_estimate_ms': round((per_stage['read_decode_preprocess']
                                              + per_stage['h2d'] + per_stage['post']
                                              + forward_ms * 0.53) * 1.3, 3),
                     't4_estimate_pass': None}
            entry['t4_estimate_pass'] = bool(entry['t4_estimate_ms'] < 100.0)
            if measured_flops:
                entry['effective_tflops_batch1'] = round(
                    measured_flops * len(group) / (forward_ms / 1000) / 1e12, 3)
            report['configs'][key] = entry
            print(json.dumps(entry), flush=True)
            del members
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'status': 'f09_latency_complete', 'out': str(args.out)}))


if __name__ == '__main__':
    main()

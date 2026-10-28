"""Dev-only unique checkpoint freeze and irreversible official-once reservation."""
import csv
import json
import math
import os
from pathlib import Path
import time

from . import experiment91 as ex
from .persistence import CheckpointStore, canonical_digest, load_file, sha256_file, atomic_json
from .ema import select_variant, tensor_digest
from .n05_metrics import metrics as exact_metrics


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def select_checkpoint(entries, *, steps_per_cycle=None):
    s = ex.schedule()['steps_per_cycle']
    if steps_per_cycle is not None and steps_per_cycle != s:
        raise ValueError('selector schedule differs from frozen S')
    expected = {(c, v) for c in range(1, 9) for v in ('online', 'ema')}
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError('all 16 internal dev readouts required')
    observed = set()
    for item in entries:
        key = item.get('cycle'), item.get('variant')
        ap, error = item.get('ap'), item.get('error_twice')
        if (key not in expected or key in observed or type(item.get('cycle')) is not int
                or item.get('successful_updates') != item['cycle'] * s
                or type(error) is not int or error < 0 or isinstance(ap, bool)
                or not isinstance(ap, (int, float)) or not math.isfinite(ap) or not 0 <= ap <= 1):
            raise ValueError('dev readout coverage/progress/metric identity differs')
        observed.add(key)
    return min(entries, key=lambda x: (x['error_twice'], -x['ap'], x['successful_updates'], 0 if x['variant'] == 'ema' else 1))


def read_vector(path, expected_rows):
    with Path(path).open(newline='') as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != len(expected_rows):
        raise ValueError('prediction vector row count differs')
    result = []
    for row, expected in zip(raw, expected_rows):
        if row['name'] != expected['name'] or int(row['label']) != expected['label']:
            raise ValueError('prediction names/order/labels differ')
        logit = float(row['logit'])
        if not math.isfinite(logit):
            raise ValueError('prediction logit is not finite')
        result.append({'name': row['name'], 'label': int(row['label']), 'logit': logit})
    return result


def collect_dev_entries(folder):
    folder = Path(folder)
    manifest = ex.metadata()['manifest.json']
    expected_rows = [manifest['rows'][i] for i in manifest['dev_indices']]
    entries = []
    for cycle in range(1, 9):
        for variant in ('online', 'ema'):
            path = folder / 'predictions' / variant / f'dev-cycle-{cycle:04d}.csv'
            vector = read_vector(path, expected_rows)
            sidecar = json.loads(Path(str(path) + '.precision.json').read_text())
            proof = sidecar.get('state_identity', {})
            update = cycle * ex.schedule()['steps_per_cycle']
            tensor_sha = sidecar.get('actual_tensor_sha256')
            if (sidecar.get('variant') != variant or sidecar.get('view') != 'identity'
                    or proof.get('successful_updates') != update or not isinstance(tensor_sha, str) or len(tensor_sha) != 64):
                raise ValueError('internal dev state/variant/tensor proof differs')
            entries.append({'cycle': cycle, 'variant': variant, 'successful_updates': update,
                **exact_metrics(vector), 'prediction_path': str(path.resolve()), 'prediction_sha256': sha256_file(path),
                'precision_path': str(Path(str(path) + '.precision.json').resolve()),
                'precision_sha256': sha256_file(str(path) + '.precision.json'), 'tensor_sha256': tensor_sha})
    select_checkpoint(entries)
    return entries


def freeze_training(folder, destination, *, technical_fixture=False):
    folder, destination = Path(folder), Path(destination)
    complete = json.loads((folder / 'complete.json').read_text())
    if bool(complete.get('cpu_fixture_only')) != technical_fixture:
        raise ValueError('CPU fixture cannot become a production selection')
    if (complete.get('status') != 'completed' or complete.get('successful_updates') != ex.schedule()['total_updates']
            or complete.get('evaluated_cycle') != 8 or complete.get('skipped_updates') != 0):
        raise ValueError('complete unskipped 8S training required before selection')
    entries = collect_dev_entries(folder)
    selected = select_checkpoint(entries)
    store = CheckpointStore(folder / 'checkpoints')
    record = store._index()['protected']['best_' + selected['variant']]
    path = store.directory / record['file']
    payload = store.load('best_' + selected['variant'])
    if bool(payload.get('cpu_fixture_only')) != technical_fixture:
        raise ValueError('checkpoint fixture status differs')
    ex.validate_checkpoint(payload)
    state = select_variant(payload, selected['variant'])
    if (payload['successful_updates'] != selected['successful_updates']
            or tensor_digest(state) != selected['tensor_sha256']
            or sha256_file(path) != record['sha256']):
        raise ValueError('selected checkpoint does not match exact dev readout')
    bundle = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(bundle)): sha256_file(p) for p in sorted([*bundle.glob('*.py'), *(bundle / 'retrain').glob('*.py')])}
    if canonical_digest(sources) != payload['identity']['source_digest']:
        raise ValueError('evaluation bundle differs from trained source')
    bn = {n: t for n, t in state.items() if n.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))}
    if bn:
        raise ValueError('Tiny freeze contains BN running buffers')
    ex.validate_normalization(payload['candidate_normalization'])
    value = {'format': 'single-model-9010-official-freeze-v1', 'experiment91': ex.binding(),
        'cpu_fixture_only': technical_fixture, 'experiment_order': ex.candidate()['experiment_order'],
        'round_id': payload['identity'].get('round_id', folder.parent.name),
        'precision': 'fp32', 'view': 'identity', 'geometry': {'image_size': 512, 'prepared_size': 512},
        'candidate_normalization': payload['candidate_normalization'], 'buffer_sha256': tensor_digest(bn), 'initialization': payload['identity']['initialization'],
        'selection_rule': ['error_twice_ascending', 'AP_descending', 'successful_update_ascending', 'EMA_priority'],
        'dev_entries': entries, 'selected': selected, 'checkpoint': str(path.resolve()), 'blob_sha256': record['sha256'],
        'variant': selected['variant'], 'tensor_sha256': tensor_digest(state), 'job': payload['job'],
        'identity': payload['identity'], 'fresh': payload['fresh'], 'source_files': sources,
        'source_digest': payload['identity']['source_digest'], 'normalization': payload['job']['normalization'],
        'official_evaluations_allowed': 1, 'official_metadata_used_for_selection': False}
    write_exclusive(destination, value)
    destination.chmod(0o444)
    return value


def validate_freeze(path, *, allow_fixture=False):
    value = json.loads(Path(path).read_text())
    if value.get('cpu_fixture_only') and not allow_fixture:
        raise ValueError('CPU fixture is barred from official evaluation')
    if (value.get('format') != 'single-model-9010-official-freeze-v1' or value.get('experiment91') != ex.binding()
            or value.get('precision') != 'fp32' or value.get('view') != 'identity' or value.get('experiment_order') != ex.candidate()['experiment_order']
            or value.get('geometry') != {'image_size': 512, 'prepared_size': 512}
            or value.get('initialization') != value['identity']['initialization']
            or value.get('official_evaluations_allowed') != 1 or value.get('official_metadata_used_for_selection') is not False
            or value.get('selected') != select_checkpoint(value['dev_entries'])
            or value.get('variant') != value['selected']['variant']
            or value.get('tensor_sha256') != value['selected']['tensor_sha256']
            or value.get('normalization') != 'imagenet'):
        raise ValueError('unique single-model freeze differs')
    ex.validate_runtime(value['identity'], value['job'], fresh=value.get('fresh'))
    ex.validate_normalization(value.get('candidate_normalization', {}))
    if value.get('buffer_sha256') != value['candidate_normalization']['buffer_sha256']:
        raise ValueError('Tiny frozen empty-buffer identity differs')
    if (value['source_digest'] != value['identity']['source_digest']
            or canonical_digest(value['source_files']) != value['source_digest']
            or sha256_file(value['checkpoint']) != value['blob_sha256']):
        raise ValueError('frozen source/checkpoint blob differs')
    bundle = Path(__file__).resolve().parents[1]
    actual = {str(p.relative_to(bundle)): sha256_file(p) for p in sorted([*bundle.glob('*.py'), *(bundle / 'retrain').glob('*.py')])}
    if actual != value['source_files']:
        raise ValueError('frozen evaluation code differs')
    for entry in value['dev_entries']:
        if sha256_file(entry['prediction_path']) != entry['prediction_sha256'] or sha256_file(entry['precision_path']) != entry['precision_sha256']:
            raise ValueError('frozen internal dev vector/proof differs')
    return value


def validate_frozen_checkpoint(value, payload, actual):
    ex.validate_checkpoint(payload)
    bn = {n: t for n, t in select_variant(payload, value['variant']).items() if n.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))}
    if (payload['job'] != value['job'] or payload['identity'] != value['identity']
            or payload['successful_updates'] != value['selected']['successful_updates']
            or actual.get('variant') != value['variant'] or actual.get('tensor_sha256') != value['tensor_sha256']
            or tensor_digest(bn) != value['buffer_sha256'] or bn
            or payload['candidate_normalization'] != value['candidate_normalization']):
        raise ValueError('actual loaded variant/source/data/BN differs from freeze')


def load_frozen_model(path, device='cpu', *, allow_fixture=False):
    value = validate_freeze(path, allow_fixture=allow_fixture)
    from .inference import _load_model
    model, job, meta, origin, selected_device = _load_model(value['checkpoint'], device=device, variant=value['variant'])
    if (job != value['job'] or origin.get('identity') != value['identity'] or origin.get('variant') != value['variant']
            or origin.get('successful_updates') != value['selected']['successful_updates']
            or origin.get('tensor_sha256') != value['tensor_sha256'] or tensor_digest(model.state_dict()) != value['tensor_sha256']):
        raise ValueError('actual strict reload variant/tensor differs from frozen single model')
    bn = {n: t for n, t in model.state_dict().items() if n.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))}
    if bn or tensor_digest(bn) != value['buffer_sha256'] or ex.describe_model(model) != value['candidate_normalization']:
        raise ValueError('strict reload Tiny noBN/LN identity differs')
    return model, job, origin, selected_device


def reserve_official(path, output_root, *, allow_fixture=False):
    value = validate_freeze(path, allow_fixture=allow_fixture)
    output_root = Path(output_root)
    record = {'status': 'reserved_do_not_retry', 'freeze_path': str(Path(path).resolve()),
        'freeze_sha256': sha256_file(path), 'variant': value['variant'], 'blob_sha256': value['blob_sha256'],
        'tensor_sha256': value['tensor_sha256'], 'created_at': time.time(), 'output_root': str(output_root.resolve())}
    # Canonical training folder owns the reservation, so changing output directories
    # or copying the freeze cannot permit a second official invocation.
    marker = Path(value['checkpoint']).parent.parent / 'OFFICIAL-ONCE.json'
    write_exclusive(marker, record)
    return value, record


def run_official_once(path, output_root, *, data_root, eval_manifest, eval_manifest_sha256, device='cuda:0'):
    # Reservation is before model construction or any official data read. Any failure
    # keeps this marker; retry/reconciliation requires a separate reviewed decision.
    frozen, reservation = reserve_official(path, output_root)
    output_root = Path(output_root)
    try:
        model, job, origin, target = load_frozen_model(path, device)
        from .data import _read_csv
        from .engine import evaluate
        if sha256_file(eval_manifest) != eval_manifest_sha256:
            raise ValueError('official sealed manifest digest differs')
        sealed = json.loads(Path(eval_manifest).read_text())
        records, csv_sha = _read_csv(data_root, 'valid')
        rows = sealed['rows']
        if (sealed.get('training_eligible') is not False or 'fit_indices' in sealed
                or sealed.get('source_identity', {}).get('valid_csv_sha256') != csv_sha
                or len(rows) != 5849 or len(records) != 5849 or len({r['name'].casefold() for r in rows}) != 5849):
            raise ValueError('complete isolated official5849 identity differs')
        for row, record in zip(rows, records):
            if (any(row[k] != record[k] for k in ('name', 'label', 'age')) or not row['relative_path'].startswith('valid/')
                    or len(row.get('sha256', '')) != 64):
                raise ValueError('official manifest row differs')
        positive = sum(r['label'] for r in rows); negative = len(rows) - positive
        if (positive, negative) != (191, 5658):
            raise ValueError('official class coverage differs from191P/5658N')
        job.update(device=str(target), precision='fp32', workers=0, eval_workers=0)
        output = output_root / 'official-raw.csv'
        native = evaluate(model, rows, list(range(len(rows))), data_root, job, target, output)
        exact = exact_metrics(read_vector(output, rows))
        result = {'status': 'completed', 'reservation': reservation, 'source': origin, 'split': 'official_valid',
            'variant': frozen['variant'], 'view': 'identity', 'precision': 'fp32', 'score': 'raw_logit',
            'rows': len(rows), 'positive': positive, 'negative': negative, 'metrics': {**native, **exact},
            'target_auroc_at_least_0_999': 1000 * exact['error_twice'] <= 2 * positive * negative,
            'output_sha256': sha256_file(output), 'eval_manifest_sha256': eval_manifest_sha256, 'input_csv_sha256': csv_sha}
        write_exclusive(output_root / 'OFFICIAL-RESULT.json', result)
        return result
    except Exception as error:
        write_exclusive(output_root / 'OFFICIAL-FAILURE.json', {'status': 'unknown_do_not_retry', 'error': repr(error), 'reservation': reservation})
        raise

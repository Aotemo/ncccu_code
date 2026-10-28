"""Compute-Pod-only 9:1 metadata freeze; no image, model, cache or official reads."""
import ast
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import socket
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / 'result'
BASE = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2')
SOURCE = BASE / 'auroc998-20261025/coordination-huazhong8-20261025/executor-preparation/assets/dev10277-manifest.json'
SOURCE_SHA = '1b76aa6b24f80179c0df492131832e4152158e48a88b9a3c5376f755f664fa0d'
TRAIN_CSV = Path('/public/home/acfwvv9ejn/SothisAI/dataset/Console/glaucoma_first/main/glaucoma_first/train.csv')
NATIVE = BASE / 'auroc999-20261027/rounds/N06/bundle/retrain'
SEED = 20261024
READS = []


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def read(path, expected=None):
    assert not path.is_symlink() and path.is_file(), str(path)
    raw = path.read_bytes()
    record = dict(path=str(path), bytes=len(raw), sha256=digest(raw))
    READS.append(record)
    if expected is not None:
        assert record['sha256'] == expected, record
    return raw


def write(name, value, compact=False):
    raw = canonical(value) + b'\n' if compact else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with (OUT / name).open('xb') as stream:
        stream.write(raw)
    return dict(path=str(OUT / name), bytes=len(raw), sha256=digest(raw))


def text_file(name, text):
    with (OUT / name).open('x') as stream:
        stream.write(text)


def environment():
    assert sys.executable == '/usr/local/python3.12/bin/python3.12'
    assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
    assert (HERE / 'POD-IDENTITY.txt').read_text().splitlines() == ['eec33bd6-b567-40ef-b866-633e272ea019', 'True']
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
    assert all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    deployment = json.loads((HERE / 'DEPLOYMENT.json').read_bytes())
    for item in deployment['files']:
        assert digest((HERE / item['relative_path']).read_bytes()) == item['sha256'], item
    preflight = json.loads((HERE / 'preflight' / 'PREFLIGHT-RECEIPT.json').read_bytes())
    assert preflight['status'] == 'passed' and preflight['exit_code'] == 0
    result = dict(pod_uid='eec33bd6-b567-40ef-b866-633e272ea019', hostname=socket.gethostname(), python=sys.executable,
                  python_version=sys.version, nice=19, threads=2, hidden_gpu=True,
                  dependencies={k: importlib.metadata.version(k) for k in ('numpy', 'torch', 'Pillow')},
                  gpu_api_called=False, script_sha256=digest(Path(__file__).read_bytes()))
    write('ENVIRONMENT.json', result)
    print(json.dumps(result), flush=True)
    return result


def pure_functions(filename, names, namespace):
    raw = read(NATIVE / filename)
    tree = ast.parse(raw.decode('utf-8'), filename=str(NATIVE / filename))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(NATIVE / filename), 'exec'), namespace)
    text_file('SOURCE-' + filename, raw.decode('utf-8'))
    return namespace


def main():
    import numpy as np
    env = environment()
    raw = read(SOURCE, SOURCE_SHA)
    original = json.loads(raw)
    rows = original['rows']
    identity = original['source_identity']
    assert len(rows) == identity['train_count'] == 51396
    assert digest(canonical(rows)) == identity['train_rows_sha256']
    assert set(original['fit_indices']).isdisjoint(original['dev_indices'])
    assert sorted(original['fit_indices'] + original['dev_indices']) == list(range(len(rows)))
    assert original['split']['include_valid'] is False and not original['audit_indices']
    assert all(type(r['label']) is int and r['label'] in (0, 1) for r in rows)
    assert all(PurePosixPath(r['relative_path']).parts[0] == 'train' and len(PurePosixPath(r['relative_path']).parts) == 2 for r in rows)
    assert len({r['name'].casefold() for r in rows}) == len(rows)
    csv_raw = read(TRAIN_CSV, identity['train_csv_sha256'])
    reader = csv.DictReader(io.StringIO(csv_raw.decode('utf-8-sig')))
    fields = reader.fieldnames
    csv_rows = list(reader)
    assert len(csv_rows) == len(rows)
    assert {'Image_name', 'True_Label', 'Age'}.issubset(fields)
    assert all(r['name'].casefold() == c['Image_name'].casefold() and r['label'] == float(c['True_Label'])
               and r['age'] == (c.get('Age', '') or '') for r, c in zip(rows, csv_rows))
    patient_tokens = ('patient', 'subject', 'person', 'participant', 'caseid', 'case_id', 'patientid', 'patient_id')
    all_keys = sorted(set(fields) | {k for r in rows for k in r})
    possible_patient_fields = [k for k in all_keys if any(t in k.lower() for t in patient_tokens)
                               or k.lower() in ('pid', 'subjectid', 'subject_id', 'individual_id', 'family_id', '双眼', '患者', '病人编号')]
    groups = defaultdict(list)
    sha_groups = defaultdict(set)
    for i, r in enumerate(rows):
        assert isinstance(r['group'], str) and r['group']
        assert len(r['sha256']) == 64 and all(c in '0123456789abcdef' for c in r['sha256'])
        groups[r['group']].append(i)
        sha_groups[r['sha256']].add(r['group'])
    schema = dict(train_csv_fields=fields, manifest_row_fields=sorted({k for r in rows for k in r}),
                  patient_field_candidates=possible_patient_fields, patient_ids_inferred_from_names=False,
                  groups=len(groups), group_size_histogram=dict(sorted(Counter(map(len, groups.values())).items())),
                  mixed_label_groups=sum(len({rows[i]['label'] for i in g}) > 1 for g in groups.values()),
                  multiple_byte_sha_groups=sum(len({rows[i]['sha256'] for i in g}) > 1 for g in groups.values()),
                  identical_byte_sha_in_multiple_groups=sum(len(g) > 1 for g in sha_groups.values()),
                  group_semantics='existing sealed content groups; not documented patient identifiers',
                  patient_leakage_risk='cannot exclude same-patient/bilateral leakage without documented patient IDs')
    write('SCHEMA.json', schema)
    assert not possible_patient_fields, 'Patient/group identifier field requires metadata interpretation before split; preserve schema receipt.'
    assert schema['identical_byte_sha_in_multiple_groups'] == 0
    namespace = pure_functions('data.py', ['_age_bin', '_group_stratum', '_partition_groups', 'build_manifest'],
                               dict(defaultdict=defaultdict, random=random, math=math))
    split = dict(kind='holdout', seed=SEED, valid_fraction=.1, include_valid=False)
    manifest = namespace['build_manifest'](dict(rows=rows, identity=identity), split)
    fit, dev = manifest['fit_indices'], manifest['dev_indices']
    # Exact duplicate/content groups stay intact. The original native algorithm is not modified to force row quotas.
    all_indices = set(range(len(rows)))
    assert len(fit) == len(set(fit)) and len(dev) == len(set(dev))
    assert set(fit).isdisjoint(dev) and set(fit) | set(dev) == all_indices
    assert not ({rows[i]['group'] for i in fit} & {rows[i]['group'] for i in dev})
    assert not ({rows[i]['sha256'] for i in fit} & {rows[i]['sha256'] for i in dev})
    assert manifest['audit_indices'] == []
    counts = {}
    for role, values in [('full_train', list(range(len(rows)))), ('fit', fit), ('dev', dev)]:
        labels = Counter(rows[i]['label'] for i in values)
        assert set(labels) == {0, 1}
        counts[role] = dict(rows=len(values), negative=labels[0], positive=labels[1],
                            groups=len({rows[i]['group'] for i in values}), positive_fraction=labels[1]/len(values))
    age_ns = pure_functions('persistence.py', ['_json_bytes', 'canonical_digest'], dict(json=json, hashlib=hashlib))
    age_ns.update(np=np, copy=copy, math=math, VERSION='age-linear-v1')
    pure_functions('age.py', ['parse_age', 'make_spec', 'validate_spec'], age_ns)
    age = age_ns['make_spec'](manifest)
    age_ns['validate_spec'](age)
    assert age['contract']['fit_indices_sha256'] == digest(canonical(fit))
    assert age['contract']['fit_count'] == counts['fit']['rows']
    missing_reasons = Counter(age_ns['parse_age'](rows[i]['age'])['reason'] for i in fit
                              if age_ns['parse_age'](rows[i]['age'])['missing'])
    negatives = counts['fit']['negative']
    steps = (negatives + 55) // 56
    schedule = dict(schema_version=1, formula='ceil(fit_negative / (effective_batch - positive_per_batch))',
                    fit_negative=negatives, effective_batch=64, positive_per_batch=8, negative_per_batch=56,
                    steps_per_cycle=steps, cycles=8, total_updates=8*steps, stop_after_updates=8*steps,
                    warmup_cycles=2, warmup_updates=2*steps, ema_warmcopy_cycles=2, ema_warmcopy_updates=2*steps,
                    early_stopping_enabled=False, min_cycles=8, evaluation_cycles=list(range(1, 9)),
                    expected_online_ema_dev_vectors=16, derived_only_from_new_fit=True)
    ema = dict(version=f'ema-warmcopy{2*steps}-decay0999-frozenbn-v1', warmcopy_updates=2*steps, decay=.999,
               dtype='torch.float32', floating_buffers='same_as_parameters_except_frozen_bn',
               integer_buffers='copy', frozen_bn_buffers='copy')
    fit_names = {rows[i]['name'].casefold() for i in fit}
    dev_names = {rows[i]['name'].casefold() for i in dev}
    assert not fit_names.intersection(dev_names)
    assert all(type(i) is int and 0 <= i < len(rows) for i in fit + dev)
    overlap = dict(fit_dev_index_intersection=0, fit_dev_name_intersection=len(fit_names & dev_names), fit_dev_group_intersection=0,
                   fit_dev_byte_sha_intersection=0, union_count=len(all_indices), original_train_count=len(rows),
                   official_rows_included=0, official_metadata_or_labels_read=False, source_rows_all_train=True,
                   fit_unique=True, dev_unique=True, index_bounds_valid=True, full_source_covered=True)
    algorithm = dict(name=manifest['split_method'], seed=SEED, native_function='data.build_manifest/_partition_groups',
                     grouping='sealed content group, indivisible', strata=['group label (mixed supported)', 'median age bin', 'median original size bin'],
                     procedure='sorted group keys; Python Random(seed) shuffle within sorted strata; stable descending size; assign maximum remaining row quota, global quota and partition index tie-break',
                     requested_dev_fraction=.1, requested_dev_rows_ceil=(len(rows)+9)//10,
                     actual_dev_rows=len(dev), actual_dev_fraction=len(dev)/len(rows),
                     deviation_from_ceil=len(dev)-(len(rows)+9)//10,
                     exact_row_target_not_forced_across_groups=True, patient_grouping_available=False,
                     raw_image_content_read=False, old_indices_not_used_for_new_allocation=True)
    records = {}
    records['manifest.json'] = write('manifest.json', manifest, compact=True)
    records['fit_indices.json'] = write('fit_indices.json', fit, compact=True)
    records['dev_indices.json'] = write('dev_indices.json', dev, compact=True)
    records['AGE.json'] = write('AGE.json', age)
    records['SCHEDULE.json'] = write('SCHEDULE.json', schedule)
    records['EMA-R50.json'] = write('EMA-R50.json', ema)
    # Explicit row-order identities for transport/inspection; no image bytes are touched.
    role_for = {i: 'fit' for i in fit}
    role_for.update({i: 'dev' for i in dev})
    with (OUT/'ASSIGNMENTS.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_index', 'role', 'name', 'relative_path', 'label', 'group', 'sha256'])
        for i, row in enumerate(rows):
            writer.writerow([i, role_for[i], row['name'], row['relative_path'], row['label'], row['group'], row['sha256']])
    native_sources = {r['path']: {k: r[k] for k in ('sha256', 'bytes')} for r in READS if str(NATIVE) in r['path']}
    contract = dict(schema_version=1, status='FROZEN_METADATA_ONLY_NO_MODEL_STARTED', counts=counts, algorithm=algorithm,
                    isolation=overlap, source_manifest=dict(path=str(SOURCE), sha256=SOURCE_SHA), source_identity=identity,
                    new_manifest=records['manifest.json'], new_manifest_canonical_sha256=digest(canonical(manifest)),
                    fit_indices_canonical_sha256=digest(canonical(fit)), dev_indices_canonical_sha256=digest(canonical(dev)),
                    age=records['AGE.json'], age_contract_sha256=age['contract_sha256'], fit_age_missing_reasons=dict(missing_reasons),
                    schedule=records['SCHEDULE.json'], schedule_canonical_sha256=digest(canonical(schedule)),
                    ema_r50=records['EMA-R50.json'], ema_r50_canonical_sha256=digest(canonical(ema)),
                    ema_integration_requirements=['new rule registration required; no engine modified here',
                       'frozen R50 BN 159 buffers copy at every successful update, including after warmcopy',
                       'do not rely on equality to the old warmcopy1422 rule to choose BN-copy behavior',
                       'non-R50 architecture BN semantics must be declared in its own candidate contract'],
                    cache_reuse=dict(prepared512_root=str(BASE/'shared-assets/prepared512-v2-pillow12.3.0'),
                       evidence='existing N06 ASSETS.json and A adaptation inventory',
                       policy='reuse sealed fulltrain image cache read-only; source rows/paths/hash unchanged',
                       cache_scanned_or_rebuilt=False, new_cache_coverage_claim=False),
                    native_metadata_source_files=native_sources, patient_grouping=schema,
                    downstream='use these exact frozen indices, new age and derived schedule; unique checkpoint selected internally then official5849 once per single model')
    write('CONTRACT.json', contract)
    write('READ-RECEIPTS.json', READS)
    # Round-trip verification of generated new metadata only; no old suite or model is run.
    reloaded = json.loads((OUT/'manifest.json').read_bytes())
    assert reloaded == manifest
    assert json.loads((OUT/'AGE.json').read_bytes()) == age
    assert digest(canonical(reloaded['fit_indices'])) == age['contract']['fit_indices_sha256']
    assert all(role_for[i] == 'fit' for i in reloaded['fit_indices'])
    assert all(role_for[i] == 'dev' for i in reloaded['dev_indices'])
    # Inputs are unchanged at completion, including the historical 8:2 manifest.
    assert digest(SOURCE.read_bytes()) == SOURCE_SHA
    summary = dict(status='SPLIT91_FROZEN', exit_code=0, counts=counts, algorithm=algorithm,
                    manifest_sha256=records['manifest.json']['sha256'], manifest_canonical_sha256=digest(canonical(manifest)),
                    age_contract=age['contract'], age_contract_sha256=age['contract_sha256'], schedule=schedule,
                    ema_r50=ema, patient_leakage='not excluded; no reliable patient ID in checked source metadata',
                    source_sha256=SOURCE_SHA, official_metadata_read=False, image_or_weights_read=False,
                    gpu_api_called=False, old_artifacts_modified=False, server_result=str(OUT))
    write('SHORT-SUMMARY.json', summary)
    text_file('HANDOFF.md', '# 9:1 single-model handoff\n\n'
              + 'Server result: '+str(OUT)+'\n\n'
              + json.dumps(dict(counts=counts, manifest=records['manifest.json'], age_sha=age['contract_sha256'], schedule=schedule, ema=ema), ensure_ascii=False, indent=2)
              + '\n\nUse exact manifest/indices and new fit-only AGE.json. Preserve frozen BN copy under the new EMA rule. No model started; no engine check claimed. Official5849 is excluded from all generation inputs. Patient IDs unavailable; content groups preserved.\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return env


if __name__ == '__main__':
    OUT.mkdir()
    start = time.time()
    code = 1
    try:
        main()
        code = 0
    except Exception as error:
        write('FAILURE.json', dict(status='FAILED_NO_FREEZE', error=repr(error), traceback=traceback.format_exc(), reads=READS))
        traceback.print_exc()
    finally:
        write('EXECUTION.json', dict(exit_code=code, started_at=start, ended_at=time.time(), script=str(Path(__file__).resolve()),
                                    script_sha256=digest(Path(__file__).read_bytes()), environment='compute Pod CPU only'))
        text_file('EXIT-CODE.txt', str(code)+'\n')
        if code == 0:
            try:
                sealed = {p.name: dict(sha256=digest(p.read_bytes()), bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.is_file()}
                freeze = write('FREEZE.json', dict(schema_version=1, status='SEALED', files=sealed, script_sha256=digest(Path(__file__).read_bytes())))
                assert all(digest((OUT/n).read_bytes()) == r['sha256'] and (OUT/n).stat().st_size == r['bytes'] for n, r in sealed.items())
                for p in OUT.iterdir():
                    p.chmod(0o444)
                OUT.chmod(0o555)
                print(json.dumps(dict(freeze=freeze, exit_code=0)), flush=True)
            except Exception:
                traceback.print_exc()
                code = 1
        sys.exit(code)

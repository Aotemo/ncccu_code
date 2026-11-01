"""Single-device training with transactional recovery at effective-batch boundaries.

The sampler cursor counts *attempts*.  The optimizer schedule and training budget
count successful updates.  Keeping these separate is essential when AMP skips a
step, and neither cursor is ever inferred from DataLoader prefetch state.
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import BalancedBatchSampler, ImageDataset, flatten_training_views
from .classification_loss import PAIRWISE_LOSS_KINDS, classification_loss
from .persistence import (CheckpointStore, RunLock, atomic_json, canonical_digest,
                          capture_rng, load_file, restore_rng)


def _autocast(device, precision):
    if precision == 'fp32':
        return contextlib.nullcontext()
    if precision not in {'fp16', 'bf16'}:
        raise ValueError(f'unsupported precision: {precision}')
    return torch.autocast(device_type=torch.device(device).type,
                          dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def _worker_init(_worker_id):
    torch.set_num_threads(1)


def _loader_options(workers, job, *, persistent=False, seed=0):
    # Iterator construction consumes a base seed.  It must not consume the model
    # RNG: resuming creates a fresh iterator at a different point in training.
    options = dict(num_workers=workers, worker_init_fn=_worker_init,
                   generator=torch.Generator().manual_seed(int(seed)),
                   pin_memory=torch.device(job.get('device', 'cpu')).type == 'cuda')
    if workers:
        options.update(persistent_workers=persistent,
                       prefetch_factor=int(job.get('prefetch_factor', 1)))
    return options


def _write_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['name', 'label', 'logit', 'probability'])
            writer.writeheader()
            writer.writerows(predictions)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def prediction_metrics(predictions):
    labels = np.asarray([row['label'] for row in predictions], dtype=np.int64)
    logits = np.asarray([row['logit'] for row in predictions], dtype=np.float64)
    positives = int(labels.sum())
    metrics = dict(count=len(predictions), positive_count=positives,
                   negative_count=len(predictions) - positives,
                   auroc=None, ap=None, pr_auc=None, operating_points={})
    if positives:
        precision, recall, _ = precision_recall_curve(labels, logits)
        metrics['ap'] = float(average_precision_score(labels, logits))
        metrics['pr_auc'] = float(-np.trapz(precision, recall))
    if 0 < positives < len(predictions):
        metrics['auroc'] = float(roc_auc_score(labels, logits))
        fpr, tpr, thresholds = roc_curve(labels, logits, drop_intermediate=False)
        for target in (0.9, 0.95):
            sensitivity_index = int(np.flatnonzero(tpr >= target)[0])
            specificity_index = int(np.flatnonzero(fpr <= 1 - target + 1e-12)[-1])
            for key, index in ((f'sensitivity_{target:.2f}', sensitivity_index),
                               (f'specificity_{target:.2f}', specificity_index)):
                threshold = float(thresholds[index])
                metrics['operating_points'][key] = dict(
                    sensitivity=float(tpr[index]), specificity=float(1 - fpr[index]),
                    logit_threshold=threshold if math.isfinite(threshold) else None)
    return metrics


def evaluate(model, rows, indices, data_root, job, device, output_csv=None):
    """Evaluate every selected example once, restoring mode and all training RNG.

    AUROC and both PR integrals use logits to avoid ranking ties introduced by
    saturated sigmoid probabilities.  A one-class split has undefined AUROC;
    it is returned as JSON null rather than represented as a misleading zero.
    """
    rng = capture_rng()
    was_training = model.training
    predictions = []
    try:
        dataset = ImageDataset(data_root, rows, int(job['image_size']), training=False,
                               normalization=job.get('normalization', 'imagenet'),
                               prepared_size=int(job.get('prepared_size', 0)),age=job.get('age'),
                               input_preprocessing=job.get('input_preprocessing'))
        loader = DataLoader(Subset(dataset, list(indices)),
                            batch_size=int(job.get('physical_batch', 32)), shuffle=False,
                            **_loader_options(int(job.get('eval_workers', 0)), job))
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                from .age import unpack_batch,forward
                images,labels,row_indices,age_features=unpack_batch(batch,job,device)
                with _autocast(device, job.get('precision', 'fp32')):
                    logits = forward(model,images,age_features,job).reshape(-1).float()
                if logits.numel() != labels.numel() or not torch.isfinite(logits).all():
                    raise FloatingPointError('evaluation produced invalid or nonfinite logits')
                probabilities = logits.sigmoid()
                for index, label, logit, probability in zip(row_indices.tolist(), labels.tolist(),
                                                          logits.cpu().tolist(), probabilities.cpu().tolist()):
                    predictions.append(dict(name=rows[index]['name'], label=int(label),
                                            logit=logit, probability=probability))
        metrics = prediction_metrics(predictions)
        if output_csv is not None:
            _write_predictions(output_csv, predictions)
        return metrics
    finally:
        model.train(was_training)
        restore_rng(rng)


class Lion(torch.optim.Optimizer):
    """Fixed FP32 Lion; algorithm reference Google Research (Apache-2.0).

    https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    The existing caller owns global gradient clipping and finite consensus.
    Named role groups bind this audited recipe's sole moment on restoration.
    """
    def __init__(self, params, lr=1e-4, betas=(.9,.99), weight_decay=0.):
        if tuple(betas) != (.9,.99) or not math.isfinite(lr) or lr < 0 or not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('fixed Lion hyperparameters differ')
        super().__init__(params,dict(lr=lr,betas=tuple(betas),weight_decay=weight_decay,optimizer_algorithm='lion10x-v1'))
        names=[];owned=[]
        for group in self.param_groups:
            ps=group['params'];ns=group.get('param_names');shapes=group.get('param_shapes')
            if (not isinstance(ns,list) or len(ns)!=len(ps) or shapes!=[list(p.shape) for p in ps]
                    or group.get('role') not in ('backbone','head','age')
                    or any(p.dtype!=torch.float32 for p in ps)
                    or tuple(group['betas'])!=(.9,.99) or group['optimizer_algorithm']!='lion10x-v1'
                    or any(type(group.get(k)) not in (int,float) or not math.isfinite(group[k]) or group[k]<0
                           for k in ('lr','initial_lr','weight_decay'))):
                raise ValueError('Lion named FP32 role group differs')
            names.extend(ns);owned.extend(id(p) for p in ps)
        if len(names)!=len(set(names)) or len(owned)!=len(set(owned)):
            raise ValueError('Lion parameter ownership repeats')

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            beta1,beta2=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                if p.grad.is_sparse or p.grad.dtype!=torch.float32:
                    raise ValueError('Lion requires dense FP32 gradients')
                state=self.state[p]
                if not state:state['exp_avg']=torch.zeros_like(p)
                p.mul_(1-group['lr']*group['weight_decay'])
                update=state['exp_avg']*beta1+p.grad*(1-beta1)
                p.add_(update.sign_(),alpha=-group['lr'])
                state['exp_avg'].mul_(beta2).add_(p.grad,alpha=1-beta2)
        return loss

    def load_state_dict(self, saved):
        if not isinstance(saved,dict) or set(saved)!={'state','param_groups'}:
            raise ValueError('Lion optimizer state schema differs')
        groups=saved['param_groups'];states=saved['state']
        if not isinstance(groups,list) or len(groups)!=len(self.param_groups) or not isinstance(states,dict):
            raise ValueError('Lion saved group count/state differs')
        cursor=0;bound={}
        for current,group in zip(self.param_groups,groups):
            if set(group)!=set(current) or any(group[k]!=current[k] for k in current if k not in ('params','lr')):
                raise ValueError('Lion saved algorithm/hyperparameters/named groups differ')
            ids=group['params'];parameters=current['params']
            if (ids!=list(range(cursor,cursor+len(parameters)))
                    or type(group['lr']) not in (int,float) or not math.isfinite(group['lr']) or group['lr']<0):
                raise ValueError('Lion saved parameter positions/LR differ')
            bound.update(zip(ids,parameters));cursor+=len(parameters)
        if not set(states)<=set(bound):raise ValueError('Lion state has an unbound parameter')
        for position,state in states.items():
            moment=state.get('exp_avg') if isinstance(state,dict) else None
            if (not isinstance(state,dict) or set(state)!={'exp_avg'} or not isinstance(moment,torch.Tensor)
                    or moment.shape!=bound[position].shape or moment.dtype!=torch.float32
                    or not bool(torch.isfinite(moment).all())):
                raise ValueError('Lion single-moment state differs')
        return super().load_state_dict(saved)


def _optimizer(model, job):
    from .attention_pool import validate_model as validate_attention_model
    from .stage3_residual import validate_model as validate_stage3_model
    validate_attention_model(model, job)
    validate_stage3_model(model, job)
    # Both supported architectures name their classification head 'classifier'.
    # Bias and normalization vectors receive no decay, including LayerNorm.
    head_ids=None
    if job.get('head_lr_scope','classifier_legacy')=='last_linear':
        linear=[module for name,module in model.named_modules()
                if 'classifier' in name.split('.') and isinstance(module,nn.Linear)]
        if not linear:raise ValueError('last_linear scope requires a classifier Linear')
        head_ids={id(p) for p in linear[-1].parameters(recurse=False)}
    buckets = {}
    age_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == 'age_weight':
            from .age import enabled
            if not enabled(job) or parameter.numel() != 2 or parameter.dtype != torch.float32:
                raise ValueError('age optimizer registration differs')
            age_parameters.append(parameter)
            continue
        head = id(parameter) in head_ids if head_ids is not None else 'classifier' in name.split('.')
        if name in ('attention_pool_weight', 'stage3_residual_weight'):
            head = True
        decay = parameter.ndim > 1 and not name.endswith('.bias')
        key = (head, decay)
        buckets.setdefault(key, []).append(parameter)
    groups = []
    for (head, decay), parameters in sorted(buckets.items()):
        lr = float(job['head_lr' if head else 'backbone_lr'])
        groups.append(dict(params=parameters, lr=lr, initial_lr=lr,
                           weight_decay=float(job['weight_decay']) if decay else 0.0,
                           role='head' if head else 'backbone'))
    from .age import enabled
    if enabled(job):
        if len(age_parameters) != 1:
            raise ValueError('age optimizer requires the two explicit residual scalars')
        if job.get('optimizer_policy') is not None:
            from .age import validate_spec, LION_VERSION
            spec=validate_spec(job['age'])
            if spec['contract']['version']!=LION_VERSION:
                raise ValueError('Lion requires the effective age optimization contract')
            lr=spec['contract']['lr'];wd=spec['contract']['weight_decay']
        else:lr,wd=1e-3,1e-4
        groups.append(dict(params=age_parameters, lr=lr, initial_lr=lr,weight_decay=wd, role='age'))
    if not groups:
        raise ValueError('model has no trainable parameters')
    if job.get('layerwise_lr_policy') is not None:
        from .layerwise_lr import build_layerwise_groups
        groups = build_layerwise_groups(model, groups, job['layerwise_lr_policy'])
    if job.get('optimizer_policy') is not None:
        from .config import LION_RULE
        if job['optimizer_policy']!=LION_RULE:raise ValueError('fixed Lion policy differs')
        names={id(p):name for name,p in model.named_parameters()}
        for group in groups:
            group.update(param_names=[names[id(p)] for p in group['params']],
                         param_shapes=[list(p.shape) for p in group['params']])
        optimizer=Lion(groups)
    else:optimizer = torch.optim.AdamW(groups)
    if job.get('low_rank_policy') is not None:
        from .low_rank import bind_optimizer
        bind_optimizer(model, optimizer)
    if job.get('gradient_centralization_policy') is not None:
        from .config import GRADIENT_CENTRALIZATION_RULE
        if job['gradient_centralization_policy'] != GRADIENT_CENTRALIZATION_RULE:
            raise ValueError('raw gradient centralization policy differs')
        targets = [(name + '.weight', module._parameters['weight'])
                   for name, module in model.named_modules()
                   if isinstance(module, (nn.Conv2d, nn.Linear))]
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        if (not targets or len({id(p) for _, p in targets}) != len(targets)
                or any(id(p) not in owned or not p.requires_grad or p.dtype != torch.float32 for _, p in targets)):
            raise ValueError('raw GC direct weight ownership differs')
        optimizer._gradient_centralization_targets = targets
    return optimizer


def _validate_layerwise_optimizer_state(model, optimizer, saved, policy):
    """Bind saved optimizer positions to the exact named layerwise groups."""
    if policy is None:
        return None
    from .layerwise_lr import group_identity
    expected = group_identity(model, optimizer.param_groups, policy)
    stored = saved.get('param_groups')
    if not isinstance(stored, list) or len(stored) != len(expected):
        raise ValueError('saved layerwise optimizer group count differs')
    keys = tuple(expected[0])
    positions = []
    for row, group in zip(expected, stored):
        if any(group.get(key) != row[key] for key in keys):
            raise ValueError('saved layerwise optimizer group identity differs')
        ids = group.get('params')
        if not isinstance(ids, list) or len(ids) != len(row['param_names']):
            raise ValueError('saved layerwise optimizer parameter positions differ')
        positions.extend(ids)
    if len(positions) != len(set(positions)):
        raise ValueError('saved layerwise optimizer parameter positions repeat')
    if not set(saved.get('state', {})).issubset(positions):
        raise ValueError('saved layerwise Adam state has an unbound parameter position')
    return expected


def _clip_gradients(parameters, role_groups, any_bad=None, *, centralization_targets=None):
    """Return finite status, preclip norms, coefficients; agree before role mutation."""
    if centralization_targets is not None:
        for _, parameter in centralization_targets:
            if parameter.grad is not None:
                parameter.grad.sub_(parameter.grad.mean(dim=tuple(range(1, parameter.grad.ndim)), keepdim=True))
    if role_groups is None:
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        local_bad = not bool(torch.isfinite(norm))
        bad = any_bad(local_bad) if any_bad is not None else local_bad
        return not bad, None, None
    from .gradient_clip import measure_role_norms, apply_role_clip_
    norms = measure_role_norms(role_groups)
    local_bad = not bool(torch.stack([torch.isfinite(value) for value in norms.values()]).all())
    bad = any_bad(local_bad) if any_bad is not None else local_bad
    if bad:
        return False, norms, None
    return True, norms, apply_role_clip_(role_groups, norms)


def _lr_factor(successful_updates, total_updates, warmup_updates):
    # The LR of the next successful update is a pure function of committed
    # progress. Overflow attempts therefore retry the same LR.
    if warmup_updates and successful_updates < warmup_updates:
        return (successful_updates + 1) / warmup_updates
    remaining = max(1, total_updates - warmup_updates - 1)
    progress = min(1.0, max(0.0, (successful_updates - warmup_updates) / remaining))
    return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2


def _apply_lr(optimizer, successful_updates, total_updates, warmup_updates):
    factor = _lr_factor(successful_updates, total_updates, warmup_updates)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * factor


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _seed(seed, device):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    # torch.manual_seed also seeds every visible CUDA device; this worker owns
    # only its selected accelerator, so seed the CPU generator explicitly.
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _validate_payload(payload, identity, manifest_digest, job, mode):
    if payload.get('format_version') != 1:
        raise ValueError('unsupported checkpoint format')
    if payload.get('identity') != identity:
        raise ValueError('resume identity does not match checkpoint')
    if payload.get('manifest_digest') != manifest_digest:
        raise ValueError('resume manifest does not match checkpoint')
    if payload.get('job', {}).get('model') != job['model']:
        raise ValueError('resume architecture does not match checkpoint')
    if payload.get('mode') != mode:
        raise ValueError('resume mode does not match checkpoint; benchmark is never train')


def _validate_stage(payload, manifest_digest, job, identity):
    if payload.get('format_version') != 1:
        raise ValueError('initial_checkpoint must use this tool checkpoint format')
    if payload.get('mode') != 'train':
        raise ValueError('a benchmark checkpoint cannot initialize formal training')
    if payload.get('manifest_digest') != manifest_digest:
        raise ValueError('initial_checkpoint manifest does not match current manifest')
    if payload.get('job', {}).get('model') != job['model']:
        raise ValueError('initial_checkpoint architecture does not match current model')
    source_identity = payload.get('identity', {})
    if (source_identity.get('format') != 'glaucoma-retrain-v1' or
            identity.get('format') != 'glaucoma-retrain-v1' or
            not identity.get('source_digest') or
            source_identity.get('source_digest') != identity.get('source_digest')):
        raise ValueError('initial_checkpoint source identity does not match this tool run')
    source_job = payload['job']
    for field in ('normalization', 'dropout', 'stochastic_depth', 'physical_batch',
                  'effective_batch', 'positive_per_batch', 'precision', 'seed'):
        if source_job.get(field) != job.get(field):
            raise ValueError(f'initial_checkpoint protocol differs: {field}')
    for field in ('kind', 'sha256'):
        if source_job.get('initialization', {}).get(field) != job.get('initialization', {}).get(field):
            raise ValueError(f'initial_checkpoint initialization protocol differs: {field}')
    for key in ('identity', 'model', 'optimizer', 'rng', 'successful_updates', 'attempted_updates'):
        if key not in payload:
            raise ValueError(f'initial_checkpoint is incomplete: missing {key}')


def _memory(device):
    if device.type != 'cuda':
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                free_bytes=free, total_bytes=total)


def run_training(job, manifest, data_root, run_dir, identity, resume=False,
                 mode='train', model_factory=None, stop_event=None):
    """Run one model on one device, returning a durable status summary.

    A caught signal is represented by ``stop_event``.  It is observed only at
    effective-batch and evaluation boundaries.  Unexpected failures retain the
    most recent committed checkpoint, so half-updated parameters never replace
    a known-good recovery point.
    """
    if job.get('optimizer_policy') is not None:
        raise ValueError('Lion requires the audited FP32 eight-rank entry')
    if job.get('gradient_centralization_policy') is not None:
        raise ValueError('raw gradient centralization requires the audited eight-rank entry')
    if job.get('mixstyle_policy') is not None:
        raise ValueError('MixStyle requires the audited eight-rank entry with explicit attempt context')
    if job.get('output_regularization') is not None:
        raise ValueError('raw-logit regularization requires the audited DDP pairwise entry point')
    if (job.get('loss') or {}).get('kind') in PAIRWISE_LOSS_KINDS:
        raise ValueError('global pairwise loss requires the formal eight-rank accumulation-one entry point')
    if job.get('age') is not None:
        raise ValueError('R8 age jobs require the audited DDP training entry point')
    if mode not in {'train', 'benchmark'}:
        raise ValueError('run_training mode must be train or benchmark')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(run_dir / '.writer.lock'):
        try:
            return _run_locked(job, manifest, data_root, run_dir, identity, resume,
                               mode, model_factory, stop_event)
        except BaseException as error:
            atomic_json(run_dir / 'status.json', dict(status='failed', mode=mode,
                                                      error=f'{type(error).__name__}: {error}'))
            raise


def _run_locked(job, manifest, data_root, run_dir, identity, resume, mode, model_factory, stop_event):
    device = torch.device(job['device'])
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('only CPU and CUDA/ROCm devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('requested accelerator is unavailable')
        torch.cuda.set_device(device)
        if job['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 is not supported by the selected accelerator')
        torch.cuda.reset_peak_memory_stats(device)
    if job['precision'] not in {'fp32', 'fp16', 'bf16'}:
        raise ValueError('precision must be fp32, fp16, or bf16')
    physical = int(job['physical_batch'])
    effective = int(job['effective_batch'])
    positive = int(job['positive_per_batch'])
    if physical < 1 or effective < physical or effective % physical or not 0 < positive < effective:
        raise ValueError('invalid physical/effective batch or positive quota')
    accumulation = effective // physical
    negatives = sum(int(manifest['rows'][index]['label']) == 0 for index in manifest['fit_indices'])
    positives = len(manifest['fit_indices']) - negatives
    if not negatives or not positives:
        raise ValueError('training requires positive and negative examples')
    steps_per_cycle = math.ceil(negatives / (effective - positive))
    warmup_benchmark = int(job.get('benchmark_warmup', 50))
    measured_benchmark = int(job.get('benchmark_updates', 200))
    if mode == 'benchmark' and (warmup_benchmark < 50 or measured_benchmark < 200):
        raise ValueError('benchmark requires at least 50 warmup and 200 measured updates')
    total_updates = (warmup_benchmark + measured_benchmark if mode == 'benchmark'
                     else int(job['cycles']) * steps_per_cycle)
    stop_after_updates = (min(total_updates, int(job['stop_after_cycles']) * steps_per_cycle)
                          if mode == 'train' and job.get('stop_after_cycles') else total_updates)
    warmup_updates = int(job['warmup_cycles']) * steps_per_cycle
    if total_updates < 1:
        raise ValueError('training budget must be positive')
    store = CheckpointStore(run_dir / 'checkpoints')
    manifest_digest = canonical_digest(manifest)
    previous = None
    if resume:
        # The store commits an empty index before its first blob. Only that
        # explicit state proves an interrupted initial save; missing/corrupt
        # indexes must fail closed instead of silently resetting a run.
        if not store.is_empty():
            previous = store.load(map_location='cpu')
            _validate_payload(previous, identity, manifest_digest, job, mode)
            schedule = previous.get('schedule', {})
            if (schedule.get('total_updates') != total_updates or
                    schedule.get('stop_after_updates', total_updates) != stop_after_updates or
                    schedule.get('warmup_updates') != warmup_updates or
                    schedule.get('steps_per_cycle') != steps_per_cycle):
                raise ValueError('resume schedule does not match checkpoint')
    elif any((run_dir / 'checkpoints').glob('checkpoint-*.pt')) or (run_dir / 'complete.json').exists():
        raise FileExistsError('run already has training outputs; use resume or a new run directory')
    stage = None
    if previous is None and job.get('initial_checkpoint'):
        stage = load_file(job['initial_checkpoint'], map_location='cpu')
        _validate_stage(stage, manifest_digest, job, identity)
    _seed(int(job['seed']), device)
    if model_factory is None:
        from .models import create_model
        model_factory = create_model
    model = model_factory(job, initialize=previous is None and stage is None).to(device)
    optimizer = _optimizer(model, job)
    from .gradient_clip import build_role_groups
    role_groups = build_role_groups(optimizer, job.get('gradient_clip_policy'))
    from .head_warmup import build_update_groups
    head_warmup_groups = build_update_groups(optimizer, job.get('head_warmup_policy'))
    scaler = torch.amp.GradScaler(device.type, enabled=job['precision'] == 'fp16')
    if previous is not None or stage is not None:
        source = previous if previous is not None else stage
        model.load_state_dict(source['model'])
        if job.get('low_rank_policy') is not None:
            from .low_rank import restore_base_reference, validate_optimizer_state
            restore_base_reference(model, job, source)
            validate_optimizer_state(model, optimizer, source['optimizer'])
        layerwise_identity = _validate_layerwise_optimizer_state(
            model, optimizer, source['optimizer'], job.get('layerwise_lr_policy'))
        optimizer.load_state_dict(source['optimizer'])
        if job.get('low_rank_policy') is not None:
            from .low_rank import optimizer_update_proof
            optimizer_update_proof(model, optimizer, source['successful_updates'])
        if layerwise_identity is not None:
            from .layerwise_lr import group_identity
            if group_identity(model, optimizer.param_groups, job['layerwise_lr_policy']) != layerwise_identity:
                raise ValueError('restored layerwise optimizer group identity differs')
        if previous is not None:
            scaler.load_state_dict(source['scaler'])
        else:
            # Optimizer moments survive a resolution branch, but its learning
            # rates and schedule are restarted identically in both branches.
            for group in optimizer.param_groups:
                group['initial_lr'] = float(job['head_lr' if group['role'] == 'head' else 'backbone_lr'])
                group['weight_decay'] = float(job['weight_decay']) if group['weight_decay'] else 0.0
        restore_rng(source['rng'])
    successful = int(previous['successful_updates']) if previous else 0
    attempted = int(previous['attempted_updates']) if previous else 0
    evaluated_cycle = int(previous['evaluated_cycle']) if previous else 0
    early_stop = dict(previous['early_stop']) if previous else dict(reference_auroc=None, bad_cycles=0, stopped=False)
    best_metrics = previous.get('best_metrics') if previous else None
    last_metrics = previous.get('last_metrics') if previous else None
    audit_metrics = previous.get('audit_metrics') if previous else None
    coverage = set(previous.get('coverage', {}).get('indices', [])) if previous else set()
    timing = dict(previous.get('timing', {})) if previous else {}
    for key in ('training_seconds', 'data_wait_seconds', 'evaluation_seconds', 'checkpoint_seconds',
                'warmup_seconds', 'measured_seconds'):
        timing.setdefault(key, 0.0)
    consecutive_overflows = int(previous.get('consecutive_overflows', 0)) if previous else 0
    last_checkpoint_time = time.monotonic()
    last_checkpoint_attempt = attempted
    wall_started = time.monotonic()
    completed_path = run_dir / 'complete.json'

    def snapshot():
        result = dict(format_version=1, identity=identity, job=dict(job), mode=mode,
                    manifest_digest=manifest_digest, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), rng=capture_rng(), successful_updates=successful,
                    attempted_updates=attempted, cycle=successful // steps_per_cycle,
                    evaluated_cycle=evaluated_cycle, early_stop=dict(early_stop), best_metrics=best_metrics,
                    last_metrics=last_metrics, audit_metrics=audit_metrics,
                    consecutive_overflows=consecutive_overflows,
                    schedule=dict(total_updates=total_updates, warmup_updates=warmup_updates,
                                  stop_after_updates=stop_after_updates,
                                  steps_per_cycle=steps_per_cycle, successful_updates=successful),
                    coverage=dict(indices=sorted(coverage), unique=len(coverage),
                                  positive=sum(int(manifest['rows'][i]['label']) for i in coverage),
                                  negative=sum(not int(manifest['rows'][i]['label']) for i in coverage)),
                    timing=dict(timing))
        if job.get('low_rank_policy') is not None:
            from .models import feature_freeze_proof
            from .low_rank import optimizer_update_proof
            result['parameter_trainability'] = feature_freeze_proof(model, job, optimizer)
            result['low_rank_optimizer'] = optimizer_update_proof(model, optimizer, successful)
        return result

    def save(best=False):
        nonlocal last_checkpoint_time, last_checkpoint_attempt
        started = time.monotonic()
        store.save(snapshot(), best=best)
        timing['checkpoint_seconds'] += time.monotonic() - started
        last_checkpoint_time = time.monotonic()
        last_checkpoint_attempt = attempted

    def status(state, **extra):
        result = dict(status=state, mode=mode, result_kind='benchmark' if mode == 'benchmark' else 'training',
                      successful_updates=successful, attempted_updates=attempted,
                      skipped_updates=attempted - successful, cycle=successful // steps_per_cycle,
                      evaluated_cycle=evaluated_cycle, steps_per_cycle=steps_per_cycle,
                      best_metrics=best_metrics, audit_metrics=audit_metrics,
                      early_stop=dict(early_stop), timing=dict(timing),
                      wall_seconds_this_process=time.monotonic() - wall_started, memory=_memory(device),
                      coverage_unique=len(coverage), **extra)
        if mode == 'benchmark':
            measured_count = max(0, successful - warmup_benchmark)
            result['benchmark'] = dict(warmup_updates=min(successful, warmup_benchmark),
                                       measured_updates=measured_count,
                                       warmup_seconds=timing['warmup_seconds'],
                                       measured_seconds=timing['measured_seconds'],
                                       seconds_per_update=(timing['measured_seconds'] / measured_count
                                                           if measured_count else None))
        atomic_json(run_dir / 'status.json', result)
        return result

    def log(event, **extra):
        record = dict(event=event, successful_updates=successful, attempted_updates=attempted,
                      cycle=successful // steps_per_cycle, **extra)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
            handle.flush()
        print(line, flush=True)

    def stopping():
        return stop_event is not None and stop_event.is_set()

    def interrupt():
        save()
        log('interrupted')
        return status('interrupted')

    _apply_lr(optimizer, successful, total_updates, warmup_updates)
    if previous is None:
        # This is intentionally before DataLoader iteration or the first
        # forward: a freshly launched container can be stopped immediately.
        save()
    if completed_path.exists() and previous is not None:
        # Recompute the completion marker from the validated checkpoint. A
        # stale marker must never hide a fallback to an older checkpoint.
        completed_path.unlink()
    status('running')
    if stopping():
        return interrupt()

    def validate_pending():
        nonlocal evaluated_cycle, best_metrics, last_metrics
        if mode == 'benchmark':
            return
        cycle = successful // steps_per_cycle
        while evaluated_cycle < cycle:
            # There can only be one pending cycle: training never crosses an
            # evaluation boundary until that evaluation has been committed.
            if cycle - evaluated_cycle != 1:
                raise ValueError('checkpoint has skipped unevaluated cycles')
            save()
            improved = False
            if manifest['dev_indices']:
                started = time.monotonic()
                metrics = evaluate(model, manifest['rows'], manifest['dev_indices'], data_root, job, device,
                                   run_dir / 'predictions' / f'dev-cycle-{cycle:04d}.csv')
                timing['evaluation_seconds'] += time.monotonic() - started
                last_metrics = metrics
                score = metrics['auroc']
                if score is None:
                    raise ValueError('development AUROC is undefined; development split needs both classes')
                improved = (best_metrics is None or score > best_metrics['auroc'] or
                            (score == best_metrics['auroc'] and metrics['ap'] > best_metrics['ap']))
                if improved:
                    best_metrics = dict(metrics, cycle=cycle, successful_updates=successful)
                reference = early_stop['reference_auroc']
                if reference is None or score > reference + float(job['min_delta']):
                    early_stop.update(reference_auroc=score, bad_cycles=0)
                else:
                    early_stop['bad_cycles'] += 1
                if cycle >= int(job['min_cycles']) and early_stop['bad_cycles'] >= int(job['patience']):
                    early_stop['stopped'] = True
                log('evaluation', metrics=metrics, best=improved, early_stop=dict(early_stop))
            evaluated_cycle = cycle
            save(best=improved)
            status('running')

    validate_pending()
    if stopping():
        return interrupt()
    train_loader = None
    iterator = None
    try:
        if successful < stop_after_updates and not early_stop['stopped']:
            dataset = ImageDataset(data_root, manifest['rows'], int(job['image_size']), training=True,
                                   normalization=job.get('normalization', 'imagenet'),
                                   prepared_size=int(job.get('prepared_size', 0)),
                                   training_scale=job.get('training_scale'),
                                   training_views=job.get('training_views'),
                                   training_photometric=job.get('training_photometric'),
                                   training_rotation=job.get('training_rotation'),
                                   input_preprocessing=job.get('input_preprocessing'))
            log('training_dataset_policy', training_photometric=dataset.training_photometric,
                training_rotation=dataset.training_rotation)
            # At most 100 consecutive skips are tolerated. The sampler is lazy;
            # this upper bound does not precompute its large attempt stream.
            batch_sampler = BalancedBatchSampler(manifest['rows'], manifest['fit_indices'], physical,
                                                  effective, positive, int(job['seed']), attempted,
                                                  (stop_after_updates - successful + 1) * 101)
            train_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                      **_loader_options(int(job['workers']), job, persistent=True,
                                                        seed=int(job['seed'])))
            iterator = iter(train_loader)
        while successful < stop_after_updates and not early_stop['stopped']:
            if stopping():
                return interrupt()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _synchronize(device)
            update_started = time.monotonic()
            attempt_indices = []
            loss_sum = 0.0
            primary_inputs = view_inputs = 0
            for _ in range(accumulation):
                wait_started = time.monotonic()
                images, labels, indices = next(iterator)
                timing['data_wait_seconds'] += time.monotonic() - wait_started
                attempt_indices.extend(indices.tolist())
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, dtype=torch.float32, non_blocking=True).reshape(-1)
                primary_inputs += images.shape[0]
                images, labels, _ = flatten_training_views(images, labels, None, job.get('training_views'))
                view_inputs += images.shape[0]
                with _autocast(device, job['precision']):
                    logits = model(images).reshape(-1)
                    loss = classification_loss(logits, labels, job.get('loss'))
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss; last checkpoint remains unchanged')
                loss_sum += float(loss.detach()) / accumulation
                scaler.scale(loss / accumulation).backward()
            # Every microbatch in the attempted effective batch has now been
            # consumed. AMP can reject its optimizer update but not unconsume it.
            attempted += 1
            coverage.update(attempt_indices)
            scaler.unscale_(optimizer)
            if head_warmup_groups is None:
                finite_grad, _, _ = _clip_gradients(model.parameters(), role_groups)
            else:
                from .head_warmup import mask_and_clip_gradients_
                finite_grad, _ = mask_and_clip_gradients_(head_warmup_groups, successful)
            if not finite_grad and not scaler.is_enabled():
                raise FloatingPointError('nonfinite gradients without enabled GradScaler')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.is_enabled() and scaler.get_scale() < old_scale
            if not finite_grad and not skipped:
                raise FloatingPointError('nonfinite gradients were not rejected by GradScaler')
            if skipped:
                consecutive_overflows += 1
            else:
                successful += 1
                consecutive_overflows = 0
                finite_parameters = torch.stack([torch.isfinite(parameter).all()
                                                 for parameter in model.parameters()]).all()
                if not bool(finite_parameters):
                    raise FloatingPointError('optimizer produced nonfinite parameters')
                _apply_lr(optimizer, successful, total_updates, warmup_updates)
            _synchronize(device)
            elapsed = time.monotonic() - update_started
            timing['training_seconds'] += elapsed
            if mode == 'benchmark':
                # Charge skipped attempts to the phase in which they occurred;
                # measured wall time must include the cost of overflow handling.
                phase = 'warmup_seconds' if successful - int(not skipped) < warmup_benchmark else 'measured_seconds'
                timing[phase] += elapsed
            if attempted % int(job.get('log_every', 20)) == 0 or skipped:
                log('update', loss=loss_sum, skipped=bool(skipped), scale=scaler.get_scale(),
                    learning_rates=[group['lr'] for group in optimizer.param_groups],
                    unique_samples=len(coverage), primary_inputs=primary_inputs, view_inputs=view_inputs)
                status('running')
            if consecutive_overflows >= 100:
                save()
                raise FloatingPointError('100 consecutive AMP overflow attempts; stopping for diagnosis')
            # Stop before evaluation if requested, preserving its pending state.
            if stopping():
                return interrupt()
            if (not skipped and successful % steps_per_cycle == 0 and mode == 'train'):
                validate_pending()
            checkpoint_updates = int(job.get('checkpoint_updates', 0))
            checkpoint_seconds = float(job.get('checkpoint_seconds', 300))
            if ((checkpoint_updates and attempted - last_checkpoint_attempt >= checkpoint_updates) or
                    time.monotonic() - last_checkpoint_time >= checkpoint_seconds):
                save()
            if stopping():
                return interrupt()
    finally:
        # Release persistent workers before the process is declared complete.
        # DataLoader currently has no public close(); iterator disposal performs
        # the same owned-worker shutdown used by its destructor.
        if iterator is not None and hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
        del iterator, train_loader
    if mode == 'train' and manifest.get('audit_indices') and audit_metrics is None:
        # Freeze selection before looking at outer-fold labels. Temporarily
        # install best weights into the same model to avoid a second GPU copy;
        # always restore final training parameters before saving recovery state.
        save()
        if stopping():
            return interrupt()
        final_model = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        selected = 'best' if best_metrics is not None else 'last'
        started = time.monotonic()
        try:
            frozen = store.load(selected, map_location='cpu')
            model.load_state_dict(frozen['model'])
            del frozen
            audit_metrics = evaluate(model, manifest['rows'], manifest['audit_indices'], data_root, job, device,
                                     run_dir / 'predictions' / 'audit.csv')
        finally:
            model.load_state_dict(final_model)
            del final_model
        timing['evaluation_seconds'] += time.monotonic() - started
        atomic_json(run_dir / 'audit.json', dict(checkpoint=selected, metrics=audit_metrics,
                                                 selection_metrics=best_metrics))
        save()
        log('audit', metrics=audit_metrics, checkpoint=selected)
        if stopping():
            return interrupt()
    save()
    result = status('completed', final_checkpoint='best' if best_metrics is not None else 'last')
    atomic_json(completed_path, result)
    log('completed', mode=mode, early_stopped=early_stop['stopped'])
    return result

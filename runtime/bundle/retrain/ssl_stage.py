"""One registered fit-only representation phase inside the fresh training owner."""
import hashlib
import copy
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ImageDataset, _seed
from .ema import tensor_digest
from .finite import all_floating_finite
from .models import create_model
from .persistence import atomic_json, canonical_digest, sha256_file
from .vicreg import gather_embeddings, vicreg_loss
from . import engine as single


LOCAL_RECIPE = {'version': 'fit-local-photometric-1600-v1', 'updates': 1600,
    'global_batch': 64, 'per_rank_batch': 8, 'training_views': 2,
    'image_size': 512, 'training_scale': [0.85, 1.0],
    'projector': [768, 1024, 256], 'backbone_lr': 1e-4,
    'projector_lr': 1e-3, 'weight_decay': 1e-4,
    'warmup_updates': 160, 'final_lr_factor': 0.01,
    'gradient_clip': 1.0, 'precision': 'fp32',
    'shared_view_geometry': True, 'local_grid': [16, 16],
    'target': 'same_position_independent_photometry', 'loss': 'local_token_vicreg',
    'learning_rng_domain': 'r115-fit-vicreg-1600-v1',
    'sampling_rng_domain': 'r115-fit-vicreg-1600-v1'}


def validate_recipe(recipe, job, manifest, world):
    """The science file fixes this one recipe; labels and dev never enter the phase."""
    expected = {'version': 'r115-fit-vicreg-1600-v1', 'updates': 1600,
        'global_batch': 64, 'per_rank_batch': 8, 'training_views': 2,
        'image_size': 512, 'training_scale': [0.85, 1.0],
        'projector': [768, 1024, 256], 'backbone_lr': 1e-4,
        'projector_lr': 1e-3, 'weight_decay': 1e-4,
        'warmup_updates': 160, 'final_lr_factor': 0.01,
        'gradient_clip': 1.0, 'precision': 'fp32'}
    if recipe.get('version') == 'fit-masked-rgb32-1600-v1':
        from .masked_reconstruction import MASKED_RECIPE
        expected = MASKED_RECIPE
    if recipe.get('version') == 'fit-masked-hog8-1600-v1':
        from .masked_reconstruction import HOG_RECIPE
        expected = HOG_RECIPE
    if recipe.get('version') == LOCAL_RECIPE['version']:
        expected = LOCAL_RECIPE
    if recipe.get('version') == 'fit-ema-assignment-1600-v1':
        from .ema_assignment import EMA_RECIPE
        expected = EMA_RECIPE
    if recipe.get('version') == 'fit-ema-assignment-9600-v1':
        from .ema_assignment import EMA_HORIZON_RECIPE
        expected = EMA_HORIZON_RECIPE
    if recipe != expected or world != 8 or job['model'] != 'convnext_small' or job['image_size'] != 512:
        raise ValueError('registered fixed fit-only SSL recipe/topology differs')
    indices = manifest['fit_indices']
    if (not indices or len(indices) < 64 or len(set(indices)) != len(indices)
            or set(indices) & set(manifest['dev_indices']) or manifest.get('audit_indices')):
        raise ValueError('SSL requires unique fit-only indices and no dev/audit overlap')
    return recipe


def ssl_phase_completed(receipt, recipe):
    """Check completion counters against an already validated registered recipe."""
    return bool(isinstance(receipt, dict) and isinstance(recipe, dict)
        and type(recipe.get('updates')) is int and recipe['updates'] > 0
        and receipt.get('status') == 'completed' and receipt.get('recipe') == recipe
        and receipt.get('successful_updates') == receipt.get('attempted_updates') == recipe['updates']
        and receipt.get('skipped_updates') == receipt.get('dev_evaluations') == 0)


def ssl_head_shape(recipe):
    return list(recipe['projector'])


def ssl_rng_derivation(recipe, seed, world):
    """Consume and report the same seeds; paired spatial targets share a stream."""
    domain = recipe['version']
    learning = recipe.get('learning_rng_domain', domain)
    sampling = recipe.get('sampling_rng_domain', domain)
    return {'domain': domain, 'learning_domain': learning, 'sampling_domain': sampling,
        'projector_seed': _seed(seed, learning, 'projector'),
        'model_seeds': [_seed(seed, learning, 'model', rank) for rank in range(world)],
        'sampler_seed': _seed(seed, sampling, 'sampler'),
        'loader_seeds': [_seed(seed, learning, 'loader', rank) for rank in range(world)]}


class UniformFitSampler:
    """Global shuffled fit stream, full 64-image batches only, rank-contiguous shards."""
    def __init__(self, indices, *, seed, updates, rank, world=8):
        if world != 8 or not 0 <= rank < world or len(indices) < 64:
            raise ValueError('SSL sampler requires eight ranks and a complete global batch')
        self.indices = tuple(indices)
        self.seed, self.updates, self.rank = seed, updates, rank

    def __len__(self):
        return self.updates

    def __iter__(self):
        produced, epoch = 0, 0
        while produced < self.updates:
            shuffled = list(self.indices)
            random.Random(_seed(self.seed, 'epoch', epoch)).shuffle(shuffled)
            for offset in range(0, len(shuffled) - 63, 64):
                if produced >= self.updates:
                    break
                batch = shuffled[offset:offset + 64]
                start = self.rank * 8
                yield [(index, _seed(self.seed, 'sample', produced, position, index))
                    for position, index in enumerate(batch[start:start + 8], start)]
                produced += 1
            epoch += 1


class SSLModel(nn.Module):
    def __init__(self, binary, *, projector_seed):
        super().__init__()
        network = binary.network
        self.features = network.features
        self.avgpool = network.avgpool
        self.final_norm = network.classifier[0]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(projector_seed)
            self.projector = nn.Sequential(nn.Linear(768, 1024), nn.GELU(approximate='none'),
                                           nn.Linear(1024, 256))

    def forward(self, images):
        value = self.avgpool(self.features(images))
        return self.projector(self.final_norm(value).flatten(1))


class LocalSSLModel(SSLModel):
    def forward(self, images):
        value = self.features(images).permute(0, 2, 3, 1)
        value = torch.nn.functional.layer_norm(value, (768,),
            self.final_norm.weight.reshape(768), self.final_norm.bias.reshape(768), self.final_norm.eps)
        return self.projector(value)


def _derived_state(binary):
    return {name: value.detach().cpu().contiguous()
        for name, value in binary.state_dict().items()
        if name.startswith('network.features.') or name.startswith('network.classifier.0.')}


def _non_encoder_state(binary):
    return {name: value for name, value in binary.state_dict().items()
        if not name.startswith('network.features.') and not name.startswith('network.classifier.0.')}


def _factor(update, *, total_updates=1600):
    if total_updates not in (1600, 9600) or not 1 <= update <= total_updates:
        raise ValueError('SSL LR update outside fixed budget')
    if update <= 160:
        return update / 160
    return 0.01 + 0.99 * (1 + math.cos(math.pi * (update - 160) / (total_updates - 160))) / 2


def _optimizer(model):
    groups = {}
    for name, parameter in model.named_parameters():
        role = 'projector' if name.startswith('projector.') else 'backbone'
        decay = parameter.ndim > 1 and not name.endswith('.bias')
        groups.setdefault((role, decay), []).append(parameter)
    result = []
    for (role, decay), parameters in sorted(groups.items()):
        initial_lr = 1e-3 if role == 'projector' else 1e-4
        result.append({'params': parameters, 'lr': initial_lr, 'initial_lr': initial_lr,
            'weight_decay': 1e-4 if decay else 0.0, 'role': role})
    return torch.optim.AdamW(result)


def _wait_barrier(path, recipe, ctx, deadline_at, stop_event):
    while True:
        value = None
        if ctx.primary:
            try:
                value = json.loads(Path(path).read_text())
            except FileNotFoundError:
                pass
        value = ctx.broadcast(value)
        if value is not None:
            if value.get('status') != 'released' or value.get('recipe') != recipe:
                raise ValueError('SSL public-initial barrier differs')
            return
        if time.time() >= deadline_at or (stop_event is not None and stop_event.is_set()):
            raise TimeoutError('SSL public-initial barrier stopped before first update')
        time.sleep(.1)


def run_ssl(job, manifest, data_root, run_dir, recipe, ctx, *, barrier_path, deadline_at,
            identity, site, precision_protocol, stop_event=None):
    validate_recipe(recipe, job, manifest, ctx.world_size)
    updates = recipe['updates']
    from .precision import configure, runtime_proof
    configure(precision_protocol)
    runtime_proof(precision_protocol, device=ctx.device)
    folder = Path(run_dir) / 'ssl-stage'
    existed = folder.exists() if ctx.primary else None
    existed = ctx.broadcast(existed)
    if existed:
        raise FileExistsError('SSL stage is fresh-only and cannot reuse an earlier attempt')
    if ctx.primary:
        folder.mkdir(parents=True)
    ctx.barrier()
    # The public head/age must match the CPU seed before any SSL parameters move.
    domain = recipe['version']
    masked = domain in ('fit-masked-rgb32-1600-v1', 'fit-masked-hog8-1600-v1')
    hog = domain == 'fit-masked-hog8-1600-v1'
    local = domain == LOCAL_RECIPE['version']
    assignment = domain in ('fit-ema-assignment-1600-v1', 'fit-ema-assignment-9600-v1')
    rng = ssl_rng_derivation(recipe, job['seed'], ctx.world_size)
    single._seed(int(job['seed']), ctx.device)
    binary = create_model(job, initialize=True).to(ctx.device)
    # Only the encoder participates; the fresh native head and age stay untouched.
    if masked:
        from .masked_reconstruction import MaskedReconstructionModel, spatial_mask, masked_l1, hog_target, masked_hog_mse
        model = MaskedReconstructionModel(binary, projector_seed=rng['projector_seed'], target='hog' if hog else 'rgb')
    elif assignment:
        from .ema_assignment import (AssignmentSSLModel, assignment_loss, freeze_prototypes,
                                     teacher_initial_proof, update_teacher)
        model = AssignmentSSLModel(binary, projector_seed=rng['projector_seed'])
    else:
        model = (LocalSSLModel if local else SSLModel)(binary, projector_seed=rng['projector_seed'])
    model = model.to(ctx.device)
    if assignment:
        teacher = copy.deepcopy(model).requires_grad_(False).eval()
        center = torch.zeros(1, 65536, device=ctx.device)
        teacher_public = teacher_initial_proof(model, teacher)
        fit_epoch_updates = len(manifest['fit_indices']) // 64
    wrapped = ctx.wrap(model)
    public_digest = tensor_digest(binary.state_dict())
    public_encoder_digest = tensor_digest(_derived_state(binary))
    public_receipt = binary.public_initialization_receipt
    public_non_encoder_digest = tensor_digest(_non_encoder_state(binary))
    publics = ctx.values({'rank': ctx.rank, 'model_sha256': public_digest,
        'encoder_tensor_sha256': public_encoder_digest,
        'public_tensor_sha256': public_receipt['public_tensor_sha256'],
        'strict_public_exact': public_receipt['all_public_tensors_exact'],
        'non_encoder_sha256': public_non_encoder_digest, 'optimizer_state_entries': 0,
        'projector_shape': ssl_head_shape(recipe), **(teacher_public if assignment else {})})
    if (len({item['model_sha256'] for item in publics}) != 1
            or len({item['encoder_tensor_sha256'] for item in publics}) != 1
            or len({item['public_tensor_sha256'] for item in publics}) != 1
            or any(item['strict_public_exact'] is not True for item in publics)):
        raise ValueError('SSL strict public encoder differs between ranks')
    if ctx.primary:
        atomic_json(folder / 'public-initial-ranks.json', publics)
        atomic_json(folder / 'status.json', {'phase': 'fit_only_ssl', 'status': 'waiting_public_initial',
            'successful_updates': 0, 'target_updates': updates, 'recipe': recipe})
    ctx.barrier()
    _wait_barrier(barrier_path, recipe, ctx, deadline_at, stop_event)
    single._seed(rng['model_seeds'][ctx.rank], ctx.device)
    if ctx.primary:
        atomic_json(folder / 'status.json', {'phase': 'fit_only_ssl', 'status': 'running',
            'successful_updates': 0, 'target_updates': updates, 'recipe': recipe})
    optimizer = _optimizer(model)
    dataset = ImageDataset(data_root, manifest['rows'], 512, training=True,
        normalization='imagenet', prepared_size=512, age=None,
        training_scale=[0.85, 1.0], training_views=recipe['training_views'], include_labels=False,
        shared_view_geometry=local)
    sampler = UniformFitSampler(manifest['fit_indices'], seed=rng['sampler_seed'],
        updates=updates, rank=ctx.rank, world=ctx.world_size)
    loader = DataLoader(dataset, batch_sampler=sampler,
        **single._loader_options(job['workers'], job, persistent=True,
                                 seed=rng['loader_seeds'][ctx.rank]))
    iterator = iter(loader)
    try:
        for update in range(1, updates + 1):
            if ctx.any(time.time() >= deadline_at or (stop_event is not None and stop_event.is_set())):
                raise TimeoutError('R115 joint training deadline or owner stop during SSL')
            optimizer.zero_grad(set_to_none=True)
            images, indices = next(iterator)
            expected_shape = (8, 3, 512, 512) if masked else (8, 2, 3, 512, 512)
            if images.shape != expected_shape:
                raise ValueError('SSL loader fit image/view geometry differs')
            images = images.to(ctx.device, non_blocking=True)
            if masked:
                mask = spatial_mask([_seed(job['seed'], rng['learning_domain'], 'mask', update, ctx.rank, int(index))
                                     for index in indices], device=ctx.device)
                prediction = wrapped(images, mask)
                loss = (masked_hog_mse(prediction.float(), hog_target(images), mask[:, :, ::8, ::8]) if hog
                        else masked_l1(prediction.float(), images.float(), mask))
            else:
                dimension = 65536 if assignment else 256
                if assignment:
                    with torch.no_grad():
                        targets = teacher(images.reshape(16, 3, 512, 512)).reshape(8, 2, dimension)
                        t1 = gather_embeddings(targets[:, 0])
                        t2 = gather_embeddings(targets[:, 1])
                embeddings = wrapped(images.reshape(16, 3, 512, 512))
                expected_embedding_shape = (16, 16, 16, 256) if local else (16, dimension)
                if embeddings.shape != expected_embedding_shape:
                    raise ValueError('SSL actual per-view embedding geometry differs')
                embeddings = embeddings.reshape(8, 2, *expected_embedding_shape[1:])
                z1 = gather_embeddings(embeddings[:, 0].reshape(-1, dimension))
                z2 = gather_embeddings(embeddings[:, 1].reshape(-1, dimension))
                expected_global_shape = (16384, 256) if local else (64, dimension)
                if z1.shape != expected_global_shape or z2.shape != expected_global_shape:
                    raise ValueError('SSL actual global two-view embedding geometry differs')
                if assignment:
                    if t1.shape != expected_global_shape or t2.shape != expected_global_shape:
                        raise ValueError('EMA global teacher geometry differs')
                    loss, assignment_stats = assignment_loss(z1.float(), z2.float(), t1.float(), t2.float(), center)
                else:
                    loss = vicreg_loss(z1.float(), z2.float())
            if ctx.any(not bool(torch.isfinite(loss))):
                raise FloatingPointError('nonfinite R115 VICReg loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if ctx.any(not bool(torch.isfinite(norm))):
                raise FloatingPointError('nonfinite R115 VICReg gradient')
            factor = _factor(update, total_updates=updates)
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            if assignment:
                prototype_frozen = freeze_prototypes(model, update=update, fit_epoch_updates=fit_epoch_updates)
            optimizer.step()
            if assignment:
                momentum = update_teacher(teacher, model, center, torch.cat((t1, t2)), update=update, total_updates=updates)
                if ctx.any(not all_floating_finite(teacher.parameters()) or not bool(torch.isfinite(center).all())):
                    raise FloatingPointError('nonfinite EMA teacher or center')
            if ctx.any(not all_floating_finite(model.parameters())
                       or not all_floating_finite(value for state in optimizer.state.values() for value in state.values())):
                raise FloatingPointError('nonfinite R115 VICReg parameter or Adam state')
            if ctx.primary and (update % 25 == 0 or update == 1):
                with (folder / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps({'phase': 'fit_only_ssl', 'successful_updates': update,
                        'primary_inputs': 64, 'view_inputs': 64 if masked else 128,
                        'global_embedding_shape': None if masked else list(z1.shape),
                        'spatial_masked_blocks_per_image': 154 if masked else None,
                        'spatial_tokens_per_primary': 256 if local else None,
                        'finite_loss_and_gradient': True,
                        'loss': float(loss.detach()), 'lr_factor': factor, 'at': time.time(),
                        **({'teacher_entropy': float(assignment_stats['teacher_entropy']),
                            'teacher_mean_entropy': float(assignment_stats['teacher_mean_entropy']),
                            'teacher_momentum': momentum, 'prototype_frozen': prototype_frozen,
                            'prototype_freeze_updates': fit_epoch_updates,
                            'teacher_forward_inputs': 128, 'encoder_view_forwards': 256}
                           if assignment else {})}) + '\n')
                atomic_json(folder / 'status.json', {'phase': 'fit_only_ssl', 'status': 'running',
                    'successful_updates': update, 'target_updates': updates,
                    'last_loss': float(loss.detach()), 'at': time.time()})
    finally:
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
        del iterator, loader
    state = _derived_state(binary)
    digest = tensor_digest(state)
    non_encoder_after = tensor_digest(_non_encoder_state(binary))
    teacher_final = ({'teacher_final_sha256': tensor_digest(teacher.state_dict()),
        'teacher_center_sha256': tensor_digest({'center': center}),
        'teacher_no_gradients': all(parameter.grad is None and not parameter.requires_grad
                                   for parameter in teacher.parameters())} if assignment else {})
    digests = ctx.values({'rank': ctx.rank, 'derived_tensor_sha256': digest,
        'non_encoder_sha256_before': publics[ctx.rank]['non_encoder_sha256'],
        'non_encoder_sha256_after': non_encoder_after,
        'head_age_unchanged': non_encoder_after == publics[ctx.rank]['non_encoder_sha256'], **teacher_final})
    if len({item['derived_tensor_sha256'] for item in digests}) != 1:
        raise ValueError('SSL derived encoder differs between ranks')
    if any(item['head_age_unchanged'] is not True for item in digests):
        raise ValueError('SSL changed the sealed public head or age parameters')
    if assignment and (len({item['teacher_final_sha256'] for item in digests}) != 1
            or len({item['teacher_center_sha256'] for item in digests}) != 1
            or any(item['teacher_no_gradients'] is not True for item in digests)):
        raise ValueError('EMA teacher/center rank agreement or gradient ownership differs')
    receipt = None
    if ctx.primary:
        asset = folder / 'derived-encoder.pt'
        temporary = folder / ('.derived-encoder.' + uuid.uuid4().hex + '.tmp')
        try:
            torch.save(state, temporary)
            os.replace(temporary, asset)
        finally:
            temporary.unlink(missing_ok=True)
        clock = json.loads((Path(site['round_root']) / 'control/round-clock.json').read_text())
        receipt = {'status': 'completed', 'phase': 'fit_only_ssl', 'recipe': recipe,
            'recipe_sha256': canonical_digest(recipe), 'round_id': site['round_id'],
            'attempt_token_sha256': hashlib.sha256(clock['token'].encode()).hexdigest(),
            'git_commit': site['git_commit'],
            'science_sha256': site['science']['configuration']['sha256'],
            'source_digest': identity['source_digest'],
            'manifest_digest': identity['manifest_digest'],
            'public_weight_sha256': job['initialization']['sha256'],
            'world_size': ctx.world_size, 'successful_updates': updates,
            'attempted_updates': updates, 'skipped_updates': 0,
            'fit_indices_only': True, 'dev_evaluations': 0,
            'labels_or_age_used_for_loss': False,
            'public_model_tensor_sha256': public_digest,
            'public_encoder_tensor_sha256': public_encoder_digest,
            'public_tensor_sha256': public_receipt['public_tensor_sha256'],
            'ssl_rank0_non_encoder_sha256': publics[0]['non_encoder_sha256'],
            'derived_tensor_sha256': digest, 'asset_path': str(asset),
            'asset_sha256': sha256_file(asset), 'rank_proofs': digests,
            **({'prototype_freeze_updates': fit_epoch_updates,
                'teacher_projector_center_persisted': False,
                'unique_augmented_views_per_update': 128, 'encoder_view_forwards_per_update': 256}
               if assignment else {}),
            'rng_derivation': rng,
            'finished_at': time.time()}
        atomic_json(folder / 'SSL-RESULT.json', receipt)
        atomic_json(folder / 'status.json', {'phase': 'fit_only_ssl', 'status': 'completed',
            'successful_updates': updates, 'target_updates': updates,
            'derived_tensor_sha256': digest, 'at': time.time()})
    receipt = ctx.broadcast(receipt)
    ctx.barrier()
    del wrapped, model, binary, optimizer, state, images, loss, norm
    if masked:
        del prediction, mask
    else:
        del embeddings, z1, z2
    if assignment:
        del teacher, center, targets, t1, t2, assignment_stats
    if ctx.device.type == 'cuda':
        torch.cuda.empty_cache()
    return receipt


def load_derived_model(job, initialize, receipt):
    if not initialize:
        return create_model(job, initialize=False)
    model = create_model(job, initialize=True)
    public_head = tensor_digest(_non_encoder_state(model))
    asset = Path(receipt['asset_path'])
    if sha256_file(asset) != receipt['asset_sha256']:
        raise ValueError('SSL derived encoder asset bytes changed')
    state = torch.load(asset, map_location='cpu', weights_only=True)
    if tensor_digest(state) != receipt['derived_tensor_sha256']:
        raise ValueError('SSL derived encoder tensor digest changed')
    target = _derived_state(model)
    if state.keys() != target.keys() or any(state[key].shape != target[key].shape or
            state[key].dtype != target[key].dtype or not bool(torch.isfinite(state[key]).all())
            for key in target):
        raise ValueError('SSL derived encoder state/key/FP32/finite identity differs')
    full = model.state_dict()
    full.update(state)
    model.load_state_dict(full, strict=True)
    if tensor_digest(_non_encoder_state(model)) != public_head:
        raise ValueError('R115 public fresh head/age changed during derived encoder load')
    return model

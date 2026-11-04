"""Production registered binary+age CPU admission; no real-data forward."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('explicit_prepare', HERE.parent / 'activation/prepare.py')
build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
CHECKS = []


def layerwise_profile(profile):
    from retrain import experiment91 as ex
    return profile in (ex.LAYERWISE_LR_SMALL_BINDING, ex.LAYERWISE_LR_SMALL_R1_BINDING)


def group_lrs_match(groups, job):
    expected = {'backbone': job['backbone_lr'], 'head': job['head_lr']}
    if job['age']['enabled']:
        expected['age'] = job['age']['contract']['lr']
    if job.get('layerwise_lr_policy') is not None:
        return ({group['role'] for group in groups} == set(expected)
                and all(group['role'] in expected
                        and group['base_initial_lr'] == expected[group['role']]
                        and group['initial_lr'] == group['lr'] ==
                            expected[group['role']] * group['lr_scale']
                        for group in groups))
    return ({group['role'] for group in groups} == set(expected)
            and all(group['role'] in expected and group['lr'] == group['initial_lr'] == expected[group['role']]
                    for group in groups))


def sigmoid_formal_contract(job, science, site):
    """Keep the sigmoid loss fixed and validate this round's registered official gate."""
    from retrain import experiment91 as ex
    from retrain.classification_loss import PAIRWISE_LOSS_KINDS
    return (job['loss'] == ex.PAIR_SIGMOID_LOSS
            and job['loss']['kind'] in PAIRWISE_LOSS_KINDS
            and science.get('official_evaluation') == ex.expected_official_policy(site, science)
            and job.get('training_views') is None and job.get('native_local_view') is None)


def check(name, condition, actual=None):
    CHECKS.append({'name': name, 'passed': bool(condition), 'actual': actual})
    if not condition:
        raise ValueError(name)


def rejected(name, operation):
    try:
        operation()
    except ValueError as error:
        check(name, True, str(error))
    else:
        check(name, False)


def create_cpu_fresh_model(job):
    """Use the production seed/precision/factory path for CPU initial identity."""
    import torch
    from retrain.engine import _seed
    from retrain.models import create_model
    from retrain.precision import RULE, configure
    configure(RULE); _seed(job['seed'], torch.device('cpu'))
    return create_model(job, initialize=True)


def main(args):
    package, cfg, job, topology, policy, digest = build.build(args)
    from retrain.site_binding import load_site, require_cpu_channel
    site = load_site(); require_cpu_channel(site)
    import torch
    # Leave room for the validation parent and helper threads inside its 12-CPU
    # preparation allocation; this does not alter the model or training site.
    torch.set_num_threads(max(1, int(os.environ['NCCCU_CPU_THREADS']) - 2))
    torch.set_num_interop_threads(1)
    check('CPU_only_before_model', not torch.cuda.is_initialized())
    from retrain import experiment91 as ex
    from retrain.engine import _optimizer, _lr_factor
    from retrain.ema import EMA, tensor_digest, validate_state
    from retrain.config import job_contract
    from retrain.fresh import validate_fresh
    from retrain.precision import RULE, configure, runtime_proof
    from retrain.persistence import CheckpointStore, canonical_digest
    sys.path.insert(0, str(package))
    import r92_initial_gate as initial_gate
    model = create_cpu_fresh_model(job)
    is_r50 = job['model'] == 'resnet50_imagenet'
    is_effnet = job['model'] == 'efficientnetv2_rw_s'
    is_siglip2 = job['model'] == 'siglip2_vitb16'
    is_eva02 = job['model'] == 'eva02_base'
    is_dinov2 = job['model'] in ('dinov2_vits14','dinov2_vitb14')
    is_dino = ex.science_configuration()['candidate_profile'] in (ex.DINO_R50_BINDING,ex.DINOV2_PAIRLOG_BINDING,ex.DINOV2_BASE_BINDING,ex.SMALL_DINO3_BINDING,ex.LARGE_DINO3_BINDING,ex.LARGE_DINO3_SEED27_BINDING,*ex.VIT_DINO3_BINDINGS)
    is_swin = job['model'] in ('swin_base','swin_v2_small')
    is_deit = job['model'] == 'deit3_small'
    is_vit_dino3 = job['model'] in ('vit_dino3_small','vit_dino3_base')
    is_caformer = job['model'] == 'caformer_s36'
    is_pair_sigmoid = ex.science_configuration()['candidate_profile'] in (
        ex.PAIR_SIGMOID_SMALL_BINDING, ex.PAIR_SIGMOID_SMALL_SEED27_BINDING, *ex.PAIR_SIGMOID_SEED_REPLICAS)
    is_photo_identity = ex.science_configuration()['candidate_profile'] == ex.PHOTO_IDENTITY_SMALL_BINDING
    is_rotation_identity = ex.science_configuration()['candidate_profile'] == ex.ROTATION_IDENTITY_SMALL_BINDING
    is_content_pool = ex.science_configuration()['candidate_profile'] == ex.CONTENT_POOL_SMALL_BINDING
    is_stage3_residual = ex.science_configuration()['candidate_profile'] == ex.STAGE3_RESIDUAL_SMALL_BINDING
    is_rolewise_clip = ex.science_configuration()['candidate_profile'] == ex.ROLEWISE_CLIP_SMALL_BINDING
    is_head_warmup = ex.science_configuration()['candidate_profile'] == ex.HEAD_WARMUP_SMALL_BINDING
    is_layerwise = layerwise_profile(ex.science_configuration()['candidate_profile'])
    is_low_rank = job.get('low_rank_policy') is not None
    is_tri3 = job.get('tri3_downsample_policy') is not None
    is_mixstyle = job.get('mixstyle_policy') is not None
    is_stem_xy = job.get('stem_xy_policy') is not None
    from retrain.config import GLOBAL_NATIVE_SIX_RULE
    is_native6 = job.get('input_preprocessing')==GLOBAL_NATIVE_SIX_RULE
    is_global768 = ex.science_configuration()['candidate_profile'] in (ex.GLOBAL_768_SMALL_BINDING,ex.GLOBAL_768_SMALL_SEED27_BINDING)
    is_raw_gc = job.get('gradient_centralization_policy') is not None
    is_lion = job.get('optimizer_policy') is not None
    structure = ex.describe_model(model); public = model.public_initialization_receipt
    contract = ex.structure_contract()
    check('strict_exact_finite_source_then_fresh_binary', bool(public)
          and public['all_public_tensors_exact'] and public['all_public_floating_FP32_finite' if is_r50 or is_effnet else 'all_public_tensors_FP32_finite']
          and public['public_tensor_count'] == contract['public_state_tensors']
          and public['full_head_shape'] == contract['full_head_shape']
          and public.get('backbone_only', False) is is_dino
          and not public['strict_missing_keys'] and not public['strict_unexpected_keys'], public)
    if is_stem_xy:
        stem=model.network.features[0][0];proof=public['stem_xy_initialization']
        check('R135_native_public_RGB_exact_zero_XY_no_extra_state',
              structure['stem_xy']['weight_shape']==[96,5,4,4]
              and proof['rgb_before']==proof['rgb_after'] and proof['bias_before']==proof['bias_after']
              and proof['added_zero_elements']==3072 and not bool(torch.count_nonzero(stem.weight[:,3:])),
              {'stem':structure['stem_xy'],'initialization':proof})
    if is_native6:
        stem=model.network.features[0][0];proof=public['stem_native6_initialization']
        check('R150_native_public_RGB_exact_zero_native_channels',
              structure['stem_native6']['weight_shape']==[96,6,4,4]
              and proof['rgb_before']==proof['rgb_after'] and proof['bias_before']==proof['bias_after']
              and proof['added_zero_elements']==4608 and not bool(torch.count_nonzero(stem.weight[:,3:])),
              {'stem':structure['stem_native6'],'initialization':proof})
    check('actual_native_EVA02_public_mean_SwiGLU_LN_binary_agezero' if is_eva02 else 'actual_native_SigLIP2_public_MAP_LN_binary_agezero' if is_siglip2 else 'actual_native_EfficientNet_publicBN_avgpool_binary_agezero' if is_effnet else 'actual_DeiT3_LN_GELU_CLS_noBN_binary_agezero' if is_deit else 'actual_DINOv2_LN_GELU_CLS_meanpatch_noBN_binary_agezero' if is_dinov2 else 'actual_measured_LN_GELU_native_avgpool_noBN_binary_agezero',
          len(structure['layer_norms']) == contract['layer_norms']
          and all(row['eps'] == ex.candidate()['internal_norm_eps'] for row in structure['layer_norms'])
          and structure['batch_norm_modules'] == contract.get('batch_norm_modules', 0)
          and structure['buffer_count'] == contract.get('static_buffer_count', contract.get('static_index_buffers', contract.get('buffer_count', 0)))
          and len(dict(model.named_buffers())) == contract.get('static_buffer_count', contract.get('static_index_buffers', contract.get('buffer_count', 0))) and model.age_metadata == job['age']
          and (not hasattr(model, 'age_weight') if not job['age']['enabled'] else
               model.age_weight.numel() == 2 and bool((model.age_weight == 0).all()))
          and (not is_content_pool or (job['attention_pool'] == ex.candidate()['attention_pool']
                and structure['attention_pool'] == {'parameter_name': 'attention_pool_weight',
                    'shape': [1, 768], 'dtype': 'torch.float32'}
                and tuple(model.attention_pool_weight.shape) == (1, 768)
                and bool((model.attention_pool_weight == 0).all())))
          and (not is_stage3_residual or (job['stage3_residual'] == ex.candidate()['stage3_residual']
                and structure['stage3_residual'] == {'parameter_name': 'stage3_residual_weight',
                    'shape': [1, 384], 'dtype': 'torch.float32', 'feature_index': 5, 'channels': 384}
                and tuple(model.stage3_residual_weight.shape) == (1, 384)
                and bool((model.stage3_residual_weight == 0).all())))
          and sum(isinstance(module, torch.nn.GELU) and module.approximate == 'none'
                  for module in model.modules()) == contract.get('gelu_modules', contract.get('stochastic_depth_modules', 0)), structure)
    if is_pair_sigmoid:
        check('fixed_sigmoid_WORLD_loss_and_conditional_official_bound',
              sigmoid_formal_contract(job, ex.science_configuration(), site))
    if ex.science_configuration()['candidate_profile'] in (ex.HARD_POSITIVE_SMALL_BINDING,ex.HARD_POSITIVE_SMALL_SEED27_BINDING):
        from retrain.classification_loss import PAIRWISE_LOSS_KINDS
        check('fixed_hard_positive_top2_mix05_stable_WORLD_and_official_bound',
              job['loss'] == ex.PAIR_HARD_POSITIVE_LOSS
              and job['loss']['kind'] in PAIRWISE_LOSS_KINDS
              and (job['effective_batch'],job['positive_per_batch']) == (64,8)
              and ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996
              and job.get('training_views') is None and job.get('native_local_view') is None)
    if ex.science_configuration()['candidate_profile'] in (ex.AUC_MARGIN_SMALL_BINDING,ex.AUC_MARGIN_SMALL_SEED27_BINDING):
        from retrain.classification_loss import PAIRWISE_LOSS_KINDS
        check('fixed_batch_primal_AUC_margin1_WORLD_and_official_bound',
              job['loss'] == ex.PAIR_AUC_MARGIN_LOSS and job['loss']['kind'] in PAIRWISE_LOSS_KINDS
              and (job['effective_batch'],job['positive_per_batch']) == (64,8)
              and ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996
              and job.get('training_views') is None and job.get('native_local_view') is None)
    if ex.science_configuration()['candidate_profile'] in (ex.PAIR_BCE_MIX_SMALL_BINDING,ex.PAIR_BCE_MIX_SMALL_SEED27_BINDING):
        from retrain.classification_loss import PAIRWISE_LOSS_KINDS
        check('fixed_pairlog_pointwise_BCE_mix05_WORLD_and_official_bound',
              job['loss'] == ex.PAIR_BCE_MIX_LOSS and job['loss']['kind'] in PAIRWISE_LOSS_KINDS
              and (job['effective_batch'],job['positive_per_batch']) == (64,8)
              and ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996
              and job.get('training_views') is None and job.get('native_local_view') is None)
    if job.get('cosine_head_policy') is not None:
        from retrain.models import describe_cosine_head
        from retrain.config import COSINE_HEAD_RULE
        check('fixed_cosine16_native_parameters_and_graph_bound',
              job['cosine_head_policy'] == COSINE_HEAD_RULE
              and structure['cosine_head'] == describe_cosine_head(model,job)
              and job['loss'] == ex.PAIR_LOGISTIC_LOSS
              and ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996,
              structure['cosine_head'])
    if is_photo_identity:
        from unittest.mock import patch
        from retrain.data import ImageDataset
        with patch('retrain.data.configured_cache', return_value=None):
            training_dataset = ImageDataset(Path('.'), [], job['image_size'], training=True,
                training_scale=job.get('training_scale'),
                training_photometric=job.get('training_photometric'))
            evaluation_dataset = ImageDataset(Path('.'), [], job['image_size'], training=False)
        check('R122_training_Dataset_photometric_identity_and_logistic_official_binding',
              job['loss'] == ex.PAIR_LOGISTIC_LOSS
              and job.get('training_photometric') == training_dataset.training_photometric == 'identity-v1'
              and evaluation_dataset.training_photometric is None
              and job.get('training_views') is None and job.get('native_local_view') is None
              and ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996,
              {'training_dataset_policy': training_dataset.training_photometric,
               'evaluation_dataset_policy': evaluation_dataset.training_photometric})
    if is_rotation_identity:
        from unittest.mock import patch
        from retrain.data import ImageDataset
        with patch('retrain.data.configured_cache', return_value=None):
            training_dataset = ImageDataset(Path('.'), [], job['image_size'], training=True,
                training_scale=job.get('training_scale'),
                training_photometric=job.get('training_photometric'),
                training_rotation=job.get('training_rotation'))
            evaluation_dataset = ImageDataset(Path('.'), [], job['image_size'], training=False,
                training_rotation=job.get('training_rotation'))
        check('R127_training_Dataset_rotation_identity_and_logistic_official_binding',
              job['loss'] == ex.PAIR_LOGISTIC_LOSS
              and job.get('training_rotation') == training_dataset.training_rotation == 'identity-v1'
              and evaluation_dataset.training_rotation == 'identity-v1'
              and job.get('training_photometric') is None
              and training_dataset.training_photometric is None
              and job.get('training_views') is None and job.get('native_local_view') is None
              and ex.candidate()['training_rotation'] == 'identity-v1'
              and ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996,
              {'training_dataset_policy': training_dataset.training_rotation,
               'evaluation_dataset_policy': evaluation_dataset.training_rotation})
    if is_content_pool:
        was_training = model.training
        feature_shapes = []
        hook = model.network.features.register_forward_hook(
            lambda _module, _input, output: feature_shapes.append(list(output.shape)))
        model.eval()
        with torch.inference_mode():
            output_512 = model(torch.zeros(1, 3, 512, 512, dtype=torch.float32),
                               torch.zeros(1, 2, dtype=torch.float32))
        hook.remove()
        model.train(was_training)
        check('R124_512_features_S256_binary_age_readout_and_q_zero',
              feature_shapes == [[1, 768, 16, 16]]
              and tuple(output_512.shape) == (1,) and bool(torch.isfinite(output_512).all())
              and bool((model.attention_pool_weight == 0).all())
              and not torch.cuda.is_initialized(),
              {'feature_shapes': feature_shapes, 'positions': 256,
               'logit': float(output_512[0]), 'GPU_initialized': torch.cuda.is_initialized()})
    if is_swin:
        check('R117_selected_internal_AUROC_0996_official_gate_bound',
              ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996)
        original_mode = model.training
        model.eval()
        with torch.inference_mode():
            output_512 = model(torch.zeros(1, 3, 512, 512, dtype=torch.float32),
                               torch.zeros(1, 2, dtype=torch.float32))
        model.train(original_mode)
        check('Swin_512_native_window_binary_age_forward_finite',
              tuple(output_512.shape) == (1,) and bool(torch.isfinite(output_512).all())
              and not torch.cuda.is_initialized(),
              {'input': [1, 3, 512, 512], 'output': list(output_512.shape),
               'logit': float(output_512[0]), 'window': contract['window'],
               'GPU_initialized': torch.cuda.is_initialized()})
    if is_effnet:
        layers=model.assert_bn_mode()
        from retrain.distributed_engine import validate_explicit_microbatch_normalization
        validate_explicit_microbatch_normalization(model,topology)
        check('native_EfficientNet_public_BN_frozen_affine_trainable',len(layers)==112
              and all(not m.training and m.weight.requires_grad and m.bias.requires_grad for m in layers)
              and structure['buffer_sha256']==ex.science_configuration()['initial_buffer_sha256'])
    elif is_r50:
        layers = model.assert_bn_mode()
        from retrain.distributed_engine import validate_explicit_microbatch_normalization
        validate_explicit_microbatch_normalization(model, topology)
        check('R50_public_or_derived_BN_frozen_affine_trainable', len(layers) == 53
              and all(not m.training and m.weight.requires_grad and m.bias.requires_grad for m in layers)
              and structure['buffer_sha256'] == ex.science_configuration()['initial_buffer_sha256'])
        if is_dino:
            check('R118_selected_internal_AUROC_0996_official_gate_bound',
                  ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996)
            original_mode = model.training
            model.eval()
            with torch.inference_mode():
                output_512 = model(torch.zeros(1, 3, 512, 512, dtype=torch.float32),
                                   torch.zeros(1, 2, dtype=torch.float32))
            model.train(original_mode)
            check('DINO_R50_512_binary_age_finite', tuple(output_512.shape) == (1,)
                  and bool(torch.isfinite(output_512).all()) and not torch.cuda.is_initialized(),
                  {'shape': list(output_512.shape), 'logit': float(output_512[0])})
    else:
        wrong = copy.deepcopy(job); wrong['final_norm_eps'] = 1e-12
        rejected('old_R92_final_LN_job_rejected', lambda: ex.validate_job(wrong))
        final_norm = model.network.fc_norm if is_eva02 else model.network.head.norm if is_caformer else model.backbone.norm if is_dinov2 else model.network.norm if is_swin or is_deit or is_vit_dino3 or is_siglip2 else model.network.classifier[0]
        final_norm.eps = 1e-12
        rejected('wrong_actual_final_LN_rejected', lambda: ex.describe_model(model))
        final_norm.eps = ex.candidate()['final_norm_eps']
    optimizer = _optimizer(model, job)
    if job.get('nadam_policy') is not None:
        from retrain.config import NADAM_RULE
        check('native_NAdam_all_existing_groups_decoupled',
              job['nadam_policy']==NADAM_RULE
              and type(optimizer)is torch.optim.NAdam and len(optimizer.param_groups)==5 and not optimizer.state
              and all(g['decoupled_weight_decay']is True and g['betas']==(.9,.999)
                  and g['eps']==1e-8 and g['momentum_decay']==.004 and g['foreach']is None
                  and not g['maximize'] and not g['capturable'] and not g['differentiable']for g in optimizer.param_groups),
              {'class':type(optimizer).__name__,'groups':[{'role':g['role'],
                  'decoupled_weight_decay':g['decoupled_weight_decay'],'betas':list(g['betas']),
                  'eps':g['eps'],'momentum_decay':g['momentum_decay'],'foreach':g['foreach']}for g in optimizer.param_groups]})
    if job.get('adamw_amsgrad_policy') is not None:
        from retrain.config import ADAMW_AMSGRAD_RULE
        check('native_AdamW_all_existing_groups_AMSGrad',
              job['adamw_amsgrad_policy']==ADAMW_AMSGRAD_RULE
              and type(optimizer)is torch.optim.AdamW and len(optimizer.param_groups)==5 and not optimizer.state
              and all(g['amsgrad']is True and g['betas']==(.9,.999)and g['eps']==1e-8 for g in optimizer.param_groups),
              {'class':type(optimizer).__name__,'groups':[{'role':g['role'],'amsgrad':g['amsgrad'],
                  'betas':list(g['betas']),'eps':g['eps']}for g in optimizer.param_groups]})
    if job.get('adamw_beta2_policy') is not None:
        check('native_AdamW_all_existing_groups_fixed_beta2',
              type(optimizer)is torch.optim.AdamW and len(optimizer.param_groups)==5 and not optimizer.state
              and all(g['betas']==(.9,.95)and g['eps']==1e-8 and g['amsgrad']is False for g in optimizer.param_groups),
              {'class':type(optimizer).__name__,'groups':[{'role':g['role'],'betas':list(g['betas']),
                  'eps':g['eps'],'amsgrad':g['amsgrad']}for g in optimizer.param_groups]})
    from retrain.models import feature_freeze_proof
    parameter_trainability = feature_freeze_proof(model, job, optimizer)
    named = dict(model.named_parameters()); names = {id(parameter): name for name, parameter in named.items()}
    trainable = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
    frozen = {name: parameter for name, parameter in named.items() if not parameter.requires_grad}
    groups = [{'role': group['role'], 'lr': group['lr'], 'initial_lr': group['initial_lr'],
        **({key:group[key] for key in ('layer_id','lr_scale','base_initial_lr','param_names')}
           if is_layerwise else {}),
        **({key:group[key] for key in ('param_names','param_shapes')} if is_low_rank else {}),
        'weight_decay': group['weight_decay'], 'parameters': [
            {'name': names[id(parameter)], 'shape': list(parameter.shape), 'numel': parameter.numel()}
            for parameter in group['params']]} for group in optimizer.param_groups]
    members = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if is_lion:
        from retrain.engine import Lion
        from retrain.config import LION_RULE
        from retrain.age import validate_spec,LION_VERSION
        source_age=copy.deepcopy(job['age']['contract']);source_sha=source_age.pop('source_contract_sha256')
        source_age.update(version='age-linear-v1',lr=1e-3,weight_decay=1e-4)
        check('R137_actual_Lion_named_groups_effective_age_empty_single_moment',
              type(optimizer) is Lion and job['optimizer_policy']==ex.candidate()['optimizer_policy']==LION_RULE
              and not optimizer.state and len(members)==345 and len({id(p) for p in members})==345
              and all(g['optimizer_algorithm']=='lion10x-v1' and tuple(g['betas'])==(.9,.99)
                      and g['param_names']==[names[id(p)] for p in g['params']]
                      and g['param_shapes']==[list(p.shape) for p in g['params']] for g in optimizer.param_groups)
              and validate_spec(job['age'])['contract']['version']==LION_VERSION
              and source_age==ex.metadata()['AGE.json']['contract']
              and canonical_digest(source_age)==source_sha==ex.binding()['age_contract_sha256'],
              {'class':type(optimizer).__name__,'policy':job['optimizer_policy'],
               'groups':[{k:g[k] for k in ('role','initial_lr','weight_decay','betas','optimizer_algorithm','param_names','param_shapes')} for g in optimizer.param_groups],
               'age':job['age'],'source_age_contract_sha256':source_sha,'optimizer_state_entries':len(optimizer.state)})
    if is_raw_gc:
        targets=optimizer._gradient_centralization_targets
        check('R136_actual_Conv2d_Linear_rawGC_targets_and_original_Adam',
              job['gradient_centralization_policy']==ex.candidate()['gradient_centralization_policy']
              and [{'name':name,'shape':list(parameter.shape),'axes':list(range(1,parameter.ndim))}
                   for name,parameter in targets] ==
                  [{k:r[k] for k in ('name','shape','axes')} for r in structure['gradient_centralization']['targets']]
              and len(targets)==113 and {id(p) for _,p in targets} <= {id(p) for p in members}
              and not optimizer.state,
              structure['gradient_centralization'])
    expected_roles = ('backbone', 'head', 'age') if job['age']['enabled'] else ('backbone', 'head')
    roles = {role: sum(row['numel'] for group in groups if group['role'] == role for row in group['parameters'])
             for role in expected_roles}
    classifier_head = model.network.head.fc.fc2 if is_caformer else (model.classifier if is_r50 or is_dinov2 or is_vit_dino3 or is_siglip2 or is_eva02 else model.network.classifier)[-1]
    head = {id(parameter) for parameter in classifier_head.parameters()}
    if is_content_pool:
        head.add(id(model.attention_pool_weight))
    if is_stage3_residual:
        head.add(id(model.stage3_residual_weight))
    check('unique_groups_emptyLion_last_linear_LN_backbone' if is_lion else 'unique_groups_emptyAdam_last_linear_LN_backbone', not optimizer.state
        and len(named) == contract.get('backbone_parameter_tensors', contract.get('backbone_state_tensors', 0) - contract.get('buffer_count', 0)) + (3 if job['age']['enabled'] else 2) + int(is_content_pool) + int(is_stage3_residual) + (144 if is_low_rank else 0)
        and len(trainable) == len(members) == len({id(parameter) for parameter in members})
        and {id(parameter) for parameter in members} == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        and roles == ({'backbone': contract['backbone_parameters'] - sum(p.numel() for p in frozen.values()) + (1082880 if is_low_rank else 0) + (3072 if is_stem_xy else 0) + (4608 if is_native6 else 0),
                       'head': contract['features'] * (4 if is_vit_dino3 and (job.get('vit_dino3_pool_policy') or {}).get('version')=='vit-dinov3-final-ln-quadrant-patch-concat-v1' else 1) + 1 + (768 if is_content_pool else 0) + (384 if is_stage3_residual else 0),
                       **({'age': 2} if job['age']['enabled'] else {})})
        and {id(parameter) for group in optimizer.param_groups if group['role'] == 'head' for parameter in group['params']} == head
        and (all(row['name'].endswith(('.lora_A', '.lora_B')) for group in groups if group['role'] == 'backbone' for row in group['parameters']) if is_low_rank else
             any(row['name'] == ('network.fc_norm.weight' if is_eva02 else 'network.bn1.weight' if is_effnet else 'network.head.norm.weight' if is_caformer else 'backbone.bn1.weight' if is_r50 else 'backbone.norm.weight' if is_dinov2 else 'network.norm.weight' if is_swin or is_deit or is_vit_dino3 or is_siglip2 else 'network.classifier.0.weight') for group in groups if group['role'] == 'backbone' for row in group['parameters']))
        and group_lrs_match(groups, job)
        and all(group['weight_decay'] == ((job['age']['contract']['weight_decay'] if group['role']=='age' else
            job['weight_decay'] if any(named[row['name']].ndim>1 and not row['name'].endswith('.bias') for row in group['parameters']) else 0.) if is_lion else
            (1e-4 if group['role'] == 'age' else job['weight_decay'] if any(named[row['name']].ndim > 1 and not row['name'].endswith('.bias') for row in group['parameters']) else 0.)) for group in groups),
        {'all_parameter_tensors': len(named), 'trainable_tensors': len(trainable),
         'frozen_tensors': len(frozen), 'parameter_trainability': parameter_trainability,
         'elements_by_role': roles, 'group_count': len(groups)})
    if is_low_rank:
        from retrain.low_rank import LOW_RANK_RULE, optimizer_update_proof
        low_rank_initial = optimizer_update_proof(model, optimizer, 0)
        check('fixed_pointwise_rank8_actual_mask_groups_emptyAdam',
              job['low_rank_policy'] == ex.candidate().get('low_rank_policy') == LOW_RANK_RULE
              and len(named) == 489 and len(frozen) == 342 and len(trainable) == 147
              and parameter_trainability['trainable_parameter_elements'] == 1083651
              and all(int(torch.count_nonzero(p)) == 0 for n, p in named.items() if n.endswith('.lora_B')),
              {'parameter_trainability': parameter_trainability, 'actual_optimizer': low_rank_initial})
    if is_rolewise_clip:
        from retrain.gradient_clip import ROLEWISE_CLIP_RULE, build_role_groups
        role_groups = build_role_groups(optimizer, job['gradient_clip_policy'])
        check('R126_exact_rolewise_policy_and_unique_optimizer_membership',
              job['gradient_clip_policy'] == ex.candidate()['gradient_clip_policy'] == ROLEWISE_CLIP_RULE
              and job['precision'] == 'fp32' and job['age']['enabled'] is True
              and job['loss'] == ex.PAIR_LOGISTIC_LOSS
              and ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996
              and not hasattr(model, 'attention_pool_weight')
              and not hasattr(model, 'stage3_residual_weight')
              and set(role_groups) == {'backbone', 'head', 'age'}
              and {id(parameter) for parameters in role_groups.values() for parameter in parameters}
                    == {id(parameter) for parameter in members}
              and sum(len(parameters) for parameters in role_groups.values()) == len(members)
              and {role:sum(parameter.numel() for parameter in parameters)
                   for role,parameters in role_groups.items()} == roles,
              {'policy': job['gradient_clip_policy'], 'elements_by_role': roles,
               'parameter_tensors_by_role': {role:len(parameters) for role,parameters in role_groups.items()}})
    if is_head_warmup:
        from retrain.head_warmup import HEAD_WARMUP_RULE, build_update_groups, expected_optimizer_steps
        update_groups = build_update_groups(optimizer, job['head_warmup_policy'])
        check('R128_exact_head_warmup_policy_and_empty_role_Adam',
              job['head_warmup_policy'] == ex.candidate()['head_warmup_policy'] == HEAD_WARMUP_RULE
              and job['precision'] == 'fp32' and job['age']['enabled'] is True
              and job['loss'] == ex.PAIR_LOGISTIC_LOSS
              and job.get('gradient_clip_policy') is None
              and job.get('training_rotation') is None and job.get('training_photometric') is None
              and job.get('attention_pool') is None and job.get('stage3_residual') is None
              and ex.science_configuration().get('official_evaluation') == ex.OFFICIAL_GATE_0996
              and set(update_groups) == {'backbone', 'head', 'age'}
              and {id(parameter) for parameters in update_groups.values() for parameter in parameters}
                    == {id(parameter) for parameter in members}
              and sum(len(parameters) for parameters in update_groups.values()) == len(members)
              and {role:sum(parameter.numel() for parameter in parameters)
                   for role,parameters in update_groups.items()} == roles
              and expected_optimizer_steps(0) == {'backbone': None, 'head': None, 'age': None}
              and not optimizer.state,
              {'policy': job['head_warmup_policy'], 'elements_by_role': roles,
               'parameter_tensors_by_role': {role:len(parameters) for role,parameters in update_groups.items()}})
    if is_layerwise:
        from retrain.layerwise_lr import LAYERWISE_LR_RULE, group_identity
        signature = group_identity(model, optimizer.param_groups, job['layerwise_lr_policy'])
        check('R130_layerwise_group_identity_and_empty_Adam',
              job['layerwise_lr_policy'] == ex.candidate()['layerwise_lr_policy'] == LAYERWISE_LR_RULE
              and job['precision'] == 'fp32' and job['age']['enabled'] is True
              and job['loss'] == ex.PAIR_LOGISTIC_LOSS
              and not optimizer.state and len(signature) == len(groups)
              and all(row['param_names'] == [part['name'] for part in group['parameters']]
                      for row, group in zip(signature, groups)),
              {'group_identity': signature, 'optimizer_state_entries': len(optimizer.state)})
    model_digest = tensor_digest(model.state_dict()); ema = EMA(model, ex.ema_rule())
    tri3_ema_structure = ex.describe_model(ema.model) if is_tri3 else None
    mixstyle_ema_structure = ex.describe_model(ema.model) if is_mixstyle else None
    stem_xy_ema_structure = ex.describe_model(ema.model) if is_stem_xy or is_native6 else None
    raw_gc_ema_structure = ex.describe_model(ema.model) if is_raw_gc else None
    common_digest = tensor_digest({name: tensor for name, tensor in model.state_dict().items()
                                   if name not in ('attention_pool_weight', 'stage3_residual_weight')
                                   and not (is_low_rank and name.endswith(('.lora_A', '.lora_B')))}) if (is_content_pool or is_stage3_residual or is_low_rank) else model_digest
    check('initial_EMA_exact_and_outside_optimizer', tensor_digest(ema.model.state_dict()) == model_digest
          and (not is_tri3 or tri3_ema_structure == structure)
          and (not is_mixstyle or mixstyle_ema_structure == structure)
          and (not (is_stem_xy or is_native6) or stem_xy_ema_structure == structure)
          and (not is_raw_gc or raw_gc_ema_structure == structure)
          and (common_digest == '50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5'
               if (is_content_pool or is_stage3_residual) else model_digest == ex.science_configuration()['initial_model_tensor_sha256'])
          and (not (is_r50 or is_effnet) or tensor_digest(dict(model.named_parameters())) == ex.science_configuration()['initial_parameters_sha256'])
          and not {id(parameter) for parameter in ema.model.parameters()} & {id(parameter) for parameter in members},
          {'online_tri3': structure['tri3_downsample'],
           'ema_tri3': tri3_ema_structure['tri3_downsample']} if is_tri3 else
          {'online_mixstyle': structure['mixstyle'],
           'ema_mixstyle': mixstyle_ema_structure['mixstyle']} if is_mixstyle else None)
    if is_low_rank:
        check('pointwise_original345_fresh_common_state_and_distinct_whole',
              common_digest == '50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5'
              and model_digest == ex.science_configuration()['initial_model_tensor_sha256']
              and model_digest != common_digest,
              {'common_tensor_sha256': common_digest, 'full_tensor_sha256': model_digest})
    fresh = build.make_fresh(package, job, topology, policy, digest, public['initialization'])
    identity = {'format': 'glaucoma-retrain-v1', 'source_digest': digest,
        'manifest_digest': ex.binding()['manifest_digest'], 'job': job_contract(job),
        'initialization': public['initialization'], 'phase_parent_sha256': None,
        'distributed': topology, 'fresh': fresh, 'experiment91': ex.binding()}
    schedule = ex.schedule()
    total, warmup = schedule['total_updates'], schedule['warmup_updates']
    validate_fresh(fresh, identity, job, topology, schedule['steps_per_cycle'], mode='train')
    check('exact_fixed_science_and_eight_rank_fresh', ex.validate_runtime(identity, job, fresh=fresh)
          and fresh['schedule'] == schedule
          and (topology['world_size'], topology['per_rank_batch'], topology['accumulation']) == (8, 8, 1))
    expected = {'source_digest': digest, 'candidate': ex.candidate(), 'model_tensor_sha256': model_digest,
        'batch_norm_modules': structure['batch_norm_modules'], 'buffer_count': structure['buffer_count'], 'buffer_sha256': structure['buffer_sha256'],
        'candidate_normalization': structure, 'optimizer_state_entries': 0, 'trainable_tensors': len(trainable),
        'parameter_elements_by_role': roles, 'no_attention_q': not is_content_pool}
    if is_content_pool:
        from retrain.attention_pool import describe as describe_attention
        expected['attention_pool'] = describe_attention(model, optimizer)
        check('R124_original_seed29_common_state_exact_and_new_q_zero',
              common_digest == '50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5'
              and expected['attention_pool']['nonzero'] == 0
              and model.age_metadata == job['age'] and job['age']['enabled'] is True,
              {'common_tensor_sha256': common_digest, 'new_full_tensor_sha256': model_digest,
               'attention_pool': expected['attention_pool']})
    if is_stage3_residual:
        weight = model.stage3_residual_weight
        expected['stage3_residual'] = {'shape': list(weight.shape), 'dtype': str(weight.dtype),
            'nonzero': int(torch.count_nonzero(weight.detach()))}
        check('R125_original_seed29_common_state_exact_and_stage3_q_zero',
              common_digest == '50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5'
              and expected['stage3_residual'] == {'shape': [1, 384], 'dtype': 'torch.float32', 'nonzero': 0}
              and model.age_metadata == job['age'] and job['age']['enabled'] is True,
              {'common_tensor_sha256': common_digest, 'new_full_tensor_sha256': model_digest,
               'stage3_residual': expected['stage3_residual']})
    if ex.science_configuration().get('ssl_recipe') is not None:
        from retrain.ssl_stage import _derived_state, _non_encoder_state
        expected.update(public_tensor_sha256=model.public_initialization_receipt['public_tensor_sha256'],
                        public_encoder_tensor_sha256=tensor_digest(_derived_state(model)),
                        public_non_encoder_sha256=tensor_digest(_non_encoder_state(model)))
    if parameter_trainability is not None:
        expected['parameter_trainability'] = parameter_trainability
    if is_low_rank:
        expected['low_rank_optimizer'] = low_rank_initial
    if is_layerwise:
        expected['layerwise_group_identity'] = signature
    initial = {'format_version': 1, 'cpu_fixture_only': True, 'identity': identity,
        'job': job, 'mode': 'train', 'age': job['age'], 'manifest_digest': ex.binding()['manifest_digest'],
        'fresh': fresh, 'successful_updates': 0, 'attempted_updates': 0, 'evaluated_cycle': 0,
        'schedule': {**ex.schedule(), 'successful_updates': 0}, 'model': model.state_dict(),
        'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': {},
        'candidate_normalization': structure}
    if parameter_trainability is not None:
        initial['parameter_trainability'] = parameter_trainability
    if is_low_rank:
        initial['low_rank_optimizer'] = low_rank_initial
    store = CheckpointStore(args.output / 'CPU-FIXTURE-NOT-FOR-TRAINING/initial-checkpoint')
    checkpoint = store.save(initial); loaded = store.load('last', map_location='cpu')
    ex.validate_checkpoint(loaded); validate_state(loaded['ema'], loaded['model'], 0, ex.ema_rule())
    with torch.no_grad():
        classifier_head.bias.add_(1.)
    result = model.load_state_dict(loaded['model'], strict=True)
    from retrain.inference import _load_model
    reloaded, reload_job, _, reload_origin, _ = _load_model(
        checkpoint, device='cpu', variant='online')
    check('CPU_initial_checkpoint_strict_roundtrip_emptyAdam', not result.missing_keys and not result.unexpected_keys
          and tensor_digest(model.state_dict()) == model_digest and loaded['optimizer']['state'] == {}
          and reload_job == job and reloaded is not model
          and (not is_eva02 or (reload_origin.get('eva02')==structure['eva02']
                                and ex.describe_model(reloaded)==structure))
          and (not is_siglip2 or (reload_origin.get('siglip2')==structure['siglip2']
                                 and ex.describe_model(reloaded)==structure))
          and (not is_effnet or (reload_origin.get('efficientnetv2')==structure['efficientnetv2']
                                and ex.describe_model(reloaded)==structure))
          and (not (is_stem_xy or is_native6) or ex.describe_model(reloaded)==structure)
          and (not is_raw_gc or ex.describe_model(reloaded)==structure)
          and (not is_deit or (reload_origin.get('deit3') == structure['deit3']
                              and ex.describe_model(reloaded) == structure))
          and (not is_vit_dino3 or (reload_origin.get('vit_dino3') == structure['vit_dino3']
                                  and ex.describe_model(reloaded) == structure))
          and (not is_caformer or (reload_origin.get('caformer') == structure['caformer']
                                  and ex.describe_model(reloaded) == structure))
          and (not is_tri3 or (reload_origin.get('tri3_downsample') == structure['tri3_downsample']
                              and ex.describe_model(reloaded) == structure))
          and (not is_mixstyle or (reload_origin.get('mixstyle') == structure['mixstyle']
                                  and ex.describe_model(reloaded) == structure))
          and (not is_content_pool or (hasattr(reloaded, 'attention_pool_weight')
               and torch.equal(reloaded.attention_pool_weight, model.attention_pool_weight)))
          and (not is_stage3_residual or (hasattr(reloaded, 'stage3_residual_weight')
               and torch.equal(reloaded.stage3_residual_weight, model.stage3_residual_weight)))
          and tensor_digest(reloaded.state_dict()) == model_digest
          and reload_origin['tensor_sha256'] == model_digest
          and loaded['successful_updates'] == loaded['attempted_updates'] == 0 and loaded['cpu_fixture_only'] is True,
          {'checkpoint': str(checkpoint), 'sha256': build.sha(checkpoint), 'fixture_only': True})
    if job.get('cosine_head_policy') is not None:
        check('saved_cosine16_default_factory_graph_exact',
              reload_job['cosine_head_policy'] == job['cosine_head_policy']
              and describe_cosine_head(reloaded,reload_job) == structure['cosine_head'])
    if is_stem_xy or is_native6 or is_global768:
        original_mode=model.training;model.eval()
        with torch.inference_mode():
            size=job['image_size'];image=torch.zeros(1,6 if is_native6 else 3,size,size,dtype=torch.float32);ages=torch.zeros(1,2,dtype=torch.float32)
            online=model(image,ages);rebuilt=reloaded(image,ages)
        model.train(original_mode)
        check('R152_default_factory_saved_global768_graph_finite' if is_global768 else ('R150_default_factory_saved_native6_graph_512_finite' if is_native6 else 'R135_default_factory_saved_RGBXY_graph_512_finite'),
              tuple(rebuilt.shape)==(1,) and bool(torch.isfinite(rebuilt).all())
              and torch.equal(online,rebuilt) and not torch.cuda.is_initialized(),
              {'input_shape':[1,6 if is_native6 else 3,size,size],'actual_stem_shape':list(reloaded.network.features[0][0].weight.shape),
               'saved_geometry_matches':(reload_job['image_size'],reload_job['prepared_size'])==(job['image_size'],job['prepared_size'])})
    if is_layerwise:
        from retrain.engine import _validate_layerwise_optimizer_state
        check('R130_strict_reload_preserves_named_optimizer_groups',
              loaded['job']['layerwise_lr_policy'] == job['layerwise_lr_policy']
              and reload_job['layerwise_lr_policy'] == job['layerwise_lr_policy']
              and _validate_layerwise_optimizer_state(model, optimizer, loaded['optimizer'],
                                                     job['layerwise_lr_policy']) == signature)
    if is_head_warmup:
        check('R128_strict_reload_preserves_non_tensor_warmup_identity',
              loaded['job']['head_warmup_policy'] == job['head_warmup_policy']
              and reload_job['head_warmup_policy'] == job['head_warmup_policy']
              and reloaded.age_metadata == job['age']
              and loaded['optimizer']['state'] == {}
              and tensor_digest(reloaded.state_dict()) == model_digest)
    if is_low_rank:
        from retrain.low_rank import low_rank_proof, validate_optimizer_state
        check('pointwise_strict_reload_named_groups_and_frozen_base',
              reload_job['low_rank_policy'] == job['low_rank_policy']
              and low_rank_proof(reloaded, reload_job)['frozen_tensor_sha256'] == parameter_trainability['frozen_tensor_sha256']
              and validate_optimizer_state(model, optimizer, loaded['optimizer']) == parameter_trainability['optimizer_groups'])
    initials = [{'rank': rank, 'model_sha256': model_digest, 'optimizer_state_entries': 0,
        'bn_modules': structure['batch_norm_modules'], 'scaler': {}, 'precision_runtime': {**runtime_proof(RULE, device='cpu'),
            'scaler_enabled': False, 'scaler_state_dict': {}, 'effective_scale': 1.0},
        'candidate_normalization': structure, 'cpu_fixture_only': True} for rank in range(8)]
    if is_content_pool:
        for record in initials: record['attention_pool'] = expected['attention_pool']
    if is_stage3_residual:
        for record in initials: record['stage3_residual'] = expected['stage3_residual']
    if parameter_trainability is not None:
        for record in initials: record['parameter_trainability'] = parameter_trainability
    if is_low_rank:
        for record in initials: record['low_rank_optimizer'] = low_rank_initial
    check('initial_gate_eight_rank_coverage_CPU_fixture_only',
        initial_gate.verify_records(initials, groups, groups, expected, expected_world=8) == model_digest)
    if is_effnet or is_siglip2 or is_eva02:
        original_mode=model.training;model.eval()
        with torch.inference_mode():
            images=torch.zeros(1,3,512,512,dtype=torch.float32);ages=torch.zeros(1,2,dtype=torch.float32)
            online=model(images,ages);rebuilt=reloaded(images,ages)
        model.train(original_mode)
        check('native_EVA02_public512_pad6_checkpoint_cold_exact' if is_eva02 else 'native_SigLIP2_public512_MAP_checkpoint_cold_exact' if is_siglip2 else 'native_EfficientNet_public512_checkpoint_cold_exact',tuple(online.shape)==(1,)
              and bool(torch.isfinite(online).all()) and torch.equal(online,rebuilt)
              and ex.describe_model(model)==ex.describe_model(reloaded)
              and not torch.cuda.is_initialized())
    if is_vit_dino3 or is_eva02:
        from retrain.single91 import immutable_buffers, immutable_buffer_contract
        buffers = immutable_buffers(loaded['model'])
        count, buffer_sha = immutable_buffer_contract(loaded['candidate_normalization'])
        check('native_ViT_checkpoint_persistent_buffer_contract',
              len(buffers) == count == 0 and tensor_digest(buffers) == buffer_sha
              and ex.describe_model(reloaded) == loaded['candidate_normalization'])
    rejected('initial_gate_missing_rank_rejected', lambda: initial_gate.verify_records(initials[:-1], groups, groups, expected, expected_world=8))
    wrong_initial = copy.deepcopy(initials); wrong_initial[0]['model_sha256'] = '0' * 64
    rejected('initial_gate_wrong_tensor_rejected', lambda: initial_gate.verify_records(wrong_initial, groups, groups, expected, expected_world=8))
    lr_horizon = getattr(optimizer, '_lr_decay_updates', total)
    cursors = (0, warmup - 1, warmup, warmup + 1, total)
    if job.get('lr_decay_updates') is not None:
        cursors += (lr_horizon - 1, lr_horizon)
    factors = {str(update): _lr_factor(update, lr_horizon, warmup) for update in cursors}
    check('LR_successful_update_boundaries', factors['0'] == 1 / warmup
          and factors[str(warmup - 1)] == factors[str(warmup)] == 1 and factors[str(warmup + 1)] < 1
          and lr_horizon == (job.get('lr_decay_updates') or total)
          and (job.get('lr_decay_updates') is None or math.isclose(factors[str(lr_horizon - 1)], .01, abs_tol=1e-15))
          and math.isclose(factors[str(total)], .01, abs_tol=1e-15), factors)
    if is_effnet:
        boundaries={}
        for update in (warmup-1,warmup,warmup+1,total):
            prior=update-1;state=ema.state_dict()
            state.update(update_count=prior,copy_count=min(prior,warmup),
                         blend_count=max(0,prior-warmup),last_update_success=prior)
            ema.successful=prior
            validate_state(state,model.state_dict(),prior,ex.ema_rule())
            with torch.no_grad():
                ema.model.network.classifier[-1].bias.fill_(float(prior))
                model.network.classifier[-1].bias.fill_(float(update))
            ema.update(model,update)
            state=ema.state_dict();validate_state(state,model.state_dict(),update,ex.ema_rule())
            expected_bias=float(update) if update<=warmup else update-ex.ema_rule()['decay']
            check('EfficientNet_EMA_synthetic_boundary_'+str(update),
                  math.isclose(ema.model.network.classifier[-1].bias.item(),expected_bias,abs_tol=.002)
                  and all(torch.equal(dict(model.named_buffers())[n],t)for n,t in ema.model.named_buffers()))
            boundaries[str(update)]={k:v for k,v in state.items()if k!='model'}
        model.load_state_dict(loaded['model'],strict=True)
    elif is_r50:
        # Boundary fixtures retain the real 159 frozen buffers. Their explicit
        # synthetic cursors are CPU evidence, never a training/initial checkpoint.
        boundaries = {}
        for update in (warmup - 1, warmup, warmup + 1, total):
            prior = update - 1
            state = ema.state_dict()
            state.update(update_count=prior, copy_count=min(prior, warmup),
                         blend_count=max(0, prior-warmup), last_update_success=prior)
            ema.successful = prior
            validate_state(state, model.state_dict(), prior, ex.ema_rule())
            with torch.no_grad():
                ema.model.classifier[-1].bias.fill_(float(prior))
                model.classifier[-1].bias.fill_(float(update))
            ema.update(model, update)
            state = ema.state_dict(); validate_state(state, model.state_dict(), update, ex.ema_rule())
            expected_bias = float(update) if update <= warmup else update - ex.ema_rule()['decay']
            check('R50_EMA_synthetic_boundary_' + str(update),
                  math.isclose(ema.model.classifier[-1].bias.item(), expected_bias, abs_tol=.002)
                  and all(torch.equal(dict(model.named_buffers())[name], value)
                          for name, value in ema.model.named_buffers()))
            boundaries[str(update)] = {key: value for key, value in state.items() if key != 'model'}
        model.load_state_dict(loaded['model'], strict=True)
    elif is_swin:
        class IndexedToy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.blocks = torch.nn.ModuleList()
                for _ in range(24):
                    block = torch.nn.Module()
                    block.register_buffer('relative_position_index', torch.zeros(4096 if job['model']=='swin_v2_small' else 2401, dtype=torch.int64))
                    if job['model']=='swin_v2_small':
                        coordinates=next(t for n,t in model.named_buffers() if n.endswith('.relative_coords_table'))
                        block.register_buffer('relative_coords_table',coordinates.clone())
                    self.blocks.append(block)
        toy = IndexedToy(); toy_ema = EMA(toy, ex.ema_rule()); boundaries = {}
        for update in (warmup - 1, warmup, warmup + 1, total):
            prior = update - 1
            state = toy_ema.state_dict()
            state.update(update_count=prior, copy_count=min(prior, warmup),
                         blend_count=max(0, prior-warmup), last_update_success=prior)
            toy_ema.successful = prior
            validate_state(state, toy.state_dict(), prior, ex.ema_rule())
            with torch.no_grad():
                toy_ema.model.weight.fill_(float(prior))
                toy.weight.fill_(float(update))
            toy_ema.update(toy, update)
            state = toy_ema.state_dict(); validate_state(state, toy.state_dict(), update, ex.ema_rule())
            expected_weight = float(update) if update <= warmup else update - ex.ema_rule()['decay']
            check('Swin_static_index_EMA_boundary_' + str(update),
                  math.isclose(toy_ema.model.weight.item(), expected_weight, abs_tol=.002)
                  and all(torch.equal(dict(toy.named_buffers())[name], value)
                          for name, value in toy_ema.model.named_buffers()))
            boundaries[str(update)] = {key: value for key, value in state.items() if key != 'model'}
    else:
        scalar = torch.nn.Linear(1, 1, bias=False); scalar.weight.data.zero_()
        scalar_ema = EMA(scalar, ex.ema_rule()); boundaries = {}
        for update in range(1, total + 1):
            scalar.weight.data.fill_(float(update)); scalar_ema.update(scalar, update)
            if update in (warmup - 1, warmup, warmup + 1, total):
                state = scalar_ema.state_dict(); validate_state(state, scalar.state_dict(), update, ex.ema_rule())
                boundaries[str(update)] = {key: value for key, value in state.items() if key != 'model'}
                if update <= warmup:
                    check('EMA_warmcopy_at_' + str(update), torch.equal(scalar.weight, scalar_ema.model.weight))
                elif update == warmup + 1:
                    check('EMA_first_blend', math.isclose(scalar_ema.model.weight.item(), warmup + 1 - ex.ema_rule()['decay'], abs_tol=.0002))
    check('EMA_successful_clock_counts_at_maximum', boundaries[str(total)]['copy_count'] == warmup
          and boundaries[str(total)]['blend_count'] == total - warmup
          and boundaries[str(total)]['update_count'] == total, boundaries)
    check('CPU_only_after_all_checks', not torch.cuda.is_initialized())
    result = {'status': 'completed', 'exit_code': 0, 'scope': 'production_binary_age_CPU_only',
        'science_configuration': site['science']['configuration'],
        'checks': CHECKS, 'source_digest': digest, 'candidate': ex.candidate(), 'initial_state_control': expected,
        'job_contract': job_contract(job), 'topology': topology, 'site_binding_sha256': build.sha(package / 'SITE_BINDING.json'),
        'source_files': build.read(package / 'SOURCE-MANIFEST.reference.json')['after'],
        'python': sys.executable, 'pod_uid': os.environ['NCCCU_POD_UID'], 'cpu_mask': sorted(os.sched_getaffinity(0)),
        'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(), 'gpu_initialized': False,
        'cpu_allocation': os.environ['NCCCU_CPU_ALLOCATION'], 'cpu_allocation_sha256': os.environ['NCCCU_CPU_ALLOCATION_SHA256'],
        'asset_receipt': str(args.asset_receipt), 'asset_receipt_sha256': args.asset_receipt_sha256,
        'training_updates': 0, 'data_images_read': False, 'official_semantics_read': False,
        'production_barrier_released': False, 'formal_clock_created': False, 'training_lock_acquired': False}
    original = args.output / 'CPU-RECEIPT.json'; build.write(original, result)
    evidence = package / 'evidence'
    build.write(package / 'FRESH-SPEC.template.json', fresh)
    build.write(evidence / 'CPU_PARAMETER_GROUPS.json', groups)
    build.write(evidence / 'FRESH_EMA_CONTRACT.json', {'rule': ex.ema_rule()})
    build.write(evidence / 'MODEL_CPU_ADMISSION.template.json', {**result,
        'formal_train_config_sha256': None, 'original_receipt_path': str(original), 'original_receipt_sha256': build.sha(original)})
    files = {str(path.relative_to(package)): build.sha(path) for path in sorted(package.rglob('*')) if path.is_file()}
    manifest = args.output / 'PACKAGE-MANIFEST.json'
    build.write(manifest, {'schema_version': 1, 'formal_root': site['round_root'], 'package_root': str(package),
        'git_commit': site['git_commit'], 'files': files, 'bundle_source_digest': digest,
        'activate_script_sha256': build.sha(args.output / 'activate.py')})
    prep = {'status': 'prepared_no_formal_clock_no_GPU', 'exit_code': 0, 'source_digest': digest,
        'initial_state_control': expected, 'cpu_receipt_path': str(original), 'cpu_receipt_sha256': build.sha(original),
        'package_manifest_sha256': build.sha(manifest), 'package_root': str(package), 'formal_root': site['round_root'],
        'source_state': site.get('source_state'), 'GPU_started': False, 'formal_clock_created': False}
    build.write(args.output / 'CPU-PREP-RECEIPT.json', prep)
    print(json.dumps({'status': result['status'], 'checks': len(CHECKS), 'passed': sum(row['passed'] for row in CHECKS),
        'cpu_receipt': str(original), 'cpu_receipt_sha256': build.sha(original),
        'prep_receipt': str(args.output / 'CPU-PREP-RECEIPT.json'), 'prep_receipt_sha256': build.sha(args.output / 'CPU-PREP-RECEIPT.json'),
        'initial_tensor_sha256': model_digest, 'GPU_started': False}, indent=2))


if __name__ == '__main__':
    arguments = build.parser().parse_args()
    started = time.time()
    try:
        main(arguments)
    except BaseException as error:
        if arguments.output.is_dir():
            build.write(arguments.output / 'FAILURE.json', {'status': 'failed', 'exit_code': 1,
                'checks': CHECKS, 'error': repr(error), 'traceback': traceback.format_exc(), 'wall_seconds': time.time() - started})
        raise

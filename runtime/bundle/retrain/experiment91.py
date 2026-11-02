"""Explicit ConvNeXt22K-to-1K source over unchanged fixed 9:1 metadata."""
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from .persistence import canonical_digest, sha256_file
from .site_binding import load_site
from .classification_loss import PAIR_HARD_POSITIVE_LOSS, PAIR_AUC_MARGIN_LOSS

FREEZE_SHA = '90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6'
BINDING_VERSION = 'single-resnet50-imagenetv2-fixed-9010-24k-earlystop-seed20261029-publicbn-v1'
FIT_BN_BINDING_VERSION = 'single-resnet50-imagenetv2-fixed-9010-24k-earlystop-seed20261029-fitbn-v1'
FROZEN_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-freezeearly-v1'
LAST_STRIDE_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-laststride1-v1'
FIT_SSL_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-fitssl1600-v1'
BASE_BINDING = 'single-convnext-base-in22k-ft-in1k-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1'
SWIN_BINDING = 'single-swin-base-ms22k-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1'
DINO_R50_BINDING = 'single-resnet50-dino-imagenet-fixedbn-9010-24k-earlystop-seed20261029-v1'
PAIR_SIGMOID_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairsigmoid-v1'
PAIR_SIGMOID_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairsigmoid-v1'
PHOTO_IDENTITY_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-photoidentity-v1'
NO_AGE_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-noage-v1'
CONTENT_POOL_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-contentpool-v1'
STAGE3_RESIDUAL_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-stage3residual-v1'
ROLEWISE_CLIP_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-roleclip1-v1'
ROTATION_IDENTITY_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-rotationidentity-v1'
HEAD_WARMUP_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-headwarmup1600-v1'
SMALL_22K_ONLY_BINDING = 'single-convnext-small-in22konly-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1'
LAYERWISE_LR_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-layerdecay08-v1'
LAYERWISE_LR_SMALL_R1_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-layerdecay08-r1-v1'
POINTWISE_LORA8_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-pointwise-lora8-v1'
LOGIT_SQUARE_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-logitsquare01-v1'
TRI3_DOWNSAMPLE_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-tri3downsample-v1'
MIXSTYLE_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-mixstyle13-local8-v1'
MIXSTYLE_SMALL_R1_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-mixstyle13-local8-r1-v1'
STEM_XY_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-stem-rgbxy-v1'
RAW_GC_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-rawgc-convlinear-v1'
LION_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-lion10x-v1'
LION_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-lion10x-v1'
LOCAL_CONTRAST_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-localcontrast10-v1'
LOCAL_CONTRAST_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-localcontrast10-v1'
DINOV2_PAIRLOG_BINDING = 'single-dinov2-vits14-7764ea0-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1'
DINOV2_PAIRLOG_SEED27_BINDING = 'single-dinov2-vits14-7764ea0-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-v1'
BRIGHT_CENTER_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-brightcenter-half-v1'
NATIVE_BRIGHT_CENTER_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-native-brightcenter-half-v1'
NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-native-brightcenter-half-v1'
GLOBAL_NATIVE_INSET_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-inset256-v1'
GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-inset256-v1'
NO_MODEL_STOCH_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-nostoch-v1'
NO_MODEL_STOCH_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-nostoch-v1'
GLOBAL_NATIVE_SIX_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-native6-v1'
GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-native6-v1'
GLOBAL_768_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-global768-v1'
GLOBAL_768_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-global768-v1'
SWIN_V2_SMALL_BINDING = 'single-swin-v2-small-db3a0c7e-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1'
SWIN_V2_SMALL_SEED27_BINDING = 'single-swin-v2-small-db3a0c7e-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-v1'
HARD_POSITIVE_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-positive-top2-mix05-v1'
HARD_POSITIVE_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-positive-top2-mix05-v1'
SMALL_384_BINDING = 'single-convnext-small-14a83735-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1'
SMALL_384_SEED27_BINDING = 'single-convnext-small-14a83735-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-v1'
AUC_MARGIN_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-batch-aucmargin1-v1'
AUC_MARGIN_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-batch-aucmargin1-v1'
COSINE_HEAD_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-cosine16-v1'
COSINE_HEAD_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-cosine16-v1'
DEIT3_SMALL_BINDING = 'single-deit3-small-8d59b281-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1'
DEIT3_SMALL_SEED27_BINDING = 'single-deit3-small-8d59b281-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-v1'
DEIT3_MEANPATCH_BINDING = 'single-deit3-small-8d59b281-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-meanpatch-v1'
DEIT3_MEANPATCH_SEED27_BINDING = 'single-deit3-small-8d59b281-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-meanpatch-v1'
ADAMW_BETA2_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-adamw-beta2-095-v1'
ADAMW_BETA2_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-adamw-beta2-095-v1'
BASE_CLIP_BINDING = 'single-convnext-base-clip-768f3849-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1'
BASE_CLIP_SEED27_BINDING = 'single-convnext-base-clip-768f3849-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-v1'
BASE_CLIP_STORAGE16_BINDING = 'single-convnext-base-clip-768f3849-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-storage16-r174-v1'
QUADRANT_MIL_SMALL_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-corner288max05-v1'
QUADRANT_MIL_SMALL_SEED27_BINDING = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-corner288max05-v1'
PAIR_SIGMOID_LOSS = {'version': 'binary-pairwise-sigmoid-v1', 'kind': 'binary_pairwise_sigmoid',
                     'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32',
                     'scope': 'global_batch'}
PAIR_LOGISTIC_LOSS = {'version': 'binary-pairwise-logistic-v1', 'kind': 'binary_pairwise_logistic',
                      'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32',
                      'scope': 'global_batch'}
SUPPORTED_BINDINGS = (BINDING_VERSION, FIT_BN_BINDING_VERSION, FROZEN_SMALL_BINDING,
                      LAST_STRIDE_SMALL_BINDING, FIT_SSL_SMALL_BINDING, BASE_BINDING, SWIN_BINDING,
                      DINO_R50_BINDING, PAIR_SIGMOID_SMALL_BINDING,
                      PAIR_SIGMOID_SMALL_SEED27_BINDING, PHOTO_IDENTITY_SMALL_BINDING,
                      NO_AGE_SMALL_BINDING, CONTENT_POOL_SMALL_BINDING, STAGE3_RESIDUAL_SMALL_BINDING,
                      ROLEWISE_CLIP_SMALL_BINDING, ROTATION_IDENTITY_SMALL_BINDING,
                      HEAD_WARMUP_SMALL_BINDING, SMALL_22K_ONLY_BINDING,
                      LAYERWISE_LR_SMALL_BINDING, LAYERWISE_LR_SMALL_R1_BINDING,
                      POINTWISE_LORA8_SMALL_BINDING, LOGIT_SQUARE_SMALL_BINDING,
                      TRI3_DOWNSAMPLE_SMALL_BINDING, MIXSTYLE_SMALL_BINDING,
                      MIXSTYLE_SMALL_R1_BINDING, STEM_XY_SMALL_BINDING, RAW_GC_SMALL_BINDING, LION_SMALL_BINDING,
                      LION_SMALL_SEED27_BINDING, LOCAL_CONTRAST_SMALL_BINDING, LOCAL_CONTRAST_SMALL_SEED27_BINDING,
                      DINOV2_PAIRLOG_BINDING,DINOV2_PAIRLOG_SEED27_BINDING,BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING,GLOBAL_NATIVE_INSET_SMALL_BINDING,
                      GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING,NO_MODEL_STOCH_SMALL_BINDING,
                      NO_MODEL_STOCH_SMALL_SEED27_BINDING,GLOBAL_NATIVE_SIX_SMALL_BINDING,
                      GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING,GLOBAL_768_SMALL_BINDING,
                      GLOBAL_768_SMALL_SEED27_BINDING,SWIN_V2_SMALL_BINDING,
                      SWIN_V2_SMALL_SEED27_BINDING,HARD_POSITIVE_SMALL_BINDING,
                      HARD_POSITIVE_SMALL_SEED27_BINDING,SMALL_384_BINDING,
                      SMALL_384_SEED27_BINDING,AUC_MARGIN_SMALL_BINDING,
                      AUC_MARGIN_SMALL_SEED27_BINDING,COSINE_HEAD_SMALL_BINDING,
                      COSINE_HEAD_SMALL_SEED27_BINDING,DEIT3_SMALL_BINDING,DEIT3_SMALL_SEED27_BINDING,DEIT3_MEANPATCH_BINDING,
                      DEIT3_MEANPATCH_SEED27_BINDING,ADAMW_BETA2_SMALL_BINDING,
                      ADAMW_BETA2_SMALL_SEED27_BINDING,BASE_CLIP_BINDING,
                      BASE_CLIP_SEED27_BINDING,BASE_CLIP_STORAGE16_BINDING,QUADRANT_MIL_SMALL_BINDING,QUADRANT_MIL_SMALL_SEED27_BINDING)
OFFICIAL_GATE_0996 = {'version': 'selected-internal-dev-auroc-0996-v1',
    'metric': 'selected_internal_dev_auroc', 'threshold': 0.996,
    'after': 'complete_internal_freeze', 'split': 'official_valid_5849',
    'below_status': 'not_run_by_policy'}


def official_allowed_for_selected(policy, selected):
    """Gate one official run only after the complete internal selector has frozen."""
    if policy is None:
        return True
    if policy != OFFICIAL_GATE_0996 or not isinstance(selected, dict):
        raise ValueError('versioned official gate differs')
    auroc = selected.get('auroc')
    if (isinstance(auroc, bool) or not isinstance(auroc, (int, float))
            or not math.isfinite(auroc) or not 0 <= auroc <= 1):
        raise ValueError('complete selected internal dev AUROC required for official gate')
    return auroc >= policy['threshold']
CANDIDATE = {'id': 'r111-resnet50-imagenetv2-24k-p4-seed20261029-publicbn-9010', 'experiment_order': 21,
    'model': 'resnet50_imagenet', 'public_revision': 'IMAGENET1K_V2-11ad3fa6',
    'readout': 'native_avgpool_dropout_linear_plus_age', 'features': 2048,
    'final_norm_eps': None, 'internal_norm_eps': None, 'batch_norm_modules': 53,
    'batch_norm_buffers': 159, 'attention_pool': None, 'bn_initialization': 'public',
    'head_initialization': 'normal_std0.01_zero_bias', 'head_seed': 20261027, 'head_lr_scope': 'last_linear'}

def science_configuration():
    """Read the exact versioned science file pinned by this package's site."""
    from .gradient_clip import ROLEWISE_CLIP_RULE
    from .head_warmup import HEAD_WARMUP_RULE
    from .layerwise_lr import LAYERWISE_LR_RULE
    from .low_rank import LOW_RANK_RULE
    from .classification_loss import OUTPUT_REGULARIZATION_RULE
    from .config import COSINE_HEAD_RULE,DEIT3_MEANPATCH_RULE,ADAMW_BETA2_RULE,QUADRANT_MIL_RULE
    site = load_site()
    try:
        reference = site['science']['configuration']
        path = Path(reference['path'])
        if (not path.is_absolute() or path.is_symlink()
                or not path.resolve().is_relative_to(Path(site['work_root']).resolve())):
            raise ValueError('pinned science configuration path/SHA differs')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('pinned science configuration path/SHA differs')
        value = json.loads(raw)
        if value.get('candidate_profile') == TRI3_DOWNSAMPLE_SMALL_BINDING:
            from .tri3_downsample import TRI3_RULE
        if value.get('candidate_profile') in (MIXSTYLE_SMALL_BINDING, MIXSTYLE_SMALL_R1_BINDING):
            from .mixstyle import MIXSTYLE_RULE
        if value.get('candidate_profile') == STEM_XY_SMALL_BINDING:
            from .models import STEM_XY_RULE
        if value.get('candidate_profile') == RAW_GC_SMALL_BINDING:
            from .config import GRADIENT_CENTRALIZATION_RULE
        if value.get('candidate_profile') in (LION_SMALL_BINDING, LION_SMALL_SEED27_BINDING):
            from .config import LION_RULE
        if value.get('candidate_profile') in (LOCAL_CONTRAST_SMALL_BINDING, LOCAL_CONTRAST_SMALL_SEED27_BINDING):
            from .config import LOCAL_CONTRAST_RULE
        if value.get('candidate_profile')==BRIGHT_CENTER_SMALL_BINDING:
            from .config import BRIGHT_CENTER_RULE
        if value.get('candidate_profile') in (NATIVE_BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING):
            from .config import NATIVE_BRIGHT_CENTER_RULE
        if value.get('candidate_profile') in (GLOBAL_NATIVE_INSET_SMALL_BINDING,GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING):
            from .config import GLOBAL_NATIVE_INSET_RULE
        if value.get('candidate_profile') in (GLOBAL_NATIVE_SIX_SMALL_BINDING,GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING):
            from .config import GLOBAL_NATIVE_SIX_RULE
        if (type(value.get('schema_version')) is not int or value['schema_version'] != 1
                or value.get('candidate_profile') not in SUPPORTED_BINDINGS
                or (value.get('candidate_profile') == FIT_SSL_SMALL_BINDING) != (value.get('ssl_recipe') is not None)
                or (value.get('candidate_profile') in (SWIN_BINDING, DINO_R50_BINDING,
                                                       PAIR_SIGMOID_SMALL_BINDING,
                                                       PAIR_SIGMOID_SMALL_SEED27_BINDING,
                                                       PHOTO_IDENTITY_SMALL_BINDING,
                                                       NO_AGE_SMALL_BINDING, CONTENT_POOL_SMALL_BINDING,
                                                       STAGE3_RESIDUAL_SMALL_BINDING,
                                                       ROLEWISE_CLIP_SMALL_BINDING,
                                                       ROTATION_IDENTITY_SMALL_BINDING,
                                                       HEAD_WARMUP_SMALL_BINDING,
                                                       SMALL_22K_ONLY_BINDING,
                                                       LAYERWISE_LR_SMALL_BINDING,
                                                       LAYERWISE_LR_SMALL_R1_BINDING,
                                                       POINTWISE_LORA8_SMALL_BINDING,
                                                       LOGIT_SQUARE_SMALL_BINDING,
                                                       TRI3_DOWNSAMPLE_SMALL_BINDING,
                                                       MIXSTYLE_SMALL_BINDING,
                                                       MIXSTYLE_SMALL_R1_BINDING, STEM_XY_SMALL_BINDING, RAW_GC_SMALL_BINDING, LION_SMALL_BINDING,
                                                       LION_SMALL_SEED27_BINDING, LOCAL_CONTRAST_SMALL_BINDING, LOCAL_CONTRAST_SMALL_SEED27_BINDING,
                                                       DINOV2_PAIRLOG_BINDING,DINOV2_PAIRLOG_SEED27_BINDING,BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING,GLOBAL_NATIVE_INSET_SMALL_BINDING,
                                                       GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING,NO_MODEL_STOCH_SMALL_BINDING,
                                                       NO_MODEL_STOCH_SMALL_SEED27_BINDING,GLOBAL_NATIVE_SIX_SMALL_BINDING,
                                                       GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING,GLOBAL_768_SMALL_BINDING,
                                                       GLOBAL_768_SMALL_SEED27_BINDING,SWIN_V2_SMALL_BINDING,
                                                       SWIN_V2_SMALL_SEED27_BINDING,HARD_POSITIVE_SMALL_BINDING,
                                                       HARD_POSITIVE_SMALL_SEED27_BINDING,SMALL_384_BINDING,
                                                       SMALL_384_SEED27_BINDING,AUC_MARGIN_SMALL_BINDING,
                                                       AUC_MARGIN_SMALL_SEED27_BINDING,COSINE_HEAD_SMALL_BINDING,
                                                       COSINE_HEAD_SMALL_SEED27_BINDING,DEIT3_SMALL_BINDING,DEIT3_SMALL_SEED27_BINDING,DEIT3_MEANPATCH_BINDING,
                                                       DEIT3_MEANPATCH_SEED27_BINDING,ADAMW_BETA2_SMALL_BINDING,
                                                       ADAMW_BETA2_SMALL_SEED27_BINDING,BASE_CLIP_BINDING,
                                                       BASE_CLIP_SEED27_BINDING,BASE_CLIP_STORAGE16_BINDING,QUADRANT_MIL_SMALL_BINDING,QUADRANT_MIL_SMALL_SEED27_BINDING)
                    and value.get('official_evaluation') != OFFICIAL_GATE_0996)
                or type(value.get('age_enabled', True)) is not bool
                or (value.get('candidate_profile') not in (QUADRANT_MIL_SMALL_BINDING,QUADRANT_MIL_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('quadrant_mil_policy')is not None
                         or value.get('candidate',{}).get('quadrant_mil_policy')is not None))
                or (value.get('candidate_profile') in (QUADRANT_MIL_SMALL_BINDING,QUADRANT_MIL_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('quadrant_mil_policy')!=QUADRANT_MIL_RULE
                         or value.get('candidate',{}).get('quadrant_mil_policy')!=QUADRANT_MIL_RULE))
                or (value.get('candidate_profile') not in (DEIT3_MEANPATCH_BINDING,DEIT3_MEANPATCH_SEED27_BINDING)
                    and (value.get('job',{}).get('deit_pool_policy')is not None
                         or value.get('candidate',{}).get('deit_pool_policy')is not None))
                or (value.get('candidate_profile') in (DEIT3_MEANPATCH_BINDING,DEIT3_MEANPATCH_SEED27_BINDING)
                    and (value.get('job',{}).get('deit_pool_policy')!=DEIT3_MEANPATCH_RULE
                         or value.get('candidate',{}).get('deit_pool_policy')!=DEIT3_MEANPATCH_RULE))
                or (value.get('candidate_profile') not in (ADAMW_BETA2_SMALL_BINDING,ADAMW_BETA2_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('adamw_beta2_policy')is not None
                         or value.get('candidate',{}).get('adamw_beta2_policy')is not None))
                or (value.get('candidate_profile') in (ADAMW_BETA2_SMALL_BINDING,ADAMW_BETA2_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('adamw_beta2_policy')!=ADAMW_BETA2_RULE
                         or value.get('candidate',{}).get('adamw_beta2_policy')!=ADAMW_BETA2_RULE
                         or value['job'].get('loss')!=PAIR_LOGISTIC_LOSS))
                or (value.get('candidate_profile') not in (COSINE_HEAD_SMALL_BINDING,COSINE_HEAD_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('cosine_head_policy') is not None
                         or value.get('candidate',{}).get('cosine_head_policy') is not None))
                or (value.get('candidate_profile') in (COSINE_HEAD_SMALL_BINDING,COSINE_HEAD_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('cosine_head_policy') != COSINE_HEAD_RULE
                         or value.get('candidate',{}).get('cosine_head_policy') != COSINE_HEAD_RULE
                         or value.get('job',{}).get('loss') != PAIR_LOGISTIC_LOSS
                         or (value['job'].get('image_size'),value['job'].get('prepared_size'),value['job'].get('seed')) != (512,512,20261027 if value['candidate_profile']==COSINE_HEAD_SMALL_SEED27_BINDING else 20261029)
                         or (value['job'].get('effective_batch'),value['job'].get('positive_per_batch')) != (64,8)
                         or any(value['job'].get(k) is not None for k in ('input_preprocessing','optimizer_policy',
                             'gradient_centralization_policy','gradient_clip_policy','head_warmup_policy',
                             'layerwise_lr_policy','low_rank_policy','stem_xy_policy','mixstyle_policy',
                             'tri3_downsample_policy','output_regularization','attention_pool','stage3_residual',
                             'freeze_features_before','final_downsample_stride','training_views','training_rotation','training_photometric'))))
                or (value.get('candidate_profile') not in (LOCAL_CONTRAST_SMALL_BINDING, LOCAL_CONTRAST_SMALL_SEED27_BINDING,BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING,GLOBAL_NATIVE_INSET_SMALL_BINDING,GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING,GLOBAL_NATIVE_SIX_SMALL_BINDING,GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('input_preprocessing') is not None
                         or value.get('candidate',{}).get('input_preprocessing') is not None))
                or (value.get('candidate_profile') in (LOCAL_CONTRAST_SMALL_BINDING, LOCAL_CONTRAST_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('input_preprocessing') != LOCAL_CONTRAST_RULE
                         or value.get('candidate',{}).get('input_preprocessing') != LOCAL_CONTRAST_RULE))
                or (value.get('candidate_profile')==BRIGHT_CENTER_SMALL_BINDING
                    and (value.get('job',{}).get('input_preprocessing')!=BRIGHT_CENTER_RULE
                         or value.get('candidate',{}).get('input_preprocessing')!=BRIGHT_CENTER_RULE))
                or (value.get('candidate_profile') in (NATIVE_BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('input_preprocessing')!=NATIVE_BRIGHT_CENTER_RULE
                         or value.get('candidate',{}).get('input_preprocessing')!=NATIVE_BRIGHT_CENTER_RULE
                         or not isinstance(value.get('prepared_cache_receipt'),dict)))
                or (value.get('candidate_profile') in (GLOBAL_NATIVE_INSET_SMALL_BINDING,GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('input_preprocessing')!=GLOBAL_NATIVE_INSET_RULE
                         or value.get('candidate',{}).get('input_preprocessing')!=GLOBAL_NATIVE_INSET_RULE
                         or not isinstance(value.get('composite_cache_receipt'),dict)))
                or (value.get('candidate_profile') in (GLOBAL_NATIVE_SIX_SMALL_BINDING,GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('input_preprocessing')!=GLOBAL_NATIVE_SIX_RULE
                         or value.get('candidate',{}).get('input_preprocessing')!=GLOBAL_NATIVE_SIX_RULE
                         or not isinstance(value.get('composite_cache_receipt'),dict)))
                or (value.get('candidate_profile') in (GLOBAL_768_SMALL_BINDING,GLOBAL_768_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('image_size')!=768
                         or value.get('job',{}).get('prepared_size')!=768
                         or not isinstance(value.get('prepared_cache_receipt'),dict)))
                or (value.get('candidate_profile') in (NO_MODEL_STOCH_SMALL_BINDING,NO_MODEL_STOCH_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('dropout'),value.get('job',{}).get('stochastic_depth'))!=(0.,0.))
                or (value.get('candidate_profile') not in (LION_SMALL_BINDING, LION_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('optimizer_policy') is not None
                         or value.get('candidate',{}).get('optimizer_policy') is not None))
                or (value.get('candidate_profile') in (LION_SMALL_BINDING, LION_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('optimizer_policy') != LION_RULE
                         or value.get('candidate',{}).get('optimizer_policy') != LION_RULE
                         or value.get('job',{}).get('loss') != PAIR_LOGISTIC_LOSS
                         or (value.get('job',{}).get('backbone_lr'),value.get('job',{}).get('head_lr'),
                             value.get('job',{}).get('weight_decay')) != (1e-5,1e-4,1e-3)
                         or any(value.get('job',{}).get(k) is not None for k in
                             ('gradient_centralization_policy','gradient_clip_policy','head_warmup_policy','layerwise_lr_policy',
                              'low_rank_policy','stem_xy_policy','mixstyle_policy','tri3_downsample_policy','output_regularization',
                              'attention_pool','stage3_residual','freeze_features_before','training_views','training_rotation',
                              'training_photometric','head_seed','bn_statistics','input_preprocessing'))
                         or 'final_downsample_stride' in value.get('job',{})))
                or (value.get('candidate_profile') != RAW_GC_SMALL_BINDING
                    and (value.get('job',{}).get('gradient_centralization_policy') is not None
                         or value.get('candidate',{}).get('gradient_centralization_policy') is not None))
                or (value.get('candidate_profile') == RAW_GC_SMALL_BINDING
                    and (value.get('job',{}).get('gradient_centralization_policy') != GRADIENT_CENTRALIZATION_RULE
                         or value.get('candidate',{}).get('gradient_centralization_policy') != value.get('job',{}).get('gradient_centralization_policy')
                         or value.get('job',{}).get('loss') != PAIR_LOGISTIC_LOSS
                         or any(value.get('job',{}).get(k) is not None for k in
                            ('gradient_clip_policy','head_warmup_policy','layerwise_lr_policy','low_rank_policy',
                             'stem_xy_policy','mixstyle_policy','tri3_downsample_policy','output_regularization',
                             'attention_pool','stage3_residual','freeze_features_before','training_views',
                             'training_rotation','training_photometric','head_seed','bn_statistics'))
                         or 'final_downsample_stride' in value.get('job',{})))
                or (value.get('candidate_profile') != STEM_XY_SMALL_BINDING
                    and (value.get('job',{}).get('stem_xy_policy') is not None
                         or value.get('candidate',{}).get('stem_xy_policy') is not None))
                or (value.get('candidate_profile') == STEM_XY_SMALL_BINDING
                    and (value.get('job',{}).get('stem_xy_policy') != STEM_XY_RULE
                         or value.get('candidate',{}).get('stem_xy_policy') != value.get('job',{}).get('stem_xy_policy')
                         or value.get('job',{}).get('loss') != PAIR_LOGISTIC_LOSS
                         or any(value.get('job',{}).get(k) is not None for k in
                             ('mixstyle_policy','tri3_downsample_policy','output_regularization','low_rank_policy','layerwise_lr_policy',
                              'attention_pool','stage3_residual','freeze_features_before','head_warmup_policy','gradient_clip_policy',
                              'training_views','training_rotation','training_photometric','head_seed','bn_statistics'))
                         or 'final_downsample_stride' in value.get('job',{})))
                or (value.get('candidate_profile') != NO_AGE_SMALL_BINDING
                    and value.get('age_enabled', True) is not True)
                or (value.get('candidate_profile') == NO_AGE_SMALL_BINDING
                    and (value.get('age_enabled') is not False
                         or value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or value.get('job', {}).get('training_photometric') is not None
                         or value.get('job', {}).get('training_views') is not None))
                or (value.get('candidate_profile') == CONTENT_POOL_SMALL_BINDING
                    and (value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or value.get('job', {}).get('training_photometric') is not None
                         or value.get('job', {}).get('training_views') is not None
                         or value.get('candidate', {}).get('attention_pool') != value.get('job', {}).get('attention_pool')))
                or (value.get('candidate_profile') == STAGE3_RESIDUAL_SMALL_BINDING
                    and (value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or value.get('job', {}).get('training_photometric') is not None
                         or value.get('job', {}).get('training_views') is not None
                         or value.get('job', {}).get('attention_pool') is not None
                         or value.get('candidate', {}).get('stage3_residual') != value.get('job', {}).get('stage3_residual')))
                or (value.get('candidate_profile') == ROLEWISE_CLIP_SMALL_BINDING
                    and (value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or value.get('job', {}).get('training_photometric') is not None
                         or value.get('job', {}).get('training_views') is not None
                         or value.get('job', {}).get('attention_pool') is not None
                         or value.get('job', {}).get('stage3_residual') is not None
                         or value.get('job', {}).get('gradient_clip_policy') != ROLEWISE_CLIP_RULE
                         or value.get('candidate', {}).get('gradient_clip_policy') != value.get('job', {}).get('gradient_clip_policy')))
                or (value.get('candidate_profile') in (PAIR_SIGMOID_SMALL_BINDING,
                                                       PAIR_SIGMOID_SMALL_SEED27_BINDING)
                    and value.get('job', {}).get('loss') != PAIR_SIGMOID_LOSS)
                or (value.get('candidate_profile') in (AUC_MARGIN_SMALL_BINDING,AUC_MARGIN_SMALL_SEED27_BINDING)
                    and (value.get('job',{}).get('loss') != PAIR_AUC_MARGIN_LOSS
                         or (value['job'].get('image_size'),value['job'].get('prepared_size'),value['job'].get('seed'))!=(512,512,20261027 if value['candidate_profile']==AUC_MARGIN_SMALL_SEED27_BINDING else 20261029)
                         or (value['job'].get('effective_batch'),value['job'].get('positive_per_batch'))!=(64,8)
                         or any(value['job'].get(key) is not None for key in
                                ('input_preprocessing','optimizer_policy','gradient_centralization_policy','mixstyle_policy',
                                 'tri3_downsample_policy','low_rank_policy','layerwise_lr_policy','output_regularization',
                                 'head_warmup_policy','gradient_clip_policy','attention_pool','stage3_residual',
                                 'training_views','training_rotation','training_photometric','freeze_features_before','final_downsample_stride'))))
                or (value.get('candidate_profile') in (HARD_POSITIVE_SMALL_BINDING,HARD_POSITIVE_SMALL_SEED27_BINDING)
                    and (value.get('job', {}).get('loss') != PAIR_HARD_POSITIVE_LOSS
                         or (value['job'].get('image_size'),value['job'].get('prepared_size')) != (512,512)
                         or (value['job'].get('effective_batch'),value['job'].get('positive_per_batch')) != (64,8)
                         or value['job'].get('seed') != (20261027 if value['candidate_profile']==HARD_POSITIVE_SMALL_SEED27_BINDING else 20261029)
                         or any(value['job'].get(key) is not None for key in
                                ('input_preprocessing','optimizer_policy','gradient_centralization_policy',
                                 'mixstyle_policy','tri3_downsample_policy','low_rank_policy','layerwise_lr_policy',
                                 'output_regularization','head_warmup_policy','gradient_clip_policy','attention_pool',
                                 'stage3_residual','training_views','training_rotation','training_photometric',
                                 'freeze_features_before','final_downsample_stride'))))
                or (value.get('candidate_profile') == PHOTO_IDENTITY_SMALL_BINDING
                    and (value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or value.get('job', {}).get('training_photometric') != 'identity-v1'))
                or (value.get('candidate_profile') == ROTATION_IDENTITY_SMALL_BINDING
                    and (value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or value.get('job', {}).get('training_rotation') != 'identity-v1'
                         or value.get('candidate', {}).get('training_rotation') != 'identity-v1'
                         or value.get('job', {}).get('training_photometric') is not None
                         or value.get('job', {}).get('training_views') is not None
                         or value.get('job', {}).get('gradient_clip_policy') is not None
                         or value.get('job', {}).get('stage3_residual') is not None
                         or value.get('job', {}).get('attention_pool') is not None))
                or (value.get('candidate_profile') == HEAD_WARMUP_SMALL_BINDING
                    and (value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or value.get('job', {}).get('head_warmup_policy') != HEAD_WARMUP_RULE
                         or value.get('candidate', {}).get('head_warmup_policy') != HEAD_WARMUP_RULE
                         or value.get('job', {}).get('training_rotation') is not None
                         or value.get('job', {}).get('training_photometric') is not None
                         or value.get('job', {}).get('training_views') is not None
                         or value.get('job', {}).get('gradient_clip_policy') is not None
                         or value.get('job', {}).get('stage3_residual') is not None
                         or value.get('job', {}).get('attention_pool') is not None))
                or (value.get('candidate_profile') == SMALL_22K_ONLY_BINDING
                    and (value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or any(value.get('job', {}).get(key) is not None for key in
                                ('head_warmup_policy', 'training_rotation', 'training_photometric',
                                 'training_views', 'gradient_clip_policy', 'stage3_residual',
                                 'attention_pool'))
                         or value.get('public_source', {}).get('format') !=
                            'torchvision-small22k-only-full-head-state-dict-v1'
                         or value.get('public_source', {}).get('num_classes') != 21841))
                or (value.get('candidate_profile') in (LAYERWISE_LR_SMALL_BINDING,
                                                       LAYERWISE_LR_SMALL_R1_BINDING)
                    and (value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or value.get('job', {}).get('layerwise_lr_policy') != LAYERWISE_LR_RULE
                         or value.get('candidate', {}).get('layerwise_lr_policy') != LAYERWISE_LR_RULE
                         or any(value.get('job', {}).get(key) is not None for key in
                                ('head_warmup_policy', 'training_rotation', 'training_photometric',
                                 'training_views', 'gradient_clip_policy', 'stage3_residual',
                                 'attention_pool'))
                         or value.get('public_source', {}).get('format') != 'torchvision-1000-state-dict-v1'))
                or (value.get('candidate_profile') == POINTWISE_LORA8_SMALL_BINDING
                    and (value.get('job', {}).get('low_rank_policy') != LOW_RANK_RULE
                         or value.get('candidate', {}).get('low_rank_policy') != LOW_RANK_RULE
                         or value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or any(value.get('job', {}).get(key) is not None for key in
                                ('layerwise_lr_policy', 'head_warmup_policy', 'gradient_clip_policy',
                                 'attention_pool', 'stage3_residual', 'freeze_features_before',
                                 'final_downsample_stride', 'training_rotation', 'training_photometric',
                                 'training_views'))
                         or value.get('public_source', {}).get('format') != 'torchvision-1000-state-dict-v1'))
                or (value.get('candidate_profile') != LOGIT_SQUARE_SMALL_BINDING
                    and (value.get('job', {}).get('output_regularization') is not None
                         or value.get('candidate', {}).get('output_regularization') is not None))
                or (value.get('candidate_profile') != TRI3_DOWNSAMPLE_SMALL_BINDING
                    and (value.get('job', {}).get('tri3_downsample_policy') is not None
                         or value.get('candidate', {}).get('tri3_downsample_policy') is not None))
                or (value.get('candidate_profile') not in (MIXSTYLE_SMALL_BINDING, MIXSTYLE_SMALL_R1_BINDING)
                    and (value.get('job', {}).get('mixstyle_policy') is not None
                         or value.get('candidate', {}).get('mixstyle_policy') is not None))
                or (value.get('candidate_profile') in (MIXSTYLE_SMALL_BINDING, MIXSTYLE_SMALL_R1_BINDING)
                    and (value.get('job', {}).get('mixstyle_policy') != MIXSTYLE_RULE
                         or value.get('candidate', {}).get('mixstyle_policy') != MIXSTYLE_RULE
                         or value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or 'final_downsample_stride' in value.get('job', {})
                         or any(value.get('job', {}).get(key) is not None for key in
                                ('tri3_downsample_policy', 'output_regularization', 'low_rank_policy',
                                 'layerwise_lr_policy', 'mixup', 'head_warmup_policy',
                                 'gradient_clip_policy', 'attention_pool', 'stage3_residual',
                                 'freeze_features_before', 'training_rotation',
                                 'training_photometric', 'training_views'))
                         or value.get('public_source', {}).get('format') != 'torchvision-1000-state-dict-v1'))
                or (value.get('candidate_profile') == TRI3_DOWNSAMPLE_SMALL_BINDING
                    and (value.get('job', {}).get('tri3_downsample_policy') != TRI3_RULE
                         or value.get('candidate', {}).get('tri3_downsample_policy') != TRI3_RULE
                         or value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or 'final_downsample_stride' in value.get('job', {})
                         or any(value.get('job', {}).get(key) is not None for key in
                                ('output_regularization', 'low_rank_policy', 'layerwise_lr_policy',
                                 'head_warmup_policy', 'gradient_clip_policy', 'attention_pool',
                                 'stage3_residual', 'freeze_features_before', 'training_rotation',
                                 'training_photometric', 'training_views'))
                         or value.get('public_source', {}).get('format') != 'torchvision-1000-state-dict-v1'))
                or (value.get('candidate_profile') == LOGIT_SQUARE_SMALL_BINDING
                    and (value.get('job', {}).get('output_regularization') != OUTPUT_REGULARIZATION_RULE
                         or value.get('candidate', {}).get('output_regularization') != OUTPUT_REGULARIZATION_RULE
                         or value.get('job', {}).get('loss') != PAIR_LOGISTIC_LOSS
                         or any(value.get('job', {}).get(key) is not None for key in
                                ('low_rank_policy', 'layerwise_lr_policy', 'head_warmup_policy',
                                 'gradient_clip_policy', 'attention_pool', 'stage3_residual',
                                 'freeze_features_before', 'final_downsample_stride',
                                 'training_rotation', 'training_photometric', 'training_views'))
                         or value.get('public_source', {}).get('format') != 'torchvision-1000-state-dict-v1'))
                or any(type(value['job'].get(key)) not in (int, float)
                       or not math.isfinite(value['job'][key]) or value['job'][key] <= 0
                       for key in ('backbone_lr', 'head_lr'))):
            raise ValueError('science configuration profile or configured LR differs')
        return value
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('complete pinned science configuration required') from error


def candidate_from_configuration(value):
    actual = value['candidate']
    source = value['public_source']
    expected = {**CANDIDATE, 'stochastic_depth': value['job']['stochastic_depth']}
    if value['candidate_profile'] == FIT_BN_BINDING_VERSION:
        expected.update(id='r112-resnet50-imagenetv2-24k-p4-seed20261029-fitbn-9010',
                        experiment_order=22, bn_initialization='fit_fixed_reference')
    elif value['candidate_profile'] == DINO_R50_BINDING:
        expected.update(id='r118-resnet50-dino-imagenet-fixedbn-24k-p4-seed20261029-9010',
                        experiment_order=28, public_revision=source['revision'],
                        bn_initialization='public')
    elif value['candidate_profile'] in (FROZEN_SMALL_BINDING, LAST_STRIDE_SMALL_BINDING,
                                        FIT_SSL_SMALL_BINDING, PAIR_SIGMOID_SMALL_BINDING,
                                        PAIR_SIGMOID_SMALL_SEED27_BINDING,
                                        PHOTO_IDENTITY_SMALL_BINDING, NO_AGE_SMALL_BINDING,
                                        CONTENT_POOL_SMALL_BINDING, STAGE3_RESIDUAL_SMALL_BINDING,
                                        ROLEWISE_CLIP_SMALL_BINDING, ROTATION_IDENTITY_SMALL_BINDING,
                                        HEAD_WARMUP_SMALL_BINDING, SMALL_22K_ONLY_BINDING,
                                        LAYERWISE_LR_SMALL_BINDING,
                                        LAYERWISE_LR_SMALL_R1_BINDING,
                                        POINTWISE_LORA8_SMALL_BINDING, LOGIT_SQUARE_SMALL_BINDING,
                                        TRI3_DOWNSAMPLE_SMALL_BINDING, MIXSTYLE_SMALL_BINDING,
                                        MIXSTYLE_SMALL_R1_BINDING, STEM_XY_SMALL_BINDING, RAW_GC_SMALL_BINDING, LION_SMALL_BINDING,
                                        LION_SMALL_SEED27_BINDING, LOCAL_CONTRAST_SMALL_BINDING, LOCAL_CONTRAST_SMALL_SEED27_BINDING,BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_BINDING,NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING,GLOBAL_NATIVE_INSET_SMALL_BINDING,
                                        GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING,NO_MODEL_STOCH_SMALL_BINDING,
                                        NO_MODEL_STOCH_SMALL_SEED27_BINDING,GLOBAL_NATIVE_SIX_SMALL_BINDING,
                                        GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING,GLOBAL_768_SMALL_BINDING,
                                        GLOBAL_768_SMALL_SEED27_BINDING,HARD_POSITIVE_SMALL_BINDING,
                                        HARD_POSITIVE_SMALL_SEED27_BINDING,SMALL_384_BINDING,
                                        SMALL_384_SEED27_BINDING,AUC_MARGIN_SMALL_BINDING,
                                        AUC_MARGIN_SMALL_SEED27_BINDING,COSINE_HEAD_SMALL_BINDING,COSINE_HEAD_SMALL_SEED27_BINDING,
                                        ADAMW_BETA2_SMALL_BINDING,ADAMW_BETA2_SMALL_SEED27_BINDING,
                                        QUADRANT_MIL_SMALL_BINDING,QUADRANT_MIL_SMALL_SEED27_BINDING):
        expected = {'id': 'r113-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-freezeearly-9010',
            'experiment_order': 23, 'model': 'convnext_small',
            'public_revision': 'fe6b358aa409414cdb522ae3d43a2c9922f7cc1b',
            'readout': 'native_avgpool_final_LN_dropout_linear_plus_age', 'features': 768,
            'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6, 'batch_norm_modules': 0,
            'batch_norm_buffers': 0, 'attention_pool': None,
            'stochastic_depth': value['job']['stochastic_depth'],
            'head_initialization': 'torch.nn.Linear_native', 'head_lr_scope': 'last_linear'}
        if value['candidate_profile'] == LAST_STRIDE_SMALL_BINDING:
            expected.update(id='r114-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-laststride1-9010',
                            experiment_order=24)
        if value['candidate_profile'] == FIT_SSL_SMALL_BINDING:
            expected.update(id='r115-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-fitssl1600-9010',
                            experiment_order=25)
        if value['candidate_profile'] == PAIR_SIGMOID_SMALL_BINDING:
            expected.update(id='r120-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairsigmoid-9010',
                            experiment_order=30)
        if value['candidate_profile'] == PAIR_SIGMOID_SMALL_SEED27_BINDING:
            expected.update(id='r121-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairsigmoid-9010',
                            experiment_order=31)
        if value['candidate_profile'] == PHOTO_IDENTITY_SMALL_BINDING:
            expected.update(id='r122-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-photoidentity-9010',
                            experiment_order=32)
        if value['candidate_profile'] == NO_AGE_SMALL_BINDING:
            expected.update(id='r123-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-noage-9010',
                            experiment_order=33, readout='native_avgpool_final_LN_dropout_linear')
        if value['candidate_profile'] == CONTENT_POOL_SMALL_BINDING:
            from .attention_pool import ATTENTION_POOL_RULE
            expected.update(id='r124-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-contentpool-9010',
                            experiment_order=34,
                            readout='native_content_attention_pool_final_LN_dropout_linear_plus_age',
                            attention_pool=ATTENTION_POOL_RULE)
        if value['candidate_profile'] == STAGE3_RESIDUAL_SMALL_BINDING:
            from .stage3_residual import STAGE3_RESIDUAL_RULE
            expected.update(id='r125-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-stage3residual-9010',
                            experiment_order=35,
                            readout='native_final_LN_dropout_linear_plus_stage3_gap_residual_plus_age',
                            stage3_residual=STAGE3_RESIDUAL_RULE)
        if value['candidate_profile'] == ROLEWISE_CLIP_SMALL_BINDING:
            from .gradient_clip import ROLEWISE_CLIP_RULE
            expected.update(id='r126-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-roleclip1-9010',
                            experiment_order=36, gradient_clip_policy=ROLEWISE_CLIP_RULE)
        if value['candidate_profile'] == ROTATION_IDENTITY_SMALL_BINDING:
            expected.update(id='r127-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-rotationidentity-9010',
                            experiment_order=37, training_rotation='identity-v1')
        if value['candidate_profile'] == HEAD_WARMUP_SMALL_BINDING:
            from .head_warmup import HEAD_WARMUP_RULE
            expected.update(id='r128-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-headwarmup1600-9010',
                            experiment_order=38, head_warmup_policy=HEAD_WARMUP_RULE)
        if value['candidate_profile'] == SMALL_22K_ONLY_BINDING:
            expected.update(id='r129-convnext-small-imagenet22k-only-24k-p4-seed20261029-scale085-pairlog-9010',
                            experiment_order=39, public_revision=source['revision'])
        if value['candidate_profile'] == LAYERWISE_LR_SMALL_BINDING:
            from .layerwise_lr import LAYERWISE_LR_RULE
            expected.update(id='r130-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010',
                            experiment_order=40, layerwise_lr_policy=LAYERWISE_LR_RULE)
        if value['candidate_profile'] == LAYERWISE_LR_SMALL_R1_BINDING:
            from .layerwise_lr import LAYERWISE_LR_RULE
            expected.update(id='r130r1-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010',
                            experiment_order=41, layerwise_lr_policy=LAYERWISE_LR_RULE)
        if value['candidate_profile'] == POINTWISE_LORA8_SMALL_BINDING:
            from .low_rank import LOW_RANK_RULE
            expected.update(id='r131-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-pointwise-lora8-9010',
                            experiment_order=42, low_rank_policy=LOW_RANK_RULE)
        if value['candidate_profile'] == LOGIT_SQUARE_SMALL_BINDING:
            from .classification_loss import OUTPUT_REGULARIZATION_RULE
            expected.update(id='r132-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-logitsquare01-9010',
                            experiment_order=43, output_regularization=OUTPUT_REGULARIZATION_RULE)
        if value['candidate_profile'] == TRI3_DOWNSAMPLE_SMALL_BINDING:
            from .tri3_downsample import TRI3_RULE
            expected.update(id='r133-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-tri3downsample-9010',
                            experiment_order=44, tri3_downsample_policy=TRI3_RULE)
        if value['candidate_profile'] == MIXSTYLE_SMALL_BINDING:
            from .mixstyle import MIXSTYLE_RULE
            expected.update(id='r134-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-mixstyle13-local8-9010',
                            experiment_order=45, mixstyle_policy=MIXSTYLE_RULE)
        if value['candidate_profile'] == MIXSTYLE_SMALL_R1_BINDING:
            from .mixstyle import MIXSTYLE_RULE
            expected.update(id='r134r1-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-mixstyle13-local8-9010',
                            experiment_order=46, mixstyle_policy=MIXSTYLE_RULE)
        if value['candidate_profile'] == STEM_XY_SMALL_BINDING:
            from .models import STEM_XY_RULE
            expected.update(id='r135-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-stem-rgbxy-9010',
                            experiment_order=47,stem_xy_policy=STEM_XY_RULE)
        if value['candidate_profile'] == RAW_GC_SMALL_BINDING:
            from .config import GRADIENT_CENTRALIZATION_RULE
            expected.update(id='r136-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-rawgc-convlinear-9010',
                            experiment_order=48,gradient_centralization_policy=GRADIENT_CENTRALIZATION_RULE)
        if value['candidate_profile'] == LION_SMALL_BINDING:
            from .config import LION_RULE
            expected.update(id='r137-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-lion10x-9010',
                            experiment_order=49,optimizer_policy=LION_RULE)
        if value['candidate_profile'] == LION_SMALL_SEED27_BINDING:
            from .config import LION_RULE
            expected.update(id='r138-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-lion10x-9010',
                            experiment_order=50,optimizer_policy=LION_RULE)
        if value['candidate_profile'] == LOCAL_CONTRAST_SMALL_BINDING:
            from .config import LOCAL_CONTRAST_RULE
            expected.update(id='r139-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-localcontrast10-9010',
                            experiment_order=51,input_preprocessing=LOCAL_CONTRAST_RULE)
        if value['candidate_profile'] == LOCAL_CONTRAST_SMALL_SEED27_BINDING:
            from .config import LOCAL_CONTRAST_RULE
            expected.update(id='r140-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-localcontrast10-9010',
                            experiment_order=52,input_preprocessing=LOCAL_CONTRAST_RULE)
        if value['candidate_profile']==BRIGHT_CENTER_SMALL_BINDING:
            from .config import BRIGHT_CENTER_RULE
            expected.update(id='r143-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-brightcenter-half-9010',
                            experiment_order=55,input_preprocessing=BRIGHT_CENTER_RULE)
        if value['candidate_profile']==NATIVE_BRIGHT_CENTER_SMALL_BINDING:
            from .config import NATIVE_BRIGHT_CENTER_RULE
            expected.update(id='r144-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-native-brightcenter-half-9010',
                            experiment_order=56,input_preprocessing=NATIVE_BRIGHT_CENTER_RULE)
        if value['candidate_profile']==NATIVE_BRIGHT_CENTER_SMALL_SEED27_BINDING:
            from .config import NATIVE_BRIGHT_CENTER_RULE
            expected.update(id='r145-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-native-brightcenter-half-9010',
                            experiment_order=57,input_preprocessing=NATIVE_BRIGHT_CENTER_RULE)
        if value['candidate_profile']==GLOBAL_NATIVE_INSET_SMALL_BINDING:
            from .config import GLOBAL_NATIVE_INSET_RULE
            expected.update(id='r146-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-inset256-9010',
                            experiment_order=58,input_preprocessing=GLOBAL_NATIVE_INSET_RULE)
        if value['candidate_profile']==GLOBAL_NATIVE_INSET_SMALL_SEED27_BINDING:
            from .config import GLOBAL_NATIVE_INSET_RULE
            expected.update(id='r147-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-inset256-9010',
                            experiment_order=59,input_preprocessing=GLOBAL_NATIVE_INSET_RULE)
        if value['candidate_profile']==NO_MODEL_STOCH_SMALL_BINDING:
            expected.update(id='r148-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-nostoch-9010',
                            experiment_order=60,stochastic_depth=0.,head_dropout=0.)
        if value['candidate_profile']==NO_MODEL_STOCH_SMALL_SEED27_BINDING:
            expected.update(id='r149-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-nostoch-9010',
                            experiment_order=61,stochastic_depth=0.,head_dropout=0.)
        if value['candidate_profile']==GLOBAL_NATIVE_SIX_SMALL_BINDING:
            from .config import GLOBAL_NATIVE_SIX_RULE
            expected.update(id='r150-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-native6-9010',
                            experiment_order=62,input_preprocessing=GLOBAL_NATIVE_SIX_RULE)
        if value['candidate_profile']==GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING:
            from .config import GLOBAL_NATIVE_SIX_RULE
            expected.update(id='r151-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-native6-9010',
                            experiment_order=63,input_preprocessing=GLOBAL_NATIVE_SIX_RULE)
        if value['candidate_profile']==GLOBAL_768_SMALL_BINDING:
            expected.update(id='r152-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-global768-9010',
                            experiment_order=64)
        if value['candidate_profile']==GLOBAL_768_SMALL_SEED27_BINDING:
            expected.update(id='r153-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-global768-9010',
                            experiment_order=65)
        if value['candidate_profile']==HARD_POSITIVE_SMALL_BINDING:
            expected.update(id='r156-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-positive-top2-mix05-9010',
                            experiment_order=68)
        if value['candidate_profile']==HARD_POSITIVE_SMALL_SEED27_BINDING:
            expected.update(id='r157-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-positive-top2-mix05-9010',
                            experiment_order=69)
        if value['candidate_profile']==AUC_MARGIN_SMALL_BINDING:
            expected.update(id='r160-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-batch-aucmargin1-9010',experiment_order=72)
        if value['candidate_profile']==AUC_MARGIN_SMALL_SEED27_BINDING:
            expected.update(id='r161-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-batch-aucmargin1-9010',experiment_order=73)
        if value['candidate_profile'] in (COSINE_HEAD_SMALL_BINDING,COSINE_HEAD_SMALL_SEED27_BINDING):
            from .config import COSINE_HEAD_RULE
            expected.update(id='r162-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-cosine16-9010',
                            experiment_order=74,readout='native_avgpool_final_LN_dropout_cosine16_plus_age',
                            cosine_head_policy=COSINE_HEAD_RULE)
            if value['candidate_profile']==COSINE_HEAD_SMALL_SEED27_BINDING:
                expected.update(id='r163-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-cosine16-9010',experiment_order=75)
        if value['candidate_profile'] in (SMALL_384_BINDING,SMALL_384_SEED27_BINDING):
            expected.update(id='r158-convnext-small-imagenet22k-to-1k-public384-24k-p4-seed20261029-scale085-pairlog-9010',
                            experiment_order=70,public_revision='14a837353b6ffeced1c635dc5c8c799356e316f4')
            if value['candidate_profile']==SMALL_384_SEED27_BINDING:
                expected.update(id='r159-convnext-small-imagenet22k-to-1k-public384-24k-p4-seed20261027-scale085-pairlog-9010',experiment_order=71)
            if (source.get('revision')!=expected['public_revision']
                    or source.get('original_sha256')!='24f8525364a67ec20ae1ab21f45a525638950a8dc057534506b7aaee63b7e2a6'
                    or value['job'].get('loss')!=PAIR_LOGISTIC_LOSS
                    or (value['job'].get('image_size'),value['job'].get('prepared_size'),value['job'].get('seed'))!=(512,512,20261027 if value['candidate_profile']==SMALL_384_SEED27_BINDING else 20261029)):
                raise ValueError('fixed Small public384 source or original512 recipe differs')
        if value['candidate_profile'] in (QUADRANT_MIL_SMALL_BINDING,QUADRANT_MIL_SMALL_SEED27_BINDING):
            from .config import QUADRANT_MIL_RULE
            expected.update(id='r172-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-corner288max05-9010',
                experiment_order=84,readout='native_global512_half_corner288_max_half_plus_age',quadrant_mil_policy=QUADRANT_MIL_RULE)
            if value['candidate_profile']==QUADRANT_MIL_SMALL_SEED27_BINDING:
                expected.update(id='r173-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-corner288max05-9010',experiment_order=85)
            if (source.get('revision')!='fe6b358aa409414cdb522ae3d43a2c9922f7cc1b'
                    or source.get('sha256')!='7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610'
                    or value['job'].get('loss')!=PAIR_LOGISTIC_LOSS
                    or (value['job'].get('model'),value['job'].get('seed'),value['job'].get('image_size'),value['job'].get('prepared_size'),
                        value['job'].get('effective_batch'),value['job'].get('positive_per_batch'))!=('convnext_small',20261027 if value['candidate_profile']==QUADRANT_MIL_SMALL_SEED27_BINDING else 20261029,512,512,64,8)
                    or any(value['job'].get(k)is not None for k in ('adamw_beta2_policy','deit_pool_policy','cosine_head_policy',
                        'input_preprocessing','optimizer_policy','gradient_centralization_policy','gradient_clip_policy',
                        'head_warmup_policy','layerwise_lr_policy','low_rank_policy','stem_xy_policy','mixstyle_policy',
                        'tri3_downsample_policy','output_regularization','attention_pool','stage3_residual',
                        'freeze_features_before','final_downsample_stride','training_views','training_rotation','training_photometric'))):
                raise ValueError('fixed original Small shared global/corner image readout differs')
        if value['candidate_profile'] in (ADAMW_BETA2_SMALL_BINDING,ADAMW_BETA2_SMALL_SEED27_BINDING):
            from .config import ADAMW_BETA2_RULE
            expected.update(id='r168-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-adamw-beta2-095-9010',
                experiment_order=80,adamw_beta2_policy=ADAMW_BETA2_RULE)
            if value['candidate_profile']==ADAMW_BETA2_SMALL_SEED27_BINDING:
                expected.update(id='r169-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-adamw-beta2-095-9010',experiment_order=81)
            if ((value['job'].get('seed'),value['job'].get('image_size'),value['job'].get('prepared_size'),
                    value['job'].get('effective_batch'),value['job'].get('positive_per_batch'))!=(20261027 if value['candidate_profile']==ADAMW_BETA2_SMALL_SEED27_BINDING else 20261029,512,512,64,8)
                    or any(value['job'].get(k)is not None for k in ('deit_pool_policy','cosine_head_policy','input_preprocessing',
                        'optimizer_policy','gradient_centralization_policy','gradient_clip_policy','head_warmup_policy',
                        'layerwise_lr_policy','low_rank_policy','stem_xy_policy','mixstyle_policy','tri3_downsample_policy',
                        'output_regularization','attention_pool','stage3_residual','freeze_features_before','final_downsample_stride',
                        'training_views','training_rotation','training_photometric'))):
                raise ValueError('fixed original Small AdamW beta2 recipe differs')
    elif value['candidate_profile'] in (DEIT3_SMALL_BINDING,DEIT3_SMALL_SEED27_BINDING,DEIT3_MEANPATCH_BINDING,DEIT3_MEANPATCH_SEED27_BINDING):
        from .models import DEIT3_LIBRARY,DEIT3_FORMAT,DEIT3_VARIANT
        expected={'id':'r164-deit3-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-9010',
            'experiment_order':76,'model':'deit3_small','public_revision':'8d59b2810f277c19d0cdd1b1e691d93c24aab6db',
            'readout':'native_LN_CLS_dropout_linear_plus_age','features':384,'final_norm_eps':1e-6,'internal_norm_eps':1e-6,
            'batch_norm_modules':0,'batch_norm_buffers':0,'attention_pool':None,'stochastic_depth':.1,
            'head_initialization':'torch.nn.Linear_native','head_lr_scope':'last_linear'}
        if value['candidate_profile']==DEIT3_SMALL_SEED27_BINDING:
            expected.update(id='r165-deit3-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-9010',experiment_order=77)
        if value['candidate_profile'] in (DEIT3_MEANPATCH_BINDING,DEIT3_MEANPATCH_SEED27_BINDING):
            from .config import DEIT3_MEANPATCH_RULE
            expected.update(id='r166-deit3-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-meanpatch-9010',
                experiment_order=78,readout='native_final_LN_meanpatch_dropout_linear_plus_age',deit_pool_policy=DEIT3_MEANPATCH_RULE)
            if value['candidate_profile']==DEIT3_MEANPATCH_SEED27_BINDING:
                expected.update(id='r167-deit3-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-meanpatch-9010',experiment_order=79)
        if (source.get('model')!=DEIT3_VARIANT or source.get('format')!=DEIT3_FORMAT or source.get('native_library')!=DEIT3_LIBRARY
                or source.get('sha256')!='88140b2a114ec4db0e3588af052162b87fcdd1bb92d5e85dd93f7d2adc1fb526'
                or value['job'].get('loss')!=PAIR_LOGISTIC_LOSS
                or (value['job'].get('model'),value['job'].get('seed'),value['job'].get('image_size'),value['job'].get('prepared_size'))!=('deit3_small',20261027 if value['candidate_profile'] in (DEIT3_SMALL_SEED27_BINDING,DEIT3_MEANPATCH_SEED27_BINDING) else 20261029,512,512)
                or (value['job'].get('effective_batch'),value['job'].get('positive_per_batch'))!=(64,8)
                or any(value['job'].get(k)is not None for k in ('cosine_head_policy','input_preprocessing','optimizer_policy',
                    'gradient_centralization_policy','gradient_clip_policy','head_warmup_policy','layerwise_lr_policy','low_rank_policy',
                    'stem_xy_policy','mixstyle_policy','tri3_downsample_policy','output_regularization','attention_pool','stage3_residual',
                    'freeze_features_before','final_downsample_stride','training_views','training_rotation','training_photometric'))):
            raise ValueError('fixed native DeiT3 source/512/original pairlog recipe differs')
    elif value['candidate_profile'] in (BASE_BINDING,BASE_CLIP_BINDING,BASE_CLIP_SEED27_BINDING,BASE_CLIP_STORAGE16_BINDING):
        expected = {'id': 'r116-convnext-base-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-9010',
            'experiment_order': 26, 'model': 'convnext_base', 'public_revision': source['revision'],
            'readout': 'native_avgpool_final_LN_dropout_linear_plus_age', 'features': 1024,
            'final_norm_eps': 1e-6, 'internal_norm_eps': 1e-6, 'batch_norm_modules': 0,
            'batch_norm_buffers': 0, 'attention_pool': None,
            'stochastic_depth': value['job']['stochastic_depth'],
            'head_initialization': 'torch.nn.Linear_native', 'head_lr_scope': 'last_linear'}
        if value['candidate_profile'] in (BASE_CLIP_BINDING,BASE_CLIP_SEED27_BINDING,BASE_CLIP_STORAGE16_BINDING):
            expected.update(id='r170-convnext-base-clip-laion2b-in12k-in1k-24k-p4-seed20261029-scale085-pairlog-9010',experiment_order=82)
            if value['candidate_profile']==BASE_CLIP_SEED27_BINDING:
                expected.update(id='r171-convnext-base-clip-laion2b-in12k-in1k-24k-p4-seed20261027-scale085-pairlog-9010',experiment_order=83)
            if value['candidate_profile']==BASE_CLIP_STORAGE16_BINDING:
                expected.update(id='r174-convnext-base-clip-laion2b-in12k-in1k-24k-p4-seed20261029-scale085-pairlog-storage16-9010',experiment_order=86)
            if (source.get('revision')!='768f3849e9b97ac2f93191ad5fdc458dfafa25f3'
                    or source.get('original_sha256')!='ee2f8ca014aa7f8362081f4c4098757e76812f5b7a35a3b018bb7a13971bb241'
                    or source.get('training_data')!='LAION-2B CLIP pretraining -> ImageNet-12K -> ImageNet-1K fine-tuning'
                    or value['job'].get('loss')!=PAIR_LOGISTIC_LOSS
                    or (value['job'].get('model'),value['job'].get('seed'),value['job'].get('normalization'),
                        value['job'].get('image_size'),value['job'].get('prepared_size'),
                        value['job'].get('effective_batch'),value['job'].get('positive_per_batch'))!=('convnext_base',20261027 if value['candidate_profile']==BASE_CLIP_SEED27_BINDING else 20261029,'imagenet',512,512,64,8)
                    or any(value['job'].get(k)is not None for k in ('adamw_beta2_policy','deit_pool_policy','cosine_head_policy',
                        'input_preprocessing','optimizer_policy','gradient_centralization_policy','gradient_clip_policy',
                        'head_warmup_policy','layerwise_lr_policy','low_rank_policy','stem_xy_policy','mixstyle_policy',
                        'tri3_downsample_policy','output_regularization','attention_pool','stage3_residual',
                        'freeze_features_before','final_downsample_stride','training_views','training_rotation','training_photometric'))):
                raise ValueError('fixed CLIP Base public initialization/original input recipe differs')
    elif value['candidate_profile'] in (SWIN_BINDING,SWIN_V2_SMALL_BINDING,SWIN_V2_SMALL_SEED27_BINDING):
        expected = {'id': 'r117-swin-base-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-9010',
            'experiment_order': 27, 'model': 'swin_base', 'public_revision': source['revision'],
            'readout': 'native_final_LN_gap_dropout_linear_plus_age', 'features': 1024,
            'final_norm_eps': 1e-5, 'internal_norm_eps': 1e-5,
            'batch_norm_modules': 0, 'batch_norm_buffers': 0, 'static_index_buffers': 24,
            'attention_pool': None, 'stochastic_depth': value['job']['stochastic_depth'],
            'head_initialization': 'torch.nn.Linear_native', 'head_lr_scope': 'last_linear'}
        if value['candidate_profile'] in (SWIN_V2_SMALL_BINDING,SWIN_V2_SMALL_SEED27_BINDING):
            expected.update(id='r154-swin-v2-small-imagenet1k-24k-p4-seed20261029-scale085-pairlog-9010',
                            experiment_order=66,model='swin_v2_small',features=768,static_buffer_count=48)
            if value['candidate_profile']==SWIN_V2_SMALL_SEED27_BINDING:
                expected.update(id='r155-swin-v2-small-imagenet1k-24k-p4-seed20261027-scale085-pairlog-9010',
                                experiment_order=67)
            if (source.get('model')!='swinv2_small_window8_256'
                    or source.get('revision')!='db3a0c7e5d97487e7e874c943838569b84fe8e54'
                    or source.get('training_data')!='ImageNet-1K'
                    or value['job'].get('loss')!=PAIR_LOGISTIC_LOSS
                    or value['job'].get('image_size')!=512 or value['job'].get('prepared_size')!=512):
                raise ValueError('fixed Swin V2 Small public/geometry/objective differs')
    elif value['candidate_profile'] in (DINOV2_PAIRLOG_BINDING,DINOV2_PAIRLOG_SEED27_BINDING):
        expected={'id':'r141-dinov2-vits14-lvd142m-24k-p4-seed20261029-scale085-pairlog-9010',
            'experiment_order':53,'model':'dinov2_vits14','public_revision':'7764ea0f912e53c92e82eb78a2a1631e92725fc8',
            'readout':'native_LN_CLS_concat_meanpatch_dropout_linear_plus_age','features':768,
            'final_norm_eps':1e-6,'internal_norm_eps':1e-6,'batch_norm_modules':0,'batch_norm_buffers':0,
            'attention_pool':None,'stochastic_depth':.1,'head_initialization':'normal_std0.01_zero_bias',
            'head_seed':20261027,'head_lr_scope':'last_linear'}
        if value['candidate_profile']==DINOV2_PAIRLOG_SEED27_BINDING:
            expected.update(id='r142-dinov2-vits14-lvd142m-24k-p4-seed20261027-scale085-pairlog-9010',experiment_order=54)
        if (source.get('kind')!='self_supervised' or source.get('training_data')!='LVD-142M'
                or source.get('format')!='dinov2-official-backbone-state-dict-v1'
                or source.get('sha256')!='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
                or value['job'].get('head_seed')!=20261027
                or value['job'].get('loss')!=PAIR_LOGISTIC_LOSS
                or any(value['job'].get(k) is not None for k in
                    ('input_preprocessing','optimizer_policy','gradient_centralization_policy','mixstyle_policy',
                     'tri3_downsample_policy','low_rank_policy','layerwise_lr_policy','output_regularization',
                     'head_warmup_policy','gradient_clip_policy','attention_pool','stage3_residual',
                     'training_views','training_rotation','training_photometric','freeze_features_before','final_downsample_stride'))):
            raise ValueError('fixed original DINOv2 ranking source/recipe differs')
    if (any(actual.get(key) != wanted for key, wanted in expected.items())
            or actual['model'] != value['job']['model']
            or actual['weight_sha256'] != source['sha256']
            or actual['provenance_sha256'] != source['provenance_sha256']
            or actual['public_revision'] != source['revision']
            or actual['features'] != value['structure']['features']
            or (actual['model'] == 'resnet50_imagenet'
                and actual['bn_initialization'] != value['job']['bn_statistics']['kind'])):
        raise ValueError('candidate, public source and measured structure differ')
    return copy.deepcopy(actual)


def public_source():
    return copy.deepcopy(science_configuration()['public_source'])


def structure_contract():
    return copy.deepcopy(science_configuration()['structure'])


def verify_public_asset(science, site):
    """Consume the new public source handoff without altering historical migration."""
    reference = science.get('public_asset_receipt')
    if reference is None:
        if science['job']['model'] in ('convnext_small', 'convnext_base', 'swin_base', 'swin_v2_small', 'deit3_small'):
            raise ValueError('registered public backbone requires its independently verified asset')
        return None
    relative = Path(reference['path'])
    root = Path(site['work_root']).resolve()
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('public handoff must be relative to work_root')
    path = root / relative
    if path.is_symlink() or sha256_file(path) != reference['sha256']:
        raise ValueError('public handoff bytes differ')
    receipt = json.loads(path.read_text())
    source = science['public_source']
    if science['candidate_profile'] in (DEIT3_SMALL_BINDING,DEIT3_SMALL_SEED27_BINDING,DEIT3_MEANPATCH_BINDING,DEIT3_MEANPATCH_SEED27_BINDING):
        if (receipt.get('passed')is not True or receipt.get('model')!=source['model']
                or receipt.get('revision')!=source['revision'] or receipt['execution']['uid']!=site['pod']['uid']
                or receipt.get('native_weights')!={'path':site['public_weight'],'sha256':source['sha256']}
                or receipt.get('provenance')!={'path':site['public_provenance'],'sha256':source['provenance_sha256']}
                or receipt.get('public_tensor_count')!=176 or receipt.get('forward_calls')!=0
                or receipt.get('GPU_initialized')is not False or receipt.get('datasets_read')is not False):
            raise ValueError('native DeiT3 source handoff differs')
        return receipt
    if science['candidate_profile'] in (DINOV2_PAIRLOG_BINDING,DINOV2_PAIRLOG_SEED27_BINDING):
        if (receipt.get('schema_version')!=1 or receipt.get('passed') is not True
                or receipt.get('execution',{}).get('NCCCU_POD_UID')!=site['pod']['uid']
                or receipt.get('model')!='dinov2_vits14' or receipt.get('revision')!=source['revision']
                or receipt.get('converted')!={'path':site['public_weight'],'sha256':source['sha256']}
                or receipt.get('provenance')!={'path':site['public_provenance'],'sha256':source['provenance_sha256']}
                or receipt.get('source_files_sha256')!=source['source_files_sha256']):
            raise ValueError('original DINOv2 public/code asset handoff differs')
        return receipt
    if science['candidate_profile'] == DINO_R50_BINDING:
        if (receipt.get('schema_version') != 1 or receipt.get('passed') is not True
                or receipt.get('execution', {}).get('uid') != site['pod']['uid']
                or receipt.get('model') != 'resnet50'
                or receipt.get('repo') != 'facebookresearch/dino'
                or receipt.get('revision') != source['revision']
                or receipt.get('converted') != {'path': site['public_weight'], 'sha256': source['sha256']}
                or receipt.get('provenance') != {'path': site['public_provenance'],
                                                'sha256': source['provenance_sha256']}
                or receipt.get('structure') != science['structure']['source_structure']
                or receipt.get('source_weight', {}).get('sha256') != source['source_weight_receipt_sha256']
                or receipt.get('strict_load', {}).get('sha256') != source['strict_receipt_sha256']):
            raise ValueError('DINO R50 strict source handoff differs from frozen science/site')
        return receipt
    if science['job']['model'] == 'resnet50_imagenet':
        from .models import IMAGENET_RESNET
        files = receipt.get('files', {})
        if (receipt.get('passed') is not True or receipt.get('scope') != 'current_Pod_public_R50_asset_bytes'
                or receipt.get('execution', {}).get('NCCCU_POD_UID') != site['pod']['uid']
                or source['sha256'] != IMAGENET_RESNET['sha256']
                or files.get('resnet50-11ad3fa6.pth', {}).get('path') != site['public_weight']
                or files['resnet50-11ad3fa6.pth']['sha256'] != source['sha256']
                or files.get('provenance.json', {}).get('path') != site['public_provenance']
                or files['provenance.json']['sha256'] != source['provenance_sha256']):
            raise ValueError('current public ImageNet R50 asset proof differs')
        return receipt
    if science['candidate_profile'] == SMALL_22K_ONLY_BINDING:
        if (receipt.get('repo') != source['repo'] or receipt.get('tag') != source['tag']
                or receipt.get('num_classes') != source['num_classes']
                or receipt.get('strict_load', {}).get('sha256') != source['strict_receipt_sha256']):
            raise ValueError('Small 22K-only complete public source identity differs')
    if (receipt.get('passed') is not True or receipt.get('schema_version') != 1
            or receipt['execution']['uid'] != site['pod']['uid']
            or receipt['model'] != source['model']
            or receipt['revision'] != source['revision']
            or receipt['converted'] != {'path': site['public_weight'], 'sha256': source['sha256']}
            or receipt['provenance'] != {'path': site['public_provenance'], 'sha256': source['provenance_sha256']}
            or receipt['original']['sha256'] != source['original_sha256']
            or receipt['structure'] != (source['structure'] if science['candidate_profile'] == SMALL_22K_ONLY_BINDING
                                        else science['structure'])):
        raise ValueError('public source, paths or measured structure differ from handoff')
    return receipt


def candidate():
    site = load_site()
    expected = candidate_from_configuration(science_configuration())
    if (site['science'].get('candidate') != expected
            or site['science'].get('binding_version') != science_configuration()['candidate_profile']):
        raise ValueError('explicit candidate/source/LN scientific identity differs')
    return expected


def metadata():
    candidate()
    return _metadata(load_site()['metadata_root'])


@lru_cache(maxsize=4)
def _metadata(root):
    root = Path(root)
    if (root / 'FREEZE.json').is_symlink() or sha256_file(root / 'FREEZE.json') != FREEZE_SHA:
        raise ValueError('9:1 FREEZE identity differs')
    freeze = json.loads((root / 'FREEZE.json').read_text())
    values = {}
    for name in ('CONTRACT.json', 'manifest.json', 'AGE.json', 'SCHEDULE.json'):
        path = root / name
        if path.is_symlink() or sha256_file(path) != freeze['files'][name]['sha256']:
            raise ValueError('9:1 frozen artifact differs: ' + name)
        values[name] = json.loads(path.read_text())
    c = values['CONTRACT.json']
    if canonical_digest(values['manifest.json']) != c['new_manifest_canonical_sha256']:
        raise ValueError('9:1 native manifest digest differs')
    warm = schedule()['warmup_updates']
    values['EMA-TINY.json'] = {'version': f'ema-warmcopy{warm}-decay0999-nobn-v1',
        'warmcopy_updates': warm, 'decay': .999, 'dtype': 'torch.float32',
        'floating_buffers': 'same_as_parameters', 'integer_buffers': 'copy', 'batch_norm_buffers': 0}
    if candidate()['model'] == 'swin_base':
        values['EMA-TINY.json'].update(static_integer_buffers='relative_position_index',
                                      static_integer_buffer_count=24)
    if candidate()['model'] == 'swin_v2_small':
        values['EMA-TINY.json'].update(static_swin_buffers='swin_v2_small_window8',static_swin_buffer_count=48,
                                      floating_buffers='same_as_parameters_except_static_swin')
    if candidate()['model'] == 'resnet50_imagenet':
        values['EMA-TINY.json'] = {'version': f'ema-warmcopy{warm}-decay0999-frozenbn-v1',
            'warmcopy_updates': warm, 'decay': .999, 'dtype': 'torch.float32',
            'floating_buffers': 'same_as_parameters_except_frozen_bn', 'integer_buffers': 'copy',
            'frozen_bn_buffers': 'copy'}
    values['binding'] = {'version': science_configuration()['candidate_profile'], 'candidate': candidate(),
        'science_config_sha256': load_site()['science']['configuration']['sha256'],
        'freeze_sha256': FREEZE_SHA, 'manifest_sha256': c['new_manifest']['sha256'],
        'manifest_digest': c['new_manifest_canonical_sha256'], 'fit_indices_sha256': c['fit_indices_canonical_sha256'],
        'dev_indices_sha256': c['dev_indices_canonical_sha256'], 'age_contract_sha256': c['age_contract_sha256'],
        'schedule_sha256': c['schedule']['sha256'], 'ema_sha256': canonical_digest(values['EMA-TINY.json']),
        'source_identity': c['source_identity']}
    return values


def binding():
    return copy.deepcopy(metadata()['binding'])


def resolved_age_spec():
    """Keep the frozen fit-only age contract while resolving this round's readout."""
    result = copy.deepcopy(metadata()['AGE.json'])
    result['enabled'] = science_configuration().get('age_enabled', True)
    if science_configuration()['candidate_profile'] in (LION_SMALL_BINDING, LION_SMALL_SEED27_BINDING):
        from .age import validate_spec,LION_VERSION
        validate_spec(result)
        c=result['contract']
        c.update(version=LION_VERSION,lr=1e-4,weight_decay=1e-3,source_contract_sha256=result['contract_sha256'])
        result['contract_sha256']=canonical_digest(c)
        validate_spec(result)
    return result


def schedule():
    raw = science_configuration()['schedule']
    return {k: raw[k] for k in ('steps_per_cycle', 'total_updates', 'warmup_updates', 'stop_after_updates')}


def early_stopping():
    return copy.deepcopy(science_configuration()['early_stopping'])


def validate_training_completion(record):
    """A full update budget or a completed patience boundary may freeze selection."""
    configured = science_configuration()
    planned = schedule()
    updates = record['successful_updates']
    cycle = record['evaluated_cycle']
    early = record.get('early_stop', {})
    if (record.get('status') != 'completed' or record.get('skipped_updates') != 0
            or record.get('attempted_updates') != updates
            or record.get('cycle') != cycle or updates != cycle * planned['steps_per_cycle']
            or not 0 < updates <= planned['total_updates']):
        raise ValueError('completed unskipped internal-dev boundary required before selection')
    if updates < planned['total_updates'] and not (
            early.get('stopped') is True and cycle >= configured['job']['min_cycles']
            and early.get('bad_cycles', 0) >= configured['job']['patience']):
        raise ValueError('training ended before its budget without completed early stopping')
    return cycle


def ema_rule():
    return copy.deepcopy(metadata()['EMA-TINY.json'])


def validate_job(job):
    configured = science_configuration()['job']
    expected = {'model': candidate()['model'], 'image_size': configured['image_size'], 'prepared_size': configured['prepared_size'],
        'precision': 'fp32', 'normalization': 'imagenet', 'seed': configured['seed'], 'effective_batch': 64,
        'positive_per_batch': configured['positive_per_batch'], **{key: configured[key] for key in (
            'cycles', 'warmup_cycles', 'min_cycles', 'patience', 'min_delta', 'stop_after_cycles')},
        'backbone_lr': configured['backbone_lr'],
        'head_lr': configured['head_lr'], 'weight_decay': (configured['weight_decay'] if candidate().get('optimizer_policy') is not None else 1e-4), 'loss': configured.get('loss'),
        'training_scale': configured.get('training_scale'),
        'training_views': configured.get('training_views'),
        'training_photometric': configured.get('training_photometric'),
        'training_rotation': configured.get('training_rotation'),
        'freeze_features_before': configured.get('freeze_features_before'),
        'final_downsample_stride': configured.get('final_downsample_stride'),
        'dropout': configured['dropout'], 'stochastic_depth': configured['stochastic_depth'], 'head_lr_scope': 'last_linear', 'mixup': None,
        'attention_pool': configured.get('attention_pool'), 'initial_checkpoint': None,
        'stage3_residual': configured.get('stage3_residual'),
        'gradient_clip_policy': configured.get('gradient_clip_policy'),
        'head_warmup_policy': configured.get('head_warmup_policy'),
        'layerwise_lr_policy': configured.get('layerwise_lr_policy'),
        # Existing profiles have no adapter identity. A future adopted profile
        # must explicitly register the same policy in its candidate mapping.
        'low_rank_policy': candidate().get('low_rank_policy'),
        # Only the exact registered R132 candidate carries this objective.
        'output_regularization': candidate().get('output_regularization'),
        # Only the exact registered R133 candidate carries these three operators.
        'tri3_downsample_policy': candidate().get('tri3_downsample_policy'),
        # Only the exact registered candidate carries the fixed MixStyle policy.
        'mixstyle_policy': candidate().get('mixstyle_policy'),
        'stem_xy_policy': candidate().get('stem_xy_policy'),
        'gradient_centralization_policy': candidate().get('gradient_centralization_policy'),
        'optimizer_policy': candidate().get('optimizer_policy'),
        'input_preprocessing': candidate().get('input_preprocessing'),
        'cosine_head_policy': candidate().get('cosine_head_policy'),
        'deit_pool_policy': candidate().get('deit_pool_policy'),
        'adamw_beta2_policy': candidate().get('adamw_beta2_policy'),
        'quadrant_mil_policy': candidate().get('quadrant_mil_policy'),
        'frozen_bn': configured['frozen_bn'],
        'final_norm_eps': configured['final_norm_eps'], 'head_seed': configured.get('head_seed'),
        'bn_statistics': configured.get('bn_statistics')}
    init = job.get('initialization', {})
    site = load_site()
    if (any(job.get(k) != v for k, v in expected.items()) or job.get('physical_batch') not in (64 // len(site['gpus']), 32)
            or job.get('experiment91') != binding() or job.get('age') != resolved_age_spec()
            or init.get('kind') != public_source().get('kind','imagenet') or init.get('sha256') != public_source()['sha256']
            or init.get('format') != public_source()['format']
            or Path(init.get('path', '')).resolve() != Path(site['public_weight']).resolve()
            or Path(init.get('provenance', '')).resolve() != Path(site['public_provenance']).resolve()):
        raise ValueError('Tiny22K-to-1K scientific/age/source/LN contract differs')
    return job


def describe_model(model):
    import torch
    from .ema import tensor_digest
    nn = torch.nn
    if candidate()['model']=='deit3_small':
        from .models import describe_deit3
        value={'candidate':candidate(),'batch_norm_modules':sum(isinstance(m,nn.modules.batchnorm._BatchNorm)for m in model.modules()),
            'buffer_count':len(dict(model.named_buffers())),'buffer_sha256':tensor_digest(dict(model.named_buffers())),
            'layer_norms':[{'name':n,'eps':float(m.eps),'normalized_shape':list(m.normalized_shape)}for n,m in model.named_modules()if isinstance(m,nn.LayerNorm)],
            'head_in_features':384,'head_out_features':1,'dropout':model.network.classifier[0].p,
            'stochastic_depth':[{'name':f'network.blocks.{i}.drop_path{j}','p':getattr(getattr(b,f'drop_path{j}'),'drop_prob',0.),'mode':'row'}
                for i,b in enumerate(model.network.blocks)for j in (1,2)],'deit3':describe_deit3(model)}
        validate_normalization(value);return value
    if candidate()['model']=='dinov2_vits14':
        from .dinov2_model import DINOv2Binary
        if not isinstance(model,DINOv2Binary):raise ValueError('original DINOv2 binary wrapper required')
        base=model.backbone;buffers=dict(model.named_buffers());final=model.classifier[-1]
        value={'candidate':candidate(),'batch_norm_modules':sum(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()),
            'buffer_count':len(buffers),'buffer_sha256':tensor_digest(buffers),
            'layer_norms':[{'name':n,'eps':float(m.eps),'normalized_shape':list(m.normalized_shape)}
                           for n,m in model.named_modules() if isinstance(m,nn.LayerNorm)],
            'head_in_features':final.in_features,'head_out_features':final.out_features,'dropout':model.classifier[0].p,
            'stochastic_depth':[{'name':'backbone.blocks.'+str(i),'p':b.sample_drop_ratio,'mode':'row'} for i,b in enumerate(base.blocks)],
            'dino':{'patch_weight_shape':list(base.patch_embed.proj.weight.shape),'patch_size':base.patch_size,
                'embed_dim':base.embed_dim,'num_register_tokens':base.num_register_tokens,
                'pos_embed_shape':list(base.pos_embed.shape),'cls_shape':list(base.cls_token.shape),
                'interpolate_antialias':base.interpolate_antialias,'interpolate_offset':base.interpolate_offset,
                'heads':[b.attn.num_heads for b in base.blocks],
                'qkv_shapes':[list(b.attn.qkv.weight.shape) for b in base.blocks],
                'frozen_unused_token':'backbone.mask_token','mask_token_shape':list(base.mask_token.shape)}}
        validate_normalization(value)
        return value
    if candidate()['model'] in ('swin_base','swin_v2_small'):
        from torchvision.models.swin_transformer import SwinTransformerBlock, ShiftedWindowAttention, SwinTransformerBlockV2, ShiftedWindowAttentionV2
        from torchvision.ops import StochasticDepth
        from .swin_model import SwinBinary, static_index_buffers, static_swin_buffers
        v2 = candidate()['model']=='swin_v2_small'
        if not isinstance(model, SwinBinary):
            raise ValueError('registered Swin V1 binary wrapper required')
        network = model.network
        buffers = dict(model.named_buffers())
        fixed = static_swin_buffers(model,'swin_v2_small_window8') if v2 else static_index_buffers(model)
        indices = {n:t for n,t in fixed.items() if n.endswith('.relative_position_index')}
        norms = [{'name': name, 'eps': float(module.eps),
                  'normalized_shape': list(module.normalized_shape)}
                 for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
        blocks = [module for module in model.modules() if type(module) is (SwinTransformerBlockV2 if v2 else SwinTransformerBlock)]
        attentions = [module for module in model.modules() if type(module) is (ShiftedWindowAttentionV2 if v2 else ShiftedWindowAttention)]
        depth = [{'name': name, 'p': module.p, 'mode': module.mode}
                 for name, module in model.named_modules() if isinstance(module, StochasticDepth)]
        final = network.classifier[-1]
        count_bn = sum(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
        stage_widths = [network.features[0][0].out_channels,
                        *[network.features[index][0].attn.qkv.in_features for index in (3, 5, 7)]]
        value = {'candidate': candidate(), 'batch_norm_modules': count_bn,
                 'buffer_count': len(buffers), 'buffer_sha256': tensor_digest(buffers),
                 'static_index_names': sorted(indices), 'layer_norms': norms,
                 'head_in_features': final.in_features, 'head_out_features': final.out_features,
                 'dropout': network.classifier[0].p, 'stochastic_depth': depth,
                 'stage_widths': stage_widths,
                 'attention': [{'heads': item.num_heads, 'window': list(item.window_size),
                                'shift': list(item.shift_size), 'dropout': item.dropout,
                                'attention_dropout': item.attention_dropout} for item in attentions]}
        if (count_bn or set(buffers) != set(fixed) or not isinstance(network.head, nn.Identity)
                or not isinstance(network.norm, nn.LayerNorm)
                or not isinstance(network.avgpool, nn.AdaptiveAvgPool2d)
                or network.avgpool.output_size != 1
                or not isinstance(network.classifier[0], nn.Dropout)
                or not isinstance(final, nn.Linear)
                or len(blocks) != 24 or len(attentions) != 24):
            raise ValueError('actual Swin V1 architecture/static buffers differ')
        validate_normalization(value)
        return value
    if candidate()['model'] == 'resnet50_imagenet':
        from .imagenet_resnet import ImageNetResnetBinary
        if not isinstance(model, ImageNetResnetBinary):
            raise ValueError('explicit ImageNet R50 wrapper required')
        layers = model.assert_bn_mode()
        final = model.classifier[-1]
        if (not isinstance(model.backbone.avgpool, nn.AdaptiveAvgPool2d)
                or not isinstance(model.backbone.fc, nn.Identity)
                or not isinstance(final, nn.Linear) or (final.in_features, final.out_features) != (2048, 1)
                or not isinstance(model.classifier[0], nn.Dropout)
                or model.classifier[0].p != science_configuration()['job']['dropout']):
            raise ValueError('actual ImageNet R50 avgpool/head differs')
        value = {'candidate': candidate(), 'batch_norm_modules': len(layers),
            'buffer_count': len(dict(model.named_buffers())), 'buffer_sha256': tensor_digest(dict(model.named_buffers())),
            'layer_norms': [], 'head_in_features': final.in_features, 'head_out_features': 1,
            'dropout': model.classifier[0].p, 'stochastic_depth': [],
            'batch_norms': [{'name': n, 'eps': m.eps, 'training': m.training, 'track_running_stats': m.track_running_stats}
                            for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]}
        validate_normalization(value)
        return value
    network = model.network
    final = network.classifier[-1]
    norms = [{'name': n, 'eps': float(m.eps), 'normalized_shape': list(m.normalized_shape)}
             for n, m in model.named_modules() if isinstance(m, nn.LayerNorm)]
    buffers = dict(model.named_buffers())
    from torchvision.ops import StochasticDepth
    depth = [{'name': n, 'p': m.p, 'mode': m.mode} for n, m in model.named_modules() if isinstance(m, StochasticDepth)]
    count_bn = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if (count_bn != 0 or buffers or not norms
            or not isinstance(network.avgpool, nn.AdaptiveAvgPool2d)
            or network.avgpool.output_size != 1
            or not isinstance(final, nn.Linear) or (final.in_features, final.out_features) != (candidate()['features'], 1)
            or not isinstance(network.classifier[-2], nn.Dropout)
            or network.classifier[-2].p != science_configuration()['job']['dropout']
            or (science_configuration()['candidate_profile'] != CONTENT_POOL_SMALL_BINDING
                and any(n == 'attention_pool_weight' or n == 'q' for n, _ in model.named_parameters()))
            or (science_configuration()['candidate_profile'] != STAGE3_RESIDUAL_SMALL_BINDING
                and any(n == 'stage3_residual_weight' for n, _ in model.named_parameters()))
            or not any(n['name'] == 'network.classifier.0' for n in norms)
            or any(n['eps'] != (candidate()['final_norm_eps'] if n['name'] == 'network.classifier.0' else 1e-6) for n in norms)):
        raise ValueError('actual Tiny native avgpool/768/noq/noBN/LN/dropout differs')
    result = {'candidate': candidate(), 'batch_norm_modules': count_bn, 'buffer_count': len(buffers),
            'buffer_sha256': tensor_digest(buffers), 'layer_norms': norms, 'head_in_features': final.in_features,
            'head_out_features': final.out_features, 'dropout': network.classifier[-2].p, 'stochastic_depth': depth}
    if science_configuration()['job'].get('cosine_head_policy') is not None:
        from .models import describe_cosine_head
        result['cosine_head'] = describe_cosine_head(model, science_configuration()['job'])
    if science_configuration()['job'].get('quadrant_mil_policy') is not None:
        from .models import describe_quadrant_mil
        result['quadrant_mil'] = describe_quadrant_mil(model, science_configuration()['job'])
    if science_configuration()['job'].get('gradient_centralization_policy') is not None:
        result['gradient_centralization']={'policy':science_configuration()['job']['gradient_centralization_policy'],
            'targets':[{'name':name+'.weight','module':type(module).__name__,
                        'shape':list(module._parameters['weight'].shape),
                        'axes':list(range(1,module._parameters['weight'].ndim))}
                       for name,module in model.named_modules() if isinstance(module,(nn.Conv2d,nn.Linear))]}
    if science_configuration()['job'].get('stem_xy_policy') is not None:
        stem=network.features[0][0]
        if not isinstance(stem,nn.Conv2d):raise ValueError('native RGBXY stem convolution missing')
        result['stem_xy']={'policy':getattr(model,'stem_xy_policy',None),'name':'network.features.0.0',
            'weight_shape':list(stem.weight.shape),'bias_shape':list(stem.bias.shape),
            'in_channels':stem.in_channels,'out_channels':stem.out_channels,
            'kernel_size':list(stem.kernel_size),'stride':list(stem.stride),'padding':list(stem.padding),
            'dilation':list(stem.dilation),'groups':stem.groups,'coordinate_generation':'stateless'}
    from .config import GLOBAL_NATIVE_SIX_RULE
    if science_configuration()['job'].get('input_preprocessing')==GLOBAL_NATIVE_SIX_RULE:
        stem=network.features[0][0]
        if not isinstance(stem,nn.Conv2d):raise ValueError('native six-channel stem convolution missing')
        result['stem_native6']={'policy':getattr(model,'stem_native6_policy',None),'name':'network.features.0.0',
            'weight_shape':list(stem.weight.shape),'bias_shape':list(stem.bias.shape),
            'in_channels':stem.in_channels,'out_channels':stem.out_channels,'kernel_size':list(stem.kernel_size),
            'stride':list(stem.stride),'padding':list(stem.padding),'dilation':list(stem.dilation),'groups':stem.groups}
    if science_configuration()['job'].get('mixstyle_policy') is not None:
        from .mixstyle import describe_mixstyle
        result['mixstyle'] = describe_mixstyle(model, {**science_configuration()['job'],
                                                    'age': resolved_age_spec()})
    if science_configuration()['job'].get('tri3_downsample_policy') is not None:
        from .tri3_downsample import describe_tri3
        result['tri3_downsample'] = describe_tri3(model, {**science_configuration()['job'],
                                                       'age': resolved_age_spec()})
    if science_configuration()['job'].get('low_rank_policy') is not None:
        from .low_rank import describe_low_rank
        result['low_rank'] = describe_low_rank(model, {**science_configuration()['job'],
                                                      'age': resolved_age_spec()})
    if science_configuration()['candidate_profile'] == CONTENT_POOL_SMALL_BINDING:
        from .attention_pool import validate_model
        validate_model(model, science_configuration()['job'])
        weight = model.attention_pool_weight
        result['attention_pool'] = {'parameter_name': 'attention_pool_weight',
                                    'shape': list(weight.shape), 'dtype': str(weight.dtype)}
    if science_configuration()['candidate_profile'] == STAGE3_RESIDUAL_SMALL_BINDING:
        from .stage3_residual import validate_model
        validate_model(model, science_configuration()['job'])
        weight = model.stage3_residual_weight
        convolution = network.features[4][1]
        if (not isinstance(convolution, nn.Conv2d) or convolution.out_channels != 384
                or len(network.features) != 8):
            raise ValueError('stage3 Small feature stack differs')
        result['stage3_residual'] = {'parameter_name': 'stage3_residual_weight',
            'shape': list(weight.shape), 'dtype': str(weight.dtype),
            'feature_index': 5, 'channels': convolution.out_channels}
    if ('final_downsample_stride' in science_configuration()['job']
            or science_configuration()['candidate_profile'] in (FIT_SSL_SMALL_BINDING, BASE_BINDING,BASE_CLIP_BINDING,BASE_CLIP_SEED27_BINDING,BASE_CLIP_STORAGE16_BINDING)):
        convolution = network.features[6][1]
        if not isinstance(convolution, nn.Conv2d):
            raise ValueError('actual final ConvNeXt downsample must remain Conv2d')
        result['final_downsample'] = {'name': 'network.features.6.1', 'stride': list(convolution.stride),
            'kernel_size': list(convolution.kernel_size), 'padding': list(convolution.padding),
            'dilation': list(convolution.dilation)}
    if science_configuration()['candidate_profile'] in (BASE_BINDING,BASE_CLIP_BINDING,BASE_CLIP_SEED27_BINDING,BASE_CLIP_STORAGE16_BINDING):
        result['stage_widths'] = [network.features[0][0].out_channels,
            network.features[2][1].out_channels, network.features[4][1].out_channels,
            network.features[6][1].out_channels]
    validate_normalization(result)
    return result


def validate_normalization(value):
    import hashlib
    if candidate()['model']=='deit3_small':
        import torch
        from .models import DEIT3_LIBRARY
        policy=candidate().get('deit_pool_policy')
        native=value.get('deit3',{});norms=value.get('layer_norms',[]);depth=value.get('stochastic_depth',[])
        rates=[float(t)for t in torch.linspace(0,.1,12)]
        if (value.get('candidate')!=candidate()or value.get('batch_norm_modules')!=0 or value.get('buffer_count')!=0
                or value.get('buffer_sha256')!=hashlib.sha256(b'').hexdigest()
                or value.get('head_in_features')!=384 or value.get('head_out_features')!=1 or value.get('dropout')!=.2
                or len(norms)!=25 or any(n['eps']!=1e-6 or n['normalized_shape']!=[384]for n in norms)
                or len(depth)!=24 or any(r['p']!=rates[i//2]or r['mode']!='row'for i,r in enumerate(depth))
                or native.get('native_library')!=DEIT3_LIBRARY or native.get('image_size')!=[512,512]
                or native.get('pos_embed_shape')!=[1,1024,384]or native.get('patch_weight_shape')!=[384,3,16,16]
                or native.get('global_pool')!=('avg'if policy is not None else'token')or native.get('no_embed_class')is not True
                or native.get('pool_policy')!=policy
                or (policy is not None and native.get('pool_include_prefix')is not False)
                or len(native.get('blocks',[]))!=12 or any(b['heads']!=6 or b['qkv_shape']!=[1152,384]or not b['fused_attn']for b in native['blocks'])):
            raise ValueError('actual native DeiT3 CLS/positions/normalization differs')
        return True
    if candidate()['model']=='dinov2_vits14':
        norms=value.get('layer_norms',[]);depth=value.get('stochastic_depth',[])
        expected={'patch_weight_shape':[384,3,14,14],'patch_size':14,'embed_dim':384,'num_register_tokens':0,
            'pos_embed_shape':[1,1370,384],'cls_shape':[1,1,384],'interpolate_antialias':False,'interpolate_offset':.1,
            'heads':[6]*12,'qkv_shapes':[[1152,384]]*12,'frozen_unused_token':'backbone.mask_token','mask_token_shape':[1,384]}
        if (value.get('candidate')!=candidate() or value.get('batch_norm_modules')!=0
                or value.get('buffer_count')!=0 or value.get('buffer_sha256')!=hashlib.sha256(b'').hexdigest()
                or value.get('head_in_features')!=768 or value.get('head_out_features')!=1 or value.get('dropout')!=.2
                or len(norms)!=25 or len({n['name'] for n in norms})!=25
                or not any(n['name']=='backbone.norm' for n in norms)
                or any(n['eps']!=1e-6 or n['normalized_shape']!=[384] for n in norms)
                or depth!=[{'name':'backbone.blocks.'+str(i),'p':.1 if i==11 else (.1/11)*i,'mode':'row'} for i in range(12)]
                or value.get('dino')!=expected):
            raise ValueError('original DINOv2 no-register/LN/CLS/patch/head graph differs')
        return True
    if candidate()['model'] in ('swin_base','swin_v2_small'):
        science = science_configuration()
        v2 = candidate()['model']=='swin_v2_small'
        norms, depth, attention = (value.get(key, []) for key in
                                   ('layer_norms', 'stochastic_depth', 'attention'))
        expected_attention = [
            {'heads': heads, 'window': [8, 8] if v2 else [7, 7], 'shift': [0 if index % 2 == 0 else 4 if v2 else 3] * 2,
             'dropout': 0.0, 'attention_dropout': 0.0}
            for heads, count in zip((3,6,12,24) if v2 else (4,8,16,32), (2,2,18,2)) for index in range(count)]
        if (value.get('candidate') != candidate() or value.get('batch_norm_modules') != 0
                or value.get('buffer_count') != (48 if v2 else 24)
                or value.get('buffer_sha256') != science.get('initial_buffer_sha256')
                or len(value.get('static_index_names', [])) != 24
                or any(not name.endswith('.relative_position_index') for name in value['static_index_names'])
                or value.get('stage_widths') != ([96,192,384,768] if v2 else [128,256,512,1024])
                or value.get('head_in_features') != (768 if v2 else 1024) or value.get('head_out_features') != 1
                or value.get('dropout') != science['job']['dropout']
                or len(norms) != structure_contract()['layer_norms']
                or any(item['eps'] != 1e-5 for item in norms)
                or len(depth) != 24
                or any(item['mode'] != 'row' or item['p'] != science['job']['stochastic_depth'] * index / 23
                       for index, item in enumerate(depth))
                or attention != expected_attention):
            raise ValueError('actual Swin V1 LN/index/window/head/SD contract differs')
        return True
    if candidate()['model'] == 'resnet50_imagenet':
        layers = value.get('batch_norms', [])
        digest = value.get('buffer_sha256')
        expected_digest = science_configuration().get('initial_buffer_sha256')
        if (value.get('candidate') != candidate() or value.get('batch_norm_modules') != 53
                or value.get('buffer_count') != 159 or len(layers) != 53
                or len({row['name'] for row in layers}) != 53
                or any(row['training'] or not row['track_running_stats']
                       or row['eps'] != structure_contract()['batch_norm_eps'] for row in layers)
                or value.get('head_in_features') != 2048 or value.get('head_out_features') != 1
                or value.get('dropout') != science_configuration()['job']['dropout']
                or value.get('layer_norms') or value.get('stochastic_depth')
                or not isinstance(digest, str) or len(digest) != 64
                or (expected_digest is not None and digest != expected_digest)):
            raise ValueError('actual R50 fixed BN/buffer/head identity differs')
        return True
    norms, depth = value.get('layer_norms', []), value.get('stochastic_depth', [])
    configured = science_configuration()['job']
    if configured.get('quadrant_mil_policy') is not None:
        from .config import QUADRANT_MIL_RULE
        if (science_configuration()['candidate_profile'] not in (QUADRANT_MIL_SMALL_BINDING,QUADRANT_MIL_SMALL_SEED27_BINDING)
                or configured['quadrant_mil_policy']!=QUADRANT_MIL_RULE
                or value.get('quadrant_mil')!={'policy':QUADRANT_MIL_RULE,'network_calls':2,
                    'shared_network':True,'parameters_added':0}):
            raise ValueError('actual fixed shared Small global/corner graph differs')
    elif value.get('quadrant_mil')is not None:
        raise ValueError('unregistered global/corner graph')
    if configured.get('cosine_head_policy') is not None:
        from .config import COSINE_HEAD_RULE
        if (science_configuration()['candidate_profile'] not in (COSINE_HEAD_SMALL_BINDING,COSINE_HEAD_SMALL_SEED27_BINDING)
                or configured['cosine_head_policy'] != COSINE_HEAD_RULE
                or value.get('cosine_head') != {'policy':COSINE_HEAD_RULE,'module':'CosineLinear',
                                              'in_features':768,'out_features':1}):
            raise ValueError('actual fixed cosine16 readout differs')
    elif value.get('cosine_head') is not None:
        raise ValueError('unregistered cosine head')
    if configured.get('gradient_centralization_policy') is not None:
        from .config import GRADIENT_CENTRALIZATION_RULE
        gc=value.get('gradient_centralization',{});targets=gc.get('targets',[])
        if (science_configuration()['candidate_profile'] != RAW_GC_SMALL_BINDING
                or configured['gradient_centralization_policy'] != GRADIENT_CENTRALIZATION_RULE
                or gc.get('policy') != GRADIENT_CENTRALIZATION_RULE or len(targets)!=113
                or len({r['name'] for r in targets})!=113
                or sum(r['module']=='Conv2d' for r in targets)!=40
                or sum(r['module']=='Linear' for r in targets)!=73
                or any(not r['name'].endswith('.weight') or r['module'] not in ('Conv2d','Linear')
                       or len(r['shape'])!=(4 if r['module']=='Conv2d' else 2)
                       or r['axes']!=list(range(1,len(r['shape']))) for r in targets)):
            raise ValueError('actual raw GC direct-weight names/shapes/axes differ')
    elif value.get('gradient_centralization') is not None:
        raise ValueError('unregistered raw gradient centralization')
    if configured.get('stem_xy_policy') is not None:
        from .models import STEM_XY_RULE
        expected={'policy':STEM_XY_RULE,'name':'network.features.0.0','weight_shape':[96,5,4,4],
            'bias_shape':[96],'in_channels':5,'out_channels':96,'kernel_size':[4,4],'stride':[4,4],
            'padding':[0,0],'dilation':[1,1],'groups':1,'coordinate_generation':'stateless'}
        if (science_configuration()['candidate_profile']!=STEM_XY_SMALL_BINDING
                or configured['stem_xy_policy']!=STEM_XY_RULE or value.get('stem_xy')!=expected):
            raise ValueError('actual RGBXY stem or coordinate semantics differ')
    elif value.get('stem_xy') is not None:raise ValueError('unregistered RGBXY stem')
    from .config import GLOBAL_NATIVE_SIX_RULE
    if configured.get('input_preprocessing')==GLOBAL_NATIVE_SIX_RULE:
        expected={'policy':GLOBAL_NATIVE_SIX_RULE,'name':'network.features.0.0','weight_shape':[96,6,4,4],
            'bias_shape':[96],'in_channels':6,'out_channels':96,'kernel_size':[4,4],'stride':[4,4],
            'padding':[0,0],'dilation':[1,1],'groups':1}
        if science_configuration()['candidate_profile'] not in (GLOBAL_NATIVE_SIX_SMALL_BINDING,GLOBAL_NATIVE_SIX_SMALL_SEED27_BINDING) or value.get('stem_native6')!=expected:
            raise ValueError('actual six-channel stem geometry or input semantics differ')
    elif value.get('stem_native6') is not None:raise ValueError('unregistered native six-channel stem')
    if configured.get('mixstyle_policy') != candidate().get('mixstyle_policy'):
        raise ValueError('MixStyle job has no matching candidate policy')
    if configured.get('mixstyle_policy') is not None:
        from .mixstyle import MIXSTYLE_RULE
        expected_mixstyle = {'policy': MIXSTYLE_RULE,
            'sites': [{'feature_index': 1, 'feature_name': 'network.features.1', 'channels': 96, 'blocks': 3},
                      {'feature_index': 3, 'feature_name': 'network.features.3', 'channels': 192, 'blocks': 3}],
            'training_only': True, 'persistent_rng_state': False}
        if (science_configuration()['candidate_profile'] not in (MIXSTYLE_SMALL_BINDING, MIXSTYLE_SMALL_R1_BINDING)
                or configured['mixstyle_policy'] != MIXSTYLE_RULE
                or value.get('mixstyle') != expected_mixstyle):
            raise ValueError('actual two-stage rank-local8 MixStyle graph differs')
    elif value.get('mixstyle') is not None:
        raise ValueError('unregistered MixStyle graph')
    if configured.get('tri3_downsample_policy') != candidate().get('tri3_downsample_policy'):
        raise ValueError('Tri3 job has no matching registered candidate identity')
    if configured.get('tri3_downsample_policy') is not None:
        from .tri3_downsample import TRI3_RULE
        expected_tri3 = {'policy': TRI3_RULE, 'targets': [
            {'feature_index': index, 'normalization_name': f'network.features.{index}.0',
             'convolution_name': f'network.features.{index}.1',
             'input_channels': inputs, 'output_channels': outputs,
             'convolution_kernel': [2, 2], 'convolution_stride': [1, 1],
             'convolution_padding': [0, 0],
             'tail': [{'kind': 'ReflectionPad2d', 'padding': [1, 1, 1, 1]},
                      *[{'kind': 'AvgPool2d', 'kernel_size': 2, 'stride': stride,
                         'padding': 0, 'ceil_mode': False, 'count_include_pad': True,
                         'divisor_override': None} for stride in (1, 2)]]}
            for index, inputs, outputs in ((2, 96, 192), (4, 192, 384), (6, 384, 768))]}
        if (science_configuration()['candidate_profile'] != TRI3_DOWNSAMPLE_SMALL_BINDING
                or configured['tri3_downsample_policy'] != TRI3_RULE
                or value.get('tri3_downsample') != expected_tri3):
            raise ValueError('actual three native LN/convolution/Tri3 descriptions differ')
    elif value.get('tri3_downsample') is not None:
        raise ValueError('unregistered Tri3 structure')
    if configured.get('low_rank_policy') != candidate().get('low_rank_policy'):
        raise ValueError('low-rank job has no matching registered candidate identity')
    if configured.get('low_rank_policy') is not None:
        from .low_rank import LOW_RANK_RULE
        proof = value.get('low_rank', {})
        if (configured['low_rank_policy'] != LOW_RANK_RULE
                or proof.get('low_rank_policy') != LOW_RANK_RULE
                or len(proof.get('targets', [])) != 72
                or len(proof.get('all_parameter_names', [])) != 489
                or len(proof.get('frozen_parameter_names', [])) != 342
                or len(proof.get('trainable_parameter_names', [])) != 147
                or proof.get('initial_frozen_tensor_sha256') != proof.get('frozen_tensor_sha256')):
            raise ValueError('actual fixed pointwise/frozen/FP32 identity differs')
    elif value.get('low_rank') is not None:
        raise ValueError('unregistered low-rank structure')
    if science_configuration()['candidate_profile'] == CONTENT_POOL_SMALL_BINDING:
        if value.get('attention_pool') != {'parameter_name': 'attention_pool_weight',
                                           'shape': [1, 768], 'dtype': 'torch.float32'}:
            raise ValueError('actual Small content pooling parameter differs')
    elif value.get('attention_pool') is not None:
        raise ValueError('unregistered attention pooling parameter')
    if science_configuration()['candidate_profile'] == STAGE3_RESIDUAL_SMALL_BINDING:
        if value.get('stage3_residual') != {'parameter_name': 'stage3_residual_weight',
                'shape': [1, 384], 'dtype': 'torch.float32', 'feature_index': 5, 'channels': 384}:
            raise ValueError('actual Small stage3 residual parameter differs')
    elif value.get('stage3_residual') is not None:
        raise ValueError('unregistered stage3 residual parameter')
    if ('final_downsample_stride' in configured
            or science_configuration()['candidate_profile'] in (FIT_SSL_SMALL_BINDING, BASE_BINDING,BASE_CLIP_BINDING,BASE_CLIP_SEED27_BINDING,BASE_CLIP_STORAGE16_BINDING)):
        stride = configured.get('final_downsample_stride', 2)
        if value.get('final_downsample') != {'name': 'network.features.6.1', 'stride': [stride, stride],
                'kernel_size': [2, 2], 'padding': [0, 0], 'dilation': [1, 1]}:
            raise ValueError('actual final ConvNeXt downsample geometry differs from pinned job')
    if (science_configuration()['candidate_profile'] in (BASE_BINDING,BASE_CLIP_BINDING,BASE_CLIP_SEED27_BINDING,BASE_CLIP_STORAGE16_BINDING)
            and value.get('stage_widths') != [128, 256, 512, 1024]):
        raise ValueError('actual Base stage widths differ')
    if (value.get('candidate') != candidate() or value.get('batch_norm_modules') != 0
            or value.get('buffer_count') != 0 or value.get('buffer_sha256') != hashlib.sha256(b'').hexdigest()
            or value.get('head_in_features') != candidate()['features'] or value.get('head_out_features') != 1
            or value.get('dropout') != science_configuration()['job']['dropout']
            or len(norms) != structure_contract()['layer_norms'] or len({n['name'] for n in norms}) != len(norms)
            or not any(n['name'] == 'network.classifier.0' for n in norms)
            or any(n['eps'] != (candidate()['final_norm_eps'] if n['name'] == 'network.classifier.0' else 1e-6) for n in norms)
            or len(depth) != structure_contract()['stochastic_depth_modules']
            or any(d['mode'] != 'row' or d['p'] != science_configuration()['job']['stochastic_depth'] * i / (len(depth)-1) for i, d in enumerate(depth))):
        raise ValueError('Tiny actual LN/noBN/stochastic-depth contract differs')
    return True


def accepted_manifest(path, digest):
    expected = metadata()['CONTRACT.json']['new_manifest']
    if (not path or Path(path).resolve() != (Path(load_site()['metadata_root']) / 'manifest.json').resolve()
            or digest != expected['sha256'] or sha256_file(path) != digest):
        raise ValueError('exact accepted 9:1 manifest required; no allocation fallback')
    return copy.deepcopy(metadata()['manifest.json'])


def validate_runtime(identity, job, manifest=None, fresh=None):
    from .config import job_contract
    validate_job(job)
    init = identity.get('initialization', {})
    if (identity.get('experiment91') != binding() or identity.get('manifest_digest') != binding()['manifest_digest']
            or identity.get('job') != job_contract(job) or len(identity.get('source_digest', '')) != 64
            or init.get('weights_sha256') != public_source()['sha256']
            or init.get('provenance_sha256') != public_source()['provenance_sha256']
            or (candidate()['model'] != 'resnet50_imagenet' and init.get('final_norm_eps') != candidate()['final_norm_eps'])
            or identity.get('phase_parent_sha256') is not None):
        raise ValueError('Tiny22K-to-1K runtime source/job/manifest/LN differs')
    if manifest is not None and canonical_digest(manifest) != binding()['manifest_digest']:
        raise ValueError('9:1 runtime manifest changed')
    spec = fresh if fresh is not None else identity.get('fresh')
    if (not spec or spec.get('experiment91') != binding() or spec.get('schedule') != schedule()
            or spec.get('ema') != ema_rule()
            or spec.get('ssl_recipe') != science_configuration().get('ssl_recipe')):
        raise ValueError('Tiny fresh schedule/noBN EMA/binding differs')
    if spec.get('ssl_recipe') is not None:
        from .ssl_stage import validate_recipe
        validate_recipe(spec['ssl_recipe'],job,metadata()['manifest.json'],len(load_site()['gpus']))
        if spec.get('ssl_startup_barrier') != str(Path(load_site()['round_root']) / 'control/ssl-startup-barrier.json'):
            raise ValueError('R115 joint SSL startup barrier differs')
    return True


def validate_checkpoint(payload):
    from .age import validate_checkpoint as validate_age
    job = payload['job']
    validate_runtime(payload['identity'], job, fresh=payload.get('fresh'))
    derived=payload.get('ssl_derived')
    recipe=payload['fresh'].get('ssl_recipe')
    if bool(derived)!=bool(recipe and not payload.get('cpu_fixture_only')) or (derived is not None and (
            derived.get('recipe')!=recipe or derived.get('successful_updates')!=1600
            or derived.get('status')!='completed' or derived.get('dev_evaluations')!=0)):
        raise ValueError('R115 fixed fit-only derived initialization lineage absent or incompatible')
    validate_age(payload, job)
    if job.get('low_rank_policy') is not None:
        from .low_rank import validate_checkpoint as validate_low_rank_checkpoint
        validate_low_rank_checkpoint(payload, job)
    validate_normalization(payload.get('candidate_normalization', {}))
    if science_configuration()['candidate_profile'] == STAGE3_RESIDUAL_SMALL_BINDING:
        import torch
        from .stage3_residual import STAGE3_RESIDUAL_RULE
        if job.get('stage3_residual') != STAGE3_RESIDUAL_RULE:
            raise ValueError('checkpoint stage3 residual protocol differs')
        for state in (payload['model'], payload['ema']['model']):
            weight = state.get('stage3_residual_weight')
            if (not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (1, 384)
                    or weight.dtype != torch.float32 or not bool(torch.isfinite(weight).all())):
                raise ValueError('checkpoint stage3 residual tensor differs')
    if job.get('freeze_features_before') is not None:
        proof = payload.get('parameter_trainability', {})
        names = sorted(payload['model'])
        frozen = [n for n in names if n.startswith(tuple('network.features.' + str(i) + '.'
                  for i in range(job['freeze_features_before'])))]
        trainable = sorted(set(names) - set(frozen))
        if (proof.get('freeze_features_before') != job['freeze_features_before']
                or proof.get('all_parameter_names') != names or proof.get('frozen_parameter_names') != frozen
                or proof.get('trainable_parameter_names') != trainable
                or proof.get('optimizer_parameter_names') != trainable):
            raise ValueError('checkpoint frozen/trainable/optimizer identity differs')
        import torch
        if any(not torch.equal(payload['model'][n], payload['ema']['model'][n]) for n in frozen):
            raise ValueError('EMA frozen features differ from online checkpoint')
    if (payload.get('manifest_digest') != binding()['manifest_digest'] or payload.get('mode') != 'train'
            or any(payload.get('schedule', {}).get(k) != v for k, v in schedule().items())
            or payload.get('ema', {}).get('rule') != ema_rule()):
        raise ValueError('Tiny checkpoint schedule/candidate/noBN contract differs')
    return True

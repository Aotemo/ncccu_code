"""Native author EfficientNetV2-RW-S with immutable public BN statistics."""
import importlib
import inspect
import json
from pathlib import Path
import torch
from torch import nn

MODEL = 'efficientnetv2_rw_s'
VARIANT = MODEL + '.ra2_in1k'
FORMAT = 'timm-efficientnetv2-rw-s-native-full1000-state-dict-v1'
REVISION = 'ee38fe1c4e8ca084db9b6ec22c3d6a7d02990704'
WEIGHT_SHA = '8bb2555726585abc07991848cfa0de5732cff2d8bf9f88d6317a687dbdc1b303'
LICENSE_SHA = 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
LIBRARY = {'version':'1.0.27','source_sha256':{
    'models/efficientnet.py':'c0a9b9ade7ca27395379217636ef2f222b33878fbd4ac81088b581594a06ad6a',
    'models/_efficientnet_builder.py':'084cf15cc4725fd926966f0dad94c8b4786f4c96aae625a8497f35ae00704000',
    'models/_efficientnet_blocks.py':'2a3de2c0371a74d7f4fe165f4ad062d768473ca9565a9ebf63fadbd0aa5ae021',
    'layers/norm_act.py':'4821c9fa7b73c19cb63d16a4c86912f36380c56e34d25bc6170ea728aa21fed6'}}
EMA_RULE = {'version':'ema-warmcopy1600-decay0999-publicbn336-v1','warmcopy_updates':1600,
    'decay':.999,'dtype':'torch.float32','floating_buffers':'same_as_parameters_except_frozen_bn',
    'integer_buffers':'copy','frozen_bn_buffers':'copy','frozen_bn_buffer_count':336}
BN = nn.modules.batchnorm._BatchNorm


def library_identity():
    import timm
    from .models import _hash
    value = {'version':timm.__version__,'source_sha256':{
        name:_hash(inspect.getfile(importlib.import_module('timm.'+name[:-3].replace('/','.'))))
        for name in LIBRARY['source_sha256']}}
    if value != LIBRARY:
        raise ValueError('native EfficientNet implementation identity differs')
    return value


def validate_initialization(initialization):
    from .models import _hash
    from .experiment91 import public_source
    source = public_source()
    path = Path(initialization['path']); meta_path = Path(initialization['provenance'])
    if (initialization.get('kind')!='imagenet' or initialization.get('format')!=FORMAT
            or source.get('model')!=VARIANT or source.get('format')!=FORMAT
            or source.get('revision')!=REVISION or source.get('sha256')!=WEIGHT_SHA
            or source.get('original_sha256')!=WEIGHT_SHA or source.get('native_library')!=LIBRARY
            or initialization.get('sha256')!=WEIGHT_SHA
            or any(not p.is_absolute() or p.is_symlink() for p in (path,meta_path))
            or _hash(path)!=WEIGHT_SHA or _hash(meta_path)!=source['provenance_sha256']):
        raise ValueError('fixed native EfficientNet public identity differs')
    meta = json.loads(meta_path.read_text())
    if (any(meta.get(k)!=source[k] for k in ('model','format','revision','source_url','training_data',
                                             'sha256','original_sha256','native_library'))
            or meta.get('redistribution_confirmed') is not True or meta.get('license_sha256')!=LICENSE_SHA
            or _hash(meta_path.parent/meta['license_file'])!=LICENSE_SHA):
        raise ValueError('native EfficientNet provenance/license differs')
    return {'kind':'imagenet','model':MODEL,'format':FORMAT,'weights_sha256':WEIGHT_SHA,
        'provenance_sha256':source['provenance_sha256'],'final_norm_eps':None,'internal_norm_eps':None,
        'source_url':source['source_url'],'training_data':source['training_data'],'source_commit':REVISION,
        'original_sha256':WEIGHT_SHA,'license_sha256':LICENSE_SHA,'native_library':LIBRARY}


class EfficientNetBinary(nn.Module):
    def __init__(self, network, age_spec):
        super().__init__()
        self.network = network
        from .age import attach_age
        attach_age(self, age_spec)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module,BN):module.eval()
        return self

    def assert_bn_mode(self):
        layers = [m for m in self.modules() if isinstance(m,BN)]
        if (len(layers)!=112 or any(m.training or isinstance(m,nn.SyncBatchNorm)
                or not m.track_running_stats or m.eps!=1e-5 or m.momentum!=.1 for m in layers)
                or self.training and any(not m.weight.requires_grad or not m.bias.requires_grad for m in layers)):
            raise ValueError('EfficientNet requires 112 fixed public BN layers with trainable affine')
        return layers

    def forward(self, images, age_features=None):
        if images.ndim!=4 or tuple(images.shape[1:])!=(3,512,512) or images.dtype!=torch.float32:
            raise ValueError('native EfficientNet requires FP32 Bx3x512x512')
        self.assert_bn_mode()
        from .age import residual
        return residual(self,self.network(images).flatten(),age_features)


def frozen_bn_names(model):
    if not isinstance(model,EfficientNetBinary):raise ValueError('native EfficientNet model required')
    model.assert_bn_mode()
    names = frozenset(prefix+'.'+suffix for prefix,module in model.named_modules() if isinstance(module,BN)
        for suffix in ('running_mean','running_var','num_batches_tracked'))
    if len(names)!=336 or names!=set(dict(model.named_buffers())):
        raise ValueError('native EfficientNet requires exactly336 BN buffers')
    return names


def frozen_bn_digest(model):
    from .ema import tensor_digest
    state = model.state_dict()
    return tensor_digest({n:state[n] for n in frozen_bn_names(model)})


def describe(model):
    names = frozen_bn_names(model); network = model.network
    head = network.classifier
    if (not isinstance(head,nn.Sequential) or len(head)!=2 or not isinstance(head[0],nn.Dropout)
            or head[0].p!=.2 or not isinstance(head[1],nn.Linear)
            or (head[1].in_features,head[1].out_features)!=(1792,1) or network.drop_rate!=0.
            or network.global_pool.pool_type!='avg' or [len(s)for s in network.blocks]!=[2,4,4,6,9,15]
            or len(dict(model.named_parameters()))!=459):
        raise ValueError('native EfficientNet binary structure differs')
    return {'version':'efficientnetv2-rws-publicbn-v1','model':MODEL,'native_variant':VARIANT,
        'feature_width':1792,'stage_depths':[2,4,4,6,9,15],'pool':'avg','dropout':.2,
        'native_drop_rate':0.,'batch_norm_modules':112,'batch_norm_buffers':len(names),
        'batch_norm_eps':1e-5,'batch_norm_momentum':.1,'batch_norm_statistics':'frozen_public',
        'batch_norm_affine':'trainable_online','parameter_tensor_count':459,'age_readouts':1}


def create_model(job, *, initialize=True):
    fixed = {'model':MODEL,'image_size':512,'precision':'fp32','normalization':'imagenet',
             'dropout':.2,'stochastic_depth':.1,'frozen_bn':True,'head_lr_scope':'last_linear'}
    if any(job.get(k)!=v for k,v in fixed.items()) or not job.get('age',{}).get('enabled'):
        raise ValueError('native EfficientNet fixed model contract differs')
    import timm
    library_identity()
    network = timm.create_model(VARIANT,pretrained=False,drop_rate=.2,drop_path_rate=.1)
    public_receipt = None
    if initialize:
        proof = validate_initialization(job['initialization'])
        state = torch.load(job['initialization']['path'],map_location='cpu',weights_only=True)
        expected = network.state_dict()
        if (not isinstance(state,dict) or len(state)!=794 or state.keys()!=expected.keys()
                or any(not torch.is_tensor(t) or t.shape!=expected[n].shape or t.dtype!=expected[n].dtype
                    or t.is_floating_point() and (t.dtype!=torch.float32 or not bool(torch.isfinite(t).all()))
                    for n,t in state.items())):
            raise ValueError('native EfficientNet full1000 public state differs')
        network.load_state_dict(state,strict=True)
        if any(not torch.equal(t,network.state_dict()[n])for n,t in state.items()):
            raise ValueError('native EfficientNet full1000 strict load is not exact')
        from .ema import tensor_digest
        public_receipt = {'initialization':proof,'strict_missing_keys':[],'strict_unexpected_keys':[],
            'public_tensor_count':794,'public_tensor_sha256':tensor_digest(state),'full_head_shape':[1000,1792],
            'backbone_only':False,'all_public_tensors_exact':True,'all_public_floating_FP32_finite':True,
            'public_integer_counter_tensors':112}
        del state
    network.classifier = nn.Sequential(nn.Dropout(.2),nn.Linear(1792,1))
    network.drop_rate = 0.
    result = EfficientNetBinary(network,job['age'])
    describe(result)
    result.public_initialization_receipt = public_receipt
    if initialize and job.get('experiment91') is not None:
        from .experiment91 import describe_model
        result.candidate_normalization = describe_model(result)
    return result

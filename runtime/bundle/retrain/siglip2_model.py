"""Full native SigLIP2 image encoder with its public MAP readout preserved."""
import importlib
import inspect
import json
from pathlib import Path
import torch
from torch import nn

MODEL='siglip2_vitb16'
VARIANT='vit_base_patch16_siglip_512.v2_webli'
FORMAT='timm-siglip2-vitb16-native512-image-encoder-state-dict-v1'
REVISION='3c4b2d3989d7cfae00a09fff172c0a6cf8cbb32a'
WEIGHT_SHA='e6cebc19e5531a0fbe55e3b8d03dbabb8f5c41a698fe01e7b27527ff946132e6'
LICENSE_SHA='cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
LIBRARY={'version':'1.0.27','source_sha256':{
    'models/vision_transformer.py':'f0f453b75952698059a1e82fa283a7a9cf2cf2217c33fc2c9cb1efc06947f552',
    'layers/attention_pool.py':'8b7d0f493adafb01aa7fedca261df245bbd1a0813884d9bc13d0b24ac253b8c9'}}


def library_identity():
    import timm
    from .models import _hash
    value={'version':timm.__version__,'source_sha256':{
        name:_hash(inspect.getfile(importlib.import_module('timm.'+name[:-3].replace('/','.'))))
        for name in LIBRARY['source_sha256']}}
    if value!=LIBRARY:raise ValueError('native SigLIP2 implementation identity differs')
    return value


def validate_initialization(initialization):
    from .models import _hash
    from .experiment91 import public_source
    source=public_source();path=Path(initialization['path']);meta_path=Path(initialization['provenance'])
    if (initialization.get('kind')!='natural_vision_language' or initialization.get('format')!=FORMAT
            or source.get('kind')!='natural_vision_language' or source.get('model')!=VARIANT
            or source.get('format')!=FORMAT or source.get('revision')!=REVISION
            or source.get('sha256')!=WEIGHT_SHA or source.get('original_sha256')!=WEIGHT_SHA
            or source.get('native_library')!=LIBRARY or initialization.get('sha256')!=WEIGHT_SHA
            or any(not p.is_absolute()or p.is_symlink()for p in(path,meta_path))
            or _hash(path)!=WEIGHT_SHA or _hash(meta_path)!=source['provenance_sha256']):
        raise ValueError('fixed natural-image SigLIP2 public identity differs')
    meta=json.loads(meta_path.read_text())
    if (any(meta.get(k)!=source[k]for k in('model','format','revision','source_url','training_data',
                                          'sha256','original_sha256','native_library'))
            or meta.get('redistribution_confirmed')is not True or meta.get('license_sha256')!=LICENSE_SHA
            or _hash(meta_path.parent/meta['license_file'])!=LICENSE_SHA):
        raise ValueError('native SigLIP2 provenance/license differs')
    return {'kind':'natural_vision_language','model':MODEL,'format':FORMAT,'weights_sha256':WEIGHT_SHA,
        'provenance_sha256':source['provenance_sha256'],'final_norm_eps':1e-6,'internal_norm_eps':1e-6,
        'source_url':source['source_url'],'training_data':source['training_data'],'source_commit':REVISION,
        'original_sha256':WEIGHT_SHA,'license_sha256':LICENSE_SHA,'native_library':LIBRARY}


class SigLIP2Binary(nn.Module):
    def __init__(self,network,age_spec):
        super().__init__();self.network=network
        self.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(768,1))
        from .age import attach_age
        attach_age(self,age_spec)

    def forward(self,images,age_features=None):
        if images.ndim!=4 or tuple(images.shape[1:])!=(3,512,512)or images.dtype!=torch.float32:
            raise ValueError('native SigLIP2 requires FP32 Bx3x512x512')
        from .age import residual
        return residual(self,self.classifier(self.network(images)).flatten(),age_features)


def describe(model):
    network=model.network;pool=network.attn_pool
    norms=[{'name':n,'eps':m.eps,'shape':list(m.normalized_shape)}for n,m in network.named_modules()
           if isinstance(m,nn.LayerNorm)]
    rates=[float(x)for x in torch.linspace(0,.1,12)]
    actual_rates=[float(getattr(getattr(b,f'drop_path{j}'),'drop_prob',0.))for b in network.blocks for j in(1,2)]
    if (not isinstance(model,SigLIP2Binary)or network.global_pool!='map' or network.num_prefix_tokens!=0
            or network.cls_token is not None or tuple(network.pos_embed.shape)!=(1,1024,768)
            or tuple(network.patch_embed.img_size)!=(512,512)or tuple(network.patch_embed.patch_size)!=(16,16)
            or not isinstance(network.head,nn.Identity)or not isinstance(network.fc_norm,nn.Identity)
            or not isinstance(network.norm_pre,nn.Identity)or network.head_drop.p!=0.
            or tuple(pool.latent.shape)!=(1,1,768)or pool.num_heads!=12 or pool.pool!='token'
            or pool.proj_drop.p!=0. or pool.mlp is None or len(network.blocks)!=12
            or len(norms)!=26 or any(n['eps']!=1e-6 or n['shape']!=[768]for n in norms)
            or actual_rates!=[p for p in rates for _ in(1,2)]
            or len(dict(model.named_parameters()))!=165 or sum(p.numel()for p in model.parameters())!=93520899
            or dict(model.named_buffers())or model.classifier[0].p!=.2
            or (model.classifier[-1].in_features,model.classifier[-1].out_features)!=(768,1)):
        raise ValueError('native SigLIP2 public MAP/position/LN/binary graph differs')
    return {'version':'siglip2-vitb16-native512-public-map-v1','native_variant':VARIANT,
        'native_library':LIBRARY,'image_size':[512,512],'patch_size':[16,16],'patch_tokens':1024,
        'features':768,'prefix_tokens':0,'public_pool':'map','public_latent_shape':[1,1,768],
        'public_pool_heads':12,'public_pool_MLP':True,'position_shape':[1,1024,768],
        'normalization_catalog':norms,'native_head_dropout':0.,'binary_dropout':.2,
        'parameter_tensor_count':165,'parameter_count':93520899,'buffer_count':0,'age_readouts':1}


def create_model(job,*,initialize=True):
    fixed={'model':MODEL,'image_size':512,'precision':'fp32','normalization':'unit','dropout':.2,
        'stochastic_depth':.1,'head_lr_scope':'last_linear','final_norm_eps':1e-6}
    if any(job.get(k)!=v for k,v in fixed.items())or not job.get('age',{}).get('enabled')or job.get('frozen_bn'):
        raise ValueError('native SigLIP2 fixed model contract differs')
    import timm
    library_identity()
    network=timm.create_model(VARIANT,pretrained=False,drop_rate=0.,drop_path_rate=.1)
    public=None
    if initialize:
        proof=validate_initialization(job['initialization'])
        state=torch.load(job['initialization']['path'],map_location='cpu',weights_only=True)
        expected=network.state_dict()
        if (not isinstance(state,dict)or len(state)!=162 or state.keys()!=expected.keys()
                or any(not torch.is_tensor(t)or t.shape!=expected[n].shape or t.dtype!=torch.float32
                       or not bool(torch.isfinite(t).all())for n,t in state.items())):
            raise ValueError('complete public SigLIP2 image encoder state differs')
        network.load_state_dict(state,strict=True)
        if any(not torch.equal(t,network.state_dict()[n])for n,t in state.items()):
            raise ValueError('complete public SigLIP2 strict load is not exact')
        from .ema import tensor_digest
        public={'initialization':proof,'strict_missing_keys':[],'strict_unexpected_keys':[],
            'public_tensor_count':162,'public_tensor_sha256':tensor_digest(state),'full_head_shape':None,
            'backbone_only':False,'all_public_tensors_exact':True,'all_public_tensors_FP32_finite':True}
        del state
    result=SigLIP2Binary(network,job['age']);describe(result)
    result.public_initialization_receipt=public
    return result

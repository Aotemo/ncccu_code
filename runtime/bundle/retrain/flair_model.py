"""FLAIR visual ResNet50 binary head with immutable public BN buffers."""
import torch
from torch import nn

class FlairResnetBinary(nn.Module):
    def __init__(self,backbone,age_spec):
        super().__init__()
        self.backbone=backbone
        with torch.random.fork_rng(devices=[]):
            self.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(2048,1))
        generator=torch.Generator(device='cpu').manual_seed(20261027)
        with torch.no_grad():
            self.classifier[-1].weight.copy_(torch.randn((1,2048),generator=generator,dtype=torch.float32)*.01)
            self.classifier[-1].bias.zero_()
        from .age import attach_age
        attach_age(self,age_spec)
        self.train(True)

    def train(self,mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def assert_bn_mode(self):
        layers=[m for m in self.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        if len(layers)!=53 or any(m.training or isinstance(m,nn.SyncBatchNorm) or not m.track_running_stats for m in layers):
            raise ValueError('N05 requires exactly 53 frozen standard BatchNorm layers')
        if any(not m.weight.requires_grad or not m.bias.requires_grad for m in layers if self.training):
            raise ValueError('N05 BN affine parameters must remain trainable')
        return layers

    def forward(self,images,age_features=None):
        if images.ndim!=4 or tuple(images.shape[1:])!=(3,512,512) or images.dtype!=torch.float32:
            raise ValueError('N05 FLAIR requires FP32 Bx3x512x512 input')
        self.assert_bn_mode()
        features=self.backbone(images)
        if features.shape!=(images.shape[0],2048):raise ValueError('FLAIR visual feature dimensions differ')
        from .age import residual
        return residual(self,self.classifier(features).flatten(),age_features)

def frozen_bn_names(model):
    if not isinstance(model,FlairResnetBinary):raise ValueError('FLAIR frozen-BN model required')
    names=[]
    for prefix,module in model.named_modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats:raise ValueError('FLAIR BN running buffers required')
            names.extend(prefix+'.'+suffix for suffix in ('running_mean','running_var','num_batches_tracked'))
    if len(names)!=159 or len(set(names))!=159:raise ValueError('FLAIR requires 159 BN buffers')
    return frozenset(names)

def frozen_bn_digest(model):
    from .ema import tensor_digest
    state=model.state_dict();names=frozen_bn_names(model)
    return tensor_digest({name:state[name] for name in names})

def create_flair_binary(job,*,initialize=True):
    if (job.get('model')!='flair_resnet50' or job.get('image_size')!=512 or job.get('precision')!='fp32'
            or job.get('normalization')!='flair_rgb01' or job.get('dropout')!=.2
            or job.get('frozen_bn') is not True or not job.get('age',{}).get('enabled')):
        raise ValueError('N05 FLAIR model protocol differs')
    from torchvision.models import resnet50
    with torch.random.fork_rng(devices=[]):
        backbone=resnet50(weights=None)
    backbone.fc=nn.Identity()
    if initialize:
        from .models import validate_initialization
        from safetensors import safe_open
        proof=validate_initialization(job['initialization'])
        if proof['kind']!='retinal_vision_language':raise ValueError('wrong FLAIR provenance')
        with safe_open(job['initialization']['path'],framework='pt',device='cpu') as source:
            state={name:source.get_tensor(name) for name in source.keys()}
        expected=backbone.state_dict()
        if len(state)!=318 or set(state)!=set(expected):raise ValueError('FLAIR visual tensor keys differ')
        for name,tensor in state.items():
            ref=expected[name]
            if tensor.shape!=ref.shape or tensor.dtype!=ref.dtype or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())):
                raise ValueError('FLAIR visual tensor shape/dtype/finite differs: '+name)
        backbone.load_state_dict(state,strict=True)
    result=FlairResnetBinary(backbone,job['age'])
    result.assert_bn_mode();frozen_bn_names(result)
    return result

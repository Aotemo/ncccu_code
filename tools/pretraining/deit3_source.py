"""Pin the one native DeiT III Small public asset; reuse the existing byte fetch."""
import argparse,json
from pathlib import Path
import sys
from source_chain import cpu_context,configure_torch_threads,require,sha256
from small_source import fetch

REPO='timm/deit3_small_patch16_384.fb_in22k_ft_in1k'
REVISION='8d59b2810f277c19d0cdd1b1e691d93c24aab6db'
WEIGHT_SHA='88140b2a114ec4db0e3588af052162b87fcdd1bb92d5e85dd93f7d2adc1fb526'
WEIGHT_BYTES=88885197
FORMAT='timm-deit3-native384-state-dict-v1'
MODEL='deit3_small_patch16_384'
TRAINING='ImageNet-22K pretraining -> ImageNet-1K fine-tuning'
SOURCE_PROFILES = {
    'deit3':dict(repo=REPO,revision=REVISION,weight_sha=WEIGHT_SHA,weight_bytes=WEIGHT_BYTES,
        format=FORMAT,model=MODEL,training=TRAINING,classes=1000,source_size=384,state_tensors=176),
    'vitdino3':dict(repo='timm/vit_small_patch16_dinov3.lvd1689m',
        revision='3bf4720a82ec2066db88137180ff1f83a675cef0',
        weight_sha='b7e7b7d1518a4dad2d92304ff47967c3dfee806f863552a06870de1e7498e7a2',
        weight_bytes=86407250,format='timm-vits16-dinov3-native-state-dict-v1',
        model='vit_small_patch16_dinov3',training='LVD-1689M self-supervised DINOv3 distilled from DINOv3 ViT-7B',
        classes=0,source_size=256,state_tensors=162,
        license_sha='25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e'),
}

def write_exact_json(path,value):
    if path.exists():
        require(json.loads(path.read_text())==value,'existing source record differs: '+str(path))
        return
    with path.open('x')as stream:
        json.dump(value,stream,indent=2);stream.write('\n')

def verified_fetch(url,target,transport_file,expected_sha=None,expected_bytes=None):
    if not target.exists():
        return fetch(url,target,expected_sha,expected_bytes,transport_file)
    require(target.is_file()and not target.is_symlink(),'existing asset must be a regular file')
    digest=sha256(target);size=target.stat().st_size
    require(digest==sha256(transport_file)and size==transport_file.stat().st_size,
            'existing asset differs from byte relay')
    require(expected_sha is None or digest==expected_sha,'existing fixed public SHA differs')
    require(expected_bytes is None or size==expected_bytes,'existing fixed public size differs')
    return {'url':url,'path':str(target),'sha256':digest,'bytes':size,
            'transport_path':str(transport_file),'transport':'existing_exact_relay_reverified_on_pod'}

def prepare(assets,transport,source_root,source_profile='deit3'):
    profile=SOURCE_PROFILES[source_profile];dino=source_profile=='vitdino3'
    repo,revision,model=profile['repo'],profile['revision'],profile['model']
    weight_sha,weight_bytes=profile['weight_sha'],profile['weight_bytes']
    license_name='LICENSE.md'if dino else'LICENSE-apache-2.0.txt'
    context=cpu_context();configure_torch_threads(context)
    root=Path(assets);require(root.is_absolute()and'/exp-v3/'in str(root),'exp-v3 asset root required')
    original=root/'original';original.mkdir(parents=True,exist_ok=True)
    relay=Path(transport);records={};base=f'https://huggingface.co/{repo}/resolve/{revision}'
    registry=(f'https://huggingface.co/api/models/{repo}?blobs=true'if dino else
              f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true')
    for name,url in [('registry-fixed.json',registry),
                     ('config.json',base+'/config.json'),('README.md',base+'/README.md'),
                     (license_name,base+'/LICENSE.md'if dino else'https://www.apache.org/licenses/LICENSE-2.0.txt')]:
        records[name]=verified_fetch(url,original/name,relay/name)
    api=json.loads((original/'registry-fixed.json').read_text())
    require(api['id']==repo and api['sha']==revision and api['private']is False and api['gated']is False
            and (api['cardData'].get('license_name')=='dinov3-license'if dino else
                 api['cardData']['license']=='apache-2.0'),'fixed public registry differs')
    row=next(s for s in api['siblings']if s['rfilename']=='pytorch_model.bin')
    require(row['lfs']['sha256']==weight_sha and row['size']==weight_bytes,'advertised public bin differs')
    config=json.loads((original/'config.json').read_text());cfg=config['pretrained_cfg']
    require((config['architecture'],config['num_classes'],config['num_features'])==(model,profile['classes'],384)
        and cfg['tag']==('lvd1689m'if dino else'fb_in22k_ft_in1k')and cfg['input_size']==[3,profile['source_size'],profile['source_size']]
        and cfg['mean']==[.485,.456,.406]and cfg['std']==[.229,.224,.225]
        and cfg['crop_pct']==1. and cfg['first_conv']=='patch_embed.proj'and cfg['classifier']=='head',
        'native public configuration differs')
    card=(original/'README.md').read_text();license_text=(original/license_name).read_text()
    if dino:
        require('license_name: dinov3-license'in card and'LVD-1689M'in card
            and records[license_name]['sha256']==profile['license_sha'],'natural DINOv3 card/full license differs')
    else:
        require('license: apache-2.0'in card and'ImageNet-22k'in card and'ImageNet-1k'in card,'public model card differs')
        require('Apache License'in license_text and'END OF TERMS AND CONDITIONS'in license_text,'license text incomplete')
    records['pytorch_model.bin']=verified_fetch(base+'/pytorch_model.bin',original/'pytorch_model.bin',relay/'pytorch_model.bin',weight_sha,weight_bytes)
    sys.path.insert(0,str(Path(source_root)/'runtime/bundle'))
    from retrain.models import _deit3_library,_vit_dino3_library
    from retrain.ema import tensor_digest
    import torch,timm
    library=_vit_dino3_library()if dino else _deit3_library()
    state=torch.load(original/'pytorch_model.bin',map_location='cpu',weights_only=True)
    native=timm.create_model(repo.split('/')[1],pretrained=False,img_size=profile['source_size'],
        **({'num_classes':0,'global_pool':'avg','drop_path_rate':.1,'drop_rate':0.}if dino else{}))
    loaded=native.load_state_dict(state,strict=True);actual=native.state_dict()
    require(len(state)==profile['state_tensors'] and not loaded.missing_keys and not loaded.unexpected_keys
        and all(t.dtype==torch.float32 and bool(torch.isfinite(t).all())and torch.equal(actual[n],t)for n,t in state.items()),
        'native full1000 strict/exact/FP32 differs')
    if dino:
        buffers=dict(native.named_buffers())
        require(len(native.blocks)==12 and native.num_prefix_tokens==5 and native.reg_token.shape==(1,4,384)
            and native.num_features==384 and native.pos_embed is None and native.global_pool=='avg'
            and isinstance(native.head,torch.nn.Identity)and isinstance(native.fc_norm,torch.nn.Identity)
            and native.norm.eps==1e-5 and native.blocks[0].attn.qkv.bias is None
            and set(buffers)=={'rope.periods'}and buffers['rope.periods'].shape==(16,)
            and buffers['rope.periods'].dtype==torch.float32 and bool(torch.isfinite(buffers['rope.periods']).all())
            and'rope.periods'not in state,'native DINOv3 public semantics differs')
    else:
        require(native.head.weight.shape==(1000,384)and len(native.blocks)==12 and native.no_embed_class
            and native.pos_embed.shape==(1,576,384)and len(dict(native.named_buffers()))==0,
            'native public non-tensor structure differs')
    eps=1e-5 if dino else 1e-6
    provenance={'model':model,'format':profile['format'],'revision':revision,'source_url':base+'/pytorch_model.bin',
        'training_data':profile['training'],'sha256':weight_sha,'original_sha256':weight_sha,'native_library':library,
        'license_file':'original/'+license_name,'license_sha256':records[license_name]['sha256'],
        'redistribution_confirmed':True,'public_tensor_count':profile['state_tensors'],'public_tensor_sha256':tensor_digest(state),
        'full_head_shape':None if dino else[1000,384],'final_norm_eps':eps,'internal_norm_eps':eps,'converted':False}
    if dino:provenance.update(kind='self_supervised',repo=repo,nonpersistent_buffers_sha256=tensor_digest(buffers))
    write_exact_json(root/'PROVENANCE.json',provenance)
    handoff={'schema_version':1,'passed':True,'model':model,'revision':revision,'execution':context,
        'native_weights':{'path':str(original/'pytorch_model.bin'),'sha256':weight_sha},
        'provenance':{'path':str(root/'PROVENANCE.json'),'sha256':sha256(root/'PROVENANCE.json')},
        'license':{'path':str(original/license_name),'sha256':provenance['license_sha256']},
        'public_tensor_count':profile['state_tensors'],'all_public_tensors_exact':True,'all_public_tensors_FP32_finite':True,
        'forward_calls':0,'datasets_read':False,'GPU_initialized':torch.cuda.is_initialized(),'records':records,
        'source_sha256':sha256(__file__)}
    require(not handoff['GPU_initialized'],'GPU initialized during CPU source check')
    write_exact_json(root/'PUBLIC-HANDOFF.json',handoff)
    return {'status':'native_public_DINOv3_strict_no_conversion_no_forward'if dino else'native_public384_strict_no_conversion_no_forward','weight_sha256':weight_sha,
        'provenance_sha256':handoff['provenance']['sha256'],'handoff_sha256':sha256(root/'PUBLIC-HANDOFF.json'),
        'license_sha256':provenance['license_sha256'],'public_tensor_count':profile['state_tensors'],'asset_root':str(root)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True);parser.add_argument('--transport-dir',required=True)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--source-profile',choices=SOURCE_PROFILES,default='deit3');args=parser.parse_args()
    print(json.dumps(prepare(args.assets,args.transport_dir,args.source_root,args.source_profile),indent=2),flush=True)

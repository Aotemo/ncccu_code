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

def prepare(assets,transport,source_root):
    context=cpu_context();configure_torch_threads(context)
    root=Path(assets);require(root.is_absolute()and'/exp-v3/'in str(root),'exp-v3 asset root required')
    original=root/'original';original.mkdir(parents=True,exist_ok=True)
    relay=Path(transport);records={};base=f'https://huggingface.co/{REPO}/resolve/{REVISION}'
    for name,url in [('registry-fixed.json',f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'),
                     ('config.json',base+'/config.json'),('README.md',base+'/README.md'),
                     ('LICENSE-apache-2.0.txt','https://www.apache.org/licenses/LICENSE-2.0.txt')]:
        records[name]=verified_fetch(url,original/name,relay/name)
    api=json.loads((original/'registry-fixed.json').read_text())
    require(api['id']==REPO and api['sha']==REVISION and api['private']is False and api['gated']is False
            and api['cardData']['license']=='apache-2.0','fixed public registry differs')
    row=next(s for s in api['siblings']if s['rfilename']=='pytorch_model.bin')
    require(row['lfs']['sha256']==WEIGHT_SHA and row['size']==WEIGHT_BYTES,'advertised public bin differs')
    config=json.loads((original/'config.json').read_text());cfg=config['pretrained_cfg']
    require((config['architecture'],config['num_classes'],config['num_features'])==(MODEL,1000,384)
        and cfg['tag']=='fb_in22k_ft_in1k'and cfg['input_size']==[3,384,384]
        and cfg['mean']==[.485,.456,.406]and cfg['std']==[.229,.224,.225]
        and cfg['crop_pct']==1. and cfg['first_conv']=='patch_embed.proj'and cfg['classifier']=='head',
        'native public configuration differs')
    card=(original/'README.md').read_text();license_text=(original/'LICENSE-apache-2.0.txt').read_text()
    require('license: apache-2.0'in card and'ImageNet-22k'in card and'ImageNet-1k'in card,'public model card differs')
    require('Apache License'in license_text and'END OF TERMS AND CONDITIONS'in license_text,'license text incomplete')
    records['pytorch_model.bin']=verified_fetch(base+'/pytorch_model.bin',original/'pytorch_model.bin',relay/'pytorch_model.bin',WEIGHT_SHA,WEIGHT_BYTES)
    sys.path.insert(0,str(Path(source_root)/'runtime/bundle'))
    from retrain.models import DEIT3_LIBRARY,_deit3_library
    from retrain.ema import tensor_digest
    import torch,timm
    _deit3_library();state=torch.load(original/'pytorch_model.bin',map_location='cpu',weights_only=True)
    native=timm.create_model(REPO.split('/')[1],pretrained=False,img_size=384)
    loaded=native.load_state_dict(state,strict=True);actual=native.state_dict()
    require(len(state)==176 and not loaded.missing_keys and not loaded.unexpected_keys
        and all(t.dtype==torch.float32 and bool(torch.isfinite(t).all())and torch.equal(actual[n],t)for n,t in state.items()),
        'native full1000 strict/exact/FP32 differs')
    require(native.head.weight.shape==(1000,384)and len(native.blocks)==12 and native.no_embed_class
        and native.pos_embed.shape==(1,576,384)and len(dict(native.named_buffers()))==0,
        'native public non-tensor structure differs')
    provenance={'model':MODEL,'format':FORMAT,'revision':REVISION,'source_url':base+'/pytorch_model.bin',
        'training_data':TRAINING,'sha256':WEIGHT_SHA,'original_sha256':WEIGHT_SHA,'native_library':DEIT3_LIBRARY,
        'license_file':'original/LICENSE-apache-2.0.txt','license_sha256':records['LICENSE-apache-2.0.txt']['sha256'],
        'redistribution_confirmed':True,'public_tensor_count':176,'public_tensor_sha256':tensor_digest(state),
        'full_head_shape':[1000,384],'final_norm_eps':1e-6,'internal_norm_eps':1e-6,'converted':False}
    write_exact_json(root/'PROVENANCE.json',provenance)
    handoff={'schema_version':1,'passed':True,'model':MODEL,'revision':REVISION,'execution':context,
        'native_weights':{'path':str(original/'pytorch_model.bin'),'sha256':WEIGHT_SHA},
        'provenance':{'path':str(root/'PROVENANCE.json'),'sha256':sha256(root/'PROVENANCE.json')},
        'license':{'path':str(original/'LICENSE-apache-2.0.txt'),'sha256':provenance['license_sha256']},
        'public_tensor_count':176,'all_public_tensors_exact':True,'all_public_tensors_FP32_finite':True,
        'forward_calls':0,'datasets_read':False,'GPU_initialized':torch.cuda.is_initialized(),'records':records,
        'source_sha256':sha256(__file__)}
    require(not handoff['GPU_initialized'],'GPU initialized during CPU source check')
    write_exact_json(root/'PUBLIC-HANDOFF.json',handoff)
    return {'status':'native_public384_strict_no_conversion_no_forward','weight_sha256':WEIGHT_SHA,
        'provenance_sha256':handoff['provenance']['sha256'],'handoff_sha256':sha256(root/'PUBLIC-HANDOFF.json'),
        'license_sha256':provenance['license_sha256'],'public_tensor_count':176,'asset_root':str(root)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True);parser.add_argument('--transport-dir',required=True)
    parser.add_argument('--source-root',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.assets,args.transport_dir,args.source_root),indent=2),flush=True)

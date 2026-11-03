"""Register the verified full SigLIP2 image encoder; no new forward or conversion."""
import argparse
import json
import sys
from pathlib import Path
from source_chain import cpu_context,configure_torch_threads,require,sha256
from deit3_source import write_exact_json


def prepare(assets,source_root):
    context=cpu_context();configure_torch_threads(context)
    root=Path(assets)
    require(root.is_absolute()and '/exp-v3/'in str(root),'exp-v3 source root required')
    sys.path.insert(0,str(Path(source_root)/'runtime/bundle'))
    from retrain.siglip2_model import VARIANT,FORMAT,REVISION,WEIGHT_SHA,LICENSE_SHA,LIBRARY,library_identity
    strict=root/'PUBLIC-SOURCE-STRICT.json'
    require(sha256(strict)=='85618b47515a75d5dfcff0190691efa06b12613e39f94cce03a49d70ecb5c9c2',
            'original full image-encoder strict evidence differs')
    proof=json.loads(strict.read_text());weight=root/'original/pytorch_model.bin'
    license_path=root/'LICENSE-apache-2.0.txt'
    require(proof['passed']is True and proof['execution']['uid']==context['uid']
        and proof['revision']==REVISION and proof['weight']['sha256']==WEIGHT_SHA
        and proof['public_tensor_count']==162 and proof['public_parameters']==93520128
        and proof['strict_missing_keys']==[]and proof['strict_unexpected_keys']==[]
        and proof['all_public_tensors_exact']is True and proof['all_public_tensors_FP32_finite']is True
        and proof['forward_calls']==0 and proof['GPU_initialized']is False and proof['datasets_read']is False,
        'full public image encoder strict identity differs')
    require(weight.is_file()and not weight.is_symlink()and weight.stat().st_size==374141074
        and sha256(weight)==WEIGHT_SHA and sha256(license_path)==LICENSE_SHA,
        'public source bytes/license differ')
    require(library_identity()==LIBRARY,'two native implementation pins differ')
    cfg=proof['source_configuration'];pretrained=cfg['pretrained_cfg']
    require(cfg['architecture']=='vit_base_patch16_siglip_512'and cfg['num_classes']==0
        and cfg['global_pool']=='map'and pretrained['tag']=='v2_webli'
        and pretrained['input_size']==[3,512,512]and pretrained['mean']==pretrained['std']==[.5,.5,.5],
        'fixed native source configuration differs')
    provenance={'schema_version':1,'model':VARIANT,'format':FORMAT,'revision':REVISION,
        'source_url':proof['weight']['url'],'training_data':'WebLI (natural image-text; SigLIP2 image encoder)',
        'sha256':WEIGHT_SHA,'original_sha256':WEIGHT_SHA,'native_library':LIBRARY,
        'redistribution_confirmed':True,'license_file':license_path.name,'license_sha256':LICENSE_SHA,
        'declared_license':'Apache-2.0','public_tensor_count':162,'source_configuration':cfg,
        'strict_record':{'path':str(strict),'sha256':sha256(strict)},
        'fixed_config':proof['fixed_config'],'fixed_card':proof['fixed_card'],
        'library_scope':'two named implementation files, not every transitive timm file',
        'source_producer_sha256':sha256(__file__)}
    path=root/'PROVENANCE.json';write_exact_json(path,provenance)
    handoff={'schema_version':1,'passed':True,'model':VARIANT,'revision':REVISION,'execution':context,
        'native_weights':{'path':str(weight),'sha256':WEIGHT_SHA},
        'provenance':{'path':str(path),'sha256':sha256(path)},'public_tensor_count':162,
        'source_strict':{'path':str(strict),'sha256':sha256(strict)},'forward_calls':0,
        'GPU_initialized':False,'datasets_read':False,
        'scope':'register verified full native image encoder including public MAP; no new load/forward'}
    handoff_path=root/'PUBLIC-HANDOFF.json';write_exact_json(handoff_path,handoff)
    print(json.dumps({'provenance':handoff['provenance'],
        'handoff':{'path':str(handoff_path),'sha256':sha256(handoff_path)}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True);parser.add_argument('--source-root',required=True)
    args=parser.parse_args();prepare(args.assets,args.source_root)

"""Register the independently strict-loaded fixed public EVA02 full1000 asset."""
import argparse,json,sys
from pathlib import Path
from source_chain import cpu_context,require,sha256
from deit3_source import write_exact_json

def prepare(assets,source_root,strict_sha):
    context=cpu_context();root=Path(assets)
    require(root.is_absolute()and '/exp-v3/'in str(root),'exp-v3 asset root required')
    sys.path.insert(0,str(Path(source_root)/'runtime/bundle'))
    from retrain.eva02_model import VARIANT,FORMAT,REVISION,WEIGHT_SHA,LIBRARY,library_identity
    strict=root/'PUBLIC-SOURCE-STRICT.json';require(sha256(strict)==strict_sha,'strict evidence SHA differs')
    proof=json.loads(strict.read_text());weight=root/'original/pytorch_model.bin';license_path=root/'original/LICENSE-MIT'
    require(proof['passed']is True and proof['execution']['uid']==context['uid']
        and proof['revision']==REVISION and proof['model']==VARIANT
        and proof['weight']['sha256']==WEIGHT_SHA and proof['public_tensor_count']==236
        and proof['public_parameters']==87117544 and proof['source_head_shape']==[1000,768]
        and proof['strict_missing_keys']==proof['strict_unexpected_keys']==[]
        and proof['all_public_tensors_exact']is True and proof['all_public_tensors_FP32_finite']is True
        and proof['forward_calls']==0 and proof['GPU_initialized']is False and proof['datasets_read']is False,
        'complete native EVA02 public strict proof differs')
    require(weight.is_file()and not weight.is_symlink()and weight.stat().st_size==348553277
        and sha256(weight)==WEIGHT_SHA and sha256(license_path)==proof['license']['sha256'],
        'public EVA02 weight/license bytes differ')
    require(library_identity()==proof['native_library']==LIBRARY,'four native implementation pins differ')
    cfg=proof['source_configuration'];pretrained=cfg['pretrained_cfg']
    require(cfg['architecture']=='eva02_base_patch14_448'and cfg['num_classes']==1000 and cfg['global_pool']=='avg'
        and pretrained['license']=='mit'and pretrained['input_size']==[3,448,448]
        and pretrained['mean']==[.48145466,.4578275,.40821073]
        and pretrained['std']==[.26862954,.26130258,.27577711],'fixed native source preprocessing differs')
    for item in proof['fixed_files'].values():require(sha256(item['path'])==item['sha256'],'source metadata bytes differ')
    provenance={'schema_version':1,'model':VARIANT,'format':FORMAT,'revision':REVISION,
        'source_url':proof['weight']['url'],'training_data':'ImageNet-22K masked-image pretraining -> ImageNet-22K -> ImageNet-1K fine-tuning',
        'sha256':WEIGHT_SHA,'original_sha256':WEIGHT_SHA,'native_library':LIBRARY,
        'redistribution_confirmed':True,'license_file':'original/LICENSE-MIT','license_sha256':proof['license']['sha256'],
        'declared_license':'MIT','public_tensor_count':236,'source_configuration':cfg,
        'strict_record':{'path':str(strict),'sha256':strict_sha},'fixed_files':proof['fixed_files'],
        'library_scope':'four named implementation files, not all transitive timm files','source_producer_sha256':sha256(__file__)}
    path=root/'PROVENANCE.json';write_exact_json(path,provenance)
    handoff={'schema_version':1,'passed':True,'model':VARIANT,'revision':REVISION,'execution':context,
        'native_weights':{'path':str(weight),'sha256':WEIGHT_SHA},'provenance':{'path':str(path),'sha256':sha256(path)},
        'public_tensor_count':236,'source_strict':{'path':str(strict),'sha256':strict_sha},'forward_calls':0,
        'GPU_initialized':False,'datasets_read':False,'scope':'register complete native full1000 public source; no new load/forward'}
    handoff_path=root/'PUBLIC-HANDOFF.json';write_exact_json(handoff_path,handoff)
    print(json.dumps({'provenance':handoff['provenance'],'handoff':{'path':str(handoff_path),'sha256':sha256(handoff_path)},
        'license_sha256':proof['license']['sha256']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--assets',required=True)
    parser.add_argument('--source-root',required=True);parser.add_argument('--strict-sha',required=True)
    args=parser.parse_args();prepare(args.assets,args.source_root,args.strict_sha)

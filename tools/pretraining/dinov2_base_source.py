"""Register the original natural-image DINOv2 Base asset; no new model/forward."""
import argparse,json,sys
from pathlib import Path
from source_chain import cpu_context,require,sha256
from deit3_source import write_exact_json

def prepare(assets,source_root,strict_sha,guard_path,guard_sha):
    context=cpu_context();root=Path(assets);strict=root/'PUBLIC-SOURCE-STRICT.json'
    require(root.is_absolute()and '/exp-v3/'in str(root),'exp-v3 public asset root required')
    require(sha256(strict)==strict_sha and sha256(guard_path)==guard_sha,'public source evidence SHA differs')
    proof=json.loads(strict.read_text());guard=json.loads(Path(guard_path).read_text())
    require(guard['exit_code']==0 and guard['task_tree_exited'] and not guard['signals_sent']
        and not guard['affinity_violations'] and guard['runnable_thread_budget_respected']
        and guard['identity']['NCCCU_POD_UID']==context['uid']
        and sha256(Path(guard_path).parent/'OUTPUT.txt')==guard['output_sha256'],
        'original strict CPU task evidence differs')
    sys.path.insert(0,str(Path(source_root)/'runtime/bundle'))
    from retrain.dinov2_model import BASE_FORMAT,BASE_WEIGHT_SHA,BASE_PROVENANCE_SHA,base_provenance,public_code_root
    init={'kind':'self_supervised','format':BASE_FORMAT,'sha256':BASE_WEIGHT_SHA,
        'path':str(root/'original/dinov2_vitb14_pretrain.pth'),'provenance':str(root/'PROVENANCE.json')}
    meta=base_provenance(init,verify_weight=True);public_code_root({'model':'dinov2_vitb14','initialization':init})
    require(proof['status']=='original_public175_strict_exact_FP32_finite'
        and proof['model']=='dinov2_vitb14'and proof['revision']==meta['revision']
        and proof['original']=={'path':init['path'],'sha256':BASE_WEIGHT_SHA,'bytes':346378731}
        and proof['public_tensor_count']==proof['parameter_tensors']==175 and proof['parameter_elements']==86580480
        and proof['strict_missing_keys']==proof['strict_unexpected_keys']==[]
        and proof['all_public_tensors_exact']is True and proof['all_public_tensors_FP32_finite']is True
        and proof['backbone_only']is True and proof['full_head_shape']is None and proof['buffer_count']==0
        and proof['position_shape']==[1,1370,768] and proof['image_forward_calls']==0
        and proof['GPU_initialized']is False and proof['dataset_or_official_read']is False
        and proof['provenance_sha256']==BASE_PROVENANCE_SHA,'original Base175 strict proof differs')
    handoff={'schema_version':1,'passed':True,'model':'dinov2_vitb14','revision':meta['revision'],
        'execution':context,'native_weights':{'path':init['path'],'sha256':BASE_WEIGHT_SHA},
        'provenance':{'path':init['provenance'],'sha256':BASE_PROVENANCE_SHA},'public_tensor_count':175,
        'source_files_sha256':meta['source_files_sha256'],'source_strict':{'path':str(strict),'sha256':strict_sha},
        'strict_CPU_guard':{'path':str(guard_path),'sha256':guard_sha},'backbone_only':True,'full_head_shape':None,
        'forward_calls':0,'GPU_initialized':False,'datasets_read':False,
        'scope':'register original natural-image full175 source and pinned code; no model construction or forward'}
    output=root/'PUBLIC-HANDOFF.json';write_exact_json(output,handoff)
    print(json.dumps({'handoff':{'path':str(output),'sha256':sha256(output)},
        'provenance':handoff['provenance'],'public_tensor_count':175},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('assets','source-root','strict-sha','guard-path','guard-sha'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();prepare(args.assets,args.source_root,args.strict_sha,args.guard_path,args.guard_sha)

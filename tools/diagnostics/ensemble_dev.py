"""Fixed F01 dev-only logit blend; reuse selected-state and pair-error audits."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def logit_grid(labels, left, right, denominator=20):
    from retrain.n05_metrics import metrics
    y=np.asarray(labels);a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64)
    if (y.ndim!=1 or a.shape!=y.shape or b.shape!=y.shape or set(y.tolist())!={0,1}
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or type(denominator)is not int or not 1<=denominator<=1000):
        raise ValueError('aligned finite binary inputs and bounded integer grid required')
    rows=[]
    for k in range(denominator+1):
        w=k/denominator;score=(1-w)*a+w*b
        actual=metrics([{'label':int(label),'logit':float(value)}for label,value in zip(y,score)])
        rows.append({'R120_weight':1-w,'R238_weight':w,**actual})
    best=min(rows,key=lambda row:(row['error_twice'],-row['ap'],row['R238_weight']))
    w=best['R238_weight']
    return {'grid':rows,'selected':dict(best),'scores':(1-w)*a+w*b}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def verify_git_inputs(metadata,commit,config):
    remote=json.loads((metadata/'REMOTE-COMMIT.json').read_text())
    tree=json.loads((metadata/'REMOTE-TREE.json').read_text())
    if remote['sha']!=commit or remote['tree']['sha']!=tree['sha'] or tree['truncated']is not False:
        raise ValueError('connector-verified commit/tree identity differs')
    entries={row['path']:row for row in tree['tree']if row['type']=='blob'}
    files={'tools/diagnostics/ensemble_dev.py':Path(__file__), 'configs/F01-LOGIT-GRID.json':config,
           'tests/test_ensemble_dev.py':Path(__file__).resolve().parents[2]/'tests/test_ensemble_dev.py'}
    for name,path in files.items():
        raw=path.read_bytes();mode='100755'if path.stat().st_mode&0o111 else'100644'
        if (path.is_symlink() or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entries[name]['sha']
                or mode!=entries[name]['mode']):
            raise ValueError('published Git blob bytes/mode differ: '+name)
    return {name:sha(path)for name,path in files.items()}


def analyze(args):
    source=args.source_root.resolve(strict=True)
    sys.path[:0]=[str(source/'runtime/bundle'),str(source/'tools/diagnostics'),str(source/'tools/pretraining')]
    from source_chain import cpu_context
    import paired_evidence as pe
    context=cpu_context()
    git_files=verify_git_inputs(args.remote_metadata,args.git_commit,args.config)
    cfg=json.loads(args.config.read_text())
    if (cfg['members']!=['R120','R238'] or cfg['weight_denominator']!=20 or cfg['view']!='identity'
            or cfg['TTA']is not False or cfg['official_read_or_forward']is not False
            or cfg['external_medical_pretraining']is not False or cfg['accumulation_dtype']!='float64'):
        raise ValueError('F01 preregistered members/grid/inference boundary differs')
    inv=pe._pinned_json(cfg['inventory']);reference=pe._reference(inv['manifest'])
    if inv['manifest']['sha256']!=cfg['manifest_sha256']:raise ValueError('F01 fixed manifest differs')
    labels=[row['label']for row in reference['dev']];fixed=inv['fixed_positive_indices']
    if (len(labels)!=5140 or labels.count(1)!=167 or len(fixed)!=20 or len(set(fixed))!=20
            or any(type(i)is not int or not 0<=i<len(labels)or labels[i]!=1 for i in fixed)):
        raise ValueError('fixed dev population or hard group differs')
    work=Path(inv['manifest']['path']).parents[2]
    out=args.output.resolve()
    if not out.is_relative_to(work/'technical-prep')or out.exists():raise ValueError('new isolated technical-prep output required')
    out.mkdir(parents=True)
    import torch
    torch.set_num_threads(max(1,int(os.environ['NCCCU_CPU_THREADS'])-2));torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():raise ValueError('CPU analysis cannot initialize GPU')
    from retrain.persistence import load_file
    from retrain.ema import select_variant,tensor_digest
    started=time.time();vectors={};members={}
    for name in cfg['members']:
        specs=[row for row in inv['models']if row['id']==name]
        if len(specs)!=1:raise ValueError('unique preregistered member missing: '+name)
        scores,selected,_=pe._selected(specs[0],reference)
        freeze=pe._pinned_json(specs[0]['freeze']);round_root=Path(specs[0]['freeze']['path']).parent
        science=json.loads((round_root/'SCIENCE.json').read_text());public=science['public_source']
        if (public['model']!='convnext_small' or public['revision']!='fe6b358aa409414cdb522ae3d43a2c9922f7cc1b'
                or public['sha256']!=cfg['public_weight_sha256']
                or public['training_data']!='ImageNet-22K pretraining -> ImageNet-1K fine-tuning'
                or science['job'].get('initial_checkpoint')is not None
                or freeze['blob_sha256']!=cfg['selected_blob_sha256'][name]
                or selected['tensor_sha256']!=cfg['selected_tensor_sha256'][name] or selected['variant']!='ema'):
            raise ValueError('general public source or selected blob/tensor identity differs: '+name)
        matches=[]
        for path in sorted(round_root.glob('outputs/*/*/checkpoints/checkpoint-*.pt')):
            side=Path(str(path)+'.json')
            if side.is_file()and json.loads(side.read_text())['sha256']==freeze['blob_sha256']:matches.append(path)
        if len(matches)!=1:raise ValueError('unique actual selected checkpoint missing: '+name)
        checkpoint=matches[0];load_started=time.time();payload=load_file(checkpoint,map_location='cpu')
        state=select_variant(payload,'ema');actual_tensor=tensor_digest(state)
        if actual_tensor!=selected['tensor_sha256']or payload['successful_updates']!=selected['successful_updates']:
            raise ValueError('actually loaded EMA differs: '+name)
        job=payload['job']
        if (job['model'],job['image_size'],job['prepared_size'],job['normalization'],job['precision'])!=('convnext_small',512,512,'imagenet','fp32'):
            raise ValueError('loaded member input/precision differs')
        members[name]={'selected':selected,'checkpoint':{'path':str(checkpoint),'blob_sha256':freeze['blob_sha256'],
                      'EMA_tensor_sha256':actual_tensor,'sidecar_sha256':sha(str(checkpoint)+'.json')},
                      'science_sha256':sha(round_root/'SCIENCE.json'),'public_source':public,
                      'current_CPU_load_seconds':time.time()-load_started}
        vectors[name]=scores;del state,payload
        print(json.dumps({'stage':'actual_member_preflight_passed','member':name,'tensor_sha256':actual_tensor}),flush=True)
    write(out/'PREFLIGHT.json',{'status':'both_actual_EMA_blobs_general_sources_and_selected_dev_verified',
          'at':time.time(),'execution':context,'members':members,'manifest':inv['manifest'],
          'model_constructed':False,'forward_calls':0,'GPU_initialized':False,'git_commit':args.git_commit,'git_files':git_files})
    grid_started=time.time();result=logit_grid(labels,vectors['R120'],vectors['R238'],cfg['weight_denominator'])
    for actual,name in ((result['grid'][0],'R120'),(result['grid'][-1],'R238')):
        if any(abs(actual[k]-members[name]['selected'][k])>1e-12 for k in ('auroc','ap','error_twice')):
            raise ValueError('grid vertex differs from original selected metrics')
    changes={name:pe._error_changes(labels,vectors[name],result['scores'],fixed)for name in cfg['members']}
    if any(row['all']['candidate']!=result['selected']['error_twice']/2 for row in changes.values()):
        raise ValueError('fused pair mass differs from exact metric')
    csvpath=out/'dev-fused.csv'
    with csvpath.open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=['name','label','logit']);writer.writeheader()
        for row,score in zip(reference['dev'],result['scores']):writer.writerow({'name':row['name'],'label':row['label'],'logit':float(score)})
    report={'status':'completed_dev_only_fusion_analysis','at':time.time(),'id':cfg['id'],
        'git_commit':args.git_commit,'git_files':git_files,'support_source_root':str(source),
        'support_files':{name:sha(source/name)for name in ('tools/diagnostics/paired_evidence.py','tools/diagnostics/persistent_errors.py',
                 'runtime/bundle/retrain/n05_metrics.py','runtime/bundle/retrain/persistence.py','runtime/bundle/retrain/ema.py')},
        'config_sha256':sha(args.config),'preflight_sha256':sha(out/'PREFLIGHT.json'),
        'manifest':inv['manifest'],'members':members,'grid':result['grid'],'selected':result['selected'],
        'changes':changes,'fixed_positive_count':20,'remaining_positive_count':147,
        'count':5140,'positive_count':167,'negative_count':4973,'accumulation_dtype':'float64',
        'strict_AUROC_gain_over_R120':result['selected']['auroc']>members['R120']['selected']['auroc'],
        'raw_dev_auroc_0995_met':result['selected']['auroc']>=.995,'dev_AP_above_09_met':result['selected']['ap']>.9,
        'prediction':{'path':str(csvpath),'sha256':sha(csvpath)},'grid_wall_seconds':time.time()-grid_started,
        'total_wall_seconds':time.time()-started,'official_labels_read':False,'official_forward':False,
        'TTA':False,'member_checkpoint_selection_changed':False,'training_modified':False,
        'forward_calls':0,'GPU_initialized':torch.cuda.is_initialized(),
        'limits':['Weights selected on repeatedly reused internal dev; this is not an independent generalization estimate.',
                  'No official score or overall-goal completion is established by this dev-only analysis.']}
    write(out/'RESULT.json',report)
    print(json.dumps({'status':report['status'],'output':str(out),'result_sha256':sha(out/'RESULT.json'),
          'selected':report['selected'],'changes':changes,'total_wall_seconds':report['total_wall_seconds']},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','config','remote-metadata','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--git-commit',required=True)
    args=p.parse_args();analyze(args)


if __name__=='__main__':main()

"""Fixed F01 dev-only logit blend; reuse selected-state and pair-error audits."""
import argparse
import csv
import hashlib
import io
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


def standardized_equal_schemes(labels, vectors, member_order, top_k=5):
    """The two preregistered equal blends; moments use no labels."""
    from retrain.n05_metrics import metrics
    y=np.asarray(labels);order=list(member_order)
    if (y.ndim!=1 or set(y.tolist())!={0,1} or len(order)<2 or len(set(order))!=len(order)
            or set(order)!=set(vectors) or type(top_k)is not int or top_k<1):
        raise ValueError('aligned binary population, distinct members and fixed top K required')
    normalized={};moments={};singles={}
    def measure(values):
        return metrics([{'label':int(label),'logit':float(value)}for label,value in zip(y,values)])
    for name in order:
        values=np.asarray(vectors[name],dtype=np.float64)
        if values.shape!=y.shape or not np.isfinite(values).all():
            raise ValueError('finite aligned member scores required')
        mean=float(values.mean(dtype=np.float64));std=float(values.std(dtype=np.float64,ddof=0))
        if not np.isfinite(mean)or not np.isfinite(std)or std<=0:
            raise ValueError('positive finite member variance required')
        moments[name]={'mean':mean,'std':std};normalized[name]=(values-mean)/std
        singles[name]=measure(values)
    top=sorted(order,key=lambda name:(singles[name]['error_twice'],order.index(name)))[:min(top_k,len(order))]
    candidates=[('all_equal_z',order)]
    if set(top)!=set(order):candidates.append(('top_k_equal_z',top))
    schemes=[];scores={}
    for identity,names in candidates:
        score=np.stack([normalized[name]for name in names]).mean(axis=0,dtype=np.float64)
        scores[identity]=score
        schemes.append({'id':identity,'members':list(names),'weights':{name:1/len(names)for name in names},**measure(score)})
    selected=min(schemes,key=lambda row:(row['error_twice'],-row['ap'],len(row['members']),row['id']!='all_equal_z'))
    return {'normalization':moments,'single_members':singles,'schemes':schemes,'selected':dict(selected),'scores':scores}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_historical_load_receipt(name,row,path,expected_sha):
    """Bind a prior real load using the member's frozen runtime to this inventory."""
    path=Path(path)
    if sha(path)!=expected_sha:raise ValueError('historical load receipt SHA differs')
    receipt=json.loads(path.read_text())
    if (receipt.get('status')!='actual_blob_and_selected_tensor_loaded_with_original_frozen_runtime'
            or receipt.get('member')!=name or receipt.get('freeze')!=row['spec']['freeze']
            or receipt.get('blob_sha256')!=row['selected_blob_sha256']
            or receipt.get('selected_tensor_sha256')!=row['selected_tensor_sha256']
            or receipt.get('variant')!=row['selected']['variant']
            or receipt.get('successful_updates')!=row['selected']['successful_updates']
            or receipt.get('checkpoint')!=row['selected_checkpoint_candidates'][0]
            or type(receipt.get('frozen_source_files_verified'))is not int
            or receipt['frozen_source_files_verified']<1
            or receipt.get('model_constructed')is not False or receipt.get('forward_calls')!=0
            or receipt.get('GPU_initialized')is not False or receipt.get('official_read')is not False):
        raise ValueError('historical load receipt identity differs')
    checkpoint=receipt['checkpoint']
    if not Path(checkpoint['path']).is_file()or sha(checkpoint['sidecar']['path'])!=checkpoint['sidecar']['sha256']:
        raise ValueError('historical loaded checkpoint or pinned sidecar no longer present')
    return {'receipt':{'path':str(path),'sha256':expected_sha},'checkpoint':checkpoint,
            'tensor_sha256':receipt['selected_tensor_sha256'],'source_commit':receipt['source_commit']}


def write(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def verify_git_inputs(metadata,commit,config):
    remote=json.loads((metadata/'REMOTE-COMMIT.json').read_text())
    tree=json.loads((metadata/'REMOTE-TREE.json').read_text())
    if remote['sha']!=commit or remote['tree']['sha']!=tree['sha'] or tree['truncated']is not False:
        raise ValueError('connector-verified commit/tree identity differs')
    entries={row['path']:row for row in tree['tree']if row['type']=='blob'}
    if config.name not in ('F01-LOGIT-GRID.json','F02-HETEROGENEOUS-EQUAL.json','F03-THRESHOLD-EQUAL.json'):
        raise ValueError('registered fusion configuration required')
    files={'tools/diagnostics/ensemble_dev.py':Path(__file__), 'configs/'+config.name:config,
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
        # The historical site identifies this model's EMA rule, never the current
        # execution Pod. Match the frozen inference entry's model-identity binding.
        model_site=round_root/'SITE_BINDING.json'
        keys=('NCCCU_SITE_BINDING','NCCCU_SITE_BINDING_SHA256')
        previous={key:os.environ.get(key)for key in keys}
        try:
            os.environ.update(NCCCU_SITE_BINDING=str(model_site),NCCCU_SITE_BINDING_SHA256=sha(model_site))
            state=select_variant(payload,'ema');actual_tensor=tensor_digest(state)
        finally:
            for key,value in previous.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
        if actual_tensor!=selected['tensor_sha256']or payload['successful_updates']!=selected['successful_updates']:
            raise ValueError('actually loaded EMA differs: '+name)
        job=payload['job']
        if (job['model'],job['image_size'],job['prepared_size'],job['normalization'],job['precision'])!=('convnext_small',512,512,'imagenet','fp32'):
            raise ValueError('loaded member input/precision differs')
        members[name]={'selected':selected,'checkpoint':{'path':str(checkpoint),'blob_sha256':freeze['blob_sha256'],
                      'EMA_tensor_sha256':actual_tensor,'sidecar_sha256':sha(str(checkpoint)+'.json')},
                      'science_sha256':sha(round_root/'SCIENCE.json'),'public_source':public,
                      'model_identity_site':{'path':str(model_site),'sha256':sha(model_site),'execution_binding_replaced':False},
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


def analyze_equal(args):
    """Saved predictions support research even when a historical weight was removed."""
    source=args.source_root.resolve(strict=True)
    sys.path[:0]=[str(source/'runtime/bundle'),str(source/'tools/diagnostics'),str(source/'tools/pretraining')]
    from source_chain import cpu_context
    import paired_evidence as pe
    from retrain.n05_metrics import metrics
    import torch
    from retrain.persistence import load_file
    from retrain.ema import select_variant,tensor_digest
    context=cpu_context();started=time.time()
    torch.set_num_threads(max(1,int(os.environ['NCCCU_CPU_THREADS'])-2));torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():raise ValueError('CPU analysis cannot initialize GPU')
    git_files=verify_git_inputs(args.remote_metadata,args.git_commit,args.config)
    cfg=json.loads(args.config.read_text())
    order=['R120','R134R1','R201','R197','R135','R107','R172','R180','R116','R182','R154']
    if (cfg['members']!=order or cfg['top_k']!=5 or cfg['normalization']!='dev_float64_zscore_ddof0'
            or cfg['view']!='identity' or cfg['TTA']is not False or cfg['official_read_or_forward']is not False
            or cfg['external_medical_pretraining']is not False or cfg['checkpoint_presence_required_for_prediction_research']is not False):
        raise ValueError('F02 preregistered equal schemes or prediction-only boundary differs')
    inv=pe._pinned_json(cfg['inventory']);reference=pe._reference(inv['manifest'])
    labels=[row['label']for row in reference['dev']];fixed=inv['fixed_positive_indices']
    if (list(inv['candidate_order'])!=order or inv['manifest']['sha256']!=cfg['manifest_sha256']
            or len(labels)!=5140 or labels.count(1)!=167 or len(fixed)!=20 or len(set(fixed))!=20
            or any(type(i)is not int or not 0<=i<len(labels)or labels[i]!=1 for i in fixed)):
        raise ValueError('F02 frozen inventory or internal population differs')
    work=Path(inv['manifest']['path']).parents[2];out=args.output.resolve()
    if not out.is_relative_to(work/'technical-prep')or out.exists():
        raise ValueError('new isolated technical-prep output required')
    external={}
    for name,path,digest in args.historical_load_receipt:
        if name not in order or name in external:raise ValueError('unique registered historical member required')
        external[name]=(Path(path),digest)
    out.mkdir(parents=True);vectors={};members={};excluded=[];loadable=[];weight_failures=[]
    for name in order:
        rows=[row for row in inv['members']if row['id']==name]
        if len(rows)!=1:raise ValueError('unique registered member inventory required')
        row=rows[0]
        try:
            if not row['status'].startswith('selected_prediction_verified'):
                raise ValueError('inventory member failed: '+row.get('reason',row['status']))
            science=pe._pinned_json(row['science']);public=science['public_source']
            expected=json.loads((source/'configs'/(name+'-SCIENCE.json')).read_text())
            if (public!=row['public_source'] or public['sha256']not in cfg['approved_general_public_weight_sha256']
                    or any(science.get(key)!=expected.get(key)for key in('candidate_profile','candidate','public_source','structure','job','ssl_recipe'))
                    or science['job'].get('initial_checkpoint')is not None):
                raise ValueError('frozen general public source or scientific identity differs')
            scores,selected,_=pe._selected(row['spec'],reference)
            if selected!=row['selected']:raise ValueError('selected prediction identity changed since inventory')
            vector=np.asarray(scores,dtype=np.float64)
            if not np.isfinite(vector).all()or vector.std(ddof=0)<=0:
                raise ValueError('positive finite member variance required')
            checkpoints=row['selected_checkpoint_candidates']
            present=len(checkpoints)==1 and Path(checkpoints[0]['path']).is_file()
            if present:
                side=pe._pinned_json(checkpoints[0]['sidecar'])
                if side['sha256']!=row['selected_blob_sha256']:raise ValueError('selected sidecar changed')
            members[name]={'selected':selected,'spec':row['spec'],'science':row['science'],'public_source':public,
                'selected_blob_sha256':row['selected_blob_sha256'],'selected_tensor_sha256':row['selected_tensor_sha256'],
                'selected_checkpoint_file_present':bool(present),'checkpoint_load_verified_by_this_analysis':False,
                'input':{key:science['job'].get(key)for key in('model','image_size','prepared_size','normalization','precision','age','native_local_view')}}
            vectors[name]=vector
        except (OSError,ValueError,KeyError,AssertionError)as error:
            excluded.append({'id':name,'reason':str(error),'type':type(error).__name__})
            continue
        if not present:
            weight_failures.append({'id':name,'reason':'unique selected checkpoint missing','type':'MissingCheckpoint'})
            continue
        try:
            if name in external:
                members[name]['historical_actual_load']=verify_historical_load_receipt(name,row,*external[name])
                members[name]['checkpoint_load_verified_by_this_analysis']=True
                loadable.append(name)
                print(json.dumps({'stage':'pinned_historical_selected_checkpoint_load_accepted','member':name}),flush=True)
                continue
            checkpoint=Path(checkpoints[0]['path']);model_site=Path(row['science']['path']).parent/'SITE_BINDING.json'
            payload=load_file(checkpoint,map_location='cpu')
            keys=('NCCCU_SITE_BINDING','NCCCU_SITE_BINDING_SHA256')
            previous={key:os.environ.get(key)for key in keys}
            try:
                os.environ.update(NCCCU_SITE_BINDING=str(model_site),NCCCU_SITE_BINDING_SHA256=sha(model_site))
                from retrain import experiment91
                experiment91._metadata.cache_clear()
                state=select_variant(payload,selected['variant']);actual_tensor=tensor_digest(state)
            finally:
                for key,value in previous.items():
                    if value is None:os.environ.pop(key,None)
                    else:os.environ[key]=value
            if actual_tensor!=row['selected_tensor_sha256']or payload['successful_updates']!=selected['successful_updates']:
                raise ValueError('loaded selected tensor or successful update differs')
            for key in ('model','image_size','prepared_size','normalization','precision'):
                if payload['job'].get(key)!=science['job'].get(key):
                    raise ValueError('loaded checkpoint input identity differs: '+key)
            members[name]['checkpoint_load_verified_by_this_analysis']=True
            members[name]['loaded_checkpoint']={'path':str(checkpoint),'blob_sha256':row['selected_blob_sha256'],
                'tensor_sha256':actual_tensor,'variant':selected['variant'],
                'sidecar_sha256':checkpoints[0]['sidecar']['sha256'],
                'model_identity_site':{'path':str(model_site),'sha256':sha(model_site)}}
            loadable.append(name)
            print(json.dumps({'stage':'actual_selected_checkpoint_loaded','member':name,'tensor_sha256':actual_tensor}),flush=True)
        except Exception as error:
            weight_failures.append({'id':name,'reason':str(error),'type':type(error).__name__})
        finally:
            if 'state'in locals():del state
            if 'payload'in locals():del payload
    admitted=[name for name in order if name in members]
    if 'R120'not in members or len(admitted)<2:raise ValueError('R120 and another verified saved-prediction member required')
    write(out/'PREFLIGHT.json',{'status':'prediction_and_weight_loadable_members_fixed_before_equal_fusion_and_correlations',
        'at':time.time(),'execution':context,'ordered_members':admitted,'members':members,'excluded':excluded,
        'manifest':inv['manifest'],'inventory':cfg['inventory'],'git_commit':args.git_commit,'git_files':git_files,
        'weight_loadable_members':loadable,'weight_load_failures':weight_failures,
        'checkpoint_weights_loaded':bool(loadable),'model_constructed':False,'forward_calls':0,'official_read':False})
    result=standardized_equal_schemes(labels,vectors,admitted,top_k=5)
    subset=standardized_equal_schemes(labels,{name:vectors[name]for name in loadable},loadable,top_k=5) if len(loadable)>=2 else None
    f01=pe._pinned_json(cfg['F01_reference']);raw=pe._pinned_bytes(f01['prediction'])
    f01_rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(f01_rows)!=len(labels)or any(item['name']!=row['name']or item['label']!=str(row['label'])for item,row in zip(f01_rows,reference['dev'])):
        raise ValueError('frozen F01 population differs')
    f01_scores=np.asarray([float(row['logit'])for row in f01_rows],dtype=np.float64)
    if not np.isfinite(f01_scores).all():raise ValueError('nonfinite frozen F01 score')
    f01_metrics=metrics([{'label':int(y),'logit':float(s)}for y,s in zip(labels,f01_scores)])
    if any(f01_metrics[key]!=f01['selected'][key]for key in('auroc','ap','error_twice')):
        raise ValueError('F01 baseline differs from its frozen result')
    changes={};predictions={}
    for scheme in result['schemes']:
        identity=scheme['id'];scores=result['scores'][identity]
        changes[identity]={name:pe._error_changes(labels,base,scores,fixed)
            for name,base in(('R120',vectors['R120']),('F01',f01_scores))}
        if any(value['all']['candidate']!=scheme['error_twice']/2 for value in changes[identity].values()):
            raise ValueError('equal fusion pair mass differs from metric')
        path=out/('dev-'+identity+'.csv')
        with path.open('x',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=['name','label','logit']);writer.writeheader()
            for row,score in zip(reference['dev'],scores):writer.writerow({'name':row['name'],'label':row['label'],'logit':float(score)})
        predictions[identity]={'path':str(path),'sha256':sha(path)}
    subset_report={'status':'insufficient_loaded_members_for_fusion','members':loadable,'failures':weight_failures}
    if subset:
        subset_changes={};subset_predictions={}
        for scheme in subset['schemes']:
            identity=scheme['id'];scores=subset['scores'][identity]
            subset_changes[identity]={name:pe._error_changes(labels,base,scores,fixed)
                for name,base in (('R120',vectors['R120']),('F01',f01_scores))}
            if any(value['all']['candidate']!=scheme['error_twice']/2 for value in subset_changes[identity].values()):
                raise ValueError('loadable subset pair mass differs from metric')
            path=out/('dev-loadable-'+identity+'.csv')
            with path.open('x',newline='')as stream:
                writer=csv.DictWriter(stream,fieldnames=['name','label','logit']);writer.writeheader()
                for row,score in zip(reference['dev'],scores):writer.writerow({'name':row['name'],'label':row['label'],'logit':float(score)})
            subset_predictions[identity]={'path':str(path),'sha256':sha(path)}
        subset_report={'status':'completed_weight_loadable_subset_dev_only','members':loadable,'failures':weight_failures,
            'normalization':subset['normalization'],'schemes':subset['schemes'],'selected':subset['selected'],
            'changes':subset_changes,'predictions':subset_predictions}
    matrix=np.stack([vectors[name]for name in admitted])
    rank_rows=[]
    for values in matrix:
        _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
        ends=np.cumsum(counts);rank_rows.append(((ends-counts+ends+1)/2)[inverse])
    pearson=np.corrcoef(matrix);spearman=np.corrcoef(np.stack(rank_rows))
    if (not np.isfinite(pearson).all()or not np.isfinite(spearman).all()
            or not np.allclose(np.diag(pearson),1)or not np.allclose(np.diag(spearman),1)):
        raise ValueError('finite member correlation identity differs')
    overlaps=[]
    for i,left in enumerate(admitted):
        for right in admitted[i+1:]:
            overlaps.append({'reference':left,'candidate':right,'error_mass':pe._error_changes(labels,vectors[left],vectors[right],fixed)})
    selected=result['selected'];missing=[name for name in selected['members']if not members[name]['selected_checkpoint_file_present']]
    report={'status':'completed_dev_only_standardized_equal_fusion','id':cfg['id'],'at':time.time(),
        'git_commit':args.git_commit,'git_files':git_files,'config_sha256':sha(args.config),
        'support_source_root':str(source),'support_files':{name:sha(source/name)for name in(
            'tools/diagnostics/paired_evidence.py','tools/diagnostics/persistent_errors.py','runtime/bundle/retrain/n05_metrics.py')},
        'preflight_sha256':sha(out/'PREFLIGHT.json'),'manifest':inv['manifest'],'members':members,'excluded':excluded,
        'normalization':result['normalization'],'single_members':result['single_members'],'schemes':result['schemes'],
        'selected':selected,'predictions':predictions,'F01_reference':cfg['F01_reference'],'F01_metrics':f01_metrics,
        'changes':changes,'correlations':{'member_order':admitted,'Pearson':pearson.tolist(),'Spearman':spearman.tolist()},
        'pair_error_overlaps':overlaps,'fixed_positive_count':20,'remaining_positive_count':147,
        'weight_loadable_subset':subset_report,
        'count':5140,'positive_count':167,'negative_count':4973,'selected_member_checkpoint_files_missing':missing,
        'deployment_ready':False,'checkpoint_weights_loaded':bool(loadable),'new_forward_calls':0,'GPU_initialized':torch.cuda.is_initialized(),
        'raw_dev_auroc_0995_met':selected['auroc']>=.995,'dev_AP_above_09_met':selected['ap']>.9,
        'official_labels_read':False,'official_forward':False,'TTA':False,'training_modified':False,
        'total_wall_seconds':time.time()-started,
        'limits':['Only two fixed equal schemes; selection still reuses internal dev.',
                  'Weight-loadable subset proves selected blob/tensor loading, not model construction or full deployment readiness.',
                  'A correction using historical load receipts may have been run after an earlier subset result was seen.',
                  'No official result or overall-goal completion is established.']}
    write(out/'RESULT.json',report)
    print(json.dumps({'status':report['status'],'result_sha256':sha(out/'RESULT.json'),'output':str(out),
        'selected':selected,'missing_checkpoint_members':missing,'changes':changes,
        'weight_loadable_subset':subset_report},indent=2),flush=True)


def f03_candidate_gate(row,minimum,approved):
    if row['status'] in ('unknown_or_excluded','selected_checkpoint_ambiguous'):
        return ('pending',row['status'])
    if row['status']!='prediction_verified_weight_pending':return ('excluded',row['status'])
    auroc=row['selected']['auroc']
    if isinstance(auroc,bool)or not isinstance(auroc,(int,float))or not np.isfinite(auroc)or not 0<=auroc<=1:
        return ('pending','invalid_selected_auroc')
    if auroc<minimum:return ('excluded','below_fixed_selected_auroc_threshold')
    public=row.get('public_source')or{};digest=public.get('sha256')
    if digest not in approved or approved[digest]!=public.get('training_data'):
        return ('pending','general_public_source_not_yet_verified')
    return ('eligible',None)


def verify_constructed_member(row,path,expected_sha):
    path=Path(path)
    if sha(path)!=expected_sha:raise ValueError('model construction receipt SHA differs')
    receipt=json.loads(path.read_text());selected=row['selected'];checkpoints=row['checkpoint_candidates']
    if (len(checkpoints)!=1 or receipt.get('status')!='strict_CPU_model_construction_and_selected_state_loaded'
            or receipt.get('member')!=row['round_id'].split('-')[0].upper()
            or receipt.get('freeze')!=row['spec']['freeze'] or receipt.get('checkpoint')!=checkpoints[0]
            or receipt.get('blob_sha256')!=row['selected_blob_sha256']
            or receipt.get('selected_tensor_sha256')!=selected['tensor_sha256']
            or receipt.get('variant')!=selected['variant'] or receipt.get('successful_updates')!=selected['successful_updates']
            or receipt.get('model_constructed')is not True or receipt.get('strict_state_load')is not True
            or receipt.get('device')!='cpu' or receipt.get('forward_calls')!=0
            or receipt.get('GPU_initialized')is not False or receipt.get('official_read')is not False):
        raise ValueError('model construction receipt identity differs')
    checkpoint=checkpoints[0]
    if not Path(checkpoint['path']).is_file()or sha(checkpoint['sidecar']['path'])!=checkpoint['sidecar']['sha256']:
        raise ValueError('selected checkpoint or sidecar no longer present')
    return {'path':str(path),'sha256':expected_sha}


def discover_f03(args):
    """Seal a fresh ordered census of complete same-population selected predictions."""
    source=args.source_root.resolve(strict=True)
    sys.path[:0]=[str(source/'runtime/bundle'),str(source/'tools/diagnostics'),str(source/'tools/pretraining')]
    from source_chain import cpu_context
    import paired_evidence as pe
    context=cpu_context();git_files=verify_git_inputs(args.remote_metadata,args.git_commit,args.config)
    cfg=json.loads(args.config.read_text());old=pe._pinned_json(cfg['reference_inventory'])
    if old['manifest']['sha256']!=cfg['manifest_sha256']:raise ValueError('fixed F03 reference differs')
    reference=pe._reference(old['manifest']);work=Path(old['manifest']['path']).parents[2]
    out=args.output.resolve()
    if not out.is_relative_to(work/'technical-prep')or out.exists():raise ValueError('new isolated technical-prep output required')
    out.mkdir(parents=True);rows=[]
    def ref(path):return {'path':str(path),'sha256':sha(path)}
    for root in sorted((work/'rounds').iterdir(),key=lambda p:p.name):
        done=root/'control/done.json';freeze=root/'SINGLE-MODEL-FREEZE.json';training=root/'evidence/TRAIN_RESULT.json';science_path=root/'SCIENCE.json'
        if root.is_symlink()or not all(p.is_file()for p in(done,freeze,training,science_path)):continue
        try:
            if json.loads(done.read_text()).get('status')!='completed':continue
            science=json.loads(science_path.read_text());frozen=json.loads(freeze.read_text())
            if frozen.get('experiment91',{}).get('manifest_sha256')!=cfg['manifest_sha256']:continue
            spec={'id':root.name,'freeze':ref(freeze),'completion':ref(training)}
            _,selected,_=pe._selected(spec,reference);checkpoints=[]
            for path in root.glob('outputs/*/*/checkpoints/checkpoint-*.pt'):
                side=Path(str(path)+'.json')
                if side.is_file()and json.loads(side.read_text()).get('sha256')==frozen.get('blob_sha256'):
                    checkpoints.append({'path':str(path),'sidecar':ref(side),'bytes':path.stat().st_size})
            state=('prediction_verified_weight_pending'if len(checkpoints)==1 else
                   'selected_checkpoint_missing'if not checkpoints else 'selected_checkpoint_ambiguous')
            rows.append({'round_id':root.name,'status':state,
                'selected':selected,'spec':spec,'science':ref(science_path),'public_source':science.get('public_source'),
                'checkpoint_candidates':checkpoints,'selected_blob_sha256':frozen.get('blob_sha256')})
        except Exception as error:
            rows.append({'round_id':root.name,'status':'unknown_or_excluded','type':type(error).__name__,'reason':str(error)})
    value={'status':'read_only_F03_candidate_discovery_no_fusion','at':time.time(),'execution':context,
        'git_commit':args.git_commit,'git_files':git_files,'manifest':old['manifest'],'members':rows,
        'fixed_positive_indices':old['fixed_positive_indices'],'checkpoint_load_or_model_construction':False,
        'official_read':False,'forward_calls':0,'GPU_initialized':False}
    write(out/'CANDIDATES.json',value)
    print(json.dumps({'status':value['status'],'candidate_count':len(rows),'output_sha256':sha(out/'CANDIDATES.json'),
        'weight_present':sum(r['status']=='prediction_verified_weight_pending'for r in rows),
        'unknown_count':sum(r['status']=='unknown_or_excluded'for r in rows)},indent=2),flush=True)


def analyze_f03(args):
    """The preregistered AUROC threshold followed by one all-member equal blend."""
    source=args.source_root.resolve(strict=True)
    sys.path[:0]=[str(source/'runtime/bundle'),str(source/'tools/diagnostics'),str(source/'tools/pretraining')]
    from source_chain import cpu_context
    import paired_evidence as pe
    context=cpu_context();git_files=verify_git_inputs(args.remote_metadata,args.git_commit,args.config)
    cfg=json.loads(args.config.read_text())
    if (cfg['id']!='F03-general-threshold-standardized-equal' or cfg['minimum_selected_auroc']!=.988
            or cfg['normalization']!='dev_float64_zscore_ddof0' or cfg['weights']!='equal_all_admitted'
            or cfg['TTA']is not False or cfg['official_read_or_forward']is not False
            or cfg['external_medical_pretraining']is not False):raise ValueError('fixed F03 contract differs')
    if not args.f03_candidate_inventory:raise ValueError('pinned F03 candidate discovery required')
    inv_path,inv_sha=args.f03_candidate_inventory;inv=pe._pinned_json({'path':inv_path,'sha256':inv_sha})
    if (inv['status']!='read_only_F03_candidate_discovery_no_fusion' or inv['official_read']is not False
            or inv['manifest']['sha256']!=cfg['manifest_sha256']):raise ValueError('F03 discovery or population differs')
    reference=pe._reference(inv['manifest']);labels=[row['label'] for row in reference['dev']]
    fixed=inv['fixed_positive_indices']
    if len(labels)!=5140 or labels.count(1)!=167 or len(fixed)!=20:raise ValueError('F03 fixed dev population differs')
    out=args.output.resolve();work=Path(inv['manifest']['path']).parents[2]
    if not out.is_relative_to(work/'technical-prep')or out.exists():raise ValueError('new isolated technical-prep output required')
    receipts={}
    for name,path,digest in args.model_load_receipt:
        if name in receipts:raise ValueError('duplicate model construction receipt')
        receipts[name]=(path,digest)
    vectors={};members={};excluded=[];pending=[]
    for row in inv['members']:
        name=row['round_id']
        state,reason=f03_candidate_gate(row,cfg['minimum_selected_auroc'],cfg['approved_general_public_weight_sha256'])
        if state=='excluded':
            excluded.append({'round_id':name,'reason':reason,
                'selected_auroc':row.get('selected',{}).get('auroc')});continue
        if state=='pending':pending.append({'round_id':name,'reason':reason});continue
        source_proof=row.get('public_source')or{}
        if name not in receipts:
            pending.append({'round_id':name,'reason':'strict_model_construction_receipt_missing'});continue
        try:
            receipt=verify_constructed_member(row,*receipts[name])
            science=pe._pinned_json(row['science'])
            if (science['public_source']!=source_proof or science['job'].get('initial_checkpoint')is not None
                    or science['candidate'].get('weight_sha256')!=source_proof['sha256']):
                raise ValueError('frozen general source or initialization identity differs')
            scores,selected,_=pe._selected(row['spec'],reference)
            if selected!=row['selected']:raise ValueError('selected identity changed after candidate discovery')
            values=np.asarray(scores,dtype=np.float64)
            if not np.isfinite(values).all()or values.std(ddof=0)<=0:raise ValueError('invalid member logit variance')
            vectors[name]=values;members[name]={'selected':selected,'science':row['science'],
                'public_source':source_proof,'checkpoint':row['checkpoint_candidates'][0],'model_load_receipt':receipt}
        except (OSError,ValueError,KeyError,AssertionError)as error:
            pending.append({'round_id':name,'reason':str(error),'type':type(error).__name__})
    if pending:raise ValueError('F03 eligible candidate verification incomplete: '+json.dumps(pending))
    order=[row['round_id']for row in inv['members']if row['round_id']in vectors]
    out.mkdir(parents=True)
    write(out/'PREFLIGHT.json',{'status':'all_admitted_members_fixed_before_F03_fusion','at':time.time(),
        'execution':context,'candidate_inventory':{'path':inv_path,'sha256':inv_sha},'ordered_members':order,
        'members':members,'excluded':excluded,'threshold':cfg['minimum_selected_auroc'],
        'git_commit':args.git_commit,'git_files':git_files,'official_read':False,'forward_calls':0})
    if len(order)<2:
        write(out/'RESULT.json',{'status':'insufficient_members_for_fusion','members':order,'preflight_sha256':sha(out/'PREFLIGHT.json')})
        print(json.dumps({'status':'insufficient_members_for_fusion','result_sha256':sha(out/'RESULT.json'),'members':order}),flush=True)
        return
    blend=standardized_equal_schemes(labels,vectors,order,top_k=len(order));scores=blend['scores']['all_equal_z']
    baseline=next((name for name in order if name.startswith('r120-')),None)
    changes=pe._error_changes(labels,vectors[baseline],scores,fixed)if baseline else None
    path=out/'dev-f03-equal.csv'
    with path.open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=['name','label','logit']);writer.writeheader()
        for row,score in zip(reference['dev'],scores):writer.writerow({'name':row['name'],'label':row['label'],'logit':float(score)})
    report={'status':'completed_F03_dev_only_fixed_threshold_equal','id':cfg['id'],'at':time.time(),
        'git_commit':args.git_commit,'git_files':git_files,'preflight_sha256':sha(out/'PREFLIGHT.json'),
        'members':members,'excluded':excluded,'normalization':blend['normalization'],'scheme':blend['schemes'][0],
        'changes_vs_R120':changes,'prediction':{'path':str(path),'sha256':sha(path)},
        'count':5140,'positive_count':167,'negative_count':4973,'official_read':False,'TTA':False,
        'forward_calls':0,'raw_dev_auroc_0995_met':blend['schemes'][0]['auroc']>=.995,
        'dev_AP_above_09_met':blend['schemes'][0]['ap']>.9,
        'limits':['Same internal dev was repeatedly used; result is not independent generalization.',
                  'Actual CPU strict model loading does not replace a final deployed inference validation.']}
    write(out/'RESULT.json',report)
    print(json.dumps({'status':report['status'],'result_sha256':sha(out/'RESULT.json'),
        'preflight_sha256':report['preflight_sha256'],'scheme':report['scheme'],'changes_vs_R120':changes},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','config','remote-metadata','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--git-commit',required=True)
    p.add_argument('--historical-load-receipt',nargs=3,action='append',default=[],metavar=('MEMBER','PATH','SHA256'))
    p.add_argument('--f03-candidate-inventory',nargs=2,metavar=('PATH','SHA256'))
    p.add_argument('--model-load-receipt',nargs=3,action='append',default=[],metavar=('ROUND','PATH','SHA256'))
    p.add_argument('--f03-discover',action='store_true')
    args=p.parse_args()
    if args.config.name=='F03-THRESHOLD-EQUAL.json'and args.f03_discover:discover_f03(args)
    elif args.config.name=='F03-THRESHOLD-EQUAL.json':analyze_f03(args)
    elif args.config.name=='F02-HETEROGENEOUS-EQUAL.json':analyze_equal(args)
    else:analyze(args)


if __name__=='__main__':main()

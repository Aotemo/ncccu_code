"""One predeclared fit subset, existing inference/evaluate, no training."""
import csv,hashlib,json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);source=Path(sys.argv[2]);formal=Path(sys.argv[3]);mode=sys.argv[4]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
result_path=Path(sys.argv[5]);assert sha(result_path)=='2d29288a7d8fb0017a1d8d875e21a72d63dad1300899d665db15ee57fa84e634'
result=read(result_path)
assert result['status']=='completed'and result['actual_safe']and result['selected_tensor_sha256']=='677606c3efe813469c73f91212719a61c3f1bb03421d13239723799827ea9444'
site_path=formal/'SITE_BINDING.json';site=read(site_path)
assert site['pod']['uid']==os.environ['NCCCU_POD_UID']and site['git_commit']=='87cc9198faa392c5cb9ed8346e3d14dd19f72516'
manifest_path=Path(site['metadata_root'])/'manifest.json';manifest=read(manifest_path)
rows=manifest['rows'];fit=manifest['fit_indices'];dev=manifest['dev_indices']
assert len(fit)==46256 and len(dev)==5140 and not set(fit)&set(dev)
assert all(rows[i]['relative_path'].startswith('train/')for i in fit+dev)
reference_path=Path('/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r213-fit-subset-20261103-32e3/SEALED-SUBSET.json')
assert sha(reference_path)=='f5f774b41069d3ede4f5bb8c5d0d326b18fdb1993aac2eea22da4cfddf167088'
reference=read(reference_path)
assert sha(manifest_path)==reference['manifest']['sha256']=='7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377'
import numpy as np
if mode=='seal':
    root.mkdir(exist_ok=False)
    rng=np.random.Generator(np.random.PCG64(20261103));indices=[]
    for label in(1,0):
        pool=[i for i in fit if rows[i]['label']==label]
        assert len(pool)==(1490 if label else 44766)
        indices.extend(map(int,rng.choice(pool,128,replace=False)))
    indices.sort();assert len(set(indices))==256 and set(indices)<=set(fit)
    assert indices==reference['indices']
    selected=result['terminal']['freeze']['selected'];prediction=Path(selected['prediction_path'])
    assert sha(prediction)==selected['prediction_sha256']and sha(selected['precision_path'])==selected['precision_sha256']
    run_dir=prediction.parents[2];index_path=run_dir/'checkpoints/index.json';index=read(index_path)
    record=index['protected']['best_ema'];checkpoint=run_dir/'checkpoints'/record['file']
    assert record['sha256']==result['selected_blob_sha256']and checkpoint.is_file()and not checkpoint.is_symlink()
    spec={'at':time.time(),'seed':20261103,'generator':'PCG64','sample_without_replacement':True,
        'indices':indices,'counts':{'total':256,'positive':128,'negative':128},'manifest':{'path':str(manifest_path),'sha256':sha(manifest_path)},
        'site':{'path':str(site_path),'sha256':sha(site_path)},'checkpoint':{'path':str(checkpoint),'sha256':record['sha256']},
        'selected_tensor_sha256':result['selected_tensor_sha256'],'selected_successful_updates':4800,'variant':'ema',
        'original_sealed_subset':{'path':str(reference_path),'sha256':sha(reference_path),'indices_exact':True},
        'source':str(source),'source_files':{n:sha(source/n)for n in('runtime/bundle/retrain/engine.py','runtime/bundle/retrain/inference.py','runtime/bundle/retrain/data.py','runtime/bundle/retrain/experiment91.py')},
        'original_complete_dev':selected,'GPU_forward':False,'new_forward_started':False,'producer_sha256':sha(__file__)}
    write(root/'SEALED-SUBSET.json',spec);print(json.dumps({'sealed':str(root/'SEALED-SUBSET.json'),'sha256':sha(root/'SEALED-SUBSET.json'),'counts':spec['counts']},indent=2))
elif mode=='evaluate':
    spec=read(root/'SEALED-SUBSET.json')
    assert sha(manifest_path)==spec['manifest']['sha256']and sha(site_path)==spec['site']['sha256']
    assert sha(__file__)==spec['producer_sha256']
    for name,digest in spec['source_files'].items():assert sha(source/name)==digest
    os.environ.update(NCCCU_SITE_BINDING=str(site_path),NCCCU_SITE_BINDING_SHA256=spec['site']['sha256'],
        TRAIN_IMAGE_CACHE=str(Path(site['prepared_cache'])/'train/sealed'),TRAIN_IMAGE_CACHE_READONLY='1',
        TRAIN_VERIFICATION_CACHE=str(root/'verification'))
    sys.path.insert(0,str(source/'runtime/bundle'))
    import torch
    torch.set_num_threads(5);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    from retrain.inference import _load_model
    from retrain.engine import evaluate
    from retrain.ema import tensor_digest
    started=time.time();model,job,metadata,origin,device=_load_model(spec['checkpoint']['path'],'cpu',variant='ema')
    assert origin['sha256']==spec['checkpoint']['sha256']and origin['tensor_sha256']==spec['selected_tensor_sha256']
    assert origin['successful_updates']==4800 and device.type=='cpu'
    assert spec['indices']==reference['indices']
    before=tensor_digest(model.state_dict());assert before==spec['selected_tensor_sha256']
    execution_job={**job,'physical_batch':4,'eval_workers':0};output=root/'fit-subset-selected.csv'
    assert not output.exists()
    values=evaluate(model,rows,spec['indices'],site['data_root'],execution_job,device,output)
    assert values['count']==256 and values['positive_count']==128 and values['negative_count']==128
    assert tensor_digest(model.state_dict())==before and not torch.cuda.is_initialized()
    report={'status':'completed_predeclared_partial_fit_only','at':time.time(),'wall_seconds':time.time()-started,
        'seal_sha256':sha(root/'SEALED-SUBSET.json'),'producer_sha256':sha(__file__),'metrics':values,
        'original_complete_dev':spec['original_complete_dev'],'origin':origin,'output_sha256':sha(output),
        'actual_tensor_before_after':before,'image_forward_count':256,'GPU_initialized':False,'official_read':False,
        'model_or_frozen_job_modified':False,'execution_overrides':{'physical_batch':4,'eval_workers':0,'torch_threads':5},
        'complete_fit_measured':False,'clinical_cause_identified':False,
        'limitations':['Balanced256 subset is not full46256 fit. AP/threshold metrics do not share dev prevalence.',
            'Fit/dev AUC gap includes sampling and dev model selection bias, and does not identify a mechanism or ceiling.']}
    write(root/'SUMMARY.json',report);print(json.dumps({'summary_sha256':sha(root/'SUMMARY.json'),'metrics':values,'wall_seconds':report['wall_seconds']},indent=2))
else:raise ValueError('seal or evaluate only')

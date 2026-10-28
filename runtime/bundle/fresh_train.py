#!/usr/bin/env python3
"""Sealed public fresh initialization; probe and formal have independent identities."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import json
from pathlib import Path
import shutil
import signal
import sys
import threading
import uuid


def runtime_contract(cfg):
    value=copy.deepcopy(cfg);value.pop('resume',None)
    return value


def worker(args):
    spec=json.loads(Path(args.worker_spec).read_text())
    if args.cpu_affinity:os.sched_setaffinity(0,set(map(int,args.cpu_affinity.split(','))))
    import torch
    torch.set_num_threads(1)
    from retrain.distributed import RankContext
    from retrain.distributed_engine import run_distributed
    ctx=RankContext(args.rank,6,args.device,spec['init_method'])
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        job=copy.deepcopy(spec['job']);job['device']=args.device
        if spec['fresh_spec'].get('cache_readonly_required'):
            from pair_runtime import rank_device_proof
            from retrain.persistence import atomic_json
            proofs=ctx.values(rank_device_proof(args.rank,ctx.device))
            if ctx.primary:atomic_json(Path(spec['run_dir'])/'rank-device-proof.json',proofs)
        if spec['cpu_width']<14:raise ValueError('frozen stage requires 14 CPU cores per rank')
        result=run_distributed(job,json.loads(Path(spec['manifest_path']).read_text()),spec['data_root'],
            spec['run_dir'],spec['identity'],ctx,resume=spec['resume'],mode=spec['mode'],
            fresh_spec=spec['fresh_spec'],stop_event=stop,deadline_at=spec['deadline_at'])
        return 0 if result['status']=='completed' else 130
    finally:ctx.close()


def run(args):
    import train
    from distributed_train import cpu_layout
    from retrain.config import load_config,job_contract
    from retrain.data import load_source,build_manifest
    from retrain.models import validate_initialization
    from retrain.distributed import layout,RankBatchSampler
    from retrain.distributed_engine import initialize_store
    from retrain.distributed_launcher import supervise_ranks
    from retrain.persistence import RunLock,atomic_json,canonical_digest,sha256_file
    from retrain.fresh import validate_fresh
    cfg=load_config(args.config,overrides={'resume':args.resume})
    fresh=json.loads(Path(args.fresh_spec).read_text())
    import math,time
    if args.deadline_at is None or not math.isfinite(args.deadline_at) or args.deadline_at<=time.time():
        raise ValueError('fresh runtime requires an unexpired original absolute deadline')
    if (cfg['mode']!='train' or len(cfg['jobs'])!=1 or cfg['execution']['devices']!=['cuda:'+str(i) for i in range(6)]
            or cfg['data']['split']['kind']!='holdout' or cfg['data']['split']['include_valid']
            or not cfg['data']['verify_images']):raise ValueError('frozen phase requires six-card train holdout')
    if sha256_file(args.plan)!=fresh['plan_sha256']:raise ValueError('frozen PLAN hash differs')
    if fresh.get('cache_readonly_required'):
        if os.environ.get('TRAIN_IMAGE_CACHE_READONLY')!='1' or not os.environ.get('TRAIN_IMAGE_CACHE'):
            raise ValueError('R8 requires sealed fail-closed read-only prepared cache')
    job=copy.deepcopy(cfg['jobs'][0]);plan=layout(job['physical_batch'],job['effective_batch'],job['positive_per_batch'],6)
    if (job['model']!='convnext_tiny' or job['physical_batch']!=96 or job['effective_batch']!=192
            or job['positive_per_batch']!=24 or not job['age']['enabled']):
        raise ValueError('R9 requires registered ConvNeXt age world6/global192/reference96')
    plan['stop_after_updates']=fresh['schedule']['stop_after_updates'];job['physical_batch']=plan['per_rank_batch']
    report=train.environment_report(cfg)
    if report['errors']:raise ValueError('stage environment failed: '+str(report['errors']))
    width,cpus=cpu_layout(6)
    if width!=14:raise ValueError('CPU quota does not permit the frozen 14-core rank layout')
    root=Path(cfg['output_root'])/cfg['run_name']
    os.environ['TRAIN_VERIFICATION_CACHE']=os.environ.get('TRAIN_VERIFICATION_CACHE',str(Path(cfg['output_root'])/'.data-verification-cache'))
    mode='fresh-probe' if fresh['purpose']=='probe' else 'train'
    with RunLock(root.parent/('.'+root.name+'.lock')):
        if root.exists()!=cfg['resume']:raise ValueError('stage needs a new run-name or exact explicit resume')
        source=load_source(cfg['data']['root'],cfg['data']['identity_path'],True,False,cfg['data']['expected_count'])
        split=dict(cfg['data']['split']);split.update(job['split']);manifest=build_manifest(source,split)
        hashes,digest=train.source_identity()
        identity={'format':'glaucoma-retrain-v1','source_digest':digest,'manifest_digest':canonical_digest(manifest),
          'job':job_contract(job),'initialization':validate_initialization(job['initialization']),
          'phase_parent_sha256':None,'distributed':plan,'fresh':fresh}
        negatives=sum(not manifest['rows'][i]['label'] for i in manifest['fit_indices'])
        import math
        steps=math.ceil(negatives/(job['effective_batch']-job['positive_per_batch']))
        validate_fresh(fresh,identity,job,plan,steps,mode=mode)
        contract={'version':1,'purpose':fresh['purpose'],'operation':mode if mode=='fresh-probe' else 'fresh-init',
                  'absolute_deadline_at':args.deadline_at,'runtime_versions':report['versions'],'jobs':{job['name']:identity},'distributed':plan,
                  'phase_runtime_config':runtime_contract(cfg)}
        if cfg['resume']:
            if json.loads((root/'contract.json').read_text())!=contract:raise ValueError('stage resume full contract differs')
        else:
            staging=root.parent/('.'+root.name+'.initializing-'+uuid.uuid4().hex);staging.mkdir()
            atomic_json(staging/'contract.json',contract);atomic_json(staging/'source-files.json',hashes)
            for relative in hashes:
                target=staging/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(Path(train.__file__).parent/relative,target)
            target=staging/'code/assets/train_identity.json';target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cfg['data']['identity_path'],target);initialize_store(staging/job['name']);os.replace(staging,root)
        atomic_json(root/'config.json',cfg);atomic_json(root/'environment.json',report)
        (root/'complete.json').unlink(missing_ok=True)
        folder=root/job['name'];folder.mkdir(exist_ok=True)
        manifest_path=folder/'manifest.json';atomic_json(manifest_path,manifest);atomic_json(folder/'job.json',job)
        # Stateless IDs and augmentation seeds, before any sampling or GPU update.
        preview=[]
        for rank in range(6):
            batches=list(RankBatchSampler(manifest['rows'],manifest['fit_indices'],plan['reference_physical_batch'],
                job['effective_batch'],job['positive_per_batch'],job['seed'],0,4,
                rank=rank,world_size=6))
            preview.append({'rank':rank,'sha256':canonical_digest(batches),'batches':batches})
        atomic_json(root/'sampler-preview.json',preview)
        rendezvous=Path('/tmp')/('retrain-fresh-'+uuid.uuid4().hex)
        spec={'job':job,'manifest_path':str(manifest_path),'data_root':cfg['data']['root'],'run_dir':str(folder),
          'identity':identity,'resume':cfg['resume'],'mode':mode,'deadline_at':args.deadline_at,'fresh_spec':fresh,'cpu_width':width,
          'init_method':'file://'+str(rendezvous)}
        path=folder/'fresh-worker-spec.json';atomic_json(path,spec)
        tasks=[]
        for rank,device in enumerate(cfg['execution']['devices']):
            argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-spec',str(path),'--rank',str(rank),
                  '--device',device,'--cpu-affinity',','.join(map(str,cpus[rank]))]
            tasks.append({'name':'rank-'+str(rank),'argv':argv})
        atomic_json(root/'status.json',{'status':'starting','mode':mode,'purpose':fresh['purpose'],'distributed':plan})
        from retrain.fresh import start_runtime_guard
        stop=threading.Event()
        guardian=start_runtime_guard(args.deadline_at,stop,lambda reason:atomic_json(root/'runtime-stop.json',{
            'at':time.time(),'reason':reason,'absolute_deadline_at':args.deadline_at}))
        try:
            result=supervise_ranks(tasks,root/'rank-control',cfg['execution']['grace_seconds'],stop_event=stop,env=os.environ.copy())
        finally:stop.set();guardian.join(timeout=2)
        result.update(mode=mode,purpose=fresh['purpose'],distributed=plan,plan_sha256=fresh['plan_sha256'])
        if result['status']=='completed':
            record=json.loads((folder/'complete.json').read_text())
            if record['status']!='completed' or record['mode']!=mode or any(result['exit_codes'].values()):
                raise ValueError('stage worker completion not proved')
            atomic_json(root/'complete.json',result)
        atomic_json(root/'status.json',result);rendezvous.unlink(missing_ok=True)
        return {'completed':0,'interrupted':130,'failed':1}[result['status']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--fresh-spec');parser.add_argument('--plan')
    parser.add_argument('--resume',action='store_true',default=None)
    parser.add_argument('--deadline-at',type=float)
    parser.add_argument('--worker-spec');parser.add_argument('--rank',type=int);parser.add_argument('--device')
    parser.add_argument('--cpu-affinity');args=parser.parse_args()
    return worker(args) if args.worker_spec else run(args)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        import traceback
        print(json.dumps({'status':'failed','error':str(exc)}),flush=True);traceback.print_exc();raise SystemExit(1)

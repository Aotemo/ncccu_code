#!/usr/bin/env python3
"""Single-model DDP entry: 1 / 2 / 4 / 8 cards, audited checkpoint continuation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import json
from pathlib import Path
import shutil
import signal
import socket
import sys
import threading
import uuid


def worker(args):
    spec=json.loads(Path(args.worker_spec).read_text())
    if args.cpu_affinity and hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,set(map(int,args.cpu_affinity.split(','))))
    import torch
    torch.set_num_threads(1)
    from retrain.distributed import RankContext
    from retrain.distributed_engine import run_distributed
    rank=args.rank;world=spec['identity']['distributed']['world_size']
    ctx=RankContext(rank,world,args.device,spec['init_method'])
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        job=copy.deepcopy(spec['job']);job['device']=args.device
        budget=spec['cpu_width']
        job['workers']=min(job['workers'],max(0,budget-2))
        job['eval_workers']=min(job['eval_workers'],max(0,budget-1))
        result=run_distributed(job,json.loads(Path(spec['manifest_path']).read_text()),
            spec['data_root'],spec['run_dir'],spec['identity'],ctx,resume=spec['resume'],mode=spec['mode'],
            parent_spec=spec['parent_spec'],stop_event=stop,limit_updates=spec['limit_updates'])
        return 0 if result['status']=='completed' else 130
    finally:ctx.close()


def cpu_layout(world):
    available=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))
    quota=Path('/sys/fs/cgroup/cpu.max')
    count=len(available)
    if quota.exists():
        q,p=quota.read_text().split()
        if q!='max':count=min(count,max(1,int(q)//int(p)))
    width=max(1,(max(world,count-2))//world)
    cpus=available[:max(world,count)]
    return width,[cpus[r*width:(r+1)*width] for r in range(world)]


def run(args):
    import train
    from retrain.config import load_config,job_contract
    from retrain.data import load_source,build_manifest
    from retrain.models import validate_initialization
    from retrain.distributed import layout
    from retrain.distributed_launcher import supervise_ranks
    from retrain.persistence import RunLock,atomic_json,canonical_digest,sha256_file
    from retrain.distributed_engine import load_parent,initialize_store
    cfg=load_config(args.config,overrides={'resume':args.resume,'devices':args.devices})
    if cfg['mode'] not in ('benchmark','train') or len(cfg['jobs'])!=1:
        raise ValueError('distributed entry requires exactly one train/benchmark model')
    world=args.world_size or len(cfg['execution']['devices'])
    if world!=len(cfg['execution']['devices']):raise ValueError('requested rank/device counts differ')
    reference=copy.deepcopy(cfg['jobs'][0])
    plan=layout(reference['physical_batch'],reference['effective_batch'],reference['positive_per_batch'],world)
    plan['stop_after_updates']=args.limit_updates
    job=copy.deepcopy(reference);job['physical_batch']=plan['per_rank_batch']
    cfg['jobs']=[job]
    if cfg['data']['split']['kind']!='holdout' or cfg['data']['split']['include_valid']:
        raise ValueError('distributed continuation currently permits only sealed train holdout')
    report=train.environment_report(cfg)
    print(json.dumps(report),flush=True)
    if report['errors']:raise ValueError('distributed environment inspection failed')
    root=Path(cfg['output_root'])/cfg['run_name']
    os.environ['TRAIN_VERIFICATION_CACHE']=os.environ.get('TRAIN_VERIFICATION_CACHE',str(Path(cfg['output_root'])/'.data-verification-cache'))
    if not cfg['data']['verify_images']:raise ValueError('distributed training requires on-access image verification')
    parent=json.loads(Path(args.parent_spec).read_text()) if args.parent_spec else None
    if parent and cfg['mode']!='train':raise ValueError('benchmark cannot inherit formal checkpoint')
    with RunLock(root.parent/('.'+root.name+'.lock')):
        if root.exists()!=cfg['resume']:raise ValueError('new run needs new name; continuation needs --resume')
        source=load_source(cfg['data']['root'],cfg['data']['identity_path'],True,False,cfg['data']['expected_count'])
        split=dict(cfg['data']['split']);split.update(job['split'])
        manifest=build_manifest(source,split)
        hashes,digest=train.source_identity()
        identity={'format':'glaucoma-retrain-v1','source_digest':digest,'manifest_digest':canonical_digest(manifest),
            'job':job_contract(job),'initialization':validate_initialization(job['initialization']),
            'phase_parent_sha256':parent['sha256'] if parent else None,'distributed':plan}
        contract={'version':1,'purpose':cfg['mode'],'runtime_versions':report['versions'],
                  'jobs':{job['name']:identity},'distributed':plan}
        if parent:load_parent(parent,identity,job,plan)
        if cfg['resume']:
            if json.loads((root/'contract.json').read_text())!=contract:raise ValueError('DDP resume contract changed')
        else:
            staging=root.parent/('.'+root.name+'.initializing-'+uuid.uuid4().hex);staging.mkdir()
            atomic_json(staging/'contract.json',contract);atomic_json(staging/'source-files.json',hashes)
            for relative in hashes:
                target=staging/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(Path(train.__file__).parent/relative,target)
            target=staging/'code/assets/train_identity.json';target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cfg['data']['identity_path'],target)
            initialize_store(staging/job['name'])
            os.replace(staging,root)
        atomic_json(root/'config.json',cfg);atomic_json(root/'reference-config.json',reference)
        atomic_json(root/'environment.json',report)
        (root/'complete.json').unlink(missing_ok=True)
        folder=root/job['name'];folder.mkdir(exist_ok=True)
        manifest_path=folder/'manifest.json';atomic_json(manifest_path,manifest);atomic_json(folder/'job.json',job)
        width,cpus=cpu_layout(world)
        rendezvous=Path('/tmp')/('retrain-ddp-'+uuid.uuid4().hex)
        spec={'job':job,'manifest_path':str(manifest_path),'data_root':cfg['data']['root'],'run_dir':str(folder),
              'identity':identity,'resume':cfg['resume'],'mode':cfg['mode'],'parent_spec':parent,
              'limit_updates':args.limit_updates,'cpu_width':width,'init_method':'file://'+str(rendezvous)}
        path=folder/'distributed-worker-spec.json';atomic_json(path,spec)
        tasks=[]
        for rank,device in enumerate(cfg['execution']['devices']):
            argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-spec',str(path),
                  '--rank',str(rank),'--device',device]
            if cpus[rank]:argv+=['--cpu-affinity',','.join(map(str,cpus[rank]))]
            tasks.append({'name':'rank-'+str(rank),'argv':argv})
        atomic_json(root/'status.json',{'status':'starting','mode':cfg['mode'],'distributed':plan})
        result=supervise_ranks(tasks,root/'rank-control',cfg['execution']['grace_seconds'],env=os.environ.copy())
        result.update(mode=cfg['mode'],distributed=plan)
        if result['status']=='completed':
            record=json.loads((folder/'complete.json').read_text())
            if record['status']!='completed' or record['mode']!=cfg['mode']:raise ValueError('missing rank0 completion')
            if set(result['exit_codes'])!={'rank-'+str(r) for r in range(world)}:raise ValueError('a rank never completed')
            atomic_json(root/'complete.json',result)
        atomic_json(root/'status.json',result)
        rendezvous.unlink(missing_ok=True)
        return {'completed':0,'interrupted':130,'failed':1}[result['status']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--world-size',type=int,choices=(1,2,4,8))
    parser.add_argument('--devices');parser.add_argument('--resume',action='store_true',default=None)
    parser.add_argument('--parent-spec');parser.add_argument('--limit-updates',type=int)
    parser.add_argument('--worker-spec');parser.add_argument('--rank',type=int);parser.add_argument('--device')
    parser.add_argument('--cpu-affinity')
    args=parser.parse_args()
    return worker(args) if args.worker_spec else run(args)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        import traceback
        print(json.dumps({'status':'failed','error':str(exc)}),flush=True);traceback.print_exc();raise SystemExit(1)

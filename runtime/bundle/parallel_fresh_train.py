#!/usr/bin/env python3
"""Configured-world public fresh training and parallel paired evaluation; legacy R9 entry remains separate."""
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


from retrain.execution_policy import resolve_topology, allocate_cpus, validate_worker_spec


def runtime_contract(cfg):
    value=copy.deepcopy(cfg);value.pop('resume',None)
    return value


def worker(args, *, model_factory=None, stop_event=None):
    spec=json.loads(Path(args.worker_spec).read_text())
    from retrain.execution_policy import validate_worker_spec, rank_proof
    requested=list(map(int,args.cpu_affinity.split(','))) if args.cpu_affinity else []
    world=validate_worker_spec(spec,args.rank,args.device,requested)
    from retrain.execution_policy import bind_worker_cpu
    bind_worker_cpu(spec,args.rank,args.device)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from retrain.distributed import RankContext
    from retrain.distributed_engine import run_distributed
    ctx=RankContext(args.rank,world,args.device,spec['init_method'],timeout_seconds=90)
    stop=stop_event if stop_event is not None else threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        job=copy.deepcopy(spec['job']);job['device']=args.device
        from retrain.persistence import atomic_json
        proofs=ctx.values(rank_proof(ctx,spec))
        from retrain.site_binding import load_site, verify_rank_devices, sha256 as site_sha
        site=load_site()
        if spec['site_binding_sha256']!=site_sha(os.environ['NCCCU_SITE_BINDING']):
            raise ValueError('worker site binding changed')
        verify_rank_devices(proofs,site)
        if ctx.primary:atomic_json(Path(spec['run_dir'])/'rank-device-proof.json',proofs)
        job['workers']=site['cpu_execution']['loader_workers']
        job['eval_workers']=site['cpu_execution']['loader_workers']
        derived_receipt=None
        if spec['fresh_spec'].get('ssl_recipe') is not None:
            if model_factory is not None or spec['resume']:
                raise ValueError('R115 SSL requires a fresh native model factory')
            from retrain.ssl_stage import run_ssl,load_derived_model
            derived_receipt=run_ssl(job,json.loads(Path(spec['manifest_path']).read_text()),
                spec['data_root'],spec['run_dir'],spec['fresh_spec']['ssl_recipe'],ctx,
                barrier_path=spec['fresh_spec']['ssl_startup_barrier'],
                deadline_at=spec['deadline_at'],identity=spec['identity'],site=site,
                precision_protocol=spec['fresh_spec']['precision_protocol'],stop_event=stop)
            model_factory=lambda supplied, initialize:load_derived_model(supplied,initialize,derived_receipt)
        result=run_distributed(job,json.loads(Path(spec['manifest_path']).read_text()),spec['data_root'],
            spec['run_dir'],spec['identity'],ctx,resume=spec['resume'],mode=spec['mode'],
            fresh_spec=spec['fresh_spec'],stop_event=stop,deadline_at=spec['deadline_at'],
            model_factory=model_factory,derived_receipt=derived_receipt)
        return 0 if result['status']=='completed' else 130
    finally:ctx.close()


def run(args):
    from retrain.site_binding import load_site, bind_cpu_role
    site=load_site();bind_cpu_role(site,'control')
    import train
    from retrain.execution_policy import resolve_topology, allocate_cpus
    from retrain.config import load_config,job_contract
    from retrain.data import load_source,build_manifest
    from retrain.models import validate_initialization
    from retrain.distributed import layout,RankBatchSampler
    from retrain.distributed_engine import initialize_store
    from retrain.distributed_launcher import supervise_ranks
    from retrain.persistence import RunLock,atomic_json,canonical_digest,sha256_file
    from retrain.fresh import validate_fresh,resolve_steps_per_cycle
    cfg=load_config(args.config,overrides={'resume':args.resume})
    fresh=json.loads(Path(args.fresh_spec).read_text())
    import math,time
    if args.deadline_at is None or not math.isfinite(args.deadline_at) or args.deadline_at<=time.time():
        raise ValueError('fresh runtime requires an unexpired original absolute deadline')
    if cfg['mode']!='train' or len(cfg['jobs'])!=1 or cfg['data']['split']['kind'] not in ('holdout','cv','official_cv','official_merge') or (cfg['data']['split']['kind']=='cv' and not args.manifest) or not cfg['data']['verify_images'] or (cfg['data']['split']['include_valid'] and cfg['data']['split']['kind'] not in ('official_cv','official_merge')):
        raise ValueError('configured fresh entry requires one model and verified holdout/official_cv')
    split=dict(cfg['data']['split']);split.update(cfg['jobs'][0]['split'])
    if (split['kind']=='official_cv') != (cfg['jobs'][0].get('f06') is not None) or (split['include_valid'] and args.manifest):
        raise ValueError('official_cv folds and their registered F06 job must appear together')
    if sha256_file(args.plan)!=fresh['plan_sha256']:raise ValueError('frozen PLAN hash differs')
    if fresh.get('cache_readonly_required') and (os.environ.get('TRAIN_IMAGE_CACHE_READONLY')!='1' or not os.environ.get('TRAIN_IMAGE_CACHE')):raise ValueError('sealed read-only cache required')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device_count=torch.cuda.device_count()
    from retrain.site_binding import load_site, verify_visible_devices, sha256 as site_sha
    site=load_site()
    if device_count!=len(site['gpus']):raise ValueError('visible accelerator count differs from admitted site')
    import ctypes,ctypes.util
    hip=ctypes.CDLL(ctypes.util.find_library('amdhip64') or 'libamdhip64.so')
    hip.hipDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype=ctypes.c_int
    actual=[]
    for ordinal in range(device_count):
        buffer=ctypes.create_string_buffer(64)
        if hip.hipDeviceGetPCIBusId(buffer,64,ordinal):raise ValueError('HIP PCI identity query failed')
        pci=buffer.value.decode().lower();actual.append(('0000:'+pci) if pci.count(':')==1 else pci)
    verify_visible_devices(actual,site)
    job,plan,policy=resolve_topology(cfg,available_accelerators=device_count)
    world=plan['world_size']
    if fresh.get('execution_policy')!=policy:raise ValueError('fresh specification and configured execution policy differ')
    plan['stop_after_updates']=fresh['schedule']['stop_after_updates']
    report=train.environment_report(cfg)
    if report['errors']:raise ValueError('stage environment failed: '+str(report['errors']))
    width,cpus=allocate_cpus(world)
    root=Path(cfg['output_root'])/cfg['run_name']
    os.environ['TRAIN_VERIFICATION_CACHE']=os.environ.get('TRAIN_VERIFICATION_CACHE',str(Path(cfg['output_root'])/'.data-verification-cache'))
    mode='fresh-probe' if fresh['purpose']=='probe' else 'train'
    with RunLock(root.parent/('.'+root.name+'.lock')):
        if root.exists()!=cfg['resume']:raise ValueError('stage needs a new run-name or exact explicit resume')
        if job.get('experiment91') is not None:
            from retrain.experiment91 import accepted_manifest
            manifest=accepted_manifest(args.manifest,args.manifest_sha256)
        elif args.manifest:
            if not args.manifest_sha256 or sha256_file(args.manifest)!=args.manifest_sha256:raise ValueError('accepted immutable manifest bytes differ')
            manifest=json.loads(Path(args.manifest).read_text())
            if manifest.get('audit_indices') or not manifest.get('fit_indices') or not manifest.get('dev_indices'):raise ValueError('complete sealed fit/dev manifest required')
        else:
            source=load_source(cfg['data']['root'],cfg['data']['identity_path'],True,split['include_valid'],cfg['data']['expected_count'])
            split=dict(cfg['data']['split']);split.update(job['split']);manifest=build_manifest(source,split)
        hashes,digest=train.source_identity()
        identity={'format':'glaucoma-retrain-v1','source_digest':digest,'manifest_digest':canonical_digest(manifest),
          'job':job_contract(job),'initialization':validate_initialization(job['initialization']),
          'phase_parent_sha256':None,'distributed':plan,'fresh':fresh}
        if job.get('experiment91') is not None:identity['experiment91']=job['experiment91']
        negatives=sum(not manifest['rows'][i]['label'] for i in manifest['fit_indices'])
        import math
        steps=resolve_steps_per_cycle(job,negatives,fresh_spec=fresh)
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
        for rank in range(world):
            batches=list(RankBatchSampler(manifest['rows'],manifest['fit_indices'],plan['reference_physical_batch'],
                job['effective_batch'],job['positive_per_batch'],job['seed'],0,4,
                rank=rank,world_size=world,per_rank_microbatch=plan.get('explicit_per_rank_microbatch')))
            preview.append({'rank':rank,'sha256':canonical_digest(batches),'batches':batches})
        atomic_json(root/'sampler-preview.json',preview)
        from retrain.n05_rendezvous import new_path,finish
        rendezvous=new_path()
        spec={'job':job,'manifest_path':str(manifest_path),'data_root':cfg['data']['root'],'run_dir':str(folder),
          'identity':identity,'resume':cfg['resume'],'mode':mode,'deadline_at':args.deadline_at,'fresh_spec':fresh,'cpu_width':width,
          'init_method':'file://'+str(rendezvous),'devices':cfg['execution']['devices'],
          'cpu_masks':cpus,'execution_policy':policy,
          'site_binding_sha256':site_sha(os.environ['NCCCU_SITE_BINDING'])}
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
        finally:
            stop.set();guardian.join(timeout=2)
            # supervise_ranks has reaped its own ranks before returning.
            finish(rendezvous,root/'rank-control')
        result.update(mode=mode,purpose=fresh['purpose'],distributed=plan,plan_sha256=fresh['plan_sha256'])
        if result['status']=='completed':
            record=json.loads((folder/'complete.json').read_text())
            if record['status']!='completed' or record['mode']!=mode or any(result['exit_codes'].values()):
                raise ValueError('stage worker completion not proved')
            atomic_json(root/'complete.json',result)
        atomic_json(root/'status.json',result)
        return {'completed':0,'interrupted':130,'failed':1}[result['status']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest');parser.add_argument('--manifest-sha256')
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

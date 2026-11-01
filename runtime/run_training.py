#!/usr/bin/env python3
"""Explicitly bound public-fresh fixed9:1 owner; native process/resource workflow."""
import argparse, contextlib, fcntl, glob, json, os, re, select, signal, socket, stat, subprocess, sys, time, traceback, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL=ROOT/'control';EVIDENCE=ROOT/'evidence'
sys.path.insert(0,str(ROOT/'bundle'))
from retrain.site_binding import load_site, expected_pcis, verify_live, instance_lock, bind_cpu_role
STOP=None

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 import hashlib
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v,*,exclusive=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if exclusive and (p.exists() or p.is_symlink()):raise FileExistsError(str(p))
 tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
 with tmp.open('x') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if exclusive:
  try:os.link(tmp,p)
  finally:tmp.unlink(missing_ok=True)
 else:os.replace(tmp,p)
def ident(pid):
 p=Path('/proc')/str(pid)
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  argv=[x.decode('utf-8','replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  return {'pid':pid,'state':s[0],'pgrp':int(s[2]),'start_ticks':s[19],'argv':argv}
 except (FileNotFoundError,ProcessLookupError):return None
def same(i):
 a=ident(i['pid']);return bool(a and a['state'] not in ('Z','X') and a['start_ticks']==i['start_ticks'] and a['argv']==i['argv'])
def matching():
 out=[]
 work_root=load_site()['work_root']
 for d in glob.glob('/proc/[0-9]*'):
  pid=int(d.rsplit('/',1)[1])
  if pid in (os.getpid(),os.getppid()):continue
  i=ident(pid)
  if not i or i['state'] in ('Z','X'):continue
  a=' '.join(i['argv'])
  if (work_root in a or '/SothisAI/' in a) and any(x in a.lower() for x in ('python','train','evaluate','controller')):
   out.append(i)
 return out
def cards():
 out=[]
 pcis=set(expected_pcis(load_site()))
 for p in Path('/sys/class/drm').glob('card[0-9]*'):
  if not p.name[4:].isdigit():continue
  d=(p/'device').resolve()
  if d.name not in pcis:continue
  out.append({'pci':d.name,'busy_percent':int((d/'gpu_busy_percent').read_text()),'vram_used_bytes':int((d/'mem_info_vram_used').read_text())})
 return sorted(out,key=lambda x:x['pci'])
def own_ranks(rank_control,coordinator=None,owned=None):
 out=[]
 for f in Path(rank_control).glob('rank-*-launch.json'):
  r=read(f);a=ident(r['pid'])
  if a and a['state'] not in ('Z','X') and str(a['start_ticks'])==str(r['start_ticks']) and a['argv']==r['argv']:
   out.append(a)
 if coordinator is not None:
  parent=ident(coordinator['pid'])
  if parent and parent['start_ticks']==coordinator['start_ticks'] and parent['state'] not in ('Z','X'):
   for p in Path('/proc').glob('[0-9]*'):
    try:
     fields=(p/'stat').read_text().rsplit(')',1)[1].split()
     if int(fields[1])!=coordinator['pid']:continue
     child=ident(int(p.name));again=ident(coordinator['pid'])
     if child and child['start_ticks']==fields[19] and again and again['start_ticks']==coordinator['start_ticks']:
      out.append(child)
    except (FileNotFoundError,ProcessLookupError):continue
 if owned is not None:
  for row in out:
   if row['pid'] not in owned:
    try:
     handle=os.pidfd_open(row['pid']);again=ident(row['pid'])
     if again and again['start_ticks']==row['start_ticks'] and again['argv']==row['argv']:
      owned[row['pid']]={'fd':handle,'identity':row,'rank':True}
     else:os.close(handle)
    except ProcessLookupError:continue
  out=[]
  for row in owned.values():
   if row['rank'] and not select.select([row['fd']],[],[],0)[0]:
    current=ident(row['identity']['pid'])
    if current and current['start_ticks']==row['identity']['start_ticks'] and current['state'] not in ('Z','X'):out.append(current)
  owner={'pid':os.getpid(),'start_ticks':ident(os.getpid())['start_ticks']}
  for f in Path(rank_control).glob('rank-*-launch.json'):
   record=read(f);tracked=owned.get(record['pid'])
   if (tracked and str(record['start_ticks'])==tracked['identity']['start_ticks']
       and record.get('supervised_by')!=owner):
    record['supervised_by']=owner;atomic(f,record)
 return out
def run_component(phase,argv,deadline,rank_control,env,*,soft_deadline_at=None,owned=None):
 token=uuid.uuid4().hex;gate=CONTROL/(phase+'-component-gate.json');launch=CONTROL/(phase+'-component-launch.json');terminal=CONTROL/(phase+'-component-terminal.json')
 gate_argv=[sys.executable,'-B','-u',str(ROOT/'component_gate.py'),'--gate',str(gate),'--token',token,'--',*argv]
 log=EVIDENCE/(phase+'-component.log');started=time.time()
 with log.open('x') as out:proc=subprocess.Popen(gate_argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
 cur=ident(proc.pid)
 if not cur or cur['argv']!=gate_argv or cur['pgrp']!=proc.pid:raise RuntimeError('launch identity failed')
 census=phase=='fit-census'
 if census:
  owned={} if owned is None else owned
  owned[proc.pid]={'fd':os.pidfd_open(proc.pid),'identity':cur,'rank':False}
 ranks=lambda:own_ranks(rank_control,cur,owned) if census else own_ranks(rank_control)
 atomic(launch,{'at':started,'phase':phase,'token':token,'gate_identity':cur,'command':argv,'deadline_at':deadline,'rank_control':str(rank_control),'log':str(log)},exclusive=True)
 atomic(gate,{'at':time.time(),'token':token,'pid':proc.pid,'start_ticks':cur['start_ticks']},exclusive=True)
 print(json.dumps({'event':'component_released','phase':phase,'pid':proc.pid,'start_ticks':cur['start_ticks'],'at':time.time()}),flush=True)
 reason=None;signals=[];term=None;last_monitor=0;cpu_known={}
 soft=soft_deadline_at if census else deadline-150
 while (proc.poll() is None or (census and ranks())) and time.time()<deadline:
  now=time.time()
  if census:ranks()
  if now-last_monitor>=15:
   stage='runtime_safety_snapshot'
   try:
    import run_round as guard
    resources=guard.runtime_safety_snapshot()
    stage='budget'
    resources.update(guard.budget(deadline))
    stage='audit_cpu_tree'
    from retrain.execution_policy import audit_cpu_tree
    resources['cpu_tree']=audit_cpu_tree(os.getpid(),cpu_known,load_site())
    stage='live_resources_receipt'
    atomic(CONTROL/'live-resources.json',resources)
    if phase=='train':
     stage='maybe_release'
     guard.maybe_release(ROOT,atomic,sha)
   except Exception as exc:
    reason=reason or 'resource_guard:%s:%r:%s'%(stage,exc,traceback.format_exc(limit=4).replace('\n',' | '))
   last_monitor=now
  if STOP and reason is None:reason='signal_'+STOP
  if census and proc.poll() is not None and ranks() and reason is None:reason='coordinator_exited_with_live_ranks'
  if reason is None and now>=soft:reason='soft_deadline'
  if reason and term is None:
   a=ident(proc.pid)
   if census and a and a['start_ticks']==cur['start_ticks'] and a['state'] not in ('Z','X'):
    signal.pidfd_send_signal(owned[proc.pid]['fd'],signal.SIGSTOP)
    until=min(time.time()+.5,deadline-1)
    while time.time()<until:
     stopped=ident(proc.pid)
     if not stopped or stopped['state'] in ('T','t','Z','X'):break
     time.sleep(.01)
    ranks()  # Snapshot direct children while their confirmed parent cannot spawn.
    for item in owned.values():
     if item['rank']:
      try:signal.pidfd_send_signal(item['fd'],signal.SIGTERM)
      except ProcessLookupError:pass
    try:signal.pidfd_send_signal(owned[proc.pid]['fd'],signal.SIGKILL)
    except ProcessLookupError:pass
   if a and a['start_ticks']==cur['start_ticks'] and (not census or (a['pgrp']==cur['pid'] and a['argv'] in (cur['argv'],argv))):
    try:os.killpg(a['pgrp'],signal.SIGTERM);signals.append({'pid':a['pid'],'sig':'TERM'})
    except ProcessLookupError:pass
   term=now
   if census:
    for r in ranks():
     if r['pgrp']==r['pid'] and same(r):
      try:os.killpg(r['pgrp'],signal.SIGTERM);signals.append({'pid':r['pid'],'sig':'TERM'})
      except ProcessLookupError:pass
  if term and now>=min(term+30,deadline-1 if census else deadline):
   for r in ranks():
    if census and (r['pgrp']!=r['pid'] or not same(r)):continue
    try:os.killpg(r['pgrp'],signal.SIGKILL);signals.append({'pid':r['pid'],'sig':'KILL'})
    except ProcessLookupError:pass
   a=ident(proc.pid)
   if a and a['start_ticks']==cur['start_ticks'] and (not census or (a['pgrp']==cur['pid'] and a['argv'] in (cur['argv'],argv))):
    try:os.killpg(a['pgrp'],signal.SIGKILL);signals.append({'pid':a['pid'],'sig':'KILL'})
    except ProcessLookupError:pass
  left=deadline-time.time()
  if left<=0:break
  time.sleep(min(0.5,left))
 code=proc.poll()
 if code is None:
  for r in own_ranks(rank_control):
   if census and (r['pgrp']!=r['pid'] or not same(r)):continue
   try:os.killpg(r['pgrp'],signal.SIGKILL)
   except ProcessLookupError:pass
  a=ident(proc.pid)
  if a and a['start_ticks']==cur['start_ticks'] and (not census or (a['pgrp']==cur['pid'] and a['argv'] in (cur['argv'],argv))):
   try:os.killpg(a['pgrp'],signal.SIGKILL)
   except ProcessLookupError:pass
  code=proc.poll();reason=reason or 'hard_deadline'
 if census:
  for item in owned.values():
   try:signal.pidfd_send_signal(item['fd'],signal.SIGKILL)
   except ProcessLookupError:pass
  while ranks() and time.time()<deadline-.1:time.sleep(.02)
 remaining=ranks();end=time.time()
 status='completed' if code==0 and reason is None and not remaining else ('resource_unknown' if remaining or code is None else 'timeout')
 result={'phase':phase,'status':status,'start_at':started,'end_at':end,'component_wall_seconds':end-started,'exit_code':code,'reason':reason,'signals':signals,'live_owned_ranks':remaining,'deadline_at':deadline,'hard_deadline_met':end<=deadline,'launch_sha256':sha(launch),'log_sha256':sha(log)}
 atomic(terminal,result,exclusive=True)
 if status!='completed':raise RuntimeError('component not completed: %r'%{k:result[k] for k in ('status','exit_code','reason')})
 return result

def run_fit_census(args):
 """Reuse the parent lock, resource checks and component supervision for one census."""
 global STOP
 from launch_once import load_fit_context
 # Count import/admission time from this owner's birth, rather than resetting at forward.
 birth=int((Path('/proc/self/stat').read_text().rsplit(')',1)[1].split())[19])/os.sysconf('SC_CLK_TCK')
 started=time.time()-(time.clock_gettime(time.CLOCK_BOOTTIME)-birth)
 soft=started+1500;hard=started+1800
 rank_control=EVIDENCE/'FIT-CENSUS-AGGREGATE.ranks/processes';fd=None;acquired=False
 owned={};published=False
 terminal={'mode':'fit-census','status':'unknown','metrics':None,'completed_count':None,
           'owner':{key:val for key,val in ident(os.getpid()).items() if key!='argv'},
           'started_at':started,'soft_deadline_at':soft,'hard_deadline_at':hard}
 def hard_stop(*_):
  # No file reads, logging, Python exception/finally or stale numeric PID signals here.
  for item in tuple(owned.values()):
   try:signal.pidfd_send_signal(item['fd'],signal.SIGKILL)
   except ProcessLookupError:pass
  pending=[item['fd'] for item in owned.values()]
  while pending and time.time()<hard:
   ready=select.select(pending,[],[],min(.02,max(0,hard-time.time())))[0]
   pending=[p for p in pending if p not in ready]
  os._exit(0 if published and not pending else 124)
 previous=signal.signal(signal.SIGALRM,hard_stop)
 signal.setitimer(signal.ITIMER_REAL,max(.001,hard-time.time()-2))
 atomic(EVIDENCE/'FIT-CENSUS-RESULT.json',terminal,exclusive=True)
 try:
  value=load_fit_context(args.fit_context,args.fit_context_sha256)
  if args.token!=value['token']:raise ValueError('diagnostic token differs')
  site=load_site();bind_cpu_role(site,'control')
  grant=read(site['cpu_execution']['allocation_receipt']['path'])
  if grant['valid_until']<hard:raise ValueError('new diagnostic grant does not cover the hard deadline')
  admission=read(CONTROL/'pod-admission.json')
  verify_live(site,pod_uid=admission['pod_uid'],ready=admission['ready'],observed_at=admission['observed_at'])
  lock=instance_lock(site);fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);state=os.fstat(fd)
  if (not stat.S_ISREG(state.st_mode) or [state.st_dev,state.st_ino]!=value['instance_lock_identity']):
   raise ValueError('global instance lock inode differs')
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
  import run_round as guard
  active=[]
  for process in matching():
   try:
    env=dict(x.split(b'=',1) for x in (Path('/proc')/str(process['pid'])/'environ').read_bytes().split(b'\0') if b'=' in x)
    if all(env.get(k)==b'' for k in (b'CUDA_VISIBLE_DEVICES',b'HIP_VISIBLE_DEVICES',b'ROCR_VISIBLE_DEVICES')) and os.getpriority(os.PRIO_PROCESS,process['pid'])>=19:continue
   except FileNotFoundError:continue
   active.append(process)
  if active or guard.device_users_before_start():raise RuntimeError('diagnostic devices/processes occupied')
  snapshot=cards()
  if len(snapshot)!=len(site['gpus']) or any(x['busy_percent'] or x['vram_used_bytes']>64*1024**2 for x in snapshot):raise RuntimeError('diagnostic cards not idle')
  guard.budget(hard,minimum_free=20*1024**3);guard.runtime_safety_snapshot()
  atomic(CONTROL/'owner.json',{'owner':terminal['owner'],'mode':'fit-census'},exclusive=True)
  atomic(CONTROL/'controller-start.json',{'owner':terminal['owner'],'mode':'fit-census'},exclusive=True)
  atomic(CONTROL/'diagnostic-clock.json',{'started_at':started,'soft_deadline_at':soft,'hard_deadline_at':hard},exclusive=True)
  env=os.environ.copy();env.update(TRAIN_IMAGE_CACHE=value['cache'],TRAIN_IMAGE_CACHE_READONLY='1',PYTHONDONTWRITEBYTECODE='1')
  output=EVIDENCE/'FIT-CENSUS-AGGREGATE.json'
  if STOP or time.time()>=soft:raise TimeoutError('diagnostic stopped before dispatch')
  command=[sys.executable,'-B','-u',str(ROOT/'bundle/parallel_evaluate.py'),'--mode','fit-census',
           '--fit-context',str(args.fit_context),'--fit-context-sha256',args.fit_context_sha256,
           '--output',str(output),'--deadline-at',repr(soft)]
  terminal['component']=run_component('fit-census',command,hard,rank_control,env,soft_deadline_at=soft,owned=owned)
  report=read(output)
  if report['status']!='awaiting_parent_safety':raise ValueError('census aggregate incomplete')
  terminal.update(completed_count=report['metrics']['count'],selected=report['selected'],candidate_metrics=report['metrics'])
 except BaseException as exc:
  terminal['error']=type(exc).__name__+': '+str(exc)
 finally:
  # Parent retains its flock over every cleanup path, including coordinator failure.
  coordinator=None
  launch=CONTROL/'fit-census-component-launch.json'
  if launch.exists():
   released=read(launch);expected=released['gate_identity'];current=ident(expected['pid'])
   if current and current['start_ticks']==expected['start_ticks'] and current['pgrp']==expected['pid']:
    if current['state'] not in ('Z','X') and current['argv'] in (expected['argv'],released['command']):
     try:
      signal.pidfd_send_signal(owned[current['pid']]['fd'],signal.SIGSTOP)
      own_ranks(rank_control,expected,owned)
      signal.pidfd_send_signal(owned[current['pid']]['fd'],signal.SIGKILL)
     except ProcessLookupError:pass
    coordinator=expected
  for sig in (signal.SIGTERM,signal.SIGKILL):
   own_ranks(rank_control,owned=owned)
   for item in owned.values():
    try:signal.pidfd_send_signal(item['fd'],sig)
    except ProcessLookupError:pass
   end=min(time.time()+1,hard-.1)
   while own_ranks(rank_control) and time.time()<end:time.sleep(.02)
  try:
   if not acquired:raise RuntimeError('global lock was not acquired')
   users=guard.device_users_before_start();snapshot=cards();remaining=own_ranks(rank_control,owned=owned)
   size=sum(p.stat().st_size for root in (EVIDENCE,CONTROL) for p in root.rglob('*') if p.is_file())
   current=ident(coordinator['pid']) if coordinator else None
   coordinator_live=bool(current and current['start_ticks']==coordinator['start_ticks'] and current['state'] not in ('Z','X'))
   registered=len(list(rank_control.glob('rank-*-launch.json')))==len(site['gpus'])
   safe=registered and not users and not remaining and not coordinator_live and len(snapshot)==len(site['gpus']) and not any(x['busy_percent'] for x in snapshot)
   terminal['actualsafe']={'safe':safe,'live_owned_ranks':remaining,'device_users':users,'scan_denied':0,'cards':snapshot,
                          'coordinator_live':coordinator_live,'parent_lock_held':True,'lock_identity':[state.st_dev,state.st_ino],'output_bytes':size}
   if safe and 'candidate_metrics' in terminal and time.time()<hard and size<=16*1024**2 and not STOP:
    metrics=terminal['candidate_metrics'];terminal.update(status='completed',metrics=metrics,
      descriptive_fit_minus_dev=metrics['auroc']-terminal['selected']['dev_auroc'])
  except BaseException as exc:terminal['safety_error']=repr(exc)
  terminal.pop('candidate_metrics',None);terminal['ended_at']=time.time()
  aggregate=EVIDENCE/'FIT-CENSUS-AGGREGATE.json'
  if aggregate.exists() and terminal['status']!='completed':atomic(aggregate,{'status':'unknown','metrics':None})
  encoded=json.dumps(terminal)
  if len(encoded.encode())>64*1024:terminal={'mode':'fit-census','status':'unknown','metrics':None,'error':'aggregate_limit'}
  atomic(EVIDENCE/'FIT-CENSUS-RESULT.json',terminal)
  published=terminal['status']=='completed'
  if acquired:fcntl.flock(fd,fcntl.LOCK_UN)
  if fd is not None:os.close(fd)
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
  for item in owned.values():os.close(item['fd'])
 return 0 if terminal['status']=='completed' else 2

def main():
 global STOP
 p=argparse.ArgumentParser();p.add_argument('--token',required=True)
 p.add_argument('--mode',choices=('train','fit-census'),default='train')
 p.add_argument('--fit-context',type=Path);p.add_argument('--fit-context-sha256');args=p.parse_args()
 def on_signal(sig,_):
  global STOP;STOP=signal.Signals(sig).name
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,on_signal)
 if args.mode=='fit-census':return run_fit_census(args)
 clock=read(CONTROL/'round-clock.json');cfg=read(ROOT/'TRAIN_CONFIG.json')
 site=load_site();world=len(site['gpus']);bind_cpu_role(site,'control')
 if site.get('scope')!='formal_exp_v3':raise ValueError('training owner requires a formal deployment')
 if args.token!=clock['token']:raise ValueError('round token differs')
 admission=read(CONTROL/'pod-admission.json')
 verify_live(site,pod_uid=admission.get('pod_uid'),ready=admission.get('ready'),observed_at=admission.get('observed_at'))
 if clock['round_id']!=site['round_id'] or cfg['execution']['devices']!=['cuda:'+str(rank) for rank in range(world)]:raise ValueError('round or configured world differs from site')
 cpu=read(EVIDENCE/'MODEL_CPU_ADMISSION.json')
 if cpu.get('status')!='completed' or cpu['formal_train_config_sha256']!=sha(ROOT/'TRAIN_CONFIG.json') or cpu['gpu_initialized'] or not cpu.get('checks') or any(x.get('passed') is not True for x in cpu['checks']):raise ValueError('CPU preparation differs')
 import run_round as guard
 guard.verify_bound()
 fresh=read(CONTROL/'train-fresh-spec.json')
 if fresh['ema']!=read(EVIDENCE/'FRESH_EMA_CONTRACT.json')['rule'] or fresh['schedule']!=read(ROOT/'ASSETS.json')['schedule']:raise ValueError('fresh EMA or schedule differs')
 if time.time()>=clock['implementation_deadline_at']:raise TimeoutError('implementation window elapsed')
 assets=read(ROOT/'ASSETS.json');manifest=assets['manifests']['fit_dev']
 if sha(manifest['remote_path'])!=manifest['sha256']:raise ValueError('manifest SHA')
 lock=instance_lock(site)
 fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW)
 rank_control=ROOT/'outputs'/cfg['run_name']/'rank-control'
 terminal={'status':'failed','at':time.time(),'token':args.token}
 acquired=False
 try:
  handoff=read(CONTROL/'predecessor-handoff.json')
  if (os.fstat(fd).st_ino!=handoff['lock_inode'] or os.fstat(fd).st_dev!=handoff['lock_device']
      or not stat.S_ISREG(os.fstat(fd).st_mode) or str(lock)!=handoff['lock_path']):raise RuntimeError('admitted instance lock identity changed')
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
  active=[];cpu_only=[]
  for process in matching():
   try:
    environment=dict(x.split(b'=',1) for x in (Path('/proc')/str(process['pid'])/'environ').read_bytes().split(b'\0') if b'=' in x)
    if all(environment.get(k)==b'' for k in (b'CUDA_VISIBLE_DEVICES',b'HIP_VISIBLE_DEVICES',b'ROCR_VISIBLE_DEVICES')) and os.getpriority(os.PRIO_PROCESS,process['pid'])>=19:
     cpu_only.append(process);continue
   except FileNotFoundError:continue
   active.append(process)
  if active:raise RuntimeError('another project GPU-capable process is active: '+repr(active))
  device_users=guard.device_users_before_start()
  if device_users:raise RuntimeError('GPU device handles occupied: '+repr(device_users))
  for item in read(CONTROL/'predecessor-handoff.json')['exact_processes']:
   if same(item['expected']):raise RuntimeError('previous exact owner/component/rank identity still live')
  atomic(CONTROL/'resource-admission.json',{'at':time.time(),'cpu_only_processes':cpu_only,'device_users':device_users,'old_exact_identities_inactive':True},exclusive=True)
  guard.budget(clock['deadline_at'],minimum_free=20*1024**3)
  guard.runtime_safety_snapshot()
  snapshot=cards()
  if len(snapshot)!=world or {c['pci'] for c in snapshot}!=set(expected_pcis(site)):raise RuntimeError('all admitted devices must be visible')
  if any(c['busy_percent'] or c['vram_used_bytes']>64*1024*1024 for c in snapshot):raise RuntimeError('devices occupied')
  import shutil
  if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('insufficient free storage')
  owner=ident(os.getpid())
  atomic(CONTROL/'owner.json',{'at':time.time(),'owner':owner,'cards':snapshot,'token':args.token},exclusive=True)
  atomic(CONTROL/'controller-start.json',{'at':time.time(),'owner':owner,'argv':sys.argv,'token':args.token},exclusive=True)
  env=os.environ.copy()
  for key in ('MODE','RUN_NAME','OUTPUT_ROOT','DATA_ROOT','RESUME','DEVICE_IDS'):env.pop(key,None)
  env.update(PYTHONDONTWRITEBYTECODE='1',XFORMERS_DISABLED='1',TRAIN_IMAGE_CACHE_READONLY='1',
             OUTPUT_ROOT=str(ROOT/'outputs'),TRAIN_IMAGE_CACHE=str(Path(assets['prepared512_cache']['remote_root'])/'train/sealed'),
             TRAIN_VERIFICATION_CACHE=str(ROOT/'verification-unused-readonly'))
  if assets.get('global_prepared512_cache'):
   env['TRAIN_GLOBAL_IMAGE_CACHE']=str(Path(assets['global_prepared512_cache']['remote_root'])/'train/sealed')
  train_started=time.time()
  policy=read(ROOT/'CLOCK_POLICY.json')
  deadline=min(train_started+policy['train_hard_seconds'],clock['deadline_at']-policy['freeze_hard_seconds']-policy['official_hard_seconds']-policy['cleanup_seconds'])
  atomic(CONTROL/'train-phase-start.json',{'at':train_started,'deadline_at':deadline,'soft_deadline_at':deadline-150,'token':args.token},exclusive=True)
  argv=[sys.executable,'-B','-u',str(ROOT/'bundle/parallel_fresh_train.py'),
        '--config',str(ROOT/'TRAIN_CONFIG.json'),'--fresh-spec',str(CONTROL/'train-fresh-spec.json'),
        '--plan',str(ROOT/'PLAN.md'),'--manifest',manifest['remote_path'],
        '--manifest-sha256',manifest['sha256'],'--deadline-at',repr(deadline-150)]
  result=run_component('train',argv,deadline,rank_control,env)
  out=ROOT/'outputs'/cfg['run_name'];model_dir=out/cfg['jobs'][0]['name']
  complete=read(out/'complete.json');record=read(model_dir/'complete.json')
  if complete['status']!='completed' or set(complete['exit_codes'])!={'rank-'+str(rank) for rank in range(world)} or any(complete['exit_codes'].values()):raise ValueError('rank completion')
  if record.get('status')!='completed' or record.get('skipped_updates')!=0 or record['attempted_updates']!=record['successful_updates']:raise ValueError('successful update completion')
  from retrain.experiment91 import validate_training_completion
  cycles=validate_training_completion(record)
  if record['ema']['rule']!=fresh['ema'] or record['ema']['copy_count']!=min(record['successful_updates'],fresh['ema']['warmcopy_updates']) or record['ema']['blend_count']!=max(0,record['successful_updates']-fresh['ema']['warmcopy_updates']):raise ValueError('runtime EMA counters differ')
  if len(list((model_dir/'predictions').glob('*/dev-cycle-*.csv')))!=2*cycles:raise ValueError('paired dev CSVs required through completed training')
  actual_initial=read(EVIDENCE/'ACTUAL_FRESH_INITIAL.json')
  expected_initial=dict(cpu['initial_state_control'])
  if fresh.get('ssl_recipe') is not None:
   derived=read(model_dir/'ssl-stage/SSL-RESULT.json')
   if (actual_initial.get('ssl_derived')!=derived or record.get('ssl_derived')!=derived
       or actual_initial.get('ssl_receipt_sha256')!=sha(model_dir/'ssl-stage/SSL-RESULT.json')):
    raise ValueError('R115 joint SSL result/derived supervised initial differs')
   expected_initial['model_tensor_sha256']=actual_initial['model_tensor_sha256']
  guard.verify_fresh_initial_identity(read(model_dir/'initial-ranks.json'),read(model_dir/'parameter-groups.json'),read(EVIDENCE/'CPU_PARAMETER_GROUPS.json'),expected_initial,expected_world=world)
  initials=read(model_dir/'initial-ranks.json')
  if actual_initial['status']!='completed' or actual_initial['rank_count']!=world or len(initials)!=world or any(x['model_sha256']!=actual_initial['model_tensor_sha256'] or x['optimizer_state_entries']!=0 for x in initials):raise ValueError('actual rank fresh initialization')
  terminal.update(status='completed',successful_updates=record['successful_updates'],cycles=record['cycle'],early_stop=record['early_stop'],
                  component=result,complete_sha256=sha(out/'complete.json'),research_only=False)
  atomic(EVIDENCE/'TRAIN_RESULT.json',terminal,exclusive=True)
  terminal['phase']='freeze'
  freeze_path=ROOT/'SINGLE-MODEL-FREEZE.json'
  freeze_env=env.copy()
  freeze_env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  freeze_deadline=min(time.time()+policy['freeze_hard_seconds'],clock['deadline_at']-policy['official_hard_seconds']-policy['cleanup_seconds'])
  if freeze_deadline-time.time()<=150:raise TimeoutError('freeze plus cleanup window unavailable')
  freeze_command=['nice','-n','19',sys.executable,'-B','-u',str(ROOT/'bundle/single_model91.py'),'freeze','--training-folder',str(model_dir),'--output',str(freeze_path)]
  freeze_result=run_component('freeze',freeze_command,freeze_deadline,CONTROL/'freeze-no-ranks',freeze_env)
  frozen=read(freeze_path)
  if frozen.get('cpu_fixture_only') or frozen['source_digest']!=fresh['source_digest']:raise ValueError('production single-model freeze differs')
  from retrain.experiment91 import official_allowed_for_selected
  evaluation=read(ROOT/'SCIENCE.json').get('official_evaluation')
  official_allowed=official_allowed_for_selected(evaluation,frozen['selected'])
  if frozen.get('official_evaluations_allowed')!=int(official_allowed):raise ValueError('official gate/freeze differs')
  if not official_allowed:
   terminal.update(status='completed',phase='complete',freeze_component=freeze_result,
     single_model_freeze_sha256=sha(freeze_path),official_status='not_run_by_policy',
     official_reason='threshold_not_met',official_evaluated=False,
     selected_internal_dev_auroc=frozen['selected']['auroc'],official_result_sha256=None,
     official_metrics=None,target_auroc_at_least_0_999=None)
   atomic(EVIDENCE/'ROUND_RESULT.json',terminal,exclusive=True)
   return
  terminal['phase']='official'
  official_env=env.copy()
  official_env['TRAIN_IMAGE_CACHE']=str(Path(assets['prepared512_cache']['remote_root'])/'valid/sealed')
  official_root=ROOT/'official'
  official_started=time.time()
  official_seconds=read(ROOT/'CLOCK_POLICY.json')['official_hard_seconds']
  if clock['deadline_at']-policy['cleanup_seconds']-official_started<official_seconds:raise TimeoutError('full official single-device budget unavailable')
  official_deadline=official_started+official_seconds
  official=assets['manifests']['official_valid']
  official_command=[sys.executable,'-B','-u',str(ROOT/'bundle/single_model91.py'),'official-once','--freeze',str(freeze_path),'--output-root',str(official_root),'--data-root',assets['data_root'],'--eval-manifest',official['remote_path'],'--eval-manifest-sha256',official['sha256'],'--device','cuda:0']
  official_component=run_component('official',official_command,official_deadline,CONTROL/'official-no-ranks',official_env)
  official_result=read(official_root/'OFFICIAL-RESULT.json')
  if official_result['status']!='completed' or (official_result['rows'],official_result['positive'],official_result['negative'])!=(5849,191,5658):raise ValueError('official full coverage not complete')
  terminal.update(status='completed',phase='complete',freeze_component=freeze_result,official_component=official_component,
    single_model_freeze_sha256=sha(freeze_path),official_status='completed',official_evaluated=True,
    selected_internal_dev_auroc=frozen['selected']['auroc'],official_result_sha256=sha(official_root/'OFFICIAL-RESULT.json'),
    official_metrics=official_result['metrics'],target_auroc_at_least_0_999=official_result['target_auroc_at_least_0_999'])
  atomic(EVIDENCE/'ROUND_RESULT.json',terminal,exclusive=True)
 except BaseException as exc:
  terminal['status']='official_unknown' if terminal.get('phase')=='official' else 'failed'
  terminal['error']=repr(exc)
  raise
 finally:
  if acquired:
   # The native launcher supervises its ranks. Retain the instance lock while
   # cleaning any surviving exact rank identities on an abnormal return.
   survivors=own_ranks(rank_control)
   for live in survivors:
    current=ident(live['pid'])
    if current and current['start_ticks']==live['start_ticks'] and current['argv']==live['argv']:
     try:os.killpg(current['pgrp'],signal.SIGTERM)
     except ProcessLookupError:pass
   if survivors:
    end=min(time.time()+30,read(CONTROL/'train-phase-start.json')['deadline_at']-5)
    while own_ranks(rank_control) and time.time()<end:time.sleep(.2)
    for live in own_ranks(rank_control):
     try:os.killpg(live['pgrp'],signal.SIGKILL)
     except ProcessLookupError:pass
   terminal.update(ended_at=time.time(),live_owned_ranks=own_ranks(rank_control),cards=cards())
   if terminal['live_owned_ranks']:terminal['status']='resource_unknown'
   atomic(CONTROL/'done.json',terminal,exclusive=True)
   fcntl.flock(fd,fcntl.LOCK_UN)
  os.close(fd)

if __name__=='__main__':sys.exit(main())

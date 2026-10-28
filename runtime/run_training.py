#!/usr/bin/env python3
"""R91 publicfresh ordinary R50 fixed9010 owner; native process/resource workflow."""
import argparse, contextlib, fcntl, glob, json, os, re, signal, socket, stat, subprocess, sys, time, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL=ROOT/'control';EVIDENCE=ROOT/'evidence'
PCIS={'0000:09:00.0','0000:77:00.0','0000:85:00.0','0000:f5:00.0'}
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
 except (OSError,ValueError,IndexError):return None
def same(i):
 a=ident(i['pid']);return bool(a and a['state'] not in ('Z','X') and a['start_ticks']==i['start_ticks'] and a['argv']==i['argv'])
def matching():
 out=[]
 for d in glob.glob('/proc/[0-9]*'):
  pid=int(d.rsplit('/',1)[1])
  if pid in (os.getpid(),os.getppid()):continue
  i=ident(pid)
  if not i or i['state'] in ('Z','X'):continue
  a=' '.join(i['argv'])
  if ('auroc999-20261027' in a or '/SothisAI/' in a) and any(x in a.lower() for x in ('python','train','evaluate','controller')):
   out.append(i)
 return out
def cards():
 out=[]
 for p in Path('/sys/class/drm').glob('card[0-9]*'):
  if not p.name[4:].isdigit():continue
  d=(p/'device').resolve()
  if d.name not in PCIS:continue
  out.append({'pci':d.name,'busy_percent':int((d/'gpu_busy_percent').read_text()),'vram_used_bytes':int((d/'mem_info_vram_used').read_text())})
 return sorted(out,key=lambda x:x['pci'])
def own_ranks(rank_control):
 out=[]
 for f in Path(rank_control).glob('rank-*-launch.json'):
  r=read(f);a=ident(r['pid'])
  if a and a['state'] not in ('Z','X') and str(a['start_ticks'])==str(r['start_ticks']) and a['argv']==r['argv']:
   out.append(a)
 return out
def run_component(phase,argv,deadline,rank_control,env):
 token=uuid.uuid4().hex;gate=CONTROL/(phase+'-component-gate.json');launch=CONTROL/(phase+'-component-launch.json');terminal=CONTROL/(phase+'-component-terminal.json')
 gate_argv=[sys.executable,'-B','-u',str(ROOT/'component_gate.py'),'--gate',str(gate),'--token',token,'--',*argv]
 log=EVIDENCE/(phase+'-component.log');started=time.time()
 with log.open('x') as out:proc=subprocess.Popen(gate_argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
 cur=ident(proc.pid)
 if not cur or cur['argv']!=gate_argv or cur['pgrp']!=proc.pid:raise RuntimeError('launch identity failed')
 atomic(launch,{'at':started,'phase':phase,'token':token,'gate_identity':cur,'command':argv,'deadline_at':deadline,'rank_control':str(rank_control),'log':str(log)},exclusive=True)
 atomic(gate,{'at':time.time(),'token':token,'pid':proc.pid,'start_ticks':cur['start_ticks']},exclusive=True)
 print(json.dumps({'event':'component_released','phase':phase,'pid':proc.pid,'start_ticks':cur['start_ticks'],'at':time.time()}),flush=True)
 reason=None;signals=[];term=None;last_monitor=0
 soft=deadline-150
 while proc.poll() is None and time.time()<deadline:
  now=time.time()
  if now-last_monitor>=15:
   try:
    import run_round as guard
    resources=guard.runtime_safety_snapshot();resources.update(guard.budget(deadline))
    atomic(CONTROL/'live-resources.json',resources)
    if phase=='train':guard.maybe_release(ROOT,atomic,sha)
   except Exception as exc:reason=reason or 'resource_guard:'+repr(exc)
   last_monitor=now
  if STOP and reason is None:reason='signal_'+STOP
  if reason is None and now>=soft:reason='soft_deadline'
  if reason and term is None:
   a=ident(proc.pid)
   if a and a['start_ticks']==cur['start_ticks']:
    try:os.killpg(a['pgrp'],signal.SIGTERM);signals.append({'pid':a['pid'],'sig':'TERM'})
    except ProcessLookupError:pass
   term=now
  if term and now>=min(term+30,deadline):
   for r in own_ranks(rank_control):
    try:os.killpg(r['pgrp'],signal.SIGKILL);signals.append({'pid':r['pid'],'sig':'KILL'})
    except ProcessLookupError:pass
   a=ident(proc.pid)
   if a and a['start_ticks']==cur['start_ticks']:
    try:os.killpg(a['pgrp'],signal.SIGKILL);signals.append({'pid':a['pid'],'sig':'KILL'})
    except ProcessLookupError:pass
  left=deadline-time.time()
  if left<=0:break
  time.sleep(min(0.5,left))
 code=proc.poll()
 if code is None:
  for r in own_ranks(rank_control):
   try:os.killpg(r['pgrp'],signal.SIGKILL)
   except ProcessLookupError:pass
  a=ident(proc.pid)
  if a and a['start_ticks']==cur['start_ticks']:
   try:os.killpg(a['pgrp'],signal.SIGKILL)
   except ProcessLookupError:pass
  code=proc.poll();reason=reason or 'hard_deadline'
 remaining=own_ranks(rank_control);end=time.time()
 status='completed' if code==0 and reason is None and not remaining else ('resource_unknown' if remaining or code is None else 'timeout')
 result={'phase':phase,'status':status,'start_at':started,'end_at':end,'component_wall_seconds':end-started,'exit_code':code,'reason':reason,'signals':signals,'live_owned_ranks':remaining,'deadline_at':deadline,'hard_deadline_met':end<=deadline,'launch_sha256':sha(launch),'log_sha256':sha(log)}
 atomic(terminal,result,exclusive=True)
 if status!='completed':raise RuntimeError('component not completed: %r'%{k:result[k] for k in ('status','exit_code','reason')})
 return result

def main():
 global STOP
 p=argparse.ArgumentParser();p.add_argument('--token',required=True);args=p.parse_args()
 def on_signal(sig,_):
  global STOP;STOP=signal.Signals(sig).name
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,on_signal)
 clock=read(CONTROL/'round-clock.json');cfg=read(ROOT/'TRAIN_CONFIG.json')
 if args.token!=clock['token']:raise ValueError('round token differs')
 admission=read(CONTROL/'pod-admission.json')
 if admission.get('pod_uid')!='eec33bd6-b567-40ef-b866-633e272ea019' or not admission.get('ready') or not 0<=time.time()-admission['observed_at']<=120:raise ValueError('Pod identity or freshness')
 if socket.gethostname()!='vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0':raise ValueError('Pod hostname')
 cpu=read(EVIDENCE/'MODEL_CPU_ADMISSION.json')
 if cpu.get('status')!='completed' or cpu['formal_train_config_sha256']!=sha(ROOT/'TRAIN_CONFIG.json') or cpu['gpu_initialized'] or not cpu.get('checks') or any(x.get('passed') is not True for x in cpu['checks']):raise ValueError('CPU preparation differs')
 import run_round as guard
 guard.verify_bound()
 fresh=read(CONTROL/'train-fresh-spec.json')
 if fresh['ema']!=read(EVIDENCE/'FRESH_EMA_CONTRACT.json')['rule'] or fresh['schedule']!=read(ROOT/'ASSETS.json')['schedule']:raise ValueError('fresh EMA or schedule differs')
 if time.time()>=clock['implementation_deadline_at']:raise TimeoutError('implementation window elapsed')
 assets=read(ROOT/'ASSETS.json');manifest=assets['manifests']['fit_dev']
 if sha(manifest['remote_path'])!=manifest['sha256']:raise ValueError('manifest SHA')
 lock='/public/home/acfwvv9ejn/SothisAI/exp-v2/.instance-owner-eec33bd6-b567-40ef-b866-633e272ea019.lock'
 fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW)
 rank_control=ROOT/'outputs'/cfg['run_name']/'rank-control'
 terminal={'status':'failed','at':time.time(),'token':args.token}
 acquired=False
 try:
  if os.fstat(fd).st_ino!=read(CONTROL/'predecessor-handoff.json')['lock_inode']:raise RuntimeError('original instance lock inode changed')
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
  if len(snapshot)!=4 or {c['pci'] for c in snapshot}!=PCIS:raise RuntimeError('four expected devices not visible')
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
  train_started=time.time()
  deadline=min(train_started+read(ROOT/'CLOCK_POLICY.json')['train_hard_seconds'],clock['deadline_at']-1200)
  atomic(CONTROL/'train-phase-start.json',{'at':train_started,'deadline_at':deadline,'soft_deadline_at':deadline-150,'token':args.token},exclusive=True)
  argv=[sys.executable,'-B','-u',str(ROOT/'bundle/parallel_fresh_train.py'),
        '--config',str(ROOT/'TRAIN_CONFIG.json'),'--fresh-spec',str(CONTROL/'train-fresh-spec.json'),
        '--plan',str(ROOT/'PLAN.md'),'--manifest',manifest['remote_path'],
        '--manifest-sha256',manifest['sha256'],'--deadline-at',repr(deadline-150)]
  result=run_component('train',argv,deadline,rank_control,env)
  out=ROOT/'outputs'/cfg['run_name'];model_dir=out/cfg['jobs'][0]['name']
  complete=read(out/'complete.json');record=read(model_dir/'complete.json')
  if complete['status']!='completed' or len(complete['exit_codes'])!=4 or any(complete['exit_codes'].values()):raise ValueError('rank completion')
  if record.get('status')!='completed' or record.get('skipped_updates')!=0 or record['attempted_updates']!=record['successful_updates']:raise ValueError('successful update completion')
  if record.get('evaluated_cycle')!=8 or record['cycle']!=8 or record['successful_updates']!=fresh['schedule']['total_updates']:raise ValueError('eight full cycles and complete new8S required')
  if record['ema']['rule']!=fresh['ema'] or record['ema']['copy_count']!=fresh['ema']['warmcopy_updates'] or record['ema']['blend_count']!=record['successful_updates']-fresh['ema']['warmcopy_updates']:raise ValueError('runtime EMA counters differ')
  if len(list((model_dir/'predictions').glob('*/dev-cycle-*.csv')))!=16:raise ValueError('16 complete dev CSVs required')
  guard.verify_fresh_initial_identity(read(model_dir/'initial-ranks.json'),read(model_dir/'bn-initial-ranks.json'),read(model_dir/'parameter-groups.json'),read(EVIDENCE/'CPU_PARAMETER_GROUPS.json'),cpu['initial_state_control'])
  initials=read(model_dir/'initial-ranks.json')
  actual_initial=read(EVIDENCE/'ACTUAL_FRESH_INITIAL.json')
  if actual_initial['status']!='completed' or len(initials)!=4 or any(x['model_sha256']!=actual_initial['model_tensor_sha256'] or x['optimizer_state_entries']!=0 for x in initials):raise ValueError('actual rank fresh initialization')
  terminal.update(status='completed',successful_updates=record['successful_updates'],cycles=record['cycle'],
                  component=result,complete_sha256=sha(out/'complete.json'),research_only=False)
  atomic(EVIDENCE/'TRAIN_RESULT.json',terminal,exclusive=True)
  terminal['phase']='freeze'
  freeze_path=ROOT/'SINGLE-MODEL-FREEZE.json'
  freeze_env=env.copy()
  freeze_env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  freeze_deadline=min(time.time()+300,clock['deadline_at']-750)
  if freeze_deadline-time.time()<=150:raise TimeoutError('freeze plus cleanup window unavailable')
  freeze_command=['nice','-n','19',sys.executable,'-B','-u',str(ROOT/'bundle/single_model91.py'),'freeze','--training-folder',str(model_dir),'--output',str(freeze_path)]
  freeze_result=run_component('freeze',freeze_command,freeze_deadline,CONTROL/'freeze-no-ranks',freeze_env)
  frozen=read(freeze_path)
  if frozen.get('cpu_fixture_only') or frozen['source_digest']!=fresh['source_digest']:raise ValueError('production single-model freeze differs')
  terminal['phase']='official'
  official_env=env.copy()
  official_env['TRAIN_IMAGE_CACHE']=str(Path(assets['prepared512_cache']['remote_root'])/'valid/sealed')
  official_root=ROOT/'official'
  official_started=time.time()
  official_seconds=read(ROOT/'CLOCK_POLICY.json')['official_hard_seconds']
  if clock['deadline_at']-150-official_started<official_seconds:raise TimeoutError('full official single-device budget unavailable')
  official_deadline=official_started+official_seconds
  official=assets['manifests']['official_valid']
  official_command=[sys.executable,'-B','-u',str(ROOT/'bundle/single_model91.py'),'official-once','--freeze',str(freeze_path),'--output-root',str(official_root),'--data-root',assets['data_root'],'--eval-manifest',official['remote_path'],'--eval-manifest-sha256',official['sha256'],'--device','cuda:0']
  official_component=run_component('official',official_command,official_deadline,CONTROL/'official-no-ranks',official_env)
  official_result=read(official_root/'OFFICIAL-RESULT.json')
  if official_result['status']!='completed' or (official_result['rows'],official_result['positive'],official_result['negative'])!=(5849,191,5658):raise ValueError('official full coverage not complete')
  terminal.update(status='completed',phase='complete',freeze_component=freeze_result,official_component=official_component,
    single_model_freeze_sha256=sha(freeze_path),official_result_sha256=sha(official_root/'OFFICIAL-RESULT.json'),
    official_metrics=official_result['metrics'],target_auroc_at_least_0_999=official_result['target_auroc_at_least_0_999'])
  atomic(EVIDENCE/'R91_RESULT.json',terminal,exclusive=True)
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

if __name__=='__main__':main()

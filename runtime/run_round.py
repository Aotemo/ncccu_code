import glob,hashlib,json,os,re,shutil,stat,sys,time
from pathlib import Path
from r92_initial_gate import verify_records as verify_fresh_initial_identity, maybe_release as release_fresh_initial
from r115_ssl_gate import maybe_release as release_ssl_initial
ROOT=Path(__file__).resolve().parent
CONTROL=ROOT/'control';EVIDENCE=ROOT/'evidence'
sys.path.insert(0,str(ROOT/'bundle'))
from retrain.site_binding import load_site, expected_pcis, verify_gpu_cpu
SCAN_DIAGNOSTICS={'tolerated_atomic_leaf_disappearances':0,'sample_relative_paths':[]}

def maybe_release(root, atomic, sha):
 if read(Path(root)/'SCIENCE.json').get('ssl_recipe') is not None:
  release_ssl_initial(root,atomic,sha)
 return release_fresh_initial(root,atomic,sha)
ATOMIC_TEMP_NAME=re.compile(r'^\..+\.[A-Za-z0-9_-]{6,}\.tmp$')
CHECKPOINT_TEMP_NAME=re.compile(r'^\.checkpoint-[A-Za-z0-9_-]{6,}\.tmp$')
CHECKPOINT_FILE_NAME=re.compile(r'^checkpoint-[0-9a-f]{32}\.pt(\.json)?$')
def transient_checkpoint_leaf(name):
 return bool(ATOMIC_TEMP_NAME.fullmatch(name) or CHECKPOINT_TEMP_NAME.fullmatch(name)
             or CHECKPOINT_FILE_NAME.fullmatch(name))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def verify_bound():
 binding=read(ROOT/'RUNTIME_BINDING.json')
 site=load_site()
 if site.get('scope')!='formal_exp_v3':raise ValueError('formal runtime cannot use a technical snapshot')
 verify_gpu_cpu(site)
 if sha(ROOT/'RUNTIME_BINDING.json')!=read(CONTROL/'binding-receipt.json')['runtime_binding_sha256']:raise ValueError('binding changed')
 if (binding['round_id']!=site['round_id'] or binding['implementation_ready'] is not True
     or binding['git_commit']!=site['git_commit'] or Path(site['round_root']).resolve()!=ROOT
     or Path(os.environ['NCCCU_SITE_BINDING']).resolve()!=ROOT/'SITE_BINDING.json'
     or binding['site_binding_sha256']!=sha(ROOT/'SITE_BINDING.json')):raise ValueError('binding identity')
 reference=read(ROOT/'SOURCE-MANIFEST.reference.json')
 if binding['source_digest']!=reference['source_digest'] or binding['bundle_source_files']!=reference['after']:raise ValueError('numerical source identity')
 if binding['clock_sha256']!=sha(CONTROL/'round-clock.json') or binding['effective_output_root']!=str(ROOT/'outputs'):raise ValueError('clock/output binding')
 for rel,value in binding['source_files'].items():
  p=ROOT/rel
  if p.is_symlink() or not p.resolve().is_relative_to(ROOT) or sha(p)!=value:raise ValueError('bound source differs: '+rel)
 if sha(read(ROOT/'TRAIN_CONFIG.json')['jobs'][0]['initialization']['path'])!=binding['public_weight_sha256']:raise ValueError('public weight changed')
 return binding

def device_users_before_start():
    users=[];denied=0
    for path in glob.glob('/proc/[0-9]*'):
        pid=int(path.rsplit('/',1)[1])
        if pid==os.getpid():continue
        try:
            values=Path(path+'/stat').read_text().rsplit(')',1)[1].split()
            if values[0] in ('Z','X'):continue
            for fd in os.listdir(path+'/fd'):
                try:
                    target=os.readlink(path+'/fd/'+fd)
                    if target=='/dev/kfd' or target.startswith('/dev/dri/renderD'):
                        users.append({'pid':pid,'start_ticks':values[19],'device':target});break
                except (FileNotFoundError,ProcessLookupError):pass
                except PermissionError:denied+=1
        except (FileNotFoundError,ProcessLookupError):pass
        except PermissionError:denied+=1
    if denied:raise PermissionError('device process scan incomplete: '+str(denied))
    return users

def card_snapshot():
    cards=[]
    pcis=set(expected_pcis(load_site()))
    for p in Path('/sys/class/drm').glob('card[0-9]*'):
        if not p.name[4:].isdigit():continue
        d=(p/'device').resolve()
        if d.name not in pcis:continue
        cards.append({'pci':d.name,'busy_percent':int((d/'gpu_busy_percent').read_text()),
                      'vram_used_bytes':int((d/'mem_info_vram_used').read_text()),
                      'vram_total_bytes':int((d/'mem_info_vram_total').read_text())})
    if {x['pci'] for x in cards}!=pcis:raise ValueError('all admitted PCI cards must be present')
    return sorted(cards,key=lambda x:x['pci'])

def real_directory(path):
    if not stat.S_ISDIR(Path(path).lstat().st_mode):raise ValueError('round directory missing or symlinked: '+str(path))

def tree_bytes():
 root=Path(ROOT);real_directory(root);total=0
 SCAN_DIAGNOSTICS['tolerated_atomic_leaf_disappearances']=0
 SCAN_DIAGNOSTICS['sample_relative_paths']=[]
 def onerror(error):raise error
 for base,directories,files in os.walk(root,followlinks=False,onerror=onerror):
  directory=Path(base);real_directory(root);real_directory(directory)
  for name in directories:real_directory(directory/name)
  for name in files:
   path=directory/name
   try:info=path.lstat()
   except FileNotFoundError as error:
    # Only the two legitimate checkpoint leaves may vanish between listing and lstat:
    # tempfile.mkstemp('.checkpoint-', '.tmp') during save, and old checkpoint blobs
    # removed by CheckpointStore._prune. Any other lost leaf stays fatal.
    if not transient_checkpoint_leaf(name):raise
    real_directory(root);real_directory(directory)
    for parent in directory.relative_to(root).parents:
     if str(parent)!='.':real_directory(root/parent)
    SCAN_DIAGNOSTICS['tolerated_atomic_leaf_disappearances']+=1
    if len(SCAN_DIAGNOSTICS['sample_relative_paths'])<32:
     SCAN_DIAGNOSTICS['sample_relative_paths'].append({'stage':'lstat','path':str(path.relative_to(root)),'error':str(error)})
    continue
   if stat.S_ISLNK(info.st_mode):raise ValueError('round file symlink forbidden: '+str(path))
   total+=info.st_size
 return total

def budget(deadline,minimum_free=17179869184):
 if time.time()>=deadline:raise TimeoutError('round absolute deadline reached')
 free=shutil.disk_usage(ROOT).free;size=tree_bytes()
 if free<minimum_free or size>12884901888:raise RuntimeError('round storage limit reached')
 unknown=SCAN_DIAGNOSTICS['tolerated_atomic_leaf_disappearances']
 return {'free_bytes':free,'new_round_bytes':size,'new_round_bytes_exact':unknown==0,
         'scan_unknown_leaves':unknown,'scan_unknown_samples':SCAN_DIAGNOSTICS['sample_relative_paths'][:8]}

def runtime_safety_snapshot():
    cards=card_snapshot();current=int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
    limit=int(Path('/sys/fs/cgroup/memory.max').read_text().strip())
    if limit<=0 or current*10>limit*9:raise RuntimeError('cgroup RAM exceeded 90 percent')
    if any(c['vram_total_bytes']<=0 or c['vram_used_bytes']*100>c['vram_total_bytes']*85 for c in cards):
        raise RuntimeError('DCU VRAM exceeded 85 percent')
    return {'at':time.time(),'cards':cards,'cgroup_memory_current':current,'cgroup_memory_limit':limit}

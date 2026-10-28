import glob,hashlib,json,os,re,shutil,stat,time
from pathlib import Path
from r92_initial_gate import verify_records as verify_fresh_initial_identity, maybe_release
ROOT=Path(__file__).resolve().parent
CONTROL=ROOT/'control';EVIDENCE=ROOT/'evidence'
PCIS={'0000:09:00.0','0000:77:00.0','0000:85:00.0','0000:f5:00.0'}
SCAN_DIAGNOSTICS={'tolerated_atomic_leaf_disappearances':0,'sample_relative_paths':[]}
ATOMIC_TEMP_NAME=re.compile(r'^\..+\.[A-Za-z0-9_-]{6,}\.tmp$')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def verify_bound():
 binding=read(ROOT/'RUNTIME_BINDING.json')
 if sha(ROOT/'RUNTIME_BINDING.json')!=read(CONTROL/'binding-receipt.json')['runtime_binding_sha256']:raise ValueError('binding changed')
 if binding['round_id']!=ROUND or binding['implementation_ready'] is not True:raise ValueError('binding identity')
 if binding['source_digest']!=SOURCE or binding['bundle_source_files']!=read(ROOT/'SOURCE-MANIFEST.reference.json')['after']:raise ValueError('numerical source identity')
 if binding['clock_sha256']!=sha(CONTROL/'round-clock.json') or binding['effective_output_root']!=str(ROOT/'outputs'):raise ValueError('clock/output binding')
 for rel,value in binding['source_files'].items():
  p=ROOT/rel
  if p.is_symlink() or not p.resolve().is_relative_to(ROOT) or sha(p)!=value:raise ValueError('bound source differs: '+rel)
 if sha(read(ROOT/'TRAIN_CONFIG.json')['jobs'][0]['initialization']['path'])!=binding['public_weight_sha256']:raise ValueError('public weight changed')
 return binding

ROUND='r92-tiny1k-9010-20261028'
SOURCE='57ef83209692eb79e682e04e2b3fe7a060225856f673505ab67396080216d51e'
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
    for p in Path('/sys/class/drm').glob('card[0-9]*'):
        if not p.name[4:].isdigit():continue
        d=(p/'device').resolve()
        if d.name not in PCIS:continue
        cards.append({'pci':d.name,'busy_percent':int((d/'gpu_busy_percent').read_text()),
                      'vram_used_bytes':int((d/'mem_info_vram_used').read_text()),
                      'vram_total_bytes':int((d/'mem_info_vram_total').read_text())})
    if {x['pci'] for x in cards}!=PCIS:raise ValueError('specified four PCI cards not present')
    return sorted(cards,key=lambda x:x['pci'])

def real_directory(path):
    if not stat.S_ISDIR(Path(path).lstat().st_mode):raise ValueError('round directory missing or symlinked: '+str(path))

def tree_bytes():
    root=Path(ROOT);real_directory(root);total=0
    def onerror(error):raise error
    for base,directories,files in os.walk(root,followlinks=False,onerror=onerror):
        directory=Path(base);real_directory(root);real_directory(directory)
        for name in directories:real_directory(directory/name)
        for name in files:
            path=directory/name
            try:info=path.lstat()
            except FileNotFoundError:
                if not ATOMIC_TEMP_NAME.fullmatch(name):raise
                real_directory(root);real_directory(directory)
                for parent in directory.relative_to(root).parents:
                    if str(parent)!='.':real_directory(root/parent)
                SCAN_DIAGNOSTICS['tolerated_atomic_leaf_disappearances']+=1
                if len(SCAN_DIAGNOSTICS['sample_relative_paths'])<32:
                    SCAN_DIAGNOSTICS['sample_relative_paths'].append(str(path.relative_to(root)))
                continue
            if stat.S_ISLNK(info.st_mode):raise ValueError('round file symlink forbidden: '+str(path))
            total+=info.st_size
    return total

def budget(deadline,minimum_free=17179869184):
    if time.time()>=deadline:raise TimeoutError('N03 absolute deadline reached')
    free=shutil.disk_usage(ROOT).free;size=tree_bytes()
    if free<minimum_free or size>12884901888:raise RuntimeError('N03 storage limit reached')
    return {'free_bytes':free,'new_round_bytes':size}

def runtime_safety_snapshot():
    cards=card_snapshot();current=int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
    limit=int(Path('/sys/fs/cgroup/memory.max').read_text().strip())
    if limit<=0 or current*10>limit*9:raise RuntimeError('cgroup RAM exceeded 90 percent')
    if any(c['vram_total_bytes']<=0 or c['vram_used_bytes']*100>c['vram_total_bytes']*85 for c in cards):
        raise RuntimeError('DCU VRAM exceeded 85 percent')
    return {'at':time.time(),'cards':cards,'cgroup_memory_current':current,'cgroup_memory_limit':limit}

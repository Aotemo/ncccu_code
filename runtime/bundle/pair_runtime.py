"""Admitted six-device proof; invoked only inside the named R9 worker."""
import ctypes
import ctypes.util
import json
import os


def normalized_pci(value):
    value=value.lower().strip()
    return '0000:'+value if value.count(':')==1 else value


def rank_device_proof(rank,device):
    import torch
    expected=json.loads(os.environ['R8_EXPECTED_PCIS'])
    parent=json.loads(os.environ['R8_PARENT_CPU_IDS'])
    if len(expected)!=6 or len(set(expected))!=6 or len(parent)!=90 or len(set(parent))!=90:
        raise ValueError('six-card PCI/CPU identity absent')
    if torch.cuda.device_count()!=6 or device.index!=rank:
        raise ValueError('R9 must see exactly logical cuda0..5')
    library=ctypes.CDLL(ctypes.util.find_library('amdhip64') or 'libamdhip64.so')
    function=library.hipDeviceGetPCIBusId
    function.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int];function.restype=ctypes.c_int
    pcis=[]
    for i in range(6):
        buffer=ctypes.create_string_buffer(64)
        if function(buffer,64,i):raise ValueError('HIP could not prove visible physical PCI')
        pcis.append(normalized_pci(buffer.value.decode()))
    if pcis!=[normalized_pci(v) for v in expected]:raise ValueError('visible HIP ordinal/PCI mapping differs')
    affinity=sorted(os.sched_getaffinity(0))
    want=parent[rank*14:rank*14+14]
    if affinity!=sorted(want):raise ValueError('rank actual CPU affinity differs from disjoint fourteen-ID mask')
    props=torch.cuda.get_device_properties(device)
    return {'rank':rank,'pid':os.getpid(),'logical_device':str(device),'visible_pcis':pcis,
            'physical_pci':pcis[rank],'cpu_ids':affinity,'parent_cpu_ids':parent,
            'device_name':props.name,'memory_bytes':props.total_memory,'at':__import__('time').time()}

"""Import the CPU dependencies without constructing a model or initializing HIP."""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys


def main():
    import torch
    threads = int(os.environ['NCCCU_CPU_THREADS'])
    mask = json.loads(os.environ['NCCCU_CPU_MASK'])
    if (sorted(os.sched_getaffinity(0)) != sorted(mask) or len(mask) != threads
            or os.getpriority(os.PRIO_PROCESS, 0) != 19
            or any(os.environ.get(key) != '' for key in
                   ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))):
        raise ValueError('explicit CPU allocation/nice19/hidden accelerator context differs')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    modules = {name: importlib.import_module(name)
               for name in ('torch', 'torchvision', 'numpy', 'PIL', 'timm')}
    from timm.models import convnext as source
    from torchvision.models import convnext as target
    from timm.layers.activations import GELU
    implementations = {}
    for name, obj in (('timm_convnext', source), ('torchvision_convnext', target),
                      ('timm_gelu', GELU)):
        path = Path(inspect.getfile(obj)).resolve(strict=True)
        implementations[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    implementations['timm_gelu']['forward_sha256'] = hashlib.sha256(inspect.getsource(GELU.forward).encode()).hexdigest()
    if torch.cuda.is_initialized():
        raise ValueError('CPU dependency import initialized an accelerator')
    print(json.dumps({'scope': 'imports_only_no_model_no_forward', 'python': sys.executable,
        'python_version': sys.version.split()[0],
        'dependencies': {name: module.__version__ for name, module in modules.items()},
        'implementations': implementations, 'mask': mask, 'nice': 19,
        'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'allocation': os.environ['NCCCU_CPU_ALLOCATION'],
        'allocation_sha256': os.environ['NCCCU_CPU_ALLOCATION_SHA256'],
        'gpu_initialized': False}, indent=2))


if __name__ == '__main__':
    main()

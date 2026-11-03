"""One process per card, reference-batch sharding and coordinated durable updates."""
from datetime import timedelta
import copy
import torch
from torch import nn
import torch.distributed as dist
from .data import BalancedBatchSampler


def layout(reference_physical, effective, positive, world_size, *, per_rank_microbatch=None):
    if world_size not in (1,2,4,6,8) or effective%world_size or effective!=2*reference_physical:
        raise ValueError('DDP supports 1/2/4/6/8 ranks and two reference microbatches per global update')
    if not 0<positive<effective or positive*reference_physical%effective:
        raise ValueError('invalid global positive quota')
    physical=reference_physical if world_size==1 else effective//world_size
    result={'version':1,'world_size':world_size,'global_effective_batch':effective,
            'global_positive_quota':positive,'reference_physical_batch':reference_physical,
            'per_rank_batch':physical,'accumulation':2 if world_size==1 else 1,
            'normalization_group_size':max(1,world_size//2),
            'normalization_policy':'reference-microbatch-sequential-ema-v1',
            'sampler_policy':'reference-seeds-contiguous-global-shards-v1'}
    if per_rank_microbatch is not None:
        if type(per_rank_microbatch)!=int or per_rank_microbatch<=0 or effective%(per_rank_microbatch*world_size):
            raise ValueError('explicit per-rank microbatch must exactly divide the fixed global batch')
        result.update(per_rank_batch=per_rank_microbatch,accumulation=effective//(per_rank_microbatch*world_size),
                      explicit_per_rank_microbatch=per_rank_microbatch,normalization_group_size=1,
                      normalization_policy='no-batchnorm-explicit-microbatch-v1')
    return result


class RankBatchSampler:
    def __init__(self,rows,fit_indices,physical_batch,effective_batch,positive_per_batch,
                 seed,start_update,num_updates,*,rank,world_size,per_rank_microbatch=None):
        self.plan=layout(physical_batch,effective_batch,positive_per_batch,world_size,per_rank_microbatch=per_rank_microbatch)
        if not 0<=rank<world_size:raise ValueError('rank outside world')
        self.rank=rank;self.world_size=world_size;self.num_updates=num_updates
        self.reference=BalancedBatchSampler(rows,fit_indices,physical_batch,effective_batch,
                                            positive_per_batch,seed,start_update,num_updates)
    def __len__(self):return self.num_updates*self.plan['accumulation']
    def __iter__(self):
        iterator=iter(self.reference)
        for _ in range(self.num_updates):
            first,second=next(iterator),next(iterator)
            batch=first+second
            width=self.plan['per_rank_batch']
            for micro in range(self.plan['accumulation']):
                start=(micro*self.world_size+self.rank)*width
                yield batch[start:start+width]


def ema_coefficients(momentum,tracked):
    m1=momentum if momentum is not None else 1/(tracked+1)
    m2=momentum if momentum is not None else 1/(tracked+2)
    return 1-m2,m2/m1,(m2/m1)*(1-m1)


def ema_merge(old,first,second,momentum,tracked):
    a,b,c=ema_coefficients(momentum,tracked)
    return a*first+b*second-c*old


class RankContext:
    def __init__(self,rank,world_size,device,init_method=None,timeout_seconds=120):
        self.rank=rank;self.world_size=world_size;self.device=torch.device(device)
        if not 0<=rank<world_size or world_size not in (1,2,4,6,8):raise ValueError('invalid rank/world')
        if self.device.type=='cuda':torch.cuda.set_device(self.device)
        self.group=None
        if world_size>1:
            dist.init_process_group('nccl' if self.device.type=='cuda' else 'gloo',
                init_method=init_method,rank=rank,world_size=world_size,
                timeout=timedelta(seconds=timeout_seconds))
            if world_size>2:
                size=world_size//2
                for start in (0,size):
                    group=dist.new_group(list(range(start,start+size)),timeout=timedelta(seconds=timeout_seconds))
                    if start<=rank<start+size:self.group=group
    @property
    def primary(self):return self.rank==0
    def values(self,value):
        if self.world_size==1:return [value]
        result=[None]*self.world_size
        dist.all_gather_object(result,value)
        return result
    def broadcast(self,value):
        if self.world_size==1:return value
        values=[value if self.primary else None]
        dist.broadcast_object_list(values,src=0)
        return values[0]
    def broadcast_flag(self,value):
        """Rank0 decides one boolean; a fixed int32 tensor broadcasts it without object pickling."""
        if self.world_size==1:return bool(value)
        flag=torch.tensor([int(bool(value))],device=self.device,dtype=torch.int32)
        dist.broadcast(flag,src=0)
        return bool(flag.item())
    def any_flags(self,values):
        """Fixed-length local booleans: one WORLD MAX and one host transfer, never per-flag collectives."""
        flags=[bool(value) for value in values]
        if self.world_size==1:return flags
        tensor=torch.tensor([int(flag) for flag in flags],device=self.device,dtype=torch.int32)
        dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
        return [bool(value) for value in tensor.cpu().tolist()]
    def any(self,value):
        flag=torch.tensor(int(bool(value)),device=self.device,dtype=torch.int32)
        if self.world_size>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        return bool(flag.item())
    def agreed(self,value):
        flag=torch.tensor(int(bool(value)),device=self.device,dtype=torch.int32)
        if self.world_size>1:dist.all_reduce(flag,op=dist.ReduceOp.SUM)
        count=int(flag.item())
        if count not in (0,self.world_size):raise RuntimeError('ranks disagree on AMP optimizer update')
        return bool(count)
    def maximum(self,value):
        tensor=torch.tensor(float(value),device=self.device,dtype=torch.float64)
        if self.world_size>1:dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
        return float(tensor.item())
    def maximum_values(self,values):
        """Reduce related telemetry in one collective and one host transfer."""
        if self.world_size==1:return [float(v) for v in values]
        tensor=torch.tensor(values,device=self.device,dtype=torch.float64)
        dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
        return tensor.cpu().tolist()
    def gather_indices(self,values):
        """Configured global batches give every rank the same exposure count."""
        if self.world_size==1:return [list(values)]
        tensor=torch.tensor(values,device=self.device,dtype=torch.int64)
        gathered=[torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered,tensor)
        return torch.stack(gathered).cpu().tolist()
    def barrier(self):
        if self.world_size>1:dist.barrier()
    def prepare_model(self,model):
        from .flair_model import FlairResnetBinary
        from .imagenet_resnet import ImageNetResnetBinary
        from .efficientnet_model import EfficientNetBinary
        if isinstance(model,(FlairResnetBinary,ImageNetResnetBinary,EfficientNetBinary)):
            model.assert_bn_mode()
            return model
        if self.world_size==6 and any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()):
            raise ValueError('frozen six-rank forbids BatchNorm models')
        if self.world_size>2 and any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()):
            if self.device.type!='cuda':raise ValueError('reference-group SyncBatchNorm needs GPU DDP')
            model=nn.SyncBatchNorm.convert_sync_batchnorm(model,self.group)
        return model
    def wrap(self,model):
        if self.world_size==1:return model
        return nn.parallel.DistributedDataParallel(model,
            device_ids=[self.device.index] if self.device.type=='cuda' else None,
            broadcast_buffers=False)
    def close(self):
        if self.world_size>1 and dist.is_initialized():dist.destroy_process_group()


class ReferenceBatchNorm:
    """Merge the two parallel reference groups as their original sequential EMA."""
    def __init__(self,model,context):
        self.context=context
        from .flair_model import FlairResnetBinary
        from .imagenet_resnet import ImageNetResnetBinary
        from .efficientnet_model import EfficientNetBinary
        if isinstance(model,(FlairResnetBinary,ImageNetResnetBinary,EfficientNetBinary)):
            model.assert_bn_mode()
            self.layers=[];self.before=[]
            return
        self.layers=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)
                     and m.track_running_stats]
        self.before=[]
    def capture(self):
        if self.context.world_size==1:return
        self.before=[(m.running_mean.detach().clone(),m.running_var.detach().clone(),
                      int(m.num_batches_tracked)) for m in self.layers]
    @torch.no_grad()
    def merge(self):
        ctx=self.context
        if ctx.world_size==1 or not self.layers:return
        group_size=ctx.world_size//2
        first=ctx.rank<group_size
        packed=[]
        for m,(mean,var,tracked) in zip(self.layers,self.before):
            a,b,c=ema_coefficients(m.momentum,tracked)
            coefficient=(a if first else b)/group_size
            packed.extend([m.running_mean.flatten()*coefficient,m.running_var.flatten()*coefficient])
        merged=torch.cat(packed)
        dist.all_reduce(merged,op=dist.ReduceOp.SUM)
        offset=0
        for m,(mean,var,tracked) in zip(self.layers,self.before):
            a,b,c=ema_coefficients(m.momentum,tracked)
            n=mean.numel()
            m.running_mean.copy_(merged[offset:offset+n].view_as(mean)-c*mean);offset+=n
            m.running_var.copy_(merged[offset:offset+n].view_as(var)-c*var);offset+=n
            m.num_batches_tracked.fill_(tracked+2)
        self.before=[]

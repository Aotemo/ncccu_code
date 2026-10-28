"""Exact dev misordered-pair count and stable AP for N05 selection."""
import math

def metrics(predictions):
    groups={}
    for row in predictions:
        label=row['label'];score=float(row['logit'])
        if label not in (0,1) or not math.isfinite(score):raise ValueError('invalid N05 dev score/label')
        bucket=groups.setdefault(score,[0,0]);bucket[label]+=1
    pos=sum(v[1] for v in groups.values());neg=sum(v[0] for v in groups.values())
    if not pos or not neg:raise ValueError('N05 dev needs two classes')
    below=correct=ties=0
    for score in sorted(groups):
        n,p=groups[score];correct+=p*below;ties+=p*n;below+=n
    error=2*pos*neg-2*correct-ties
    tp=fp=0;last_recall=0.;terms=[]
    for score in sorted(groups,reverse=True):
        n,p=groups[score];tp+=p;fp+=n;recall=tp/pos
        terms.append((recall-last_recall)*(tp/(tp+fp)));last_recall=recall
    return {'error_twice':error,'ap':math.fsum(terms),'auroc':(2*pos*neg-error)/(2*pos*neg)}

def improved(new,old):
    def key(value):
        e=value.get('error_twice');ap=value.get('ap');update=value.get('successful_updates')
        if type(e) is not int or e<0 or type(update) is not int or update<0:
            raise ValueError('exact error and update required')
        if isinstance(ap,bool) or not isinstance(ap,(int,float)) or not math.isfinite(ap) or not 0<=ap<=1:
            raise ValueError('finite AP required')
        return e,-ap,update
    newkey=key(new)
    return old is None or newkey<key(old)

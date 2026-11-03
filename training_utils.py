"""Shared train-only data labels, normalization and ranking pairs."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import expit
from common import DATA,read_jsonl,parse_answer

def labels(bank, split):
    assert split in ('train','validation'), 'Training never opens test references'
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    return np.array([[int((v:=parse_answer(c['text'])) is not None and v==parse_answer(r['answer'])) for c in b['candidates']] for b,r in zip(bank,refs)])

def accuracy(y, scores, mask=None):
    correct = y[np.arange(len(y)),scores.argmax(axis=1)]
    return float(correct.mean() if mask is None else correct[mask].mean())

def save_json(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def normalize(train,other):
    mean=train.reshape(-1,train.shape[-1]).mean(0); scale=train.reshape(-1,train.shape[-1]).std(0); scale[scale<1e-6]=1
    return np.clip((train-mean)/scale,-10,10),np.clip((other-mean)/scale,-10,10),mean,scale

def pairs(x,y,groups,weight):
    differences=[]; weights=[]
    for i,labels in enumerate(y):
        pos=np.flatnonzero(labels); neg=np.flatnonzero(1-labels)
        for j in pos:
            for k in neg:
                differences.append(x[i,j]-x[i,k]); weights.append((weight if groups[i][k] is not None else 1)/max(1,len(pos)*len(neg)))
    dx=np.asarray(differences)
    return np.r_[dx,-dx],np.r_[np.ones(len(dx)),np.zeros(len(dx))],np.tile(weights,2)

def fit_baseline_neural(xt,yt,hidden=64,objective='pairwise',seed=7506,epochs=20):
    """Mini-batch Adam with group-level pairwise or listwise loss and BCE auxiliary."""
    rng=np.random.default_rng(seed); d=xt.shape[-1]
    params={'w1':(rng.standard_normal((d,hidden))*np.sqrt(2/d)).astype(np.float32),'b1':np.zeros(hidden,dtype=np.float32),'w2':(rng.standard_normal((hidden,1))*np.sqrt(1/hidden)).astype(np.float32),'b2':np.zeros(1,dtype=np.float32)}
    m={k:np.zeros_like(v) for k,v in params.items()}; v={k:np.zeros_like(a) for k,a in params.items()}
    best=(-1,None,None,None); step=0; tick=time.perf_counter()
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(yt))
        for begin in range(0,len(order),64):
            take=order[begin:begin+64]; x=xt[take].reshape(-1,d); y=yt[take]; size=len(take)
            pre=x@params['w1']+params['b1']; h=np.maximum(pre,0)
            z=(h@params['w2']+params['b2']).reshape(-1,4)
            if objective=='bce':
                dz=(expit(z)-y)/y.size
            else:
                if objective=='pairwise':
                    valid=(y[:,:,None]==1)&(y[:,None,:]==0)
                    counts=valid.sum((1,2)); w=valid/np.maximum(counts,1)[:,None,None]
                    diff=z[:,None,:]-z[:,:,None]
                    g=expit(diff)*w
                    dz=-g.sum(2)+g.sum(1)
                else:
                    shifted=z-z.max(1,keepdims=True); p=np.exp(shifted); p/=p.sum(1,keepdims=True)
                    correct=p*y; correct/=np.maximum(correct.sum(1,keepdims=True),1e-12)
                    dz=p-correct
                # Preserve a correctness-probability anchor even for all-correct groups.
                dz=dz/size + .2*(expit(z)-y)/y.size
            gz=dz.astype(np.float32).reshape(-1,1)
            dh=(gz@params['w2'].T)*(pre>0)
            grads={'w2':h.T@gz,'b2':gz.sum(0),'w1':x.T@dh,'b1':dh.sum(0)}
            step+=1
            for k,g in grads.items():
                if k.startswith('w'): g+=1e-4*params[k]
                m[k]=.9*m[k]+.1*g; v[k]=.999*v[k]+.001*g*g
                params[k]-=.001*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    return params

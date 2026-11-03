"""Train-internal selection of question/solution interaction models and matched controls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='4'
import time
import argparse
import shutil
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from common import load_inputs
from features import extract_features
from math_consistency import extract_math_features
from training_utils import labels,accuracy
from training_utils import save_json
from training_utils import normalize,pairs
from selector import enhanced_input
from selector import documents,encode,linear_input,bilinear_logits
from selector import aggregate,member_scores,score_v2


def fit_encoder(bank):
    q,r=documents(bank)
    vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=10000,min_df=3,sublinear_tf=True,dtype=np.float32)
    matrix=vectorizer.fit_transform(q+r)
    svd=TruncatedSVD(n_components=128,n_iter=5,random_state=7506).fit(matrix)
    return dict(vectorizer=vectorizer,components=svd.components_.astype(np.float32))


def gradients(x,q,y,w,l2):
    """Pairwise Bradley-Terry plus .2 BCE, with a learnable low-rank bilinear term."""
    z=bilinear_logits(x,q,w); valid=(y[:,:,None]==1)&(y[:,None,:]==0)
    count=valid.sum((1,2)); weights=valid/np.maximum(count,1)[:,None,None]
    g=expit(z[:,None,:]-z[:,:,None])*weights
    dz=(-g.sum(2)+g.sum(1))/len(y)+.2*(expit(z)-y)/y.size
    denom=np.sqrt(w['u'].shape[1]); qu=q@w['u']; xv=x@w['v']
    grads=dict(coef=x.reshape(-1,x.shape[-1]).T@dz.ravel(),bias=np.array([dz.sum()]),u=q.T@(dz[:,:,None]*xv).sum(1)/denom,v=x.reshape(-1,x.shape[-1]).T@(dz[:,:,None]*qu[:,None,:]).reshape(-1,qu.shape[-1])/denom)
    value=np.sum(np.logaddexp(0,z[:,None,:]-z[:,:,None])*weights)/len(y)+.2*np.mean(np.logaddexp(0,z)-y*z)
    for k in ['coef','u','v']:
        reg=.0001 if k=='coef' else l2
        value+=.5*reg*np.sum(w[k]**2); grads[k]+=reg*w[k]
    return float(value),grads


def fit_bilinear(x,q,y,xv,qv,yv,groups,rank,l2,epochs,select_epoch):
    rng=np.random.default_rng(7506); d=x.shape[-1]
    w=dict(coef=np.zeros(d,dtype=np.float32),bias=np.zeros(1,dtype=np.float32),u=(rng.standard_normal((q.shape[-1],rank))*.1/np.sqrt(q.shape[-1])).astype(np.float32),v=(rng.standard_normal((d,rank))/np.sqrt(d)).astype(np.float32))
    first={k:np.zeros_like(a) for k,a in w.items()}; second={k:np.zeros_like(a) for k,a in w.items()}; step=0; best=None; checks=[]
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(y))
        for begin in range(0,len(order),64):
            take=order[begin:begin+64]; _,grad=gradients(x[take],q[take],y[take],w,l2); step+=1
            for k,g in grad.items():
                first[k]=.9*first[k]+.1*g; second[k]=.999*second[k]+.001*g*g
                w[k]-=.001*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        if (select_epoch and epoch in [10,20,40]) or (not select_epoch and epoch==epochs):
            p=aggregate(expit(bilinear_logits(xv,qv,w)),groups); acc=accuracy(yv,p); checks.append(dict(epoch=epoch,accuracy=100*acc))
            if best is None or acc>best[0]: best=(acc,{k:a.copy() for k,a in w.items()},epoch,p)
    return best,checks


def check_gradients():
    rng=np.random.default_rng(14); x=rng.normal(size=(2,4,5)); q=rng.normal(size=(2,3)); y=np.array([[1,0,1,0],[1,1,1,1]])
    w=dict(coef=rng.normal(size=5)*.1,bias=np.array([.1]),u=rng.normal(size=(3,2))*.1,v=rng.normal(size=(5,2))*.1)
    _,g=gradients(x,q,y,w,.001); errors=[]
    for k in w:
        for index in [tuple([0]*w[k].ndim),tuple(s-1 for s in w[k].shape)]:
            a={n:v.copy() for n,v in w.items()}; b={n:v.copy() for n,v in w.items()}; a[k][index]+=1e-6; b[k][index]-=1e-6
            numeric=(gradients(x,q,y,a,.001)[0]-gradients(x,q,y,b,.001)[0])/(2e-6)
            errors.append(abs(numeric-g[k][index]))
    assert max(errors)<1e-6,errors
    return max(errors)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=Path('retraining/search')); parser.add_argument('--baseline',type=Path,default=Path('checkpoint/v2')); args=parser.parse_args()
    start=time.perf_counter(); gradient_error=check_gradients(); out=args.output; cand=out/'candidates'; cand.mkdir(parents=True,exist_ok=True)
    bt,at=load_inputs('train'); yt=labels(bt,'train'); ft,_,gt=extract_features(bt); base=enhanced_input(at,ft,extract_math_features(bt))
    tr,va=train_test_split(np.arange(len(bt)),test_size=.2,random_state=7506,stratify=yt.sum(1)); trainbank=[bt[i] for i in tr]; heldbank=[bt[i] for i in va]; tg=[gt[i] for i in tr]; vg=[gt[i] for i in va]
    encoder=fit_encoder(trainbank); internal=[]
    for dimensions in [64,128]:
        qt,rt=encode(trainbank,encoder,dimensions); qv,rv=encode(heldbank,encoder,dimensions)
        for interact in [False,True]:
            tx,vx,_,_=normalize(linear_input(base[tr],qt,rt,interact),linear_input(base[va],qv,rv,interact)); dx,dy,sw=pairs(tx,yt[tr],tg,3)
            for c in [.003,.01,.03]:
                m=LogisticRegression(C=c,fit_intercept=False,max_iter=1000,random_state=7506).fit(dx,dy,sample_weight=sw); assert m.n_iter_[0]<1000
                question_start=1160+dimensions; assert np.max(np.abs(m.coef_.ravel()[question_start:question_start+dimensions]))<1e-10
                p=aggregate(expit(vx@m.coef_.ravel()),vg)
                record=dict(kind='linear',dimensions=dimensions,interact=interact,C=c,internal_accuracy=100*accuracy(yt[va],p)); internal.append(record); print(record,flush=True)
    qt,rt=encode(trainbank,encoder,128); qv,rv=encode(heldbank,encoder,128)
    tx,vx,_,_=normalize(np.concatenate([base[tr],rt],axis=-1),np.concatenate([base[va],rv],axis=-1)); qm=qt.mean(0); qs=qt.std(0); qs[qs<1e-6]=1; qt=np.clip((qt-qm)/qs,-10,10); qv=np.clip((qv-qm)/qs,-10,10)
    for rank in [16,32]:
        for l2 in [.001,.01]:
            best,checks=fit_bilinear(tx,qt,yt[tr],vx,qv,yt[va],vg,rank,l2,40,True)
            record=dict(kind='bilinear',dimensions=128,rank=rank,l2=l2,epoch=best[2],internal_accuracy=100*best[0],epoch_checks=checks); internal.append(record); print(record,flush=True)
    selections=[max([r for r in internal if r['kind']=='linear' and r['interact']==flag],key=lambda r:r['internal_accuracy']) for flag in [False,True]]
    selections.append(max([r for r in internal if r['kind']=='bilinear'],key=lambda r:r['internal_accuracy']))
    save_json(out/'internal_selection.json',dict(settings=internal,selected=selections,training_questions=len(tr),heldout_questions=len(va),official_validation_used=False,gradient_max_error=gradient_error))
    # All architecture/regularization/epoch choices above precede loading official validation.
    bv,av=load_inputs('validation'); yv=labels(bv,'validation'); fv,_,gv=extract_features(bv); basev=enhanced_input(av,fv,extract_math_features(bv))
    encoder=fit_encoder(bt); encoder_file=cand/'text_encoder.joblib'; joblib.dump(encoder,encoder_file,compress=0)
    models=[]; records=[]
    for i,s in enumerate(selections):
        qt,rt=encode(bt,encoder,s['dimensions']); qv,rv=encode(bv,encoder,s['dimensions'])
        if s['kind']=='linear':
            tx,vx,mean,scale=normalize(linear_input(base,qt,rt,s['interact']),linear_input(basev,qv,rv,s['interact'])); dx,dy,sw=pairs(tx,yt,gt,3)
            m=LogisticRegression(C=s['C'],fit_intercept=False,max_iter=1000,random_state=7506).fit(dx,dy,sample_weight=sw); assert m.n_iter_[0]<1000
            w=dict(mean=mean,scale=scale,coef=m.coef_.ravel().astype(np.float32)); p=aggregate(expit(vx@w['coef']),gv)
        else:
            tx,vx,mean,scale=normalize(np.concatenate([base,rt],axis=-1),np.concatenate([basev,rv],axis=-1)); qm=qt.mean(0); qs=qt.std(0); qs[qs<1e-6]=1; qt=np.clip((qt-qm)/qs,-10,10); qv=np.clip((qv-qm)/qs,-10,10)
            best,_=fit_bilinear(tx,qt,yt,vx,qv,yv,gv,s['rank'],s['l2'],s['epoch'],False); w=dict(mean=mean,scale=scale,qmean=qm,qscale=qs,**best[1]); p=best[3]
        file=cand/f'text_{i}.npz'; np.savez(file,**w); models.append((p,file,s)); records.append(dict(**s,validation_accuracy=100*accuracy(yv,p))); print(records[-1],flush=True)
    old=score_v2(bv,av,args.baseline); hard=np.array([len(set(v for v in g if v is not None))>=2 for g in gv]); best=(accuracy(yv,old),None,0,old); fusion=[]
    for i,(p,file,s) in enumerate(models):
        for weight in [.1,.25,.5,.75,1]:
            score=weight*p+(1-weight)*old; acc=accuracy(yv,score)
            fusion.append(dict(index=i,kind=s['kind'],interact=s.get('interact'),text_weight=weight,validation_accuracy=100*acc,hard_validation_accuracy=100*accuracy(yv,score,hard)))
            if acc>best[0]: best=(acc,i,weight,score)
    acc,index,weight,p=best; bundle=out/'model_bundle'; bundle.mkdir(exist_ok=True)
    def config(s,weight): return dict(kind=s['kind'],dimensions=s['dimensions'],interact=s.get('interact'),text_weight=weight,test_used_for_selection=False)
    def copy_text(dst,file,s,weight):
        dst.mkdir(exist_ok=True); shutil.copyfile(file,dst/'text_model.npz'); shutil.copyfile(encoder_file,dst/'text_encoder.joblib'); save_json(dst/'config.json',config(s,weight))
    if index is None: save_json(bundle/'config.json',dict(text_weight=0,test_used_for_selection=False))
    else: copy_text(bundle,models[index][1],models[index][2],weight)
    if weight<1: shutil.copytree(args.baseline,bundle/'v2',dirs_exist_ok=True)
    for i,(_,file,s) in enumerate(models): copy_text(out/f'text_{i}_bundle',file,s,1)
    summary=dict(model_fits=19,preprocessing_fits=4,experiments=records,fusion_grid=fusion,selected_index=index,text_weight=weight,validation_accuracy=100*acc,hard_validation_accuracy=100*accuracy(yv,p,hard),strict_improvement=bool(acc>accuracy(yv,old)),test_used_for_selection=False,gradient_max_error=gradient_error,seconds=time.perf_counter()-start)
    save_json(out/'decision.json',summary); np.savez(out/'validation_analysis.npz',labels=yv,selected_scores=p,candidate_scores=np.array([m[0] for m in models]),baseline_scores=old,hard=hard)
    print('FROZEN',summary,flush=True)


if __name__=='__main__': main()

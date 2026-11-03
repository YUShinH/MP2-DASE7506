"""Reproduce the selected checkpoint from training data with frozen settings.

No validation or test inputs/references are loaded here. Use search.py only to
replay the documented final development search, rather than fixed retraining.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='4'
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import load_inputs
from features import extract_features
from math_consistency import extract_math_features
from training_utils import labels,normalize,pairs,fit_baseline_neural,save_json
from selector import model_input,enhanced_input,encode
from search import fit_encoder,fit_bilinear


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=Path('retraining/checkpoint')); args=parser.parse_args()
    if args.output.resolve()==Path('checkpoint').resolve(): parser.error('Use a separate output folder to preserve the submitted checkpoint')
    started=time.perf_counter(); out=args.output; (out/'v2').mkdir(exist_ok=True,parents=True)
    bank,arrays=load_inputs('train'); y=labels(bank,'train'); f,_,groups=extract_features(bank)
    x=model_input(arrays['embeddings'],f); tx,_,mean,scale=normalize(x,x[:1]); dx,dy,sw=pairs(tx,y,groups,3)
    pair=LogisticRegression(C=.01,fit_intercept=False,max_iter=500,random_state=7506).fit(dx,dy,sample_weight=sw)
    np.savez(out/'v2/pair.npz',mean=mean,scale=scale,coef=pair.coef_.ravel().astype(np.float32))
    print('Selected pairwise baseline reproduced',flush=True)
    x=model_input(arrays['embeddings'],f,True); tx,_,mean,scale=normalize(x,x[:1])
    params=fit_baseline_neural(tx,y,hidden=64,objective='pairwise',seed=7506,epochs=20)
    np.savez(out/'v2/neural.npz',mean=mean,scale=scale,**params)
    shutil.copyfile('checkpoint/v2/config.json',out/'v2/config.json')
    print('Selected 20-epoch neural baseline reproduced',flush=True)
    encoder=fit_encoder(bank); joblib.dump(encoder,out/'text_encoder.joblib',compress=0)
    q,r=encode(bank,encoder,128); base=enhanced_input(arrays,f,extract_math_features(bank))
    x=np.concatenate([base,r],axis=-1); tx,_,mean,scale=normalize(x,x[:1]); qmean=q.mean(0); qscale=q.std(0); qscale[qscale<1e-6]=1; q=np.clip((q-qmean)/qscale,-10,10)
    best,_=fit_bilinear(tx,q,y,tx[:1],q[:1],y[:1],groups[:1],rank=32,l2=.01,epochs=40,select_epoch=False)
    np.savez(out/'text_model.npz',mean=mean,scale=scale,qmean=qmean,qscale=qscale,**best[1])
    shutil.copyfile('checkpoint/config.json',out/'config.json')
    assets={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file()}
    assert sum(r['bytes'] for r in assets.values())<16*2**20
    summary=dict(seed=7506,training_questions=len(bank),settings='frozen selected recipe; no validation/test read',seconds=time.perf_counter()-started,asset_bytes=sum(r['bytes'] for r in assets.values()),assets=assets)
    save_json(out.parent/'training_summary.json',summary)
    print(json.dumps(dict(output=str(out),seconds=summary['seconds'],asset_bytes=summary['asset_bytes'])),flush=True)


if __name__=='__main__': main()

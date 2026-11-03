"""Explicit question/solution text interactions, trained only on supplied training data."""
import json
import re
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from features import extract_features
from math_consistency import extract_math_features


def model_input(embeddings, features, contextual=False):
    e=np.array(embeddings,dtype=np.float32,copy=True)
    e/=np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-12)
    parts=[e,features.astype(np.float32)]
    if contextual: parts.append(e-e.mean(axis=1,keepdims=True))
    return np.concatenate(parts,axis=-1)

def aggregate(scores, groups, mode='mean'):
    out=scores.copy()
    for i, values in enumerate(groups):
        for j,v in enumerate(values):
            if v is None:
                out[i,j]=-1.0
            else:
                same=scores[i,[k for k,w in enumerate(values) if w==v]]
                out[i,j]=same.mean() if mode=='mean' else same.max() if mode=='max' else same.sum()
    return out

def member_scores(x, weights):
    x=(x-weights['mean'])/weights['scale']
    x=np.clip(x,-10,10)
    if 'w1' in weights:
        hidden=np.maximum(x@weights['w1']+weights['b1'],0)
        z=(hidden@weights['w2']+weights['b2']).reshape(-1,4)
    else:
        z=(x@weights['coef']).reshape(-1,4)
    return expit(z)

def enhanced_input(arrays,features,math_features):
    return np.concatenate([model_input(arrays['embeddings'],features),math_features],axis=-1)

def score_v2(bank,arrays,checkpoint):
    root=Path(checkpoint)
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    assert not config['v1_weight'], 'Submitted baseline has no v1 inference member'
    f,_,groups=extract_features(bank); inputs={}; result=np.zeros((len(bank),4),dtype=np.float64)
    for member in config['members']:
        context=member['contextual']
        if context not in inputs:
            x=model_input(arrays['embeddings'],f,context); inputs[context]=x.reshape(-1,x.shape[-1])
        with np.load(root/member['file'],allow_pickle=False) as weights: p=member_scores(inputs[context],weights)
        result+=member['weight']*aggregate(p,groups,member['pool'])
    return result


def documents(bank):
    clean=lambda t:re.sub(r'\d+(?:[.,/]\d+)*',' NUM ',t.lower())
    return [clean(r['question']) for r in bank],[clean(c['text']) for r in bank for c in r['candidates']]


def encode(bank,encoder,dimensions):
    qdocs,rdocs=documents(bank); vectorizer=encoder['vectorizer']; components=encoder['components'][:dimensions]
    q=np.asarray(vectorizer.transform(qdocs)@components.T,dtype=np.float32)
    r=np.asarray(vectorizer.transform(rdocs)@components.T,dtype=np.float32).reshape(len(bank),4,dimensions)
    q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    r/=np.maximum(np.linalg.norm(r,axis=-1,keepdims=True),1e-12)
    return q,r


def linear_input(base,q,r,interact):
    question=np.broadcast_to(q[:,None,:],r.shape)
    products=question*r; distance=np.abs(question-r); cosine=products.sum(-1,keepdims=True)
    if not interact:
        products=np.zeros_like(products); distance=np.zeros_like(distance); cosine=np.zeros_like(cosine)
    return np.concatenate([base,r,question,products,distance,cosine],axis=-1).astype(np.float32)


def bilinear_logits(x,q,w):
    rank=w['u'].shape[1]
    return x@w['coef']+((q@w['u'])[:,None,:]*(x@w['v'])).sum(-1)/np.sqrt(rank)+w['bias']


def score_bank(bank,arrays,checkpoint):
    root=Path(checkpoint) if checkpoint else Path(__file__).parent/'checkpoint'
    cfg=json.loads((root/'config.json').read_text())
    weight=cfg['text_weight']
    if not weight: return score_v2(bank,arrays,root/'v2')
    encoder=joblib.load(root/'text_encoder.joblib')
    q,r=encode(bank,encoder,cfg['dimensions']); f,_,groups=extract_features(bank)
    base=enhanced_input(arrays,f,extract_math_features(bank))
    with np.load(root/'text_model.npz',allow_pickle=False) as w:
        if cfg['kind']=='linear':
            x=linear_input(base,q,r,cfg['interact']); p=member_scores(x.reshape(-1,x.shape[-1]),w)
        else:
            x=np.concatenate([base,r],axis=-1)
            x=np.clip((x-w['mean'])/w['scale'],-10,10)
            q=np.clip((q-w['qmean'])/w['qscale'],-10,10)
            p=expit(bilinear_logits(x,q,w))
    p=aggregate(p,groups)
    return weight*p+(1-weight)*score_v2(bank,arrays,root/'v2') if weight<1 else p


def predict(bank,arrays,checkpoint):
    return score_bank(bank,arrays,checkpoint).argmax(1).tolist() if bank else []

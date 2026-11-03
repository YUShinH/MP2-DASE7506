"""Post-freeze reliability and paired analysis; does not train or select models."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='4'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from common import load_inputs,read_jsonl,parse_answer,DATA
from selector import score_bank,score_v2


def reliability(scores,y,mask):
    confidence=scores.max(1)[mask]; correct=y[np.arange(len(y)),scores.argmax(1)][mask]; order=np.argsort(-confidence,kind='stable')
    coverage=[dict(coverage=c,n=int(np.ceil(len(correct)*c)),accuracy=float(correct[order[:int(np.ceil(len(correct)*c))]].mean())) for c in [.25,.5,.75,1]]
    bins=[]; ece=0.
    for i in range(5):
        m=(confidence>=i/5)&((confidence<(i+1)/5) if i<4 else (confidence<=1))
        if m.any():
            score=float(confidence[m].mean()); accuracy=float(correct[m].mean()); ece+=float(m.mean())*abs(score-accuracy)
            bins.append(dict(lower=i/5,upper=(i+1)/5,n=int(m.sum()),mean_score=score,accuracy=accuracy))
    return dict(n=len(correct),brier=float(np.mean((confidence-correct)**2)),ece_5_bins=ece,bins=bins,risk_coverage=coverage,calibrated=False)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('results/analysis.json')); args=p.parse_args()
    bank,arrays=load_inputs('validation'); references=read_jsonl(DATA/'gsm8k_validation.jsonl')
    y=np.array([[int(parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer'])) for c in b['candidates']] for b,r in zip(bank,references)])
    current=score_bank(bank,arrays,Path('checkpoint')); baseline=score_v2(bank,arrays,Path('checkpoint/v2'))
    groups=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]; hard=np.array([len(set(v for v in g if v is not None))>=2 for g in groups])
    rows=np.arange(len(y)); a=y[rows,current.argmax(1)]; b=y[rows,baseline.argmax(1)]; delta=a-b; rng=np.random.default_rng(7506)
    boot=np.array([rng.choice(delta,size=len(delta),replace=True).mean()*100 for _ in range(5000)])
    wins=int((delta>0).sum()); losses=int((delta<0).sum())
    result=dict(split='validation',post_freeze_only=True,current_correct=int(a.sum()),baseline_correct=int(b.sum()),repairs=wins,regressions=losses,paired_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),mcnemar_exact_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,all_validation=reliability(current,y,np.ones(len(y),dtype=bool)),hard_validation=reliability(current,y,hard))
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('all_validation','hard_validation')},indent=2))


if __name__=='__main__': main()

"""Verify frozen assets, input invariants and unchanged official tools."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='4'
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from common import load_inputs
from selector import predict,score_bank


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,default=Path('checkpoint')); args=p.parse_args()
    if args.checkpoint.resolve()==Path('checkpoint').resolve():
        manifest=json.loads(Path('evidence/asset_manifest.json').read_text())
        assert set(manifest)=={f.relative_to(args.checkpoint).as_posix() for f in args.checkpoint.rglob('*') if f.is_file()}
        for name,record in manifest.items(): assert hashlib.sha256((args.checkpoint/name).read_bytes()).hexdigest()==record['sha256']
    assert sum(f.stat().st_size for f in args.checkpoint.rglob('*') if f.is_file())<16*2**20
    official=json.loads(Path('evidence/official_tools.json').read_text())['files']
    for name,digest in official.items(): assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    bank,arrays=load_inputs('validation'); bank=deepcopy(bank[:12]); arrays={k:v[:12].copy() for k,v in arrays.items()}
    expected=predict(bank,arrays,args.checkpoint)
    assert len(expected)==12 and all(type(i) is int and 0<=i<4 for i in expected)
    assert predict([],{'embeddings':np.zeros((0,4,1024))},args.checkpoint)==[]
    batched=sum([predict(bank[i:i+3],{k:v[i:i+3] for k,v in arrays.items()},args.checkpoint) for i in range(0,12,3)],[])
    assert batched==expected
    original=score_bank(bank,arrays,args.checkpoint)
    for i,row in enumerate(bank): row['id']=f'unseen-submission-id-{i}'
    assert expected==predict(bank,arrays,args.checkpoint)
    perm=np.array([2,0,3,1]); changed=deepcopy(bank)
    for row in changed:
        row['candidates']=[row['candidates'][j] for j in perm]
        for j,c in enumerate(row['candidates']): c['index']=j
    actual=score_bank(changed,{k:v[:,perm] for k,v in arrays.items()},args.checkpoint)
    np.testing.assert_allclose(original[:,perm],actual,rtol=1e-5,atol=1e-6)
    print('PASS: asset hashes/size, official tools, output contract, empty/new IDs, batching and candidate permutation')


if __name__=='__main__': main()

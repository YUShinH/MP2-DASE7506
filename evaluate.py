"""Measure fresh official prediction + independent scoring, including startup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def cpu_model():
    if sys.platform=='win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    info=Path('/proc/cpuinfo')
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith('model name'): return line.split(':',1)[1].strip()
    return platform.processor()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--checkpoint',type=Path,default=Path('checkpoint'))
    parser.add_argument('--method',type=Path,default=Path('selector.py'))
    parser.add_argument('--no-checkpoint',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[name]='4'
    predictions=args.output_dir/f'{args.split}-predictions.json'
    score=args.output_dir/f'{args.split}-score.json'
    checkpoint_args=[] if args.no_checkpoint else ['--checkpoint',str(args.checkpoint)]
    commands=[
        [sys.executable,'predict.py','--method',str(args.method),*checkpoint_args,'--split',args.split,'--output',str(predictions)],
        [sys.executable,'score_predictions.py','--split',args.split,'--predictions',str(predictions),'--output',str(score)],
    ]
    elapsed=[]
    for cmd in commands:
        tick=time.perf_counter(); subprocess.run(cmd,check=True,env=env); elapsed.append(time.perf_counter()-tick)
    assets={} if args.no_checkpoint else {f.relative_to(args.checkpoint).as_posix():dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in args.checkpoint.rglob('*') if f.is_file()}
    result=dict(split=args.split,prediction_process_seconds=elapsed[0],scoring_process_seconds=elapsed[1],total_seconds=sum(elapsed),threads=4,python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),cpu_model=cpu_model(),asset_bytes=sum(a['bytes'] for a in assets.values()),assets=assets,commands=commands)
    (args.output_dir/f'{args.split}-resources.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

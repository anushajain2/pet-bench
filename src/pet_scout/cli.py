from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description='PET Scout reproducible analysis')
    parser.add_argument('stage',choices=['prepare','embed','evaluate','report','all'])
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args(); root=args.root.resolve()
    config=json.loads((root/'config.json').read_text())
    if args.stage in ['prepare','all']:
        from .data import prepare
        prepare(root,config)
    if args.stage in ['embed','all']:
        from .features import embed
        table=pd.read_csv(root/'data/processed/candidates.csv')
        embed(root,table,config['model'],args.batch_size,args.device)
    if args.stage in ['evaluate','all']:
        from .evaluate import evaluate
        evaluate(root,config)
    if args.stage in ['report','all']:
        from .report import build_report
        build_report(root)

if __name__=='__main__': main()

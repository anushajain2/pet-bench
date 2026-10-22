"""Run PETBench in isolated model processes, with no paid services."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='PETBench condition-specific model benchmark')
    parser.add_argument('stage', choices=['prepare', 'embed', 'evaluate', 'report', 'all'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--model', help='One configured model key; omit to run all sequentially.')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / 'benchmark_config.json').read_text())
    if args.stage in ['prepare', 'all']:
        from .bench_data import prepare_benchmark
        prepare_benchmark(root, config)
    if args.stage in ['embed', 'all']:
        if args.model:
            if args.model not in [m['key'] for m in config['models']]: parser.error('Unknown model key')
            from .bench_features import embed_model
            embed_model(root, config, args.model, args.device)
        else:
            for spec in config['models']:
                subprocess.run([sys.executable, '-m', 'pet_scout.bench_cli', 'embed', '--root', str(root),
                                '--model', spec['key'], '--device', args.device], check=True)
    if args.stage in ['evaluate', 'all']:
        from .bench_eval import evaluate_benchmark
        evaluate_benchmark(root, config)
    if args.stage in ['report', 'all']:
        from .bench_report import build_benchmark_report
        build_benchmark_report(root, config)


if __name__ == '__main__':
    main()

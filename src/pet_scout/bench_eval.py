"""Matched-budget evaluation of representations, not an unqualified model leaderboard."""
import hashlib
import json
import time
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from .data import digest, write_json
from .features import composition
from .bench_features import check_embedding_cache
from .evaluate import fit_ridge, selection_metrics, validate_split

BASELINES = ['Nearest sequence', 'Composition ridge']


def validate_data_inputs(root, config):
    base = root / 'benchmark'
    manifest = json.loads((base / 'data_manifest.json').read_text())
    files = {'config_sha256': root / 'benchmark_config.json', 'catalog_sha256': base / 'proteins.csv',
             'split_sha256': base / 'splits.csv', 'identity_sha256': base / 'identity.npy',
             'source_sha256': root / 'data/raw/source.xlsx'}
    for key, path in files.items():
        if digest(path) != manifest[key]:
            raise ValueError(f'Prepared data changed: {key}. Run prepare again.')
    for task in config['tasks']:
        if digest(base / 'tasks' / task['id'] / 'candidates.csv') != manifest['task_sha256'][task['id']]:
            raise ValueError('Task data checksum mismatch.')
    return manifest


def summarize_metrics(metrics):
    values = ['selected_mean', 'random_expected_mean', 'oracle_mean', 'lift_over_random',
              'opportunity_captured', 'spearman', 'mae_log1p']
    keys = ['task', 'regime', 'training_size', 'model', 'budget']
    seed_scores = metrics.groupby(keys + ['seed'], sort=False)[values].mean().reset_index()
    summary = seed_scores.groupby(keys, sort=False).agg(
        selected_mean=('selected_mean', 'mean'), random_mean=('random_expected_mean', 'mean'),
        oracle_mean=('oracle_mean', 'mean'), lift=('lift_over_random', 'mean'),
        min_seed_lift=('lift_over_random', 'min'), max_seed_lift=('lift_over_random', 'max'),
        opportunity_captured=('opportunity_captured', 'mean'),
        spearman=('spearman', 'mean'), mae_log1p=('mae_log1p', 'mean')).reset_index()
    nearest = seed_scores[seed_scores.model.eq('Nearest sequence')].drop(columns='model')
    paired = seed_scores.merge(nearest[keys[:3]+['budget', 'seed', 'selected_mean']],
                              on=keys[:3]+['budget', 'seed'], suffixes=('', '_nearest'), validate='many_to_one')
    paired['difference_from_nearest'] = paired.selected_mean - paired.selected_mean_nearest
    diffs = paired.groupby(keys, sort=False).difference_from_nearest.agg(['mean', 'min', 'max']).reset_index()
    summary = summary.merge(diffs.rename(columns={'mean': 'delta_nearest', 'min': 'min_seed_delta', 'max': 'max_seed_delta'}), on=keys)
    coverage = metrics.groupby(keys, sort=False).spearman.agg(spearman_valid_folds='count', total_folds='size').reset_index()
    summary = summary.merge(coverage, on=keys)
    return seed_scores, summary, paired


def evaluate_benchmark(root, config):
    start = time.monotonic()
    base = root / 'benchmark'
    data_manifest = validate_data_inputs(root, config)
    catalog = pd.read_csv(base / 'proteins.csv')
    identity = np.load(base / 'identity.npy')
    features = {'Composition ridge': composition(catalog)}
    embedding_metadata = {}
    # Require the entire prespecified roster. No silent model omissions.
    for spec in config['models']:
        revision = json.loads((base / 'revisions' / f'{spec["key"]}.json').read_text())['revision']
        matrix, meta = check_embedding_cache(base / 'embeddings' / f'{spec["key"]}.npz', catalog, spec, revision)
        features[spec['label']] = matrix
        embedding_metadata[spec['label']] = meta
    index = {pid: i for i, pid in enumerate(catalog.sample_id)}
    splits = pd.read_csv(base / 'splits.csv', dtype={'training_size': str})
    predictions, scores, timings = [], [], []
    for task in config['tasks']:
        table = pd.read_csv(base / 'tasks' / task['id'] / 'candidates.csv')
        ids = table.sample_id.to_numpy()
        task_index = {pid: i for i, pid in enumerate(ids)}
        global_idx = np.array([index[pid] for pid in ids])
        y = table.activity.to_numpy()
        groups = table.sequence_group.to_numpy()
        similarities = identity[np.ix_(global_idx, global_idx)]
        matrices = {name: values[global_idx] for name, values in features.items()}
        for number, split in enumerate(splits[splits.task.eq(task['id'])].itertuples(index=False)):
            train = np.array([task_index[pid] for pid in split.train_ids.split(';')])
            test = np.array([task_index[pid] for pid in split.test_ids.split(';')])
            validate_split(train, test, groups, similarities, config['identity_threshold'], split.regime == 'grouped')
            # For tied sequence identities use candidate ID, never activity or subset ordering.
            sorted_train = train[np.argsort(ids[train])]
            cross = similarities[np.ix_(test, sorted_train)]
            closest = sorted_train[cross.argmax(axis=1)]
            nearest_identity = cross.max(axis=1)
            context = {'task': task['id'], 'regime': split.regime, 'seed': split.seed,
                       'fold': split.fold, 'training_size': split.training_size,
                       'n_train': len(train), 'n_test': len(test)}
            for name in ['Nearest sequence', *matrices]:
                t = time.monotonic()
                with threadpool_limits(limits=4):
                    if name == 'Nearest sequence':
                        prediction, alpha = y[closest], None
                    else:
                        prediction, alpha = fit_ridge(matrices[name], y, train, test, groups,
                            split.regime, split.seed + split.fold, config['ridge_alphas'], config['inner_folds'])
                elapsed = time.monotonic() - t
                if not np.isfinite(prediction).all(): raise ValueError('Nonfinite predictions.')
                timings.append({**context, 'model': name, 'fit_predict_seconds': elapsed,
                    'ridge_alpha': alpha, 'fixed_alpha_fallback': name != 'Nearest sequence' and split.regime == 'grouped' and len(np.unique(groups[train])) < 2})
                rho = float(spearmanr(y[test], prediction).statistic) if np.ptp(y[test]) > 0 and np.ptp(prediction) > 0 else None
                for k in config['selection_budgets']:
                    scores.append({**context, 'model': name, 'budget': k, 'spearman': rho,
                        'mae_log1p': float(np.mean(abs(np.log1p(y[test])-np.log1p(prediction)))),
                        **selection_metrics(y[test], prediction, ids[test], k)})
                for j, ix in enumerate(test):
                    predictions.append({**context, 'model': name, 'sample_id': ids[ix],
                        'observed': float(y[ix]), 'predicted': float(prediction[j]),
                        'nearest_train_id': ids[closest[j]], 'nearest_identity': float(nearest_identity[j]),
                        'sequence_group': int(groups[ix]), 'ridge_alpha': alpha})
            if number % 25 == 0:
                print(f'{task["id"]}: evaluated split/subset {number+1}/{sum(splits.task.eq(task["id"]))}', flush=True)
    out = base / 'results'
    out.mkdir(exist_ok=True)
    metrics = pd.DataFrame(scores)
    seed_scores, summary, paired = summarize_metrics(metrics)
    outputs = {'predictions': pd.DataFrame(predictions), 'metrics': metrics,
               'seed_metrics': seed_scores, 'summary': summary, 'paired_comparison': paired,
               'fit_timings': pd.DataFrame(timings)}
    for name, df in outputs.items(): df.to_csv(out / f'{name}.csv', index=False)
    compute = pd.DataFrame([{'model': name, 'inference_seconds': m['inference_seconds'],
        'loading_seconds': m['loading_seconds'], 'process_peak_rss_gib': m['process_peak_rss_gib'],
        'embedding_dimensions': m['shape'][1], 'parameter_count': m['parameter_count'],
        'device': m['device'], 'dtype': m['dtype'], 'batch_size': m['batch_size'],
        'checkpoint_disk_gib': m['checkpoint_disk_bytes']/1024**3} for name, m in embedding_metadata.items()])
    compute.to_csv(out / 'compute.csv', index=False)
    code_paths = [root / 'src/pet_scout' / name for name in ['bench_data.py', 'bench_features.py', 'bench_eval.py', 'evaluate.py', 'features.py', 'data.py']]
    write_json(out / 'manifest.json', {'config': config, 'data_manifest': data_manifest,
        'models': ['Nearest sequence', *matrices], 'embedding_metadata': embedding_metadata,
        'n_split_subsets': len(splits), 'n_outer_model_fits': len(timings),
        'n_predictions': len(predictions), 'evaluation_seconds': round(time.monotonic()-start, 3),
        'code_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in code_paths)).hexdigest(),
        'result_sha256': {p.name: digest(p) for p in out.glob('*.csv')},
        'versions': {p: version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','transformers','sentencepiece']},
        'paid_compute_usd': 0,
        'limitations': [
            'Exploratory extension of an already inspected published dataset. No blind novelty or mutation-effect claim.',
            'Task populations differ because some proteins were not measured at every condition.',
            'Frozen sequence embeddings plus one shared ridge protocol; rankings apply to this pipeline, not every use of each foundation model.',
            'A large connected group causes uneven held-out pools; matched random controls fold sizes only.',
            'Each model receives exactly the same examples at each budget. Other-condition labels are not used.',
            'Mean results weight each test pool equally. Seed ranges describe partition sensitivity, not independent replications or confidence intervals.',
            'Protein pretraining exposure is unknown; label isolation does not establish pretraining decontamination.',
            'Single-study, ML-screened, expressed proteins; experimental uncertainty is not fully propagated.',
            'Compute is one local float32 CPU run per model with different batch sizes, not a standardized hardware throughput benchmark.'
        ]})
    print(f'Completed {len(timings)} fits and {len(predictions)} held-out predictions in {time.monotonic()-start:.1f}s', flush=True)

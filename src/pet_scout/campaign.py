"""Finite-pool experiment selection; acquisition never receives unrevealed labels."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import digest, write_json
from .bench_eval import validate_data_inputs
from .bench_features import check_embedding_cache


def validate_config(config, n):
    a, b, m = (config[k] for k in ['initial_size', 'batch_size', 'max_budget'])
    if not (0 < a < m <= n and b > 0 and (m-a) % b == 0):
        raise ValueError('Invalid experimental budget or batch size.')
    if config['primary_budget'] not in range(a, m+1, b):
        raise ValueError('Primary budget must be a measured checkpoint.')
    if config['bootstrap_members'] < 2 or config['ridge_alpha'] <= 0:
        raise ValueError('Require >=2 bootstrap members and positive regularization.')
    if not 0 < config['hit_quantile'] < 1 or not 0 <= config['diversity_weight'] <= 1:
        raise ValueError('Invalid quantile or diversity weight.')
    if config['n_seeds'] < 1 or not 1 <= config['target_hits'] <= m:
        raise ValueError('Invalid replicate count or target.')
    if config['strategies'] != ['Random', 'Nearest sequence', 'AI greedy', 'AI explore']:
        raise ValueError('The complete four-policy roster is required.')


def bootstrap_predict(x, observed, observed_y, pool, seed, config):
    """Train-only scaling; bootstrap ridge with intercept on log1p activity.

    Both AI policies use this same ensemble. Dispersion is a heuristic, not a
    calibrated posterior. The dual solve is algebraically equivalent to Ridge.
    """
    scaler = StandardScaler().fit(x[observed])
    train = scaler.transform(x[observed]).astype(float)
    test = scaler.transform(x[pool]).astype(float)
    y = np.log1p(observed_y)
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(config['bootstrap_members']):
        indices = rng.integers(0, len(observed), len(observed))
        xb, yb = train[indices], y[indices]
        center = xb.mean(axis=0)
        centered = xb-center
        weights = np.linalg.solve(centered @ centered.T + config['ridge_alpha']*np.eye(len(xb)), yb-yb.mean())
        predictions.append((test-center) @ centered.T @ weights + yb.mean())
    predictions = np.asarray(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)


def percentile_rank(values):
    if len(values) == 1:
        return np.ones(1)
    return (rankdata(values, method='average')-1)/(len(values)-1)


def choose_batch(strategy, ids, x, identity, observed, observed_y, seed, config):
    """Only observed labels cross this interface; all candidates' sequences may be known.

    Select the entire batch before any of its outcomes are revealed.
    """
    observed = np.asarray(observed, dtype=int)
    observed_y = np.asarray(observed_y, dtype=float)
    if len(observed) != len(observed_y) or len(np.unique(observed)) != len(observed):
        raise ValueError('Observed indices must be unique and match observed labels.')
    if not np.isfinite(observed_y).all() or (observed_y < 0).any():
        raise ValueError('Observed activity must be finite and nonnegative.')
    pool = np.setdiff1d(np.arange(len(ids)), observed)
    k = config['batch_size']
    if len(pool) < k:
        raise ValueError('Insufficient untested candidates.')
    mean = np.full(len(pool), np.nan)
    std = np.full(len(pool), np.nan)
    if strategy == 'Random':
        score = np.random.default_rng(seed).random(len(pool))
    elif strategy == 'Nearest sequence':
        order = np.argsort(ids[observed])
        nearest = identity[np.ix_(pool, observed[order])].argmax(axis=1)
        score = observed_y[order][nearest]
    elif strategy in ['AI greedy', 'AI explore']:
        mean, std = bootstrap_predict(x, observed, observed_y, pool, seed, config)
        score = mean if strategy == 'AI greedy' else mean + config['ucb_beta']*std
    else:
        raise ValueError(strategy)
    chosen, details = [], []
    unit = x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    for _ in range(k):
        remaining = np.flatnonzero(~np.isin(pool, chosen))
        acquisition = score[remaining].copy()
        distance = np.full(len(remaining), np.nan)
        if strategy == 'AI explore':
            reference = np.r_[observed, chosen].astype(int)
            distance = np.maximum(0, 1-(unit[pool[remaining]] @ unit[reference].T).max(axis=1))
            w = config['diversity_weight']
            acquisition = (1-w)*percentile_rank(acquisition) + w*percentile_rank(distance)
        selected = np.lexsort((ids[pool[remaining]], -acquisition))[0]
        j = remaining[selected]
        chosen.append(int(pool[j]))
        details.append({'sample_id': str(ids[pool[j]]), 'predicted_log1p': mean[j],
                        'bootstrap_std_log1p': std[j], 'acquisition_score': acquisition[selected],
                        'diversity_distance': distance[selected]})
    return np.array(chosen), details


def simulate(ids, x, y, identity, seed, config):
    validate_config(config, len(ids))
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Finite, nonnegative measured activities required.')
    # The full-label quantile is exclusively an evaluator-side retrospective target.
    threshold = float(np.quantile(y, config['hit_quantile']))
    if threshold <= 0:
        raise ValueError('Top-quantile threshold must be positive.')
    hit = y >= threshold
    initial = np.random.default_rng(seed).choice(len(ids), config['initial_size'], replace=False)
    checkpoints, selections = [], []
    for strategy in config['strategies']:
        observed = initial.copy()
        for i in observed:
            selections.append({'strategy': strategy, 'seed': seed, 'batch': 0,
                'revealed_at': len(initial), 'sample_id': ids[i], 'activity': y[i],
                'hit': bool(hit[i]), 'predicted_log1p': np.nan, 'bootstrap_std_log1p': np.nan,
                'acquisition_score': np.nan, 'diversity_distance': np.nan})
        batch = 0
        while True:
            checkpoints.append({'strategy': strategy, 'seed': seed, 'tested': len(observed),
                'hits': int(hit[observed].sum()), 'new_hits': int(hit[observed[len(initial):]].sum()),
                'best_activity': float(y[observed].max()), 'threshold': threshold,
                'n_available_hits': int(hit.sum()), 'target_reached': int(hit[observed].sum()) >= config['target_hits']})
            if len(observed) == config['max_budget']:
                break
            batch += 1
            # Shared random streams by starting set and round; no hidden outcomes.
            chosen, details = choose_batch(strategy, ids, x, identity, observed,
                                          y[observed], seed*100+batch, config)
            observed = np.r_[observed, chosen]
            for i, detail in zip(chosen, details):
                selections.append({**detail, 'strategy': strategy, 'seed': seed, 'batch': batch,
                    'revealed_at': len(observed), 'activity': y[i], 'hit': bool(hit[i])})
    return checkpoints, selections


def summarize(checkpoints, config):
    keys = ['task', 'strategy', 'tested']
    summary = checkpoints.groupby(keys, sort=False).agg(
        mean_hits=('hits', 'mean'), p10_hits=('hits', lambda s: s.quantile(.1)),
        p90_hits=('hits', lambda s: s.quantile(.9)), mean_new_hits=('new_hits', 'mean'),
        mean_best_activity=('best_activity', 'mean'), target_success_rate=('target_reached', 'mean')).reset_index()
    paired = checkpoints.copy()
    for label, suffix in [('Random', 'random'), ('Nearest sequence', 'nearest'), ('AI greedy', 'greedy')]:
        baseline = checkpoints[checkpoints.strategy.eq(label)][['task','seed','tested','hits']].rename(columns={'hits':f'hits_{suffix}'})
        paired = paired.merge(baseline, on=['task','seed','tested'], validate='many_to_one')
        paired[f'delta_{suffix}'] = paired.hits-paired[f'hits_{suffix}']
    pairs = paired.groupby(keys, sort=False).agg(
        delta_random=('delta_random','mean'), delta_nearest=('delta_nearest','mean'),
        delta_greedy=('delta_greedy','mean'), win_rate_random=('delta_random',lambda s:(s>0).mean()),
        tie_rate_random=('delta_random',lambda s:(s==0).mean()),
        p10_delta_random=('delta_random',lambda s:s.quantile(.1)),
        p90_delta_random=('delta_random',lambda s:s.quantile(.9))).reset_index()
    times = []
    for (task, strategy, seed), frame in checkpoints.groupby(['task','strategy','seed']):
        reached = frame[frame.target_reached].tested
        times.append({'task':task,'strategy':strategy,'seed':seed,
            'reached_by_max': len(reached)>0, 'first_reached_at': int(reached.min()) if len(reached) else np.nan,
            'capped_assays': int(reached.min()) if len(reached) else config['max_budget']})
    return summary.merge(pairs, on=keys), paired, pd.DataFrame(times)


def run(root):
    started = time.monotonic()
    config = json.loads((root/'campaign_config.json').read_text())
    bench_config = json.loads((root/'benchmark_config.json').read_text())
    data_manifest = validate_data_inputs(root, bench_config)
    base = root/'benchmark'
    catalog = pd.read_csv(base/'proteins.csv')
    spec = next(m for m in bench_config['models'] if m['key'] == config['model_key'])
    revision = json.loads((base/'revisions'/f'{spec["key"]}.json').read_text())['revision']
    features, metadata = check_embedding_cache(base/'embeddings'/f'{spec["key"]}.npz', catalog, spec, revision)
    identity = np.load(base/'identity.npy')
    lookup = {pid:i for i,pid in enumerate(catalog.sample_id)}
    checkpoints, selections = [], []
    with threadpool_limits(limits=1):
        for task in bench_config['tasks']:
            table = pd.read_csv(base/'tasks'/task['id']/'candidates.csv').sort_values('sample_id')
            ids = table.sample_id.to_numpy()
            indices = np.array([lookup[pid] for pid in ids])
            x, y = features[indices], table.activity.to_numpy()
            for count, seed in enumerate(range(config['seed_start'], config['seed_start']+config['n_seeds'])):
                points, choices = simulate(ids, x, y, identity[np.ix_(indices,indices)], seed, config)
                checkpoints.extend({'task':task['id'], **p} for p in points)
                selections.extend({'task':task['id'], **p} for p in choices)
                if count % 10 == 0:
                    print(f'{task["id"]}: start {count+1}/{config["n_seeds"]}; {time.monotonic()-started:.1f}s', flush=True)
    out = root/'campaign/results'
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = pd.DataFrame(checkpoints)
    summary, paired, target_times = summarize(checkpoints, config)
    outputs = {'checkpoints':checkpoints, 'selections':pd.DataFrame(selections),
               'summary':summary,'paired':paired,'target_times':target_times}
    for name, table in outputs.items():
        table.to_csv(out/f'{name}.csv', index=False)
    write_json(out/'manifest.json', {'config':config, 'task_labels':{t['id']:t['label'] for t in bench_config['tasks']},
        'data_manifest':data_manifest, 'embedding_metadata':metadata,
        'config_sha256':digest(root/'campaign_config.json'),
        'code_sha256':digest(Path(__file__)), 'result_sha256':{f'{name}.csv':digest(out/f'{name}.csv') for name in outputs},
        'simulation_seconds':round(time.monotonic()-started,3), 'paid_compute_usd':0,
        'n_campaigns':len(bench_config['tasks'])*config['n_seeds']*len(config['strategies']),
        'scope':config['scope']})
    from .campaign_report import report
    report(root)
    print(f'Completed {len(outputs["selections"])} selections in {time.monotonic()-started:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == '__main__':
    main()

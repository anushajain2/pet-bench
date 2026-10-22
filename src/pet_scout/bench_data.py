"""A canonical sequence catalog and separately measured, condition-specific tasks."""
from pathlib import Path
import shutil
import hashlib
import numpy as np
import pandas as pd
from .data import download, prepare, write_json, digest
from .features import pairwise_identity, sequence_groups
from .evaluate import make_folds, validate_split


def nested_training_subset(train, size, seed, fold):
    """Independent of labels; the first 16 samples are contained in the first 32."""
    order = np.random.default_rng(seed * 1009 + fold).permutation(train)
    if size == 'all':
        return order
    if int(size) > len(order):
        raise ValueError('Requested training size exceeds the training pool.')
    return order[:int(size)]


def prepare_benchmark(root, config):
    source = download(root)
    base = root / 'benchmark'
    base.mkdir(exist_ok=True)
    frames = {}
    for task in config['tasks']:
        taskroot = base / 'tasks' / task['id']
        raw = taskroot / 'data/raw'
        raw.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, raw / 'source.xlsx')
        frames[task['id']] = prepare(taskroot, {'condition': task})
    union = pd.concat(frames.values(), ignore_index=True)
    # Canonical IDs are the same across tasks even when source aliases differ.
    catalog = union.sort_values('sample_id').drop_duplicates('sequence').copy()
    catalog = catalog[['sample_id', 'sequence', 'sequence_sha256', 'length']].sort_values('sample_id').reset_index(drop=True)
    mapping = catalog.set_index('sequence').sample_id.to_dict()
    catalog.to_csv(base / 'proteins.csv', index=False)
    identity = pairwise_identity(catalog, base / 'identity.npy')
    groups = sequence_groups(identity, config['identity_threshold'])
    groupmap = dict(zip(catalog.sample_id, groups))
    pd.DataFrame({'sample_id': catalog.sample_id, 'sequence_group': groups}).to_csv(base / 'groups.csv', index=False)
    catalog_index = {pid: i for i, pid in enumerate(catalog.sample_id)}
    splits, counts = [], []
    for task in config['tasks']:
        frame = frames[task['id']].copy()
        frame['sample_id'] = frame.sequence.map(mapping)
        frame = frame.sort_values('sample_id').reset_index(drop=True)
        frame['sequence_group'] = frame.sample_id.map(groupmap)
        frame.to_csv(base / 'tasks' / task['id'] / 'candidates.csv', index=False)
        ids = frame.sample_id.to_numpy()
        g = frame.sequence_group.to_numpy()
        idx = np.array([catalog_index[pid] for pid in ids])
        task_identity = identity[np.ix_(idx, idx)]
        if len(np.unique(g)) < config['n_folds']:
            raise ValueError('Insufficient global sequence groups for requested outer folds.')
        all_folds = [(regime, seed, make_folds(len(frame), g, regime, config['n_folds'], seed))
                     for regime in config['regimes'] for seed in config['seeds']]
        minimum_train = min(len(train) for _, _, fs in all_folds for train, _ in fs)
        sizes = [n for n in config['training_sizes'] if n == 'all' or n <= minimum_train]
        for regime, seed, fs in all_folds:
            seen = []
            for fold, (train, test) in enumerate(fs):
                validate_split(train, test, g, task_identity, config['identity_threshold'], regime == 'grouped')
                seen.extend(test)
                if len(test) < max(config['selection_budgets']):
                    raise ValueError('A held-out pool cannot support the fixed selection budgets.')
                for size in sizes:
                    sub = nested_training_subset(train, size, seed, fold)
                    validate_split(sub, test, g, task_identity, config['identity_threshold'], regime == 'grouped')
                    splits.append({'task': task['id'], 'regime': regime, 'seed': seed, 'fold': fold,
                        'training_size': str(size), 'n_train': len(sub), 'n_test': len(test),
                        'n_train_groups': len(np.unique(g[sub])),
                        'max_cross_identity': float(task_identity[np.ix_(test, sub)].max()),
                        'train_ids': ';'.join(ids[sub]), 'test_ids': ';'.join(ids[test])})
            if sorted(seen) != list(range(len(frame))):
                raise ValueError('Outer test pools do not partition task sequences exactly once.')
        counts.append({'task': task['id'], 'label': task['label'], 'n_proteins': len(frame),
            'n_assay_rows': int(frame.n_replicates.sum()), 'n_groups': len(np.unique(g)),
            'largest_group': int(frame.sequence_group.value_counts().max()),
            'minimum_training_pool': minimum_train, 'training_sizes': ';'.join(map(str, sizes)),
            'single_measurement_proteins': int(frame.n_replicates.eq(1).sum())})
    pd.DataFrame(splits).to_csv(base / 'splits.csv', index=False)
    pd.DataFrame(counts).to_csv(base / 'task_summary.csv', index=False)
    write_json(base / 'data_manifest.json', {'config_sha256': digest(root / 'benchmark_config.json'),
        'source_sha256': digest(source), 'catalog_sha256': digest(base / 'proteins.csv'),
        'split_sha256': digest(base / 'splits.csv'), 'identity_sha256': digest(base / 'identity.npy'),
        'task_sha256': {t['id']: digest(base / 'tasks' / t['id'] / 'candidates.csv') for t in config['tasks']},
        'n_unique_proteins': len(catalog), 'n_sequence_groups': int(len(np.unique(groups))),
        'grouping_scope': 'Connected components computed on the union of all three tasks, without assay labels.',
        'missingness': 'Only observed assays used per task. Tasks have different populations; no zero imputation.',
        'scope': config['scope']})
    print(pd.DataFrame(counts).to_string(index=False), flush=True)

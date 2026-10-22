"""Nested, target-isolated evaluation and within-fold experimental selection."""
from __future__ import annotations
import json
import platform
import time
import hashlib
from importlib.metadata import version
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import composition, pairwise_identity, sequence_groups, sequence_fingerprint
from .data import write_json, digest


def grouped_folds(groups, n_splits, seed):
    """Balance sample counts while holding connected components intact."""
    labels, counts=np.unique(groups,return_counts=True)
    n_splits=min(n_splits,len(labels))
    if n_splits<2: raise ValueError('At least two sequence groups are required.')
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(labels))
    order=order[np.argsort(-counts[order],kind='stable')]
    sizes=np.zeros(n_splits,int); allocation={}
    for i in order:
        choices=np.flatnonzero(sizes==sizes.min()); fold=int(rng.choice(choices))
        allocation[labels[i]]=fold; sizes[fold]+=counts[i]
    folds=np.array([allocation[g] for g in groups])
    return [(np.flatnonzero(folds!=f),np.flatnonzero(folds==f)) for f in range(n_splits)]


def make_folds(n, groups, regime, n_splits, seed):
    if regime=='grouped': return grouped_folds(groups,n_splits,seed)
    if regime=='matched':
        reference=grouped_folds(groups,n_splits,seed)
        shuffled=np.random.default_rng(seed+10000).permutation(n)
        result=[]; start=0
        for _, heldout in reference:
            test=shuffled[start:start+len(heldout)]; start+=len(heldout)
            train=np.setdiff1d(np.arange(n),test)
            result.append((train,test))
        return result
    if regime!='random': raise ValueError(regime)
    return list(KFold(n_splits=min(n_splits,n),shuffle=True,random_state=seed).split(np.arange(n)))


def validate_split(train, test, groups, identity, threshold, grouped):
    if len(set(train)&set(test)) or not len(train) or not len(test):
        raise ValueError('Empty or overlapping train/test split.')
    if grouped:
        if set(groups[train])&set(groups[test]): raise ValueError('Sequence-group leakage.')
        if np.max(identity[np.ix_(test,train)])>=threshold:
            raise ValueError('Cross-split pair exceeds identity threshold.')


def fit_ridge(x, y, train, test, groups, regime, seed, alphas, inner_folds=3):
    # All feature scaling, tuning and regression happen within outer training data.
    local_groups=groups[train]
    if regime=='grouped' and len(np.unique(local_groups))<2:
        estimator=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
    else:
        folds=make_folds(len(train),local_groups,'grouped' if regime=='grouped' else 'random',inner_folds,seed)
        estimator=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),
            {'ridge__alpha':alphas},scoring='neg_mean_absolute_error',cv=folds,
            n_jobs=1,error_score='raise')
    estimator.fit(x[train],np.log1p(y[train]))
    prediction=np.maximum(np.expm1(estimator.predict(x[test])),0)
    alpha=float(estimator.best_params_['ridge__alpha']) if hasattr(estimator,'best_params_') else 100.0
    return prediction,alpha


def select_indices(scores, ids, k):
    """Stable score ordering, with ID tie-break fixed independently of outcomes."""
    if not 1<=k<=len(scores): raise ValueError('Budget must fit candidate pool.')
    return np.lexsort((np.asarray(ids,dtype=str),-np.asarray(scores)))[:k]


def selection_metrics(y, prediction, ids, k):
    chosen=select_indices(prediction,ids,k)
    value=float(np.mean(y[chosen])); random=float(np.mean(y))
    oracle=float(np.mean(np.sort(y)[-k:])); gap=oracle-random
    return {'selected_mean':value,'random_expected_mean':random,'oracle_mean':oracle,
        'lift_over_random':value-random,
        'opportunity_captured':(value-random)/gap if gap>1e-10 else None,
        'selected_ids':';'.join(np.asarray(ids)[chosen])}


def evaluate(root: Path, config: dict):
    start_time=time.monotonic()
    out=root/'reports';out.mkdir(exist_ok=True)
    table=pd.read_csv(root/'data/processed/candidates.csv')
    with np.load(root/'data/processed/embeddings.npz') as f:
        embeddings=f['embeddings']; embedding_ids=f['ids']
    emeta=json.loads((root/'data/processed/embeddings.json').read_text())
    if not np.array_equal(embedding_ids,table.sample_id.to_numpy()) or emeta['sequence_sha256']!=sequence_fingerprint(table):
        raise ValueError('Embedding identity mismatch; regenerate embeddings.')
    if emeta['model_id']!=config['model']: raise ValueError('Wrong embedding model.')
    if not np.isfinite(embeddings).all(): raise ValueError('Nonfinite embeddings.')
    identity=pairwise_identity(table,root/'data/processed/identity.npy')
    groups=sequence_groups(identity,config['identity_threshold']); n=len(table)
    labels,counts=np.unique(groups,return_counts=True)
    pd.DataFrame({'sample_id':table.sample_id,'sequence_group':groups}).to_csv(out/'sequence_groups.csv',index=False)
    if len(labels)<config['n_folds']:
        raise ValueError(f'Only {len(labels)} independent groups; reduce prespecified fold count and document change.')
    xcomp=composition(table); y=table.activity.to_numpy(); ids=table.sample_id.to_numpy()
    predictions=[]; metrics=[]; split_audit=[]; random_batches=[]
    for regime in ['random','matched','grouped']:
      for seed in config['seeds']:
        folds=make_folds(n,groups,regime,config['n_folds'],seed)
        seen=[]
        for fold,(train,test) in enumerate(folds):
            validate_split(train,test,groups,identity,config['identity_threshold'],regime=='grouped');seen.extend(test)
            if min(config['budgets'])>len(test): raise ValueError('Test pool smaller than every requested budget.')
            similar=identity[np.ix_(test,train)]
            closest=train[similar.argmax(axis=1)]; nearest=similar.max(axis=1)
            audit={'regime':regime,'seed':seed,'fold':fold,'n_train':len(train),'n_test':len(test),
                'train_groups':len(np.unique(groups[train])),'test_groups':len(np.unique(groups[test])),
                'shared_groups':len(set(groups[train])&set(groups[test])),
                'max_cross_identity':float(nearest.max()),'median_nearest_identity':float(np.median(nearest)),
                'train_ids':';'.join(ids[train]),'test_ids':';'.join(ids[test])}
            split_audit.append(audit)
            model_predictions={'Nearest sequence':(y[closest],None)}
            for name,x in [('Composition ridge',xcomp),('ESM-2 ridge',embeddings)]:
                model_predictions[name]=fit_ridge(x,y,train,test,groups,regime,seed+fold,config['ridge_alphas'])
            for name,(pred,alpha) in model_predictions.items():
                rho=float(spearmanr(y[test],pred).statistic) if np.ptp(y[test])>0 and np.ptp(pred)>0 else None
                for j,index in enumerate(test):
                    predictions.append({'regime':regime,'seed':seed,'fold':fold,'model':name,
                        'sample_id':ids[index],'observed':float(y[index]),'predicted':float(pred[j]),
                        'nearest_identity':float(nearest[j]),'nearest_train_id':ids[closest[j]],
                        'sequence_group':int(groups[index]),'ridge_alpha':alpha})
                for k in config['budgets']:
                    if k>len(test): continue
                    metrics.append({'regime':regime,'seed':seed,'fold':fold,'model':name,'budget':k,
                        'n_test':len(test),'spearman':rho,'mae_log1p':float(np.mean(np.abs(np.log1p(y[test])-np.log1p(pred)))),
                        'ridge_alpha':alpha,**selection_metrics(y[test],pred,ids[test],k)})
            rng=np.random.default_rng(seed*100+fold)
            for k in config['budgets']:
                if k>len(test): continue
                draws=np.array([y[test][rng.choice(len(test),size=k,replace=False)].mean() for _ in range(config['random_draws'])])
                random_batches.append({'regime':regime,'seed':seed,'fold':fold,'budget':k,
                    'expected_mean':float(y[test].mean()),'p05':float(np.quantile(draws,.05)),
                    'p95':float(np.quantile(draws,.95)), 'n_draws':len(draws)})
            print(f'{regime} seed={seed} fold={fold+1}: train={len(train)}, test={len(test)}, max identity={nearest.max():.3f}',flush=True)
        if sorted(seen)!=list(range(n)): raise ValueError('Outer folds do not partition the data exactly once.')
    p=pd.DataFrame(predictions);m=pd.DataFrame(metrics)
    p.to_csv(out/'predictions.csv',index=False);m.to_csv(out/'selection_metrics.csv',index=False)
    pd.DataFrame(split_audit).to_csv(out/'split_audit.csv',index=False)
    pd.DataFrame(random_batches).to_csv(out/'random_reference.csv',index=False)
    # Equal-weight outer test batches; dispersion is across seed-level means.
    numeric=['selected_mean','random_expected_mean','oracle_mean','lift_over_random','opportunity_captured','spearman','mae_log1p']
    seed_scores=m.groupby(['regime','model','budget','seed'])[numeric].mean().reset_index()
    seed_scores.to_csv(out/'seed_metrics.csv',index=False)
    summary=seed_scores.groupby(['regime','model','budget']).agg(
        mean_selected_activity=('selected_mean','mean'), mean_random_activity=('random_expected_mean','mean'),
        mean_oracle_activity=('oracle_mean','mean'),mean_lift=('lift_over_random','mean'),
        mean_opportunity_captured=('opportunity_captured','mean'),
        split_min_lift=('lift_over_random','min'),split_max_lift=('lift_over_random','max'),
        mean_spearman=('spearman','mean'),mean_mae_log1p=('mae_log1p','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    # Paired comparison on exactly the same held-out pools. Seed ranges are descriptive.
    paired=seed_scores.pivot(index=['regime','budget','seed'],columns='model',values='selected_mean').reset_index()
    paired['esm_minus_nearest']=paired['ESM-2 ridge']-paired['Nearest sequence']
    paired.to_csv(out/'paired_comparison.csv',index=False)
    manifest={'evaluation_seconds':round(time.monotonic()-start_time,2),
        'package_versions':{p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','biopython','torch','transformers']},
        'source_code_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sorted((root/'src/pet_scout').glob('*.py')))).hexdigest(),
        'diagnostic_addition':'Size-matched random control was added after the initial random/grouped run revealed an 86-protein connected component. Primary model/config were not retuned.',
        'config':config,'n_candidates':n,'n_groups':len(labels),'largest_group':int(counts.max()),
        'group_sizes':sorted([int(c) for c in counts],reverse=True),'python':platform.python_version(),
        'embedding_metadata':emeta,'dataset_sha256':digest(root/'data/processed/candidates.csv'),
        'config_sha256':digest(root/'config.json'),'identity_sha256':digest(root/'data/processed/identity.npy'),
        'predictions_sha256':digest(out/'predictions.csv'),'n_outer_fits':len(split_audit),
        'uncertainty':'Min/max across seed-level mean metrics describes split sensitivity, not independent trials or a confidence interval.',
        'limitations':['One selected assay condition and non-random, ML-screened candidate population.',
         'Sequence grouping uses full construct global identity; distant conserved domains may cross groups.',
         'A large sequence component makes grouped folds imbalanced; matched random is a diagnostic with the same train/test sizes, not the same candidate pools.',
         'Unknown overlap with protein-model pretraining; frozen embeddings do not eliminate pretraining exposure.',
         'Missing activity assays and failed expression are excluded, limiting applicability to tested enzymes.',
         'Outcome is 48-hour product release under this assay, not industrial recycling yield or mutation effect.',
         'The original authors already studied ML prediction; this is an evaluation extension, not a novelty claim.']}
    write_json(out/'run_manifest.json',manifest)
    print(summary.to_string(index=False),flush=True)

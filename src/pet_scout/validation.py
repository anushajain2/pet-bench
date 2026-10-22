"""Fixed-policy robustness and ablation study; preserves original campaign outputs."""
import argparse
import json
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from .campaign import bootstrap_predict, choose_batch, percentile_rank, validate_config
from .bench_eval import validate_data_inputs
from .bench_features import check_embedding_cache
from .data import digest, write_json

STRATEGIES = ['Random','Nearest sequence','AI greedy','AI uncertainty','AI diversity','AI explore']


def anchors(groups, config):
    labels, counts = np.unique(groups, return_counts=True)
    return [int(g) for g,n in zip(labels,counts) if n >= config['initial_size'] and
            len(groups)-n >= config['max_budget']-config['initial_size']]


def starting_pool(groups, seed, config, anchor=None):
    """No outcomes used. Exclude every unmeasured member of the starting group."""
    rng = np.random.default_rng(seed)
    if anchor is None:
        initial = rng.choice(len(groups), config['initial_size'], replace=False)
        return np.arange(len(groups)), initial
    if anchor not in anchors(groups, config):
        raise ValueError('Ineligible starting group.')
    initial_global = rng.choice(np.flatnonzero(groups == anchor), config['initial_size'], replace=False)
    allowed = np.sort(np.r_[initial_global, np.flatnonzero(groups != anchor)])
    initial = np.searchsorted(allowed, initial_global)
    return allowed, initial


def acquire(strategy, ids, x, identity, observed, observed_y, seed, config):
    if strategy in ['Random','Nearest sequence','AI greedy','AI explore']:
        return choose_batch(strategy, ids, x, identity, observed, observed_y, seed, config)
    if strategy not in ['AI uncertainty','AI diversity']:
        raise ValueError(strategy)
    observed = np.asarray(observed, dtype=int)
    if len(observed) != len(observed_y) or len(set(observed)) != len(observed):
        raise ValueError('Invalid revealed observations.')
    if not np.isfinite(observed_y).all() or (np.asarray(observed_y)<0).any():
        raise ValueError('Invalid revealed labels.')
    pool = np.setdiff1d(np.arange(len(ids)), observed)
    mean, std = bootstrap_predict(x, observed, observed_y, pool, seed, config)
    score = mean+config['ucb_beta']*std if strategy=='AI uncertainty' else mean
    unit = x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    chosen, details = [], []
    for _ in range(config['batch_size']):
        remain = np.flatnonzero(~np.isin(pool,chosen))
        acquisition = score[remain].copy()
        distance = np.full(len(remain),np.nan)
        if strategy == 'AI diversity':
            reference = np.r_[observed,chosen].astype(int)
            distance = np.maximum(0,1-(unit[pool[remain]] @ unit[reference].T).max(axis=1))
            w = config['diversity_weight']
            acquisition = (1-w)*percentile_rank(acquisition)+w*percentile_rank(distance)
        pick = np.lexsort((ids[pool[remain]],-acquisition))[0]
        j = remain[pick]
        chosen.append(int(pool[j]))
        details.append(dict(sample_id=str(ids[pool[j]]),predicted_log1p=mean[j],
            bootstrap_std_log1p=std[j],acquisition_score=acquisition[pick],diversity_distance=distance[pick]))
    return np.asarray(chosen), details


def correlation(a,b):
    return float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan


def campaign(ids,x,y,identity,groups,seed,config,strategies,anchor=None):
    validate_config(config,len(ids))
    if not np.isfinite(y).all() or (y<0).any(): raise ValueError('Invalid activities.')
    threshold = float(np.quantile(y,config['hit_quantile']))
    if threshold<=0: raise ValueError('Nonpositive hit threshold.')
    allowed,initial = starting_pool(groups,seed,config,anchor)
    ids,x,y,groups = ids[allowed],x[allowed],y[allowed],groups[allowed]
    identity = identity[np.ix_(allowed,allowed)]
    hit = y>=threshold
    initial_groups = set(groups[initial])
    untested = np.setdiff1d(np.arange(len(ids)),initial)
    available_hits = int(hit[untested].sum())
    available_hit_groups = len(set(groups[untested[hit[untested]]]))
    # Offline diagnostic on identical initial observations, not on adaptive policy pools.
    mean,std = bootstrap_predict(x,initial,y[initial],untested,seed*100+1,config)
    error = abs(np.log1p(y[untested])-mean)
    # Rank-based quartiles with deterministic ID tie breaks; equal spread has no information.
    order = np.lexsort((ids[untested],std))
    quartile = np.empty(len(untested),int)
    quartile[order] = np.minimum(np.arange(len(order))*4//len(order)+1,4)
    informative = bool(np.ptp(std)>0)
    if not informative:
        quartile[:] = 0  # No artificial uncertainty ordering when every score ties.
    diagnostics = [dict(sample_id=str(ids[i]),predicted_log1p=mean[j],
        observed_log1p=float(np.log1p(y[i])),bootstrap_std_log1p=std[j],
        abs_error_log1p=error[j],uncertainty_quartile=int(quartile[j]),
        max_initial_identity=float(identity[i,initial].max())) for j,i in enumerate(untested)]
    diagnostic = dict(uncertainty_error_spearman=correlation(std,error),
        prediction_spearman=correlation(mean,y[untested]),n_candidates=len(untested),
        mean_abs_error=float(error.mean()),mean_std=float(std.mean()),
        informative_uncertainty=informative,
        high_minus_low_error=float(error[quartile==4].mean()-error[quartile==1].mean()) if informative else np.nan)
    points, choices = [], []
    for strategy in strategies:
        observed = initial.copy()
        for i in initial:
            choices.append(dict(strategy=strategy,batch=0,revealed_at=len(initial),sample_id=ids[i],
                                activity=y[i],hit=bool(hit[i]),sequence_group=int(groups[i]),initial=True))
        batch = 0
        while True:
            acquired=observed[len(initial):]
            strong=acquired[hit[acquired]]
            new_groups=set(groups[strong])
            unfamiliar = np.array([g not in initial_groups for g in groups[strong]],dtype=bool)
            points.append(dict(strategy=strategy,tested=len(observed),hits=int(hit[observed].sum()),
                new_hits=len(strong),new_hit_groups=len(new_groups),unfamiliar_hits=int(unfamiliar.sum()),
                unfamiliar_hit_groups=len(new_groups-initial_groups),initial_hits=int(hit[initial].sum()),
                best_new_activity=float(y[acquired].max()) if len(acquired) else np.nan,
                threshold=threshold,n_available_hits=available_hits,n_available_hit_groups=available_hit_groups,
                new_hit_recall=len(strong)/available_hits if available_hits else np.nan,
                candidate_pool_size=len(untested),initial_group_count=len(initial_groups)))
            if len(observed)==config['max_budget']: break
            batch+=1
            selected, details=acquire(strategy,ids,x,identity,observed,y[observed],seed*100+batch,config)
            observed=np.r_[observed,selected]
            for i,detail in zip(selected,details):
                choices.append({**detail,'strategy':strategy,'batch':batch,'revealed_at':len(observed),
                    'activity':y[i],'hit':bool(hit[i]),'sequence_group':int(groups[i]),'initial':False})
    return points, choices, diagnostics, diagnostic


def summarize(points):
    pair_keys=['task','scenario','anchor','seed','tested']
    paired=points.copy()
    for strategy,suffix in [('Random','random'),('Nearest sequence','nearest'),('AI greedy','greedy')]:
        baseline=points[points.strategy.eq(strategy)][pair_keys+['new_hits','new_hit_groups']]
        baseline=baseline.rename(columns={c:f'{c}_{suffix}' for c in ['new_hits','new_hit_groups']})
        paired=paired.merge(baseline,on=pair_keys,validate='many_to_one')
        paired[f'delta_{suffix}']=paired.new_hits-paired[f'new_hits_{suffix}']
        paired[f'group_delta_{suffix}']=paired.new_hit_groups-paired[f'new_hit_groups_{suffix}']
    values=['hits','new_hits','new_hit_groups','unfamiliar_hits','unfamiliar_hit_groups','best_new_activity',
            'new_hit_recall','delta_random','delta_nearest','delta_greedy',
            'group_delta_random','group_delta_nearest','group_delta_greedy']
    anchor_summary=paired.groupby(['task','scenario','anchor','strategy','tested'])[values].mean().reset_index()
    summary=anchor_summary.groupby(['task','scenario','strategy','tested'])[values].mean().reset_index()
    spread=paired.groupby(['task','scenario','strategy','tested']).agg(
        p10_new_hits=('new_hits',lambda s:s.quantile(.1)),p90_new_hits=('new_hits',lambda s:s.quantile(.9)),
        win_rate_nearest=('delta_nearest',lambda s:(s>0).mean()),n_starts=('seed','size')).reset_index()
    summary=summary.merge(spread,on=['task','scenario','strategy','tested'])
    # Paired 2x2 acquisition effects, never tune weights to improve them.
    effects=[]
    for key,frame in paired.groupby(pair_keys):
        frame=frame.set_index('strategy')
        for metric in ['new_hits','new_hit_groups']:
            g,u,d,c=[float(frame.loc[s,metric]) for s in ['AI greedy','AI uncertainty','AI diversity','AI explore']]
            effects.append(dict(zip(pair_keys,key))|dict(metric=metric,uncertainty_added=u-g,
                diversity_added=d-g,combined_added=c-g,interaction=c-u-d+g))
    return summary,anchor_summary,paired,pd.DataFrame(effects)


def run(root):
    started=time.monotonic()
    config=json.loads((root/'campaign_config.json').read_text())
    extension=json.loads((root/'validation_config.json').read_text())
    if extension['strategies']!=STRATEGIES or extension['scenarios']!=['random_start','group_shift']:
        raise ValueError('Full fixed policy/scenario roster required.')
    bench=json.loads((root/'benchmark_config.json').read_text())
    data_manifest=validate_data_inputs(root,bench)
    base=root/'benchmark'
    catalog=pd.read_csv(base/'proteins.csv')
    spec=next(m for m in bench['models'] if m['key']==config['model_key'])
    revision=json.loads((base/'revisions'/f'{spec["key"]}.json').read_text())['revision']
    x,meta=check_embedding_cache(base/'embeddings'/f'{spec["key"]}.npz',catalog,spec,revision)
    identity=np.load(base/'identity.npy')
    lookup={pid:i for i,pid in enumerate(catalog.sample_id)}
    rows,choices,diagnostics,diagnostic_runs,design=[],[],[],[],[]
    with threadpool_limits(limits=1):
        for task in bench['tasks']:
            table=pd.read_csv(base/'tasks'/task['id']/'candidates.csv').sort_values('sample_id')
            ids=table.sample_id.to_numpy(); y=table.activity.to_numpy(); groups=table.sequence_group.to_numpy()
            indices=np.array([lookup[pid] for pid in ids]); task_x=x[indices]; sim=identity[np.ix_(indices,indices)]
            eligible=anchors(groups,config)
            if not eligible: raise ValueError('No eligible shifted-start group; do not silently omit a condition.')
            for anchor in [None,*eligible]:
                scenario='random_start' if anchor is None else 'group_shift'
                context={'task':task['id'],'scenario':scenario,'anchor':-1 if anchor is None else anchor}
                allowed,initial=starting_pool(groups,config['seed_start'],config,anchor)
                design.append({**context,'n_task':len(ids),'starting_group_size':len(ids) if anchor is None else int((groups==anchor).sum()),
                    'search_pool_size':len(allowed)-len(initial),'threshold':float(np.quantile(y,config['hit_quantile'])),
                    'n_seeds':config['n_seeds'],'max_cross_identity':None if anchor is None else float(sim[np.ix_(groups==anchor,groups!=anchor)].max())})
                for seed in range(config['seed_start'],config['seed_start']+config['n_seeds']):
                    p,c,d,dr=campaign(ids,task_x,y,sim,groups,seed,config,STRATEGIES,anchor)
                    rows.extend({**context,'seed':seed,**r} for r in p)
                    choices.extend({**context,'seed':seed,**r} for r in c)
                    diagnostics.extend({**context,'seed':seed,**r} for r in d)
                    diagnostic_runs.append({**context,'seed':seed,**dr})
                print(f'{task["id"]} {scenario} anchor={anchor}: complete ({time.monotonic()-started:.1f}s)',flush=True)
    points=pd.DataFrame(rows)
    summary,anchor_summary,paired,effects=summarize(points)
    outputs={'checkpoints':points,'selections':pd.DataFrame(choices),'summary':summary,
        'anchor_summary':anchor_summary,'paired':paired,'effects':effects,'design':pd.DataFrame(design),
        'initial_predictions':pd.DataFrame(diagnostics),'diagnostics':pd.DataFrame(diagnostic_runs)}
    out=root/'validation/results';out.mkdir(parents=True,exist_ok=True)
    for name,frame in outputs.items(): frame.to_csv(out/f'{name}.csv',index=False)
    code=['validation.py','campaign.py','bench_features.py','bench_eval.py','features.py','data.py']
    write_json(out/'manifest.json',dict(config=config,extension=extension,data_manifest=data_manifest,
        task_labels={t['id']:t['label'] for t in bench['tasks']},embedding_metadata=meta,
        code_sha256={name:digest(root/'src/pet_scout'/name) for name in code},
        config_sha256={name:digest(root/name) for name in ['campaign_config.json','validation_config.json']},
        result_sha256={f'{name}.csv':digest(out/f'{name}.csv') for name in outputs},
        n_campaigns=len(design)*config['n_seeds']*len(STRATEGIES),
        simulation_seconds=round(time.monotonic()-started,3),paid_compute_usd=0,
        versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
        prior_campaign_results=json.loads((root/'campaign/results/manifest.json').read_text())['result_sha256']))
    from .validation_report import report
    report(root)
    print(f'Completed {len(choices)} assay records in {time.monotonic()-started:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    args=parser.parse_args()
    run(args.root.resolve())


if __name__=='__main__': main()

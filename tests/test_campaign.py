import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pet_scout.campaign import bootstrap_predict, choose_batch, simulate, summarize
from pet_scout.campaign_report import verified_results

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'campaign_config.json').read_text())


def fixture():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(32,7))
    ids = np.array([f'p{i:03d}' for i in range(len(x))])
    y = np.arange(1,len(x)+1,dtype=float)
    identity = np.eye(len(x))*.5+.5
    config = {**CONFIG,'initial_size':8,'batch_size':4,'max_budget':24,
              'primary_budget':16,'bootstrap_members':3,'target_hits':3}
    return ids,x,y,identity,config


def test_bootstrap_solver_matches_sklearn_reference():
    ids,x,y,identity,config = fixture()
    observed, pool = np.arange(8),np.arange(8,20)
    mean,std = bootstrap_predict(x,observed,y[observed],pool,19,config)
    scaler = StandardScaler().fit(x[observed])
    train,test = scaler.transform(x[observed]),scaler.transform(x[pool])
    rng = np.random.default_rng(19)
    pred=[]
    for _ in range(config['bootstrap_members']):
        sample=rng.integers(0,8,8)
        model=Ridge(alpha=config['ridge_alpha']).fit(train[sample],np.log1p(y[observed][sample]))
        pred.append(model.predict(test))
    np.testing.assert_allclose(mean,np.mean(pred,axis=0),atol=1e-10)
    np.testing.assert_allclose(std,np.std(pred,axis=0,ddof=1),atol=1e-10)


def test_first_batch_invariant_to_every_hidden_outcome():
    ids,x,y,identity,config = fixture()
    seed=21
    initial=np.random.default_rng(seed).choice(len(ids),config['initial_size'],replace=False)
    changed=y.copy()
    hidden=np.setdiff1d(np.arange(len(ids)),initial)
    changed[hidden]=y[hidden][::-1]*100
    _, a=simulate(ids,x,y,identity,seed,config)
    _, b=simulate(ids,x,changed,identity,seed,config)
    for strategy in config['strategies']:
        first=lambda rows:[r['sample_id'] for r in rows if r['strategy']==strategy and r['batch']==1]
        assert first(a)==first(b)


def test_full_batch_reveal_and_metrics_are_exact():
    ids,x,y,identity,config=fixture()
    points,choices=simulate(ids,x,y,identity,21,config)
    table=pd.DataFrame(choices)
    starts=[]
    for strategy,frame in table.groupby('strategy'):
        assert len(frame)==config['max_budget']
        assert frame.sample_id.is_unique
        starts.append(tuple(frame[frame.batch.eq(0)].sample_id))
        for p in (p for p in points if p['strategy']==strategy):
            seen=frame[frame.revealed_at.le(p['tested'])]
            assert len(seen)==p['tested']
            assert seen.hit.sum()==p['hits']
            assert seen[seen.batch.gt(0)].hit.sum()==p['new_hits']
            assert seen.activity.max()==p['best_activity']
        assert frame.groupby('batch').revealed_at.nunique().eq(1).all()
    assert len(set(starts))==1


def test_nonreaching_runs_are_censored_not_dropped():
    config={**CONFIG,'max_budget':32}
    rows=[]
    for seed in [1,2]:
        for strategy in config['strategies']:
            for tested in [16,24,32]:
                hits=5 if seed==1 and tested>=24 else 1
                rows.append(dict(task='x',strategy=strategy,seed=seed,tested=tested,
                                 hits=hits,new_hits=hits-1,best_activity=10,target_reached=hits>=5))
    summary,paired,times=summarize(pd.DataFrame(rows),config)
    assert summary[summary.tested.eq(32)].target_success_rate.eq(.5).all()
    assert times.groupby('strategy').capped_assays.mean().eq(28).all()
    assert times[times.seed.eq(2)].first_reached_at.isna().all()
    assert paired.delta_random.eq(0).all()


def test_nearest_ties_use_ids_not_activity_and_no_duplicates():
    ids,x,y,identity,config=fixture()
    observed=np.array([7,1,4])
    # All cross-similarities tie. Nearest observed ID is p001, not the most active.
    chosen,details=choose_batch('Nearest sequence',ids,x,identity,observed,y[observed],31,config)
    assert [d['acquisition_score'] for d in details]==[y[1]]*4
    assert ids[chosen].tolist()==['p000','p002','p003','p005']


@pytest.mark.skipif(not (ROOT/'campaign/results/manifest.json').exists(),reason='Run campaign first')
def test_saved_campaign_integrity_and_all_metrics():
    manifest,tables=verified_results(ROOT)
    cfg=manifest['config']
    assert len(tables['checkpoints'])==manifest['n_campaigns']*7
    assert len(tables['selections'])==manifest['n_campaigns']*cfg['max_budget']
    lookup={k:g for k,g in tables['selections'].groupby(['task','strategy','seed'])}
    for row in tables['checkpoints'].itertuples():
        seen=lookup[(row.task,row.strategy,row.seed)]
        seen=seen[seen.revealed_at.le(row.tested)]
        assert len(seen)==row.tested and seen.sample_id.is_unique
        assert seen.hit.sum()==row.hits
        assert seen.activity.max()==pytest.approx(row.best_activity)
    for _,frame in tables['selections'].query('batch == 0').groupby(['task','seed']):
        assert frame.groupby('strategy').sample_id.apply(tuple).nunique()==1
    for task in manifest['task_labels']:
        actual=pd.read_csv(ROOT/'benchmark/tasks'/task/'candidates.csv').set_index('sample_id').activity
        rows=tables['selections'][tables['selections'].task.eq(task)]
        np.testing.assert_allclose(rows.activity,rows.sample_id.map(actual))
        threshold=actual.quantile(cfg['hit_quantile'])
        assert rows.hit.eq(rows.activity.ge(threshold)).all()


@pytest.mark.skipif(not (ROOT/'campaign/results/manifest.json').exists(),reason='Run campaign first')
def test_campaign_ui_filters_and_reveal():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    app.switch_page('pages/2_Experiment_planner.py').run()
    assert not app.exception
    app.selectbox(key='campaign_task').select('cool_acidic')
    app.selectbox(key='campaign_batch').select(1).run()
    assert not app.exception
    chosen=[df.value for df in app.dataframe if 'acquisition_score' in df.value.columns]
    assert len(chosen)==1 and len(chosen[0])==8 and 'activity' not in chosen[0]
    app.toggle(key='campaign_reveal').set_value(True).run()
    chosen=[df.value for df in app.dataframe if 'acquisition_score' in df.value.columns]
    assert not app.exception and 'activity' in chosen[0]


def test_invalid_budget_is_rejected():
    ids,x,y,identity,config=fixture()
    with pytest.raises(ValueError,match='budget'):
        simulate(ids,x,y,identity,1,{**config,'max_budget':100})

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pet_scout.campaign import simulate, choose_batch
from pet_scout.validation import anchors, starting_pool, acquire, campaign, summarize, STRATEGIES
from pet_scout.validation_report import verified

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'campaign_config.json').read_text())


def fixture():
    rng=np.random.default_rng(51)
    ids=np.array([f'p{i:03d}' for i in range(48)])
    x=rng.normal(size=(48,8));y=np.arange(1,49,dtype=float)
    groups=np.repeat([0,1,2],16)
    identity=np.where(groups[:,None]==groups[None,:],.8,.3)
    config={**CONFIG,'initial_size':8,'batch_size':4,'primary_budget':16,'max_budget':24,'bootstrap_members':3}
    return ids,x,y,identity,groups,config


def test_starting_group_eligibility_and_exclusion():
    ids,x,y,sim,groups,cfg=fixture()
    assert anchors(groups,cfg)==[0,1,2]
    allowed,initial=starting_pool(groups,21,cfg,1)
    assert len(initial)==8 and (groups[allowed[initial]]==1).all()
    remaining=np.setdiff1d(allowed,allowed[initial])
    assert len(remaining)==32 and (groups[remaining]!=1).all()
    with pytest.raises(ValueError,match='Ineligible'):
        starting_pool(groups,21,cfg,4)


def test_ablation_zero_weights_recover_greedy():
    ids,x,y,sim,groups,cfg=fixture()
    observed=np.arange(8)
    expected,_=choose_batch('AI greedy',ids,x,sim,observed,y[observed],7,cfg)
    u,_=acquire('AI uncertainty',ids,x,sim,observed,y[observed],7,{**cfg,'ucb_beta':0})
    d,_=acquire('AI diversity',ids,x,sim,observed,y[observed],7,{**cfg,'diversity_weight':0})
    np.testing.assert_array_equal(u,expected);np.testing.assert_array_equal(d,expected)


def test_reference_campaign_matches_original_policy_choices():
    ids,x,y,sim,groups,cfg=fixture()
    _,original=simulate(ids,x,y,sim,21,cfg)
    points,actual,_,_=campaign(ids,x,y,sim,groups,21,cfg,STRATEGIES)
    for strategy in cfg['strategies']:
        select=lambda rows:[r['sample_id'] for r in rows if r['strategy']==strategy]
        assert select(original)==select(actual)


@pytest.mark.parametrize('anchor',[None,1])
def test_hidden_outcomes_cannot_change_first_batch_including_new_policies(anchor):
    ids,x,y,sim,groups,cfg=fixture()
    allowed,initial=starting_pool(groups,21,cfg,anchor)
    changed=y.copy(); hidden=np.setdiff1d(np.arange(len(ids)),allowed[initial])
    changed[hidden]=changed[hidden][::-1]*10
    _,a,_,_=campaign(ids,x,y,sim,groups,21,cfg,STRATEGIES,anchor)
    _,b,_,_=campaign(ids,x,changed,sim,groups,21,cfg,STRATEGIES,anchor)
    for strategy in STRATEGIES:
        first=lambda rows:[r['sample_id'] for r in rows if r['strategy']==strategy and r['batch']==1]
        assert first(a)==first(b)


def test_shifted_metrics_and_diagnostics_are_evaluator_only():
    ids,x,y,sim,groups,cfg=fixture()
    points,choices,diagnostics,run=campaign(ids,x,y,sim,groups,21,cfg,STRATEGIES,1)
    choices=pd.DataFrame(choices)
    assert choices[choices.batch.gt(0)].sequence_group.ne(1).all()
    for point in points:
        seen=choices[(choices.strategy==point['strategy'])&choices.revealed_at.le(point['tested'])]
        new=seen[seen.batch.gt(0)];hits=new[new.hit]
        assert len(seen)==point['tested'] and seen.sample_id.is_unique
        assert len(hits)==point['new_hits']==point['unfamiliar_hits']
        assert hits.sequence_group.nunique()==point['new_hit_groups']==point['unfamiliar_hit_groups']
    assert len(diagnostics)==32==run['n_candidates']
    assert set(r['sample_id'] for r in diagnostics).isdisjoint(choices[choices.batch.eq(0)].sample_id)
    for row in diagnostics:
        assert row['abs_error_log1p']==pytest.approx(abs(row['observed_log1p']-row['predicted_log1p']))


def test_paired_ablation_effects_are_not_assumed_additive():
    rows=[]
    values={'Random':1,'Nearest sequence':2,'AI greedy':3,'AI uncertainty':4,'AI diversity':1,'AI explore':5}
    for strategy,value in values.items():
        rows.append(dict(task='t',scenario='group_shift',anchor=0,seed=1,tested=40,strategy=strategy,
            hits=value,new_hits=value,new_hit_groups=value,unfamiliar_hits=value,unfamiliar_hit_groups=value,
            best_new_activity=10,new_hit_recall=.5))
    summary,_,paired,effects=summarize(pd.DataFrame(rows))
    row=effects[effects.metric.eq('new_hits')].iloc[0]
    assert row.uncertainty_added==1 and row.diversity_added==-2 and row.combined_added==2
    assert row.interaction==3
    assert summary.set_index('strategy').loc['AI greedy','delta_nearest']==1


def test_constant_uncertainty_has_no_artificial_quartile_ranking():
    ids,x,y,sim,groups,cfg=fixture()
    y[groups==1]=0
    _,_,diagnostics,run=campaign(ids,x,y,sim,groups,21,cfg,STRATEGIES,1)
    assert not run['informative_uncertainty']
    assert np.isnan(run['uncertainty_error_spearman']) and np.isnan(run['high_minus_low_error'])
    assert all(row['uncertainty_quartile']==0 for row in diagnostics)


@pytest.mark.skipif(not (ROOT/'validation/results/manifest.json').exists(),reason='Run validation study')
def test_saved_study_reference_parity_preservation_and_metrics():
    from pet_scout.data import digest
    manifest=verified(ROOT);base=ROOT/'validation/results'
    for name,expected in manifest['prior_campaign_results'].items():
        assert digest(ROOT/'campaign/results'/name)==expected
    choices=pd.read_csv(base/'selections.csv');points=pd.read_csv(base/'checkpoints.csv')
    assert len(choices)==manifest['n_campaigns']*64
    keys=['task','scenario','anchor','seed','strategy']
    assert choices.groupby(keys).sample_id.nunique().eq(64).all()
    initial=choices[choices.batch.eq(0)]
    for _,frame in initial.groupby(keys[:-1]):
        assert frame.groupby('strategy').sample_id.apply(tuple).nunique()==1
    shifted=choices[choices.scenario.eq('group_shift')]
    assert shifted[shifted.batch.eq(0)].sequence_group.eq(shifted[shifted.batch.eq(0)].anchor).all()
    assert shifted[shifted.batch.gt(0)].sequence_group.ne(shifted[shifted.batch.gt(0)].anchor).all()
    reference=choices[choices.scenario.eq('random_start')&choices.strategy.isin(CONFIG['strategies'])]
    old=pd.read_csv(ROOT/'campaign/results/selections.csv')
    cols=['task','strategy','seed','batch','sample_id','activity','hit']
    sort=['task','strategy','seed','batch','sample_id']
    pd.testing.assert_frame_equal(reference[cols].sort_values(sort).reset_index(drop=True),old[cols].sort_values(sort).reset_index(drop=True))
    # Recompute every new-hit count using only the independently logged chosen assays.
    grouped={k:g for k,g in choices.groupby(keys)}
    for key,frame in points.groupby(keys):
        acquired=grouped[key].query('batch > 0')
        for row in frame.itertuples():
            hits=acquired[acquired.revealed_at.le(row.tested)&acquired.hit]
            assert len(hits)==row.new_hits and hits.sequence_group.nunique()==row.new_hit_groups
    for task in choices.task.unique():
        source=pd.read_csv(ROOT/'benchmark/tasks'/task/'candidates.csv').set_index('sample_id')
        current=choices[choices.task.eq(task)]
        np.testing.assert_allclose(current.activity,current.sample_id.map(source.activity))
        assert current.sequence_group.eq(current.sample_id.map(source.sequence_group)).all()
        assert current.hit.eq(current.activity.ge(source.activity.quantile(.9))).all()


@pytest.mark.skipif(not (ROOT/'validation/results/manifest.json').exists(),reason='Run validation study')
def test_validation_dashboard_filters():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    app.switch_page('pages/3_Validation_&_case_study.py').run()
    assert not app.exception and len(app.error)==0
    app.selectbox(key='validation_task').select('cool_acidic')
    app.selectbox(key='validation_anchor').select(0)
    app.radio(key='validation_metric').set_value('new_hit_groups').run()
    assert not app.exception and len(app.error)==0
    app.selectbox(key='validation_scenario').select('random_start').run()
    assert not app.exception and len(app.error)==0
    assert app.selectbox(key='validation_anchor').value=='all'


def test_stale_validation_configuration_rejected(tmp_path):
    base=tmp_path/'validation/results';base.mkdir(parents=True)
    (tmp_path/'validation_config.json').write_text('{}')
    (base/'manifest.json').write_text(json.dumps({'config_sha256':{'validation_config.json':'old'}}))
    with pytest.raises(ValueError,match='configuration changed'): verified(tmp_path)

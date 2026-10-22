"""Scientific contracts for reusable features and matched-budget benchmark splits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pet_scout.bench_data import nested_training_subset
from pet_scout.bench_features import amino_acid_mean, check_embedding_cache, POOLING_VERSION
from pet_scout.data import digest, write_json
from pet_scout.features import sequence_fingerprint

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'benchmark'


def test_nested_subsets_have_exact_budgets_and_no_extra_examples():
    train = np.arange(100,180)
    small = nested_training_subset(train,16,11,0)
    medium = nested_training_subset(train,32,11,0)
    all_examples = nested_training_subset(train,'all',11,0)
    assert len(set(small)) == 16 and len(set(medium)) == 32
    np.testing.assert_array_equal(small,medium[:16])
    np.testing.assert_array_equal(medium,all_examples[:32])
    assert set(all_examples) == set(train)
    with pytest.raises(ValueError): nested_training_subset(train,81,11,0)


@pytest.mark.parametrize('specials,lengths,expected', [
    ([[1,0,0,1,1],[1,0,0,0,1]],[2,3],[1.5,7.0]),
    ([[0,0,1,1,1],[0,0,0,1,1]],[2,3],[.5,6.0])
])
def test_pooling_excludes_specials_and_padding_for_esm_and_t5(specials,lengths,expected):
    import torch
    hidden = torch.arange(10,dtype=torch.float32).reshape(2,5,1)
    result = amino_acid_mean(hidden,torch.ones((2,5)),torch.tensor(specials),lengths)
    np.testing.assert_allclose(result.numpy().ravel(),expected)
    with pytest.raises(ValueError,match='one non-special token'):
        amino_acid_mean(hidden,torch.ones((2,5)),torch.tensor(specials),[3,3])


def test_feature_cache_rejects_wrong_order_revision_and_corruption(tmp_path):
    path = tmp_path/'test.npz'
    table = pd.DataFrame({'sample_id':['a','b'],'sequence':['ACD','EFG']})
    spec = {'model_id':'test/model'}
    np.savez_compressed(path,embeddings=np.ones((2,3)),ids=np.array(['a','b']))
    meta = {'model_id':spec['model_id'],'revision':'exact-revision',
            'sequence_sha256':sequence_fingerprint(table),'pooling_version':POOLING_VERSION,
            'embedding_sha256':digest(path)}
    write_json(path.with_suffix('.json'),meta)
    check_embedding_cache(path,table,spec,'exact-revision')
    with pytest.raises(ValueError,match='provenance'): check_embedding_cache(path,table,spec,'wrong-revision')
    swapped = table.copy();swapped['sample_id']=['b','a']
    with pytest.raises(ValueError,match='ID order'): check_embedding_cache(path,swapped,spec,'exact-revision')
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'): check_embedding_cache(path,table,spec,'exact-revision')


@pytest.mark.skipif(not (BASE/'splits.csv').exists(),reason='Run pet-bench prepare')
def test_actual_splits_respect_global_groups_and_nested_training_budgets():
    splits = pd.read_csv(BASE/'splits.csv',dtype={'training_size':str})
    groups = pd.read_csv(BASE/'groups.csv').set_index('sample_id').sequence_group.to_dict()
    for _, frame in splits.groupby(['task','regime','seed','fold']):
        assert frame.test_ids.nunique() == 1
        test = set(frame.test_ids.iloc[0].split(';'))
        previous = set()
        order = frame.assign(order=frame.training_size.map({'16':0,'32':1,'64':2,'all':3})).sort_values('order')
        for row in order.itertuples():
            train = set(row.train_ids.split(';'))
            assert not train & test
            assert previous <= train
            assert len(train) == row.n_train and len(test) == row.n_test
            if row.training_size != 'all': assert len(train) == int(row.training_size)
            if row.regime == 'grouped':
                assert not {groups[p] for p in train} & {groups[p] for p in test}
                assert row.max_cross_identity < .6
            previous = train
    for (task,_,_,_), frame in splits.groupby(['task','regime','seed','training_size']):
        actual = [pid for ids in frame.test_ids for pid in ids.split(';')]
        expected = pd.read_csv(BASE/'tasks'/task/'candidates.csv').sample_id.tolist()
        assert sorted(actual) == sorted(expected)


@pytest.mark.skipif(not (BASE/'results/manifest.json').exists(),reason='Run pet-bench evaluate')
def test_completed_benchmark_is_complete_and_results_are_auditable():
    manifest = json.loads((BASE/'results/manifest.json').read_text())
    for name,checksum in manifest['result_sha256'].items():
        assert digest(BASE/'results'/name) == checksum
    pred = pd.read_csv(BASE/'results/predictions.csv',dtype={'training_size':str})
    metrics = pd.read_csv(BASE/'results/metrics.csv',dtype={'training_size':str})
    assert set(pred.model) == set(manifest['models'])
    assert len(pred) == manifest['n_predictions']
    keys = ['task','regime','seed','fold','training_size']
    # Every method must use the same candidate pool and observations in every setting.
    for _, frame in pred.groupby(keys):
        assert frame.groupby('model').size().nunique() == 1
        assert frame.groupby('sample_id').model.nunique().eq(len(manifest['models'])).all()
        assert frame.groupby('sample_id')['observed'].nunique().eq(1).all()
    # Independently recompute all top-five selections in seed 11, full-data runs.
    chosen_metrics = metrics[(metrics.seed==11)&(metrics.training_size=='all')&(metrics.budget==5)]
    lookup = {k:g for k,g in pred[(pred.seed==11)&(pred.training_size=='all')].groupby(keys+['model'])}
    for row in chosen_metrics.itertuples():
        pool = lookup[tuple(getattr(row,k) for k in keys+['model'])]
        chosen = pool.sort_values(['predicted','sample_id'],ascending=[False,True]).head(5)
        assert ';'.join(chosen.sample_id) == row.selected_ids
        assert chosen.observed.mean() == pytest.approx(row.selected_mean)
        assert pool.observed.mean() == pytest.approx(row.random_expected_mean)


@pytest.mark.skipif(not (BASE/'results/manifest.json').exists(),reason='Run pet-bench evaluate')
def test_dashboard_filters_and_hidden_results():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    assert not app.exception
    assert 'five experiments' in app.title[0].value
    app.selectbox(key='bench_task').select('cool_acidic')
    app.selectbox(key='bench_regime').select('matched')
    app.selectbox(key='bench_size').select('16').run()
    assert not app.exception
    app.selectbox(key='bench_model').select('ProtT5-XL encoder')
    app.toggle(key='bench_reveal').set_value(False).run()
    assert not app.exception
    # Hidden outcomes must be absent from the selected-batch table.
    batch_tables = [df.value for df in app.dataframe if 'nearest_train_id' in df.value.columns]
    assert len(batch_tables) == 1 and 'observed' not in batch_tables[0]


def test_dashboard_refuses_stale_results(tmp_path):
    from streamlit.testing.v1 import AppTest
    (tmp_path/'benchmark/results').mkdir(parents=True)
    write_json(tmp_path/'benchmark/results/manifest.json',{'config':{'version':'old'},'data_manifest':{}})
    write_json(tmp_path/'benchmark/data_manifest.json',{})
    write_json(tmp_path/'benchmark_config.json',{'version':'new'})
    app = AppTest.from_string(f'from pathlib import Path\nfrom pet_scout.bench_ui import render\nrender(Path({str(tmp_path)!r}))',default_timeout=30).run()
    assert not app.exception
    assert 'earlier configuration' in app.warning[0].value
    assert len(app.dataframe) == 0

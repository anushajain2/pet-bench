"""Integration checks against the actual completed experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pet_scout.data import digest

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'reports/run_manifest.json').exists(),reason='Run pet-scout all first')


def test_artifact_provenance_and_coverage():
    manifest=json.loads((ROOT/'reports/run_manifest.json').read_text())
    data=pd.read_csv(ROOT/'data/processed/candidates.csv')
    pred=pd.read_csv(ROOT/'reports/predictions.csv')
    assert manifest['dataset_sha256']==digest(ROOT/'data/processed/candidates.csv')
    assert manifest['predictions_sha256']==digest(ROOT/'reports/predictions.csv')
    assert not data.sequence.duplicated().any()
    assert not set(data.sample_id)&{'blank','LCC-ICCG','sumo_LCC-ICCG','RFP'}
    for _,frame in pred.groupby(['regime','model','seed']):
        assert len(frame)==len(data)
        assert set(frame.sample_id)==set(data.sample_id)
        assert not frame.sample_id.duplicated().any()
    np.testing.assert_allclose(pred.observed,pred.sample_id.map(data.set_index('sample_id').activity))


def test_all_grouped_folds_respect_identity():
    splits=pd.read_csv(ROOT/'reports/split_audit.csv')
    grouped=splits[splits.regime.eq('grouped')]
    assert grouped.shared_groups.eq(0).all()
    assert grouped.max_cross_identity.lt(.6).all()
    for r in splits.itertuples():
        assert not set(r.train_ids.split(';'))&set(r.test_ids.split(';'))


def test_selected_candidates_belong_to_same_pool():
    pred=pd.read_csv(ROOT/'reports/predictions.csv')
    metrics=pd.read_csv(ROOT/'reports/selection_metrics.csv')
    for row in metrics.itertuples():
        pool=pred[(pred.regime==row.regime)&(pred.model==row.model)&(pred.seed==row.seed)&(pred.fold==row.fold)]
        expected=pool.sort_values(['predicted','sample_id'],ascending=[False,True]).head(row.budget)
        assert row.selected_ids==';'.join(expected.sample_id)
        assert np.isclose(row.selected_mean,expected.observed.mean())
        assert np.isclose(row.random_expected_mean,pool.observed.mean())

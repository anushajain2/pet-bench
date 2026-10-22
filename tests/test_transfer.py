from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from pet_scout.external import translate_insert, construct_overlap, evaluate_scores, import_study, verified
from pet_scout.ranking import parse_csv, parse_fasta, rank_table

ROOT = Path(__file__).resolve().parents[1]


def upload():
    rng = np.random.default_rng(91)
    aa = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    return pd.DataFrame({'sample_id':[f'p{i:02d}' for i in range(12)],
        'sequence':[''.join(rng.choice(aa, 40)) for _ in range(12)],
        'activity':[str(i) for i in range(8)]+['']*4})


def fake_embeddings(root, sequences):
    # Stable sequence-only representation; deliberately no access to activities.
    return np.array([[s.count(c) for c in 'ACDEFGHIKLMNPQRSTVWY'] for s in sequences], dtype=float)


def test_documented_dna_boundaries_and_translation():
    assert translate_insert('CATATGGCTTGCTAAGCGGCCGC') == 'MAC'
    for bad in ['ATGGCTTAA', 'CATATGNGCTAAGCGGCCGC', 'CATATGTAGTAAGCGGCCGC', 'CATATGATAAGCGGCCGC']:
        with pytest.raises(ValueError):
            translate_insert(bad)


def test_construct_overlap_distinguishes_tags_from_mutations():
    sequence = 'ACDEFGHIKLMNPQRSTVWY'*10
    assert construct_overlap(sequence, 'M'+sequence)
    assert not construct_overlap(sequence, 'Y'+sequence[1:])
    assert not construct_overlap(sequence, sequence[:100])


def test_exact_random_distribution_and_tie_breaks():
    ids = np.array(['c','b','a','d'])
    y = np.array([1.,2.,8.,4.])
    result = evaluate_scores(ids, y, np.ones(4), 2)
    assert result['selected_ids'] == 'a, b'
    assert result['selected_mean_um'] == 5
    assert result['random_mean_um'] == y.mean()
    triples = [y[list(x)].mean() for x in itertools.combinations(range(4),2)]
    assert result['random_fraction_at_least_selected'] == np.mean(np.array(triples)>=5)
    assert np.isnan(result['spearman'])


@pytest.mark.parametrize('value', ['NaN','NA','inf','-1','not measured'])
def test_missing_is_not_invalid_measurement(value):
    frame = upload(); frame.loc[0,'activity'] = value
    with pytest.raises(ValueError, match='finite'):
        parse_csv(frame.to_csv(index=False))


@pytest.mark.parametrize('kind', ['duplicate_id','duplicate_sequence','invalid_aa','long','mixed','all_same','too_few','formula'])
def test_upload_rejects_ambiguous_inputs(kind):
    frame = upload()
    if kind == 'duplicate_id': frame.loc[1,'sample_id'] = frame.sample_id[0]
    if kind == 'duplicate_sequence': frame.loc[1,'sequence'] = frame.sequence[0]
    if kind == 'invalid_aa': frame.loc[1,'sequence'] += 'X'
    if kind == 'long': frame.loc[1,'sequence'] = 'A'*1023
    if kind == 'mixed': frame['condition'] = ['a','b']*6
    if kind == 'all_same': frame.loc[:7,'activity'] = '1'
    if kind == 'too_few': frame.loc[7,'activity'] = ''
    if kind == 'formula': frame.loc[0,'sample_id'] = '=SUM(1,2)'
    with pytest.raises(ValueError):
        parse_csv(frame.to_csv(index=False))


def test_fasta_and_csv_equivalence_and_unknown_labels():
    frame = upload()
    fasta = ''.join(f'>{r.sample_id} description\n{r.sequence}\n' for r in frame.itertuples())
    labels = frame.loc[:7,['sample_id','activity']]
    pd.testing.assert_frame_equal(parse_csv(frame.to_csv(index=False)), parse_fasta(fasta, labels.to_csv(index=False)))
    labels.loc[0,'sample_id'] = 'absent'
    with pytest.raises(ValueError, match='absent'):
        parse_fasta(fasta, labels.to_csv(index=False))


def test_ranking_is_deterministic_order_invariant_and_does_not_write(tmp_path):
    frame = upload()
    table = parse_csv(frame.to_csv(index=False))
    result, manifest = rank_table(tmp_path, table, '40 C PET assay', 'umol/mg', 3, fake_embeddings)
    shuffled = parse_csv(frame.sample(frac=1,random_state=9).to_csv(index=False))
    again, meta = rank_table(tmp_path, shuffled, '40 C PET assay', 'umol/mg', 3, fake_embeddings)
    pd.testing.assert_frame_equal(result, again)
    assert manifest == meta
    assert result.selected_for_batch.sum() == 3
    assert set(result.sample_id) == set(frame.sample_id[8:])
    assert set(result.nearest_measured_id).issubset(set(frame.sample_id[:8]))
    assert not list(tmp_path.iterdir())
    assert 'sequence' not in result.columns
    assert manifest['n_measured'] == 8
    assert manifest['n_untested'] == 4


@pytest.mark.skipif(not (ROOT/'external/manifest.json').exists(), reason='Run external evaluation')
def test_real_external_import_and_provenance():
    import json
    config = json.loads((ROOT/'external_protocol.json').read_text())
    sequences, assays = import_study(ROOT, config)
    assert len(sequences) == 34 and len(assays) == 70
    assert assays.groupby('temperature_c').size().eq(14).all()
    assert set(assays.sample_id).issubset(set(sequences.sample_id))
    manifest = verified(ROOT)
    assert manifest['n_excluded_overlap'] == 1
    scores = pd.read_csv(ROOT/'external/metrics.csv')
    assert set(scores.n) == {11,13}
    assert scores.selected_ids.str.contains('APET-15').sum() == 0


@pytest.mark.skipif(not (ROOT/'external/manifest.json').exists(), reason='Run external evaluation')
def test_external_dashboard_conditions_and_cohorts():
    import json
    app = AppTest.from_file(str(ROOT/'pages/4_External_validation.py'), default_timeout=30).run()
    assert not app.exception
    chart = json.loads(app.get('plotly_chart')[0].proto.spec)
    assert {trace['name'] for trace in chart['data']} == {'ESM-2 35M','Composition','Nearest sequence','Random expectation','Oracle (knows outcomes)'}
    assert app.metric[0].value == '13'
    app.toggle[0].set_value(True).run()
    assert not app.exception and app.metric[0].value == '11'
    app.selectbox[0].select(60).run()
    assert not app.exception
    assert len(json.loads(app.get('plotly_chart')[0].proto.spec)['data']) == 5


@pytest.mark.skipif(not (ROOT/'benchmark/embeddings/esm2_35m.npz').exists(), reason='Need public cache')
def test_public_demo_and_stale_result_clearing():
    app = AppTest.from_file(str(ROOT/'pages/5_Rank_candidates.py'), default_timeout=45).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert 'candidate_run' in app.session_state
    assert len(app.session_state['candidate_run']['manifest']['selected_ids']) == 3
    app.number_input[0].set_value(2).run()
    assert not app.exception
    assert 'candidate_run' not in app.session_state

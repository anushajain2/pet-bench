"""Public external evaluation and local candidate ranking interfaces."""
import hashlib
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from .external import verified
from .ranking import parse_csv, parse_fasta, rank_table


def external_page(root):
    st.set_page_config(page_title='PETBench · External validation', page_icon='🧪', layout='wide')
    st.caption('PETBENCH / INDEPENDENT STUDY')
    st.title('Does the ranking travel to another lab?')
    st.markdown('Train on one published study. Choose three enzymes from another. **Would those picks break down more PET?**')
    try:
        manifest = verified(root)
    except (OSError, ValueError, KeyError) as error:
        st.error(f'Results unavailable or stale: {error}')
        st.stop()
    st.info('Small, retrospective transfer test: 14 preselected candidates → 13 after removing one overlapping construct. Different PET material, pH, duration and activity units. No external-label tuning.')
    a, b = st.columns(2)
    temperature = a.selectbox('External assay temperature', [40, 60], format_func=lambda t: f'{t}°C · {"primary" if t == 40 else "secondary"}')
    distant = b.toggle('Only candidates below 60% identity to training catalog', value=False)
    cohort = 'identity_below_60pct' if distant else 'nonoverlapping'
    metrics = pd.read_csv(root/'external/metrics.csv')
    selected = metrics[metrics.external_temperature_c.eq(temperature) & metrics.cohort.eq(cohort)].copy()
    if selected.empty:
        st.warning('Too few eligible candidates for the fixed evaluation protocol.')
        st.stop()
    ai = selected[selected.model.eq('ESM-2 35M')].iloc[0]
    cols = st.columns(4)
    cols[0].metric('Candidates evaluated', int(ai.n))
    cols[1].metric('AI top-three mean', f'{ai.selected_mean_um:.1f} µM')
    cols[2].metric('Random top-three expectation', f'{ai.random_mean_um:.1f} µM')
    cols[3].metric('AI rank correlation', f'{ai.spearman:.2f}')
    chart = selected[['model', 'selected_mean_um']].rename(columns={'model':'Method','selected_mean_um':'PET products (µM)'}).reset_index(drop=True)
    chart.loc[len(chart)] = ['Random expectation', ai.random_mean_um]
    chart.loc[len(chart)] = ['Oracle (knows outcomes)', ai.oracle_mean_um]
    figure = px.bar(chart, y='Method', x='PET products (µM)', color='Method', orientation='h',
                    color_discrete_sequence=['#236b8e','#89b4c2','#ca8b62','#b8bec6','#688d70'])
    figure.update_layout(showlegend=False, height=360, margin=dict(t=15,b=15))
    figure.update_yaxes(autorange='reversed', title=None)
    st.plotly_chart(figure, width='stretch')
    st.caption('Mean measured TPA + MHET concentration for each method’s three picks. Compare methods within this panel; concentrations are not interchangeable with the training study’s activity units.')
    if not distant and temperature == 40:
        st.success('AI finds a productive trio, but composition nearly ties it. This is evidence that some ranking information transfers—not proof that a large protein model is necessary.')
    elif not distant:
        st.info('At 60°C, nearest-sequence selection slightly beats AI on the top-three outcome, even though AI has better overall rank correlation.')
    else:
        st.info('Removing close relatives leaves 11 candidates. AI’s advantage is smaller in absolute product concentration; the full-panel headline is partly driven by close relatives of strong enzymes.')
    st.dataframe(selected[['model','spearman','selected_mean_um','random_mean_um','selected_ids']], hide_index=True, width='stretch')
    with st.expander('Exact random-selection comparison'):
        st.write(f'All possible triples were enumerated. A random batch falls between {ai.random_p05_um:.1f} and {ai.random_p95_um:.1f} µM in the central 90% of triples; {100*ai.random_fraction_at_least_selected:.1f}% score at least as high as AI’s picks.')
        st.caption('These are selection probabilities within this fixed panel, not confidence intervals or a generalization significance claim. Assay replicate uncertainty is not propagated into the model comparison.')
    with st.expander('Sequence overlap and source audit'):
        st.write('APET-15 overlaps training construct DP093 and is excluded. APET-14 and APET-17 have nearest training identities of 69% and 81%; the distant-candidate toggle removes both. Identity includes terminal gaps.')
        st.dataframe(pd.read_csv(root/'external/sequence_audit.csv'), hide_index=True, width='stretch')
        st.write('Supplementary Data 1 supplies synthesized DNA. The importer validates the documented NdeI/NotI boundaries, translates complete coding inserts and checks for internal stops. The unavailable vector-derived His tag is not invented.')
    st.markdown('Source: [Wu et al., Nature Communications (2025)](https://www.nature.com/articles/s41467-025-61599-z), Figure 2c and Supplementary Data 1, CC BY 4.0. Training: Norton-Baker et al. (2025), CC BY-NC 4.0. This is a cross-study test, not a claim of discovering a new enzyme.')
    st.download_button('Download all transfer results', (root/'external/metrics.csv').read_bytes(), 'petbench-transfer-results.csv', 'text/csv')
    st.download_button('Download fixed protocol', (root/'external_protocol.json').read_bytes(), 'petbench-transfer-protocol.json', 'application/json')
    st.caption(f'Provenance verified for {len(manifest["input_and_output_sha256"])} files. Pretraining sequence overlap is unknown; this selected panel cannot establish performance on arbitrary proteins or mutation effects.')


def ranking_page(root):
    st.set_page_config(page_title='PETBench · Rank candidates', page_icon='🧬', layout='wide')
    st.caption('PETBENCH / YOUR NEXT EXPERIMENT')
    st.title('Which enzymes should you test next?')
    st.write('Supply proteins you have measured and candidates you have not tested. Fit a small model to your measurements and download a proposed batch with a sequence-similarity comparison.')
    st.info('Local research tool. Uploads stay in this app session and are processed on this machine. No paid API, external sequence upload, or automatic file storage. Exported results include IDs and measured activities—keep them private when needed.')
    mode = st.radio('Input', ['Public example', 'Combined CSV', 'FASTA + measurements CSV'], horizontal=True)
    table = None
    if mode == 'Public example':
        raw = (root/'examples/ranking_demo.csv').read_bytes()
        st.caption('Published 40°C / pH 7.5 measurements: 16 labeled proteins and 24 candidates with labels withheld. This demonstrates the workflow; it is not a new validation result.')
        st.download_button('Download example input', raw, 'ranking_demo.csv', 'text/csv')
        table = parse_csv(raw)
    elif mode == 'Combined CSV':
        st.code('sample_id,sequence,activity\nmeasured_1,MALW...,12.5\ncandidate_1,MKLV...,', language='text')
        st.caption('Use actual full protein sequences. At least 8 measured proteins, 1 untested candidate, at most 500 total. Blank activity means untested; zero means a measured zero. Merge comparable replicates before upload.')
        uploaded = st.file_uploader('Combined CSV', type=['csv'])
        if uploaded:
            try:
                table = parse_csv(uploaded.getvalue())
            except ValueError as error:
                st.error(str(error))
    else:
        st.caption('FASTA first header token is the sample ID. Measurements CSV needs sample_id,activity; omitted IDs are untested. Use at least 8 measured and 1 untested sequence, at most 500 total.')
        fasta = st.file_uploader('Protein sequences', type=['fasta','fa','faa'])
        labels = st.file_uploader('Measured activities', type=['csv'])
        if fasta and labels:
            try:
                table = parse_fasta(fasta.getvalue(), labels.getvalue())
            except ValueError as error:
                st.error(str(error))
    a, b = st.columns([2,1])
    demo = mode == 'Public example'
    condition = a.text_input('Assay condition', value='40°C, pH 7.5, crystalline PET powder, 48 hours' if demo else '', key=f'condition-{mode}')
    units = b.text_input('Activity units', value='µmol products / mg enzyme' if demo else '', key=f'units-{mode}')
    comparable = st.checkbox('All supplied measurements use the same substrate, conditions, duration, units and normalization.', value=demo, key=f'confirm-{mode}')
    if table is None:
        st.session_state.pop('candidate_run', None)
        return
    n_pool = int(table.activity.isna().sum())
    st.write(f'**{table.activity.notna().sum()} measured · {n_pool} untested proteins.**')
    batch = int(st.number_input('How many candidates can you test?', min_value=1, max_value=n_pool, value=min(3,n_pool)))
    fingerprint = hashlib.sha256((table.to_csv(index=False)+condition+units+str(batch)+str(comparable)).encode()).hexdigest()
    if st.session_state.get('candidate_run', {}).get('fingerprint') != fingerprint:
        st.session_state.pop('candidate_run', None)
    if st.button('Rank candidates', type='primary', disabled=not comparable or not condition.strip() or not units.strip()):
        try:
            with st.spinner('Embedding sequences locally and fitting to your measured proteins…'):
                result, manifest = rank_table(root, table, condition, units, batch)
            st.session_state.candidate_run = {'fingerprint':fingerprint, 'result':result, 'manifest':manifest}
        except (ValueError, OSError) as error:
            st.error(str(error))
    if 'candidate_run' in st.session_state:
        run = st.session_state.candidate_run
        result, manifest = run['result'], run['manifest']
        st.subheader('Your proposed batch')
        st.write(', '.join(manifest['selected_ids']))
        st.dataframe(result, hide_index=True, width='stretch')
        st.caption('AI ranks by ensemble mean; the sequence baseline uses the activity of each candidate’s closest measured neighbor. Positive rank disagreement means AI ranks that candidate higher. Bootstrap spread is model variability, not a calibrated confidence interval.')
        st.warning('Recommendations need experimental validation. The evidence here concerns distinct natural enzymes; this has not been validated for single-mutation design. A large score or small bootstrap spread does not guarantee success.')
        c, d = st.columns(2)
        c.download_button('Download ranked candidates', result.to_csv(index=False).encode(), 'ranking.csv', 'text/csv')
        d.download_button('Download run manifest', json.dumps(manifest, indent=2, allow_nan=False).encode(), 'ranking-manifest.json', 'application/json')
    with st.expander('How it works and local compute'):
        st.write('Frozen ESM-2 35M embeddings → scaling fitted only to measured proteins → 20 bootstrap ridge fits with fixed regularization. Activity is transformed with log1p. Candidate labels never enter training. Scores are for prioritization, not calibrated absolute activity.')
        st.write('The public example reuses checked embeddings. New proteins use a pinned local checkpoint on CPU, with no truncation. First install the checkpoint using the command in TRANSFER_AND_RANKING.md. User sequences and embeddings are not cached to disk by this application.')

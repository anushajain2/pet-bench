"""Read-only interface to the complete PETBench results."""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

REGIMES = {'grouped': 'Sequence-group holdout', 'matched': 'Size-matched random', 'random': 'Random holdout'}
COLORS = {'Nearest sequence':'#81968e', 'Composition ridge':'#c79a55', 'ESM-2 35M':'#75b59f',
          'ESM-2 150M':'#2d9377', 'ESM-2 650M':'#075f4c', 'ProtT5-XL encoder':'#727fb0'}


@st.cache_data
def load_tables(basepath, stamp):
    base = __import__('pathlib').Path(basepath)
    names = ['summary','seed_metrics','predictions','compute','fit_timings']
    tables = {name: pd.read_csv(base/'results'/f'{name}.csv', dtype={'training_size':str}) for name in names}
    tables['tasks'] = pd.read_csv(base/'task_summary.csv')
    tables['splits'] = pd.read_csv(base/'splits.csv', dtype={'training_size':str})
    return tables


def render(root):
    st.set_page_config(page_title='PETBench | Protein models under test', page_icon='🧬', layout='wide')
    st.markdown('''<style>
    .block-container {max-width:1320px;padding-top:3.6rem;}
    h1 {letter-spacing:-1.5px;font-size:3rem!important;}
    [data-testid="stMetric"] {border:1px solid #dbe5dc;border-radius:12px;padding:16px;background:#fff;}
    .eyebrow {letter-spacing:2px;font-size:12px;font-weight:700;color:#126b54;}
    </style>''', unsafe_allow_html=True)
    base = root/'benchmark'
    path = base/'results/manifest.json'
    st.markdown('<div class="eyebrow">PETBENCH / REPRODUCIBLE PROTEIN MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.title('Which model would you trust with five experiments?')
    st.write('Compare how six methods select plastic-degrading enzymes across temperature, acidity and training-data budgets.')
    if not path.exists():
        st.info('Benchmark preparation is in progress. Run `pet-bench all` to produce the complete results. No mock scores are shown.')
        config_path = root/'benchmark_config.json'
        if config_path.exists():
            planned = json.loads(config_path.read_text())
            progress = []
            for spec in planned['models']:
                metadata = base/'embeddings'/f'{spec["key"]}.json'
                meta = json.loads(metadata.read_text()) if metadata.exists() else {}
                progress.append({'Model':spec['label'],'Embedding status':'Complete' if meta else 'Pending',
                                 'Inference seconds':meta.get('inference_seconds')})
            st.dataframe(pd.DataFrame(progress),hide_index=True,width='stretch')
        st.page_link('pages/1_Original_study.py', label='Explore the completed original PET Scout study')
        return
    manifest = json.loads(path.read_text())
    config = manifest['config']
    prepared = json.loads((base/'data_manifest.json').read_text())
    requested = json.loads((root/'benchmark_config.json').read_text())
    if config != requested or manifest['data_manifest'] != prepared:
        st.warning('Saved results belong to an earlier configuration or prepared dataset. Rerun `pet-bench all` before comparing models.')
        st.stop()
    data = load_tables(str(base), path.stat().st_mtime_ns)
    st.page_link('pages/2_Experiment_planner.py', label='🧪 New: Can AI find strong enzymes with fewer experiments?')
    task_labels = {task['id']: task['label'] for task in config['tasks']}
    with st.sidebar:
        st.subheader('PETBench v1')
        st.caption('One dataset. Fixed tasks. Shared test pools.')
        task = st.selectbox('Assay task', list(task_labels), format_func=task_labels.get, key='bench_task')
        regime = st.selectbox('Evaluation split', list(REGIMES), format_func=REGIMES.get, key='bench_regime')
        taskrow = data['tasks'].set_index('task').loc[task]
        sizes = str(taskrow.training_sizes).split(';')
        size = st.selectbox('Training examples', sizes, index=sizes.index('all'), key='bench_size')
        budget = st.select_slider('Candidates to select', options=config['selection_budgets'], value=5, key='bench_budget')
        st.caption('All = every example in that fold’s training pool. Each task is learned separately.')
        st.divider()
        st.link_button('Source experiment', 'https://doi.org/10.1021/acscatal.5c03460')
        st.download_button('Download findings', (base/'results/FINDINGS.md').read_text() if (base/'results/FINDINGS.md').exists() else 'Report generation pending.', 'PETBench-findings.md')
        st.caption('Published measurements; no new proteins were made. Experimental data: CC BY-NC 4.0.')
    cols = st.columns(4)
    cols[0].metric('Models and baselines', len(manifest['models']))
    cols[1].metric('Measured proteins · this task', int(taskrow.n_proteins))
    cols[2].metric('Training budgets', len(sizes))
    cols[3].metric('Paid compute', '$0')
    st.caption(f'{task_labels[task]} · crystalline PET powder · 48-hour assay · {int(taskrow.n_groups)} sequence groups. Different tasks contain different measured protein populations.')
    st.info(f'The largest sequence group in this task contains {int(taskrow.largest_group)} proteins. Grouped test pools are uneven; compare each method with baselines on the same pools. Seed ranges are not confidence intervals.')
    ranking, learning, batches, costs, audit = st.tabs(['01  Model comparison','02  Learning curves','03  Choose a batch','04  Compute costs','05  Methods & audit'])
    summary = data['summary']
    current = summary[(summary.task == task)&(summary.regime == regime)&(summary.training_size == size)&(summary.budget == budget)].copy()
    current = current.sort_values('selected_mean', ascending=False)
    with ranking:
        st.subheader(f'What do {budget} selected enzymes achieve?')
        st.write('The score is their average measured PET-breakdown activity, evaluated on proteins held out from fitting. Higher is better.')
        chart = current.copy()
        chart['plus'] = chart.max_seed_lift-chart.lift
        chart['minus'] = chart.lift-chart.min_seed_lift
        fig = px.bar(chart, x='model', y='lift', color='model', color_discrete_map=COLORS,
                     error_y='plus', error_y_minus='minus', labels={'model':'Method','lift':'Activity above random selection (µmol/mg)'}, template='plotly_white')
        fig.add_hline(y=0, line_color='#587168')
        fig.update_layout(showlegend=False, height=380, margin=dict(t=10,b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption('Bars: mean lift above random selection from identical candidate pools. Whiskers: minimum and maximum seed-level averages across five partitions.')
        best = current.iloc[0]
        if best.model == 'Nearest sequence':
            st.write('**Highest mean: nearest sequence.** The simple baseline leads this comparison.')
        elif best.min_seed_delta > 0:
            st.write(f'**Highest mean: {best.model}.** Its selections outperform nearest sequence in all five evaluated partition seeds. These partitions reuse proteins; this is not independent experimental replication.')
        else:
            st.write(f'**Highest mean: {best.model}.** Its advantage over nearest sequence is not positive in every partition seed. The mean alone does not establish a reliable winner.')
        view = current[['model','selected_mean','random_mean','delta_nearest','min_seed_delta','max_seed_delta','spearman','spearman_valid_folds','total_folds']]
        st.dataframe(view, hide_index=True, width='stretch', column_config={c:st.column_config.NumberColumn(c,format='%.2f') for c in view.columns if c!='model'})
        st.caption('These rankings compare frozen representations with the same ridge-training protocol. They do not rank all possible uses of each foundation model.')
        st.download_button('Download this comparison',current.to_csv(index=False),'petbench-comparison.csv','text/csv')
    with learning:
        st.subheader('How much measured data does each method need?')
        curves = summary[(summary.task == task)&(summary.regime == regime)&(summary.budget == budget)].copy()
        fig = px.line(curves, x='training_size', y='delta_nearest', color='model', markers=True,
            category_orders={'training_size':sizes}, color_discrete_map=COLORS,
            labels={'training_size':'Number of training proteins','delta_nearest':'Selected activity minus nearest-sequence baseline (µmol/mg)','model':'Method'}, template='plotly_white')
        fig.add_hline(y=0, line_color='#587168')
        fig.update_xaxes(type='category')
        fig.update_layout(height=420, legend=dict(orientation='h',y=-.25), margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
        st.caption('At each budget all methods see exactly the same labels; smaller training subsets are contained in larger ones. “All” varies by outer fold and is not equally spaced from 64.')
        st.dataframe(curves.pivot(index='model',columns='training_size',values='selected_mean').reindex(columns=sizes).round(2),width='stretch')
        st.caption('Table: mean measured activity of selected batches, µmol/mg. No extra labels from other assay conditions are used.')
    with batches:
        st.subheader('Inspect an actual held-out selection')
        controls = st.columns(3)
        seed = controls[0].selectbox('Partition seed',config['seeds'],key='bench_seed')
        fold = controls[1].selectbox('Fold',list(range(config['n_folds'])),format_func=lambda x:f'Fold {x+1}',key='bench_fold')
        model = controls[2].selectbox('Selection method',manifest['models'],key='bench_model')
        pred = data['predictions']
        pool = pred[(pred.task==task)&(pred.regime==regime)&(pred.training_size==size)&(pred.seed==seed)&(pred.fold==fold)&(pred.model==model)].sort_values(['predicted','sample_id'],ascending=[False,True]).copy()
        chosen = pool.head(budget)
        reveal = st.toggle('Reveal measured results',value=True,key='bench_reveal')
        st.caption(f'{len(pool)} held-out candidates; {int(pool.n_train.iloc[0])} training proteins. Selection uses predictions only.')
        if reveal:
            c = st.columns(3)
            c[0].metric('Chosen batch',f'{chosen.observed.mean():.2f}')
            c[1].metric('Random expectation',f'{pool.observed.mean():.2f}')
            c[2].metric('Hindsight best',f'{pool.observed.nlargest(budget).mean():.2f}')
        cols = ['sample_id','predicted','nearest_train_id','nearest_identity']+(['observed'] if reveal else [])
        st.dataframe(chosen[cols],hide_index=True,width='stretch')
        if reveal:
            pool['Selection'] = np.where(pool.sample_id.isin(chosen.sample_id),'Chosen','Other')
            fig = px.scatter(pool,x='predicted',y='observed',color='Selection',hover_name='sample_id',
                color_discrete_map={'Chosen':'#126b54','Other':'#b8c8c0'},template='plotly_white',
                labels={'predicted':'Predicted activity','observed':'Measured activity (µmol/mg)'})
            fig.update_layout(height=350,margin=dict(t=10))
            st.plotly_chart(fig,width='stretch')
        st.download_button('Download candidate ranking',pool[cols].to_csv(index=False),'petbench-candidates.csv','text/csv')
        st.caption('Individual batches are examples. Use the aggregate comparison for performance claims.')
    with costs:
        st.subheader('What did these models cost to run?')
        compute = data['compute']
        fig = px.bar(compute,x='model',y='inference_seconds',color='model',color_discrete_map=COLORS,
            labels={'model':'Model','inference_seconds':'One-time embedding inference (seconds)'},template='plotly_white')
        fig.update_layout(showlegend=False,height=330,margin=dict(t=10))
        st.plotly_chart(fig,width='stretch')
        st.dataframe(compute.round(3),hide_index=True,width='stretch')
        st.write(f'All models embedded the same {manifest["data_manifest"]["n_unique_proteins"]} unique proteins. Features are reused for every condition and split. Evaluation took {manifest["evaluation_seconds"]:.1f} seconds for {manifest["n_outer_model_fits"]:,} outer method runs.')
        st.caption('One local CPU run per checkpoint, float32, four Torch threads. Loading includes downloads/cache reads. Peak memory is whole-process RSS, not GPU memory. Different batch sizes mean this is practical run accounting, not a controlled throughput benchmark.')
        st.download_button('Download compute measurements',compute.to_csv(index=False),'petbench-compute.csv','text/csv')
    with audit:
        st.subheader('A comparison you can inspect and rerun')
        st.write('Models are trained separately for each assay condition. All sequence-group assignments use the union of proteins across tasks. Replicates and identical sequences are combined before any split.')
        st.markdown('''- Same outer splits and nested training subsets for every method.
- Same ridge alpha grid and up to three inner folds for training-only tuning.
- Frozen embeddings with pinned checkpoint revisions and residue-only pooling.
- Source, feature, split and prediction checksums; missing assays stay missing.
- Equal weight per candidate-selection batch, then per partition seed.
- Random, size-matched random and grouped evaluations reported separately.''')
        st.warning('This is a retrospective extension of public experimental data, including a condition already examined in PET Scout. It is not a blind prospective test, a mutation-effect benchmark, or proof of industrial recycling performance.')
        for limitation in manifest['limitations']: st.caption(limitation)
        st.dataframe(data['tasks'],hide_index=True,width='stretch')
        selected_splits=data['splits'][(data['splits'].task==task)&(data['splits'].regime==regime)&(data['splits'].training_size==size)]
        st.dataframe(selected_splits.drop(columns=['train_ids','test_ids']),hide_index=True,width='stretch')
        with st.expander('Run configuration and provenance'): st.json(manifest)
        st.download_button('Download split membership',data['splits'].to_csv(index=False),'petbench-splits.csv','text/csv')
        st.download_button('Download complete summary',summary.to_csv(index=False),'petbench-summary.csv','text/csv')
        st.page_link('pages/1_Original_study.py',label='Open the original PET Scout study')

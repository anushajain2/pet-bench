"""Read-only screening-campaign dashboard and auditable batch replay."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .campaign_report import verified_results, COLORS


def render(root):
    st.set_page_config(page_title='PETBench | Experiment planner',page_icon='🧪',layout='wide')
    st.markdown('<style>.block-container{max-width:1300px;padding-top:3rem}h1{letter-spacing:-1px}</style>',unsafe_allow_html=True)
    st.caption('PETBENCH / EXPERIMENT SELECTION')
    st.title('Can AI help a lab find better enzymes with fewer tests?')
    st.write('Replay published PET-degradation experiments. Every strategy starts with the same 16 measured enzymes, then chooses eight more at a time.')
    if not (root/'campaign/results/manifest.json').exists():
        st.info('Run `.venv/bin/python -m pet_scout.campaign` to create the simulation results.')
        return
    try:
        manifest, tables = verified_results(root)
    except ValueError as error:
        st.error(str(error))
        return
    config = manifest['config']
    st.page_link('pages/3_Validation_&_case_study.py',label='Explore the harder starting-set test, ablations and case study')
    with st.sidebar:
        task = st.selectbox('Assay condition',list(manifest['task_labels']),format_func=manifest['task_labels'].get,key='campaign_task')
        budget = st.select_slider('Total experimental budget',options=list(range(config['initial_size'],config['max_budget']+1,config['batch_size'])),value=config['primary_budget'],key='campaign_budget')
        st.caption('Budgets include the 16 starting assays. The primary comparison was fixed at 40; 64 is secondary.')
        st.download_button('Download findings',(root/'campaign/results/FINDINGS.md').read_text(),'experiment-selection-findings.md')
        st.page_link('app.py',label='Model benchmark')
    summary = tables['summary'][tables['summary'].task.eq(task)]
    checkpoints = tables['checkpoints'][tables['checkpoints'].task.eq(task)]
    primary = summary[summary.tested.eq(budget)]
    threshold = checkpoints.threshold.iloc[0]
    st.info(f'“Strong” = measured activity ≥ {threshold:.2f} µmol/mg, the 90th percentile for this condition. This threshold is used only to score discoveries; the selection algorithms never see it.')
    cols = st.columns(4)
    for column,strategy in zip(cols,config['strategies']):
        row = primary[primary.strategy.eq(strategy)].iloc[0]
        column.metric(strategy,f'{row.mean_hits:.2f} found',delta=None if strategy=='Random' else f'{row.delta_random:+.2f} vs random',delta_color='normal')
    st.caption(f'Mean strong enzymes found after {budget} total tests, including starting discoveries; {config["n_seeds"]} paired starting sets. These reuse the same proteins, not independent lab replications.')
    curves, replay, audit = st.tabs(['Discovery curves','Replay a campaign','Methods & audit'])
    with curves:
        st.subheader('How quickly do strong enzymes get found?')
        band = st.toggle('Show 10th–90th percentile spread across starting sets',value=False,key='campaign_bands')
        fig = go.Figure()
        for strategy,color in COLORS.items():
            frame = summary[summary.strategy.eq(strategy)].sort_values('tested')
            if band:
                fig.add_trace(go.Scatter(x=frame.tested,y=frame.p90_hits,mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=frame.tested,y=frame.p10_hits,mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba('+','.join(str(int(color[i:i+2],16)) for i in (1,3,5))+',0.10)',showlegend=False,hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=frame.tested,y=frame.mean_hits,mode='lines+markers',name=strategy,line=dict(color=color)))
        fig.add_vline(x=config['primary_budget'],line_dash='dash',line_color='#bbb')
        fig.update_layout(template='plotly_white',xaxis_title='Total enzymes tested',yaxis_title='Strong enzymes found (mean)',height=460,legend=dict(orientation='h',y=-.26,yanchor='top'),margin=dict(t=15,b=110))
        st.plotly_chart(fig,width='stretch')
        st.caption('Shaded ranges, when enabled, show simulation sensitivity to the starting set. They are not confidence intervals. Dashed line: the fixed 40-assay primary endpoint.')
        fig = px.line(summary,x='tested',y='target_success_rate',color='strategy',markers=True,color_discrete_map=COLORS,template='plotly_white',labels={'tested':'Total enzymes tested','target_success_rate':'Campaigns finding ≥5 strong enzymes','strategy':'Strategy'})
        fig.update_yaxes(tickformat='.0%',range=[0,1.03])
        fig.update_layout(height=400,legend=dict(orientation='h',y=-.3,yanchor='top',title_text=''),margin=dict(t=15,b=110))
        st.plotly_chart(fig,width='stretch')
        st.dataframe(primary[['strategy','mean_hits','mean_new_hits','delta_random','delta_nearest','delta_greedy','win_rate_random','tie_rate_random','target_success_rate']].round(3),hide_index=True,width='stretch')
        st.caption('Win rate = fraction of paired starts with strictly more hits than random; ties are shown separately. New hits exclude the shared starting samples. Δ greedy tests the combined uncertainty/diversity policy against prediction-only selection.')
        times = tables['target_times'][tables['target_times'].task.eq(task)].groupby('strategy').agg(reached_by_64=('reached_by_max','mean'),mean_capped_assays=('capped_assays','mean')).reset_index()
        with st.expander('Assays needed to find five: include unsuccessful campaigns'):
            st.dataframe(times.round(3),hide_index=True,width='stretch')
            st.caption('Runs that never reach five are assigned 64 in the capped mean. This is a budget-truncated measure, not eventual time to success. Targets are checked only when a full batch returns.')
    with replay:
        c = st.columns(3)
        seed = c[0].selectbox('Starting-set seed',sorted(checkpoints.seed.unique()),key='campaign_seed')
        strategy = c[1].selectbox('Selection strategy',config['strategies'],index=2,key='campaign_strategy')
        batch = c[2].selectbox('Completed batch',list(range((config['max_budget']-config['initial_size'])//config['batch_size']+1)),key='campaign_batch')
        selections = tables['selections']
        run = selections[(selections.task==task)&(selections.seed==seed)&(selections.strategy==strategy)]
        current = run[run.batch.eq(batch)]
        st.write(f'Batch {batch}: {len(current)} enzymes. All choices were made before any outcome in this batch was revealed.')
        reveal = st.toggle('Reveal this batch’s measured outcomes',value=False,key='campaign_reveal')
        columns = ['sample_id','predicted_log1p','bootstrap_std_log1p','acquisition_score','diversity_distance']
        if reveal: columns += ['activity','hit']
        st.dataframe(current[columns].round(4),hide_index=True,width='stretch')
        st.caption('AI predictions and spread use log(1 + activity). Random and nearest-sequence policies have no model uncertainty. Acquisition scores have different meanings across policies and should not be compared numerically.')
        previous = run[run.batch.lt(batch)]
        with st.expander(f'Previously revealed samples ({len(previous)})'):
            st.dataframe(previous[['batch','sample_id','activity','hit']],hide_index=True,width='stretch')
        st.download_button('Download displayed batch',current[columns].to_csv(index=False),'campaign-batch.csv','text/csv')
    with audit:
        st.warning('Exploratory replay of previously inspected published measurements. No new enzyme was made, and uncertainty is not calibrated. The simulation does not establish performance on unseen enzyme families or a new laboratory study.')
        st.markdown('''- Four policies share each initial sample set, budget, candidate pool and assay outcomes.
- ESM-2 35M features are frozen and cached. Both AI policies use the same 20-member bootstrap ridge predictor, alpha 100, with scaling fitted only on revealed samples.
- AI explore combines 75% rank of mean + one bootstrap standard deviation with 25% rank of cosine diversity. Weights were fixed before campaign runs.
- The full batch is selected before its measurements are revealed. Every revealed assay counts toward the budget.
- The top-decile threshold is an evaluator-only, retrospective target; ties count as hits.
- All assays have equal simulated cost. Experimental noise and expression failures are not simulated.
- This pool contains previously screened, expressed natural enzyme constructs. It is not a mutation-design benchmark.
- A failed campaign remains in all success-rate calculations; it is never dropped to make savings look better.''')
        st.write(f'{manifest["n_campaigns"]} campaigns · {manifest["simulation_seconds"]:.1f} seconds simulation · $0 paid compute · reused embeddings.')
        st.link_button('Source study','https://doi.org/10.1021/acscatal.5c03460')
        st.caption('Source experimental data: CC BY-NC 4.0. Model selection was informed by the earlier benchmark; all reported findings are exploratory.')
        for name in ['summary','checkpoints','selections','paired','target_times']:
            st.download_button(f'Download {name}',tables[name].to_csv(index=False),f'campaign-{name}.csv','text/csv')
        with st.expander('Configuration and provenance'): st.json(manifest)

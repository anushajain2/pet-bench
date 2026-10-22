"""Robustness, acquisition components and finished case study."""
import pandas as pd
import plotly.express as px
import streamlit as st
from .validation_report import verified, diagnostic_summary, COLORS, SCENARIOS


def render(root):
    st.set_page_config(page_title='PETBench | Validation & case study',page_icon='🧬',layout='wide')
    st.markdown('<style>.block-container{max-width:1300px;padding-top:3rem}h1{letter-spacing:-1px}</style>',unsafe_allow_html=True)
    st.caption('PETBENCH / ROBUSTNESS AND FAILURE ANALYSIS')
    st.title('Does AI help beyond familiar enzymes?')
    st.write('Start with measurements from one sequence group. Spend the next 24 assays searching outside it. Then separate the effects of uncertainty and diversity.')
    base=root/'validation/results'
    if not (base/'manifest.json').exists():
        st.info('Run `.venv/bin/python -m pet_scout.validation` to generate results.');return
    try: manifest=verified(root)
    except ValueError as error:
        st.error(str(error));return
    summary=pd.read_csv(base/'summary.csv'); anchors=pd.read_csv(base/'anchor_summary.csv')
    design=pd.read_csv(base/'design.csv'); config=manifest['config']
    with st.sidebar:
        task=st.selectbox('Assay condition',list(manifest['task_labels']),format_func=manifest['task_labels'].get,key='validation_task')
        scenario=st.selectbox('Starting scenario',list(SCENARIOS),index=1,format_func=SCENARIOS.get,key='validation_scenario')
        available=sorted(design[(design.task==task)&(design.scenario==scenario)].anchor.unique())
        anchor=st.selectbox('Starting group',['all',*available] if scenario=='group_shift' else ['all'],
                            format_func=lambda a:'Both groups · equal weight' if a=='all' and scenario=='group_shift' else ('All random starts' if a=='all' else f'Group {a}'),key='validation_anchor')
        st.caption('Primary budget: 40 total tests = 16 starting measurements + 24 acquired assays. No acquisition settings were retuned.')
        st.download_button('Download case study',(base/'CASE_STUDY.md').read_text(),'PETBench-case-study.md')
        st.download_button('Download 2-minute demo script',(base/'DEMO.md').read_text(),'PETBench-demo.md')
    current=summary[(summary.task==task)&(summary.scenario==scenario)]
    if anchor!='all': current=anchors[(anchors.task==task)&(anchors.scenario==scenario)&(anchors.anchor==anchor)]
    primary=current[current.tested.eq(config['primary_budget'])]
    st.info('Scores here count NEW strong enzymes, excluding starting discoveries. “Strong” keeps the original condition’s top-decile threshold. Groups are sequence-identity clusters, not established biological families.')
    robustness,ablations,diagnostics,case=st.tabs(['Harder starting sets','Acquisition ablations','Uncertainty diagnostic','Case study & audit'])
    def style(fig):
        fig.update_layout(height=460,legend=dict(orientation='h',y=-.27,yanchor='top',title_text=''),margin=dict(t=15,b=130),template='plotly_white')
        return fig
    with robustness:
        label=manifest['task_labels'][task]
        st.subheader(f'{label}: what do 24 new tests find?')
        metric=st.radio('Discovery measure',['new_hits','new_hit_groups'],format_func=lambda m:'New strong enzymes' if m=='new_hits' else 'Groups containing new strong enzymes',horizontal=True,key='validation_metric')
        fig=px.bar(primary,x='strategy',y=metric,color='strategy',color_discrete_map=COLORS,
                   labels={'strategy':'Selection policy',metric:'Mean discoveries at 40 total assays'})
        fig.update_layout(showlegend=False,height=370,margin=dict(t=15,b=90),template='plotly_white')
        st.plotly_chart(fig,width='stretch')
        st.dataframe(primary[['strategy','new_hits','new_hit_groups','unfamiliar_hit_groups','delta_random','delta_nearest','delta_greedy']].round(3),hide_index=True,width='stretch')
        fig=px.line(current,x='tested',y=metric,color='strategy',markers=True,color_discrete_map=COLORS,
            labels={'tested':'Total assays (including 16 initial)',metric:'New discoveries','strategy':'Policy'})
        fig.add_vline(x=config['primary_budget'],line_dash='dash',line_color='#999')
        st.plotly_chart(style(fig),width='stretch')
        st.caption('Compare policies within a scenario. Shifted and random-start searches have different candidate pools and hit prevalence. After each batch, models can learn from new-group measurements; this is not permanent group holdout.')
        if scenario=='group_shift':
            table=anchors[(anchors.task==task)&(anchors.scenario==scenario)&(anchors.tested==config['primary_budget'])]
            st.subheader('Does the starting group change the result?')
            st.dataframe(table[['anchor','strategy','new_hits','new_hit_groups','delta_nearest']].round(3),hide_index=True,width='stretch')
        st.dataframe(design[(design.task==task)&(design.scenario==scenario)],hide_index=True,width='stretch')
    with ablations:
        st.subheader('Which addition helps or hurts?')
        st.write('The same prediction model is used in all four AI policies. Each bar is the paired difference from prediction-only AI at 40 total tests.')
        frame=primary[primary.strategy.isin(['AI uncertainty','AI diversity','AI explore'])]
        fig=px.bar(frame,x='strategy',y='delta_greedy',color='strategy',color_discrete_map=COLORS,
                   labels={'strategy':'Addition to prediction-only AI','delta_greedy':'Change in new strong enzymes'})
        fig.add_hline(y=0,line_color='#555');fig.update_layout(showlegend=False,height=350,template='plotly_white')
        st.plotly_chart(fig,width='stretch')
        st.dataframe(frame[['strategy','delta_greedy','group_delta_greedy']].round(3),hide_index=True,width='stretch')
        st.markdown('''- **AI uncertainty:** predicted mean + one bootstrap standard deviation.
- **AI diversity:** 75% rank of predicted mean + 25% rank of distance from observed/already-chosen sequences.
- **AI explore:** the original combination of both additions.

These are fixed settings. Results do not establish that every uncertainty or diversity method behaves the same way.''')
        effects=pd.read_csv(base/'effects.csv')
        effects=effects[(effects.task==task)&(effects.scenario==scenario)&(effects.tested==config['primary_budget'])]
        if anchor!='all': effects=effects[effects.anchor.eq(anchor)]
        st.caption('Interaction = combined − uncertainty − diversity + greedy. Nonzero values mean component effects are not additive.')
        st.dataframe(effects.groupby('metric')[['uncertainty_added','diversity_added','combined_added','interaction']].mean().round(3),width='stretch')
    with diagnostics:
        st.subheader('Does “uncertain” actually mean a larger prediction error?')
        st.write('This analysis uses every untested candidate at the first shared decision. Its outcomes are withheld from the selection policies.')
        runs=pd.read_csv(base/'diagnostics.csv')
        runs=runs[(runs.task==task)&(runs.scenario==scenario)]
        if anchor!='all': runs=runs[runs.anchor.eq(anchor)]
        result=diagnostic_summary(runs)
        row=result.iloc[0]
        c=st.columns(2)
        c[0].metric('Uncertainty/error rank correlation',f'{row.uncertainty_error_spearman:.2f}')
        c[1].metric('High minus low uncertainty error',f'{row.high_minus_low_error:+.2f}')
        st.caption(f'{int(row.valid_correlations)}/{int(row.n_starts)} defined within-start correlations. Positive correlation means larger uncertainty scores tend to accompany larger errors; it does not establish calibrated intervals or better acquisition.')
        predictions=pd.read_csv(base/'initial_predictions.csv')
        predictions=predictions[(predictions.task==task)&(predictions.scenario==scenario)]
        if anchor!='all': predictions=predictions[predictions.anchor.eq(anchor)]
        predictions=predictions[predictions.uncertainty_quartile.gt(0)]
        q=predictions.groupby(['anchor','seed','uncertainty_quartile']).abs_error_log1p.mean().reset_index()
        q=q.groupby(['anchor','uncertainty_quartile']).abs_error_log1p.mean().reset_index()
        q=q.groupby('uncertainty_quartile').abs_error_log1p.mean().reset_index()
        fig=px.line(q,x='uncertainty_quartile',y='abs_error_log1p',markers=True,template='plotly_white',
                    labels={'uncertainty_quartile':'Uncertainty quartile (1 = lowest)','abs_error_log1p':'Mean absolute log-activity error'})
        fig.update_xaxes(dtick=1);fig.update_layout(height=350,margin=dict(t=15))
        st.plotly_chart(fig,width='stretch')
        st.caption('Runs with constant uncertainty scores have no ranking and are excluded from quartile comparisons. Eligible runs contribute equally within each starting group; group means are then averaged. Repeated candidate predictions are not independent observations. This diagnostic alone cannot explain later adaptive trajectories.')
    with case:
        st.write(f'{manifest["n_campaigns"]:,} campaigns · {manifest["simulation_seconds"]:.1f} seconds for simulation/export · $0 paid compute · existing embeddings reused.')
        st.warning('Exploratory extension of previously inspected data. Only two starting groups per condition are eligible. No new enzyme, external study or prospective laboratory result is claimed.')
        with st.expander('Read the finished case study'): st.markdown((base/'CASE_STUDY.md').read_text())
        with st.expander('Two-minute walkthrough'): st.markdown((base/'DEMO.md').read_text())
        with st.expander('Protocol'): st.markdown((root/'VALIDATION_STUDY.md').read_text())
        with st.expander('Configuration and provenance'): st.json(manifest)
        for name in ['summary','anchor_summary','effects','diagnostics','design']:
            st.download_button(f'Download {name}',(base/f'{name}.csv').read_bytes(),f'validation-{name}.csv','text/csv')
        st.caption('Published data: CC BY-NC 4.0. Source DOI: 10.1021/acscatal.5c03460. Original benchmark and campaign outputs are preserved.')

"""Read-only explorer of completed, genuinely held-out experimental predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='PET Scout | Evidence for enzyme selection',page_icon='🧬',layout='wide')
st.markdown('''<style>
.block-container {max-width:1280px;padding-top:2.2rem;}
h1 {letter-spacing:-1.5px;font-size:3rem!important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #dbe5dc;border-radius:12px;padding:18px;}
[data-testid="stMetricLabel"] {color:#587168;}
[data-testid="stSidebar"] {border-right:1px solid #dbe5dc;}
.eyebrow {letter-spacing:2px;font-size:12px;font-weight:700;color:#126b54;}
.hero-copy {font-size:18px;max-width:850px;line-height:1.6;color:#4c675e;}
</style>''',unsafe_allow_html=True)

@st.cache_data
def load_results(fingerprint):
    report=ROOT/'reports'
    return (pd.read_csv(report/'predictions.csv'),pd.read_csv(report/'summary.csv'),
        pd.read_csv(report/'seed_metrics.csv'),pd.read_csv(ROOT/'data/processed/candidates.csv'),
        pd.read_csv(report/'split_audit.csv'),pd.read_csv(report/'paired_comparison.csv'))

manifest_path=ROOT/'reports/run_manifest.json'
if not manifest_path.exists():
    st.title('PET Scout')
    st.info('Run `pet-scout all` from the project folder to generate real experimental results. No demonstration data is substituted.')
    st.stop()
manifest=json.loads(manifest_path.read_text())
pred,summary,seeds,candidates,splits,paired=load_results(manifest_path.stat().st_mtime_ns)
audit=json.loads((ROOT/'data/processed/audit.json').read_text())
LABELS={'random':'Random holdout','matched':'Size-matched random','grouped':'Sequence-group holdout'}
COLORS={'Nearest sequence':'#90a6a0','Composition ridge':'#dca557','ESM-2 ridge':'#126b54'}

with st.sidebar:
    st.markdown('## PET Scout')
    st.caption('A small study. An auditable result.')
    st.divider()
    st.markdown('**The question**')
    st.write('Can AI choose useful plastic-degrading enzymes when close relatives are absent from training?')
    st.markdown('**One experimental condition**')
    st.write('60°C · pH 7.5\n\nCrystalline PET powder\n\n48-hour reaction')
    st.caption('Published measurements; retrospective evaluation. No new enzymes were made.')
    st.divider()
    st.link_button('Read the source study','https://doi.org/10.1021/acscatal.5c03460')
    st.download_button('Download findings',data=(ROOT/'reports/FINDINGS.md').read_text() if (ROOT/'reports/FINDINGS.md').exists() else 'Run pet-scout report.',file_name='PET-Scout-Findings.md',mime='text/markdown')

st.markdown('<div class="eyebrow">PROTEIN ENGINEERING / EVIDENCE BEFORE EXPERIMENTS</div>',unsafe_allow_html=True)
st.title('Better choices for plastic recycling?')
st.markdown('<p class="hero-copy">A protein language model proposes which enzymes to test. Published experiments reveal whether those choices actually break down more PET.</p>',unsafe_allow_html=True)
a,b,c,d=st.columns(4)
a.metric('Measured proteins',len(candidates))
b.metric('Sequence groups',manifest['n_groups'])
c.metric('Held-out folds',manifest['n_outer_fits'])
d.metric('Protein model','ESM-2 · 35M')
st.caption('Sequence groups connect proteins at ≥60% global alignment identity. Frozen model; only the small regression model is trained on assay results.')

st.info(f"The largest sequence group contains {manifest['largest_group']} of {len(candidates)} proteins. Grouped test pools are uneven. A size-matched random diagnostic helps assess the effect of smaller training sets.")

benchmark,selection,evidence,methods=st.tabs(['01  The finding','02  Choose a test batch','03  Explore the evidence','04  Methods & limits'])
with benchmark:
    st.subheader('Does the advantage survive a harder test?')
    st.write('Random splits can place close relatives on both sides. The grouped split keeps those relatives together, testing how predictions transfer to less familiar proteins.')
    budget=st.select_slider('Candidates selected per held-out fold',options=manifest['config']['budgets'],value=5,key='overview_budget')
    selected=seeds[seeds.budget.eq(budget)].copy();selected['Split']=selected.regime.map(LABELS)
    plot=selected.groupby(['Split','model']).lift_over_random.agg(['mean','min','max']).reset_index()
    plot['upper']=plot['max']-plot['mean'];plot['lower']=plot['mean']-plot['min']
    fig=px.bar(plot,x='Split',y='mean',color='model',barmode='group',color_discrete_map=COLORS,
        error_y='upper',error_y_minus='lower',labels={'mean':'Activity above random selection (µmol/mg)','model':'Method'},template='plotly_white',category_orders={'Split':['Random holdout','Size-matched random','Sequence-group holdout']})
    fig.add_hline(y=0,line_color='#687e75');fig.update_layout(height=390,legend_title_text='',margin=dict(t=10))
    st.plotly_chart(fig,width='stretch')
    st.caption('Bars: mean gain over random selection within the same test pools. Whiskers: range of seed-level averages, not confidence intervals. Each fold gets its own selection budget.')
    diff=paired[(paired.regime=='grouped')&(paired.budget==budget)].esm_minus_nearest
    st.metric('ESM vs nearest sequence · grouped holdout',f'{diff.mean():+.2f} µmol/mg',help='Difference in measured activity of selected batches, averaged over folds then seeds.')
    if diff.min()>0: st.success('ESM selected better batches in every evaluated grouped partition. This is evidence for this dataset and protocol, not universal performance.')
    elif diff.max()<0: st.warning('The nearest-sequence baseline selected better batches in every evaluated grouped partition. This model does not establish an AI selection advantage here.')
    else: st.info('The advantage changes across grouped partitions. This dataset does not establish a robust ESM advantage over sequence similarity.')
    view=summary[summary.budget.eq(budget)][['regime','model','mean_selected_activity','mean_random_activity','mean_opportunity_captured','mean_spearman']].copy()
    view.regime=view.regime.map(LABELS)
    st.dataframe(view.rename(columns={'regime':'Split','model':'Method','mean_selected_activity':'Selected mean (µmol/mg)','mean_random_activity':'Random mean (µmol/mg)','mean_opportunity_captured':'Opportunity captured','mean_spearman':'Mean fold Spearman'}),hide_index=True,width='stretch')

with selection:
    st.subheader('Which five would you test?')
    st.write('Choose one held-out candidate pool. The ranking was generated without using that pool’s activity measurements.')
    controls=st.columns(4)
    regime=controls[0].selectbox('Evaluation split',['grouped','random','matched'],format_func=LABELS.get)
    seed=controls[1].selectbox('Partition seed',manifest['config']['seeds'])
    fold=controls[2].selectbox('Held-out fold',list(range(manifest['config']['n_folds'])),format_func=lambda x:f'Fold {x+1}')
    model=controls[3].selectbox('Selection method',['ESM-2 ridge','Nearest sequence','Composition ridge'])
    pool=pred[(pred.regime==regime)&(pred.seed==seed)&(pred.fold==fold)&(pred.model==model)].copy()
    k=st.select_slider('Testing budget',options=[v for v in manifest['config']['budgets'] if v<=len(pool)],value=min(5,len(pool)),key='batch_budget')
    ranked=pool.sort_values(['predicted','sample_id'],ascending=[False,True]);chosen=ranked.head(k)
    reveal=st.toggle('Reveal published experimental results',value=True)
    st.caption(f'{len(pool)} candidates in this test pool. Predictions are rankings, not probabilities. Units are µmol aromatic products per mg enzyme after 48 hours.')
    if reveal:
        a,b,c=st.columns(3)
        a.metric('Selected batch · measured mean',f'{chosen.observed.mean():.2f}')
        b.metric('Random batch · expected mean',f'{pool.observed.mean():.2f}')
        c.metric('Best possible batch · hindsight',f'{pool.observed.nlargest(k).mean():.2f}')
    cols=['sample_id','predicted','nearest_train_id','nearest_identity']+(['observed'] if reveal else [])
    st.dataframe(chosen[cols].rename(columns={'sample_id':'Candidate','predicted':'Predicted activity','nearest_train_id':'Closest training candidate','nearest_identity':'Sequence identity','observed':'Measured activity'}),hide_index=True,width='stretch')
    if reveal:
        plot=ranked.copy();plot['Selected']=np.where(plot.sample_id.isin(chosen.sample_id),'Selected','Other candidates')
        fig=px.scatter(plot,x='predicted',y='observed',color='Selected',hover_name='sample_id',hover_data=['nearest_identity','nearest_train_id'],color_discrete_map={'Selected':'#126b54','Other candidates':'#c2cfc7'},labels={'predicted':'Predicted activity','observed':'Measured activity (µmol/mg)'},template='plotly_white')
        fig.update_layout(height=360,margin=dict(t=10));st.plotly_chart(fig,width='stretch')
    st.download_button('Download this held-out ranking',ranked[cols].to_csv(index=False),f'pet-scout-{regime}-{seed}-{fold}-{model}.csv','text/csv')
    st.caption('Browsing different folds is exploratory. Use the complete aggregate evaluation above for claims about performance.')

with evidence:
    st.subheader('How unfamiliar are the test proteins?')
    s=st.selectbox('Partition to inspect',manifest['config']['seeds'],key='evidence_seed')
    inspect=pred[(pred.seed==s)&(pred.model=='ESM-2 ridge')].copy();inspect['Split']=inspect.regime.map(LABELS)
    inspect['Absolute log error']=abs(np.log1p(inspect.predicted)-np.log1p(inspect.observed))
    fig=px.scatter(inspect,x='nearest_identity',y='Absolute log error',color='Split',hover_name='sample_id',hover_data=['observed','predicted','nearest_train_id'],color_discrete_sequence=['#dca557','#8091b5','#126b54'],template='plotly_white',labels={'nearest_identity':'Identity to closest training sequence'})
    fig.add_vline(x=.6,line_dash='dot',line_color='#647a6c');fig.update_layout(height=370,margin=dict(t=10));st.plotly_chart(fig,width='stretch')
    st.caption('Grouped holdouts must stay below the 0.60 identity threshold. This scatter is descriptive; low similarity alone does not establish why a prediction failed.')
    candidate=st.selectbox('Inspect a measured protein',candidates.sample_id.tolist())
    row=candidates.set_index('sample_id').loc[candidate]
    a,b,c=st.columns(3);a.metric('Measured activity',f'{row.activity:.2f} µmol/mg');b.metric('Assay measurements',int(row.n_replicates));c.metric('Sequence length',int(row.length))
    st.write(f'**Source aliases:** {row.aliases} · **Screening round:** {row["round"]} · **Accession:** {row.accession}')
    if pd.notna(row.tm_1): st.caption(f'Source thermal transition tm_1: {row.tm_1:.1f}°C. Descriptive only; not used as a sequence-model input or equated with operating lifetime.')
    st.code(row.sequence,language=None,wrap_lines=True)
    st.dataframe(pred[(pred.sample_id==candidate)&(pred.seed==s)][['regime','model','observed','predicted','nearest_train_id','nearest_identity']],hide_index=True,width='stretch')

with methods:
    st.subheader('What this study can establish')
    st.write('A reproducible comparison of candidate-selection methods for one published assay. It cannot establish a new enzyme, industrial recycling performance, or the benefit of a proposed mutation.')
    st.markdown('''**Controls against misleading results**

- Average experimental replicates and collapse identical sequences before splitting.
- Fit normalization and tune regression only within the outer training data.
- Keep connected sequence groups intact in outer and inner grouped splits.
- Score every method on exactly the same held-out pool within a split.
- Treat unmeasured activity as missing, never as zero.
- Record source checksums, model revision, split membership, and cached feature provenance.

**Limits**

One 86-protein group makes test pools unequal. The size-matched random diagnostic
was added after the initial run exposed this imbalance; it preserves fold sizes,
not candidate composition. Model settings and the primary condition were unchanged.

The source candidates were already selected through an ML-assisted screening process.
Not all proteins expressed or were measured at every condition. Full-sequence identity
can miss similarities within individual domains. ESM may have seen these sequences or
homologs during pretraining. Repeated splits reuse proteins and are not independent trials.
The original paper already explored ML prediction; this project makes no novelty claim.
''')
    st.dataframe(splits[['regime','seed','fold','n_train','n_test','shared_groups','max_cross_identity']],hide_index=True,width='stretch')
    with st.expander('Data preparation audit'): st.json(audit)
    with st.expander('Run configuration and model provenance'): st.json(manifest)
    st.download_button('Download all held-out predictions',pred.to_csv(index=False),'pet-scout-predictions.csv','text/csv')
    st.caption('Experimental data: Norton-Baker et al. (2025), ACS Catalysis. Source supplement: CC BY-NC 4.0. Project code: MIT.')

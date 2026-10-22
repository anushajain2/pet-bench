"""Generate a findings report and shareable figure from actual saved results."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np


def md_table(frame):
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def build_report(root: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'reports'
    manifest=json.loads((out/'run_manifest.json').read_text())
    audit=json.loads((root/'data/processed/audit.json').read_text())
    summary=pd.read_csv(out/'summary.csv'); seeds=pd.read_csv(out/'seed_metrics.csv')
    paired=pd.read_csv(out/'paired_comparison.csv'); splits=pd.read_csv(out/'split_audit.csv')
    budget=5; rows=summary[summary.budget.eq(budget)].copy()
    rows['regime']=rows.regime.map({'random':'Random holdout','matched':'Size-matched random','grouped':'Sequence-group holdout'})
    table=rows[['regime','model','mean_selected_activity','mean_random_activity','mean_lift','mean_opportunity_captured']].copy()
    table.columns=['Split','Method','Selected activity','Random expectation','Lift','Opportunity captured']
    for c in table.columns[2:]: table[c]=table[c].map(lambda v:f'{v:.2f}')
    conclusions=[]
    for regime in ['random','matched','grouped']:
        d=paired.loc[paired.regime.eq(regime)&paired.budget.eq(budget),'esm_minus_nearest']
        conclusions.append(f'- **{regime.capitalize()} split:** ESM minus nearest-sequence top-{budget} mean activity = **{d.mean():+.2f} µmol/mg**, with seed-level differences from {d.min():+.2f} to {d.max():+.2f}.')
    g=paired.loc[paired.regime.eq('grouped')&paired.budget.eq(budget),'esm_minus_nearest']
    if g.min()>0:
        verdict='ESM improved candidate selection over the similarity baseline in every evaluated grouped partition. This is evidence for this dataset and protocol, not a guarantee on unseen enzyme families.'
    elif g.max()<0:
        verdict='ESM did not improve candidate selection over the similarity baseline in any evaluated grouped partition. These results do not justify preferring this ESM representation and regressor for this selection task.'
    else:
        verdict='The ESM advantage changed sign across grouped partitions. This small study does not establish a robust selection advantage over sequence similarity.'
    max_group=splits.loc[splits.regime.eq('grouped'),'max_cross_identity'].max()
    text=f'''# PET Scout — experimental findings

## Question and result

Can frozen protein-language-model embeddings help select PET-degrading enzymes,
beyond choosing the nearest known sequence, when close relatives are held out?

{verdict}

{chr(10).join(conclusions)}

These are completed retrospective experiments on published measurements. No enzyme
was synthesized, mutated, or newly validated. The source authors already investigated
ML prediction; PET Scout is a focused reproducible extension, not a discovery claim.

## Data

- Source: Norton-Baker et al., *ACS Catalysis* (2025), https://doi.org/10.1021/acscatal.5c03460.
- Condition: crystalline PET powder, 60°C, pH 7.5, 48 hours; 3.5 mg PET and 5 µg enzyme per 0.5 mL reaction.
- Outcome: released aromatic products, µmol per mg enzyme, as reported in the source workbook.
- {audit['unique_sequences']} unique construct sequences from {audit['selected_assay_rows']} assay rows.
- {audit['single_measurement_sequences']} sequences have only one measurement in this condition.
- {audit['duplicate_sequences_collapsed']} identical-sequence aliases collapsed before splitting.
- Ambiguous sequence IDs excluded: {', '.join(audit['invalid_sequence_ids_excluded']) or 'none'}.
- Missing assays and failed expression are never treated as inactive enzymes.
- Condition fixed in config before model evaluation; no favorable-condition search.

## Top-five selection results

Each number averages the mean measured activity of five selections **within an
outer held-out fold**, first over folds and then over seeds. Fold pools differ
between split regimes; compare against their own random expectation. The five-fold
protocol spends five selections per fold, not five selections for the whole dataset.

{md_table(table)}

Opportunity captured = (selected mean − random expected mean) /
(oracle top-five mean − random expected mean). It can be negative. It is undefined
for pools with no activity variation and is omitted for those pools.

## Reproducible protocol

1. Average source replicates per candidate ID, then average identical-sequence ID means.
2. Compute global BLOSUM62 alignments, gap-open −10 and gap-extension −0.5.
   Identity = identical residues / all alignment columns, including gaps.
3. Build connected components of sequence pairs with identity ≥60%.
   There are {manifest['n_groups']} groups; the largest contains {manifest['largest_group']} sequences.
4. Compare five-fold random holdouts with size-balanced group holdouts, using
   seeds {manifest['config']['seeds']}. Each candidate is tested once per seed.
5. Freeze ESM-2 35M; mean-pool amino-acid tokens, excluding special and padding tokens.
   Compare ESM ridge, amino-acid-composition ridge, and nearest-sequence prediction.
6. Fit log1p(activity) with ridge regression; choose alpha from
   {manifest['config']['ridge_alphas']} using training-only inner validation.
   Fit scaling inside each training fold. Inner validation respects the outer grouping policy.
7. Select by model prediction, using candidate ID only to break equal-score ties.
   Score selections on held-out observed activity. Report exact random expectations
   and Monte Carlo random batch ranges for the same pools.

Largest observed cross-split identity in grouped evaluation: {max_group:.3f}.
Sequence grouping is label-independent and transitive. It does not prove that
proteins belong to different functional families or share no conserved domains.

## Fold-size diagnostic

The largest connected component contains {manifest['largest_group']} of {manifest['n_candidates']} proteins,
so grouped folds have unequal test pools and training-set sizes. After observing
this in the initial run, we added a **size-matched random** diagnostic: the same
train/test sizes as each grouped fold, but randomly assigned proteins. It is an
exploratory control, not a new prespecified primary analysis. Models, condition,
threshold and budgets were unchanged. Candidate composition still differs, so
this does not isolate sequence novelty as a causal effect.

## Uncertainty and interpretation

Seed ranges describe sensitivity to partitions. They are **not confidence intervals**:
all seeds reuse the same proteins, so they are not independent experimental replications.
Monte Carlo ranges describe random selection of candidates, not model confidence.
Experimental replicate variation is recorded but not propagated through all models.

The experiment is conditional on an already screened, expressible, measured candidate
population. It does not estimate discovery rates across nature. Full construct sequences
are used without unverified trimming. Related domains may cross the 60% whole-sequence
threshold. ESM pretraining may contain these proteins or homologs; only assay-label
training leakage is controlled. Scores are not calibrated probabilities, predicted
melting temperatures, or evidence about the effect of a proposed point mutation.

## What this contributes

A checked, condition-specific dataset; auditable splits; cached frozen-model features;
a fair same-pool selection comparison; and a visible account of generalization limits.
A lab could use this evidence to judge whether this particular ranking approach merits
prospective testing. Choosing improved variants still requires new experiments.

## Files

- `summary.csv`: aggregate selection and ranking metrics.
- `paired_comparison.csv`: same-seed ESM-minus-nearest comparisons.
- `predictions.csv`: every outer-held-out prediction and its nearest training neighbor.
- `split_audit.csv`: train/test memberships and cross-split similarity checks.
- `run_manifest.json`: config, checksums, model revision, and limitations.
- `selection_lift.png`: static figure for a portfolio or report.

Source data retain CC BY-NC 4.0 terms; see DATA_LICENSE.md.
'''
    (out/'FINDINGS.md').write_text(text)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    colors={'Nearest sequence':'#90a6a0','Composition ridge':'#e7b467','ESM-2 ridge':'#126b54'}
    for ax,regime in zip(axes,['random','matched','grouped']):
        for i,model in enumerate(colors):
            d=seeds[(seeds.regime==regime)&(seeds.model==model)&(seeds.budget==budget)].lift_over_random
            ax.bar(i,d.mean(),color=colors[model],width=.6)
            ax.scatter(np.full(len(d),i)+np.linspace(-.08,.08,len(d)),d,color='#152d2a',s=15,zorder=3)
        ax.axhline(0,color='#536660',linewidth=.8)
        ax.set_xticks(range(3),['Nearest\nsequence','Composition\nridge','ESM-2\nridge'])
        ax.set_title({'random':'Random holdout','matched':'Size-matched random','grouped':'Sequence-group holdout'}[regime],fontweight='bold')
    axes[0].set_ylabel('Top-5 activity above same-pool random expectation\n(µmol aromatic products / mg enzyme)')
    fig.suptitle('Does AI improve selection of PET-degrading enzymes?',fontweight='bold')
    fig.text(.5,.01,'60°C · pH 7.5 · crystalline PET · 48 h | Dots = seed-level means; not independent replicates',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(out/'selection_lift.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(f'Wrote {out / "FINDINGS.md"}',flush=True)

"""Audited findings, static research figures and portfolio case study."""
import json
import pandas as pd
from .data import digest
from .bench_eval import validate_data_inputs

COLORS={'Random':'#87958e','Nearest sequence':'#c28b41','AI greedy':'#146954',
        'AI uncertainty':'#398cb3','AI diversity':'#b46d8c','AI explore':'#6862aa'}
SCENARIOS={'random_start':'Random starting samples','group_shift':'Start in one group; search outside it'}


def verified(root):
    base=root/'validation/results'
    manifest=json.loads((base/'manifest.json').read_text())
    for name,value in manifest['config_sha256'].items():
        if digest(root/name)!=value: raise ValueError('Validation configuration changed. Rerun the study.')
    for name,value in manifest['code_sha256'].items():
        if digest(root/'src/pet_scout'/name)!=value: raise ValueError('Validation code changed. Rerun the study.')
    data=validate_data_inputs(root,json.loads((root/'benchmark_config.json').read_text()))
    if data!=manifest['data_manifest']: raise ValueError('Prepared data changed. Rerun the study.')
    for name,value in manifest['result_sha256'].items():
        if digest(base/name)!=value: raise ValueError(f'Validation output changed: {name}. Rerun the study.')
    return manifest


def diagnostic_summary(frame):
    # Equal weight per initial run, then per anchor. Undefined correlations stay missing.
    keys=['task','scenario','anchor']
    cols=['uncertainty_error_spearman','prediction_spearman','mean_abs_error','mean_std','high_minus_low_error']
    per_anchor=frame.groupby(keys)[cols].mean().reset_index()
    summary=per_anchor.groupby(['task','scenario'])[cols].mean().reset_index()
    counts=frame.groupby(['task','scenario']).agg(valid_correlations=('uncertainty_error_spearman','count'),
                                                n_starts=('seed','size')).reset_index()
    return summary.merge(counts,on=['task','scenario'])


def markdown_table(frame, columns):
    lines=['| '+' | '.join(columns.values())+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        values=[f'{row[key]:.2f}' if isinstance(row[key],float) else str(row[key]) for key in columns]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    manifest=verified(root); out=root/'validation/results'
    cfg=manifest['config']; labels=manifest['task_labels']
    summary=pd.read_csv(out/'summary.csv')
    primary=summary[summary.tested.eq(cfg['primary_budget'])].copy()
    diag=diagnostic_summary(pd.read_csv(out/'diagnostics.csv'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for i,(scenario,scenario_label) in enumerate(SCENARIOS.items()):
        for j,(task,label) in enumerate(labels.items()):
            frame=primary[(primary.scenario==scenario)&(primary.task==task)].set_index('strategy').reindex(COLORS)
            ax=axes[i,j]
            ax.barh(frame.index[::-1],frame.new_hits.iloc[::-1],color=[COLORS[s] for s in frame.index[::-1]])
            ax.set_title(label)
            ax.set_xlabel('New hits after 24 additional assays')
            ax.grid(axis='x',alpha=.15)
            for k,value in enumerate(frame.new_hits.iloc[::-1]): ax.text(value+.04,k,f'{value:.2f}',va='center',fontsize=9)
            ax.set_xlim(0,max(primary.new_hits)*1.2)
    fig.suptitle('Does AI still help after starting from a narrow sequence group?',fontsize=16)
    fig.text(.5,.915,'Top: random starts   •   Bottom: group shift (both eligible starting groups, equally weighted)',ha='center')
    fig.text(.5,.015,'40 total assays = 16 starting measurements + 24 new tests. Hits use the original condition’s 90th percentile.\nCompare policies within each scenario: the searchable pools and available hits differ between scenarios.',ha='center')
    fig.tight_layout(rect=[0,.07,1,.9]);fig.savefig(out/'robustness.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    added=['AI uncertainty','AI diversity','AI explore']
    for ax,(scenario,label) in zip(axes,SCENARIOS.items()):
        for j,(task,tasklabel) in enumerate(labels.items()):
            f=primary[(primary.scenario==scenario)&(primary.task==task)].set_index('strategy').reindex(added)
            ax.bar(np.arange(3)+(j-1)*.24,f.delta_greedy,width=.24,label=tasklabel)
        ax.axhline(0,color='#555',lw=1);ax.set_xticks(range(3),['Add uncertainty','Add diversity','Add both'])
        ax.set_title(label);ax.set_ylabel('Change in new hits vs prediction-only AI');ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Which exploration component changes discovery performance?',fontsize=15)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'ablations.png',dpi=160);plt.close(fig)
    # Per-run quartile means avoid pretending repeated candidate predictions are independent.
    raw=pd.read_csv(out/'initial_predictions.csv')
    raw=raw[raw.uncertainty_quartile.gt(0)]
    q=raw.groupby(['task','scenario','anchor','seed','uncertainty_quartile']).abs_error_log1p.mean().reset_index()
    q=q.groupby(['task','scenario','anchor','uncertainty_quartile']).abs_error_log1p.mean().reset_index()
    q=q.groupby(['task','scenario','uncertainty_quartile']).abs_error_log1p.mean().reset_index()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,(scenario,label) in zip(axes,SCENARIOS.items()):
        for task,tasklabel in labels.items():
            f=q[(q.task==task)&(q.scenario==scenario)]
            ax.plot(f.uncertainty_quartile,f.abs_error_log1p,'o-',label=tasklabel)
        ax.set_title(label);ax.set_xticks([1,2,3,4],['Lowest','Q2','Q3','Highest'])
        ax.set_xlabel('Predicted uncertainty quartile · first decision only')
        ax.set_ylabel('Mean absolute error in log(1 + activity)');ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Do larger uncertainty scores identify larger errors?',fontsize=15)
    fig.tight_layout(rect=[0,.02,1,.93]);fig.savefig(out/'uncertainty_diagnostic.png',dpi=160);plt.close(fig)
    primary['condition']=primary.task.map(labels)
    primary['start']=primary.scenario.map(SCENARIOS)
    columns={'condition':'Condition','start':'Starting scenario','strategy':'Policy','new_hits':'New hits',
             'new_hit_groups':'Groups with new hits','delta_nearest':'Δ hits vs nearest'}
    anchor_results=pd.read_csv(out/'anchor_summary.csv')
    text=['# When does AI help select PET-degrading enzymes under a limited experimental budget?', '',
        '## Main findings', '',
        'At the fixed 40-assay budget, the shifted-start test preserves a prediction-only AI advantage over sequence similarity at 40°C/pH 7.5 for both eligible starting groups. The 60°C/pH 7.5 average advantage is sensitive to which group supplied the initial measurements. In acidic conditions, random selection has the highest mean new-hit count after the shift. These are conditional, retrospective findings.', '',
        'The random-start ablation attributes the larger hit-count penalty to the fixed diversity addition: adding diversity loses roughly 2.3–2.5 hits versus prediction-only AI, while adding uncertainty loses 0.1–1.0. Diversity can nevertheless increase the number of groups represented among discoveries, particularly in the shifted warm-neutral setting. The choice between finding many hits and finding hits across more groups changes how a strategy should be judged.', '',
        '## The practical problem', '',
        'A laboratory has measurements for 16 PET-degrading enzyme candidates and can afford 24 more assays. Which enzymes should it test next? This project evaluates a reproducible decision system using published measurements, so selection strategies can be compared without making new proteins.', '',
        '## What I built', '',
        'A protein-model benchmark, a sequential experiment-selection simulator, and a robustness/ablation study with an interactive dashboard. The pipeline validates source data and cached embeddings, hides unrevealed labels from selection policies, records every batch, and exports checksummed results. Frozen ESM-2 35M embeddings feed a 20-member bootstrap ridge predictor. Sequence similarity and random selection provide practical baselines.', '',
        '## Evaluation design', '',
        '- Three PET assay conditions, kept separate: 40°C/pH 7.5, 40°C/pH 4.5 and 60°C/pH 7.5; crystalline powder, 48-hour measurements.',
        '- A fixed primary budget of 40 total assays, including 16 starting measurements. Curves through 64 assays are secondary.',
        '- Random starts: 50 shared initial sets per condition. The four existing policies must exactly reproduce the earlier campaign selections.',
        '- Group shift: start with 16 samples from one sequence group and exclude all remaining members of that group from acquisition. Evaluate every eligible starting group, with 50 starts each; this dataset has two eligible groups per condition.',
        '- Sequence groups are 60%-identity connected components, not validated biological families. After the first acquired batch returns, the model can learn from new-group measurements. This tests initial transfer and adaptation, not permanent group holdout.',
        '- A hit is activity at or above the original condition’s 90th percentile, including ties. The evaluator uses this definition; policies never receive the threshold or unrevealed outcomes.',
        '- The primary outcome counts newly acquired hits, excluding starting discoveries. A secondary outcome counts distinct sequence groups represented among new hits. In the shifted scenario, all such groups are outside the initial group.',
        '- Searchable pools and hit prevalence differ between scenarios. Compare policies against paired baselines within a scenario; raw cross-scenario differences do not isolate the effect of starting-set breadth.', '',
        '## Results at 40 total assays', '',markdown_table(primary,columns),'',
        'These are means across 50 starts for the random scenario and across 50 starts per eligible group for the shifted scenario, with equal weight per starting group. Repeated starts reuse proteins and measurements; they are not independent laboratory replications.', '',
        '## Does the advantage survive unfamiliar starting conditions?', '']
    for task,label in labels.items():
        f=primary[(primary.task==task)&primary.scenario.eq('group_shift')].set_index('strategy')
        g=f.loc['AI greedy'];n=f.loc['Nearest sequence'];r=f.loc['Random']
        text.append(f'- **{label}:** prediction-only AI finds {g.new_hits:.2f} new hits versus {n.new_hits:.2f} for nearest sequence and {r.new_hits:.2f} for random. Its difference from nearest is {g.delta_nearest:+.2f}; it discovers hits in {g.new_hit_groups:.2f} new groups on average.')
        a=anchor_results[(anchor_results.task==task)&anchor_results.scenario.eq('group_shift')&anchor_results.strategy.eq('AI greedy')&anchor_results.tested.eq(cfg['primary_budget'])]
        text.append('  Starting-group breakdown, AI minus nearest: '+ '; '.join(f'group {int(row.anchor)}: {row.delta_nearest:+.2f} new hits' for row in a.itertuples())+'.')
    text+=['','## Why did the original exploration strategy lose?', '',
        'The ablation keeps the predictor, initial measurements, batch sizes and acquisition weights fixed. “Add uncertainty” ranks mean + one bootstrap standard deviation. “Add diversity” combines 75% rank of predicted mean with 25% rank of minimum cosine distance to observed or already selected embeddings. “Add both” is the original combined policy. This isolates additions to the greedy policy at these settings; it does not establish which component would work best after tuning.', '']
    for scenario,label in SCENARIOS.items():
        text.append(f'### {label}')
        for task,tasklabel in labels.items():
            f=primary[(primary.task==task)&primary.scenario.eq(scenario)].set_index('strategy')
            u,d,c=[f.loc[s].delta_greedy for s in added]
            text.append(f'- **{tasklabel}:** uncertainty changes new hits by {u:+.2f}; diversity by {d:+.2f}; both by {c:+.2f}.')
    text+=['','## Does the uncertainty score identify errors?', '',
        'An offline diagnostic compares bootstrap spread with absolute log-activity error over all untested candidates at the first shared decision. These measurements are used only for analysis, never selection. A positive rank correlation means larger scores tend to accompany larger errors; it does not demonstrate calibrated intervals or useful exploration. Later policy trajectories diverge, so this diagnostic alone cannot explain every downstream outcome. When every uncertainty score is identical, the run has no uncertainty ranking and is excluded from quartile comparisons rather than assigning an artificial ID-based ordering; valid correlation counts are shown.', '']
    for row in diag.itertuples():
        text.append(f'- **{labels[row.task]}, {SCENARIOS[row.scenario]}:** mean within-start uncertainty/error Spearman correlation {row.uncertainty_error_spearman:.2f}; highest minus lowest uncertainty-quartile error {row.high_minus_low_error:+.2f} log-activity units. Defined correlations: {row.valid_correlations}/{row.n_starts}.')
    text+=['','## Engineering safeguards and reproducibility', '',
        '- Acquisition APIs receive revealed labels only. Every batch is chosen before any of its measurements returns.',
        '- Feature scaling uses observed samples only. Frozen embeddings and sequence similarities can be known for all candidates.',
        '- Full source, configuration, code and output hashes; deterministic seeds; paired baseline comparisons; no best-seed selection.',
        '- Original model-benchmark and campaign outputs are preserved. Automated checks cover label isolation, original-policy parity, eligibility, group exclusion, numerical regression, reconstructed metrics and UI controls.',
        f'- This extension completed {manifest["n_campaigns"]:,} campaigns in {manifest["simulation_seconds"]:.1f} seconds for simulation and CSV export on the existing local CPU environment, reusing cached embeddings. Paid compute: $0. Figure rendering is additional.', '',
        '## Limits', '',
        'This is an exploratory single-study replay of previously screened and expressed natural enzyme constructs. The original data and initial campaign results were inspected before this validation extension; it is not preregistered or blind. Only two groups per condition can supply 16 starting examples, which limits the breadth of the shift test. Assay noise, expression failures and varying experimental costs are not simulated. Group counts describe sequence breadth, not novel mechanisms or independently useful enzyme scaffolds. No new enzyme was designed or experimentally validated.', '',
        '## How I would present the contribution', '',
        '> Built and evaluated a reproducible AI-guided PET-enzyme screening system under fixed experimental budgets, with sequence-group shift tests, acquisition ablations, uncertainty diagnostics and an auditable interactive dashboard.', '',
        'Use the condition-specific results above when describing performance. Do not present one favorable condition as a universal advantage or turn simulated hit gains into unmeasured monetary savings.', '',
        '## Reproduce and demo', '',
        '```sh','.venv/bin/python -m pet_scout.validation','.venv/bin/python -m pytest -q',
        '.venv/bin/python -m streamlit run app.py','```','',
        'Open **Validation & case study** in the sidebar. See VALIDATION_STUDY.md for the full protocol and DEMO.md for a two-minute walkthrough.', '',
        'Source: [Norton-Baker et al., ACS Catalysis (2025)](https://doi.org/10.1021/acscatal.5c03460). Experimental data CC BY-NC 4.0; project software MIT.']
    (out/'CASE_STUDY.md').write_text('\n'.join(text)+'\n')
    demo=['# Two-minute dashboard walkthrough','','This is a narration script, not a recorded video.','',
        '**0:00–0:20 — The problem.** “A lab knows the activity of 16 enzyme candidates and has budget for 24 more tests. I built a system to compare which candidates it should test next.”','',
        '**0:20–0:45 — The original result.** Open Experiment planner at 40°C/pH 7.5 and budget 40. Explain that AI greedy finds 8.34 strong enzymes in total versus 7.20 for sequence similarity and 4.18 for random. These totals include starting hits. Then switch to acidic conditions to show that sequence similarity wins there.','',
        '**0:45–1:10 — The harder test.** Open Validation & case study → Group shift. Explain that all 16 initial measurements come from one sequence group and remaining candidates from that group are excluded. Show new hits and distinct groups with new hits. This page counts only newly acquired hits, so its values differ from the original totals. Use the condition-specific results shown; do not claim universal AI superiority.','',
        '**1:10–1:35 — Diagnose the failure.** Open Acquisition ablations. Compare uncertainty added, diversity added and both added against prediction-only AI. Point to the uncertainty diagnostic and distinguish an informative score from a calibrated error estimate.','',
        '**1:35–2:00 — Reproducibility and limits.** Show the protocol/audit tab and downloads. “Policies never see hidden outcomes; every result can be reconstructed. This uses published experiments and does not claim a new enzyme or prospective lab validation.”','',
        'Finish with the strongest result actually supported across the two eligible starting groups, including any negative or mixed outcomes.']
    (out/'DEMO.md').write_text('\n'.join(demo)+'\n')

"""Scientific figures and plain-language findings, generated from saved results."""
import json
import pandas as pd
from .data import digest

COLORS = {'Random':'#87958e', 'Nearest sequence':'#c28b41', 'AI greedy':'#146954', 'AI explore':'#6862aa'}


def verified_results(root):
    base = root/'campaign/results'
    manifest = json.loads((base/'manifest.json').read_text())
    if digest(root/'campaign_config.json') != manifest['config_sha256']:
        raise ValueError('Campaign configuration changed; rerun the simulation.')
    if json.loads((root/'benchmark/data_manifest.json').read_text()) != manifest['data_manifest']:
        raise ValueError('Prepared benchmark changed; rerun the simulation.')
    from .bench_eval import validate_data_inputs
    validate_data_inputs(root, json.loads((root/'benchmark_config.json').read_text()))
    if digest(root/'src/pet_scout/campaign.py') != manifest['code_sha256']:
        raise ValueError('Simulation code changed; rerun the simulation.')
    for filename, expected in manifest['result_sha256'].items():
        if digest(base/filename) != expected:
            raise ValueError(f'Campaign output changed: {filename}. Rerun the simulation.')
    return manifest, {name:pd.read_csv(base/f'{name}.csv') for name in ['summary','checkpoints','selections','paired','target_times']}


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest, tables = verified_results(root)
    config = manifest['config']
    summary = tables['summary']
    out = root/'campaign/results'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(15,8), sharex=True)
    for j,(task,label) in enumerate(manifest['task_labels'].items()):
        for strategy,color in COLORS.items():
            data = summary[(summary.task==task)&(summary.strategy==strategy)].sort_values('tested')
            axes[0,j].plot(data.tested,data.mean_hits,label=strategy,color=color,lw=2)
            axes[1,j].plot(data.tested,data.target_success_rate*100,label=strategy,color=color,lw=2)
        axes[0,j].set_title(label)
        axes[1,j].set_xlabel('Total enzymes tested (includes 16 starting assays)')
        axes[1,j].set_ylim(0,103)
        for axis in axes[:,j]:
            axis.axvline(config['primary_budget'],color='#bbb',linestyle='--',lw=1)
            axis.grid(alpha=.15)
            axis.set_xticks(range(config['initial_size'],config['max_budget']+1,config['batch_size']))
    axes[0,0].set_ylabel(f'Strong enzymes found\n(mean across {config["n_seeds"]} starts)')
    axes[1,0].set_ylabel('Campaigns finding ≥5\nstrong enzymes (%)')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Can AI spend a limited PET-enzyme screening budget better?',fontsize=17,y=.99)
    fig.text(.5,.015,'Strong = measured activity ≥ within-condition 90th percentile. Same starting sets; public measurements replayed.\nRepeated starts reuse proteins: these are simulation outcomes, not independent lab replications.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.95])
    fig.savefig(out/'discovery_curves.png',dpi=160)
    plt.close(fig)
    primary = summary[summary.tested.eq(config['primary_budget'])]
    lines = ['# PETBench: choosing the next experiment', '',
        'A retrospective screening simulation asks whether AI finds more strong PET-degrading enzymes with a fixed experimental budget.', '',
        f'**Protocol:** {config["n_seeds"]} shared random starts per condition; {config["initial_size"]} initially measured enzymes, batches of {config["batch_size"]}, primary endpoint {config["primary_budget"]} total assays, secondary budget {config["max_budget"]}. {manifest["n_campaigns"]} campaigns completed.', '',
        '**Strong** means activity at or above that condition’s 90th percentile. Ties are included. This is a hindsight, dataset-relative scoring rule, not an industrial threshold, and is never provided to a selection policy.', '',
        '## Results at 40 total assays', '',
        '| Condition | Strategy | Mean strong enzymes found | Difference vs random | Difference vs nearest | Campaigns finding ≥5 |',
        '|---|---|---:|---:|---:|---:|']
    for row in primary.itertuples():
        lines.append(f'| {manifest["task_labels"][row.task]} | {row.strategy} | {row.mean_hits:.2f} | {row.delta_random:+.2f} | {row.delta_nearest:+.2f} | {row.target_success_rate:.0%} |')
    lines += ['', '## What this tells us', '']
    for task,label in manifest['task_labels'].items():
        p = primary[primary.task.eq(task)].set_index('strategy')
        greedy, explore = p.loc['AI greedy'], p.loc['AI explore']
        lines.append(f'- **{label}:** AI greedy found {greedy.mean_hits:.2f} strong enzymes on average at 40 assays ({greedy.delta_random:+.2f} versus random; {greedy.delta_nearest:+.2f} versus nearest sequence). Adding uncertainty and diversity changed that by {explore.delta_greedy:+.2f} strong enzymes. This is the observed result, not proof of a general advantage.')
    times = tables['target_times']
    lines += ['', '## Time to five strong enzymes', '',
        '| Condition | Strategy | Reached by 64 | Mean capped assays to target |', '|---|---|---:|---:|']
    for (task,strategy),frame in times.groupby(['task','strategy'],sort=False):
        lines.append(f'| {manifest["task_labels"][task]} | {strategy} | {frame.reached_by_max.mean():.0%} | {frame.capped_assays.mean():.2f} |')
    lines += ['', 'Unsuccessful campaigns remain in the denominator. Capped assays assigns 64 to non-reaching runs; it is NOT an estimate of their eventual time to success. Only batch-completion times are observed. No savings percentage is calculated from successful runs alone.', '',
        '## Policies', '',
        '- Random: uniform sampling without replacement from remaining candidates.',
        '- Nearest sequence: predict each untested enzyme from the measured activity of its closest observed sequence; sample-ID tie breaks.',
        '- AI greedy: rank the mean log1p activity predicted by 20 bootstrapped ridge models using frozen ESM-2 35M embeddings.',
        '- AI explore: 75% percentile rank of ensemble mean + one bootstrap standard deviation, plus 25% percentile rank of minimum cosine distance to observed and already selected sequences. Recompute diversity after each batch choice; do not reveal batch outcomes until all eight are selected.',
        '- Both AI policies use the same train-only scaler, bootstrap procedure and fixed ridge alpha 100. No tuning against this simulation’s outcomes. The embedding model was chosen for low cost after the earlier benchmark, so this remains exploratory.', '',
        '## Interpretation limits', '',
        '- The algorithm receives only revealed labels, but the dataset was previously inspected by the project author. This is not a blind evaluation.',
        '- These are sequential searches within a finite, previously screened and expressed candidate pool, not sequence-family holdout tests or searches across arbitrary natural sequences.',
        '- The same measured outcomes are reused across 50 starts. Percentile bands and paired win rates describe starting-set sensitivity, not biological confidence intervals.',
        '- Bootstrap spread is an acquisition heuristic, not calibrated prediction uncertainty. Its contribution is combined with diversity; this four-policy experiment does not isolate the two.',
        '- All assays are treated as equal-cost and repeat measurements are represented by their existing means. Expression failures, assay noise and real scheduling costs are not modeled.',
        '- Different conditions have different measured candidate populations. Do not pool the condition results or treat their scores as directly comparable.',
        '- No new enzyme is designed or experimentally validated. External or prospective validation remains future work.', '',
        f'**Compute:** {manifest["simulation_seconds"]:.1f} seconds for simulation and result export; existing frozen embeddings reused; $0 paid compute. Report rendering is additional.', '',
        '## Reproduce', '', 'From the project folder:', '', '```sh',
        '.venv/bin/python -m pet_scout.campaign', '.venv/bin/python -m pytest -q',
        '.venv/bin/python -m streamlit run app.py', '```', '',
        'Requires prepared PETBench data and the validated ESM-2 35M cache. See CAMPAIGN.md for setup. `manifest.json` records configuration, source and result hashes. `selections.csv` records every revealed sample and its acquisition-time score. `checkpoints.csv` records every campaign checkpoint.', '',
        'Source: [Norton-Baker et al., ACS Catalysis (2025)](https://doi.org/10.1021/acscatal.5c03460). Experimental data CC BY-NC 4.0.']
    (out/'FINDINGS.md').write_text('\n'.join(lines)+'\n')

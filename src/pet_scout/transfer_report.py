"""Export a checked transfer report and standalone scientific figure."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .external import verified


def run(root):
    verified(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = root/'external'
    data = pd.read_csv(base/'metrics.csv')
    colors = ['#236b8e','#89b4c2','#ca8b62']
    fig, axes = plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
    methods = ['ESM-2 35M','Composition','Nearest sequence']
    for i, cohort in enumerate(['nonoverlapping','identity_below_60pct']):
        for j, temperature in enumerate([40,60]):
            ax = axes[i,j]
            subset = data[data.cohort.eq(cohort)&data.external_temperature_c.eq(temperature)].set_index('model').loc[methods]
            row = subset.iloc[0]
            bars = ax.bar(np.arange(3),subset.selected_mean_um,color=colors,width=.65)
            ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=10)
            ax.axhline(row.random_mean_um,color='#646b74',linestyle='--',label='Random expectation')
            ax.axhline(row.oracle_mean_um,color='#608265',linestyle=':',label='Outcome-aware oracle')
            ax.set_xticks(np.arange(3),['ESM-2 35M','Composition','Nearest sequence'])
            ax.set_ylabel('Mean products of top 3 picks (µM)')
            ax.set_ylim(0,max(subset.selected_mean_um.max(),row.oracle_mean_um)*1.24)
            label = 'Overlap excluded' if i == 0 else 'Below 60% identity'
            ax.set_title(f'{temperature}°C · {label} · n={int(row.n)}',loc='left',weight='bold')
            ax.spines[['top','right']].set_visible(False)
            ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('PETBench: ranking enzymes from an independent study',fontsize=17,weight='bold')
    fig.supxlabel('Wu et al. (2025), Fig. 2c · pH 9 · amorphous PET film · 72 h\nRetrospective, preselected panel. Different vertical scales. No assay-error confidence intervals.',fontsize=10)
    fig.savefig(base/'transfer.png',dpi=180)
    fig.savefig(base/'transfer.pdf')
    plt.close(fig)
    rows = []
    for r in data.itertuples():
        cohort = '13 eligible' if r.cohort == 'nonoverlapping' else '11 below 60%'
        rows.append(f'| {r.external_temperature_c}°C | {cohort} | {r.model} | {r.spearman:.2f} | {r.selected_mean_um:.2f} | {r.random_mean_um:.2f} | {r.selected_ids} |')
    report = '''# PETBench: can a ranking transfer between labs?

**A useful, limited positive result:** a fixed model trained on one study selects
productive enzymes in a second published study, but simple baselines remain
competitive and close relatives explain much of the large headline gain.

At 40°C, the AI-selected trio averages **280.57 µM** of PET breakdown products,
versus **75.31 µM** for random selection and **29.03 µM** for nearest-sequence
selection. Composition reaches **277.56 µM**, almost the same as AI. At 60°C,
nearest-sequence selection slightly beats AI on top-three activity (**67.37 vs
67.11 µM**). Higher overall rank correlation does not automatically mean a better
top-three batch.

![Independent transfer comparison](transfer.png)

## What survives the harder test?

One overlapping construct (APET-15 / DP093) was excluded before evaluation,
leaving 13 candidates. Removing APET-14 and APET-17, which have close training
relatives, leaves 11 candidates below 60% global identity.

On that distant subset, AI's selected mean is **17.82 vs 14.12 µM random** at 40°C
and **17.53 vs 15.14 µM random** at 60°C. AI beats both simple baselines in this
subset, but the absolute differences are small. Its rank correlations are **0.28**
and **0.72**, respectively. This suggests transferable ranking information in
this panel; it does not establish that the small gains exceed assay variability.

## Complete comparison

All activity values below are measured external TPA + MHET concentrations after
72 hours, averaged over three selected proteins. They are not model-calibrated
predictions or estimates of industrial recycling yield.

| Assay | Cohort | Method | Spearman | Selected mean µM | Random mean µM | Selected IDs |
|---|---|---|---:|---:|---:|---|
'''+ '\n'.join(rows) + '''

## Why this is credible—and where it stops

- Fixed comparison protocol written before calculating model performance;
  no external-label tuning. This was not preregistered, blinded or prospective.
- Public workbooks, explicit DNA-to-protein reconstruction, checksum validation,
  per-candidate audit, scores, source rows and all three replicate values.
- Every possible random triple is enumerated. These random-selection references
  describe the fixed panel; they are not confidence intervals for other studies.
- The test changes PET form, pH, duration and normalization. Rank transfer is the
  question; cross-study absolute prediction error would be misleading.
- The external panel was already selected using models, expression and an
  activity screen. It does not establish discovery rates among arbitrary proteins.
- Both temperatures reuse the same small panel. Assay uncertainty was not
  propagated, and overlap with model pretraining sequences is unknown.
- No new enzyme was designed, discovered or experimentally tested by PETBench.

## A defensible portfolio claim

“Built a reproducible PET-hydrolase selection benchmark and tested cross-study
transfer with construct-overlap checks, sequence-distance sensitivity and simple
baselines. Delivered a local workflow that turns measured proteins and untested
sequences into an auditable experimental batch.”

## Reproduce and use

Run `python -m pet_scout.external`, then `python -m pet_scout.transfer_report`
from the project root with the installed environment. See
`TRANSFER_AND_RANKING.md` for exact commands, upload formats, privacy and compute.
The dashboard includes **External validation** and **Rank candidates**.

Sources: [Wu et al., Nature Communications (2025)](https://doi.org/10.1038/s41467-025-61599-z),
CC BY 4.0; [Norton-Baker et al., ACS Catalysis (2025)](https://doi.org/10.1021/acscatal.5c03460),
supplement CC BY-NC 4.0. PETBench analysis is independent of the authors.
'''
    (base/'FINDINGS.md').write_text(report)
    print(f'Saved {base}/FINDINGS.md and transfer.png/pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    run(parser.parse_args().root.resolve())

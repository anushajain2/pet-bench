# When does AI help select PET-degrading enzymes under a limited experimental budget?

## Main findings

At the fixed 40-assay budget, the shifted-start test preserves a prediction-only AI advantage over sequence similarity at 40°C/pH 7.5 for both eligible starting groups. The 60°C/pH 7.5 average advantage is sensitive to which group supplied the initial measurements. In acidic conditions, random selection has the highest mean new-hit count after the shift. These are conditional, retrospective findings.

The random-start ablation attributes the larger hit-count penalty to the fixed diversity addition: adding diversity loses roughly 2.3–2.5 hits versus prediction-only AI, while adding uncertainty loses 0.1–1.0. Diversity can nevertheless increase the number of groups represented among discoveries, particularly in the shifted warm-neutral setting. The choice between finding many hits and finding hits across more groups changes how a strategy should be judged.

## The practical problem

A laboratory has measurements for 16 PET-degrading enzyme candidates and can afford 24 more assays. Which enzymes should it test next? This project evaluates a reproducible decision system using published measurements, so selection strategies can be compared without making new proteins.

## What I built

A protein-model benchmark, a sequential experiment-selection simulator, and a robustness/ablation study with an interactive dashboard. The pipeline validates source data and cached embeddings, hides unrevealed labels from selection policies, records every batch, and exports checksummed results. Frozen ESM-2 35M embeddings feed a 20-member bootstrap ridge predictor. Sequence similarity and random selection provide practical baselines.

## Evaluation design

- Three PET assay conditions, kept separate: 40°C/pH 7.5, 40°C/pH 4.5 and 60°C/pH 7.5; crystalline powder, 48-hour measurements.
- A fixed primary budget of 40 total assays, including 16 starting measurements. Curves through 64 assays are secondary.
- Random starts: 50 shared initial sets per condition. The four existing policies must exactly reproduce the earlier campaign selections.
- Group shift: start with 16 samples from one sequence group and exclude all remaining members of that group from acquisition. Evaluate every eligible starting group, with 50 starts each; this dataset has two eligible groups per condition.
- Sequence groups are 60%-identity connected components, not validated biological families. After the first acquired batch returns, the model can learn from new-group measurements. This tests initial transfer and adaptation, not permanent group holdout.
- A hit is activity at or above the original condition’s 90th percentile, including ties. The evaluator uses this definition; policies never receive the threshold or unrevealed outcomes.
- The primary outcome counts newly acquired hits, excluding starting discoveries. A secondary outcome counts distinct sequence groups represented among new hits. In the shifted scenario, all such groups are outside the initial group.
- Searchable pools and hit prevalence differ between scenarios. Compare policies against paired baselines within a scenario; raw cross-scenario differences do not isolate the effect of starting-set breadth.

## Results at 40 total assays

| Condition | Starting scenario | Policy | New hits | Groups with new hits | Δ hits vs nearest |
| --- | --- | --- | --- | --- | --- |
| 40°C · pH 4.5 | Start in one group; search outside it | AI diversity | 1.70 | 1.31 | 0.22 |
| 40°C · pH 4.5 | Start in one group; search outside it | AI explore | 1.66 | 1.26 | 0.18 |
| 40°C · pH 4.5 | Start in one group; search outside it | AI greedy | 1.66 | 1.07 | 0.18 |
| 40°C · pH 4.5 | Start in one group; search outside it | AI uncertainty | 1.25 | 0.85 | -0.23 |
| 40°C · pH 4.5 | Start in one group; search outside it | Nearest sequence | 1.48 | 0.85 | 0.00 |
| 40°C · pH 4.5 | Start in one group; search outside it | Random | 2.01 | 1.32 | 0.53 |
| 40°C · pH 4.5 | Random starting samples | AI diversity | 2.72 | 1.42 | -4.14 |
| 40°C · pH 4.5 | Random starting samples | AI explore | 2.16 | 1.28 | -4.70 |
| 40°C · pH 4.5 | Random starting samples | AI greedy | 5.02 | 1.14 | -1.84 |
| 40°C · pH 4.5 | Random starting samples | AI uncertainty | 4.04 | 1.14 | -2.82 |
| 40°C · pH 4.5 | Random starting samples | Nearest sequence | 6.86 | 1.28 | 0.00 |
| 40°C · pH 4.5 | Random starting samples | Random | 2.74 | 1.54 | -4.12 |
| 40°C · pH 7.5 | Start in one group; search outside it | AI diversity | 3.48 | 2.93 | 0.24 |
| 40°C · pH 7.5 | Start in one group; search outside it | AI explore | 3.42 | 2.87 | 0.18 |
| 40°C · pH 7.5 | Start in one group; search outside it | AI greedy | 4.98 | 2.57 | 1.74 |
| 40°C · pH 7.5 | Start in one group; search outside it | AI uncertainty | 4.93 | 2.58 | 1.69 |
| 40°C · pH 7.5 | Start in one group; search outside it | Nearest sequence | 3.24 | 2.06 | 0.00 |
| 40°C · pH 7.5 | Start in one group; search outside it | Random | 2.71 | 2.16 | -0.53 |
| 40°C · pH 7.5 | Random starting samples | AI diversity | 4.46 | 3.38 | -1.22 |
| 40°C · pH 7.5 | Random starting samples | AI explore | 4.54 | 3.60 | -1.14 |
| 40°C · pH 7.5 | Random starting samples | AI greedy | 6.82 | 3.10 | 1.14 |
| 40°C · pH 7.5 | Random starting samples | AI uncertainty | 6.64 | 3.22 | 0.96 |
| 40°C · pH 7.5 | Random starting samples | Nearest sequence | 5.68 | 2.18 | 0.00 |
| 40°C · pH 7.5 | Random starting samples | Random | 2.66 | 2.10 | -3.02 |
| 60°C · pH 7.5 | Start in one group; search outside it | AI diversity | 5.02 | 4.08 | 0.19 |
| 60°C · pH 7.5 | Start in one group; search outside it | AI explore | 5.09 | 4.33 | 0.26 |
| 60°C · pH 7.5 | Start in one group; search outside it | AI greedy | 6.49 | 3.55 | 1.66 |
| 60°C · pH 7.5 | Start in one group; search outside it | AI uncertainty | 6.74 | 4.03 | 1.91 |
| 60°C · pH 7.5 | Start in one group; search outside it | Nearest sequence | 4.83 | 2.94 | 0.00 |
| 60°C · pH 7.5 | Start in one group; search outside it | Random | 3.20 | 2.55 | -1.63 |
| 60°C · pH 7.5 | Random starting samples | AI diversity | 4.44 | 3.80 | -0.46 |
| 60°C · pH 7.5 | Random starting samples | AI explore | 4.56 | 4.14 | -0.34 |
| 60°C · pH 7.5 | Random starting samples | AI greedy | 6.96 | 4.02 | 2.06 |
| 60°C · pH 7.5 | Random starting samples | AI uncertainty | 6.86 | 4.24 | 1.96 |
| 60°C · pH 7.5 | Random starting samples | Nearest sequence | 4.90 | 3.18 | 0.00 |
| 60°C · pH 7.5 | Random starting samples | Random | 2.00 | 1.74 | -2.90 |

These are means across 50 starts for the random scenario and across 50 starts per eligible group for the shifted scenario, with equal weight per starting group. Repeated starts reuse proteins and measurements; they are not independent laboratory replications.

## Does the advantage survive unfamiliar starting conditions?

- **40°C · pH 7.5:** prediction-only AI finds 4.98 new hits versus 3.24 for nearest sequence and 2.71 for random. Its difference from nearest is +1.74; it discovers hits in 2.57 new groups on average.
  Starting-group breakdown, AI minus nearest: group 0: +1.80 new hits; group 8: +1.68 new hits.
- **40°C · pH 4.5:** prediction-only AI finds 1.66 new hits versus 1.48 for nearest sequence and 2.01 for random. Its difference from nearest is +0.18; it discovers hits in 1.07 new groups on average.
  Starting-group breakdown, AI minus nearest: group 0: +2.10 new hits; group 8: -1.74 new hits.
- **60°C · pH 7.5:** prediction-only AI finds 6.49 new hits versus 4.83 for nearest sequence and 3.20 for random. Its difference from nearest is +1.66; it discovers hits in 3.55 new groups on average.
  Starting-group breakdown, AI minus nearest: group 0: -0.46 new hits; group 8: +3.78 new hits.

## Why did the original exploration strategy lose?

The ablation keeps the predictor, initial measurements, batch sizes and acquisition weights fixed. “Add uncertainty” ranks mean + one bootstrap standard deviation. “Add diversity” combines 75% rank of predicted mean with 25% rank of minimum cosine distance to observed or already selected embeddings. “Add both” is the original combined policy. This isolates additions to the greedy policy at these settings; it does not establish which component would work best after tuning.

### Random starting samples
- **40°C · pH 7.5:** uncertainty changes new hits by -0.18; diversity by -2.36; both by -2.28.
- **40°C · pH 4.5:** uncertainty changes new hits by -0.98; diversity by -2.30; both by -2.86.
- **60°C · pH 7.5:** uncertainty changes new hits by -0.10; diversity by -2.52; both by -2.40.
### Start in one group; search outside it
- **40°C · pH 7.5:** uncertainty changes new hits by -0.05; diversity by -1.50; both by -1.56.
- **40°C · pH 4.5:** uncertainty changes new hits by -0.41; diversity by +0.04; both by +0.00.
- **60°C · pH 7.5:** uncertainty changes new hits by +0.25; diversity by -1.47; both by -1.40.

## Does the uncertainty score identify errors?

An offline diagnostic compares bootstrap spread with absolute log-activity error over all untested candidates at the first shared decision. These measurements are used only for analysis, never selection. A positive rank correlation means larger scores tend to accompany larger errors; it does not demonstrate calibrated intervals or useful exploration. Later policy trajectories diverge, so this diagnostic alone cannot explain every downstream outcome. When every uncertainty score is identical, the run has no uncertainty ranking and is excluded from quartile comparisons rather than assigning an artificial ID-based ordering; valid correlation counts are shown.

- **40°C · pH 4.5, Start in one group; search outside it:** mean within-start uncertainty/error Spearman correlation 0.18; highest minus lowest uncertainty-quartile error +0.14 log-activity units. Defined correlations: 94/100.
- **40°C · pH 4.5, Random starting samples:** mean within-start uncertainty/error Spearman correlation 0.30; highest minus lowest uncertainty-quartile error +0.37 log-activity units. Defined correlations: 50/50.
- **40°C · pH 7.5, Start in one group; search outside it:** mean within-start uncertainty/error Spearman correlation 0.10; highest minus lowest uncertainty-quartile error +0.30 log-activity units. Defined correlations: 100/100.
- **40°C · pH 7.5, Random starting samples:** mean within-start uncertainty/error Spearman correlation 0.06; highest minus lowest uncertainty-quartile error +0.12 log-activity units. Defined correlations: 50/50.
- **60°C · pH 7.5, Start in one group; search outside it:** mean within-start uncertainty/error Spearman correlation 0.21; highest minus lowest uncertainty-quartile error +0.41 log-activity units. Defined correlations: 100/100.
- **60°C · pH 7.5, Random starting samples:** mean within-start uncertainty/error Spearman correlation 0.36; highest minus lowest uncertainty-quartile error +0.68 log-activity units. Defined correlations: 50/50.

## Engineering safeguards and reproducibility

- Acquisition APIs receive revealed labels only. Every batch is chosen before any of its measurements returns.
- Feature scaling uses observed samples only. Frozen embeddings and sequence similarities can be known for all candidates.
- Full source, configuration, code and output hashes; deterministic seeds; paired baseline comparisons; no best-seed selection.
- Original model-benchmark and campaign outputs are preserved. Automated checks cover label isolation, original-policy parity, eligibility, group exclusion, numerical regression, reconstructed metrics and UI controls.
- This extension completed 2,700 campaigns in 30.5 seconds for simulation and CSV export on the existing local CPU environment, reusing cached embeddings. Paid compute: $0. Figure rendering is additional.

## Limits

This is an exploratory single-study replay of previously screened and expressed natural enzyme constructs. The original data and initial campaign results were inspected before this validation extension; it is not preregistered or blind. Only two groups per condition can supply 16 starting examples, which limits the breadth of the shift test. Assay noise, expression failures and varying experimental costs are not simulated. Group counts describe sequence breadth, not novel mechanisms or independently useful enzyme scaffolds. No new enzyme was designed or experimentally validated.

## How I would present the contribution

> Built and evaluated a reproducible AI-guided PET-enzyme screening system under fixed experimental budgets, with sequence-group shift tests, acquisition ablations, uncertainty diagnostics and an auditable interactive dashboard.

Use the condition-specific results above when describing performance. Do not present one favorable condition as a universal advantage or turn simulated hit gains into unmeasured monetary savings.

## Reproduce and demo

```sh
.venv/bin/python -m pet_scout.validation
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py
```

Open **Validation & case study** in the sidebar. See VALIDATION_STUDY.md for the full protocol and DEMO.md for a two-minute walkthrough.

Source: [Norton-Baker et al., ACS Catalysis (2025)](https://doi.org/10.1021/acscatal.5c03460). Experimental data CC BY-NC 4.0; project software MIT.

# PETBench: choosing the next experiment

A retrospective screening simulation asks whether AI finds more strong PET-degrading enzymes with a fixed experimental budget.

**Protocol:** 50 shared random starts per condition; 16 initially measured enzymes, batches of 8, primary endpoint 40 total assays, secondary budget 64. 600 campaigns completed.

**Strong** means activity at or above that condition’s 90th percentile. Ties are included. This is a hindsight, dataset-relative scoring rule, not an industrial threshold, and is never provided to a selection policy.

## Results at 40 total assays

| Condition | Strategy | Mean strong enzymes found | Difference vs random | Difference vs nearest | Campaigns finding ≥5 |
|---|---|---:|---:|---:|---:|
| 40°C · pH 7.5 | Random | 4.18 | +0.00 | -3.02 | 42% |
| 40°C · pH 7.5 | Nearest sequence | 7.20 | +3.02 | +0.00 | 80% |
| 40°C · pH 7.5 | AI greedy | 8.34 | +4.16 | +1.14 | 94% |
| 40°C · pH 7.5 | AI explore | 6.06 | +1.88 | -1.14 | 90% |
| 40°C · pH 4.5 | Random | 4.40 | +0.00 | -4.12 | 50% |
| 40°C · pH 4.5 | Nearest sequence | 8.52 | +4.12 | +0.00 | 98% |
| 40°C · pH 4.5 | AI greedy | 6.68 | +2.28 | -1.84 | 94% |
| 40°C · pH 4.5 | AI explore | 3.82 | -0.58 | -4.70 | 34% |
| 60°C · pH 7.5 | Random | 3.88 | +0.00 | -2.90 | 34% |
| 60°C · pH 7.5 | Nearest sequence | 6.78 | +2.90 | +0.00 | 88% |
| 60°C · pH 7.5 | AI greedy | 8.84 | +4.96 | +2.06 | 94% |
| 60°C · pH 7.5 | AI explore | 6.44 | +2.56 | -0.34 | 92% |

## What this tells us

- **40°C · pH 7.5:** AI greedy found 8.34 strong enzymes on average at 40 assays (+4.16 versus random; +1.14 versus nearest sequence). Adding uncertainty and diversity changed that by -2.28 strong enzymes. This is the observed result, not proof of a general advantage.
- **40°C · pH 4.5:** AI greedy found 6.68 strong enzymes on average at 40 assays (+2.28 versus random; -1.84 versus nearest sequence). Adding uncertainty and diversity changed that by -2.86 strong enzymes. This is the observed result, not proof of a general advantage.
- **60°C · pH 7.5:** AI greedy found 8.84 strong enzymes on average at 40 assays (+4.96 versus random; +2.06 versus nearest sequence). Adding uncertainty and diversity changed that by -2.40 strong enzymes. This is the observed result, not proof of a general advantage.

## Time to five strong enzymes

| Condition | Strategy | Reached by 64 | Mean capped assays to target |
|---|---|---:|---:|
| 40°C · pH 4.5 | AI explore | 100% | 47.36 |
| 40°C · pH 4.5 | AI greedy | 100% | 33.60 |
| 40°C · pH 4.5 | Nearest sequence | 100% | 30.56 |
| 40°C · pH 4.5 | Random | 94% | 43.84 |
| 40°C · pH 7.5 | AI explore | 100% | 35.84 |
| 40°C · pH 7.5 | AI greedy | 98% | 30.56 |
| 40°C · pH 7.5 | Nearest sequence | 96% | 34.24 |
| 40°C · pH 7.5 | Random | 88% | 48.48 |
| 60°C · pH 7.5 | AI explore | 100% | 32.16 |
| 60°C · pH 7.5 | AI greedy | 100% | 29.44 |
| 60°C · pH 7.5 | Nearest sequence | 98% | 34.72 |
| 60°C · pH 7.5 | Random | 76% | 49.28 |

Unsuccessful campaigns remain in the denominator. Capped assays assigns 64 to non-reaching runs; it is NOT an estimate of their eventual time to success. Only batch-completion times are observed. No savings percentage is calculated from successful runs alone.

## Policies

- Random: uniform sampling without replacement from remaining candidates.
- Nearest sequence: predict each untested enzyme from the measured activity of its closest observed sequence; sample-ID tie breaks.
- AI greedy: rank the mean log1p activity predicted by 20 bootstrapped ridge models using frozen ESM-2 35M embeddings.
- AI explore: 75% percentile rank of ensemble mean + one bootstrap standard deviation, plus 25% percentile rank of minimum cosine distance to observed and already selected sequences. Recompute diversity after each batch choice; do not reveal batch outcomes until all eight are selected.
- Both AI policies use the same train-only scaler, bootstrap procedure and fixed ridge alpha 100. No tuning against this simulation’s outcomes. The embedding model was chosen for low cost after the earlier benchmark, so this remains exploratory.

## Interpretation limits

- The algorithm receives only revealed labels, but the dataset was previously inspected by the project author. This is not a blind evaluation.
- These are sequential searches within a finite, previously screened and expressed candidate pool, not sequence-family holdout tests or searches across arbitrary natural sequences.
- The same measured outcomes are reused across 50 starts. Percentile bands and paired win rates describe starting-set sensitivity, not biological confidence intervals.
- Bootstrap spread is an acquisition heuristic, not calibrated prediction uncertainty. Its contribution is combined with diversity; this four-policy experiment does not isolate the two.
- All assays are treated as equal-cost and repeat measurements are represented by their existing means. Expression failures, assay noise and real scheduling costs are not modeled.
- Different conditions have different measured candidate populations. Do not pool the condition results or treat their scores as directly comparable.
- No new enzyme is designed or experimentally validated. External or prospective validation remains future work.

**Compute:** 5.4 seconds for simulation and result export; existing frozen embeddings reused; $0 paid compute. Report rendering is additional.

## Reproduce

From the project folder:

```sh
.venv/bin/python -m pet_scout.campaign
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py
```

Requires prepared PETBench data and the validated ESM-2 35M cache. See CAMPAIGN.md for setup. `manifest.json` records configuration, source and result hashes. `selections.csv` records every revealed sample and its acquisition-time score. `checkpoints.csv` records every campaign checkpoint.

Source: [Norton-Baker et al., ACS Catalysis (2025)](https://doi.org/10.1021/acscatal.5c03460). Experimental data CC BY-NC 4.0.

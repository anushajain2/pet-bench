# Robustness and acquisition ablations

The settings in `validation_config.json` were fixed before this extension ran, after the earlier dataset and campaign results had been inspected. This is an exploratory analysis, not a preregistration.

## Harder starting sets

Reuse the prepared 60%-identity connected components. Eligible starting groups have at least 16 measured members and at least 48 measured candidates outside them. Evaluate every eligible group, using 50 seeds (1001–1050) per group, then weight anchor groups equally. The present data has two eligible groups per condition: group 0 and group 8. These labels describe graph components, not verified biological families.

Each campaign reveals 16 measurements sampled uniformly within its starting group. Exclude every unmeasured member of that group from acquisition. The remaining pool consists only of outside-group candidates. Select in batches of eight up to 64 total assays, with 40 fixed as the primary budget. Later batches can train on previously acquired outside-group labels: this is a test of initial distribution shift and adaptation, not permanent family holdout.

Run the original random-start scenario with the same six policies as a paired reference. Four policies must exactly reproduce the earlier campaign records. Candidate-pool sizes and available hits differ across scenarios: compare methods within each scenario, not a raw random-start versus shifted-start count as a causal estimate of difficulty.

## Fixed acquisitions

Use cached ESM-2 35M features, 20-member bootstrap ridge, alpha 100, train-only scaling and log1p activity. No hyperparameter search or winner-dependent model choice.

1. Random selection.
2. Nearest observed sequence.
3. AI greedy: largest ensemble mean.
4. AI uncertainty: largest mean + one bootstrap standard deviation.
5. AI diversity: 75% rank of mean + 25% rank of minimum cosine distance to observed/already-chosen embeddings.
6. AI explore: 75% rank of mean + standard deviation, plus 25% diversity rank (the original combined rule).

“Uncertainty-only” and “diversity-only” refer to the addition to the prediction baseline; neither discards predicted performance. Full batches are selected before their labels are revealed.

## Outcomes and diagnostics

The primary outcome is newly acquired hits after 24 additional assays, excluding starting discoveries. Keep the original whole-condition 90th-percentile hit threshold and include ties; it stays in the evaluator. Report distinct groups containing new hits, unfamiliar-group hits, hit recall among available candidates, and best newly observed activity as secondary metrics. Group-shift discoveries all occur outside the starting group. Random-start group counts may include groups represented initially; unfamiliar-group counts are separately available.

Report differences against random, nearest and greedy within the same task/scenario/anchor/seed/budget. Report each starting group as well as the equal-anchor mean. No pooling across conditions. The ablation includes the paired interaction (combined − uncertainty − diversity + greedy), so effects are not assumed additive.

At the first common decision, use an offline evaluator to relate bootstrap spread to absolute log1p prediction error across all untested candidates. Report within-run Spearman correlation, valid-correlation counts, and error by ranked uncertainty quartile (ID breaks partial ties). Runs with constant uncertainty scores have no ranking: mark their quartile as 0 and exclude them from quartile comparisons. Aggregate within runs before pooling. This is diagnostic data only; it cannot inform selection, tune weights, establish calibrated intervals, or by itself explain later adaptive trajectories. The constant-score handling was clarified during validation of the first run; it changes diagnostics only, not acquisitions or performance results.

## Run

Existing project setup and cached embeddings are reused:

```sh
.venv/bin/python -m pet_scout.validation
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py
```

Requires the completed original campaign for the reference-output manifest. No new dependencies or paid services. New results live in `validation/results/`; existing `campaign/results/` and `benchmark/results/` are preserved. The main module imports the original predictor and four policies rather than changing them.

Outputs: design eligibility and pool sizes; all checkpoints and selections; aggregate and per-anchor results; paired effects; first-decision candidate predictions and diagnostic summaries; hashes; three scientific figures; CASE_STUDY.md; DEMO.md.

The case study documents measured limitations and condition-specific findings. Further scientific validation would require compatible independent data or new laboratory measurements. A two-minute narrated video is not automatically recorded; DEMO.md provides the finished walkthrough script.

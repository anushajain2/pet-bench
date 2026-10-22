# PETBench experiment selection

Question: with a fixed laboratory screening budget, can an AI selection policy find more strong PET-degrading enzymes than random or sequence-based selection?

This is an exploratory extension of a previously inspected public dataset. A simulated campaign reveals existing activity measurements after each batch; it does not design or validate new enzymes. The four policies operate on the same candidate pool and shared starting measurements. Other-condition outcomes never enter a campaign.

## Fixed protocol

`campaign_config.json` is the machine-readable protocol. It was written before the first campaign run; the dataset and earlier model benchmark were already inspected.

- All three existing PETBench assay conditions, analyzed separately.
- 50 paired random starting sets per condition (seeds 1001–1050).
- 16 initial assays, followed by batches of eight; 40 total assays is the primary endpoint, 64 is secondary. Initial assays count toward cost and discoveries.
- Four policies: uniform random; nearest observed sequence; bootstrap-ensemble AI greedy; the same predictor with an uncertainty/diversity acquisition rule.
- Frozen ESM-2 35M embeddings for both AI policies. The model was selected for low cost after the earlier benchmark, not as a universally best representation.
- 20 bootstrap ridge fits per acquisition step, fixed alpha 100, log1p activity target. The scaler is fitted on revealed samples only. Each bootstrap fit has its own intercept. No simulation-based hyperparameter search.
- Greedy chooses highest ensemble mean log activity. Explore uses 75% percentile rank of mean + one standard deviation and 25% percentile rank of minimum cosine distance to observed or already selected embeddings. Diversity is updated within a batch; labels are not.
- Outcome targets: hits at 40, paired differences from random/nearest/greedy, best activity, fraction finding at least five hits, and batch-completion time to five. Unsuccessful runs remain in denominators; capped time assigns 64 to non-reaching runs and is explicitly not eventual time to success.
- A hit is activity at or above the within-condition 90th percentile (ties included). This evaluator-only definition uses full published outcomes to define a retrospective discovery target. Selection policies cannot access the threshold or unrevealed labels. It does not mean industrial usefulness.
- Starting-set bands are descriptive 10th–90th percentiles, not biological confidence intervals. All starts reuse the same measured enzymes.

Sequence embeddings and similarities for all candidates can be known before selection. The `choose_batch` API receives only observed activities; the full measurement array exists only in the simulation/evaluation layer. Every batch is chosen before any of its outcomes are passed back to the policy. Candidate IDs resolve ties independently of outcomes.

## Run

From the existing installed project:

```sh
.venv/bin/python -m pet_scout.campaign
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py
```

Open **Experiment planner** in the sidebar. Existing benchmarks remain accessible.

For a fresh checkout, install the project as described in README, then prepare data and embeddings:

```sh
.venv/bin/pet-bench prepare
.venv/bin/pet-bench embed --model esm2_35m
.venv/bin/python -m pet_scout.campaign
```

No new dependencies, API keys, paid GPU or model download are needed if the previous ESM cache is present. The full benchmark is not required to run campaigns, but its prepared catalog, source workbook, splits and validated feature cache are required.

Outputs under `campaign/results/`: checkpoint curves, every selected sample and acquisition score, paired comparisons, censored target times, summary, scientific figure, findings and provenance manifest. Results are reproducible; timings may vary. The dashboard checks source, configuration, simulation-code and result checksums before rendering.

## Limits and useful next validation

This evaluates search within a finite pool of already screened and expressed proteins. It is not sequence-family holdout generalization, mutation design, a prospective experiment, or industrial yield prediction. Repeated assays are represented by their existing means; label noise, expression failure and varying assay costs are not simulated. Bootstrap spread is uncalibrated; the combined exploration policy does not isolate uncertainty from diversity. Model pretraining overlap is unknown. External validation requires another compatible experimental dataset or prospective measurements, not a new random seed on this dataset.

Do not choose a favorable seed, condition or budget after seeing the output and present it as the primary result. Report all three conditions and the fixed 40-assay endpoint, even when exploration loses.

Experimental data: Norton-Baker et al. (2025), https://doi.org/10.1021/acscatal.5c03460, CC BY-NC 4.0. Software license remains MIT.

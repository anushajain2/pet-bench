# PET Scout — experimental findings

## Question and result

Can frozen protein-language-model embeddings help select PET-degrading enzymes,
beyond choosing the nearest known sequence, when close relatives are held out?

The ESM advantage changed sign across grouped partitions. This small study does not establish a robust selection advantage over sequence similarity.

- **Random split:** ESM minus nearest-sequence top-5 mean activity = **-3.25 µmol/mg**, with seed-level differences from -8.10 to +4.30.
- **Matched split:** ESM minus nearest-sequence top-5 mean activity = **-0.92 µmol/mg**, with seed-level differences from -4.80 to +2.34.
- **Grouped split:** ESM minus nearest-sequence top-5 mean activity = **+2.32 µmol/mg**, with seed-level differences from -5.14 to +8.68.

These are completed retrospective experiments on published measurements. No enzyme
was synthesized, mutated, or newly validated. The source authors already investigated
ML prediction; PET Scout is a focused reproducible extension, not a discovery claim.

## Data

- Source: Norton-Baker et al., *ACS Catalysis* (2025), https://doi.org/10.1021/acscatal.5c03460.
- Condition: crystalline PET powder, 60°C, pH 7.5, 48 hours; 3.5 mg PET and 5 µg enzyme per 0.5 mL reaction.
- Outcome: released aromatic products, µmol per mg enzyme, as reported in the source workbook.
- 159 unique construct sequences from 236 assay rows.
- 82 sequences have only one measurement in this condition.
- 0 identical-sequence aliases collapsed before splitting.
- Ambiguous sequence IDs excluded: DP085.
- Missing assays and failed expression are never treated as inactive enzymes.
- Condition fixed in config before model evaluation; no favorable-condition search.

## Top-five selection results

Each number averages the mean measured activity of five selections **within an
outer held-out fold**, first over folds and then over seeds. Fold pools differ
between split regimes; compare against their own random expectation. The five-fold
protocol spends five selections per fold, not five selections for the whole dataset.

| Split | Method | Selected activity | Random expectation | Lift | Opportunity captured |
|---|---|---|---|---|---|
| Sequence-group holdout | Composition ridge | 9.10 | 10.51 | -1.41 | -0.00 |
| Sequence-group holdout | ESM-2 ridge | 19.52 | 10.51 | 9.01 | 0.29 |
| Sequence-group holdout | Nearest sequence | 17.20 | 10.51 | 6.68 | 0.33 |
| Size-matched random | Composition ridge | 18.99 | 6.87 | 12.13 | 0.57 |
| Size-matched random | ESM-2 ridge | 20.02 | 6.87 | 13.15 | 0.67 |
| Size-matched random | Nearest sequence | 20.94 | 6.87 | 14.07 | 0.53 |
| Random holdout | Composition ridge | 19.38 | 7.14 | 12.25 | 0.38 |
| Random holdout | ESM-2 ridge | 23.03 | 7.14 | 15.89 | 0.47 |
| Random holdout | Nearest sequence | 26.28 | 7.14 | 19.14 | 0.53 |

Opportunity captured = (selected mean − random expected mean) /
(oracle top-five mean − random expected mean). It can be negative. It is undefined
for pools with no activity variation and is omitted for those pools.

## Reproducible protocol

1. Average source replicates per candidate ID, then average identical-sequence ID means.
2. Compute global BLOSUM62 alignments, gap-open −10 and gap-extension −0.5.
   Identity = identical residues / all alignment columns, including gaps.
3. Build connected components of sequence pairs with identity ≥60%.
   There are 30 groups; the largest contains 86 sequences.
4. Compare five-fold random holdouts with size-balanced group holdouts, using
   seeds [11, 23, 37, 51, 71]. Each candidate is tested once per seed.
5. Freeze ESM-2 35M; mean-pool amino-acid tokens, excluding special and padding tokens.
   Compare ESM ridge, amino-acid-composition ridge, and nearest-sequence prediction.
6. Fit log1p(activity) with ridge regression; choose alpha from
   [10.0, 100.0, 1000.0] using training-only inner validation.
   Fit scaling inside each training fold. Inner validation respects the outer grouping policy.
7. Select by model prediction, using candidate ID only to break equal-score ties.
   Score selections on held-out observed activity. Report exact random expectations
   and Monte Carlo random batch ranges for the same pools.

Largest observed cross-split identity in grouped evaluation: 0.598.
Sequence grouping is label-independent and transitive. It does not prove that
proteins belong to different functional families or share no conserved domains.

## Fold-size diagnostic

The largest connected component contains 86 of 159 proteins,
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

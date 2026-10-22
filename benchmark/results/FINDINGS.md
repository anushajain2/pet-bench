# PETBench v1 — findings and protocol

## What was tested

Which frozen protein representation helps select five PET-degrading enzymes for a
specified temperature and pH? Six methods were evaluated on the same labeled data,
training subsets and held-out pools: nearest sequence, composition ridge, ESM-2 35M,
ESM-2 150M, ESM-2 650M and the ProtT5-XL encoder, each embedding method using ridge.
This compares complete frozen-embedding-plus-ridge pipelines; it does not rank every
possible use of these foundation models or test designed mutations.

Completed 5,400 outer model fits / predictions and
190,440 candidate-level held-out predictions. The unit of an
outer fit is one task, split regime, partition seed, fold, training budget and method;
inner ridge validation adds additional fits. The nearest-sequence method needs no
parameter fitting. Paid compute used: $0.

## Data coverage

| label | n_proteins | n_assay_rows | n_groups | largest_group | single_measurement_proteins |
| --- | --- | --- | --- | --- | --- |
| 40°C · pH 7.5 | 211 | 288 | 36 | 94 | 134 |
| 40°C · pH 4.5 | 159 | 222 | 25 | 88 | 96 |
| 60°C · pH 7.5 | 159 | 236 | 27 | 86 | 82 |

All tasks use crystalline PET powder, 48 hours, 3.5 mg PET, 5 µg enzyme, 0.5 mL reaction.
Outcome: µmol aromatic products per mg enzyme. Measurements are source assay endpoints,
not industrial recycling yields. Replicates are averaged before splitting. Controls
and ambiguous sequences are excluded. Unmeasured assays are not treated as zero.
The catalog contains 211 unique sequences;
measurements across conditions are not additional independent proteins.

## Primary comparison

The tables below use sequence-group holdouts, all available training examples, and a
five-candidate selection budget per test pool. Rows are sorted by observed mean for
readability. A first-place mean alone does not establish a reliable model advantage.
Selected activity and differences use µmol/mg. Seed ranges describe split sensitivity.

## 40°C · pH 7.5

ESM-2 35M exceeds nearest-sequence selection in every evaluated partition seed. This is consistent within these partitions, not independent experimental confirmation.

| Method | Selected activity | Random expectation | Difference vs nearest | Min seed difference | Max seed difference | Mean fold Spearman | Valid rank folds / 25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ESM-2 35M | 34.90 | 14.70 | 17.85 | 14.34 | 23.34 | 0.41 | 25 |
| ProtT5-XL encoder | 24.06 | 14.70 | 7.00 | 2.62 | 13.70 | 0.30 | 25 |
| ESM-2 150M | 23.38 | 14.70 | 6.33 | 1.04 | 13.72 | 0.37 | 25 |
| Composition ridge | 18.48 | 14.70 | 1.42 | -3.56 | 6.48 | 0.27 | 25 |
| ESM-2 650M | 18.34 | 14.70 | 1.29 | -2.98 | 4.34 | 0.28 | 25 |
| Nearest sequence | 17.05 | 14.70 | 0.00 | 0.00 | 0.00 | 0.20 | 25 |

## 40°C · pH 4.5

ProtT5-XL encoder exceeds nearest-sequence selection in every evaluated partition seed. This is consistent within these partitions, not independent experimental confirmation.

| Method | Selected activity | Random expectation | Difference vs nearest | Min seed difference | Max seed difference | Mean fold Spearman | Valid rank folds / 25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ProtT5-XL encoder | 3.50 | 1.97 | 1.78 | 1.54 | 2.44 | 0.20 | 24 |
| ESM-2 150M | 3.46 | 1.97 | 1.74 | 1.34 | 2.84 | 0.15 | 24 |
| ESM-2 35M | 3.06 | 1.97 | 1.35 | 0.70 | 1.80 | 0.03 | 24 |
| ESM-2 650M | 2.46 | 1.97 | 0.75 | 0.48 | 1.18 | 0.16 | 24 |
| Nearest sequence | 1.71 | 1.97 | 0.00 | 0.00 | 0.00 | -0.06 | 21 |
| Composition ridge | 0.58 | 1.97 | -1.14 | -1.62 | -0.70 | -0.04 | 24 |

## 60°C · pH 7.5

ProtT5-XL encoder exceeds nearest-sequence selection in every evaluated partition seed. This is consistent within these partitions, not independent experimental confirmation.

| Method | Selected activity | Random expectation | Difference vs nearest | Min seed difference | Max seed difference | Mean fold Spearman | Valid rank folds / 25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ProtT5-XL encoder | 25.00 | 10.74 | 6.85 | 3.96 | 9.64 | 0.37 | 25 |
| Nearest sequence | 18.15 | 10.74 | 0.00 | 0.00 | 0.00 | 0.28 | 25 |
| ESM-2 35M | 17.39 | 10.74 | -0.76 | -6.92 | 7.58 | 0.30 | 25 |
| ESM-2 650M | 16.29 | 10.74 | -1.86 | -4.86 | 1.36 | 0.29 | 25 |
| ESM-2 150M | 16.27 | 10.74 | -1.88 | -3.54 | -0.26 | 0.30 | 25 |
| Composition ridge | 10.49 | 10.74 | -7.66 | -14.94 | -3.24 | 0.05 | 25 |

## Learning curves and controls

Training budgets were fixed to 16, 32, 64 and all available examples. Subsets are nested,
drawn without labels, and shared across all six methods. Each numeric budget is used
only if every fold in that task can support it. All three tasks support all budgets.
The `all` budget varies with the outer fold size; it is not a fixed sample count.

Random, size-matched random and sequence-group holdouts are reported separately.
Size-matched random has the same train/test sizes as grouped holdout, but different
candidate composition. It does not isolate a causal effect of sequence novelty.
See `summary.csv` for every task/regime/budget/model and `paired_comparison.csv` for
same-seed differences against nearest sequence. Budgets one and three are secondary
selection analyses. No single pooled winner across conditions is reported.

## Evaluation rules

- Five outer folds, repeated for seeds [11, 23, 37, 51, 71].
- Global sequence groups are connected components at ≥60% identity across the union
  of task proteins. Identity uses global BLOSUM62 alignment, gap-open −10,
  gap-extension −0.5, with all gap columns in the denominator.
- Each task uses only labels measured at that condition. No other-condition labels
  are supplied to models. Related proteins remain together in grouped outer folds.
- Pretrained representations are frozen; amino-acid tokens are mean-pooled, excluding
  special tokens and padding. IDs, sequences, exact model revisions and files are checked.
- Ridge predicts log1p(activity); StandardScaler and alpha selection operate only on
  outer training data. Up to three inner folds use the outer grouping policy.
  The shared alpha grid is [10.0, 100.0, 1000.0]. If fewer than two training
  groups remain, alpha 100 is used and recorded instead of leaking across groups.
- Selection uses predicted rank, with candidate ID as the tie-breaker. The outcome
  is mean measured activity among selected candidates. Random expectation is exactly
  the mean of the same test pool; the oracle uses its best measured candidates.
- Aggregate folds equally, then seeds. Small and large pools get equal weight because
  the experimental selection budget is the same per pool. Spearman is undefined for
  constant predictions/outcomes; missing correlations are omitted, never set to zero.
- All checkpoint choices, conditions and budgets were fixed before these new runs.
  The 60°C/pH 7.5 condition was studied earlier; this is an exploratory extension of
  public data, not a prospective blind benchmark.

## Measured compute

| Model | Inference seconds | Load/download seconds | Peak process GiB | Dimensions |
| --- | --- | --- | --- | --- |
| ESM-2 35M | 6.85 | 0.68 | 1.03 | 480 |
| ESM-2 150M | 20.80 | 76.13 | 1.62 | 640 |
| ESM-2 650M | 58.43 | 345.28 | 3.61 | 1280 |
| ProtT5-XL encoder | 99.97 | 797.00 | 5.65 | 1024 |

All four checkpoints ran locally, sequentially, on CPU in float32 with four Torch
threads. Each model ran in a separate process. Inference timing excludes loading;
load timing includes downloads/cache access. Peak RSS includes the whole process.
Batch sizes differ, so these are practical costs of this run, not controlled hardware
throughput comparisons. Embeddings cover the same complete protein catalog and are
reused across tasks, seeds and training budgets. Evaluation took
49.0 seconds. Model metadata records precision, revisions,
parameter counts, peak memory, package versions and download sizes.

## Limits

- Exploratory extension of an already inspected published dataset. No blind novelty or mutation-effect claim.
- Task populations differ because some proteins were not measured at every condition.
- Frozen sequence embeddings plus one shared ridge protocol; rankings apply to this pipeline, not every use of each foundation model.
- A large connected group causes uneven held-out pools; matched random controls fold sizes only.
- Each model receives exactly the same examples at each budget. Other-condition labels are not used.
- Mean results weight each test pool equally. Seed ranges describe partition sensitivity, not independent replications or confidence intervals.
- Protein pretraining exposure is unknown; label isolation does not establish pretraining decontamination.
- Single-study, ML-screened, expressed proteins; experimental uncertainty is not fully propagated.
- Compute is one local float32 CPU run per model with different batch sizes, not a standardized hardware throughput benchmark.

The source paper already studied PET-hydrolase discovery and machine learning.
PETBench provides an auditable comparison on this source; it is not the first PET
benchmark, a discovery of a new enzyme, or evidence that a mutation improves function.
A follow-up prospective or genuinely independent experimental dataset is needed to
test how far these rankings generalize. Choosing methods after viewing this leaderboard
requires fresh evaluation before claiming an unbiased performance estimate.

## Reproduce

From the project directory: `pet-bench all`, then `python -m pytest -q`.
See `BENCHMARK.md` for installation and individual stages. The original PET Scout
study remains under `reports/` and its original `config.json`.

## Attribution

- Norton-Baker et al. (2025), *Machine Learning-Guided Identification of PET Hydrolases
  from Natural Diversity*: https://doi.org/10.1021/acscatal.5c03460
- Experimental workbook: https://acs.figshare.com/articles/dataset/30044876
  (CC BY-NC 4.0 per source record; see `DATA_LICENSE.md`).
- ESM-2: https://github.com/facebookresearch/esm
- ProtT5 encoder: https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc

Code is MIT. Do not treat the code license as replacing dataset or model licenses.

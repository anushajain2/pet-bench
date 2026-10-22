# PETBench v1

Which protein representation helps choose PET-degrading enzymes for a particular
temperature and pH, and how many measured examples does it need?

PETBench extends the original PET Scout study into a six-method benchmark across
three fixed conditions. It measures actual held-out candidate selection, ranking,
data efficiency and practical local compute costs. It is a retrospective single-study
benchmark of **natural enzyme selection**, not a mutation-effect or generation benchmark.

## Run the dashboard

```bash
.venv/bin/streamlit run app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Select task, split regime, training budget and number of
candidates in the sidebar. The original PET Scout study remains available through
the **Original study** page; its data, reports and CLI are unchanged.

## Reproduce

Use Python 3.12. Exact dependency versions are in `requirements.lock`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/pet-bench all
.venv/bin/python -m pytest -q
```

Or run individual stages:

```bash
.venv/bin/pet-bench prepare
.venv/bin/pet-bench embed
.venv/bin/pet-bench evaluate
.venv/bin/pet-bench report
```

`--root /absolute/path/to/pet-scout` works from another directory. Embedding runs use
separate sequential processes to release each model's memory and make peak-memory
measurements interpretable. CPU float32 is the reference execution. CUDA or MPS can be
selected with `--device`, but device changes may affect values and timing. Existing
validated caches are reused; switching the flag alone does not recompute them.

Run just one checkpoint with, for example:

```bash
.venv/bin/pet-bench embed --model prott5
```

The first run downloads public model checkpoints; no paid inference API, cloud GPU,
private credential or sequence upload is required. Hugging Face receives normal public
model-download requests. Subsequent runs use pinned checkpoint and embedding caches.
Treat the downloaded models as separately licensed dependencies. ESM's unused masked-
language-model head can appear as unexpected keys when loading the encoder; it is not
used for these embeddings. ProtT5 requires SentencePiece.
Automatic remote safetensors conversion is disabled; ProtT5 loads the official pinned
PyTorch checkpoint. This avoids an unrelated background conversion thread keeping
an otherwise completed embedding process alive.

## Fixed protocol

`benchmark_config.json` specifies three crystalline-PET conditions: 40°C/pH7.5,
40°C/pH4.5 and 60°C/pH7.5. All use a 48-hour assay. Temperature and pH define separate
tasks; they are not additional independent proteins or regression input features.

Six methods: nearest sequence; amino-acid composition plus length with ridge;
ESM-2 35M, 150M and 650M; and ProtT5-XL encoder. Every representation uses residue-only
mean pooling and the same StandardScaler + ridge pipeline, log1p outcome, and
training-only tuning over alpha [10,100,1000]. This isolates a practical representation
comparison rather than optimizing each architecture separately.

Outer protocol: five folds, five partition seeds, random / size-matched random /
sequence-group holdout. Groups are computed over the union of proteins, at 60% global
alignment identity. Train budgets are 16,32,64 and all; subsets are nested and shared.
Numeric sizes are included only if feasible for every fold in a task. Each task uses
only its own observed assay labels; unmeasured outcomes are never zero-filled.

Global grouping may connect proteins through additional sequences from another task.
Consequently the 60°C/pH7.5 grouped results can differ from the original PET Scout
study, which grouped only that condition's proteins. Both protocols are preserved;
do not interpret that difference as an improvement caused by a model change.

Primary metric: mean measured activity of five candidates selected **inside each
held-out pool**, compared with the same-pool random expectation and nearest-sequence
selection. Selection budgets one and three are secondary. Folds get equal weight,
then seeds get equal weight. Report seed-level ranges as partition sensitivity, not
confidence intervals. All scores are condition-specific; there is no overall winner
obtained by pooling incompatible assays or different candidate populations.

The same 60°C/pH7.5 data were explored in PET Scout before this benchmark extension.
This is not a pristine blind evaluation. Roster and protocol were fixed before the
new comparison runs. Choosing a model after inspecting the leaderboard requires a
new independent evaluation before claiming unbiased selection performance.

## Outputs

```text
benchmark_config.json                Fixed task/model/evaluation specification
benchmark/proteins.csv                Canonical sequence catalog
benchmark/tasks/*/candidates.csv      Condition-specific measured labels
benchmark/groups.csv                 Global sequence groups
benchmark/splits.csv                 Every training subset and held-out pool
benchmark/data_manifest.json          Data and split checksums
benchmark/revisions/*.json            Exact model checkpoint revisions
benchmark/embeddings/*.npz            Cached sequence-aligned representations
benchmark/embeddings/*.json           Provenance, timing, peak memory and precision
benchmark/results/summary.csv         Task/split/budget-specific comparison
benchmark/results/seed_metrics.csv    Partition sensitivity
benchmark/results/paired_comparison.csv  Same-seed contrasts with nearest sequence
benchmark/results/predictions.csv     Every outer-held-out prediction
benchmark/results/metrics.csv         Selected IDs, activity, oracle and random baseline
benchmark/results/fit_timings.csv     Downstream fit/predict timing and selected alpha
benchmark/results/compute.csv         One-time embedding costs
benchmark/results/manifest.json       Run configuration, checksums and limitations
benchmark/results/FINDINGS.md         Actual computed findings
benchmark/results/benchmark_overview.png  Static shareable figure
```

Feature caches check sequence order, model revision, pooling version, ID order,
finite values and file checksum. Evaluation requires every configured model; it
fails rather than quietly ranking an incomplete roster. Data/config changes require
preparing again; mismatched feature caches fail explicitly rather than being silently
reused. To intentionally rerun a model, move its `.npz` and `.json` cache pair to a
separate archival directory first. Keep the revision pin for reproducibility.

## Interpretation and attribution

The models have unknown protein-pretraining overlap. Sequence grouping controls
assay-label leakage but cannot establish absence of pretraining exposure. The source
proteins were selected through an ML-assisted campaign and successfully expressed;
these scores do not describe random proteins from nature. Large connected groups
produce unequal test pools. Full-construct identity may hide shared domains.
One local timing run with model-specific batch sizes is not a hardware benchmark.

Dataset: Norton-Baker et al. (2025), *Machine Learning-Guided Identification of PET
Hydrolases from Natural Diversity*, https://doi.org/10.1021/acscatal.5c03460.
The original authors already investigated ML and condition-specific activity.
PETBench is an evaluation extension, not a new enzyme discovery or the first PET benchmark.

The experimental supplement and derived measurements retain CC BY-NC 4.0 terms
according to their Figshare record. Code is MIT; see `DATA_LICENSE.md`.

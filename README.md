# PETBench

Reproducible AI-guided screening of PET-degrading enzymes, with model benchmarks,
sequential experiment selection, robustness tests and an interactive dashboard.

```sh
git clone https://github.com/anushajain2/pet-bench.git
cd pet-bench
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m streamlit run app.py
```

Published measurements, prepared features and completed results are included;
viewing the dashboard does not require downloading model weights or paid APIs.
Code is MIT-licensed; experimental data and derived measurement tables retain
their [CC BY-NC 4.0 terms](DATA_LICENSE.md).

**Portfolio case study: when does AI help select PET-degrading enzymes under a
limited experimental budget?** The final extension tests narrow starting groups,
separates uncertainty from diversity, and audits prediction uncertainty.

- Read: [case study](validation/results/CASE_STUDY.md)
- Present: [two-minute demo script](validation/results/DEMO.md)
- Reproduce: `.venv/bin/python -m pet_scout.validation`
- Audit: [fixed protocol](VALIDATION_STUDY.md) and [completion checklist](VALIDATION_TASKS.md)
- Explore: **Validation & case study** in the dashboard sidebar.

**New: experiment-selection simulator.** Can a lab find more strong PET-degrading
enzymes with only 40 assays? Four policies, 50 paired starting sets per condition,
three conditions, and complete batch-by-batch audit trails. Open **Experiment planner**
in the dashboard sidebar.

- Run: `.venv/bin/python -m pet_scout.campaign`
- Protocol: [CAMPAIGN.md](CAMPAIGN.md)
- Results: [campaign/results/FINDINGS.md](campaign/results/FINDINGS.md)
- Checklist: [CAMPAIGN_TASKS.md](CAMPAIGN_TASKS.md)

**PETBench is the current dashboard:** a reproducible comparison of six enzyme-selection
methods across three PET assay conditions and four training-data budgets.

- Start: `.venv/bin/streamlit run app.py --server.address 127.0.0.1`
- Reproduce: `.venv/bin/pet-bench all`
- Protocol and installation: [BENCHMARK.md](BENCHMARK.md)
- Completed findings: [benchmark/results/FINDINGS.md](benchmark/results/FINDINGS.md)
- Build checklist: [BENCHMARK_TASKS.md](BENCHMARK_TASKS.md)

The original PET Scout study is preserved below and in the dashboard's **Original study** page.
Its standalone entry point is `original_app.py`; its commands and artifacts remain unchanged.

---

# PET Scout

**Can protein-language-model embeddings help select plastic-degrading enzymes
when close relatives are absent from the training data?**

PET Scout is a completed, reproducible small-data study and an interactive explorer
of its actual held-out predictions. It uses published enzyme measurements rather
than synthetic demonstration data.

The practical decision is: *which five proteins should a researcher test from a
candidate pool?* The experiment compares frozen ESM-2 embeddings with sequence
similarity, amino-acid composition, and random selection.

## Start the explorer

From this folder, using the environment installed during the build:

```bash
.venv/bin/streamlit run original_app.py --server.address 127.0.0.1
```

Open http://localhost:8501. The supplied results and cached features make the
explorer usable without downloading a model or training anything. It only reads
local artifacts. It does not call a paid API, upload sequences, or run a wet lab.

## What is included

- Original public experimental workbook, download URL, checksum and attribution.
- A checked dataset for **60°C, pH 7.5, crystalline PET powder, 48 hours**.
- Frozen **ESM-2 35M** sequence embeddings, with an exact model revision.
- Global alignment identities and transitive sequence groups at a 60% threshold.
- Five-fold evaluation repeated over five partition seeds.
- Random, sequence-grouped, and diagnostic size-matched random holdouts.
- Training-only inner tuning for ridge regressors; no held-out labels used to select candidates.
- Every prediction, nearest training neighbor, fold membership and selected batch.
- Interactive results, a static chart, tests and a written findings report.

Read **[reports/FINDINGS.md](reports/FINDINGS.md)** for the actual findings and
**[TASKS.md](TASKS.md)** for the build checklist. A larger model is not automatically
better; no hyperparameter or condition search was performed to manufacture an advantage.

## Reproduce from a fresh environment

Python 3.12 is recommended. The lock file records the exact versions used on macOS
Apple Silicon; platform-specific wheels may differ on other systems.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/pet-scout all
.venv/bin/python -m pytest -q
.venv/bin/streamlit run original_app.py --server.address 127.0.0.1
```

Alternatively, use `uv venv --python 3.12` and `uv pip install -r requirements.lock`
followed by `uv pip install --no-deps -e .`.

Stages can also be run independently:

```bash
pet-scout prepare
pet-scout embed --device cpu --batch-size 8
pet-scout evaluate
pet-scout report
```

Run these commands in the project directory, or pass `--root /absolute/path/to/pet-scout`.
The first embedding run requires internet access to download the model. Subsequent
runs reuse cache files after checking sequence identity. CUDA and MPS are optional;
this build ran on CPU. Small numerical differences between devices can affect ties.
No GPU rental is needed for this dataset.

To force fresh embeddings, remove `data/processed/embeddings.npz` and its `.json`
sidecar. The model remains pinned in `model_revision.json`. To choose another model,
change `config.json`, regenerate embeddings and rerun evaluation; label that as a
new experiment rather than replacing the original finding without explanation.

## Scientific design

**Outcome:** µmol aromatic products released per mg enzyme after 48 hours. This is
an assay of PET breakdown. It is not a melting temperature, a probability, a rate
constant, or an industrial recycling yield.

**Data units:** assay replicates are averaged per candidate ID. Identical sequences
are collapsed before splitting. Controls are removed. One ambiguous sequence is
excluded and recorded. Missing assays and expression failures are never imputed as
zero activity. Source constructs are used without guessed tag/domain trimming.

**Models:** nearest-sequence activity; 20 amino-acid frequencies plus log sequence
length with ridge; frozen ESM-2 mean-pooled representations with ridge. Regressors
fit `log1p(activity)`; inverse predictions are clipped at zero. Scaling and alpha
selection happen inside outer training data. Inner splits follow the same homology
policy as outer splits. Alphas are fixed to [10, 100, 1000].

**Similarity:** global BLOSUM62 alignment with gap-open −10 and extension −0.5.
Identity is identical residues divided by all alignment columns, including gaps.
Connected components ensure that no pair at or above 60% identity crosses a grouped
split. This is full-construct identity, not a claim of family-level independence.

**Selection metric:** select the top five predictions **within each held-out fold**
and compute their mean measured activity. Compare against the exact expected mean
of randomly selecting five from that same pool, plus a hindsight oracle. One full
five-fold pass uses five separate selection batches. Budgets one and three are
secondary sensitivity analyses. Ties are broken by candidate ID, never by outcomes.

**Uncertainty:** seed-level minima/maxima describe partition sensitivity, not
confidence intervals. Monte Carlo random-batch ranges describe random selection,
not uncertainty in model predictions. Repeated partitions reuse the same proteins.

**Imbalance diagnostic:** an 86-protein connected component creates unequal grouped
folds. The size-matched random control was added after the initial run exposed this
issue. It matches training/test counts but not candidate composition. This addition
is explicitly exploratory; the primary condition and models were not retuned.

## Limits and intended use

This is a retrospective evaluation on proteins already selected through a published
ML-assisted discovery campaign. Its results are conditional on successful expression
and measurement. The source authors already investigated supervised ML; this is a
focused evaluation extension, not a new discovery or a reproduction of every paper
result. Frozen ESM pretraining may contain these proteins or relatives; only assay-label
training leakage is controlled.

The study does not generate or validate a new enzyme, establish effects of mutations,
or prove performance on unseen industrial conditions. Use the experiment to decide
whether this particular ranking method deserves further testing.

## Repository map

```text
app.py                         Interactive, read-only results explorer
config.json                    Fixed assay/model/evaluation specification
model_revision.json            Exact pretrained-model revision
src/pet_scout/data.py           Acquisition, integrity checks, preprocessing
src/pet_scout/features.py       Alignments, grouping and frozen embeddings
src/pet_scout/evaluate.py       Nested evaluation and candidate selection
src/pet_scout/report.py         Findings report and static figure
src/pet_scout/cli.py            Reproduction commands
data/raw/                      Original source and provenance
data/processed/                Checked candidates and cached features
reports/                       Results, audit trail, report and figure
tests/                         Scientific isolation and interface checks
```

## Attribution

Norton-Baker B, Komp E, Gado JE, et al. (2025). *Machine Learning-Guided Identification
of PET Hydrolases from Natural Diversity.* ACS Catalysis 15, 16070–16083.
https://doi.org/10.1021/acscatal.5c03460

Source data: https://acs.figshare.com/articles/dataset/30044876

ESM-2: Lin et al. (2023). *Evolutionary-scale prediction of atomic-level protein
structure with a language model.* Science 379, 1123–1130.
https://doi.org/10.1126/science.ade2574
Model: https://huggingface.co/facebook/esm2_t12_35M_UR50D

Project code: MIT. Experimental supplement and derived measurements: **CC BY-NC 4.0**
according to its Figshare record. See [DATA_LICENSE.md](DATA_LICENSE.md).

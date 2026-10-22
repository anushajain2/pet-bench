# Independent transfer test and candidate ranking

## The practical question

Can measurements from one lab help choose three PET-degrading enzymes from another
lab's candidate panel? Separately, can a researcher upload their own measured
proteins and get a transparent proposal for the next batch?

The **External validation** page answers the first question using a fixed
retrospective analysis. **Rank candidates** implements the second workflow.
They are distinct: uploaded rankings train only on the supplied measurements;
they do not silently borrow activities from either public study.

## External-data feasibility audit

Training data: Norton-Baker et al., ACS Catalysis (2025),
<https://doi.org/10.1021/acscatal.5c03460>. Crystalline PET powder, pH 7.5,
48 h; 211 constructs at 40°C and 159 at 60°C. Activities are product amounts
normalized by enzyme mass.

External data: Wu et al., Nature Communications 16, 6211 (2025),
<https://doi.org/10.1038/s41467-025-61599-z>. Figure 2c supplies triplicate
TPA + MHET concentrations after 72 h on amorphous PET film at pH 9.
This changes substrate, pH, duration and normalization. We evaluate **ranking
within each external assay**, not calibrated cross-study activity predictions.
The external paper predates the training paper: this is not a temporal forecast.

Public inputs are archived under `external/raw/` with checksums in
`external/manifest.json`. The original supplement archive can be retrieved at
<https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12228687/supplementaryFiles>.
Source license: CC BY 4.0; see DATA_LICENSE.md for separate training-data terms.

- Supplementary Data 1 contains 34 synthesized **DNA** constructs. The paper
  documents NdeI/NotI cloning. The importer requires the exact flanks, removes
  the leading CAT and terminal TAA/NotI sequence, translates complete codons,
  retains the start methionine and rejects internal stops. Vector-derived
  His-tag residues are unavailable and are not invented.
- Figure 2c provides 14 APET candidates at five temperatures. The panel was
  selected through a model-based discovery pipeline, expression and a pNPB
  activity screen. It is not an unbiased sample of arbitrary sequences.
- Only actual PET measurements are used. pNPB activity and melting temperature
  are not substitute labels. IsPETase and other comparison enzymes are excluded.
  Missing/unexpressed proteins are not assigned zero activity.
- APET-15 is the same near-full-length construct as DP093 (one extra terminal
  residue), so it is excluded against the entire training catalog.
- APET-14 and APET-17 have nearest global training identities of 69% and 81%.
  The secondary similarity filter removes them, leaving 11 candidates below 60%.
  Global identity includes gap columns; this filter does not prove family independence.

## Fixed evaluation rules

`external_protocol.json` was written before model-performance calculation.
This is retrospective and was not preregistered or blinded: the source paper
identifies its successful enzyme. No hyperparameters were selected on external
labels. Both temperatures, all three methods and both eligible cohorts are shown.

The primary pair is training 40°C / pH 7.5 → external 40°C / pH 9. The secondary
pair is 60°C → 60°C. Frozen ESM-2 35M and amino-acid composition each use the same
20-member bootstrap ridge ensemble, alpha 100, train-only standardization,
log1p activity and seed 2026. Nearest-sequence prediction uses only its closest
measured training neighbor. Ties are resolved by sample ID.

We report Spearman correlation and the measured mean activity of three selected
candidates. Every possible random triple is enumerated for an exact finite-panel
reference, along with an outcome-aware oracle. The random-tail fraction is not
a generalization p-value; assay replicate uncertainty is not propagated.
The two temperatures reuse the same proteins and are not independent replications.
The distant subset has only 11 proteins; small absolute activity differences
should not be interpreted as established improvements beyond experimental noise.
Unknown overlap with protein-model pretraining remains a limitation.

Reproduce without paid APIs or GPU:

```sh
.venv/bin/python -m pet_scout.external --root .
.venv/bin/python -m pet_scout.transfer_report --root .
```

The included, checksummed public embeddings allow this to run without downloading
weights. Changing scientific inputs invalidates the dashboard manifest instead
of silently displaying stale results. Read `external/FINDINGS.md` and
`external/transfer.png` for the resulting comparison.

## Rank your own candidates

Use the dashboard's **Rank candidates** page, or the CLI below. A combined UTF-8
CSV needs `sample_id,sequence,activity`. Leave activity **blank** only for untested
proteins; `0` is a measured zero. NaN, NA, infinity and negative values are rejected.
Alternatively supply a FASTA plus `sample_id,activity` CSV; first FASTA header
tokens identify records, and omitted measurement IDs are untested.

Requirements: 8 or more measured sequences with varying nonnegative activities,
one or more untested candidates, at most 500 total and 1–1022 canonical amino
acids per sequence. IDs and sequences must be unique. Measured/candidate exact
near-full-length substring duplicates are rejected. No ambiguous residues or
silent sequence truncation. Average genuinely comparable replicates first.

Every measured protein must use the same substrate, pH, temperature, duration,
units and normalization. Optional `condition`, `assay` or `units` columns must
be consistent. The app requires an explicit comparability confirmation; it
cannot determine from sequences whether experimental conditions actually match.
Eight measurements is an operational minimum, not proof of sufficient sample size.

Public examples use a deterministic 40-protein subset of the original 40°C assay
(pandas random_state 2026), with 16 activities retained and 24 hidden. They are
workflow demonstrations, not additional evaluation results. Attribution and
CC BY-NC 4.0 apply to the example measurements.

```sh
.venv/bin/python -m pet_scout.ranking --root . \
  --csv examples/ranking_demo.csv \
  --condition '40 C, pH 7.5, crystalline PET powder, 48 h' \
  --units 'umol products / mg enzyme' --batch-size 3 \
  --output user_runs/demo

.venv/bin/python -m pet_scout.ranking --root . \
  --fasta examples/ranking_demo.fasta \
  --measurements examples/ranking_demo_measurements.csv \
  --condition '40 C, pH 7.5, crystalline PET powder, 48 h' \
  --units 'umol products / mg enzyme' --batch-size 3 \
  --output user_runs/fasta-demo
```

Both produce identical rankings. Choose a new output directory for each run;
existing output directories are never overwritten. `user_runs/` and `uploads/`
are ignored by Git. Files written elsewhere remain the user's responsibility.

The ranking includes the proposed batch, a log1p activity score, bootstrap spread,
nearest measured neighbor and identity, and disagreement with nearest-sequence
ranking. Positive disagreement means AI ranks the candidate higher. The run
manifest records the input hash, model revision, settings, assay description and
output checksum. This is a **relative prioritization score**, not a calibrated
activity or success probability. Bootstrap spread is not a confidence interval.
The tool has not been validated for single-mutation design or prospectively in a lab.

## Privacy and compute

The application processes uploaded sequences in memory on the machine running
Streamlit. It does not send them to a model API, log them, or cache their embeddings
to disk. Session state holds the active upload/results; changing input clears
the displayed ranking. Downloads/CLI output contain IDs and nearest measured
activities, but not raw sequences. Treat exports as private when appropriate.
Do not expose this local research app publicly with confidential uploads without
adding authentication and deployment-specific data controls.

Public examples reuse verified embeddings. Novel sequences require pinned
ESM-2 35M weights locally. Inference always uses `local_files_only=True` and fails
with an actionable error if missing. Install the public checkpoint separately:

```sh
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esm2_t12_35M_UR50D', revision='6fbf070e65b0b7291e7bbcd451118c216cff79d8', cache_dir='.cache/huggingface', allow_patterns=['config.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','pytorch_model.bin'])"
```

No paid services, GPU rental or model fine-tuning were used in this extension.
New-sequence inference and pairwise alignments take local CPU time, increasing
with the number and length of sequences. Runtime depends on hardware; no fixed
speed guarantee is made.

## Completed scope

- Independent-source feasibility audit, licensing and sequence reconstruction.
- Fixed transfer protocol, construct-overlap exclusion and distant-subset analysis.
- Three comparison methods, exact random reference, reproducible outputs and figure.
- CSV / FASTA input, strict validation, local inference, ranking and export.
- Public demo inputs, CLI, two dashboard pages and reproducibility documentation.
- Automated scientific, parsing, privacy, deterministic-output and UI checks.

No new enzyme discovery, laboratory validation or hosted deployment is claimed.

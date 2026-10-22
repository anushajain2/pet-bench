# PET Scout build tasks

- [x] Inspect destination and applicable repository instructions.
- [x] Acquire public experimental source data and record provenance/checksums.
- [x] Validate sequences, assay conditions, missing values and replicate handling.
- [x] Select one condition without inspecting model performance.
- [x] Implement sequence similarity and connected-component sequence groups.
- [x] Implement random and group-held-out splits with leakage assertions.
- [x] Generate frozen ESM-2 embeddings and cache sequence/model identities.
- [x] Compare nearest-sequence, composition, ESM regression and random selection.
- [x] Run repeated evaluation, candidate-selection metrics and uncertainty analysis.
- [x] Create an interactive explorer of actual out-of-fold predictions.
- [x] Test data validation, splitting, model isolation, metrics and dashboard.
- [x] Write methods, limitations, reproducibility instructions and findings.
- [x] Verify the completed project in the requested folder and launch the demo.

Scope: retrospective enzyme selection from one published screening experiment;
no claims of a new enzyme, wet-lab validation, or universal generalization.
No paid cloud resources required for this build.

## Completed build

- 159 measured proteins, 30 sequence groups and 75 outer evaluation folds.
- Three prediction methods plus exact same-pool random selection expectations.
- Frozen ESM-2 embeddings generated locally on CPU; no paid compute used.
- 15 automated tests passed, covering scientific isolation, artifacts and the app.
- Dashboard rendered successfully in the browser; candidate-selection view verified.
- Main result: no robust ESM advantage over the nearest-sequence baseline.
- Source provenance, exact dependency versions and reproducibility commands included.

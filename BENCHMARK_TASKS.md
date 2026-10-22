# PETBench build checklist

- [x] Inspect the existing study and local hardware.
- [x] Fix assay tasks, model roster, data budgets and evaluation rules before new runs.
- [x] Prepare three assay datasets and one canonical protein catalog.
- [x] Build global sequence groups and save reusable outer splits and nested training subsets.
- [x] Generate and validate four frozen protein-model embedding caches.
- [x] Record model revisions, runtime, peak memory and provenance.
- [x] Evaluate six methods with identical training labels, tuning budgets and test pools.
- [x] Report task-specific selection, ranking and learning-curve results.
- [x] Build a benchmark dashboard while retaining the original study.
- [x] Test split isolation, feature integrity, metrics and UI interactions.
- [x] Write findings, reproduction commands and limitations; verify the completed app.

No paid compute is authorized. All initial runs use local CPU, sequential models,
and cached embeddings. Earlier PET Scout files remain in their original locations.

## Completed run

- 211 unique proteins, three assay tasks, six methods, four training budgets.
- 900 split/subset settings and 5,400 outer method runs.
- 190,440 held-out predictions; full rerun reproduced prediction and summary hashes exactly.
- All four protein models ran on local CPU; $0 paid compute.
- 23 automated tests passed, plus fresh ProtT5 loading/inference and cached CLI exit checks.
- Browser layout and static results figure visually checked.
- Original study retained in the sidebar with original data and reports.

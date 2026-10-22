# Validation of PETBench v1

- 23 automated tests passed after the completed run.
- Verified disjoint train/test IDs and sequence groups for all grouped settings.
- Checked identical candidate pools and observed outcomes across all six methods.
- Independently recalculated top-five selections for the full-data seed-11 runs.
- Checked model revisions, feature checksums and source/result provenance.
- Reran all 5,400 method evaluations: predictions.csv and summary.csv hashes were identical.
- Fresh ProtT5 checkpoint load plus two-sequence inference exited successfully after disabling an unrelated remote conversion side job.
- Cached `pet-bench embed` exited with status 0 for the full roster.
- Dashboard interaction checks cover task, split, budget, model selection and hiding outcomes.
- Dashboard rejects stale configuration/data manifests.
- Original PET Scout tests still pass.
- Dashboard and exported figure were visually inspected.

Paid compute used: $0. All outputs are retrospective results from published experiments.

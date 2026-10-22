# Final-phase verification

Completed 2026-10-22 in the existing local Python environment, reusing frozen ESM-2 35M embeddings. Paid compute: $0.

- 2,700 campaigns: three conditions × (one random-start scenario + two eligible starting groups) × 50 seeds × six policies.
- 18,900 checkpoints and 172,800 assay-selection records, including initial assays.
- Final full regression suite: **42 passed in 11.37 seconds**. The previous 31 tests remain passing.
- All nine finalized result CSV files reproduced with identical SHA256 hashes across two full runs. Timing values are not expected to be identical.
- The four original policies reproduce every original random-start selected sample and observed outcome. All five original campaign output hashes remain unchanged; the earlier benchmark was not rerun or modified.
- Tests verify anchor eligibility, exclusion of all unmeasured starting-group members, no repeated tests, shared initial samples, and first-batch invariance to every hidden outcome for all six strategies in both scenarios.
- Zeroing either acquisition addition recovers the greedy ranking. Paired factorial effects are independently checked, including a non-additive interaction example.
- Every saved checkpoint’s new-hit count and discovered-group count is independently recomputed from logged selections. All logged activities and group labels are reconciled with prepared source tables.
- Undefined uncertainty rankings are explicitly handled: if every uncertainty score ties, quartile is marked 0 and omitted from quartile comparisons. This diagnostic correction left all seven acquisition/performance/design CSV files unchanged. It was documented after the initial run; no acquisition weights or performance thresholds were changed.
- Dashboard tests navigate from the actual app entrypoint, switch condition, starting group, scenario and group-discovery measure, and verify no exceptions. Stale configuration rejection is tested.

The case study reports per-anchor reversals rather than only favorable averages. The two-minute demo deliverable is a written walkthrough for the working dashboard, not a recorded video. All analyses remain exploratory and single-study; there is no external or prospective validation claim.

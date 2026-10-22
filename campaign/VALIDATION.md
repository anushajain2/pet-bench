# Campaign validation

Completed 2026-10-22 using the existing project Python environment and frozen ESM-2 35M cache. No paid compute, external service calls or model downloads were required.

- 600 complete campaigns: three conditions × 50 shared starts × four strategies.
- 4,200 checkpoints and 38,400 selected-assay records (including shared starting samples).
- Two complete simulation runs produced identical SHA256 hashes for all five result CSV files: checkpoints, selections, summary, paired and target_times. Wall-clock timing is not expected to be identical.
- Full regression suite: **31 passed in 4.19 seconds**, including eight campaign tests and the previous 23 tests.
- Numerical test checks bootstrap dual-ridge predictions and spread against independently fitted scikit-learn Ridge models.
- Hidden-label test changes every unrevealed outcome and confirms that the first acquired batch stays unchanged for all four strategies. The acquisition API receives only observed activities.
- Campaign tests independently reconstruct hits, activity maxima and sample counts from every saved checkpoint, check shared initial sets and no duplicate selections, and reconcile measurements against source candidate tables.
- Censoring test confirms non-reaching campaigns remain in success rates and capped-time means.
- Dashboard test navigates through the actual app entrypoint, changes assay condition and batch, and checks that measured outcomes stay hidden until revealed.
- Dashboard and scientific figure visually inspected. Figure axis labels shortened after inspection to prevent overlapping vertical text.

At the fixed 40-assay endpoint, AI greedy found more hits than random in all three conditions and more than nearest sequence in the two neutral-pH conditions. Nearest sequence led the acidic condition. The fixed uncertainty/diversity policy found fewer hits than AI greedy in all three conditions. No acquisition parameters were adjusted after observing these results.

The source dataset was already inspected before protocol creation. This remains an exploratory, single-study, finite-pool replay, not a blind or prospective validation.

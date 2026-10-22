# Two-minute dashboard walkthrough

This is a narration script, not a recorded video.

**0:00–0:20 — The problem.** “A lab knows the activity of 16 enzyme candidates and has budget for 24 more tests. I built a system to compare which candidates it should test next.”

**0:20–0:45 — The original result.** Open Experiment planner at 40°C/pH 7.5 and budget 40. Explain that AI greedy finds 8.34 strong enzymes in total versus 7.20 for sequence similarity and 4.18 for random. These totals include starting hits. Then switch to acidic conditions to show that sequence similarity wins there.

**0:45–1:10 — The harder test.** Open Validation & case study → Group shift. Explain that all 16 initial measurements come from one sequence group and remaining candidates from that group are excluded. Show new hits and distinct groups with new hits. This page counts only newly acquired hits, so its values differ from the original totals. Use the condition-specific results shown; do not claim universal AI superiority.

**1:10–1:35 — Diagnose the failure.** Open Acquisition ablations. Compare uncertainty added, diversity added and both added against prediction-only AI. Point to the uncertainty diagnostic and distinguish an informative score from a calibrated error estimate.

**1:35–2:00 — Reproducibility and limits.** Show the protocol/audit tab and downloads. “Policies never see hidden outcomes; every result can be reconstructed. This uses published experiments and does not claim a new enzyme or prospective lab validation.”

Finish with the strongest result actually supported across the two eligible starting groups, including any negative or mixed outcomes.

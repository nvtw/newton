# Kamino DVI Notes

Last updated: June 22, 2026.

## Current State

- Sparse direct-bilateral DVI runs inside Kamino and avoids dense Delassus materialization for DR Legs.
- On measured DR Legs contact runs, DVI is much faster than `padmm-fast`; recent 60-step CUDA graph runs were roughly 5x-7x faster depending on world count.
- DVI is not proven as good as PADMM in all cases. Joint/contact residuals are close in the tested DR Legs cases, but NCP/natural-map quality still needs broader validation and tuning.
- Preserve Kamino behavior. Only keep mraksha ideas that improve matrix structure, reuse, or hot-path cost without changing contact/friction semantics.
- Keep DVI lean and selectable through normal Kamino config, e.g. `dynamics_solver="dvi"` plus `DVISolverConfig`; avoid separate setup paths or unused solver variants.

## What Mattered

- The mraksha-style speedup came mainly from avoiding the dense contact/limit Schur complement, not from graph coloring.
- Prefactoring the bilateral robot block is already used: DVI factors the joint-joint block once per solve and reuses it for the direct bilateral re-solves.
- The remaining hot path is the repeated sparse unilateral/contact sweep plus bilateral RHS rebuilds.
- Sparse DVI now mirrors dense DVI's active bilateral dimension for repeated direct-block solves: worlds with no active limits/contacts keep the first bilateral result and skip later LLT solves.
- For the focused sparse DR Legs benchmark, `bilateral_solve_period=2` is the best measured default so far. Period `1` spends too much time in repeated LLT solves; period `4` worsened contact/NCP residuals and was not faster in the measured accuracy run.
- Do not copy behavior-changing mraksha choices, such as tangential-only friction projection, unless Kamino intentionally changes its solver semantics.

## Failed Experiment

- Tried an incremental body-space contact update that initialized `J^T lambda` once per block and updated it with per-contact impulse deltas.
- It compiled and produced comparable residuals, but DR Legs timing regressed:
  - 4 worlds, 180 steps: about 10.06 ms baseline mean vs 10.40 ms prototype mean.
  - 16 worlds, 180 steps: about 11.00 ms baseline median vs 12.07 ms prototype median.
- Do not resurrect this exact approach unless atomics and per-row block scans are removed or clearly reduced.
- Also tried reducing joint-row sparse matvec work using the existing joint nonzero prefix, with and without an active-row initializer replacing `v_aug.zero_()`.
  - Both variants were neutral-to-worse on the 180-step DR Legs CUDA graph benchmark.
  - The extra launch/branching was not justified by the reduced block scan.
- Tried fusing sparse bilateral diagonal setup into sparse bilateral block assembly.
  - It passed focused tests but tied the baseline within noise, so it was not kept.
- Tried a DVI-local filtered transpose for bilateral RHS rebuilds.
  - It passed focused tests but regressed DR Legs contact timing, so it was not kept.
- Tried a graph-compatible device-side adaptive skip for repeated bilateral solves.
  - Aggressive thresholds improved one FPS run but hurt NCP/natural-map residuals; conservative thresholds were only a small noisy gain with extra launches/atomics, so this was not kept.

## Next Hot-Path Targets

- Target the contact/unilateral sweep first; small bilateral row-matvec filtering changes have not produced a measured win.
- Keep DVI-specific changes isolated in `solvers/dvi/`; existing Kamino files should stay minimally touched.
- Measure every optimization against the committed sparse baseline before keeping it.

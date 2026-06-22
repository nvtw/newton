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
- Do not treat the bilateral robot block as numerically constant. Its topology is fixed, but its values depend on current joint Jacobians and world-space inertia. A stale-factor test on DR Legs made contacts much worse, including multi-decimeter penetration when reusing one factor for a run.
- Mraksha/Roblox-style "once per mechanism" applies to the symbolic factorization, ordering, fill pattern, and storage plan. The numeric block values are still assembled and factored at runtime.
- Prefactoring the bilateral robot block is already used within a DVI solve: DVI factors the joint-joint block once per substep and reuses it for the direct bilateral re-solves in that substep.
- The remaining hot path is the repeated sparse unilateral/contact sweep plus bilateral RHS rebuilds.
- The PR 2998 native tile factorization update was slower on DR Legs. Only the native left-transpose back-solve fallback was kept; it gives a small LLT solve win.
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

## Next Hot-Path Targets

- Target the contact/unilateral sweep first; small bilateral row-matvec filtering changes have not produced a measured win.
- Keep DVI-specific changes isolated in `solvers/dvi/`; existing Kamino files should stay minimally touched.
- Measure every optimization against the committed sparse baseline before keeping it.

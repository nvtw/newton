# Explicit bilateral row cache proposal

The measured RHS prefix costs about31us out of51us for the complete bilateral
callback on the frozen Colibri workload. Current iteration obtains each wrench
through block row -> global row -> row-local index, plus the structural index.
It repeats wrench reads when accumulating body impulses.

The local cached_bilateral_rows.py prototype adds explicit wrench0, wrench1
and bias buffers indexed directly by (constraint ID, block row). It copies the
original FP32 values inside the existing factor-preparation kernel. Factor
arithmetic remains unchanged. The iterate factory changes only these reads;
FP64 dots, triangular solve, FP32 impulse accumulation and body response retain
their original order. No production fields are reused.

For38blocks x6rows this costs11,856bytes. Copying the values adds writes to
preparation, which must be included in any end-to-end timing. Whether removing
dependent gathers compensates for these writes is not yet measured.

## Lifetime and correctness obligations

- Refresh with every prepare_and_factor, including post-integration geometry
  refresh and changing mass-split copy counts.
- Reallocate/rebind if BlockJointSystem refresh_joint_properties rebuilds row
  ownership. A stale row permutation would apply incorrect impulses.
- Leave accumulated impulses, row_dynamic, dynamic_mass and reference live:
  finite-bound activation may alter them, and the impulse is mutable each sweep.
- The cached wrenches are unscaled physical rows; mass-copy scaling remains in
  the unchanged body response/factor code.
- Ignore cached inactive rows through the original count/valid checks.
- Keep the existing global finite-bound correction path untouched.

## Completed local validation: not promoted

The process-local installer now binds the cache on every BlockJointSystem
rebind and replaces only grouped iterate/relax launches plus block preparation.
Ungrouped and global finite-bound correction paths retain their original code.
There is no canonical runtime modification.

Validation passed:
- Actual revolute/prismatic motion and property-transition fixtures compare
  baseline/candidate q, qd, derived fields, multipliers and bilateral impulses
  byte-exactly. They cover finite-limit crossings, stale releases, friction,
  speed limits, drive changes and joint-row rebinding; P/L guards remain active.
- Captured finite-drive torque bounds and contact momentum/energy tests.
- Mixed joint/friction per-phase P/L and final-energy checks in four
  endpoint-order/group-width configurations.
- Full public330-frame trajectory: all10 shared saved arrays byte-identical to
  /tmp/colibri_fused_begin_baseline_ab330.npz, including q/qd histories.

Candidate timing is12.9499ms/frame (77.22FPS). An immediately following
fresh canonical330 baseline measured12.5075ms (79.95FPS), also with all10
arrays byte-identical, making the candidate3.54% slower in the adjacent pair.
The earlier12.4116ms baseline alone was insufficient to establish regression;
the fresh reverse control confirms no gain. No improvement was demonstrated;
do not promote this cache. Extra preparation writes and dispatch arguments may
outweigh reduced gathers; the measurement alone does not isolate their costs.

Artifacts: /tmp/colibri_cached_rows_candidate330.json and matching NPZ.
Run:

    python -m local_studies.colibri.cached_bilateral_rows \
      local_studies.colibri.check_public_analytic_gradient --frames 330 --substeps 30 \
      --save-history --output /tmp/colibri_cached_rows_candidate330.json

Reverse control: /tmp/colibri_cached_rows_reverse_baseline330.json.
Environment metadata: /tmp/colibri_cached_rows_pair_environment.json.
Before the fresh control, GPU temperature37C, SM clock2625MHz, memory13365MHz,
power79.6W and utilization2%. A transient100%CPU Python process exited before
its command could be identified; this is recorded rather than assumed absent.

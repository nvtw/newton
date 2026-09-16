# Constant-index bilateral iteration study

The dominant ordered sweep includes a serial 6x6 FP64 triangular joint solve.
Unlike block preparation, iteration still has runtime loop bounds. The local
static_bilateral_iterate.py expands only these loops, retaining guards, all
active arithmetic in its original order, dynamic drive rows and impulse scatter.

CPU interpretation of the actual original/transformed callback is byte-exact
for 42 cases: counts 0-6, biased/unbiased phases and diagonal scales from
1e-6 to 1e6. Six existing CUDA block-policy/color-group tests pass, including
bounded motors and momentum checks.

A full public default 330-frame run passes, with every shared saved array
byte-identical to /tmp/colibri_fused_begin_baseline_ab330.npz, including all
q/qd history frames. Isolated timing is 12.38914 ms versus 12.41164 ms:
approximately 0.18%, below a useful confidence margin. The candidate is
NOT promoted; no production source is changed.

The earlier geometric-default Nsight capture reports 146/103 registers per
thread for the two frequent sweep kernels and zero local memory. Thus there
was no measured register spill to fix. Constant matrix indexing alone has
not materially reduced the sweep cost.

Artifacts: /tmp/colibri_static_iterate_candidate330.json/.npz.
The local installer can be invoked before a diagnostic module with:

    python -m local_studies.colibri.static_bilateral_iterate \
      local_studies.colibri.check_public_analytic_gradient \
      --frames 330 --save-history --output /tmp/candidate.json

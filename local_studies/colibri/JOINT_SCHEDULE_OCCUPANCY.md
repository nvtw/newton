# Joint scheduling audit

CPU reconstruction uses the final330 captured rigid graph from
`/tmp/colibri_support_bridge_counts.npz`. First-fit coloring and four-color
partitions reproduce every captured original body-copy count exactly.
This is the final relaxation snapshot, not the separately profiled next
biased sweep; its graph has481rows and175colors.

All38joint rows occur in colors0–12, hence only groups0–3. Their active
joint-lane counts per color are10,7,4,3,3,3,2,1,1,1,1,1,1. The median is
two joint lanes per32-thread block. Those colors contain10–16total rows
including contacts. The source dispatcher executes four colors serially
within each group with a block barrier after each color.

This identifies sparse joint work and a small number of joint-carrying
blocks; it does not by itself prove hardware occupancy is the bottleneck.
A cooperative per-joint RHS/solve microbenchmark is the next targeted
experiment. Changing which colors share copies would change finite-iteration
physics and requires separate validation, even if momentum is preserved.

Report: `/tmp/colibri_joint_schedule_audit.json`. No solver code changed.

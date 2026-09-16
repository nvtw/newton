# Fused joint preparation experiment

The current profile spends0.308ms/frame preparing direct joint rows and
0.242ms/frame snapshotting dynamic drive state. These are consecutive launches
at every substep. The local prototype executes both operations in one kernel,
with one thread per structural joint and an explicit list of that joint's rows.
Every joint prepares its wrenches before reading them for its drive snapshot.
No cross-joint dependency is introduced.

The prototype preserves the original source expressions using generated local
helpers; it is not a proposed production code-generation mechanism. Production
integration, if warranted by timing, would share normal Warp helper functions.
The original path handles multi-axis dynamic rows, disabled systems and changed
row topologies. Geometry-only refreshes outside begin_substep remain unchanged.

Validation:

-10-frame smoke is bit-identical to the canonical trajectory.
-330-frame run passes and all ten saved state/history/model arrays are identical.
-40 live transition comparisons cover revolute/prismatic, split/unsplit,
 finite-limit crossings, friction and velocity-limit changes, finite drives and
 model notifications. All compared fields match bitwise; independent momentum
 checks pass in both paths.

Isolated A/B timing at30 substeps/update: baseline12.411642ms/frame
(80.57FPS), fused12.573828ms/frame (79.53FPS). All ten saved arrays
match byte-for-byte and both runs pass. This provides no performance benefit,
so the prototype is not promoted; canonical code is unchanged. Other agent
GPU jobs were held throughout both runs. Reports:
`/tmp/colibri_fused_begin_baseline_ab330.json` and
`/tmp/colibri_fused_begin_candidate_ab330.json`.
Artifacts: `/tmp/colibri_fused_joint_begin330.json` and `.npz`,
`/tmp/colibri_fused_begin_transitions.log`.

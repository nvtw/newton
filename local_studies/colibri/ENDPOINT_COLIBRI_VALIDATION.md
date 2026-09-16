# Colibri after correcting mesh endpoint suppression

The correction removes only the unsafe static vertex-owner rejection from
the two mesh/SDF sampling kernels. Query bounds, physical gaps, reduction
ranking and solver equations remain unchanged. Six focused cases fail before
and pass after; the independent reversal fixture now passes stopping,
dormant-zero-impulse, energy and six-component momentum checks.

The public example passes4500frames (75seconds), crossing the previous
67.9167-second failure. Group4, grid64, 30substeps per120Hz update, SOR1,
no global damping and exact source offsets remain unchanged.

- Peak fresh depth:0.351057mm
- Peak joint anchor error:0.567253mm
- Peak joint axis error:0.0153767rad
- Full pose history and drive-motion statistics saved.

This run overlapped correctness jobs; its timing is not accepted. A five-minute
extension is running separately. The geometric-envelope admission experiment
remains local; it was not enabled for this result.

Artifact: `/tmp/colibri_public_endpoint_fix4500.json` and `.npz`.

The crank completes38.6985turns in the saved75-second run, with mean speed
-3.24272rad/s against-3.49066target and no stalled samples. Passing the old
failure time does not by itself prove that the endpoint correction directly
restores the particular old missing flank; frozen candidate comparisons are
being used to separate that causal question from trajectory divergence.

The same minimal geometry fix and six-case regression are backported to
Kamino. Its native stage15 test remains unconverged and fails atframe3 with
1.474mm penetration. This is not a Kamino stability fix.

## Five-minute extension

The extension fails at124.1333seconds. The first assertion reports2.741mm
TailMount/Tail_Feather_A separation; an independent all-joint audit of the
saved failed pose finds5.338mm maximum tail anchor error and0.43137rad axis
error. The preceding accepted frame has maximum tail anchor0.26244mm.
This is another abrupt event, not sustained robustness.

Frozen original update8147 shows why the endpoint change alone is not a
complete coverage solution: public admission still excludes the approaching
opposite flank after the fix (minimum transported gap+549.8µm). Geometric
admission within the same envelope retains a point ending348µm penetrating,
which lets the solver respond during that interval. Its separate75-second
scene screen passes, but canonical opt-in plumbing and longer validation
remain pending. Physical activation and authored gaps are unchanged.

Artifacts: `/tmp/colibri_public_endpoint_fix18000.json`, `.npz`,
`/tmp/colibri_endpoint124s_joint_detail.json`, and
`/tmp/colibri_public8147_endpoint_coverage.json`.

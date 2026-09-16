# Redundant broadcasts around block joints

The single-world mass-splitting dispatcher interleaves the direct equality
solve with ordinary contacts. Unbounded BlockJointSystem.solve is a no-op,
but this path still broadcasts copies before preparation, then writes body
state and broadcasts again after warm-start averaging.

Local experiment check_block_broadcasts.py retains block preparation and
uses the ordinary no-direct-solve scheduling afterward. It asserts local
block joints, no finite drive bounds, no direct contact response, no maximal
projector, and no reduced constraints. All seven saved arrays after330frames
are byte-identical to the baseline. No canonical dispatch changes yet.
Its observed75.25FPS overlapped the longer validation; do not treat that
as an isolated performance result.
Artifact: `/tmp/colibri_block_broadcasts_330.json` and `.state.npz`.

## Canonical integration

BlockJointSystem reports whether finite drive limits require an original-body correction. The single-world mass-splitting solve retains prepare_and_factor and bypasses only the no-op global solve scheduling when ordinary PGS owns the rows and no articulated contact path is active. Relaxation and other dispatch modes are unchanged.

A counting regression failed before the fix (three broadcasts instead of one), then passed after it. Direct/finite corrections and all articulated contact guards retain their original synchronization. This change does not alter contact geometry, row order, masses, or drive bounds. The earlier overlapping timing is not an isolated performance claim.

Canonical validation: three scheduling tests and six captured drive/contact-policy tests pass. The canonical 330-frame Colibri run passes all existing checks and all seven saved state arrays are byte-identical to `/tmp/colibri_grid64_width4_30x1_330.state.npz`. Report: `/tmp/colibri_block_broadcasts_canonical_330.json`. Timing is excluded because concurrent correctness work was allowed.

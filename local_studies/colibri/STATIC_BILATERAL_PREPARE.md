# Static bilateral preparation experiment

The current profile attributes about2.58ms/frame to bilateral block
preparation. The production kernel uses runtime row counts to index local
6x6 FP64 matrices during factorization. This local experiment expands
those small loops into constant indices while retaining row-count guards,
active arithmetic order, precision, dynamic row compliance and validity
checks. It does not cache geometry across time or weaken joint solves.

`static_bilateral_prepare.py` compares every output (body responses, lower
factor, diagonal, validity) against the production kernel, for row counts
0 through6 with authored revolute-drive and randomized wrench geometry.
All14 cases are bitexact: `/tmp/colibri_static_bilateral_prepare.json`.
This is initial equivalence evidence, not comprehensive high-condition
validation and not a demonstrated speed improvement.

A process-local trial can replace
`articulations.block_joint_system.prepare_bilateral_joint_blocks` with
`static_bilateral_prepare.candidate()` before solver construction.
No production implementation is changed. The current candidate creates
expanded source as a study artifact; any production implementation needs
a maintainable fixed-bound kernel and appropriate regression coverage.

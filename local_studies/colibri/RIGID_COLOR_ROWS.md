# Rigid-row coloring specialization

The generic deterministic greedy color builder scans eight endpoint slots for
every row, both when finding an available color and when marking it occupied.
The existing Colibri color-group policy only accepts rigid contact/joint rows,
which have at most two endpoints. The local `rigid_color_rows.py` changes only
those loop bounds from eight to two. No coloring order, mask arithmetic,
first-free-bit choice or row schedule changes.

This specialization is valid only for confirmed two-endpoint rows. It is not
a replacement for the generic multibody-row builder. A production integration
would keep the generic default and select the specialization only under the
existing rigid-only group policy.

Validation so far:

- CPU/CUDA compare every allocated graph buffer against the generic builder.
-48cases cover four group widths, empty/shrunk/regrown graphs, repeated bodies,
 absent endpoints and more than64colors.
- Public330-frame trajectory passes, with all saved pose/velocity history values
 byte-identical to the validated geometric-default baseline.

Integrated after isolated timing confirmed the improvement in both orders:
12.99005→12.71753ms/frame and13.07834→12.35561ms/frame. All ten saved arrays
match exactly in each pair. The spread indicates timing noise; do not claim a
fixed percentage across scenes. Generic eight-endpoint behavior remains the
default; only the existing rigid-only solver policy selects two endpoints.

The production regression fails before integration with the missing selection
argument and passes after. A separate CPU/CUDA multibody test verifies that
nodes in slots2–7 still affect generic coloring. The four pre-existing local-import PLC0415 issues in the solver file were
subsequently cleaned up; full-file Ruff and the PhoenXWorld import pass.
No commit has been made.

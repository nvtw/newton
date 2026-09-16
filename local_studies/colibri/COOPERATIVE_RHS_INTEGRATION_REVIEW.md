# Cooperative RHS integration review

Scope: uncommitted local cooperative_bilateral_rhs.py and canonical grouped
dispatch/body-load/bilateral call paths at base
f0073be5332b1caf155d54a448943f924d856ef7. This is not a review of the
unrelated pending main merge or the entire PhoenX branch.

Fit: Fits as a private hardware scheduling optimization of existing grouped
rigid joint blocks. No new public configuration or authored USD field is
needed. The selected scope changes neither equations nor color/copy order.
The failed interleaved-group experiment is a materially different alternative
and is excluded because it failed joint accuracy.

Requirements: the local CUDA prototype preserves per-row FP64 dot order,
uses all eight lanes before leader-only scalar solve/scatter, and invokes
original joint inequalities only once. The exact eight-lane mask agrees
with256-thread blocks. Count/enabled/valid predicates are shared across each
tile. Body-pair loading is read-only. Frozen355-array and two independent
330-frame ten-array comparisons pass exactly, as do lifecycle, finite-drive
and mixed contact conservation checks. Paired timing11.153/11.137ms versus
12.342ms supports the optimization; it does not establish other-scene speed.

Integration requirements: CPU must remain scalar (the native shuffle's CPU
stub cannot implement the tile operation); contact-only paths should remain
scalar to avoid eightfold idle threads. Preparation and bounded-drive global
correction must retain their original paths. Zero/disabled/partial tiles,
row counts0–6, lifecycle changes, graph capture and generic fallbacks need
production tests. Replace local AST/launch monkeypatching with shared normal
Warp helpers. Add a focused changelog fragment.

No demonstrated CUDA arithmetic/ownership defect was found in the local
prototype within the tested scope. Production integration and long-trajectory/cross-scene validation are now
complete within the scope below. No commit or approval request.


## Canonical integration validation

The canonical runner completed successfully after integration, with unchanged
SHA-256 hashes for bilateral_joint.py, dispatch/color_groups.py and
solver_phoenx.py throughout validation. It compared array shapes, dtypes and
bytes, not floating-point tolerances:

- Colibri, 330 frames: all ten saved arrays match the scalar baseline.
- Kapla, 180 measured frames: all six saved arrays match the packed-layout baseline.
- G1, 20 controller steps: both complete position/velocity histories match.
- Colibri, 18,000 frames (300 simulated seconds): all ten saved arrays match
  the previous geometric-admission baseline.

The long Colibri run measured 11.18856 ms/frame (89.377 FPS), excluding
rendering and physical audits, after 30 warmup frames. Peak penetration was
0.753578 mm, joint-anchor error 0.938996 mm and axis error 0.028822 rad.
These results preserve the existing finite-duration acceptance, including
its unresolved free-assembly drift; they do not prove indefinite stability
or resolve high-mass-ratio convergence. Cross-scene equality does not establish
Kapla/G1 speed improvements or determinism across different hardware.

Report: /tmp/colibri_cooperative_canonical_validation.json.
The report records source hashes, exact commands, references and comparisons.

## Integration hygiene audit

Read-only Ruff check of all 88 changed/untracked production Python files
reports nine PLC0415 local-import findings in solver.py and
graph_coloring_incremental.py. Each flagged import is already present in HEAD;
no new lint findings were detected in this scope. This is not a full
pre-commit pass, and the pending main merge is not committed. Raw results:
/tmp/colibri_integration_ruff.json.

The two public Colibri examples and asset README contain no hard-coded /home
or /tmp path, private newton._src import, or TODO/FIXME marker. Asset provenance,
SI conversion, cylinder reconstruction, runtime cache generation, source
gravity and current solver limitations are documented in the asset README.

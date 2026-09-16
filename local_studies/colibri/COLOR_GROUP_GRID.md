# Color-group grid integration

The canonical color-group sweep now uses64blocks of32threads. One shared `DEFAULT_SWEEP_BLOCK_COUNT` controls both the factory stride and solver launch dimensions. Independent groups can use more SMs; each group retains its original sequential color order and copy mapping. Other dispatch modes are unchanged.

Validation: five existing topology/policy tests pass, including captured finite-drive bounds, contact momentum/energy and invalid-option checks. New `test_color_group_grid.py` compares32and64block kernels over0,1,33,65and131ragged groups. It checks exact-once row coverage, group ownership, per-group color dependency order, and byte-identical results. All six tests pass.

Parent full-scene evidence: paired isolated330-frame group4/30step runs measured67.6841FPS at32blocks and72.2468FPS at64blocks; all seven saved state arrays are byte-identical (byte comparison handles NaN-bearing headers). Parent then confirmed the full1200-frame/20second motion history, final poses/velocities and timestamps are byte-identical between grids and pass existing checks. These are this Colibri workload measurements, not a general solver speed guarantee. No commits.

## Exact first-fit bit search

The canonical builder now selects the first unoccupied bit with five integer binary-search tests, preserving the previous greedy row order. A preserved linear-search kernel remains in `color_groups_linear_reference.py`. CPU and CUDA frozen topology checks compare every buffer across widths 1, 4, 8, and 33 and shrink/empty/regrow sequences. Independent Python set-based coloring checks twelve randomized 300-row graphs, eight endpoints, empty rows, and a repeated pair spanning three mask words.

A captured 345-row frozen topology replay measured 0.36235 ms per build before and 0.32559 ms after, including buffer zeroing. At two builds per frame this predicts only about 0.074 ms/frame savings; it is not a measured full-scene FPS improvement.

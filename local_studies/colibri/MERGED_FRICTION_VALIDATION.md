# Staged friction proposal validation

Production source is unchanged. The staged tree is `/tmp/colibri_merged_friction_proposal`; its manifest records both original and staged SHA256 values. `run_merged_friction.py` installs module mappings before Newton imports and checks source, asset, staged-file, and actual loaded-module paths after execution.

## Scope and limits

The proposal records actual point-friction cone clipping, carries that state through matching and packed scatter, and consumes it at contact refresh rather than each temporal prepare. It uses the actually applied accumulated normal impulse for the single-channel stabilized Coulomb bound. Ordinary and reduced point callbacks share the projector; direct and maximal point callbacks have explicit state writes. The reduced patch load path and G1 diagnostic load were also updated.

Patch friction uses the total-normal bound, but its separate broken-anchor lifecycle is not migrated. Its legacy reset policy remains limited to actual patch operation, not point fallback. Shared cloth storage grows by one unused persistent row; cloth does not consume the rigid anchor flag. Persistent current/previous storage grows by eight bytes per capacity point, plus four bytes where a packed current buffer is allocated.

This staged proposal does not include the separate positive-gap identity/fresh-geometry matching experiment. No claim of complete base stationarity or correct coupled drive equilibrium follows from the history fix.

## Results

- Staged CPU constitutive/lifecycle gate: 3 tests pass, `/tmp/colibri_merged_cpu_gate.log`.
- Initial native batch: first five modules, 14 tests pass. Contact coupling then exited139 because three synthetic projector fixtures allocated impulses without persistent history. No physical result was accepted from that process.
- Fixtures now allocate `CC_DWORDS_PER_CONTACT` history rows. All three actual CPU projector energy/rank-one controls pass with exit0; `/tmp/colibri_merged_synthetic_cpu.log`. Other bare container tests are dispatcher mocks or already allocate history.
- Remaining GPU contact conservation, owned articulation paths, patch, Kapla and learned G1 checks are pending. Batch runner rejects zero-test/all-skipped runs and preserves every return code. It supports stopping only between modules after a time budget.

## Resume command

```bash
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
  -m local_studies.colibri.run_merged_regressions \
  --output /tmp/colibri_merged_native_remaining.json --max-seconds 180 \
  @/tmp/colibri_merged_regression_remaining.txt
```

Run only in the coordinated GPU window. An incomplete time-bounded batch is explicitly not an overall pass.

## Combined-history trajectory

The separate combined history control has 0.147181 mm net base translation over20–60 seconds (3.67952 micrometres/second), and 0.008697 degrees rotation. Over50–60 seconds its net translation is0.039856 mm (3.98556 micrometres/second). Final hinge angle is25.081925 degrees. It still creeps; a PhysX angle comparison is not an analytical equilibrium criterion. See `/tmp/colibri_friction_break_paired_motion.json` for all controlled windows.

## Second GPU window

Contact coupling passed all7 tests (483.942 seconds including cold compilation), including cross-articulation energy and sustained internal-contact/motor momentum. Source and staged SHA guards passed. The180-second budget stopped the runner at this module boundary; its exit1 records an incomplete suite, not failed tests. Cumulative6modules/21tests passed,12modules pending in `/tmp/colibri_merged_regression_remaining2.txt`. Report: `/tmp/colibri_merged_native_remaining.json`.

## Prioritized remaining breadth

`/tmp/colibri_merged_priority_modules.txt` starts with unequal-copy/moving-COM conservation, packed relaxation, ordinary energy and predictive impacts (seven tests). Next come maximal/direct momentum (six tests), direct nullspace/kinematic ownership (six) and reduced nullspace (four). Normal-first/frictionless equivalence, patch friction and Kapla follow. `/tmp/colibri_merged_owned_methods.txt` selects four direct/tree mass-splitting and multi-world coupling methods, excluding the separately failing cloth baseline case from this rigid change gate. The learned G1 check is `run_merged_friction local_studies.colibri.check_g1_policy --frames20 --output PATH` (insert a space before20).

Ordinary and reduced kernels have partial cache coverage. Direct/maximal mode specializations can still cold-compile for minutes; a180-second boundary limit cannot bound one active module. Do not interpret an incomplete run as a complete integration gate.

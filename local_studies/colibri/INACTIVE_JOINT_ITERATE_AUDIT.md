# Inactive axial iteration prototype

Canonical runtime is unchanged. `inactive_joint_iterate.install()` replaces only the inequality callback; the dispatcher still solves the bilateral block first.

The shortcut applies to rigid block-PGS revolute/prismatic rows with clamp NONE and friction <= 0. The original axial helper then returns exactly zero impulse and clears accumulated friction. The shortcut retains that clearing.

An initial unconditional return preserved all nine public 330-frame arrays and 40 live transition comparisons but changed angular signed zeros in a direct frozen diagnostic. The corrected prototype loads only endpoint linear/angular velocities and falls back to original arithmetic whenever any component is exactly zero. It avoids inertia loads, wrench arithmetic and stores only for fully nonzero endpoint velocities. This is an exact-zero guard, not a physical threshold.

Validation:

- Forty revolute/prismatic, split/unsplit transition steps: byte-identical q/qd, used geometry, limit/friction multipliers and bilateral impulses; independent all-six momentum checks.
- Existing three-body/two-manifold grouped moving-COM biased/relax momentum and energy fixture passes.
- Direct stale-friction tests with all negative-zero and nonzero velocities: linear/angular/multiplier bit patterns identical.
- Corrected full public330: `/tmp/colibri_inactive_iterate_zero_guard330.{json,npz}`; all nine numeric arrays byte-identical to `/tmp/colibri_inactive_joint_canonical330.npz`.
- Timing pending isolated GPU window. Correctness timings overlap the long stability run and are not performance evidence.

## Isolated result: do not promote

After the long stability run completed and nvidia-smi reported no compute processes, the reference measured12.68580ms/frame (78.828FPS), the strict signed-zero candidate12.66399ms (78.964FPS): only0.17%, within timing noise. P95slightly worsened13.47785->13.55122ms. Both330checks passed and all nine saved numeric arrays were byte-identical. Artifacts `/tmp/colibri_inactive_iterate_{reference,candidate}_isolated330.{json,npz}`. Keep this optimization local; no demonstrated useful gain.

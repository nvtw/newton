# Interleaved color-group experiment (rejected)

The prototype keeps first-fit colors but assigns partition=color%group_count
instead of color//width. Each partition visits its colors in ascending order.
Both topology ownership and dispatch use the same mapping; width4,30substeps
per120Hz update, SOR1 and all physical inputs remain unchanged. This changes
copy counts and finite-iteration physics, so exact baseline trajectory
equivalence is neither expected nor claimed.

Three block-policy tests pass. Four mixed joint/contact conservation cases
pass unchanged. The original contact-only fixture fails its unequal-copy
precondition for width4: all three bodies now have exactly2copies. A separate
local fixture asserts this actual topology and retains every momentum,
energy, motion and activity assertion; all four cases pass. Original
production tests were not changed.

Full Colibri fails at0.366667simulated seconds: TailMount/Tail_Feather_A__2x_
axis error0.108276rad exceeds0.03rad. The screen stops before30warmupframes,
so there is no usable FPS measurement. No promotion; production stays with
consecutive color groups. This demonstrates that conserving momentum alone
does not establish adequate finite-iteration joint accuracy.

Artifacts:
-`/tmp/colibri_interleaved_groups_tests.log` (includes topology precondition failures)
-`/tmp/colibri_interleaved_groups_physics.log` (adapted fixture and mixed tests pass)
-`/tmp/colibri_interleaved_groups330.json` and `.npz` (failed screen)

Reproduce the scene screen:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.interleaved_color_groups local_studies.colibri.check_public_analytic_gradient --frames 330 --substeps 30 --save-history --output /tmp/colibri_interleaved_repeat.json
```

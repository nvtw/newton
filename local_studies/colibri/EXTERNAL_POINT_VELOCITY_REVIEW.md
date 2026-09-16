# External point-velocity numerical review

Scope: local `external_point_velocity.py`, complete root/child revolute component against an external endpoint. No core edits or GPU runs in this review. Fits as a private, guarded optimization; it is not an unconditional general numerical replacement.

## Algebra

The native scalar external branch is exactly equivalent in real arithmetic to:

- root point: `v_root + omega_root × r`;
- child point: `v_root + omega_root × (r - shift) + s*(linear + axis × r)`;
- `s = axis · (omega_child - omega_root)`;
- external point: `v_external + omega_external × r`;
- relative point velocity: endpoint1 minus endpoint0.

The candidate has the correct shift sign, endpoint ordering and child motion contribution. It preserves the original use of joint-coordinate velocity rather than independently reading child linear velocity. This equivalence is to the existing revolute path, not a new assertion of arbitrary joint-type coverage. Internal-component contacts retain the original cancellation-aware fallback.

## Actual CPU helper evidence

`test_external_point_velocity_review.py` invokes the candidate Warp function and the unchanged scalar joint-coordinate function, with FP64 used only in the existing reference. All tests use resolved FP32 inputs.

1. 128 randomized root/child contacts, reversed endpoints and moving external bodies: pass absolute row-vector difference `1e-8 m/s` at modest velocity scale.
2. Common16m/s translation plus a resolved1e-7m/s rotational slip: original candidate loses the slip. Kepler's translation-difference-first revision passes unchanged test. Fail-before `/tmp/colibri_external_point_velocity_review.log`; pass-after `/tmp/colibri_external_point_velocity_review_after.log`.
3. Equal10rad/s angular velocities on root and moving external body, lever lengths1m and nextfloat(1m): revised candidate−9.536743e-7 versus scalar reference−1.192093e-6m/s. Absolute difference2.384186e-7, relative20%. A genuinely static external endpoint guard excludes this case; inverse mass zero alone does not exclude moving kinematics.
4. **Stationary external endpoint, near rolling center:** root translation `nextfloat(10)` m/s, angular velocity10rad/s, lever `nextfloat(1)` m. Candidate0 versus reference+2.384186e-7m/s. Static-only scope alone does not exclude this cancellation. `/tmp/colibri_external_static_rolling_review.log` preserves both remaining failures.

Cases3/4 quantify numerical limitations, not a demonstrated momentum failure or evidence that the current low-speed Colibri fixture is inaccurate. Kepler's actual134-point trace test reports errors approximately1e-11m/s. The local counterexample fixture deliberately remains failing for the unguarded raw helper; it must not be presented as an accepted regression suite until the appropriate guarded call path is tested.

## Recommendation

Keep the successful relative-translation-first algebra. For a general accepted path, bound FP32 evaluation error or detect cancellation and retain the original scalar fallback when the physical row-accuracy budget is not met. Do not loosen that budget to accept the new arithmetic. Do not infer a universal guarantee from external-static ownership or one low-speed trajectory. A guarded fast path is proportionate; arbitrary moving endpoints require additional relative-angular algebra or fallback evidence.

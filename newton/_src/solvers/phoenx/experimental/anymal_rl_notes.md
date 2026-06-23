# PhoenX Anymal RL Notes

## Steering curriculum findings

- The forward and disturbed-forward phases can produce a stable forward policy.
- Adding a turn/stand phase directly after forward initially made the policy stand for yaw commands. The clean fix was not more random tuning: use a minimum non-zero yaw command magnitude and Anymal left/right mirror loss.
- With `command_yaw_min_abs=0.45` and `mirror_loss_coeff=0.02`, the turn/stand phase passed no-reset evaluation for idle, positive yaw, and negative yaw from the disturbed-forward checkpoint.
- A broad omni command phase can collapse back to standing if the dense velocity reward is too weak relative to orientation/yaw terms. The recipe therefore gives omni phases a stronger linear velocity reward and a small command-progress term.

## Latest measured checkpoints

- `/tmp/phoenx_anymal_turn_yaw_mirror3/checkpoint_04_turn_stand_978.npz`: passes the turn/stand battery over 300 no-reset steps with zero falls.
- `/tmp/phoenx_anymal_omni_full/checkpoint_05_omni_steering_1298.npz`: failed the full omni battery; it preserved idle standing but did not recover forward/lateral tracking. Do not use this as a final steering policy.

## Playback

Use the viewer example with a checkpoint trained for command steering:

```bash
uv run --extra examples -m newton.examples robot_anymal_rl_phoenx --device cuda:0 --checkpoint /path/to/policy.npz
```

Controls are body-frame velocity commands: `W/S` forward/backward, `A/D` lateral, `Q/E` yaw about the up axis. With no key pressed, the command is zero.

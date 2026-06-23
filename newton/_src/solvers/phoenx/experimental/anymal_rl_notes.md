# PhoenX Anymal RL Notes

## Steering curriculum findings

- The forward and disturbed-forward phases can produce a stable forward policy.
- Adding a turn/stand phase directly after forward initially made the policy stand for yaw commands. The clean fix was not more random tuning: use a minimum non-zero yaw command magnitude and Anymal left/right mirror loss.
- With `command_yaw_min_abs=0.45` and `mirror_loss_coeff=0.02`, the turn/stand phase passed no-reset evaluation for idle, positive yaw, and negative yaw from the disturbed-forward checkpoint.
- A broad omni command phase can collapse back to standing if the dense velocity reward is too weak relative to orientation/yaw terms. The recipe therefore gives omni phases a stronger linear velocity reward and a small command-progress term.

## Latest measured checkpoints

- `/tmp/phoenx_anymal_turn_yaw_mirror3/checkpoint_04_turn_stand_978.npz`: passes the turn/stand battery over 300 no-reset steps with zero falls.
- `/tmp/phoenx_anymal_omni_full/checkpoint_05_omni_steering_1298.npz`: failed the full omni battery; it preserved idle standing but did not recover forward/lateral tracking. Do not use this as a final steering policy.


## Full-control educational runner

- `train_anymal_full_control_curriculum.py` is the readable from-scratch runner for a steerable Anymal policy. It keeps the phase list explicit: balance/forward, faster forward, robust forward, turn-in-place, recover forward, curved walking, reverse, side-step, full command mix, then robust full control.
- The split is intentional. The earlier broad `omni_steering` phase exposed forward, lateral, reverse, and yaw commands at the same time and could collapse a useful forward/yaw policy back toward standing. The new runner expands one command family at a time and evaluates every phase without auto-reset.
- The final policy is intended for `robot_anymal_rl_phoenx`, whose viewer controls are body-frame velocity commands: `W/S` forward/backward, `A/D` lateral, and `Q/E` yaw. No key pressed sends a zero command.

## Playback

Use the viewer example with a checkpoint trained for command steering:

```bash
uv run --extra examples -m newton.examples robot_anymal_rl_phoenx --device cuda:0 --checkpoint /path/to/policy.npz
```

Controls are body-frame velocity commands: `W/S` forward/backward, `A/D` lateral, `Q/E` yaw about the up axis. With no key pressed, the command is zero.

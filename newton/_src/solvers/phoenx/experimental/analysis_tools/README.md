# PhoenX analysis tools

Diagnostic helpers for PhoenX solver performance work. Not part of the solver;
used to decide *which* optimization is worth doing.

## `ncu_profile_iterate.sh`

Nsight Compute profile of the multi-world hot iterate (`prepare_plus_iterate`).
Determines whether the kernel is **bandwidth-, compute-, or latency-bound** —
the question that should precede any micro-optimization.

```bash
# Needs sudo (GPU performance counters are privileged).
sudo bash ncu_profile_iterate.sh [SCENE] [NUM_WORLDS] [OUT_FILE]
# e.g.  sudo bash ncu_profile_iterate.sh dr_legs 4096
```

Reading the result:
- **Compute (SM) Throughput** high (≳70%) → compute-bound: cut instructions.
- **DRAM Throughput** high → bandwidth-bound: cut bytes moved (e.g. the packed
  symmetric `inverse_inertia_world`).
- Both low + **low occupancy** + high **Stall Long Scoreboard** → latency-bound,
  occupancy-limited: the load latency isn't hidden because too few warps are
  resident. Fix by *raising occupancy* (shed registers, or give each world its
  own CTA via the `block_world` scheduler) — **not** by software prefetching,
  which adds registers and lowers occupancy further.

### Known result (RTX PRO 6000, dr_legs @ 4096, 2026-06-25)
SM 36% / DRAM 14% / occupancy 25% (register-limited) / 42% Long-Scoreboard
stalls / ~12 of 32 active threads per warp → **latency-bound + lane-underfilled**,
not bandwidth- or compute-bound. This is why:
- packing `inverse_inertia_world` to vec6 won (+22%): fewer registers/outstanding
  loads → more latency hidden;
- recompute-from-axis tricks (tangent basis, mode-cache subset reads) were
  neutral-to-negative: they trade loads for ALU/registers, the wrong direction;
- selecting `block_world` for dr_legs won (+17%): one CTA per world fixes the
  lane underfill (see `_choose_multi_world_scheduler`).

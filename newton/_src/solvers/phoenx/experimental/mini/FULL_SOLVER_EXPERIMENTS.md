# Full PhoenX Throughput Experiments

Compact ledger of mini-to-full solver transfers. Use `docs/PERF_NOTES.md` for
current architecture conclusions and benchmark artifacts for raw traces.

## Reference workload

The original transfer baseline used 32,768 worlds, eight free boxes per world,
one substep, four PGS iterations, point friction, sticky matching, and captured
replay on an RTX PRO 6000 Blackwell. Baseline: 4.025 ms/frame, 8.140 M
world-steps/s, 65.122 M body-steps/s. Contact-count differences make unmatched
settled-manifold timings diagnostic only.

## Decisions

| ID | Decision | Result |
| --- | --- | --- |
| F1 | Keep prepared contact lever arms | Fused prepare/solve 669.58 -> 573.37 us (-14.4%); end-to-end gain unresolved. |
| F2 | Keep deterministic serial greedy coloring for many small worlds | Coloring 528.41 -> 83.20 us; controlled frame 3.880 -> 3.211 ms. |
| F3 | Keep direct endpoint color ownership | Removes multi-world adjacency rebuild; retain fallback above 64 colors. |
| F4 | Keep scheduler-shaped color output | Avoid family prefixes when block-world consumes only color prefixes. |
| F6 | Keep stable monotone-run world bucketing where its predicate holds | Avoid general sorting machinery. |
| F7 | Keep sparse optional revolute rows | Do not materialize absent rows. |
| F10 | Keep family-aliased mutable joint state | Reduces redundant generic state without changing the public model. |
| F11 | Keep vec4-packed generic multiplier sidecar only where measured | Layout-specific, not a blanket packing rule. |
| F12 | Keep subwarp auto-scheduling for small rigid worlds | Select by workload. |
| F13 | Keep coalesced node-color mask reset | Direct ownership still needs an efficient reset. |
| F14 | Keep compact-articulation fleet scheduling | Useful for regular fleets; preserve generic fallback. |
| F17 | Match the canonical current contact stream | Avoid maintaining a parallel solver-only truth. |
| F18--F21 | Keep narrowed/deduplicated contact history and fused forward map where qualified | Preserve exact sticky matching semantics. |
| F22 | Keep fused contact-coloring priorities | Removes a redundant pass. |
| S2 | Remove unreachable contact-owner loads | Safe single-world cleanup. |

## Rejected transfers

| IDs | Reason |
| --- | --- |
| F5 | Phase splitting added launches/traffic. |
| F8 | Final state-export fusion increased live state. |
| F9 | Cooperative merge/basis reconstruction did not amortize synchronization. |
| F15 | Explicit color-major contact copies added more traffic than locality saved. |
| F16 | Mini world striping did not fit full-solver heterogeneity. |
| S1 | Color-ordered contact locality alone did not justify reordering cost. |

## Transfer rule

Accept a mini result in full PhoenX only after matched end-to-end timing,
contact/work accounting, graph-capture validation, and representative tests for
contacts, friction, stacks, mixed joints, chains, and fallback paths. Kernel-only
wins must be labeled as such.

## Complete transfer ledger

| ID | Outcome and evidence retained |
| --- | --- |
| F0 | Baseline: 4.025 ms, 8.140M world-steps/s, 65.122M body-steps/s on the 32K eight-box workload. |
| F1 | Accepted hot-kernel change: prepared lever arms cut fused prepare/solve 14.4%; contact variation prevented a clean frame claim. |
| F2 | Accepted: serial per-world greedy cut coloring 84.3% and controlled frame time 17.2%; exact-priority local search regressed. |
| F3 | Accepted: direct endpoint masks remove multi-world adjacency rebuild; JP rebuild/fallback remains above 64 colors. |
| F4 | Accepted: compile scheduler-shaped output so block-world avoids unused family prefixes. |
| F5 | Rejected: phase splitting added launch and round-trip costs. |
| F6 | Accepted only for stable monotone runs: cheap world bucketing without a general sort. |
| F7 | Accepted: omit optional revolute rows instead of materializing zeros. |
| F8 | Rejected: export fusion increased live state and did not improve the frame. |
| F9 | Rejected: cooperative merge/basis reconstruction did not repay synchronization. |
| F10 | Accepted: alias mutable state by constraint family while preserving canonical ownership. |
| F11 | Accepted where measured: vec4 multiplier sidecar is layout-specific. |
| S0 | Large Kapla baseline establishes the single-world regime; do not extrapolate many-world scheduling results. |
| S1 | Rejected/control: color-order locality alone did not repay reordering. |
| S2 | Accepted: remove loads proven unreachable for contact ownership. |
| F12 | Accepted: auto-select subwarp scheduling for small rigid worlds. |
| F13 | Accepted: coalesced mask reset supports direct endpoint ownership. |
| F14 | Accepted: compact-articulation scheduling for regular fleets with generic fallback. |
| F15 | Rejected: explicit color-major copies added more traffic than locality saved. |
| F16 | Rejected for full: mini world striping did not survive heterogeneous solver state. |
| F17 | Accepted principle: consume the canonical current stream rather than duplicate it. |
| F18 | Accepted after qualification: narrow previous-contact fields that tolerate it. |
| F19 | Accepted where matched: contact-major warm-start gather reduces scattered history reads. |
| F20 | Accepted: fuse the forward contact map without duplicating canonical state. |
| F21 | Accepted: remove duplicate previous-contact geometry. |
| F22 | Accepted: fuse contact-color priorities into an existing pass. |

F2/F3 change deterministic PGS order and therefore trajectories. Their
qualification covered multi-world ordering, contacts, stacking/friction, mixed
joint modes, chains, prismatic, ball-socket, and fixed joints; bitwise equality
was not the acceptance criterion. Modeled bandwidth figures reused mini useful
byte counts and deliberately excluded full matching/ingest/sort traffic.

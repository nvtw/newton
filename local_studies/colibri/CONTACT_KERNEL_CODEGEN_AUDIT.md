# Two-body contact kernel code-generation audit

## Executed candidate and measured cost

Candidate generated source:
`/tmp/colibri_cached_component_0doupexh/cached_contact_gs.py`.
Executed Warp module hash72fc378; local helper `cached_component_response.py`.
The accepted candidate applies `serial_native_pair`, not the rejected independent
dense body-response contraction. It still builds12 basis responses per refresh
for external mobility queries.

Parent's actual warmed profile:
`/tmp/colibri_cached_serial_warm_profile_stats.csv`.

- Iterate:26.214ms/frame,422.8microseconds/call,55.9%.
- Mobility refresh:8.326ms/frame,17.8%.
- Cache construction:1.294ms/frame.
- Warm start:1.420ms/frame, versus native3.155.
- Main iterate was native27.468ms/frame: removing tree barriers barely reduced it.

## Resource evidence: no spill finding

Exact cached PTX, with only its trailing NUL string terminator removed, was
assembled offline by CUDA13.1 ptxas for sm120. No GPU launch or source edit.
This is compiler resource evidence, not claimed driver-JIT binary identity.

| Kernel | Registers | Stack | Spill loads/stores | Barriers |
| --- | ---: | ---: | ---: | ---: |
| Candidate iterate |128|0|0/0|0|
| Native iterate10b3364 |128|0|0/0|1|
| Candidate refresh |64|0|0/0|0|
| Native refresh |96|0|0/0|0|

Logs: `/tmp/colibri_cached_72fc378_ptxas.log`,
`/tmp/colibri_native_10b3364_ptxas.log`.
Raw PTX has no local load/store instructions. The1214 virtual FP64 registers
are compiler SSA names, NOT1214 allocated hardware registers or proof of spills.

## Work that remains

The candidate launches one active thread per component. That thread visits every
point in order, evaluates three generic FP64 joint-coordinate row velocities,
reads/writes global body and impulse arrays, executes native metric friction,
and performs serial exact two-body elimination. The source has no mechanism
to amortize all these body/factor loads in register-held component state.
Dependencies between successive point corrections remain fully serialized.

Its mobility refresh is likewise serial: six cross-response queries perpoint,
each with endpoint/coordinate loops over cached spatial coefficients. This is
additional arithmetic/memory work, not a free lookup. The profile confirms that
refresh plus cache construction does not improve this part of the budget.

The metric helper includes a safeguarded40-iteration sliding root and a
conditional FP64 fallback. Neither worst case may be assumed to execute:
the actual final candidate capture has67/67 blocks selecting FP32,14 loaded,
zero FP64 candidates. See `/tmp/colibri_cached_final_metric_branch.json`.
This is one final snapshot, not a runtime branch histogram.

An older byte-gated frozen67point32sweep diagnostic had zero FP64 calls and
at most7 Newton iterations in biased solves (5 in relaxation), with most calls
inactive. `/tmp/colibri_condensed_metric_counts.json` cannot establish present
trajectory iteration counts.

## Bounded next attribution

Before another optimization, capture one actual current biased call and replay
identical restored inputs with counted native metric iterations/branches; verify
all output bytes against the uninstrumented helper. Attribute Jv evaluation,
projection and physical scatter separately, treating ablated replay only as
cost attribution, never physics evidence. The current evidence rejects a spill
explanation and shows a persistent serialized perpoint path. It does not yet
rank global-memory latency, unconditional FP64 Jv or rare nonlinear root work.

A potential substantial replacement would keep the small component's physical
velocity and invariant response state local/cooperative across the ordered
contact loop, then scatter once. Preserve each original point law and drive
reaction; prove numerical acceptance before timing. Do not substitute radial
clamping, truncate physical modes or remove precision solely to claim speed.

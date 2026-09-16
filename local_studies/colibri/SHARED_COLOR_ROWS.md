# Local shared-mask color builder: rejected

The local prototype retains serial greedy row order and first-free-color selection, caching the first eight uint32 mask words per body in 4 KiB CUDA shared memory. Higher colors use global storage; CPU, generic eight-endpoint graphs, and graphs above 128 bodies retain canonical dispatch. All mask output words are preserved; no graph rows or colors are truncated.

Independent Python-set first-fit reference plus canonical CUDA comparison passes 144 cases, including captured repeats, active growth/shrinkage/zero, nonzero inactive sentinels, duplicate endpoints, body zero, all-inactive endpoints, 256/257/300-color transitions, and 128/129-body fallback. Local source is shared_color_rows.py; harness check_shared_color_rows.py. CUDA log: /tmp/shared_color_rows_correctness.log.

Frozen actual historical first-frame Colibri graph: 345 active rows, 148 colors, 38 bodies, capacity8230. Input /tmp/colibri_public_internals_first.internals.npz, SHA256 b47fcafb2af37c8abb5d04e1a28bbadf6432d0ce39ae6e70695370fd80130595. This is actual captured topology but predates the current reducer/manifold changes; it is not a current public trajectory proof.

Twelve forward/reverse interleaved samples of 128-call CUDA graphs include the same mask/count zero fills for both paths. Median complete builder cost: canonical181.136 microseconds, shared212.449 microseconds, approximately17.3% slower. GPU compute-process query was empty immediately before launch and peers held the reserved window. All output buffers match independent oracle bytes.

Artifact: /tmp/shared_color_rows_frozen_timing.json and .log; harness benchmark_shared_color_rows.py. The candidate is rejected and remains local. No full-scene run, new FPS claim, production edit, or promotion is justified. Shared-memory initialization/flush, native code generation, and serial dependency costs were not individually isolated; this timing does not establish which caused the slowdown.

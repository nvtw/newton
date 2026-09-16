# Cooperative forward substitution

The local cooperative_bilateral_forward.py installer extends the accepted
eight-lane RHS callback. Each lane retains one row's forward-substitution
value. Solved values are broadcast in increasing row order; each later row
subtracts those terms in the same order as the scalar implementation.
Diagonal division remains FP64 division and occurs after forward substitution.

The leader gathers the scaled rows, then executes the original back
substitution and impulse accumulation. In particular, each descending
back-substitution row still subtracts its higher-row terms in increasing
index order. No reciprocal approximation, mass scaling, altered row order,
or changed contact scheduling is introduced.

## Validation

Eight focused methods pass, covering scalar/cooperative byte identity for
counts 0 through 6, disabled and invalid rows, partial tiles, live geometry
and joint-property changes, finite drive bounds, contact momentum/energy,
and mixed joint/contact momentum through relaxation.

Three isolated 330-frame runs use the same public 30 temporal steps,
120 Hz collision updates, one solver iteration and SOR 1:

| Run | Mean frame time | FPS |
| --- | ---: | ---: |
| Candidate | 11.0745 ms | 90.30 |
| Fresh canonical baseline | 11.2605 ms | 88.81 |
| Reverse candidate | 10.9529 ms | 91.30 |

Both candidates pass all scene checks. All ten saved arrays, including the
complete pose and velocity histories, are byte-identical to the fresh
baseline. The measured reduction is 1.65-2.73%; this is modest and the
candidate runs also show variability. No canonical code changed.

Reproduce:

    python -m local_studies.colibri.cooperative_bilateral_forward \
      local_studies.colibri.check_public_analytic_gradient \
      --frames 330 --substeps 30 --save-history \
      --output /tmp/colibri_cooperative_forward_candidate330.json

Artifacts use the /tmp/colibri_cooperative_forward_ prefix; the paired
timing summary is paired_summary.json.

## Frozen profile and integration gate

A restored first biased sweep after 330 frames matches all 355 mutable
world arrays. Nsight reports:

| Kernel | Mean time | Registers/thread | Local memory/thread |
| --- | ---: | ---: | ---: |
| Canonical RHS-only cooperation | 54.943 us | 128 | 0 |
| Cooperative forward candidate | 52.538 us | 136 | 0 |

Both launch 256 threads per block. The measured sweep reduction is 4.38%,
consistent with the modest full-step improvement. There is no measured spill.

The canonical implementation now extracts shared backward substitution and
impulse scatter, retaining scalar forward substitution for scalar execution.
The cooperative path broadcasts increasing forward rows and performs ordinary
FP64 diagonal division before gathering to the leader. Existing scalar
equivalence, lifecycle and physical invariant tests are the arithmetic gate;
the earlier launch-policy fail-before remains documented separately.

The local installer is now a compatibility entry when the canonical forward
path is present. The recorded pre-integration profile, rather than rerunning
two aliases of the canonical code, is the comparison evidence:
 /tmp/colibri_cooperative_forward_frozen_profile.nsys-rep and .sqlite.


## Canonical validation

The canonical runner completed all four cases with unchanged production source
hashes: Colibri330 (ten arrays), Kapla180 (six arrays), G1's20-step controller
history (two arrays), and Colibri18,000 (ten arrays) are byte-identical to
their respective references. The five-minute Colibri run measured11.082846
ms/frame (90.2295 FPS), p95 12.31738 ms, excluding rendering and physical audits.
Peak penetration0.753578 mm, anchor0.938996 mm and axis0.028822 rad are unchanged.
This preserves the baseline trajectory and its known drive/support convergence
limitations; it does not resolve them.

Report: /tmp/colibri_cooperative_forward_canonical_validation.json.

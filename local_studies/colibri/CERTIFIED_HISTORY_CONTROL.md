# Certified material-history controls

These are diagnostic local studies, not accepted production fixes.

## Identity retry excludes a proposed cause

`/tmp/colibri_history_retry600.json` preserved every existing contact-match winner,
then retried unclaimed previous identities using the original eligibility rules.
Across 1,200 collision updates it recovered **zero** additional identities:
25,358 originally matched and 54,966 unmatched. Original duplicate owners were
zero after the independent canonical ownership fix. The complete saved position,
velocity, and time histories were byte-identical to
`/tmp/colibri_native_direct_owned_fixed600.npz`.

Thus nearest-candidate contention is not this Colibri trajectory's creep cause.
The retry does fix a separate synthetic avoidable-loss example. The canonical
fix for duplicate fingerprint ownership is independently justified by failing
regressions and does not establish that ownership caused Colibri creep.

## Certified common material reference

The prototype samples one rigid relative-transform birth field at each current
original common contact point. It preserves every original point's normal law,
Coulomb cone, impulse location, and joint equations. It neither pools friction
capacity nor substitutes two force anchors. Membership uses the certified
connected coplanar source triangle union, preserving holes. Birth requires the
whole source facet to touch within an operand-aware FP32 arithmetic bound;
contact-offset distance is not used as the touching tolerance.

`/tmp/colibri_certified_patch600.json` was an **inactive control**: a COM
subtraction error bound omitted the original operands and rejected all 1,200
updates. It therefore did not test persistent common history.

`/tmp/colibri_certified_patch_fresh600.json` corrected the arithmetic guard and
sampled the field at the fresh common physical point. It activated 1,190 of
1,200 updates, but the conservative any-member-broken veto caused **1,190
rebirths and zero retentions**. This still does not test a retained common birth.

Results of that rebirth-only run:

- 5–10 second creep: 3.04399 micrometers/second.
- Final hinge: 18.20759 degrees.
- Final hard-row residual: 2.90e-11; drive residual: 8.34e-10.
- Final paired joint linear and world-angular impulse defects: zero.
- 13.91 FPS, including the deliberately serial source-triangle membership
  diagnostic. This performance is unacceptable and is not a production claim.

The next diagnostic records whether the previous-member veto comes from
zero-load quadrature points or positive-load sliding points. It must not silently
ignore loaded slip. `certified_patch_break_probe.instrument(source)` adds a
read-only 128-ingest ring without changing the veto or any physical equation.

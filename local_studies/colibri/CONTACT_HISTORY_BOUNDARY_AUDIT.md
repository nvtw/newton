# Two-body contact identity/history audit

CPU/read-only audit of `/tmp/colibri_base_frame_totalnormal_frame330.npz` at actual collision-query boundary: final relaxation of microstep29 → warm-before of microstep30. The latter occurs after ingestion but before preparation/warm solve. Therefore these resets are distinct from the98percent friction-cap preparation bug.

## Measured observations

-67active points before and after,12contact columns.27six-component friction-anchor tuples retained byte-exact;40not found anywhere in previous tuples. No permutation among27retained anchors.
- All67contact normals remain byte-identical. Normal-dot threshold rejection is not supported by this sample.
-39of40reset points have positive normal bias at new warm preparation. Eight reset slots previously had positive normal impulses, totaling71.124µNs out of161.608µNs (44.01percent).
- Seven loaded reset slots11–17have maximum local anchor-component changes only44.7–216.1nm; these all have positive new normal bias. The eighth,slot35, has nonpositive bias and5.970mm anchor change, consistent with a changed feature rather than harmless reorder. No proof of that feature classification without raw shape/point keys.
- Ring lacks raw matcher indices, previous/new sorted shape keys, candidate distances and claim winners. Thus reset is established, but exact rejection reason and one-to-one identity of changed slots are not certified. Loaded fractions use unchanged slot correspondence; retained anchors support stable ordering but do not prove all changed slots retain identity.

## Exact source mechanisms

`newton/_src/geometry/contact_match.py:236` immediately marks a sticky contact broken whenever freshly regenerated gap is positive, before searching old contacts. This is consistent with39positive-bias resets and tiny anchor motion, but fresh-gap values are not captured directly. A real separating contact must not retain active friction impulses; whether its dormant identity should survive tiny separation is a distinct policy question.

The matcher searches nearest prior point within the same shape pair, ignores unstable subkeys, applies500µm tangential distance threshold andnormaldot.995. It claims old points deterministically by distance andkey. `_resolve_claims_kernel:318` demotes collision losers with **no second-closest fallback**. This is deterministic but can unnecessarily lose valid correspondences. Example in1Dmillimeters: old points[0,.4], new[.1,.15], threshold.5. Both select old0; onlyone survives although new.15→old.4 is valid. This is a source-level opportunity, not evidence it occurred in the saved two-body query.

PhoenX `contact_ingest.py:967` copies matched friction references if normaldot≥.95, independently of impulse warm starting. Its gather reconstructs fresh anchors otherwise. Sorting translates match indices through the previous inverse permutation (`contact_ingest.py:378`), so row reorder alone is intentionally supported. Current data shows no failure of that translation.

## Bounded improvements worth testing

1. First add exact read-only query capture of rawmatch result, shape-pair keys, freshgap, nearest distances, claimwinner and finaltranslated index. Keep native trajectory equivalence gate. This distinguishes gapbreak, feature replacement and avoidable ownership conflict.
2. For proven conflicts, deterministic per-pair maximum-cardinality/minimum-distance matching or bounded second-choice arbitration can retain additional valid unique matches. Preserve existing distance/normal gates and deterministic tie ordering; never transfer history across shape pairs. Test many-to-one ties, permutation, duplicates, capacity, normals and disappearing features.
3. Separately evaluate dormant identity across a small geometric separation within existing contact-search eligibility: active normal/tangent impulses remain zero, no friction force without physical normal load, and anchors still expire by actual tangential slip/normal/feature incompatibility. This is a policy experiment requiring lift/recontact/sliding/energy checks, not permission to widen contact force activation or freeze old normal-gap witnesses. Coordinate with Kepler’s explicit broken-bit work to avoid overriding a genuine slip release.

No production contact-file changes, no GPU runs. Numeric artifact `/tmp/colibri_contact_history_boundary_audit.json`.

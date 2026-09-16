"""Local-only dormant contact identity experiment; never reuse separated witnesses."""

import inspect
import sys
import tempfile
import types
from pathlib import Path

import warp as wp

from newton._src.geometry import contact_match


@wp.kernel(enable_backward=False)
def _geometry_matches(
    count: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    margin0: wp.array[wp.float32],
    margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    identity: wp.array[wp.int32],
    geometry: wp.array[wp.int32],
):
    k = wp.tid()
    geometry[k] = wp.int32(-1)
    if k >= count[0]:
        return
    source = permutation[k]
    a = shape_body[shape0[source]]
    b = shape_body[shape1[source]]
    p0 = point0[source]
    p1 = point1[source]
    if a >= 0:
        p0 = wp.transform_point(body_q[a], p0)
    if b >= 0:
        p1 = wp.transform_point(body_q[b], p1)
    gap = wp.dot(p1 - p0, normal[source]) - (margin0[source] + margin1[source])
    if gap <= 0.0:
        geometry[k] = identity[k]


def _candidate_kernel():
    source = inspect.getsource(contact_match._match_contacts_kernel.func)
    old = "fresh_gap > wp.float32(0.0)"
    assert source.count(old) == 1
    source = source.replace(old, "fresh_gap > wp.sqrt(data.pos_threshold_sq)")
    source = source.replace("def _match_contacts_kernel(", "def _match_contacts_history_only_kernel(")
    module = types.ModuleType("colibri_history_match_candidate")
    module.__dict__.update(vars(contact_match))
    module.__name__ = "colibri_history_match_candidate"
    path = Path(tempfile.gettempdir()) / "colibri_history_match_candidate.py"
    path.write_text(source)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module._match_contacts_history_only_kernel


def install(pipeline):
    """Install before graph capture; combine with positive-gap impulse clearing.

    This changes only candidate identity eligibility and routes a distinct
    geometry map to sorting. It does not by itself change PhoenX's warm-start
    gather; a live experiment must disable impulse warmstarting or install
    Kepler's separate clear-positive-impulse gather transformation.
    """
    assert pipeline._matching_sticky
    matcher = pipeline._contact_matcher
    sorter = pipeline._contact_sorter
    geometry = wp.empty(matcher._capacity, dtype=wp.int32, device=pipeline.device)
    kernel = _candidate_kernel()
    old_match = matcher.match
    old_sort = sorter.sort_full

    def match(**kwargs):
        assert kwargs["canonical_to_source"] is not None
        original = contact_match._match_contacts_kernel
        contact_match._match_contacts_kernel = kernel
        try:
            old_match(**kwargs)
        finally:
            contact_match._match_contacts_kernel = original
        wp.launch(
            _geometry_matches,
            matcher._capacity,
            [
                kwargs[k]
                for k in (
                    "contact_count",
                    "canonical_to_source",
                    "point0",
                    "point1",
                    "shape0",
                    "shape1",
                    "normal",
                    "margin0",
                    "margin1",
                    "body_q",
                    "shape_body",
                    "match_index_out",
                )
            ]
            + [geometry],
            device=pipeline.device,
        )

    def sort_full(**kwargs):
        if kwargs.get("sticky_match_index") is not None:
            kwargs["sticky_match_index"] = geometry
        return old_sort(**kwargs)

    matcher.match = match
    sorter.sort_full = sort_full
    return {"geometry_matches": geometry, "scope": "Dormant identity only; positive-gap geometry always fresh"}

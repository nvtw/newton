"""Local causal control: disable only negative numerical contact recovery."""

import runpy

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_contact_count,
    contact_get_contact_first,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import _get_parallel_contact_prepare_kernel
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer


@wp.kernel(enable_backward=False)
def clamp_recovery(columns: ContactColumnContainer, active: wp.array[wp.int32], cc: ContactContainer):
    col = wp.tid()
    if col < active[0]:
        first = contact_get_contact_first(columns, col)
        count = contact_get_contact_count(columns, col)
        for index in range(count):
            k = first + index
            if cc.derived[8, k] <= 0.0 and cc.derived[3, k] < 0.0:
                cc.derived[3, k] = 0.0


kernels = tuple(_get_parallel_contact_prepare_kernel(True, soft) for soft in (False, True))
original = wp.launch
calls = [0]


def launch(kernel, *args, **kwargs):
    result = original(kernel, *args, **kwargs)
    if kernel in kernels:
        inputs = kwargs.get("inputs", args[1] if len(args) > 1 else None)
        assert inputs is not None
        original(
            clamp_recovery, inputs[0].data.shape[1], [inputs[0], inputs[1], inputs[6]], device=kwargs.get("device")
        )
        calls[0] += 1
    return result


wp.launch = launch
try:
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
finally:
    wp.launch = original
    print("NO_NEGATIVE_RECOVERY_CAPTURE_CALLS", calls[0], flush=True)
    assert calls[0] > 0, "Preparation hook did not run"

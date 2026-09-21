import inspect
import textwrap
import unittest

import newton._src.solvers.phoenx.tests.test_contact_coupling as t

source = textwrap.dedent(
    inspect.getsource(t.TestContactCoupling.test_sustained_self_contact_and_motor_conserve_momentum)
)
source = source.replace("range(200)", "range(1)")
source = source.replace(
    "np.testing.assert_allclose(after, before, rtol=0.0, atol=2.0e-4)",
    "print('MOMENTUM', after-before, state.joint_q.numpy(), flush=True)",
)
source = source.replace("self.assertGreater(peak_friction, 1.0e-6)", "print('FRICTION',peak_friction,flush=True)")
source = source.replace(
    "np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0.0, atol=2.0e-4)", "pass"
)
for mu, force, bend in [(0.0, 0.0, True), (0.5, 0.0, True), (0.0, 0.1, True), (0.5, 0.1, True), (0.5, 0.1, False)]:
    print("CASE", mu, force, bend, flush=True)
    s = source.replace("fill_(0.5)", f"fill_({mu})").replace("joint_force[-1] = 0.1", f"joint_force[-1] = {force}")
    if not bend:
        s = s.replace("[0.3, -0.6]", "[0.0, 0.0]")
    scope = dict(vars(t))
    exec(s, scope)
    scope["test_sustained_self_contact_and_motor_conserve_momentum"](unittest.TestCase())

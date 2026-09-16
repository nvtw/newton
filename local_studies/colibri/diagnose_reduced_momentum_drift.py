import inspect, textwrap, unittest
import newton._src.solvers.phoenx.tests.test_contact_coupling as t
source = textwrap.dedent(inspect.getsource(t.TestContactCoupling.test_sustained_self_contact_and_motor_conserve_momentum))
source = source.replace("range(200)", "range(41)")
source = source.replace("step % 20", "step % 10")
source = source.replace("np.testing.assert_allclose(after, before, rtol=0.0, atol=2.0e-4)", "print('MOMENTUM',step,after-before,flush=True)")
source = source.replace("self.assertGreater(peak_friction, 1.0e-6)", "print('FRICTION',peak_friction,flush=True)")
source = source.replace("np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0.0, atol=2.0e-4)", "pass")
for contacts, relax in [(False,True),(True,True),(True,False)]:
    print("CASE",contacts,relax,flush=True)
    s=source
    if not contacts: s=s.replace("pipeline.collide(state, contacts)", "contacts.rigid_contact_count.zero_()")
    if not relax:s=s.replace("solver_iterations=8,", "solver_iterations=8, velocity_iterations=0,")
    scope=dict(vars(t)); exec(s,scope)
    scope["test_sustained_self_contact_and_motor_conserve_momentum"](unittest.TestCase())

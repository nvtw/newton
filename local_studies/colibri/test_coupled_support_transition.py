"""Frozen live mode-transition regression; all contacts remain in acceptance."""
import unittest
import numpy as np
from local_studies.colibri import coupled_support_online as reference
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator

class TestCoupledModeTransition(unittest.TestCase):
    def test_resolved_two_point_transition(self):
        """Resolve the live sticking/sliding stall without deleting its small mode."""
        d=np.load("/tmp/colibri_two_body_coupled_live60.rejected.npz")
        lam,result=reference.solve_online(d["A"],d["rhs"],d["gamma"],d["mu"],d["old"])
        self.assertTrue(result["accepted"],result)
        self.assertLess(result["final_physical_error"],1e-8)
        evaluate,*_=natural_map_evaluator(d["A"],d["rhs"],d["gamma"],d["mu"])
        self.assertLess(np.max(np.abs(evaluate(lam)[0])),1e-8)
        p=np.flatnonzero(lam[::3]>1e-12)
        tangent=(3*p[:,None]+np.array([1,2])).ravel()
        smallest=np.linalg.svd(d["C"].T[:,tangent],compute_uv=False)[-1]
        self.assertGreater(smallest,1e-8,"Fixture must retain a small resolved physical mode")
        self.assertLess(smallest,1e-6)
        joint=np.linalg.solve(d["K"],d["targets"]-d["B"]@d["free"]-d["B"]@d["W"]@d["C"].T@lam)
        v=d["free"]+d["W"]@(d["C"].T@lam+d["B"].T@joint)
        np.testing.assert_allclose(d["B"]@v+d["diagonal"]*joint,d["targets"],atol=1e-10,rtol=0)

    def test_many_contact_sticking_boundary(self):
        """Cross a live sticking boundary while retaining every physical row."""
        d=np.load("/tmp/colibri_two_body_coupled_live60_newton.rejected.npz")
        lam,result=reference.solve_online(d["A"],d["rhs"],d["gamma"],d["mu"],d["old"],d["C"])
        self.assertTrue(result["accepted"],result)
        evaluate,*_=natural_map_evaluator(d["A"],d["rhs"],d["gamma"],d["mu"])
        self.assertLess(np.max(np.abs(evaluate(lam)[0])),1e-8)
        self.assertGreaterEqual(np.min(lam[::3]),-1e-12)
        self.assertTrue(any(stage.get("boundary_search") for stage in result["stages"]))
        joint=np.linalg.solve(d["K"],d["targets"]-d["B"]@d["free"]-d["B"]@d["W"]@d["C"].T@lam)
        v=d["free"]+d["W"]@(d["C"].T@lam+d["B"].T@joint)
        np.testing.assert_allclose(d["B"]@v+d["diagonal"]*joint,d["targets"],atol=1e-10,rtol=0)

if __name__=="__main__":unittest.main()

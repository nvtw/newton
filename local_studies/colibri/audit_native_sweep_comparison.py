"""Read-only matched-window drift and provenance audit of native sweep counts."""
import json
from pathlib import Path
import numpy as np
from local_studies.colibri.audit_native_conditioned_motion import relative,world_relative,angles

def main():
    """Compare the same five-to-ten-second interval without changing any check."""
    prefixes={1:"/tmp/colibri_native_direct_normalized3600",
              2:"/tmp/colibri_native_normalized_sweeps2_600",
              4:"/tmp/colibri_native_normalized_sweeps4_600"}
    reports=[]
    for n,prefix in prefixes.items():
        if not Path(prefix+".npz").exists():
            continue
        z=np.load(prefix+".npz")
        t=z["history_times"];q=z["q_history"].astype(float)
        lo=int(np.argmin(abs(t-5)));hi=int(np.argmin(abs(t-10)))
        assert abs(t[lo]-5)<1e-8 and abs(t[hi]-10)<1e-8
        delta=q[hi,0,:3]-q[lo,0,:3]
        provenance=json.load(open(prefix+".native_conditioned.json"))
        assert provenance["production_unchanged"] and provenance["differential_history_rows"]==27
        assert provenance["normalized_rotation_increment"]
        result=json.load(open(prefix+".json"))
        hinge=relative(q[hi,0,3:],q[hi,1,3:])
        reports.append(dict(sweeps=n,source=prefix,window=[float(t[lo]),float(t[hi])],
            delta_um=(delta*1e6).tolist(),xy_rate_um_s=float(np.linalg.norm(delta[:2])*1e6/(t[hi]-t[lo])),
            max_excursion_um=float(np.linalg.norm(q[lo:hi+1,0,:2]-q[lo,0,:2],axis=1).max()*1e6),
            world_rotation_deg=angles(world_relative(q[lo,0,3:],q[hi,0,3:])),
            hinge_deg=float(np.degrees(2*np.arctan2(hinge[2],hinge[3]))),
            timing=result["timing"],bounds_passed=result["passed"]))
    Path("/tmp/colibri_native_sweep_comparison.json").write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))

if __name__=="__main__":
    main()

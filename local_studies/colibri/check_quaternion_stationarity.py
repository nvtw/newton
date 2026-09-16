"""CPU FP32 zero/small-angular-velocity quaternion stationarity audit."""
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.access_mode import integrate_orientation


@wp.kernel(enable_backward=False)
def repeat(qin:wp.array[wp.quatf],out:wp.array2d[wp.quatf],omega:wp.float32):
    """At zero omega this is exactly the native final normalization map."""
    i=wp.tid()
    q=qin[i]
    out[i,0]=q
    for k in range(512):
        q=integrate_orientation(q,wp.vec3f(0.,0.,omega),wp.float32(1./3600.))
        if k<4:
            out[i,k+1]=q
    out[i,5]=q

def main():
    """Test saved actual base quaternions without initializing CUDA."""
    birth=np.load("/tmp/colibri_analytical_birth_capture.birth.npz")
    native=np.load("/tmp/colibri_native_direct_combined3600.npz")
    q=np.vstack([birth["orientation"][1],native["initial_q"][0,3:],native["q_history"][:,0,3:]]).astype(np.float32)
    report={}
    for omega in (0.,1e-8,1e-6):
        out=wp.zeros((len(q),6),dtype=wp.quatf,device="cpu")
        wp.launch(repeat,dim=len(q),inputs=[wp.array(q,dtype=wp.quatf,device="cpu"),out,wp.float32(omega)],device="cpu")
        values=out.numpy()
        changes=np.any(values[:,1:].view(np.uint32)!=values[:,:-1].view(np.uint32),axis=2)
        norm=np.linalg.norm(values.astype(float),axis=2)
        report[str(omega)]={"count": len(q),"changed_each_interval": np.sum(changes,axis=0).tolist(),
            "birth_values": values[0].tolist(),"max_component_change": float(np.max(abs(values[:,-1]-q))),
            "norm_min": float(norm.min()),"norm_max": float(norm.max()),
            "cycles_2": int(np.sum(np.all(values[:,1].view(np.uint32)==values[:,3].view(np.uint32),axis=1)&
                               np.any(values[:,1].view(np.uint32)!=values[:,2].view(np.uint32),axis=1)))}
        np.savez("/tmp/colibri_quaternion_stationarity_"+str(omega)+".npz",q=values)
    report["native_history"]={"norm_min": float(np.linalg.norm(q[2:].astype(float),axis=1).min()),
        "norm_max": float(np.linalg.norm(q[2:].astype(float),axis=1).max())}
    report["scope"]="Actual Warp CPU FP32 polynomial helper; zero omega matches native final normalization. GPU rounding not established. Nonzero helper is copy-prediction polynomial, not full momentum-preserving native integration."
    Path("/tmp/colibri_quaternion_stationarity.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()

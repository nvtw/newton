import json
import numpy as np
from local_studies.colibri.analyze_late_contacts import transform
t=np.load("/tmp/colibri_slabs8_tail_trace.trace.npz")
step=746
slot=int(np.flatnonzero(t["step_ids"]==step)[0])
q,qd=t["pre_q"][slot],t["pre_qd"][slot]
raw=np.load(f"/tmp/colibri_slab_tail_pair_{step}_raw_predictive.npz")
reduced=np.load(f"/tmp/colibri_slab_tail_pair_{step}_reduced_predictive.npz")
shape_body=t["shape_body"]
bodies=shape_body[raw["shapes"][0]]
a,b=map(int,bodies)
p0=transform(q[a],raw["point0"])
p1=transform(q[b],raw["point1"])
point=.5*(p0+p1)
com0=transform(q[a],t["body_com"][a])
com1=transform(q[b],t["body_com"][b])
v0=qd[a,:3]+np.cross(qd[a,3:],point-com0)
v1=qd[b,:3]+np.cross(qd[b,3:],point-com1)
vn=np.sum((v1-v0)*raw["normals"],axis=1)
gap=raw["pre_gap"]
pred=gap+vn/120
after=raw["transported_post_gap"]
future=after < -.0001
dot=raw["normals"]@reduced["normals"].T
maxdot=dot.max(axis=1)
result={}
for name,mask in [("all",np.ones(len(gap),bool)),("future_penetrating",future),("future_predicted_impact",future&(pred<0)),("future_initially_receding",future&(vn>=0))]:
 ids=np.flatnonzero(mask)
 result[name]={"count":len(ids),"pre_gap_range":[float(gap[ids].min()),float(gap[ids].max())] if len(ids) else None,"vn_range":[float(vn[ids].min()),float(vn[ids].max())] if len(ids) else None,"pred_gap_range":[float(pred[ids].min()),float(pred[ids].max())] if len(ids) else None,"max_dot_reduced_range":[float(maxdot[ids].min()),float(maxdot[ids].max())] if len(ids) else None}
result["worst_future_samples"]=[{"id":int(i),"pre_gap":float(gap[i]),"vn":float(vn[i]),"pred_gap":float(pred[i]),"post_gap":float(after[i]),"normal":raw["normals"][i].tolist(),"max_dot_reduced":float(maxdot[i])} for i in np.argsort(after)[:8]]
print(json.dumps(result,indent=2))
open("/tmp/colibri_slab_tail_746_classification.json","w").write(json.dumps(result,indent=2))

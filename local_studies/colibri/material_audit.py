import ast
import json
from collections import Counter
from pathlib import Path
from pxr import Usd, UsdShade, UsdPhysics
root="/World/Colibri/FrameAsm/"
stage=Usd.Stage.Open("/home/twidmer/Documents/colibri/Colibri.usd")
unit=float(stage.GetMetadata("metersPerUnit"))
values={}
for node in ast.parse(Path("newton/examples/kamino/example_kamino_colibri.py").read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        try: values[node.targets[0].id]=ast.literal_eval(node.value)
        except Exception: pass
materials=[]
for p in stage.Traverse():
    if p.IsA(UsdShade.Material) and p.HasAPI(UsdPhysics.MaterialAPI):
        api=UsdPhysics.MaterialAPI(p)
        materials.append(dict(path=str(p.GetPath()),density=api.GetDensityAttr().Get(),mu=api.GetDynamicFrictionAttr().Get(),mu_static=api.GetStaticFrictionAttr().Get(),restitution=api.GetRestitutionAttr().Get(),extra={a.GetName():str(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValueOpinion() and a.GetName().startswith("physxMaterial:")}))
shapes=[]; mismatch=[]
for name,kind,label,data,dimensions in values["SHAPES"]:
    p=stage.GetPrimAtPath(root+label)
    if not p: continue
    mat,rel=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
    expected=values["SHAPE_MATERIALS"].get(label,(1000.,.5))
    actual=(1000.,.5)
    if mat and mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        api=UsdPhysics.MaterialAPI(mat)
        actual=(api.GetDensityAttr().Get()/unit**3,api.GetDynamicFrictionAttr().Get())
    record=dict(label=label,material=str(mat.GetPath()) if mat else None,relation=str(rel.GetPath()) if rel else None,source=actual,literal=expected,collision=p.GetAttribute("physics:collisionEnabled").Get(),approx=p.GetAttribute("physics:approximation").Get(),attrs={a.GetName():str(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValueOpinion() and a.GetName().startswith(("physxCollision:","physxSDFMeshCollision:","physics:mass","physics:density","physics:diagonalInertia","physics:centerOfMass","physics:principalAxes"))})
    shapes.append(record)
    if abs(actual[0]-expected[0])>max(1e-3,actual[0]*1e-6) or actual[1]!=expected[1]: mismatch.append(record)
bodies=[]
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        attrs={a.GetName():str(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValueOpinion() and a.GetName().startswith(("physxRigidBody:","physics:mass","physics:density","physics:diagonalInertia","physics:centerOfMass","physics:principalAxes"))}
        bodies.append(dict(path=str(p.GetPath()),attrs=attrs))
report=dict(materials=materials,shapes=shapes,mismatches=mismatch,bodies=bodies)
Path("/tmp/colibri_material_audit.json").write_text(json.dumps(report,indent=2))
print("MATERIALS",json.dumps(materials,indent=2))
print("SHAPES",len(shapes),"BOUND",sum(x["material"] is not None for x in shapes),"MISMATCH",json.dumps(mismatch))
print("APPROX",Counter(x["approx"] for x in shapes))
print("COLLISION_EXTRA",[(x["label"],x["attrs"]) for x in shapes if x["attrs"] and any(not key.startswith("physxSDFMeshCollision:") for key in x["attrs"])])
print("BODIES",json.dumps(bodies,indent=2))

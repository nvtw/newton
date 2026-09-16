"""One-time extraction of Colibri assembly constants; never used by the example."""

from pathlib import Path
import pprint
import re

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import cKDTree

SOURCE = Path('/home/twidmer/Documents/colibri')
DEST = Path('newton/examples/assets/colibri')
DEST.mkdir(parents=True, exist_ok=True)
stage = Usd.Stage.Open(str(SOURCE / 'Colibri.usd'))
cache = UsdGeom.XformCache()
root = '/World/Colibri/FrameAsm/'
# Centimeters and the authored 1.5 assembly scale, baked into meter assets.
scale = 0.015
bodies = {str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
flower = root + 'FrameGround/Flower'
# The flower is kinematic with its only joint disabled: attach it to the base.
del bodies[flower]

def owner(p):
    while p and str(p.GetPath()) not in bodies:
        p = p.GetParent()
    return str(p.GetPath()) if p else None

def pose(mat):
    t = Gf.Transform(mat)
    q = t.GetRotation().GetQuat()
    return tuple(round(float(x), 9) for x in (*t.GetTranslation(), *q.GetImaginary(), q.GetReal()))

def rigid_matrix(mat):
    t = Gf.Transform(mat)
    out = Gf.Matrix4d().SetRotate(t.GetRotation())
    out.SetTranslateOnly(t.GetTranslation() * 0.01)
    return out

def local_matrix(p, body):
    return cache.GetLocalToWorldTransform(p) * cache.GetLocalToWorldTransform(body).GetInverse()

def local_pose(mat):
    t = Gf.Transform(mat)
    mat = Gf.Matrix4d().SetRotate(t.GetRotation())
    mat.SetTranslateOnly(t.GetTranslation() * scale)
    return pose(mat)

obj_files = {}
for f in SOURCE.rglob('*.obj'):
    obj_files.setdefault(f.name, f)
obj_vertices = {name: np.array([list(map(float, line.split()[1:4])) for line in f.read_text().splitlines() if line.startswith('v ')]) for name, f in obj_files.items()}
normalized = {re.sub(r'[^a-z0-9]', '', Path(n).stem.lower()): n for n in obj_files}
shapes = []
copied = set()
fallbacks = []
for p in stage.Traverse():
    if not p.IsA(UsdGeom.Mesh) or not str(p.GetPath()).startswith(root):
        continue
    body_path = owner(p)
    if body_path is None:
        continue
    body = bodies[body_path]
    mesh = UsdGeom.Mesh(p)
    vertices = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    mat = local_matrix(p, body)
    label = str(p.GetPath()).removeprefix(root)
    # USD cylinders are canonical 100-unit Y-axis meshes with radius 50.
    if p.GetName().startswith('Cylinder'):
        assert len(vertices) == 130
        center = mat.Transform(Gf.Vec3d(0)) * scale
        axis = mat.TransformDir(Gf.Vec3d(0, 1, 0))
        radial_x = mat.TransformDir(Gf.Vec3d(1, 0, 0)).GetLength()
        radial_z = mat.TransformDir(Gf.Vec3d(0, 0, 1)).GetLength()
        assert abs(radial_x-radial_z) < 1e-5
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), axis.GetNormalized())
        cylinder_mat = Gf.Matrix4d().SetRotate(rotation)
        cylinder_mat.SetTranslateOnly(center)
        shapes.append((body.GetName(), 'cylinder', label, pose(cylinder_mat), (round(50*radial_x*scale, 9), round(50*axis.GetLength()*scale, 9))))
        continue
    name = re.sub(r'[^a-z0-9]', '', p.GetName().lower().replace('_mesh', '').replace('_mirrored', ''))
    candidates = [n for key, n in normalized.items() if name.startswith(key)]
    if p.GetName() == 'FrameMesh':
        candidates = ['Frame.obj']
    candidates.sort(key=len, reverse=True)
    match = None
    for candidate in candidates:
        v = obj_vertices[candidate]
        if len(v) == len(vertices) and cKDTree(v).query(vertices)[0].max() < 2e-5:
            match = candidate
            break
    if match:
        if match not in copied:
            lines = obj_files[match].read_text().splitlines()
            lines = ['v '+' '.join(f'{float(x)*scale:.10g}' for x in line.split()[1:4]) if line.startswith('v ') else line for line in lines]
            (DEST/match).write_text('\n'.join(lines)+'\n')
            copied.add(match)
        # Bake the mesh-local USD transform into a small affine matrix in SI.
        linear = np.array(mat)[:3,:3].T
        translation = np.array(mat)[3,:3]*scale
        transform = tuple(tuple(round(float(x), 10) for x in row) for row in np.column_stack((linear, translation)))
    else:
        # Mirrored/edited parts and the slider have no identical source OBJ.
        match = p.GetName()+'.usd.obj'
        fallback_vertices = np.array([mat.Transform(Gf.Vec3d(*v)) for v in vertices])*scale
        lines = ['# Geometry extracted from Colibri.usd; meters.']
        lines += ['v '+' '.join(f'{x:.10g}' for x in v) for v in fallback_vertices]
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        offset = 0
        for count in counts:
            face = list(indices[offset:offset+count])
            for k in range(1, count-1):
                lines.append(f'f {face[0]+1} {face[k]+1} {face[k+1]+1}')
            offset += count
        (DEST/match).write_text('\n'.join(lines)+'\n')
        transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0))
        fallbacks.append(label)
    shapes.append((body.GetName(), 'mesh', label, match, transform))

joints = []
seen = set()
for p in stage.Traverse():
    if not p.IsA(UsdPhysics.Joint) or not UsdPhysics.Joint(p).GetJointEnabledAttr().Get():
        continue
    j = UsdPhysics.Joint(p)
    targets = [j.GetBody0Rel().GetTargets()[0], j.GetBody1Rel().GetTargets()[0]]
    body_paths = [owner(stage.GetPrimAtPath(t)) for t in targets]
    frames = []
    for i, (target, body_path) in enumerate(zip(targets, body_paths, strict=True)):
        pos = p.GetAttribute(f'physics:localPos{i}').Get()
        rot = p.GetAttribute(f'physics:localRot{i}').Get()
        frame = Gf.Matrix4d().SetRotate(Gf.Quatd(rot))
        frame.SetTranslateOnly(Gf.Vec3d(pos))
        frame = frame * local_matrix(stage.GetPrimAtPath(target), bodies[body_path])
        frames.append(local_pose(frame))
    names = [bodies[b].GetName() for b in body_paths]
    axis = str(p.GetAttribute('physics:axis').Get() or 'Z')
    kind = 'ball' if p.IsA(UsdPhysics.SphericalJoint) else 'revolute'
    signature = (tuple(names), tuple(frames), axis, kind)
    if signature in seen:
        continue
    seen.add(signature)
    drive = {}
    for usd_name, newton_name in [('stiffness','target_ke'),('damping','target_kd'),('targetPosition','target_pos'),('targetVelocity','target_vel')]:
        attr = p.GetAttribute('drive:angular:physics:'+usd_name)
        if p.HasAPI(UsdPhysics.DriveAPI, "angular") and attr and attr.HasAuthoredValueOpinion():
            # USD angular drive gains are torque/degree, with torque in kg*cm²/s².
            factor = 0.01**2*180/np.pi if usd_name in ('stiffness','damping') else np.pi/180
            drive[newton_name] = float(attr.Get())*factor
    joints.append((names[0],names[1],kind,axis,*frames,drive))

# Traverse revolute parents first; spherical joints close the wing loops.
order = ['FrameGround']
while True:
    children = [b for a,b,kind,*_ in joints if kind == 'revolute' and a in order and b not in order]
    if not children:
        break
    order.extend(dict.fromkeys(children))
missing = [p.GetName() for p in bodies.values() if p.GetName() not in order]
assert missing == ['TailRack'], missing
order.extend(missing)
poses = {p.GetName(): pose(rigid_matrix(cache.GetLocalToWorldTransform(p))) for p in bodies.values()}
collision_labels = [str(p.GetPath()).removeprefix(root) for p in stage.Traverse() if str(p.GetPath()).startswith(root) and p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
data = dict(BODY_ORDER=order, BODY_POSES=poses, SHAPES=shapes, JOINTS=joints, COLLISION_LABELS=collision_labels)
Path('/tmp/colibri_constants.py').write_text('\n\n'.join(name+' = '+pprint.pformat(value, width=115, sort_dicts=False) for name,value in data.items())+'\n')
print('Bodies',len(bodies),'joints',len(joints),'shapes',len(shapes),'cylinders',sum(x[1]=='cylinder' for x in shapes),'matched OBJ files',len(copied),'USD fallback meshes',len(fallbacks))
print('ORDER',order)
print('FALLBACKS',fallbacks)

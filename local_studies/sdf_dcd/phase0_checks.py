"""Phase 0: check interior SDF values and hw-filter precision on texture SDFs."""

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd as newton_usd
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf, texture_sample_sdf_hw


@wp.kernel
def sample(sdf: TextureSDFData, pts: wp.array[wp.vec3], sw: wp.array[float], hw: wp.array[float]):
    i = wp.tid()
    sw[i] = texture_sample_sdf(sdf, pts[i])
    hw[i] = texture_sample_sdf_hw(sdf, pts[i])


@wp.kernel
def exact(mesh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], 1.0e6)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        out[i] = q.sign * wp.length(pts[i] - cp)


def main():
    from pxr import Usd

    stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
    mesh = newton_usd.get_mesh(stage.GetPrimAtPath("/root/bunny"))
    for band, res in [((-0.005, 0.005), 256), ((-0.1, 0.1), 128), ((-0.001, 0.001), 128)]:
        m = newton.Mesh(mesh.vertices, mesh.indices)
        m.build_sdf(max_resolution=res, narrow_band_range=band, margin=0.01)
        sdf = m.sdf.texture_data
        lo, hi = np.min(m.vertices, 0), np.max(m.vertices, 0)
        rng = np.random.default_rng(0)
        pts = rng.uniform(lo, hi, size=(20000, 3)).astype(np.float32)
        wpts = wp.array(pts, dtype=wp.vec3)
        sw = wp.zeros(len(pts), dtype=float)
        hw = wp.zeros(len(pts), dtype=float)
        ex = wp.zeros(len(pts), dtype=float)
        wp.launch(sample, len(pts), inputs=[sdf, wpts, sw, hw])
        wmesh = wp.Mesh(wp.array(m.vertices, dtype=wp.vec3), wp.array(m.indices.flatten(), dtype=int))
        wp.launch(exact, len(pts), inputs=[wmesh.id, wpts, ex])
        sw, hw, ex = sw.numpy(), hw.numpy(), ex.numpy()
        inside = ex < 0
        deep = ex < 2 * band[0]
        L = np.max(hi - lo)
        print(f"band={band} res={res} L={L:.3f} voxel_radius={float(sdf.voxel_radius):.5f}")
        print(f"  inside n={inside.sum()} deep n={deep.sum()} min exact={ex.min():.4f} min sw={sw.min():.4f}")
        err = np.abs(sw - ex)
        print(f"  |sw-exact| all: mean={err.mean():.5f} p99={np.percentile(err, 99):.5f}")
        if deep.any():
            print(f"  |sw-exact| deep: mean={err[deep].mean():.5f} p99={np.percentile(err[deep], 99):.5f}")
        d = np.abs(hw - sw)
        print(f"  |hw-sw|: mean={d.mean():.6f} p99={np.percentile(d, 99):.6f} max={d.max():.6f}")
        print(f"  edges kept: {0 if m._collision_edges is None else len(m._collision_edges)}, verts {len(m.vertices)}")


if __name__ == "__main__":
    main()

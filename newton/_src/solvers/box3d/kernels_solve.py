# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fused contact + joint solve kernel for the Box3D solver.

Uses ``@wp.func_native`` with shared memory for body velocities and
``__syncthreads()`` barriers between graph colors — colored Gauss-Seidel
within a single CUDA thread block per world.

The kernel handles:
  - Loading body velocities into shared memory
  - Warm starting (first substep only)
  - Colored contact solve (biased or relaxation)
  - Restitution (last substep only)
  - Writing velocities back to global memory
  - Joint solve via global memory reads/writes (colored, after contacts)

Full 3x3 inverse inertia is read from global memory (L1/L2 cached).
"""

from __future__ import annotations

import warp as wp

# Maximum bodies in shared memory.  6 floats per body (vx,vy,vz,wx,wy,wz).
# 128 bodies * 6 * 4 bytes = 3072 bytes — well within per-block limit.
_MAX_SMEM_BODIES = 128

_CONTACT_SOLVE_SNIPPET = r"""
    __shared__ float smem[""" + str(_MAX_SMEM_BODIES * 6) + r"""];

    // Load velocities into shared memory
    for (int b = tid; b < num_bodies; b += blockDim.x) {
        auto v = *wp::address(body_vel, wid, b);
        auto w = *wp::address(body_ang_vel, wid, b);
        float* s = smem + b * 6;
        s[0] = v[0]; s[1] = v[1]; s[2] = v[2];
        s[3] = w[0]; s[4] = w[1]; s[5] = w[2];
    }
    __syncthreads();

    int total_c = *wp::address(color_offsets, wid, max_colors);

    // ═══ Warm start (first substep only) ═══
    if (do_warm_start > 0) {
        for (int ci = tid; ci < total_c; ci += blockDim.x) {
            float ni = *wp::address(c_ni, wid, ci);
            float fi1 = *wp::address(c_fi1, wid, ci);
            float fi2 = *wp::address(c_fi2, wid, ci);
            if (ni == 0.f && fi1 == 0.f && fi2 == 0.f) continue;

            auto n = *wp::address(c_normal, wid, ci);
            auto t1 = *wp::address(c_tangent1, wid, ci);
            auto t2 = *wp::address(c_tangent2, wid, ci);
            float px = ni*n[0] + fi1*t1[0] + fi2*t2[0];
            float py = ni*n[1] + fi1*t1[1] + fi2*t2[1];
            float pz = ni*n[2] + fi1*t1[2] + fi2*t2[2];

            int a = *wp::address(c_body_a, wid, ci);
            if (a >= 0) {
                float ima = *wp::address(inv_m, wid, a);
                auto iia = *wp::address(inv_I, wid, a);
                auto ra = *wp::address(c_r_a, wid, ci);
                float* sa = smem + a * 6;
                sa[0] -= ima*px; sa[1] -= ima*py; sa[2] -= ima*pz;
                // angular: inv_I * cross(r, P)
                float cx = ra[1]*pz - ra[2]*py;
                float cy = ra[2]*px - ra[0]*pz;
                float cz = ra[0]*py - ra[1]*px;
                sa[3] -= iia.data[0][0]*cx + iia.data[0][1]*cy + iia.data[0][2]*cz;
                sa[4] -= iia.data[1][0]*cx + iia.data[1][1]*cy + iia.data[1][2]*cz;
                sa[5] -= iia.data[2][0]*cx + iia.data[2][1]*cy + iia.data[2][2]*cz;
            }
            int bi = *wp::address(c_body_b, wid, ci);
            if (bi >= 0) {
                float imb = *wp::address(inv_m, wid, bi);
                auto iib = *wp::address(inv_I, wid, bi);
                auto rb = *wp::address(c_r_b, wid, ci);
                float* sb = smem + bi * 6;
                sb[0] += imb*px; sb[1] += imb*py; sb[2] += imb*pz;
                float cx = rb[1]*pz - rb[2]*py;
                float cy = rb[2]*px - rb[0]*pz;
                float cz = rb[0]*py - rb[1]*px;
                sb[3] += iib.data[0][0]*cx + iib.data[0][1]*cy + iib.data[0][2]*cz;
                sb[4] += iib.data[1][0]*cx + iib.data[1][1]*cy + iib.data[1][2]*cz;
                sb[5] += iib.data[2][0]*cx + iib.data[2][1]*cy + iib.data[2][2]*cz;
            }
            *wp::address(c_tni, wid, ci) = ni;
        }
        __syncthreads();
    }

    // ═══ Colored contact solve ═══
    for (int color = 0; color < max_colors; color++) {
        int cstart = *wp::address(color_offsets, wid, color);
        int cend = *wp::address(color_offsets, wid, color + 1);
        if (tid < cend - cstart) {
            int ci = cstart + tid;
            int a = *wp::address(c_body_a, wid, ci);
            int bi = *wp::address(c_body_b, wid, ci);
            auto ra = *wp::address(c_r_a, wid, ci);
            auto rb = *wp::address(c_r_b, wid, ci);
            auto n = *wp::address(c_normal, wid, ci);
            float nMass = *wp::address(c_nmass, wid, ci);
            float friction = *wp::address(c_fric, wid, ci);

            float ima=0,imb=0;
            float vax=0,vay=0,vaz=0,wax=0,way=0,waz=0;
            float vbx=0,vby=0,vbz=0,wbx=0,wby=0,wbz=0;

            if (a >= 0) { ima = *wp::address(inv_m,wid,a);
                float* sa = smem+a*6; vax=sa[0];vay=sa[1];vaz=sa[2];wax=sa[3];way=sa[4];waz=sa[5]; }
            if (bi >= 0) { imb = *wp::address(inv_m,wid,bi);
                float* sb = smem+bi*6; vbx=sb[0];vby=sb[1];vbz=sb[2];wbx=sb[3];wby=sb[4];wbz=sb[5]; }

            // Relative velocity
            float vrx = (vbx + wby*rb[2]-wbz*rb[1]) - (vax + way*ra[2]-waz*ra[1]);
            float vry = (vby + wbz*rb[0]-wbx*rb[2]) - (vay + waz*ra[0]-wax*ra[2]);
            float vrz = (vbz + wbx*rb[1]-wby*rb[0]) - (vaz + wax*ra[1]-way*ra[0]);
            float vn = vrx*n[0] + vry*n[1] + vrz*n[2];

            // Bias
            float velocityBias = 0.f, ms = 1.f, isv = 0.f;
            if (use_bias > 0) {
                float base_sep = *wp::address(c_sep, wid, ci);
                float dpax=0,dpay=0,dpaz=0,dpbx=0,dpby=0,dpbz=0;
                if (a >= 0) { auto dp = *wp::address(delta_pos,wid,a); dpax=dp[0];dpay=dp[1];dpaz=dp[2]; }
                if (bi >= 0) { auto dp = *wp::address(delta_pos,wid,bi); dpbx=dp[0];dpby=dp[1];dpbz=dp[2]; }
                float sep = base_sep + (dpbx-dpax)*n[0] + (dpby-dpay)*n[1] + (dpbz-dpaz)*n[2];

                int is_s = *wp::address(c_is_static, wid, ci);
                float br_val = is_s ? static_br : bias_rate;
                float ms_val = is_s ? static_ms : mass_scale;
                float is_val = is_s ? static_is : impulse_scale;
                if (sep > 0.f) {
                    velocityBias = sep * inv_sub_dt;
                } else {
                    velocityBias = fmaxf(ms_val * br_val * sep, -contact_speed);
                    ms = ms_val;
                    isv = is_val;
                }
            }

            // Normal impulse
            float lambda_n = *wp::address(c_ni, wid, ci);
            float imp = -nMass * (ms*vn + velocityBias) - isv * lambda_n;
            float newL = fmaxf(lambda_n + imp, 0.f);
            imp = newL - lambda_n;
            *wp::address(c_ni, wid, ci) = newL;
            *wp::address(c_tni, wid, ci) = *wp::address(c_tni, wid, ci) + imp;

            // Apply normal impulse
            float px = imp*n[0], py = imp*n[1], pz = imp*n[2];
            if (a >= 0) {
                auto iia = *wp::address(inv_I, wid, a);
                float* sa = smem+a*6;
                sa[0] -= ima*px; sa[1] -= ima*py; sa[2] -= ima*pz;
                float cx=ra[1]*pz-ra[2]*py, cy=ra[2]*px-ra[0]*pz, cz=ra[0]*py-ra[1]*px;
                sa[3] -= iia.data[0][0]*cx+iia.data[0][1]*cy+iia.data[0][2]*cz;
                sa[4] -= iia.data[1][0]*cx+iia.data[1][1]*cy+iia.data[1][2]*cz;
                sa[5] -= iia.data[2][0]*cx+iia.data[2][1]*cy+iia.data[2][2]*cz;
            }
            if (bi >= 0) {
                auto iib = *wp::address(inv_I, wid, bi);
                float* sb = smem+bi*6;
                sb[0] += imb*px; sb[1] += imb*py; sb[2] += imb*pz;
                float cx=rb[1]*pz-rb[2]*py, cy=rb[2]*px-rb[0]*pz, cz=rb[0]*py-rb[1]*px;
                sb[3] += iib.data[0][0]*cx+iib.data[0][1]*cy+iib.data[0][2]*cz;
                sb[4] += iib.data[1][0]*cx+iib.data[1][1]*cy+iib.data[1][2]*cz;
                sb[5] += iib.data[2][0]*cx+iib.data[2][1]*cy+iib.data[2][2]*cz;
            }

            // ═══ Friction (2 tangent directions) ═══
            float maxF = friction * newL;
            auto t1 = *wp::address(c_tangent1, wid, ci);
            auto t2 = *wp::address(c_tangent2, wid, ci);
            float t1M = *wp::address(c_t1m, wid, ci);
            float t2M = *wp::address(c_t2m, wid, ci);

            // Re-read velocity after normal impulse
            if (a >= 0) { float* sa=smem+a*6; vax=sa[0];vay=sa[1];vaz=sa[2];wax=sa[3];way=sa[4];waz=sa[5]; }
            if (bi >= 0) { float* sb=smem+bi*6; vbx=sb[0];vby=sb[1];vbz=sb[2];wbx=sb[3];wby=sb[4];wbz=sb[5]; }
            vrx=(vbx+wby*rb[2]-wbz*rb[1])-(vax+way*ra[2]-waz*ra[1]);
            vry=(vby+wbz*rb[0]-wbx*rb[2])-(vay+waz*ra[0]-wax*ra[2]);
            vrz=(vbz+wbx*rb[1]-wby*rb[0])-(vaz+wax*ra[1]-way*ra[0]);

            // Tangent 1
            { float vt = vrx*t1[0]+vry*t1[1]+vrz*t1[2];
              float lam = *wp::address(c_fi1,wid,ci);
              float i1 = -t1M*vt;
              float nL = fminf(fmaxf(lam+i1,-maxF),maxF); i1=nL-lam;
              *wp::address(c_fi1,wid,ci) = nL;
              float p1x=i1*t1[0],p1y=i1*t1[1],p1z=i1*t1[2];
              if (a>=0) { auto iia=*wp::address(inv_I,wid,a); float*sa=smem+a*6;
                sa[0]-=ima*p1x;sa[1]-=ima*p1y;sa[2]-=ima*p1z;
                float cx=ra[1]*p1z-ra[2]*p1y,cy=ra[2]*p1x-ra[0]*p1z,cz=ra[0]*p1y-ra[1]*p1x;
                sa[3]-=iia.data[0][0]*cx+iia.data[0][1]*cy+iia.data[0][2]*cz;
                sa[4]-=iia.data[1][0]*cx+iia.data[1][1]*cy+iia.data[1][2]*cz;
                sa[5]-=iia.data[2][0]*cx+iia.data[2][1]*cy+iia.data[2][2]*cz; }
              if (bi>=0) { auto iib=*wp::address(inv_I,wid,bi); float*sb=smem+bi*6;
                sb[0]+=imb*p1x;sb[1]+=imb*p1y;sb[2]+=imb*p1z;
                float cx=rb[1]*p1z-rb[2]*p1y,cy=rb[2]*p1x-rb[0]*p1z,cz=rb[0]*p1y-rb[1]*p1x;
                sb[3]+=iib.data[0][0]*cx+iib.data[0][1]*cy+iib.data[0][2]*cz;
                sb[4]+=iib.data[1][0]*cx+iib.data[1][1]*cy+iib.data[1][2]*cz;
                sb[5]+=iib.data[2][0]*cx+iib.data[2][1]*cy+iib.data[2][2]*cz; } }

            // Re-read for tangent 2
            if (a>=0) { float*sa=smem+a*6; vax=sa[0];vay=sa[1];vaz=sa[2];wax=sa[3];way=sa[4];waz=sa[5]; }
            if (bi>=0) { float*sb=smem+bi*6; vbx=sb[0];vby=sb[1];vbz=sb[2];wbx=sb[3];wby=sb[4];wbz=sb[5]; }
            vrx=(vbx+wby*rb[2]-wbz*rb[1])-(vax+way*ra[2]-waz*ra[1]);
            vry=(vby+wbz*rb[0]-wbx*rb[2])-(vay+waz*ra[0]-wax*ra[2]);
            vrz=(vbz+wbx*rb[1]-wby*rb[0])-(vaz+wax*ra[1]-way*ra[0]);

            // Tangent 2
            { float vt = vrx*t2[0]+vry*t2[1]+vrz*t2[2];
              float lam = *wp::address(c_fi2,wid,ci);
              float i2 = -t2M*vt;
              float nL = fminf(fmaxf(lam+i2,-maxF),maxF); i2=nL-lam;
              *wp::address(c_fi2,wid,ci) = nL;
              float p2x=i2*t2[0],p2y=i2*t2[1],p2z=i2*t2[2];
              if (a>=0) { auto iia=*wp::address(inv_I,wid,a); float*sa=smem+a*6;
                sa[0]-=ima*p2x;sa[1]-=ima*p2y;sa[2]-=ima*p2z;
                float cx=ra[1]*p2z-ra[2]*p2y,cy=ra[2]*p2x-ra[0]*p2z,cz=ra[0]*p2y-ra[1]*p2x;
                sa[3]-=iia.data[0][0]*cx+iia.data[0][1]*cy+iia.data[0][2]*cz;
                sa[4]-=iia.data[1][0]*cx+iia.data[1][1]*cy+iia.data[1][2]*cz;
                sa[5]-=iia.data[2][0]*cx+iia.data[2][1]*cy+iia.data[2][2]*cz; }
              if (bi>=0) { auto iib=*wp::address(inv_I,wid,bi); float*sb=smem+bi*6;
                sb[0]+=imb*p2x;sb[1]+=imb*p2y;sb[2]+=imb*p2z;
                float cx=rb[1]*p2z-rb[2]*p2y,cy=rb[2]*p2x-rb[0]*p2z,cz=rb[0]*p2y-rb[1]*p2x;
                sb[3]+=iib.data[0][0]*cx+iib.data[0][1]*cy+iib.data[0][2]*cz;
                sb[4]+=iib.data[1][0]*cx+iib.data[1][1]*cy+iib.data[1][2]*cz;
                sb[5]+=iib.data[2][0]*cx+iib.data[2][1]*cy+iib.data[2][2]*cz; } }
        }
        __syncthreads();
    }

    // ═══ Restitution (last substep only) ═══
    if (do_restitution > 0) {
        for (int ci = tid; ci < total_c; ci += blockDim.x) {
            float rest = *wp::address(c_rest, wid, ci);
            if (rest == 0.f) continue;
            float rv = *wp::address(c_rvel, wid, ci);
            float tni = *wp::address(c_tni, wid, ci);
            if (rv > -rest_thresh || tni == 0.f) continue;

            int a = *wp::address(c_body_a, wid, ci);
            int bi = *wp::address(c_body_b, wid, ci);
            auto ra = *wp::address(c_r_a, wid, ci);
            auto rb = *wp::address(c_r_b, wid, ci);
            auto n = *wp::address(c_normal, wid, ci);
            float nM = *wp::address(c_nmass, wid, ci);

            float ima=0,imb=0;
            float vax=0,vay=0,vaz=0,wax=0,way=0,waz=0;
            float vbx=0,vby=0,vbz=0,wbx=0,wby=0,wbz=0;
            if (a>=0) { ima=*wp::address(inv_m,wid,a);
                float*sa=smem+a*6; vax=sa[0];vay=sa[1];vaz=sa[2];wax=sa[3];way=sa[4];waz=sa[5]; }
            if (bi>=0) { imb=*wp::address(inv_m,wid,bi);
                float*sb=smem+bi*6; vbx=sb[0];vby=sb[1];vbz=sb[2];wbx=sb[3];wby=sb[4];wbz=sb[5]; }

            float vrx=(vbx+wby*rb[2]-wbz*rb[1])-(vax+way*ra[2]-waz*ra[1]);
            float vry=(vby+wbz*rb[0]-wbx*rb[2])-(vay+waz*ra[0]-wax*ra[2]);
            float vrz=(vbz+wbx*rb[1]-wby*rb[0])-(vaz+wax*ra[1]-way*ra[0]);
            float vn = vrx*n[0]+vry*n[1]+vrz*n[2];

            float ln = *wp::address(c_ni, wid, ci);
            float imp = -nM*(vn + rest*rv);
            float nL = fmaxf(ln+imp, 0.f); imp = nL-ln;
            *wp::address(c_ni, wid, ci) = nL;
            *wp::address(c_tni, wid, ci) = *wp::address(c_tni, wid, ci) + imp;

            float px=imp*n[0],py=imp*n[1],pz=imp*n[2];
            if (a>=0) { auto iia=*wp::address(inv_I,wid,a); float*sa=smem+a*6;
                sa[0]-=ima*px;sa[1]-=ima*py;sa[2]-=ima*pz;
                float cx=ra[1]*pz-ra[2]*py,cy=ra[2]*px-ra[0]*pz,cz=ra[0]*py-ra[1]*px;
                sa[3]-=iia.data[0][0]*cx+iia.data[0][1]*cy+iia.data[0][2]*cz;
                sa[4]-=iia.data[1][0]*cx+iia.data[1][1]*cy+iia.data[1][2]*cz;
                sa[5]-=iia.data[2][0]*cx+iia.data[2][1]*cy+iia.data[2][2]*cz; }
            if (bi>=0) { auto iib=*wp::address(inv_I,wid,bi); float*sb=smem+bi*6;
                sb[0]+=imb*px;sb[1]+=imb*py;sb[2]+=imb*pz;
                float cx=rb[1]*pz-rb[2]*py,cy=rb[2]*px-rb[0]*pz,cz=rb[0]*py-rb[1]*px;
                sb[3]+=iib.data[0][0]*cx+iib.data[0][1]*cy+iib.data[0][2]*cz;
                sb[4]+=iib.data[1][0]*cx+iib.data[1][1]*cy+iib.data[1][2]*cz;
                sb[5]+=iib.data[2][0]*cx+iib.data[2][1]*cy+iib.data[2][2]*cz; } }
        __syncthreads();
    }

    // ═══ Store velocities back to global ═══
    for (int b = tid; b < num_bodies; b += blockDim.x) {
        float* s = smem + b * 6;
        wp::vec_t<3,float> v; v[0]=s[0];v[1]=s[1];v[2]=s[2];
        *wp::address(body_vel, wid, b) = v;
        wp::vec_t<3,float> w; w[0]=s[3];w[1]=s[4];w[2]=s[5];
        *wp::address(body_ang_vel, wid, b) = w;
    }
"""


@wp.func_native(snippet=_CONTACT_SOLVE_SNIPPET)
def _contact_solve_native(
    body_vel: wp.array2d(dtype=wp.vec3),
    body_ang_vel: wp.array2d(dtype=wp.vec3),
    inv_m: wp.array2d(dtype=float),
    inv_I: wp.array2d(dtype=wp.mat33),
    delta_pos: wp.array2d(dtype=wp.vec3),
    c_body_a: wp.array2d(dtype=wp.int32),
    c_body_b: wp.array2d(dtype=wp.int32),
    c_normal: wp.array2d(dtype=wp.vec3),
    c_tangent1: wp.array2d(dtype=wp.vec3),
    c_tangent2: wp.array2d(dtype=wp.vec3),
    c_r_a: wp.array2d(dtype=wp.vec3),
    c_r_b: wp.array2d(dtype=wp.vec3),
    c_sep: wp.array2d(dtype=float),
    c_nmass: wp.array2d(dtype=float),
    c_t1m: wp.array2d(dtype=float),
    c_t2m: wp.array2d(dtype=float),
    c_fric: wp.array2d(dtype=float),
    c_rest: wp.array2d(dtype=float),
    c_rvel: wp.array2d(dtype=float),
    c_ni: wp.array2d(dtype=float),
    c_fi1: wp.array2d(dtype=float),
    c_fi2: wp.array2d(dtype=float),
    c_tni: wp.array2d(dtype=float),
    c_is_static: wp.array2d(dtype=wp.int32),
    color_offsets: wp.array2d(dtype=wp.int32),
    num_bodies: int,
    max_colors: int,
    use_bias: int,
    do_warm_start: int,
    do_restitution: int,
    inv_sub_dt: float,
    bias_rate: float,
    mass_scale: float,
    impulse_scale: float,
    static_br: float,
    static_ms: float,
    static_is: float,
    contact_speed: float,
    rest_thresh: float,
    wid: int,
    tid: int,
):
    ...


@wp.kernel
def contact_solve_kernel(
    body_vel: wp.array2d(dtype=wp.vec3),
    body_ang_vel: wp.array2d(dtype=wp.vec3),
    inv_m: wp.array2d(dtype=float),
    inv_I: wp.array2d(dtype=wp.mat33),
    delta_pos: wp.array2d(dtype=wp.vec3),
    c_body_a: wp.array2d(dtype=wp.int32),
    c_body_b: wp.array2d(dtype=wp.int32),
    c_normal: wp.array2d(dtype=wp.vec3),
    c_tangent1: wp.array2d(dtype=wp.vec3),
    c_tangent2: wp.array2d(dtype=wp.vec3),
    c_r_a: wp.array2d(dtype=wp.vec3),
    c_r_b: wp.array2d(dtype=wp.vec3),
    c_sep: wp.array2d(dtype=float),
    c_nmass: wp.array2d(dtype=float),
    c_t1m: wp.array2d(dtype=float),
    c_t2m: wp.array2d(dtype=float),
    c_fric: wp.array2d(dtype=float),
    c_rest: wp.array2d(dtype=float),
    c_rvel: wp.array2d(dtype=float),
    c_ni: wp.array2d(dtype=float),
    c_fi1: wp.array2d(dtype=float),
    c_fi2: wp.array2d(dtype=float),
    c_tni: wp.array2d(dtype=float),
    c_is_static: wp.array2d(dtype=wp.int32),
    color_offsets: wp.array2d(dtype=wp.int32),
    num_bodies: int,
    max_colors: int,
    use_bias: int,
    do_warm_start: int,
    do_restitution: int,
    inv_sub_dt: float,
    bias_rate: float,
    mass_scale: float,
    impulse_scale: float,
    static_br: float,
    static_ms: float,
    static_is: float,
    contact_speed: float,
    rest_thresh: float,
):
    """Fused contact solve: shared-memory colored Gauss-Seidel.

    Launched with ``wp.launch_tiled(dim=[num_worlds], block_dim=block_dim)``.
    """
    wid, tid = wp.tid()
    _contact_solve_native(
        body_vel, body_ang_vel, inv_m, inv_I, delta_pos,
        c_body_a, c_body_b, c_normal, c_tangent1, c_tangent2,
        c_r_a, c_r_b, c_sep, c_nmass, c_t1m, c_t2m,
        c_fric, c_rest, c_rvel,
        c_ni, c_fi1, c_fi2, c_tni, c_is_static,
        color_offsets, num_bodies, max_colors,
        use_bias, do_warm_start, do_restitution,
        inv_sub_dt, bias_rate, mass_scale, impulse_scale,
        static_br, static_ms, static_is,
        contact_speed, rest_thresh,
        wid, tid,
    )

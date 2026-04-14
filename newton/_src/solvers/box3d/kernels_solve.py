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

from .mat3sym import mat3sym

# Maximum bodies in shared memory.  6 floats per body (vx,vy,vz,wx,wy,wz).
# 128 bodies * 6 * 4 bytes = 3072 bytes — well within per-block limit.
_MAX_SMEM_BODIES = 1024

_CONTACT_SOLVE_SNIPPET = r"""
    __shared__ float smem[""" + str(_MAX_SMEM_BODIES * 6) + r"""];

    // Load velocities into shared memory (optionally integrate velocity first)
    for (int b = tid; b < num_bodies; b += blockDim.x) {
        auto v = *wp::address(body_vel, wid, b);
        auto w = *wp::address(body_ang_vel, wid, b);
        if (do_int_vel > 0) {
            float im = *wp::address(inv_m, wid, b);
            if (im > 0.0f) {
                // Box2D-style: gravity delta undamped, velocity damped
                float ld = 1.0f / (1.0f + sub_dt * lin_damp);
                float ad = 1.0f / (1.0f + sub_dt * ang_damp);
                v[0] = gx*sub_dt + v[0]*ld; v[1] = gy*sub_dt + v[1]*ld; v[2] = gz*sub_dt + v[2]*ld;
                w[0] *= ad; w[1] *= ad; w[2] *= ad;
            }
        }
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
                sa[3] -= iia.m00*cx + iia.m01*cy + iia.m02*cz;
                sa[4] -= iia.m01*cx + iia.m11*cy + iia.m12*cz;
                sa[5] -= iia.m02*cx + iia.m12*cy + iia.m22*cz;
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
                sb[3] += iib.m00*cx + iib.m01*cy + iib.m02*cz;
                sb[4] += iib.m01*cx + iib.m11*cy + iib.m12*cz;
                sb[5] += iib.m02*cx + iib.m12*cy + iib.m22*cz;
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

            // Bias — following Box2D: speculative bias always, soft bias only when use_bias
            float velocityBias = 0.f, ms = 1.f, isv = 0.f;
            {
                float base_sep = *wp::address(c_sep, wid, ci);
                float dpax=0,dpay=0,dpaz=0,dpbx=0,dpby=0,dpbz=0;
                if (a >= 0) { auto dp = *wp::address(delta_pos,wid,a); dpax=dp[0];dpay=dp[1];dpaz=dp[2]; }
                if (bi >= 0) { auto dp = *wp::address(delta_pos,wid,bi); dpbx=dp[0];dpby=dp[1];dpbz=dp[2]; }
                float sep = base_sep + (dpbx-dpax)*n[0] + (dpby-dpay)*n[1] + (dpbz-dpaz)*n[2];

                if (sep > 0.005f) {
                    // Speculative: always active (even during relaxation).
                    // Small slop (5mm) prevents jitter from floating-point separation noise.
                    velocityBias = sep * inv_sub_dt;
                } else if (use_bias > 0) {
                    // Soft position correction: only during biased pass
                    int is_s = *wp::address(c_is_static, wid, ci);
                    float br_val = is_s ? static_br : bias_rate;
                    float ms_val = is_s ? static_ms : mass_scale;
                    float is_val = is_s ? static_is : impulse_scale;
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
                sa[3] -= iia.m00*cx+iia.m01*cy+iia.m02*cz;
                sa[4] -= iia.m01*cx+iia.m11*cy+iia.m12*cz;
                sa[5] -= iia.m02*cx+iia.m12*cy+iia.m22*cz;
            }
            if (bi >= 0) {
                auto iib = *wp::address(inv_I, wid, bi);
                float* sb = smem+bi*6;
                sb[0] += imb*px; sb[1] += imb*py; sb[2] += imb*pz;
                float cx=rb[1]*pz-rb[2]*py, cy=rb[2]*px-rb[0]*pz, cz=rb[0]*py-rb[1]*px;
                sb[3] += iib.m00*cx+iib.m01*cy+iib.m02*cz;
                sb[4] += iib.m01*cx+iib.m11*cy+iib.m12*cz;
                sb[5] += iib.m02*cx+iib.m12*cy+iib.m22*cz;
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
                sa[3]-=iia.m00*cx+iia.m01*cy+iia.m02*cz;
                sa[4]-=iia.m01*cx+iia.m11*cy+iia.m12*cz;
                sa[5]-=iia.m02*cx+iia.m12*cy+iia.m22*cz; }
              if (bi>=0) { auto iib=*wp::address(inv_I,wid,bi); float*sb=smem+bi*6;
                sb[0]+=imb*p1x;sb[1]+=imb*p1y;sb[2]+=imb*p1z;
                float cx=rb[1]*p1z-rb[2]*p1y,cy=rb[2]*p1x-rb[0]*p1z,cz=rb[0]*p1y-rb[1]*p1x;
                sb[3]+=iib.m00*cx+iib.m01*cy+iib.m02*cz;
                sb[4]+=iib.m01*cx+iib.m11*cy+iib.m12*cz;
                sb[5]+=iib.m02*cx+iib.m12*cy+iib.m22*cz; } }

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
                sa[3]-=iia.m00*cx+iia.m01*cy+iia.m02*cz;
                sa[4]-=iia.m01*cx+iia.m11*cy+iia.m12*cz;
                sa[5]-=iia.m02*cx+iia.m12*cy+iia.m22*cz; }
              if (bi>=0) { auto iib=*wp::address(inv_I,wid,bi); float*sb=smem+bi*6;
                sb[0]+=imb*p2x;sb[1]+=imb*p2y;sb[2]+=imb*p2z;
                float cx=rb[1]*p2z-rb[2]*p2y,cy=rb[2]*p2x-rb[0]*p2z,cz=rb[0]*p2y-rb[1]*p2x;
                sb[3]+=iib.m00*cx+iib.m01*cy+iib.m02*cz;
                sb[4]+=iib.m01*cx+iib.m11*cy+iib.m12*cz;
                sb[5]+=iib.m02*cx+iib.m12*cy+iib.m22*cz; } }
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
                sa[3]-=iia.m00*cx+iia.m01*cy+iia.m02*cz;
                sa[4]-=iia.m01*cx+iia.m11*cy+iia.m12*cz;
                sa[5]-=iia.m02*cx+iia.m12*cy+iia.m22*cz; }
            if (bi>=0) { auto iib=*wp::address(inv_I,wid,bi); float*sb=smem+bi*6;
                sb[0]+=imb*px;sb[1]+=imb*py;sb[2]+=imb*pz;
                float cx=rb[1]*pz-rb[2]*py,cy=rb[2]*px-rb[0]*pz,cz=rb[0]*py-rb[1]*px;
                sb[3]+=iib.m00*cx+iib.m01*cy+iib.m02*cz;
                sb[4]+=iib.m01*cx+iib.m11*cy+iib.m12*cz;
                sb[5]+=iib.m02*cx+iib.m12*cy+iib.m22*cz; } }
        __syncthreads();
    }

    // ═══ Store velocities shared → global ═══
    for (int b = tid; b < num_bodies; b += blockDim.x) {
        float* s = smem + b * 6;
        wp::vec_t<3,float> v; v[0]=s[0];v[1]=s[1];v[2]=s[2];
        *wp::address(body_vel, wid, b) = v;
        wp::vec_t<3,float> w; w[0]=s[3];w[1]=s[4];w[2]=s[5];
        *wp::address(body_ang_vel, wid, b) = w;
    }
    __syncthreads();

    // ═══ Joint solve (colored, global memory) ═══
    #define QROT(qx,qy,qz,qw, vx,vy,vz, rx,ry,rz) { \
        float _cx=(qy)*(vz)-(qz)*(vy); float _cy=(qz)*(vx)-(qx)*(vz); float _cz=(qx)*(vy)-(qy)*(vx); \
        (rx)=(vx)+2.f*((qw)*_cx+(qy)*_cz-(qz)*_cy); \
        (ry)=(vy)+2.f*((qw)*_cy+(qz)*_cx-(qx)*_cz); \
        (rz)=(vz)+2.f*((qw)*_cz+(qx)*_cy-(qy)*_cx); }

    if (num_joints > 0) {
        for (int jc = 0; jc < max_jcolors; jc++) {
            int jstart = *wp::address(jc_off, wid, jc);
            int jend   = *wp::address(jc_off, wid, jc + 1);
            if (tid < jend - jstart) {
                int ji = jstart + tid;
                int ja = *wp::address(j_ba, wid, ji);
                int jbi = *wp::address(j_bb, wid, ji);
                int jtype = *wp::address(j_type, wid, ji);

                float jima=0,jimb=0;
                float jvax=0,jvay=0,jvaz=0,jwax=0,jway=0,jwaz=0;
                float jvbx=0,jvby=0,jvbz=0,jwbx=0,jwby=0,jwbz=0;
                float qax=0,qay=0,qaz=0,qaw=1;
                float qbx=0,qby=0,qbz=0,qbw=1;
                // Read a dummy mat3sym to get the correct type, then zero it
                auto jiia = *wp::address(inv_I, wid, 0);
                jiia.m00=0;jiia.m01=0;jiia.m02=0;jiia.m11=0;jiia.m12=0;jiia.m22=0;
                auto jiib = jiia;

                if (ja >= 0) {
                    jima = *wp::address(inv_m, wid, ja);
                    jiia = *wp::address(inv_I, wid, ja);
                    auto v = *wp::address(body_vel, wid, ja); jvax=v[0]; jvay=v[1]; jvaz=v[2];
                    auto w = *wp::address(body_ang_vel, wid, ja); jwax=w[0]; jway=w[1]; jwaz=w[2];
                    auto q = *wp::address(body_ori, wid, ja); qax=q[0]; qay=q[1]; qaz=q[2]; qaw=q[3];
                }
                if (jbi >= 0) {
                    jimb = *wp::address(inv_m, wid, jbi);
                    jiib = *wp::address(inv_I, wid, jbi);
                    auto v = *wp::address(body_vel, wid, jbi); jvbx=v[0]; jvby=v[1]; jvbz=v[2];
                    auto w = *wp::address(body_ang_vel, wid, jbi); jwbx=w[0]; jwby=w[1]; jwbz=w[2];
                    auto q = *wp::address(body_ori, wid, jbi); qbx=q[0]; qby=q[1]; qbz=q[2]; qbw=q[3];
                }

                // Compute world-space anchor offsets
                auto la = *wp::address(j_la, wid, ji);
                float jrax, jray, jraz;
                QROT(qax,qay,qaz,qaw, la[0],la[1],la[2], jrax,jray,jraz);
                auto lb = *wp::address(j_lb, wid, ji);
                float jrbx, jrby, jrbz;
                QROT(qbx,qby,qbz,qbw, lb[0],lb[1],lb[2], jrbx,jrby,jrbz);

                // ── Point-to-point (3x3 K solve) ──
                float cdx = (jvbx+jwby*jrbz-jwbz*jrby) - (jvax+jway*jraz-jwaz*jray);
                float cdy = (jvby+jwbz*jrbx-jwbx*jrbz) - (jvay+jwaz*jrax-jwax*jraz);
                float cdz = (jvbz+jwbx*jrby-jwby*jrbx) - (jvaz+jwax*jray-jway*jrax);

                float jbx=0, jby=0, jbz=0, jms=1.f, jisv=0.f;
                if (use_bias > 0) {
                    auto pA = *wp::address(body_pos, wid, ja >= 0 ? ja : 0);
                    auto pB = *wp::address(body_pos, wid, jbi >= 0 ? jbi : 0);
                    float sx = (ja >= 0 ? pA[0] : 0.f) + jrax;
                    float sy = (ja >= 0 ? pA[1] : 0.f) + jray;
                    float sz = (ja >= 0 ? pA[2] : 0.f) + jraz;
                    float ex = (jbi >= 0 ? pB[0] : 0.f) + jrbx;
                    float ey = (jbi >= 0 ? pB[1] : 0.f) + jrby;
                    float ez = (jbi >= 0 ? pB[2] : 0.f) + jrbz;
                    jbx = jbias_rate * (ex - sx);
                    jby = jbias_rate * (ey - sy);
                    jbz = jbias_rate * (ez - sz);
                    jms = jmass_scale; jisv = jimpulse_scale;
                }

                // Build K = (mA+mB)*I + [rA]_x^T @ IA^-1 @ [rA]_x + [rB]_x^T @ IB^-1 @ [rB]_x
                // Following Box2D's approach: K_rot[i][j] = dot(cross(r, e_i), I^-1 * cross(r, e_j))
                // Compute via: s_j = I^-1 @ [r]_x column j, then K_rot[i][j] = dot([r]_x^T row i, s_j)
                // [r]_x cols: c0=(0,rz,-ry), c1=(-rz,0,rx), c2=(ry,-rx,0)
                // [r]_x^T rows (= -[r]_x rows): r0=(0,rz,-ry), r1=(-rz,0,rx), r2=(ry,-rx,0)

                float d = jima + jimb;
                float K00 = d, K01 = 0, K02 = 0, K11 = d, K12 = 0, K22 = d;

                // Macro to add [r]_x^T @ I^-1 @ [r]_x contribution for one body
                #define ADD_SKEW_K(rx,ry,rz,II) { \
                    /* s0 = II @ (0, rz, -ry) — mat3sym: symmetric, so [1][0]=m01 etc */ \
                    float s0x = II.m01*rz - II.m02*ry; \
                    float s0y = II.m11*rz - II.m12*ry; \
                    float s0z = II.m12*rz - II.m22*ry; \
                    /* s1 = II @ (-rz, 0, rx) */ \
                    float s1x = -II.m00*rz + II.m02*rx; \
                    float s1y = -II.m01*rz + II.m12*rx; \
                    float s1z = -II.m02*rz + II.m22*rx; \
                    /* s2 = II @ (ry, -rx, 0) */ \
                    float s2x = II.m00*ry - II.m01*rx; \
                    float s2y = II.m01*ry - II.m11*rx; \
                    float s2z = II.m02*ry - II.m12*rx; \
                    /* [r]_x^T rows: r0=(0,rz,-ry), r1=(-rz,0,rx), r2=(ry,-rx,0) */ \
                    K00 += rz*s0y - ry*s0z; \
                    K01 += rz*s1y - ry*s1z; \
                    K02 += rz*s2y - ry*s2z; \
                    K11 += -rz*s1x + rx*s1z; \
                    K12 += -rz*s2x + rx*s2z; \
                    K22 += ry*s2x - rx*s2y; \
                }
                if (ja >= 0) { ADD_SKEW_K(jrax, jray, jraz, jiia) }
                if (jbi >= 0) { ADD_SKEW_K(jrbx, jrby, jrbz, jiib) }
                #undef ADD_SKEW_K

                float rx = cdx+jbx, ry = cdy+jby, rz = cdz+jbz;
                float det = K00*(K11*K22-K12*K12) - K01*(K01*K22-K12*K02) + K02*(K01*K12-K11*K02);
                if (fabsf(det) > 1e-12f) {
                    float id = 1.f/det;
                    float I00=(K11*K22-K12*K12)*id, I01=(K02*K12-K01*K22)*id, I02=(K01*K12-K02*K11)*id;
                    float I11=(K00*K22-K02*K02)*id, I12=(K01*K02-K00*K12)*id;
                    float I22=(K00*K11-K01*K01)*id;
                    float svx=I00*rx+I01*ry+I02*rz, svy=I01*rx+I11*ry+I12*rz, svz=I02*rx+I12*ry+I22*rz;

                    auto li = *wp::address(j_li, wid, ji);
                    float ix = -jms*svx - jisv*li[0];
                    float iy = -jms*svy - jisv*li[1];
                    float iz = -jms*svz - jisv*li[2];
                    wp::vec_t<3,float> nli; nli[0]=li[0]+ix; nli[1]=li[1]+iy; nli[2]=li[2]+iz;
                    *wp::address(j_li, wid, ji) = nli;

                    // Apply linear impulse
                    if (ja >= 0) {
                        jvax -= jima*ix; jvay -= jima*iy; jvaz -= jima*iz;
                        float cx=jray*iz-jraz*iy, cy=jraz*ix-jrax*iz, cz=jrax*iy-jray*ix;
                        jwax -= jiia.m00*cx+jiia.m01*cy+jiia.m02*cz;
                        jway -= jiia.m01*cx+jiia.m11*cy+jiia.m12*cz;
                        jwaz -= jiia.m02*cx+jiia.m12*cy+jiia.m22*cz;
                    }
                    if (jbi >= 0) {
                        jvbx += jimb*ix; jvby += jimb*iy; jvbz += jimb*iz;
                        float cx=jrby*iz-jrbz*iy, cy=jrbz*ix-jrbx*iz, cz=jrbx*iy-jrby*ix;
                        jwbx += jiib.m00*cx+jiib.m01*cy+jiib.m02*cz;
                        jwby += jiib.m01*cx+jiib.m11*cy+jiib.m12*cz;
                        jwbz += jiib.m02*cx+jiib.m12*cy+jiib.m22*cz;
                    }
                }

                // ── Angular constraint (REVOLUTE: 2 DOF ⊥ hinge, FIXED: 3 DOF) ──
                if (jtype == 1 || jtype == 3) {  // REVOLUTE=1 or FIXED=3
                    auto ha = *wp::address(j_ha, wid, ji);
                    float whx, why, whz;
                    // Rotate hinge axis to world frame (use child body orientation)
                    QROT(qbx,qby,qbz,qbw, ha[0],ha[1],ha[2], whx,why,whz);

                    auto ai = *wp::address(j_ai, wid, ji);
                    float dwx = jwbx-jwax, dwy = jwby-jway, dwz = jwbz-jwaz;

                    // Combined angular inertia (from mat3sym — symmetric, so only 6 values)
                    float ka00 = (ja>=0 ? jiia.m00 : 0) + (jbi>=0 ? jiib.m00 : 0);
                    float ka01 = (ja>=0 ? jiia.m01 : 0) + (jbi>=0 ? jiib.m01 : 0);
                    float ka02 = (ja>=0 ? jiia.m02 : 0) + (jbi>=0 ? jiib.m02 : 0);
                    float ka11 = (ja>=0 ? jiia.m11 : 0) + (jbi>=0 ? jiib.m11 : 0);
                    float ka12 = (ja>=0 ? jiia.m12 : 0) + (jbi>=0 ? jiib.m12 : 0);
                    float ka22 = (ja>=0 ? jiia.m22 : 0) + (jbi>=0 ? jiib.m22 : 0);
                    // Build as mat_t for existing code that uses .data[][]
                    wp::mat_t<3,3,float> k_ang;
                    k_ang.data[0][0]=ka00; k_ang.data[0][1]=ka01; k_ang.data[0][2]=ka02;
                    k_ang.data[1][0]=ka01; k_ang.data[1][1]=ka11; k_ang.data[1][2]=ka12;
                    k_ang.data[2][0]=ka02; k_ang.data[2][1]=ka12; k_ang.data[2][2]=ka22;

                    if (jtype == 1) {
                        // REVOLUTE: constrain 2 DOF perpendicular to hinge
                        float b1x,b1y,b1z;
                        if (fabsf(why) < 0.9f) { b1x=whz; b1y=0; b1z=-whx; }
                        else { b1x=0; b1y=-whz; b1z=why; }
                        float bl = sqrtf(b1x*b1x+b1y*b1y+b1z*b1z);
                        if (bl > 1e-8f) { float iv=1.f/bl; b1x*=iv; b1y*=iv; b1z*=iv; }
                        float b2x=why*b1z-whz*b1y, b2y=whz*b1x-whx*b1z, b2z=whx*b1y-why*b1x;

                        // Effective angular mass along each basis
                        float kb1x=k_ang.data[0][0]*b1x+k_ang.data[0][1]*b1y+k_ang.data[0][2]*b1z;
                        float kb1y=k_ang.data[1][0]*b1x+k_ang.data[1][1]*b1y+k_ang.data[1][2]*b1z;
                        float kb1z=k_ang.data[2][0]*b1x+k_ang.data[2][1]*b1y+k_ang.data[2][2]*b1z;
                        float k1 = b1x*kb1x+b1y*kb1y+b1z*kb1z;
                        float kb2x=k_ang.data[0][0]*b2x+k_ang.data[0][1]*b2y+k_ang.data[0][2]*b2z;
                        float kb2y=k_ang.data[1][0]*b2x+k_ang.data[1][1]*b2y+k_ang.data[1][2]*b2z;
                        float kb2z=k_ang.data[2][0]*b2x+k_ang.data[2][1]*b2y+k_ang.data[2][2]*b2z;
                        float k2 = b2x*kb2x+b2y*kb2y+b2z*kb2z;

                        float am1 = k1 > 0 ? 1.f/k1 : 0;
                        float am2 = k2 > 0 ? 1.f/k2 : 0;

                        float i1 = -jms*am1*(dwx*b1x+dwy*b1y+dwz*b1z) - jisv*ai[0];
                        float i2 = -jms*am2*(dwx*b2x+dwy*b2y+dwz*b2z) - jisv*ai[1];
                        float apx=i1*b1x+i2*b2x, apy=i1*b1y+i2*b2y, apz=i1*b1z+i2*b2z;

                        if (ja >= 0) {
                            jwax -= jiia.m00*apx+jiia.m01*apy+jiia.m02*apz;
                            jway -= jiia.m01*apx+jiia.m11*apy+jiia.m12*apz;
                            jwaz -= jiia.m02*apx+jiia.m12*apy+jiia.m22*apz;
                        }
                        if (jbi >= 0) {
                            jwbx += jiib.m00*apx+jiib.m01*apy+jiib.m02*apz;
                            jwby += jiib.m01*apx+jiib.m11*apy+jiib.m12*apz;
                            jwbz += jiib.m02*apx+jiib.m12*apy+jiib.m22*apz;
                        }

                        // Motor on hinge axis
                        float mspd = *wp::address(j_ms_arr, wid, ji);
                        float mmt = *wp::address(j_mmt, wid, ji);
                        float mi_val = ai[2];
                        if (mmt > 0) {
                            float kh_x=k_ang.data[0][0]*whx+k_ang.data[0][1]*why+k_ang.data[0][2]*whz;
                            float kh_y=k_ang.data[1][0]*whx+k_ang.data[1][1]*why+k_ang.data[1][2]*whz;
                            float kh_z=k_ang.data[2][0]*whx+k_ang.data[2][1]*why+k_ang.data[2][2]*whz;
                            float kh = whx*kh_x+why*kh_y+whz*kh_z;
                            if (kh > 0) {
                                float am = 1.f/kh;
                                float cdm = (jwbx-jwax)*whx+(jwby-jway)*why+(jwbz-jwaz)*whz - mspd;
                                float mimp = -am*cdm;
                                float max_i = sub_dt*mmt;
                                float new_mi = fminf(fmaxf(mi_val+mimp,-max_i),max_i);
                                mimp = new_mi - mi_val; mi_val = new_mi;
                                if (ja >= 0) {
                                    jwax -= jiia.m00*mimp*whx+jiia.m01*mimp*why+jiia.m02*mimp*whz;
                                    jway -= jiia.m01*mimp*whx+jiia.m11*mimp*why+jiia.m12*mimp*whz;
                                    jwaz -= jiia.m02*mimp*whx+jiia.m12*mimp*why+jiia.m22*mimp*whz;
                                }
                                if (jbi >= 0) {
                                    jwbx += jiib.m00*mimp*whx+jiib.m01*mimp*why+jiib.m02*mimp*whz;
                                    jwby += jiib.m01*mimp*whx+jiib.m11*mimp*why+jiib.m12*mimp*whz;
                                    jwbz += jiib.m02*mimp*whx+jiib.m12*mimp*why+jiib.m22*mimp*whz;
                                }
                            }
                        }
                        // Limits on hinge axis (if enabled)
                        int lim_en = *wp::address(j_lim_en, wid, ji);
                        if (lim_en > 0) {
                            float kh_x2=k_ang.data[0][0]*whx+k_ang.data[0][1]*why+k_ang.data[0][2]*whz;
                            float kh_y2=k_ang.data[1][0]*whx+k_ang.data[1][1]*why+k_ang.data[1][2]*whz;
                            float kh_z2=k_ang.data[2][0]*whx+k_ang.data[2][1]*why+k_ang.data[2][2]*whz;
                            float kh2 = whx*kh_x2+why*kh_y2+whz*kh_z2;
                            if (kh2 > 0) {
                                float am_lim = 1.f/kh2;
                                // Compute joint angle around hinge axis from quaternions
                                // relQ = conj(qA) * qB; angle = 2*atan2(dot(relQ.xyz, h_local), relQ.w)
                                // But we have world-frame h. Use: angle = 2*atan2(dot(relQ.xyz, h_world_A), relQ.w)
                                // where relQ = conj(qA_world) * qB_world
                                float rqx = qaw*qbx - qax*qbw - qay*qbz + qaz*qby;
                                float rqy = qaw*qby + qax*qbz - qay*qbw - qaz*qbx;
                                float rqz = qaw*qbz - qax*qby + qay*qbx - qaz*qbw;
                                float rqw = qaw*qbw + qax*qbx + qay*qby + qaz*qbz;
                                float joint_angle = 2.f * atan2f(rqx*whx+rqy*why+rqz*whz, rqw);

                                float lo = *wp::address(j_lim_lo, wid, ji);
                                float hi = *wp::address(j_lim_hi, wid, ji);
                                float lo_imp = *wp::address(j_lo_imp, wid, ji);
                                float hi_imp = *wp::address(j_hi_imp, wid, ji);

                                // Lower limit
                                { float C_lo = joint_angle - lo;
                                  float bias_lo=0, ms_lo=1.f, is_lo=0.f;
                                  if (C_lo > 0.f) { bias_lo = C_lo * inv_sub_dt; }
                                  else if (use_bias > 0) { bias_lo=jms*jbias_rate*C_lo; ms_lo=jms; is_lo=jisv; }
                                  float cdot_lo = (jwbx-jwax)*whx+(jwby-jway)*why+(jwbz-jwaz)*whz;
                                  float imp_lo = -ms_lo*am_lim*(cdot_lo+bias_lo) - is_lo*lo_imp;
                                  float new_lo = fmaxf(lo_imp+imp_lo, 0.f); imp_lo=new_lo-lo_imp; lo_imp=new_lo;
                                  if (ja>=0){jwax-=jiia.m00*imp_lo*whx+jiia.m01*imp_lo*why+jiia.m02*imp_lo*whz;
                                             jway-=jiia.m01*imp_lo*whx+jiia.m11*imp_lo*why+jiia.m12*imp_lo*whz;
                                             jwaz-=jiia.m02*imp_lo*whx+jiia.m12*imp_lo*why+jiia.m22*imp_lo*whz;}
                                  if (jbi>=0){jwbx+=jiib.m00*imp_lo*whx+jiib.m01*imp_lo*why+jiib.m02*imp_lo*whz;
                                              jwby+=jiib.m01*imp_lo*whx+jiib.m11*imp_lo*why+jiib.m12*imp_lo*whz;
                                              jwbz+=jiib.m02*imp_lo*whx+jiib.m12*imp_lo*why+jiib.m22*imp_lo*whz;} }

                                // Upper limit (signs flipped)
                                { float C_hi = hi - joint_angle;
                                  float bias_hi=0, ms_hi=1.f, is_hi=0.f;
                                  if (C_hi > 0.f) { bias_hi = C_hi * inv_sub_dt; }
                                  else if (use_bias > 0) { bias_hi=jms*jbias_rate*C_hi; ms_hi=jms; is_hi=jisv; }
                                  float cdot_hi = -((jwbx-jwax)*whx+(jwby-jway)*why+(jwbz-jwaz)*whz);
                                  float imp_hi = -ms_hi*am_lim*(cdot_hi+bias_hi) - is_hi*hi_imp;
                                  float new_hi = fmaxf(hi_imp+imp_hi, 0.f); imp_hi=new_hi-hi_imp; hi_imp=new_hi;
                                  if (ja>=0){jwax+=jiia.m00*imp_hi*whx+jiia.m01*imp_hi*why+jiia.m02*imp_hi*whz;
                                             jway+=jiia.m01*imp_hi*whx+jiia.m11*imp_hi*why+jiia.m12*imp_hi*whz;
                                             jwaz+=jiia.m02*imp_hi*whx+jiia.m12*imp_hi*why+jiia.m22*imp_hi*whz;}
                                  if (jbi>=0){jwbx-=jiib.m00*imp_hi*whx+jiib.m01*imp_hi*why+jiib.m02*imp_hi*whz;
                                              jwby-=jiib.m01*imp_hi*whx+jiib.m11*imp_hi*why+jiib.m12*imp_hi*whz;
                                              jwbz-=jiib.m02*imp_hi*whx+jiib.m12*imp_hi*why+jiib.m22*imp_hi*whz;} }

                                *wp::address(j_lo_imp, wid, ji) = lo_imp;
                                *wp::address(j_hi_imp, wid, ji) = hi_imp;
                            }
                        }

                        wp::vec_t<3,float> nai; nai[0]=ai[0]+i1; nai[1]=ai[1]+i2; nai[2]=mi_val;
                        *wp::address(j_ai, wid, ji) = nai;

                    } else if (jtype == 2) {
                        // BALL: no angular constraint (only P2P from above)
                        // Nothing additional to do

                    } else {
                        // FIXED: constrain all 3 angular DOF
                        // Solve k_ang @ impulse = -(jms * dw) - jisv * accumulated
                        float r0 = jms*dwx, r1 = jms*dwy, r2 = jms*dwz;
                        float det_a = k_ang.data[0][0]*(k_ang.data[1][1]*k_ang.data[2][2]-k_ang.data[1][2]*k_ang.data[1][2])
                                    - k_ang.data[0][1]*(k_ang.data[0][1]*k_ang.data[2][2]-k_ang.data[1][2]*k_ang.data[0][2])
                                    + k_ang.data[0][2]*(k_ang.data[0][1]*k_ang.data[1][2]-k_ang.data[1][1]*k_ang.data[0][2]);
                        if (fabsf(det_a) > 1e-12f) {
                            float ida = 1.f/det_a;
                            float a00=(k_ang.data[1][1]*k_ang.data[2][2]-k_ang.data[1][2]*k_ang.data[1][2])*ida;
                            float a01=(k_ang.data[0][2]*k_ang.data[1][2]-k_ang.data[0][1]*k_ang.data[2][2])*ida;
                            float a02=(k_ang.data[0][1]*k_ang.data[1][2]-k_ang.data[0][2]*k_ang.data[1][1])*ida;
                            float a11=(k_ang.data[0][0]*k_ang.data[2][2]-k_ang.data[0][2]*k_ang.data[0][2])*ida;
                            float a12=(k_ang.data[0][1]*k_ang.data[0][2]-k_ang.data[0][0]*k_ang.data[1][2])*ida;
                            float a22=(k_ang.data[0][0]*k_ang.data[1][1]-k_ang.data[0][1]*k_ang.data[0][1])*ida;
                            float sx = a00*r0+a01*r1+a02*r2;
                            float sy = a01*r0+a11*r1+a12*r2;
                            float sz = a02*r0+a12*r1+a22*r2;
                            float ix = -sx - jisv*ai[0];
                            float iy = -sy - jisv*ai[1];
                            float iz = -sz - jisv*ai[2];
                            if (ja >= 0) {
                                jwax -= jiia.m00*ix+jiia.m01*iy+jiia.m02*iz;
                                jway -= jiia.m01*ix+jiia.m11*iy+jiia.m12*iz;
                                jwaz -= jiia.m02*ix+jiia.m12*iy+jiia.m22*iz;
                            }
                            if (jbi >= 0) {
                                jwbx += jiib.m00*ix+jiib.m01*iy+jiib.m02*iz;
                                jwby += jiib.m01*ix+jiib.m11*iy+jiib.m12*iz;
                                jwbz += jiib.m02*ix+jiib.m12*iy+jiib.m22*iz;
                            }
                            wp::vec_t<3,float> nai; nai[0]=ai[0]+ix; nai[1]=ai[1]+iy; nai[2]=ai[2]+iz;
                            *wp::address(j_ai, wid, ji) = nai;
                        }
                    }
                }

                // Write back velocities
                if (ja >= 0) {
                    wp::vec_t<3,float> v; v[0]=jvax;v[1]=jvay;v[2]=jvaz; *wp::address(body_vel,wid,ja)=v;
                    wp::vec_t<3,float> w; w[0]=jwax;w[1]=jway;w[2]=jwaz; *wp::address(body_ang_vel,wid,ja)=w;
                }
                if (jbi >= 0) {
                    wp::vec_t<3,float> v; v[0]=jvbx;v[1]=jvby;v[2]=jvbz; *wp::address(body_vel,wid,jbi)=v;
                    wp::vec_t<3,float> w; w[0]=jwbx;w[1]=jwby;w[2]=jwbz; *wp::address(body_ang_vel,wid,jbi)=w;
                }
            }
            __syncthreads();
        }
    }
    #undef QROT
"""


@wp.func_native(snippet=_CONTACT_SOLVE_SNIPPET)
def _contact_solve_native(
    body_vel: wp.array2d(dtype=wp.vec3),
    body_ang_vel: wp.array2d(dtype=wp.vec3),
    inv_m: wp.array2d(dtype=float),
    inv_I: wp.array2d(dtype=mat3sym),
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
    do_int_vel: int,
    gx: float, gy: float, gz: float,
    lin_damp: float, ang_damp: float,
    # Joint parameters
    body_pos: wp.array2d(dtype=wp.vec3),
    body_ori: wp.array2d(dtype=wp.quat),
    j_ba: wp.array2d(dtype=wp.int32),
    j_bb: wp.array2d(dtype=wp.int32),
    j_type: wp.array2d(dtype=wp.int32),
    j_la: wp.array2d(dtype=wp.vec3),
    j_lb: wp.array2d(dtype=wp.vec3),
    j_ha: wp.array2d(dtype=wp.vec3),
    j_li: wp.array2d(dtype=wp.vec3),
    j_ai: wp.array2d(dtype=wp.vec3),
    j_ms_arr: wp.array2d(dtype=float),
    j_mmt: wp.array2d(dtype=float),
    j_lim_lo: wp.array2d(dtype=float),
    j_lim_hi: wp.array2d(dtype=float),
    j_lim_en: wp.array2d(dtype=wp.int32),
    j_lo_imp: wp.array2d(dtype=float),
    j_hi_imp: wp.array2d(dtype=float),
    jc_off: wp.array2d(dtype=wp.int32),
    num_joints: int,
    max_jcolors: int,
    jbias_rate: float,
    jmass_scale: float,
    jimpulse_scale: float,
    sub_dt: float,
    wid: int,
    tid: int,
):
    ...


@wp.kernel
def contact_solve_kernel(
    body_vel: wp.array2d(dtype=wp.vec3),
    body_ang_vel: wp.array2d(dtype=wp.vec3),
    inv_m: wp.array2d(dtype=float),
    inv_I: wp.array2d(dtype=mat3sym),
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
    do_int_vel: int,
    gx: float, gy: float, gz: float,
    lin_damp: float, ang_damp: float,
    # Joint parameters
    body_pos: wp.array2d(dtype=wp.vec3),
    body_ori: wp.array2d(dtype=wp.quat),
    j_ba: wp.array2d(dtype=wp.int32),
    j_bb: wp.array2d(dtype=wp.int32),
    j_type: wp.array2d(dtype=wp.int32),
    j_la: wp.array2d(dtype=wp.vec3),
    j_lb: wp.array2d(dtype=wp.vec3),
    j_ha: wp.array2d(dtype=wp.vec3),
    j_li: wp.array2d(dtype=wp.vec3),
    j_ai: wp.array2d(dtype=wp.vec3),
    j_ms_arr: wp.array2d(dtype=float),
    j_mmt: wp.array2d(dtype=float),
    j_lim_lo: wp.array2d(dtype=float),
    j_lim_hi: wp.array2d(dtype=float),
    j_lim_en: wp.array2d(dtype=wp.int32),
    j_lo_imp: wp.array2d(dtype=float),
    j_hi_imp: wp.array2d(dtype=float),
    jc_off: wp.array2d(dtype=wp.int32),
    num_joints: int,
    max_jcolors: int,
    jbias_rate: float,
    jmass_scale: float,
    jimpulse_scale: float,
    sub_dt_val: float,
):
    """Fused contact + joint solve: shared-memory colored Gauss-Seidel.

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
        do_int_vel, gx, gy, gz, lin_damp, ang_damp,
        body_pos, body_ori,
        j_ba, j_bb, j_type, j_la, j_lb, j_ha, j_li, j_ai,
        j_ms_arr, j_mmt,
        j_lim_lo, j_lim_hi, j_lim_en, j_lo_imp, j_hi_imp,
        jc_off,
        num_joints, max_jcolors,
        jbias_rate, jmass_scale, jimpulse_scale, sub_dt_val,
        wid, tid,
    )

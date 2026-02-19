/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

// PBR material evaluation and BSDF sampling - CUDA/OptiX version
// Converted from Vulkan GLSL pbr_common.glsl
// This is a 1:1 conversion including all features: thin-film iridescence, sheen, clearcoat, transmission, dispersion
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include func_common.h content before this file.

#ifndef PBR_COMMON_H
#define PBR_COMMON_H

//=============================================================================
// Random (from nvvkhl/shaders/random.h)
//=============================================================================
#ifndef RANDOM_H
#define RANDOM_H 1

static __forceinline__ __device__ unsigned int xxhash32(uint3 p)
{
    const uint4 primes = make_uint4(2246822519U, 3266489917U, 668265263U, 374761393U);
    unsigned int h32;
    h32 = p.z + primes.w + p.x * primes.y;
    h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * primes.y;
    h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = primes.x * (h32 ^ (h32 >> 15));
    h32 = primes.y * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

static __forceinline__ __device__ unsigned int pcg(unsigned int& state)
{
    unsigned int prev = state * 747796405u + 2891336453u;
    unsigned int word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

static __forceinline__ __device__ float rand(unsigned int& seed)
{
    unsigned int r = pcg(seed);
    return float(r) * (1.0f / float(0xffffffffu));
}

#endif  // RANDOM_H

//=============================================================================
// Ray utilities (from nvvkhl/shaders/ray_util.h)
//=============================================================================
#ifndef RAY_UTIL_H
#define RAY_UTIL_H 1

static __forceinline__ __device__ float3 offsetRay(float3 p, float3 n)
{
    const float epsilon = 1.0f / 65536.0f;
    float magnitude = length(p);
    float offset = epsilon * magnitude;
    float3 offsetVector = n * offset;
    float3 offsetPoint = p + offsetVector;
    return offsetPoint;
}

static __forceinline__ __device__ float3
pointOffset(float3 p, float3 pa, float3 pb, float3 pc, float3 na, float3 nb, float3 nc, float3 bary)
{
    float3 tmpu = p - pa;
    float3 tmpv = p - pb;
    float3 tmpw = p - pc;
    float dotu = fminf(0.0f, dot(tmpu, na));
    float dotv = fminf(0.0f, dot(tmpv, nb));
    float dotw = fminf(0.0f, dot(tmpw, nc));
    tmpu = tmpu - dotu * na;
    tmpv = tmpv - dotv * nb;
    tmpw = tmpw - dotw * nc;
    float3 pP = p + tmpu * bary.x + tmpv * bary.y + tmpw * bary.z;
    return pP;
}

#endif  // RAY_UTIL_H

//=============================================================================
// GGX functions (from nvvkhl/shaders/ggx.h)
//=============================================================================
#ifndef NVVKHL_GGX_H
#define NVVKHL_GGX_H 1

static __forceinline__ __device__ float schlickFresnel(float F0, float F90, float VdotH)
{
    return F0 + (F90 - F0) * powf(1.0f - VdotH, 5.0f);
}

static __forceinline__ __device__ float3 schlickFresnel(float3 F0, float3 F90, float VdotH)
{
    float t = powf(1.0f - VdotH, 5.0f);
    return F0 + (F90 - F0) * t;
}

static __forceinline__ __device__ float schlickFresnelIor(float ior, float VdotH)
{
    float R0 = powf((1.0f - ior) / (1.0f + ior), 2.0f);
    return R0 + (1.0f - R0) * powf(1.0f - VdotH, 5.0f);
}

static __forceinline__ __device__ float3 mix_rgb(float3 base, float3 layer, float3 factor)
{
    float maxFactor = fmaxf(factor.x, fmaxf(factor.y, factor.z));
    return (1.0f - maxFactor) * base + factor * layer;
}

static __forceinline__ __device__ float sqr(float x) { return x * x; }

static __forceinline__ __device__ bool isTIR(float2 ior, float kh)
{
    const float b = ior.x / ior.y;
    return (1.0f < (b * b * (1.0f - kh * kh)));
}

static __forceinline__ __device__ float hvd_ggx_eval(float2 invRoughness, float3 h)
{
    const float x = h.x * invRoughness.x;
    const float y = h.y * invRoughness.y;
    const float aniso = x * x + y * y;
    const float f = aniso + h.z * h.z;
    return M_1_PI * invRoughness.x * invRoughness.y * h.z / (f * f);
}

static __forceinline__ __device__ float3 hvd_ggx_sample_vndf(float3 k, float2 roughness, float2 xi)
{
    const float3 v = normalize(make_float3(k.x * roughness.x, k.y * roughness.y, k.z));
    const float3 t1
        = (v.z < 0.99999f) ? normalize(cross(v, make_float3(0.0f, 0.0f, 1.0f))) : make_float3(1.0f, 0.0f, 0.0f);
    const float3 t2 = cross(t1, v);
    const float a = 1.0f / (1.0f + v.z);
    const float r = sqrtf(xi.x);
    const float phi = (xi.y < a) ? xi.y / a * M_PI : M_PI + (xi.y - a) / (1.0f - a) * M_PI;
    float sp = sinf(phi);
    float cp = cosf(phi);
    const float p1 = r * cp;
    const float p2 = r * sp * ((xi.y < a) ? 1.0f : v.z);
    float3 h = p1 * t1 + p2 * t2 + sqrtf(fmaxf(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;
    h.x *= roughness.x;
    h.y *= roughness.y;
    h.z = fmaxf(0.0f, h.z);
    return normalize(h);
}

static __forceinline__ __device__ float smith_shadow_or_mask(float3 k, float2 roughness)
{
    float kz2 = k.z * k.z;
    if (kz2 == 0.0f)
        return 0.0f;
    const float ax = k.x * roughness.x;
    const float ay = k.y * roughness.y;
    const float inv_a2 = (ax * ax + ay * ay) / kz2;
    return 2.0f / (1.0f + sqrtf(1.0f + inv_a2));
}

static __forceinline__ __device__ float
ggx_smith_shadow_mask(float& G1, float& G2, float3 k1, float3 k2, float2 roughness)
{
    G1 = smith_shadow_or_mask(k1, roughness);
    G2 = smith_shadow_or_mask(k2, roughness);
    return G1 * G2;
}

static __forceinline__ __device__ float2 fresnel_conductor(
    float2& phase_shift_sin, float2& phase_shift_cos, float n_a, float n_b, float k_b, float cos_a, float sin_a_sqd
)
{
    const float k_b2 = k_b * k_b;
    const float n_b2 = n_b * n_b;
    const float n_a2 = n_a * n_a;
    const float tmp0 = n_b2 - k_b2;
    const float half_U = 0.5f * (tmp0 - n_a2 * sin_a_sqd);
    const float half_V = sqrtf(fmaxf(0.0f, half_U * half_U + k_b2 * n_b2));
    const float u_b2 = half_U + half_V;
    const float v_b2 = half_V - half_U;
    const float u_b = sqrtf(fmaxf(0.0f, u_b2));
    const float v_b = sqrtf(fmaxf(0.0f, v_b2));
    const float tmp1 = tmp0 * cos_a;
    const float tmp2 = n_a * u_b;
    const float tmp3 = (2.0f * n_b * k_b) * cos_a;
    const float tmp4 = n_a * v_b;
    const float tmp5 = n_a * cos_a;
    const float tmp6 = (2.0f * tmp5) * v_b;
    const float tmp7 = (u_b2 + v_b2) - tmp5 * tmp5;
    const float tmp8 = (2.0f * tmp5) * ((2.0f * n_b * k_b) * u_b - tmp0 * v_b);
    const float tmp9 = sqr((n_b2 + k_b2) * cos_a) - n_a2 * (u_b2 + v_b2);
    const float tmp67 = tmp6 * tmp6 + tmp7 * tmp7;
    const float inv_sqrt_x = (0.0f < tmp67) ? (1.0f / sqrtf(tmp67)) : 0.0f;
    const float tmp89 = tmp8 * tmp8 + tmp9 * tmp9;
    const float inv_sqrt_y = (0.0f < tmp89) ? (1.0f / sqrtf(tmp89)) : 0.0f;
    phase_shift_cos = make_float2(tmp7 * inv_sqrt_x, tmp9 * inv_sqrt_y);
    phase_shift_sin = make_float2(tmp6 * inv_sqrt_x, tmp8 * inv_sqrt_y);
    return make_float2(
        (sqr(tmp5 - u_b) + v_b2) / (sqr(tmp5 + u_b) + v_b2),
        (sqr(tmp1 - tmp2) + sqr(tmp3 - tmp4)) / (sqr(tmp1 + tmp2) + sqr(tmp3 + tmp4))
    );
}

static __forceinline__ __device__ float2 fresnel_dielectric(float n_a, float n_b, float cos_a, float cos_b)
{
    const float naca = n_a * cos_a;
    const float nbcb = n_b * cos_b;
    const float r_s = (naca - nbcb) / (naca + nbcb);
    const float nacb = n_a * cos_b;
    const float nbca = n_b * cos_a;
    const float r_p = (nbca - nacb) / (nbca + nacb);
    return make_float2(r_s * r_s, r_p * r_p);
}

// CIE XYZ color matching functions (sampled at 16 wavelengths from 400-700nm)
__device__ static const float3 cie_xyz_table[16]
    = { { 0.02986f, 0.00310f, 0.13609f }, { 0.20715f, 0.02304f, 0.99584f }, { 0.36717f, 0.06469f, 1.89550f },
        { 0.28549f, 0.13661f, 1.67236f }, { 0.08233f, 0.26856f, 0.76653f }, { 0.01723f, 0.48621f, 0.21889f },
        { 0.14400f, 0.77341f, 0.05886f }, { 0.40957f, 0.95850f, 0.01280f }, { 0.74201f, 0.97967f, 0.00060f },
        { 1.03325f, 0.84591f, 0.00000f }, { 1.08385f, 0.62242f, 0.00000f }, { 0.79203f, 0.36749f, 0.00000f },
        { 0.38751f, 0.16135f, 0.00000f }, { 0.13401f, 0.05298f, 0.00000f }, { 0.03531f, 0.01375f, 0.00000f },
        { 0.00817f, 0.00317f, 0.00000f } };

static __forceinline__ __device__ float3
thin_film_factor(float coating_thickness, float coating_ior, float base_ior, float incoming_ior, float kh)
{
    coating_thickness = fmaxf(0.0f, coating_thickness);
    const float sin0_sqr = fmaxf(0.0f, 1.0f - kh * kh);
    const float eta01 = incoming_ior / coating_ior;
    const float eta01_sqr = eta01 * eta01;
    const float sin1_sqr = eta01_sqr * sin0_sqr;
    if (1.0f < sin1_sqr)
        return make_float3(1.0f, 1.0f, 1.0f);
    const float cos1 = sqrtf(fmaxf(0.0f, 1.0f - sin1_sqr));
    const float2 R01 = fresnel_dielectric(incoming_ior, coating_ior, kh, cos1);
    float2 phi12_sin, phi12_cos;
    const float2 R12 = fresnel_conductor(phi12_sin, phi12_cos, coating_ior, base_ior, 0.0f, cos1, sin1_sqr);
    const float tmp = (4.0f * M_PI) * coating_ior * coating_thickness * cos1;
    const float R01R12_s = fmaxf(0.0f, R01.x * R12.x);
    const float r01r12_s = sqrtf(R01R12_s);
    const float R01R12_p = fmaxf(0.0f, R01.y * R12.y);
    const float r01r12_p = sqrtf(R01R12_p);
    float3 xyz = make_float3(0.0f, 0.0f, 0.0f);
    float lambda_min = 400.0f;
    float lambda_step = ((700.0f - 400.0f) / 16.0f);
    float lambda = lambda_min + 0.5f * lambda_step;
    for (int i = 0; i < 16; ++i) {
        const float phi = tmp / lambda;
        float phi_s = sinf(phi);
        float phi_c = cosf(phi);
        const float cos_phi_s = phi_c * phi12_cos.x - phi_s * phi12_sin.x;
        const float tmp_s = 2.0f * r01r12_s * cos_phi_s;
        const float R_s = (R01.x + R12.x + tmp_s) / (1.0f + R01R12_s + tmp_s);
        const float cos_phi_p = phi_c * phi12_cos.y - phi_s * phi12_sin.y;
        const float tmp_p = 2.0f * r01r12_p * cos_phi_p;
        const float R_p = (R01.y + R12.y + tmp_p) / (1.0f + R01R12_p + tmp_p);
        xyz = xyz + cie_xyz_table[i] * (0.5f * (R_s + R_p));
        lambda += lambda_step;
    }
    xyz = xyz * (1.0f / 16.0f);
    return clamp(
        make_float3(
            xyz.x * (3.2406f / 0.433509f) + xyz.y * (-1.5372f / 0.433509f) + xyz.z * (-0.4986f / 0.433509f),
            xyz.x * (-0.9689f / 0.341582f) + xyz.y * (1.8758f / 0.341582f) + xyz.z * (0.0415f / 0.341582f),
            xyz.x * (0.0557f / 0.32695f) + xyz.y * (-0.204f / 0.32695f) + xyz.z * (1.057f / 0.32695f)
        ),
        0.0f, 1.0f
    );
}

static __forceinline__ __device__ float3
compute_half_vector(float3 k1, float3 k2, float3 normal, float2 ior, float nk2, bool transmission, bool thinwalled)
{
    float3 h;
    if (transmission) {
        if (thinwalled)
            h = k1 + (normal * (nk2 + nk2) + k2);
        else {
            h = k2 * ior.y + k1 * ior.x;
            if (ior.y > ior.x)
                h = h * (-1.0f);
        }
    } else
        h = k1 + k2;
    return normalize(h);
}

static __forceinline__ __device__ float3 ggx_refract(float3 k, float3 n, float b, float nk, bool& tir)
{
    const float refraction = b * b * (1.0f - nk * nk);
    tir = (1.0f <= refraction);
    if (tir)
        return (n * (nk + nk) - k);
    return normalize((-k * b + n * (b * nk - sqrtf(1.0f - refraction))));
}

static __forceinline__ __device__ float ior_fresnel(float eta, float kh)
{
    float costheta = 1.0f - (1.0f - kh * kh) / (eta * eta);
    if (costheta <= 0.0f)
        return 1.0f;
    costheta = sqrtf(costheta);
    const float n1t1 = kh;
    const float n1t2 = costheta;
    const float n2t1 = kh * eta;
    const float n2t2 = costheta * eta;
    const float r_p = (n1t2 - n2t1) / (n1t2 + n2t1);
    const float r_o = (n1t1 - n2t2) / (n1t1 + n2t2);
    const float fres = 0.5f * (r_p * r_p + r_o * r_o);
    return fminf(fmaxf(fres, 0.0f), 1.0f);
}

static __forceinline__ __device__ float hvd_sheen_eval(float invRoughness, float nh)
{
    const float sinTheta2 = fmaxf(0.0f, 1.0f - nh * nh);
    const float sinTheta = sqrtf(sinTheta2);
    return (invRoughness + 2.0f) * powf(sinTheta, invRoughness) * 0.5f * M_1_PI * nh;
}

static __forceinline__ __device__ float vcavities_mask(float nh, float kh, float nk)
{
    return fminf(2.0f * nh * nk / kh, 1.0f);
}

static __forceinline__ __device__ float
vcavities_shadow_mask(float& G1, float& G2, float nh, float3 k1, float k1h, float3 k2, float k2h)
{
    G1 = vcavities_mask(nh, k1h, k1.z);
    G2 = vcavities_mask(nh, k2h, k2.z);
    return fminf(G1, G2);
}

static __forceinline__ __device__ float3 hvd_sheen_sample(float2 xi, float invRoughness)
{
    const float phi = 2.0f * M_PI * xi.x;
    float sinPhi = sinf(phi);
    float cosPhi = cosf(phi);
    const float sinTheta = powf(1.0f - xi.y, 1.0f / (invRoughness + 2.0f));
    const float cosTheta = sqrtf(1.0f - sinTheta * sinTheta);
    return normalize(make_float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta));
}

static __forceinline__ __device__ float3 flip(float3 h, float3 k, float xi)
{
    const float a = h.z * k.z;
    const float b = h.x * k.x + h.y * k.y;
    const float kh = fmaxf(0.0f, a + b);
    const float kh_f = fmaxf(0.0f, a - b);
    const float p_flip = kh_f / (kh + kh_f);
    return (xi < p_flip) ? make_float3(-h.x, -h.y, h.z) : h;
}

#endif  // NVVKHL_GGX_H

//=============================================================================
// BSDF structs (from nvvkhl/shaders/bsdf_structs.h)
//=============================================================================
#ifndef NVVKHL_BSDF_STRUCTS_H
#define NVVKHL_BSDF_STRUCTS_H 1

#define BSDF_EVENT_ABSORB 0
#define BSDF_EVENT_DIFFUSE 1
#define BSDF_EVENT_GLOSSY (1 << 1)
#define BSDF_EVENT_IMPULSE (1 << 2)
#define BSDF_EVENT_REFLECTION (1 << 3)
#define BSDF_EVENT_TRANSMISSION (1 << 4)
#define BSDF_EVENT_DIFFUSE_REFLECTION (BSDF_EVENT_DIFFUSE | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_DIFFUSE_TRANSMISSION (BSDF_EVENT_DIFFUSE | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_GLOSSY_REFLECTION (BSDF_EVENT_GLOSSY | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_GLOSSY_TRANSMISSION (BSDF_EVENT_GLOSSY | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_IMPULSE_REFLECTION (BSDF_EVENT_IMPULSE | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_IMPULSE_TRANSMISSION (BSDF_EVENT_IMPULSE | BSDF_EVENT_TRANSMISSION)

struct BsdfEvaluateData {
    float3 k1;
    float3 k2;
    float3 xi;
    float3 bsdf_diffuse;
    float3 bsdf_glossy;
    float pdf;
};

struct BsdfSampleData {
    float3 k1;
    float3 k2;
    float3 xi;
    float pdf;
    float3 bsdf_over_pdf;
    int event_type;
};

#endif  // NVVKHL_BSDF_STRUCTS_H

//=============================================================================
// PBR Material struct (from nvvkhl/shaders/pbr_mat_struct.h)
//=============================================================================
#ifndef NVVKHL_PBR_MAT_STRUCT_H
#define NVVKHL_PBR_MAT_STRUCT_H 1

struct PbrMaterial {
    float3 baseColor;
    float opacity;
    float2 roughness;
    float metallic;
    float3 emissive;
    float occlusion;
    float3 N;
    float3 T;
    float3 B;
    float3 Ng;
    float ior1;
    float ior2;
    float dispersion;
    float specular;
    float3 specularColor;
    float transmission;
    float3 attenuationColor;
    float attenuationDistance;
    bool isThinWalled;
    float clearcoat;
    float clearcoatRoughness;
    float3 Nc;
    float iridescence;
    float iridescenceIor;
    float iridescenceThickness;
    float3 sheenColor;
    float sheenRoughness;
    float diffuseTransmissionFactor;
    float3 diffuseTransmissionColor;
};

static __forceinline__ __device__ PbrMaterial defaultPbrMaterial()
{
    PbrMaterial mat;
    mat.baseColor = make_float3(1.0f, 1.0f, 1.0f);
    mat.opacity = 1.0f;
    mat.roughness = make_float2(1.0f, 1.0f);
    mat.metallic = 1.0f;
    mat.emissive = make_float3(0.0f, 0.0f, 0.0f);
    mat.N = make_float3(0.0f, 0.0f, 1.0f);
    mat.Ng = make_float3(0.0f, 0.0f, 1.0f);
    mat.T = make_float3(1.0f, 0.0f, 0.0f);
    mat.B = make_float3(0.0f, 1.0f, 0.0f);
    mat.ior1 = 1.0f;
    mat.ior2 = 1.5f;
    mat.dispersion = 0.0f;
    mat.specular = 1.0f;
    mat.specularColor = make_float3(1.0f, 1.0f, 1.0f);
    mat.transmission = 0.0f;
    mat.attenuationColor = make_float3(1.0f, 1.0f, 1.0f);
    mat.attenuationDistance = 1.0f;
    mat.isThinWalled = true;
    mat.clearcoat = 0.0f;
    mat.clearcoatRoughness = 0.01f;
    mat.Nc = make_float3(0.0f, 0.0f, 1.0f);
    mat.iridescence = 0.0f;
    mat.iridescenceIor = 1.5f;
    mat.iridescenceThickness = 0.1f;
    mat.sheenColor = make_float3(0.0f, 0.0f, 0.0f);
    mat.sheenRoughness = 0.0f;
    mat.occlusion = 1.0f;
    mat.diffuseTransmissionFactor = 0.0f;
    mat.diffuseTransmissionColor = make_float3(1.0f, 1.0f, 1.0f);
    return mat;
}

static __forceinline__ __device__ PbrMaterial
defaultPbrMaterial(float3 baseColor, float metallic, float roughness, float3 N, float3 Ng)
{
    PbrMaterial mat = defaultPbrMaterial();
    mat.baseColor = baseColor;
    mat.metallic = metallic;
    mat.roughness = make_float2(roughness * roughness, roughness * roughness);
    mat.Ng = Ng;
    mat.N = N;
    orthonormalBasis(mat.N, mat.T, mat.B);
    return mat;
}

#endif  // NVVKHL_PBR_MAT_STRUCT_H

//=============================================================================
// BSDF functions (from nvvkhl/shaders/bsdf_functions.h) - Full version
//=============================================================================
#ifndef NVVKHL_BSDF_FUNCTIONS_H
#define NVVKHL_BSDF_FUNCTIONS_H 1

#define DIRAC -1.0f

#define LOBE_DIFFUSE_REFLECTION 0
#define LOBE_SPECULAR_TRANSMISSION 1
#define LOBE_SPECULAR_REFLECTION 2
#define LOBE_METAL_REFLECTION 3
#define LOBE_SHEEN_REFLECTION 4
#define LOBE_CLEARCOAT_REFLECTION 5
#define LOBE_DIFFUSE_TRANSMISSION 6
#define LOBE_COUNT 7

static __forceinline__ __device__ float3 absorptionCoefficient(const PbrMaterial& mat)
{
    float3 logColor = make_float3(
        -logf(fmaxf(mat.attenuationColor.x, 0.001f)), -logf(fmaxf(mat.attenuationColor.y, 0.001f)),
        -logf(fmaxf(mat.attenuationColor.z, 0.001f))
    );
    return logColor / fmaxf(mat.attenuationDistance, 0.001f);
}

static __forceinline__ __device__ float fresnelCosineApproximation(float VdotN, float roughness)
{
    return mix(VdotN, sqrtf(0.5f + 0.5f * VdotN), sqrtf(roughness));
}

// Compute lobe weights for importance sampling
static __forceinline__ __device__ void
computeLobeWeights(float weightLobe[LOBE_COUNT], const PbrMaterial& mat, float VdotN, float3& tint)
{
    float frCoat = 0.0f;
    if (mat.clearcoat > 0.0f) {
        float frCosineClearcoat = fresnelCosineApproximation(VdotN, mat.clearcoatRoughness);
        frCoat = mat.clearcoat * ior_fresnel(1.5f / mat.ior1, frCosineClearcoat);
    }

    float frDielectric = 0.0f;
    if (mat.specular > 0.0f) {
        float frCosineDielectric = fresnelCosineApproximation(VdotN, (mat.roughness.x + mat.roughness.y) * 0.5f);
        frDielectric = ior_fresnel(mat.ior2 / mat.ior1, frCosineDielectric);
        frDielectric *= mat.specular;
    }

    if (mat.iridescence > 0.0f) {
        float3 frIridescence
            = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, VdotN);
        frDielectric
            = mix(frDielectric, fmaxf(frIridescence.x, fmaxf(frIridescence.y, frIridescence.z)), mat.iridescence);
        tint = mix_rgb(tint, mat.specularColor, frIridescence * mat.iridescence);
    }

    float sheen = 0.0f;
    if ((mat.sheenColor.x != 0.0f || mat.sheenColor.y != 0.0f || mat.sheenColor.z != 0.0f)) {
        sheen = powf(1.0f - fabsf(VdotN), mat.sheenRoughness);
        sheen = sheen / (sheen + 0.5f);
    }

    weightLobe[LOBE_CLEARCOAT_REFLECTION] = 0.0f;
    weightLobe[LOBE_SHEEN_REFLECTION] = 0.0f;
    weightLobe[LOBE_METAL_REFLECTION] = 0.0f;
    weightLobe[LOBE_SPECULAR_REFLECTION] = 0.0f;
    weightLobe[LOBE_SPECULAR_TRANSMISSION] = 0.0f;
    weightLobe[LOBE_DIFFUSE_TRANSMISSION] = 0.0f;
    weightLobe[LOBE_DIFFUSE_REFLECTION] = 0.0f;

    float weightBase = 1.0f;
    weightLobe[LOBE_CLEARCOAT_REFLECTION] = frCoat;
    weightBase *= 1.0f - frCoat;
    weightLobe[LOBE_SHEEN_REFLECTION] = weightBase * sheen;
    weightBase *= 1.0f - sheen;
    weightLobe[LOBE_METAL_REFLECTION] = weightBase * mat.metallic;
    weightBase *= 1.0f - mat.metallic;
    weightLobe[LOBE_SPECULAR_REFLECTION] = weightBase * frDielectric;
    weightBase *= 1.0f - frDielectric;
    weightLobe[LOBE_SPECULAR_TRANSMISSION] = weightBase * mat.transmission;
    float remainingWeight = weightBase * (1.0f - mat.transmission);
    weightLobe[LOBE_DIFFUSE_TRANSMISSION] = remainingWeight * mat.diffuseTransmissionFactor;
    weightLobe[LOBE_DIFFUSE_REFLECTION] = remainingWeight * (1.0f - mat.diffuseTransmissionFactor);
}

static __forceinline__ __device__ int findLobe(const PbrMaterial& mat, float VdotN, float rndVal, float3& tint)
{
    float weightLobe[LOBE_COUNT];
    computeLobeWeights(weightLobe, mat, VdotN, tint);
    int lobe = LOBE_COUNT;
    float weight = 0.0f;
    while (--lobe > 0) {
        weight += weightLobe[lobe];
        if (rndVal < weight)
            break;
    }
    return lobe;
}

// Diffuse reflection BRDF
static __forceinline__ __device__ void brdf_diffuse_eval(BsdfEvaluateData& data, const PbrMaterial& mat, float3 tint)
{
    if (dot(data.k2, mat.Ng) <= 0.0f)
        return;
    data.pdf = fmaxf(0.0f, dot(data.k2, mat.N) * M_1_PI);
    data.bsdf_diffuse = tint * data.pdf;
}

static __forceinline__ __device__ void brdf_diffuse_eval(BsdfEvaluateData& data, const PbrMaterial& mat)
{
    brdf_diffuse_eval(data, mat, mat.baseColor);
}

static __forceinline__ __device__ void brdf_diffuse_sample(BsdfSampleData& data, const PbrMaterial& mat, float3 tint)
{
    data.k2 = cosineSampleHemisphere(data.xi.x, data.xi.y);
    data.k2 = mat.T * data.k2.x + mat.B * data.k2.y + mat.N * data.k2.z;
    data.k2 = normalize(data.k2);
    data.pdf = dot(data.k2, mat.N) * M_1_PI;
    data.bsdf_over_pdf = tint;
    data.event_type = (0.0f < dot(data.k2, mat.Ng)) ? BSDF_EVENT_DIFFUSE_REFLECTION : BSDF_EVENT_ABSORB;
}

static __forceinline__ __device__ void brdf_diffuse_sample(BsdfSampleData& data, const PbrMaterial& mat)
{
    brdf_diffuse_sample(data, mat, mat.baseColor);
}

// Diffuse transmission BRDF
static __forceinline__ __device__ void
brdf_diffuse_transmission_eval(BsdfEvaluateData& data, const PbrMaterial& mat, float3 tint)
{
    if (dot(data.k2, -mat.Ng) <= 0.0f)
        return;
    data.pdf = fmaxf(0.0f, dot(data.k2, -mat.N) * M_1_PI);
    data.bsdf_diffuse = tint * data.pdf * mat.diffuseTransmissionColor;
}

static __forceinline__ __device__ void
brdf_diffuse_transmission_sample(BsdfSampleData& data, const PbrMaterial& mat, float3 tint)
{
    float3 sampledDir = cosineSampleHemisphere(data.xi.x, data.xi.y);
    data.k2 = mat.T * sampledDir.x + mat.B * sampledDir.y - mat.N * sampledDir.z;
    data.k2 = normalize(data.k2);
    data.pdf = fmaxf(0.0f, dot(data.k2, -mat.N) * M_1_PI);
    data.bsdf_over_pdf = tint * mat.diffuseTransmissionColor;
    data.event_type = (dot(data.k2, -mat.Ng) > 0.0f) ? BSDF_EVENT_DIFFUSE_TRANSMISSION : BSDF_EVENT_ABSORB;
}

// GGX-Smith BRDF (glossy reflection)
static __forceinline__ __device__ void
brdf_ggx_smith_eval(BsdfEvaluateData& data, const PbrMaterial& mat, int lobe, float3 tint)
{
    const bool backside = (dot(data.k2, mat.Ng) <= 0.0f);
    if (backside && false) {
        data.pdf = 0.0f;
        data.bsdf_glossy = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    const float nk1 = fabsf(dot(data.k1, mat.N));  // |N·V|
    const float nk2 = fabsf(dot(data.k2, mat.N));  // |N·L|
    const float3 h = normalize(data.k1 + data.k2);
    const float nh = dot(mat.N, h);
    const float k1h = dot(data.k1, h);
    const float k2h = dot(data.k2, h);
    if (nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f) {
        data.pdf = 0.0f;
        data.bsdf_glossy = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    const float3 h0 = make_float3(dot(mat.T, h), dot(mat.B, h), nh);

    data.pdf = hvd_ggx_eval(make_float2(1.0f / mat.roughness.x, 1.0f / mat.roughness.y), h0);
    float G1, G2;
    float G12 = ggx_smith_shadow_mask(
        G1, G2, make_float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1),
        make_float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), mat.roughness
    );
    data.pdf *= 0.25f / (nk1 * nh);
    float3 bsdf = make_float3(G12 * data.pdf, G12 * data.pdf, G12 * data.pdf);
    data.pdf *= G1;

    if (mat.iridescence > 0.0f) {
        const float3 factor = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, k1h);
        if (lobe == LOBE_SPECULAR_REFLECTION)
            tint = tint * mix(make_float3(1.0f, 1.0f, 1.0f), factor, mat.iridescence);
        else if (lobe == LOBE_METAL_REFLECTION)
            tint = mix_rgb(tint, mat.specularColor, factor * mat.iridescence);
    }
    data.bsdf_glossy = bsdf * tint;
}

static __forceinline__ __device__ void
brdf_ggx_smith_sample(BsdfSampleData& data, const PbrMaterial& mat, int lobe, float3 tint)
{
    data.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
    data.pdf = 0.0f;
    const float nk1 = dot(data.k1, mat.N);
    if (nk1 <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    const float3 k10 = make_float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);
    const float3 h0 = hvd_ggx_sample_vndf(k10, mat.roughness, make_float2(data.xi.x, data.xi.y));
    if (h0.z == 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;
    const float kh = dot(data.k1, h);
    if (kh <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    data.k2 = (2.0f * kh) * h - data.k1;
    const float gnk2 = dot(data.k2, mat.Ng);
    if (gnk2 <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    const float nk2 = fabsf(dot(data.k2, mat.N));
    const float k2h = fabsf(dot(data.k2, h));
    float G1, G2;
    float G12
        = ggx_smith_shadow_mask(G1, G2, k10, make_float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), mat.roughness);
    if (G12 <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    data.bsdf_over_pdf = make_float3(G2, G2, G2);
    data.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    data.pdf = hvd_ggx_eval(make_float2(1.0f / mat.roughness.x, 1.0f / mat.roughness.y), h0) * G1;
    data.pdf *= 0.25f / (nk1 * h0.z);
    if (mat.iridescence > 0.0f) {
        const float3 factor = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, kh);
        if (lobe == LOBE_SPECULAR_REFLECTION)
            tint = tint * mix(make_float3(1.0f, 1.0f, 1.0f), factor, mat.iridescence);
        else if (lobe == LOBE_METAL_REFLECTION)
            tint = mix_rgb(tint, mat.specularColor, factor * mat.iridescence);
    }
    data.bsdf_over_pdf = data.bsdf_over_pdf * tint;
}

// Dispersion helpers
static __forceinline__ __device__ float rerandomize(float v)
{
    unsigned int word = __float_as_uint(v);
    word = ((word >> ((word >> 28) + 4)) ^ word) * 277803737U;
    word = (word >> 22) ^ word;
    return float(word) / __uint_as_float(0x4f800000U);
}

#define WAVELENGTH_MIN 399.43862850585765f
#define WAVELENGTH_MAX 668.6617899434457f

static __forceinline__ __device__ float3 wavelengthToRGB(float x)
{
    float3 rgb = make_float3(0.0f, 0.0f, 0.0f);
    if (399.43862850585765f < x) {
        if (x < 435.3450352446586f)
            rgb.x = 2.6268757476158464e-05f * x + -0.010492756458829732f;
        else if (x < 452.7741480943567f)
            rgb.x = -5.383671438883332e-05f * x + 0.024380763013525125f;
        else if (x < 550.5919453498173f)
            rgb.x = 1.2536207000814165e-07f * x + -5.187018452935683e-05f;
        else if (x < 600.8694441891222f)
            rgb.x = 0.00032842519537482f * x + -0.18081111406184644f;
        else if (x < 668.6617899434457f)
            rgb.x = -0.0002438262071743009f * x + 0.16303726812428945f;
    }
    if (467.41924217251835f < x) {
        if (x < 532.3927928594046f)
            rgb.y = 0.00020126149345609334f * x + -0.0940734947497564f;
        else if (x < 552.5312202450474f)
            rgb.y = -4.3718474429905034e-05f * x + 0.03635207454767751f;
        else if (x < 605.5304635656746f)
            rgb.y = -0.00023012125757884968f * x + 0.13934543177803685f;
    }
    if (400.68666327204835f < x) {
        if (x < 447.59688835108466f)
            rgb.z = 0.00042519082480799777f * x + -0.1703682928462067f;
        else if (x < 501.2110070697423f)
            rgb.z = -0.00037202508909921054f * x + 0.18646306956262593f;
    }
    return rgb;
}

static __forceinline__ __device__ float computeDispersedIOR(float base_ior, float dispersion, float wavelength_nm)
{
    float abbeNumber = 20.0f / dispersion;
    return fmaxf(
        base_ior + (base_ior - 1.0f) * (523655.0f / (wavelength_nm * wavelength_nm) - 1.5168f) / abbeNumber, 1.0f
    );
}

// GGX-Smith BTDF (transmission)
static __forceinline__ __device__ void btdf_ggx_smith_eval(BsdfEvaluateData& data, const PbrMaterial& mat, float3 tint)
{
    bool isThinWalled = mat.isThinWalled;
    float2 ior = make_float2(mat.ior1, mat.ior2);
    if (mat.dispersion > 0.0f) {
        float wavelength = mix(WAVELENGTH_MIN, WAVELENGTH_MAX, rerandomize(data.xi.z));
        ior.x = computeDispersedIOR(ior.x, mat.dispersion, wavelength);
        tint = tint * (WAVELENGTH_MAX - WAVELENGTH_MIN) * wavelengthToRGB(wavelength);
    }
    const float nk1 = fabsf(dot(data.k1, mat.N));  // |N·V|
    const float nk2 = fabsf(dot(data.k2, mat.N));  // |N·L|
    const bool backside = (dot(data.k2, mat.Ng) < 0.0f);
    const float3 h = compute_half_vector(data.k1, data.k2, mat.N, ior, nk2, backside, isThinWalled);
    const float nh = dot(mat.N, h);
    const float k1h = dot(data.k1, h);
    const float k2h = dot(data.k2, h) * (backside ? -1.0f : 1.0f);
    if (nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f) {
        data.pdf = 0.0f;
        data.bsdf_glossy = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    float fr;
    if (!backside) {
        if (!isTIR(ior, k1h)) {
            data.pdf = 0.0f;
            data.bsdf_glossy = make_float3(0.0f, 0.0f, 0.0f);
            return;
        } else
            fr = 1.0f;
    } else
        fr = 0.0f;
    const float3 h0 = make_float3(dot(mat.T, h), dot(mat.B, h), nh);

    data.pdf = hvd_ggx_eval(make_float2(1.0f / mat.roughness.x, 1.0f / mat.roughness.y), h0);
    float G1, G2;
    float G12 = ggx_smith_shadow_mask(
        G1, G2, make_float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1),
        make_float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), mat.roughness
    );
    if (!isThinWalled && backside) {
        const float tmp = k1h * ior.x - k2h * ior.y;
        data.pdf *= k1h * k2h / (nk1 * nh * tmp * tmp);
    } else
        data.pdf *= 0.25f / (nk1 * nh);
    const float prob = (backside) ? 1.0f - fr : fr;
    const float3 bsdf = make_float3(prob * G12 * data.pdf, prob * G12 * data.pdf, prob * G12 * data.pdf);
    data.pdf *= prob * G1;
    data.bsdf_glossy = bsdf * tint;
}

static __forceinline__ __device__ void btdf_ggx_smith_sample(BsdfSampleData& data, const PbrMaterial& mat, float3 tint)
{
    bool isThinWalled = mat.isThinWalled;
    data.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
    data.pdf = 0.0f;
    float2 ior = make_float2(mat.ior1, mat.ior2);
    if (mat.dispersion > 0.0f) {
        float wavelength = mix(WAVELENGTH_MIN, WAVELENGTH_MAX, rerandomize(data.xi.z));
        ior.x = computeDispersedIOR(ior.x, mat.dispersion, wavelength);
        tint = tint * (WAVELENGTH_MAX - WAVELENGTH_MIN) * wavelengthToRGB(wavelength);
    }
    const float nk1 = fabsf(dot(data.k1, mat.N));
    const float3 k10 = make_float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);
    const float3 h0 = hvd_ggx_sample_vndf(k10, mat.roughness, make_float2(data.xi.x, data.xi.y));
    if (fabsf(h0.z) == 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;
    const float kh = dot(data.k1, h);
    if (kh <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    bool tir = false;
    if (isThinWalled) {
        data.k2 = (2.0f * kh) * h - data.k1;
        data.k2 = normalize(data.k2 - 2.0f * mat.N * dot(data.k2, mat.N));
    } else
        data.k2 = ggx_refract(data.k1, h, ior.x / ior.y, kh, tir);
    data.bsdf_over_pdf = make_float3(1.0f, 1.0f, 1.0f);
    data.event_type = (tir) ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;
    const float gnk2 = dot(data.k2, mat.Ng) * ((data.event_type == BSDF_EVENT_GLOSSY_REFLECTION) ? 1.0f : -1.0f);
    if (gnk2 <= 0.0f || isnan(data.k2.x)) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    const float nk2 = fabsf(dot(data.k2, mat.N));
    const float k2h = fabsf(dot(data.k2, h));
    float G1, G2;
    float G12
        = ggx_smith_shadow_mask(G1, G2, k10, make_float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), mat.roughness);
    if (G12 <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    data.bsdf_over_pdf = data.bsdf_over_pdf * G2;
    data.pdf = hvd_ggx_eval(make_float2(1.0f / mat.roughness.x, 1.0f / mat.roughness.y), h0) * G1;
    if (!isThinWalled && (data.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION)) {
        const float tmp = kh * ior.x - k2h * ior.y;
        if (tmp > 0)
            data.pdf *= kh * k2h / (nk1 * h0.z * tmp * tmp);
    } else
        data.pdf *= 0.25f / (nk1 * h0.z);
    data.bsdf_over_pdf = data.bsdf_over_pdf * tint;
}

// Sheen BRDF
static __forceinline__ __device__ void brdf_sheen_eval(BsdfEvaluateData& data, const PbrMaterial& mat)
{
    const bool backside = (dot(data.k2, mat.Ng) <= 0.0f);
    if (backside)
        return;
    const float nk1 = fabsf(dot(data.k1, mat.N));  // |N·V|
    const float nk2 = fabsf(dot(data.k2, mat.N));  // |N·L|
    const float3 h = normalize(data.k1 + data.k2);
    const float nh = dot(mat.N, h);
    const float k1h = dot(data.k1, h);
    const float k2h = dot(data.k2, h);
    if (nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
        return;
    const float invRoughness = 1.0f / (mat.sheenRoughness * mat.sheenRoughness);
    const float3 h0 = make_float3(dot(mat.T, h), dot(mat.B, h), nh);

    data.pdf = hvd_sheen_eval(invRoughness, h0.z);
    float G1, G2;
    const float G12 = vcavities_shadow_mask(
        G1, G2, h0.z, make_float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1), k1h,
        make_float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), k2h
    );
    data.pdf *= 0.25f / (nk1 * nh);
    const float3 bsdf = make_float3(G12 * data.pdf, G12 * data.pdf, G12 * data.pdf);
    data.pdf *= G1;
    data.bsdf_glossy = bsdf * mat.sheenColor;
}

static __forceinline__ __device__ void brdf_sheen_sample(BsdfSampleData& data, const PbrMaterial& mat)
{
    data.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
    data.pdf = 0.0f;
    const float invRoughness = 1.0f / (mat.sheenRoughness * mat.sheenRoughness);
    const float nk1 = fabsf(dot(data.k1, mat.N));
    const float3 k10 = make_float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);
    float xiFlip = data.xi.z;
    const float3 h0 = flip(hvd_sheen_sample(make_float2(data.xi.x, data.xi.y), invRoughness), k10, xiFlip);
    if (fabsf(h0.z) == 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;
    const float k1h = dot(data.k1, h);
    if (k1h <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    data.k2 = (2.0f * k1h) * h - data.k1;
    data.bsdf_over_pdf = make_float3(1.0f, 1.0f, 1.0f);
    data.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    const float gnk2 = dot(data.k2, mat.Ng);
    if (gnk2 <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    const float nk2 = fabsf(dot(data.k2, mat.N));
    const float k2h = fabsf(dot(data.k2, h));
    float G1, G2;
    const float G12 = vcavities_shadow_mask(
        G1, G2, h0.z, k10, k1h, make_float3(dot(data.k2, mat.T), dot(mat.B, data.k2), nk2), k2h
    );
    if (G12 <= 0.0f) {
        data.event_type = BSDF_EVENT_ABSORB;
        return;
    }
    data.bsdf_over_pdf = data.bsdf_over_pdf * (G12 / G1);
    data.pdf = hvd_sheen_eval(invRoughness, h0.z) * G1;
    data.pdf *= 0.25f / (nk1 * h0.z);
    data.bsdf_over_pdf = data.bsdf_over_pdf * mat.sheenColor;
}

// Main BSDF evaluate function - full version with all lobes
static __forceinline__ __device__ void bsdfEvaluate(BsdfEvaluateData& data, PbrMaterial mat)
{
    float3 tint = mat.baseColor;
    float VdotN = dot(data.k1, mat.N);
    int lobe = findLobe(mat, VdotN, data.xi.z, tint);
    data.bsdf_diffuse = make_float3(0.0f, 0.0f, 0.0f);
    data.bsdf_glossy = make_float3(0.0f, 0.0f, 0.0f);
    data.pdf = 0.0f;

    if (lobe == LOBE_DIFFUSE_REFLECTION)
        brdf_diffuse_eval(data, mat, tint);
    else if (lobe == LOBE_DIFFUSE_TRANSMISSION)
        brdf_diffuse_transmission_eval(data, mat, tint);
    else if (lobe == LOBE_SPECULAR_REFLECTION)
        brdf_ggx_smith_eval(data, mat, LOBE_SPECULAR_REFLECTION, mat.specularColor);
    else if (lobe == LOBE_SPECULAR_TRANSMISSION)
        btdf_ggx_smith_eval(data, mat, tint);
    else if (lobe == LOBE_METAL_REFLECTION)
        brdf_ggx_smith_eval(data, mat, LOBE_METAL_REFLECTION, mat.baseColor);
    else if (lobe == LOBE_CLEARCOAT_REFLECTION) {
        mat.roughness = make_float2(
            mat.clearcoatRoughness * mat.clearcoatRoughness, mat.clearcoatRoughness * mat.clearcoatRoughness
        );
        mat.N = mat.Nc;
        mat.iridescence = 0.0f;
        brdf_ggx_smith_eval(data, mat, LOBE_CLEARCOAT_REFLECTION, make_float3(1.0f, 1.0f, 1.0f));
    } else if (lobe == LOBE_SHEEN_REFLECTION)
        brdf_sheen_eval(data, mat);

    data.bsdf_diffuse = data.bsdf_diffuse * mat.occlusion;
    data.bsdf_glossy = data.bsdf_glossy * mat.occlusion;
}

// Main BSDF sample function - full version with all lobes
static __forceinline__ __device__ void bsdfSample(BsdfSampleData& data, PbrMaterial mat)
{
    float3 tint = mat.baseColor;
    float VdotN = dot(data.k1, mat.N);
    int lobe = findLobe(mat, VdotN, data.xi.z, tint);
    data.pdf = 0.0f;
    data.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
    data.event_type = BSDF_EVENT_ABSORB;

    if (lobe == LOBE_DIFFUSE_REFLECTION)
        brdf_diffuse_sample(data, mat, tint);
    else if (lobe == LOBE_DIFFUSE_TRANSMISSION)
        brdf_diffuse_transmission_sample(data, mat, tint);
    else if (lobe == LOBE_SPECULAR_REFLECTION)
        brdf_ggx_smith_sample(data, mat, LOBE_SPECULAR_REFLECTION, mat.specularColor);
    else if (lobe == LOBE_SPECULAR_TRANSMISSION)
        btdf_ggx_smith_sample(data, mat, tint);
    else if (lobe == LOBE_METAL_REFLECTION)
        brdf_ggx_smith_sample(data, mat, LOBE_METAL_REFLECTION, mat.baseColor);
    else if (lobe == LOBE_CLEARCOAT_REFLECTION) {
        mat.roughness = make_float2(
            mat.clearcoatRoughness * mat.clearcoatRoughness, mat.clearcoatRoughness * mat.clearcoatRoughness
        );
        mat.N = mat.Nc;
        mat.B = normalize(cross(mat.N, mat.T));
        mat.T = cross(mat.B, mat.N);
        mat.iridescence = 0.0f;
        brdf_ggx_smith_sample(data, mat, LOBE_CLEARCOAT_REFLECTION, make_float3(1.0f, 1.0f, 1.0f));
    } else if (lobe == LOBE_SHEEN_REFLECTION)
        brdf_sheen_sample(data, mat);

    if (data.pdf <= 0.00001f || any_isnan(data.bsdf_over_pdf))
        data.event_type = BSDF_EVENT_ABSORB;
    if ((isnan(data.pdf) || isinf(data.pdf)) && data.event_type != BSDF_EVENT_ABSORB) {
        data.event_type = (data.event_type & (~BSDF_EVENT_GLOSSY)) | BSDF_EVENT_IMPULSE;
        data.pdf = DIRAC;
    }
}

// Simple BSDF model (for fallback/testing)
static __forceinline__ __device__ float bsdfSimpleGlossyProbability(float NdotV, float metallic)
{
    return mix(schlickFresnel(0.04f, 1.0f, NdotV), 1.0f, metallic);
}

static __forceinline__ __device__ void bsdfEvaluateSimple(BsdfEvaluateData& data, const PbrMaterial& mat)
{
    float3 H = normalize(data.k1 + data.k2);
    float NdotV = clampedDot(mat.N, data.k1);
    float NdotL = clampedDot(mat.N, data.k2);
    float VdotH = clampedDot(data.k1, H);
    float NdotH = clampedDot(mat.N, H);
    if (NdotV == 0.0f || NdotL == 0.0f || VdotH == 0.0f || NdotH == 0.0f) {
        data.bsdf_diffuse = data.bsdf_glossy = make_float3(0.0f, 0.0f, 0.0f);
        data.pdf = 0.0f;
        return;
    }
    float c_min_reflectance = 0.04f;
    float3 f0 = mix(make_float3(c_min_reflectance, c_min_reflectance, c_min_reflectance), mat.baseColor, mat.metallic);
    float3 fGlossy = schlickFresnel(f0, make_float3(1.0f, 1.0f, 1.0f), VdotH);
    float fDiffuse = schlickFresnel(1.0f - c_min_reflectance, 0.0f, VdotH) * (1.0f - mat.metallic);
    float3 localH = make_float3(dot(mat.T, H), dot(mat.B, H), NdotH);
    float d = hvd_ggx_eval(make_float2(1.0f / mat.roughness.x, 1.0f / mat.roughness.y), localH);
    float3 localK1 = make_float3(dot(mat.T, data.k1), dot(mat.B, data.k1), NdotV);
    float3 localK2 = make_float3(dot(mat.T, data.k2), dot(mat.B, data.k2), NdotL);
    float G1 = 0.0f, G2 = 0.0f;
    ggx_smith_shadow_mask(G1, G2, localK1, localK2, mat.roughness);
    float diffusePdf = M_1_PI * NdotL;
    float specularPdf = G1 * d * 0.25f / (NdotV * NdotH);
    data.pdf = mix(diffusePdf, specularPdf, bsdfSimpleGlossyProbability(NdotV, mat.metallic));
    data.bsdf_diffuse = mat.baseColor * fDiffuse * diffusePdf;
    data.bsdf_glossy = fGlossy * G2 * specularPdf;
}

static __forceinline__ __device__ void bsdfSampleSimple(BsdfSampleData& data, const PbrMaterial& mat)
{
    float3 tint = mat.baseColor;
    data.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
    float nk1 = clampedDot(mat.N, data.k1);
    if (data.xi.z <= bsdfSimpleGlossyProbability(nk1, mat.metallic)) {
        data.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
        float3 localK1 = make_float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1);
        float3 halfVector = hvd_ggx_sample_vndf(localK1, mat.roughness, make_float2(data.xi.x, data.xi.y));
        halfVector = mat.T * halfVector.x + mat.B * halfVector.y + mat.N * halfVector.z;
        data.k2 = reflect(-data.k1, halfVector);
    } else {
        data.event_type = BSDF_EVENT_DIFFUSE_REFLECTION;
        float3 localDir = cosineSampleHemisphere(data.xi.x, data.xi.y);
        data.k2 = mat.T * localDir.x + mat.B * localDir.y + mat.N * localDir.z;
    }
    BsdfEvaluateData evalData;
    evalData.k1 = data.k1;
    evalData.k2 = data.k2;
    bsdfEvaluateSimple(evalData, mat);
    data.pdf = evalData.pdf;
    float3 bsdf_total = evalData.bsdf_diffuse + evalData.bsdf_glossy;
    if (data.pdf <= 0.00001f || any_isnan(bsdf_total)) {
        data.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
        data.event_type = BSDF_EVENT_ABSORB;
    } else
        data.bsdf_over_pdf = bsdf_total / data.pdf;
}

static __forceinline__ __device__ float3 bsdfSimpleAverageReflectance(float3 k1, const PbrMaterial& mat)
{
    float NdotV = clampedDot(mat.N, k1);
    float c_min_reflectance = 0.04f;
    float3 f0 = mix(make_float3(c_min_reflectance, c_min_reflectance, c_min_reflectance), mat.baseColor, mat.metallic);
    float3 bsdf_glossy_average = schlickFresnel(f0, make_float3(1.0f, 1.0f, 1.0f), NdotV);
    float3 bsdf_diffuse_average
        = mat.baseColor * schlickFresnel(1.0f - c_min_reflectance, 0.0f, NdotV) * (1.0f - mat.metallic);
    return bsdf_glossy_average + bsdf_diffuse_average;
}

#endif  // NVVKHL_BSDF_FUNCTIONS_H

#endif  // PBR_COMMON_H

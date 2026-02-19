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

// Shared utility functions - CUDA/OptiX version
// Converted from Vulkan GLSL func_common.glsl
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include constants.h content before this file.

#ifndef FUNC_COMMON_H
#define FUNC_COMMON_H

// Also define the nvvkhl guard for compatibility
#define NVVKHL_FUNC_H 1

// Constants should already be defined - provide fallbacks
// Force float constants if host headers already defined double variants.
#ifdef M_PI
#undef M_PI
#endif
#ifdef M_TWO_PI
#undef M_TWO_PI
#endif
#ifdef M_PI_2
#undef M_PI_2
#endif
#ifdef M_PI_4
#undef M_PI_4
#endif
#ifdef M_1_OVER_PI
#undef M_1_OVER_PI
#endif
#ifdef M_1_PI
#undef M_1_PI
#endif
#define M_PI        3.14159265358979323846f
#define M_TWO_PI    6.28318530717958648f
#define M_PI_2      1.57079632679489661923f
#define M_PI_4      0.785398163397448309616f
#define M_1_OVER_PI 0.318309886183790671538f
#define M_1_PI      0.318309886183790671538f

// Force fast float-only intrinsics in OptiX kernels.
// This avoids hidden libdevice f64 range-reduction paths for transcendental math.
#ifndef WARP_USE_FAST_INTRINSICS
#define WARP_USE_FAST_INTRINSICS 1
#endif

#if WARP_USE_FAST_INTRINSICS
#define sinf   __sinf
#define cosf   __cosf
#define tanf   __tanf
#define expf   __expf
#define logf   __logf
#define powf   __powf
#endif

//-----------------------------------------------------------------------------
// Matrix types (CUDA doesn't have built-in matrix types like GLSL)
//-----------------------------------------------------------------------------
#ifndef FLOAT4X4_DEFINED
#define FLOAT4X4_DEFINED

struct float4x4 {
    float4 rows[4];

    __forceinline__ __device__ float4& operator[](int i) { return rows[i]; }
    __forceinline__ __device__ const float4& operator[](int i) const { return rows[i]; }
};

struct float3x3 {
    float3 rows[3];

    __forceinline__ __device__ float3& operator[](int i) { return rows[i]; }
    __forceinline__ __device__ const float3& operator[](int i) const { return rows[i]; }
};

// Row-major matrix-vector multiplication: result_i = dot(row_i, v)
static __forceinline__ __device__ float4 mul(const float4x4& m, float4 v)
{
    return make_float4(
        m[0].x * v.x + m[0].y * v.y + m[0].z * v.z + m[0].w * v.w,
        m[1].x * v.x + m[1].y * v.y + m[1].z * v.z + m[1].w * v.w,
        m[2].x * v.x + m[2].y * v.y + m[2].z * v.z + m[2].w * v.w,
        m[3].x * v.x + m[3].y * v.y + m[3].z * v.z + m[3].w * v.w
    );
}

// Column-major matrix-vector multiplication (GLSL convention).
// The matrix memory is laid out row-by-row (GLM/NumPy style), but we treat
// each stored "row" as a column.  This matches GLSL `mat4 * vec4`.
static __forceinline__ __device__ float4 mul_cm(const float4x4& m, float4 v)
{
    return make_float4(
        m[0].x * v.x + m[1].x * v.y + m[2].x * v.z + m[3].x * v.w,
        m[0].y * v.x + m[1].y * v.y + m[2].y * v.z + m[3].y * v.w,
        m[0].z * v.x + m[1].z * v.y + m[2].z * v.z + m[3].z * v.w,
        m[0].w * v.x + m[1].w * v.y + m[2].w * v.z + m[3].w * v.w
    );
}

// Column-major mat3 * vec3 (GLSL convention).
static __forceinline__ __device__ float3 mul_cm(const float3x3& m, float3 v)
{
    return make_float3(
        m[0].x * v.x + m[1].x * v.y + m[2].x * v.z, m[0].y * v.x + m[1].y * v.y + m[2].y * v.z,
        m[0].z * v.x + m[1].z * v.y + m[2].z * v.z
    );
}

static __forceinline__ __device__ float3 mul(const float3x3& m, float3 v)
{
    return make_float3(
        m[0].x * v.x + m[0].y * v.y + m[0].z * v.z, m[1].x * v.x + m[1].y * v.y + m[1].z * v.z,
        m[2].x * v.x + m[2].y * v.y + m[2].z * v.z
    );
}

// Matrix-matrix multiplication
static __forceinline__ __device__ float4x4 mul(const float4x4& a, const float4x4& b)
{
    float4x4 result;
    for (int i = 0; i < 4; i++) {
        result[i].x = a[i].x * b[0].x + a[i].y * b[1].x + a[i].z * b[2].x + a[i].w * b[3].x;
        result[i].y = a[i].x * b[0].y + a[i].y * b[1].y + a[i].z * b[2].y + a[i].w * b[3].y;
        result[i].z = a[i].x * b[0].z + a[i].y * b[1].z + a[i].z * b[2].z + a[i].w * b[3].z;
        result[i].w = a[i].x * b[0].w + a[i].y * b[1].w + a[i].z * b[2].w + a[i].w * b[3].w;
    }
    return result;
}

// Make identity matrix
static __forceinline__ __device__ float4x4 make_float4x4_identity()
{
    float4x4 m;
    m[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    m[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    m[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
    m[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    return m;
}

static __forceinline__ __device__ float3x3 make_float3x3_identity()
{
    float3x3 m;
    m[0] = make_float3(1.0f, 0.0f, 0.0f);
    m[1] = make_float3(0.0f, 1.0f, 0.0f);
    m[2] = make_float3(0.0f, 0.0f, 1.0f);
    return m;
}

// Extract 3x3 from 4x4
static __forceinline__ __device__ float3x3 make_float3x3(const float4x4& m)
{
    float3x3 result;
    result[0] = make_float3(m[0].x, m[0].y, m[0].z);
    result[1] = make_float3(m[1].x, m[1].y, m[1].z);
    result[2] = make_float3(m[2].x, m[2].y, m[2].z);
    return result;
}

// float2x3 for UV transforms (2 rows, 3 columns)
struct float2x3 {
    float3 rows[2];

    __forceinline__ __device__ float3& operator[](int i) { return rows[i]; }
    __forceinline__ __device__ const float3& operator[](int i) const { return rows[i]; }
};

// Transform UV with 2x3 matrix
static __forceinline__ __device__ float2 transformUV(const float2x3& m, float2 uv)
{
    return make_float2(m[0].x * uv.x + m[0].y * uv.y + m[0].z, m[1].x * uv.x + m[1].y * uv.y + m[1].z);
}

static __forceinline__ __device__ float2x3 make_float2x3_identity()
{
    float2x3 m;
    m[0] = make_float3(1.0f, 0.0f, 0.0f);
    m[1] = make_float3(0.0f, 1.0f, 0.0f);
    return m;
}

#endif  // FLOAT4X4_DEFINED

//-----------------------------------------------------------------------------
// Vector math helpers (CUDA doesn't have all GLSL built-ins)
//-----------------------------------------------------------------------------

static __forceinline__ __device__ float square(float x) { return x * x; }

static __forceinline__ __device__ float saturate(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

static __forceinline__ __device__ float3 saturate(float3 x)
{
    return make_float3(saturate(x.x), saturate(x.y), saturate(x.z));
}

static __forceinline__ __device__ float luminance(float3 color)
{
    return color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
}

static __forceinline__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static __forceinline__ __device__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

static __forceinline__ __device__ float length(float3 v) { return sqrtf(dot(v, v)); }

static __forceinline__ __device__ float length(float2 v) { return sqrtf(dot(v, v)); }

static __forceinline__ __device__ float3 normalize(float3 v)
{
    float inv_len = rsqrtf(fmaxf(dot(v, v), 1e-20f));
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

static __forceinline__ __device__ float2 normalize(float2 v)
{
    float inv_len = rsqrtf(fmaxf(dot(v, v), 1e-20f));
    return make_float2(v.x * inv_len, v.y * inv_len);
}

static __forceinline__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float clampedDot(float3 x, float3 y) { return fmaxf(fminf(dot(x, y), 1.0f), 0.0f); }

// Vector operations
static __forceinline__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __device__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }

static __forceinline__ __device__ float3 operator*(float s, float3 a) { return make_float3(a.x * s, a.y * s, a.z * s); }

static __forceinline__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

static __forceinline__ __device__ float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }

static __forceinline__ __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }

static __forceinline__ __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

static __forceinline__ __device__ float2 operator*(float2 a, float s) { return make_float2(a.x * s, a.y * s); }

static __forceinline__ __device__ float2 operator*(float s, float2 a) { return make_float2(a.x * s, a.y * s); }

static __forceinline__ __device__ float2 operator/(float2 a, float s)
{
    float inv = 1.0f / s;
    return make_float2(a.x * inv, a.y * inv);
}

static __forceinline__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static __forceinline__ __device__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

static __forceinline__ __device__ float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

// Mix/lerp functions
static __forceinline__ __device__ float mix(float a, float b, float t) { return a + (b - a) * t; }

static __forceinline__ __device__ float3 mix(float3 a, float3 b, float t) { return a + (b - a) * t; }

static __forceinline__ __device__ float3 mix(float3 a, float3 b, float3 t)
{
    return make_float3(a.x + (b.x - a.x) * t.x, a.y + (b.y - a.y) * t.y, a.z + (b.z - a.z) * t.z);
}

static __forceinline__ __device__ float2 mix(float2 a, float2 b, float t)
{
    return make_float2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
}

static __forceinline__ __device__ float4 mix(float4 a, float4 b, float t)
{
    return make_float4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

// Clamp functions
static __forceinline__ __device__ float3 clamp(float3 v, float3 lo, float3 hi)
{
    return make_float3(fminf(fmaxf(v.x, lo.x), hi.x), fminf(fmaxf(v.y, lo.y), hi.y), fminf(fmaxf(v.z, lo.z), hi.z));
}

static __forceinline__ __device__ float3 clamp(float3 v, float lo, float hi)
{
    return make_float3(fminf(fmaxf(v.x, lo), hi), fminf(fmaxf(v.y, lo), hi), fminf(fmaxf(v.z, lo), hi));
}

static __forceinline__ __device__ float2 clamp(float2 v, float lo, float hi)
{
    return make_float2(fminf(fmaxf(v.x, lo), hi), fminf(fmaxf(v.y, lo), hi));
}

// Max/min for vectors
static __forceinline__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

static __forceinline__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

static __forceinline__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

static __forceinline__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

// Abs for vectors
static __forceinline__ __device__ float3 fabsf(float3 v) { return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }

// Reflect
static __forceinline__ __device__ float3 reflect(float3 I, float3 N) { return I - 2.0f * dot(N, I) * N; }

// Sign
static __forceinline__ __device__ float sign(float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); }

// Any/all for checking NaN/Inf
static __forceinline__ __device__ bool any_isnan(float3 v) { return isnan(v.x) || isnan(v.y) || isnan(v.z); }

static __forceinline__ __device__ bool any_isinf(float3 v) { return isinf(v.x) || isinf(v.y) || isinf(v.z); }

//-----------------------------------------------------------------------------
// Original func_common functions
//-----------------------------------------------------------------------------

static __forceinline__ __device__ void orthonormalBasis(float3 normal, float3& tangent, float3& bitangent)
{
    if (normal.z < -0.99998796f) {
        tangent = make_float3(0.0f, -1.0f, 0.0f);
        bitangent = make_float3(-1.0f, 0.0f, 0.0f);
        return;
    }
    float a = 1.0f / (1.0f + normal.z);
    float b = -normal.x * normal.y * a;
    tangent = make_float3(1.0f - normal.x * normal.x * a, b, -normal.x);
    bitangent = make_float3(b, 1.0f - normal.y * normal.y * a, -normal.y);
}

static __forceinline__ __device__ float4 makeFastTangent(float3 normal)
{
    float3 tangent, unused;
    orthonormalBasis(normal, tangent, unused);
    return make_float4(tangent.x, tangent.y, tangent.z, 1.0f);
}

static __forceinline__ __device__ float3 rotate(float3 v, float3 k, float theta)
{
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    return (v * cos_theta) + (cross(k, v) * sin_theta) + (k * dot(k, v)) * (1.0f - cos_theta);
}

static __forceinline__ __device__ float2 getSphericalUv(float3 v)
{
    float gamma = asinf(-v.y);
    float theta = atan2f(v.z, v.x);
    float2 uv = make_float2(theta * M_1_OVER_PI * 0.5f, gamma * M_1_OVER_PI) + make_float2(0.5f, 0.5f);
    return uv;
}

static __forceinline__ __device__ float2 mixBary(float2 a, float2 b, float2 c, float3 bary)
{
    return a * bary.x + b * bary.y + c * bary.z;
}

static __forceinline__ __device__ float3 mixBary(float3 a, float3 b, float3 c, float3 bary)
{
    return a * bary.x + b * bary.y + c * bary.z;
}

static __forceinline__ __device__ float4 mixBary(float4 a, float4 b, float4 c, float3 bary)
{
    return a * bary.x + b * bary.y + c * bary.z;
}

static __forceinline__ __device__ float3 cosineSampleHemisphere(float r1, float r2)
{
    float r = sqrtf(r1);
    float phi = M_TWO_PI * r2;
    float3 dir;
    dir.x = r * cosf(phi);
    dir.y = r * sinf(phi);
    dir.z = sqrtf(1.0f - r1);
    return dir;
}

static __forceinline__ __device__ float powerHeuristic(float a, float b)
{
    const float t = a * a;
    return t / (b * b + t);
}

#endif  // FUNC_COMMON_H

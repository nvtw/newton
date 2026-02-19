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

/*
 * Tonemapping compute kernel - CUDA version
 * Converted from Vulkan GLSL tonemap.comp
 * Based on nvpro_core/nvvkhl tonemapper
 *
 * NOTE: This file is designed to be used as a standalone CUDA kernel,
 *       not as part of the OptiX path tracing pipeline.
 */

#ifndef TONEMAP_H
#define TONEMAP_H

#define TONEMAP_WORKGROUP_SIZE 16

//-----------------------------------------------------------------------------
// Tonemapper settings - matches nvvkhl_shaders::Tonemapper
//-----------------------------------------------------------------------------
struct TonemapperParams {
    int method;  // 0=Filmic, 1=Uncharted2, 2=Clip, 3=ACES, 4=AgX, 5=KhronosPBR
    int isActive;
    float exposure;
    float brightness;
    float contrast;
    float saturation;
    float vignette;
    float _padding;
};

// Constants for tonemap methods
#define eTonemapFilmic     0
#define eTonemapUncharted2 1
#define eTonemapClip       2
#define eTonemapACES       3
#define eTonemapAgX        4
#define eTonemapKhronosPBR 5

//-----------------------------------------------------------------------------
// Vector helper functions (if not already defined)
//-----------------------------------------------------------------------------
#ifndef TONEMAP_VECTOR_HELPERS
#define TONEMAP_VECTOR_HELPERS

static __forceinline__ __device__ float3 make_float3_scalar(float s) { return make_float3(s, s, s); }

static __forceinline__ __device__ float3 tonemap_max(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

static __forceinline__ __device__ float3 tonemap_clamp(float3 v, float lo, float hi)
{
    return make_float3(fminf(fmaxf(v.x, lo), hi), fminf(fmaxf(v.y, lo), hi), fminf(fmaxf(v.z, lo), hi));
}

static __forceinline__ __device__ float3 tonemap_mix(float3 a, float3 b, float t)
{
    return make_float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

static __forceinline__ __device__ float3 tonemap_mix3(float3 a, float3 b, float3 t)
{
    return make_float3(a.x + (b.x - a.x) * t.x, a.y + (b.y - a.y) * t.y, a.z + (b.z - a.z) * t.z);
}

static __forceinline__ __device__ float3 tonemap_pow(float3 v, float3 e)
{
    return make_float3(powf(v.x, e.x), powf(v.y, e.y), powf(v.z, e.z));
}

static __forceinline__ __device__ float3 tonemap_log2(float3 v)
{
    return make_float3(log2f(v.x), log2f(v.y), log2f(v.z));
}

static __forceinline__ __device__ float tonemap_dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Vector operators
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

static __forceinline__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

static __forceinline__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

static __forceinline__ __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

static __forceinline__ __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }

static __forceinline__ __device__ float2 operator/(float2 a, float2 b) { return make_float2(a.x / b.x, a.y / b.y); }

static __forceinline__ __device__ float tonemap_dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

#endif  // TONEMAP_VECTOR_HELPERS

//-----------------------------------------------------------------------------
// Linear to sRGB conversion
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 toSrgb(float3 rgb)
{
    float3 lo = rgb * 12.92f;
    float3 hi = tonemap_pow(rgb, make_float3_scalar(1.0f / 2.4f)) * 1.055f - make_float3_scalar(0.055f);
    // mix(lo, hi, greaterThan(rgb, vec3(0.0031308)))
    return make_float3(
        rgb.x > 0.0031308f ? hi.x : lo.x, rgb.y > 0.0031308f ? hi.y : lo.y, rgb.z > 0.0031308f ? hi.z : lo.z
    );
}

//-----------------------------------------------------------------------------
// Filmic tonemapping (Jim Hejl / Richard Burgess-Dawson)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 tonemapFilmic(float3 color)
{
    float3 temp = tonemap_max(make_float3_scalar(0.0f), color - make_float3_scalar(0.004f));
    float3 result = (temp * (make_float3_scalar(6.2f) * temp + make_float3_scalar(0.5f)))
        / (temp * (make_float3_scalar(6.2f) * temp + make_float3_scalar(1.7f)) + make_float3_scalar(0.06f));
    return result;
}

//-----------------------------------------------------------------------------
// Uncharted 2 tonemapping (John Hable)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 tonemapUncharted2Impl(float3 color)
{
    const float a = 0.15f;
    const float b = 0.50f;
    const float c = 0.10f;
    const float d = 0.20f;
    const float e = 0.02f;
    const float f = 0.30f;
    return ((color * (a * color + make_float3_scalar(c * b)) + make_float3_scalar(d * e))
            / (color * (a * color + make_float3_scalar(b)) + make_float3_scalar(d * f)))
        - make_float3_scalar(e / f);
}

static __forceinline__ __device__ float3 tonemapUncharted2(float3 color)
{
    const float W = 11.2f;
    const float exposure_bias = 2.0f;
    color = tonemapUncharted2Impl(color * exposure_bias);
    float3 white_scale = make_float3_scalar(1.0f) / tonemapUncharted2Impl(make_float3_scalar(W));
    return tonemap_pow(color * white_scale, make_float3_scalar(1.0f / 2.2f));
}

//-----------------------------------------------------------------------------
// ACES approximation (Stephen Hill)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 tonemapACES(float3 color)
{
    // ACESInputMat * color (row-major multiplication)
    float3 c;
    c.x = 0.59719f * color.x + 0.35458f * color.y + 0.04823f * color.z;
    c.y = 0.07600f * color.x + 0.90834f * color.y + 0.01566f * color.z;
    c.z = 0.02840f * color.x + 0.13383f * color.y + 0.83777f * color.z;

    float3 a = c * (c + make_float3_scalar(0.0245786f)) - make_float3_scalar(0.000090537f);
    float3 b = c * (make_float3_scalar(0.983729f) * c + make_float3_scalar(0.4329510f)) + make_float3_scalar(0.238081f);
    c = a / b;

    // ACESOutputMat * c
    float3 result;
    result.x = 1.60475f * c.x - 0.53108f * c.y - 0.07367f * c.z;
    result.y = -0.10208f * c.x + 1.10813f * c.y - 0.00605f * c.z;
    result.z = -0.00327f * c.x - 0.07276f * c.y + 1.07602f * c.z;

    return toSrgb(result);
}

//-----------------------------------------------------------------------------
// AgX tonemapping (Benjamin Wrensch / Troy Sobotka)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 tonemapAgX(float3 color)
{
    // agx_mat * color
    float3 c;
    c.x = 0.842479062253094f * color.x + 0.0784335999999992f * color.y + 0.0792237451477643f * color.z;
    c.y = 0.0423282422610123f * color.x + 0.878468636469772f * color.y + 0.0791661274605434f * color.z;
    c.z = 0.0423756549057051f * color.x + 0.0784336f * color.y + 0.879142973793104f * color.z;

    const float min_ev = -12.47393f;
    const float max_ev = 4.026069f;
    c = tonemap_clamp(tonemap_log2(c), min_ev, max_ev);
    c = (c - make_float3_scalar(min_ev)) / (max_ev - min_ev);

    float3 v = c * 15.5f + make_float3_scalar(-40.14f);
    v = c * v + make_float3_scalar(31.96f);
    v = c * v + make_float3_scalar(-6.868f);
    v = c * v + make_float3_scalar(0.4298f);
    v = c * v + make_float3_scalar(0.1191f);
    v = c * v + make_float3_scalar(-0.0023f);

    // agx_mat_inv * v
    float3 result;
    result.x = 1.19687900512017f * v.x - 0.0980208811401368f * v.y - 0.0990297440797205f * v.z;
    result.y = -0.0528968517574562f * v.x + 1.15190312990417f * v.y - 0.0989611768448433f * v.z;
    result.z = -0.0529716355144438f * v.x - 0.0980434501171241f * v.y + 1.15107367264116f * v.z;

    return result;
}

//-----------------------------------------------------------------------------
// Khronos PBR neutral tonemapper
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 tonemapKhronosPBR(float3 color)
{
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = fminf(color.x, fminf(color.y, color.z));
    float peak = fmaxf(color.x, fmaxf(color.y, color.z));

    float offset = x < 0.08f ? x * (-6.25f * x + 1.0f) : 0.04f;
    color = color - make_float3_scalar(offset);

    if (peak >= startCompression) {
        const float d = 1.0f - startCompression;
        float newPeak = 1.0f - d * d / (peak + d - startCompression);
        color = color * (newPeak / peak);

        float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
        color = tonemap_mix(color, make_float3_scalar(newPeak), g);
    }
    return toSrgb(color);
}

//-----------------------------------------------------------------------------
// Apply tonemapping with all color corrections
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 applyTonemap(float3 color, float2 uv, const TonemapperParams& tm)
{
    // Exposure
    color = color * tm.exposure;

    float3 c;
    // Tonemap
    switch (tm.method) {
    case eTonemapFilmic:
        c = tonemapFilmic(color);
        break;
    case eTonemapUncharted2:
        c = tonemapUncharted2(color);
        break;
    case eTonemapClip:
        c = toSrgb(tonemap_clamp(color, 0.0f, 1.0f));
        break;
    case eTonemapACES:
        c = tonemapACES(color);
        break;
    case eTonemapAgX:
        c = tonemapAgX(color);
        break;
    case eTonemapKhronosPBR:
        c = tonemapKhronosPBR(color);
        break;
    default:
        c = tonemapFilmic(color);
        break;
    }

    // Contrast and clamp
    c = tonemap_clamp(tonemap_mix(make_float3_scalar(0.5f), c, tm.contrast), 0.0f, 1.0f);

    // Brightness
    c = tonemap_pow(c, make_float3_scalar(1.0f / tm.brightness));

    // Saturation
    float3 i = make_float3_scalar(tonemap_dot(c, make_float3(0.299f, 0.587f, 0.114f)));
    c = tonemap_mix(i, c, tm.saturation);

    // Vignette
    float2 center_uv = (uv - make_float2(0.5f, 0.5f)) * make_float2(2.0f, 2.0f);
    c = c * (1.0f - tonemap_dot2(center_uv, center_uv) * tm.vignette);

    return c;
}

//-----------------------------------------------------------------------------
// Tonemap kernel launch parameters
//-----------------------------------------------------------------------------
struct TonemapLaunchParams {
    cudaTextureObject_t inputTexture;  // Input HDR image as texture
    cudaSurfaceObject_t outputSurface;  // Output LDR image as surface
    unsigned int width;
    unsigned int height;
    TonemapperParams tonemapper;
};

//-----------------------------------------------------------------------------
// Tonemap CUDA kernel
// Note: This is a template - actual kernel should be defined in your main file
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void tonemapKernel(TonemapLaunchParams params)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= params.width || y >= params.height)
        return;

    float2 pixel_center = make_float2(float(x) + 0.5f, float(y) + 0.5f);
    float2 uv = pixel_center / make_float2(float(params.width), float(params.height));

    // Sample input texture
    float4 R = tex2D<float4>(params.inputTexture, uv.x, uv.y);

    if (params.tonemapper.isActive == 1)
    {
        float3 color = make_float3(R.x, R.y, R.z);
        color = applyTonemap(color, uv, params.tonemapper);
        R = make_float4(color.x, color.y, color.z, R.w);
    }

    // Write to output surface
    surf2Dwrite(R, params.outputSurface, x * sizeof(float4), y);
}
*/

#endif  // TONEMAP_H

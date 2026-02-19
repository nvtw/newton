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

// Shared types and constants for ray tracing shaders - CUDA/OptiX version
// Converted from Vulkan GLSL shared_types.glsl
// NOTE: This file is designed to be inlined into a single CUDA source.
//       OptiX device headers should be included before this file.

#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

// OptiX and CUDA types should already be available from optix_device.h

//-----------------------------------------------------------------------------
// Grid size for compute shaders
//-----------------------------------------------------------------------------
#define GRID_SIZE 16

//-----------------------------------------------------------------------------
// Light configuration
//-----------------------------------------------------------------------------
#define NB_LIGHTS 0

//-----------------------------------------------------------------------------
// Shader Binding Table offsets and payload locations
// We have two sets of shaders: primary (lightweight for finding primary surface)
// and secondary (for Monte Carlo path tracing)
//-----------------------------------------------------------------------------
#define PAYLOAD_PRIMARY     0
#define PAYLOAD_SECONDARY   1
#define SBTOFFSET_PRIMARY   0
#define SBTOFFSET_SECONDARY 1
#define MISSINDEX_PRIMARY   0
#define MISSINDEX_SECONDARY 1

//-----------------------------------------------------------------------------
// Descriptor set bindings (for reference - OptiX uses launch params instead)
//-----------------------------------------------------------------------------
// Scene bindings
#define eFrameInfo              0
#define eSceneDesc              1
#define eTextures               2
#define eInstanceTransforms     3
#define ePrevInstanceTransforms 4
#define ePickResult             5

// RTX bindings
#define eTlas 0

// Post-processing bindings
#define ePostImage 0

// Ray tracing output bindings
#define eViewZ              0
#define eMotionVectors      1
#define eNormal_Roughness   2
#define eBaseColor_Metallicity 3
#define eSpecAlbedo         4
#define eColor              5
#define eSpecHitDist        6

// TAA bindings
#define eInImage  0
#define eOutImage 1

// Environment bindings
#define eHdr        0
#define eImpSamples 1

// Sky bindings
#define eSkyParam 0

//-----------------------------------------------------------------------------
// Flags and utility macros
//-----------------------------------------------------------------------------
#define TEST_FLAG(flags, flag) (((flags) & (flag)) != 0)
#define BIT(x) (1 << (x))

#define FLAGS_ENVMAP_SKY              BIT(0)
#define FLAGS_USE_PSR                 BIT(1)
#define FLAGS_USE_PATH_REGULARIZATION BIT(2)

//-----------------------------------------------------------------------------
// DLSS infinite distance marker
// FP16 max value - must match what DLSS RR expects for "infinite" distance
//-----------------------------------------------------------------------------
#ifndef DLSS_INF_DISTANCE
#define DLSS_INF_DISTANCE 65504.0f
#endif

//-----------------------------------------------------------------------------
// Alpha modes
//-----------------------------------------------------------------------------
#define ALPHA_OPAQUE 0
#define ALPHA_MASK   1
#define ALPHA_BLEND  2

//-----------------------------------------------------------------------------
// Shared data structures
//-----------------------------------------------------------------------------

struct Light {
    float3 position;
    float intensity;
    float3 color;
    int type;
};

struct FrameInfo {
    float4x4 view;
    float4x4 proj;
    float4x4 viewInv;
    float4x4 projInv;
    float4x4 prevMVP;
    float4 envIntensity;
    float2 jitter;
    float envRotation;
    unsigned int flags;
    unsigned int frameIndex;
    unsigned int renderWidth;
    unsigned int renderHeight;
    float _padding;
#if NB_LIGHTS > 0
    Light light[NB_LIGHTS];
#endif
};

struct RtxPushConstant {
    int frame;
    float maxLuminance;
    unsigned int maxDepth;
    float meterToUnitsMultiplier;
    float overrideRoughness;
    float overrideMetallic;
    int2 mouseCoord;
    float bitangentFlip;
};

// Environment acceleration structure for importance sampling
// Matches C# MiniOptixScene EnvAccel struct exactly - only alias and q
// PDF is stored in the alpha channel of the environment map texture
struct EnvAccel {
    unsigned int alias;
    float q;
};

// Pick result - written at mouse cursor position
struct PickResult {
    int instanceId;  // Instance ID (-1 if miss)
    int primitiveId;  // Primitive ID
    float hitT;  // Hit distance
    unsigned int meshId;  // Mesh ID (RenderPrimitive index)
    float3 hitPos;  // World-space hit position
    float _pad;
};

//-----------------------------------------------------------------------------
// Matrix helper - CUDA uses row-major, need conversion utilities
//-----------------------------------------------------------------------------

// 3x4 transform matrix (row-major, as in OptiX)
struct TransformMatrix3x4 {
    float4 row0;  // [m00, m01, m02, tx]
    float4 row1;  // [m10, m11, m12, ty]
    float4 row2;  // [m20, m21, m22, tz]
};

// Helper to transform a point by a 3x4 matrix
static __forceinline__ __device__ float3 transformPoint(const TransformMatrix3x4& m, float3 p)
{
    return make_float3(
        m.row0.x * p.x + m.row0.y * p.y + m.row0.z * p.z + m.row0.w,
        m.row1.x * p.x + m.row1.y * p.y + m.row1.z * p.z + m.row1.w,
        m.row2.x * p.x + m.row2.y * p.y + m.row2.z * p.z + m.row2.w
    );
}

// Helper to transform a vector (no translation) by a 3x4 matrix
static __forceinline__ __device__ float3 transformVector(const TransformMatrix3x4& m, float3 v)
{
    return make_float3(
        m.row0.x * v.x + m.row0.y * v.y + m.row0.z * v.z, m.row1.x * v.x + m.row1.y * v.y + m.row1.z * v.z,
        m.row2.x * v.x + m.row2.y * v.y + m.row2.z * v.z
    );
}

#endif  // SHARED_TYPES_H

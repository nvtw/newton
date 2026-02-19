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

// Ray payload structures and common ray utilities - CUDA/OptiX version
// Converted from Vulkan GLSL ray_common.glsl
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include shared_types.h and func_common.h content before this file.

#ifndef RAY_COMMON_H
#define RAY_COMMON_H

// Fallback for DLSS_INF_DISTANCE if not defined
#ifndef DLSS_INF_DISTANCE
#define DLSS_INF_DISTANCE 65504.0f
#endif

// Useful for debugging results at individual pixels.
#define ATCURSOR(x, pixelPos, mouseCoord) \
    if(pixelPos.x == mouseCoord.x && pixelPos.y == mouseCoord.y) { x; }

//-----------------------------------------------------------------------------
// Ray payload structures
//-----------------------------------------------------------------------------

struct PayloadSecondary {
    unsigned int seed;
    float hitT;
    float3 contrib;  // Output: Radiance (times MIS factors) at this point.
    float3 weight;  // Output of closest-hit shader: BRDF sample weight of this bounce.
    float3 rayOrigin;  // Input and output.
    float3 rayDirection;  // Input and output.
    float bsdfPDF;  // Input and output: Probability that the BSDF sampling generated rayDirection.
    float2 maxRoughness;
};

struct PayloadPrimary {
    unsigned int renderNodeIndex;
    unsigned int renderPrimIndex;  // what mesh we hit
    float hitT;  // where we hit the mesh along the ray
    float3 tangent;
    float3 normal_envmapRadiance;  // when hitT == DLSS_INF_DISTANCE we hit the environment map and return its radiance
                                   // here
    float2 uv;
    float bitangentSign;
};

// RayPayload for rt shaders - geometry-only from hit shader, material evaluation in rgen
// Uses normal_envmapRadiance like PayloadPrimary (dual-purpose field)
struct RayPayload {
    float hitT;  // Hit distance; DLSS_INF_DISTANCE = miss
    float3 normal_envmapRadiance;  // Shading normal for hits, environment radiance for miss
    float3 tangent;  // Tangent for TBN matrix reconstruction
    float2 uv;  // Texture coordinates
    float2 uv1;  // Secondary texture coordinates
    unsigned int materialId;  // Material ID for lookup
    float bitangentSign;  // Sign for bitangent reconstruction
    int instanceId;  // Instance index for transform lookup (motion vectors)
    unsigned int meshId;  // Mesh index (gl_InstanceCustomIndexEXT) for RenderPrimitive lookup
    unsigned int primitiveId;  // Triangle index within mesh (for deformable motion vectors)
    float3 barycentrics;  // Barycentric coordinates of hit (for deformable motion vectors)
};

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------

// Build a mirror reflection matrix from a normal
static __forceinline__ __device__ void buildMirrorMatrix(float3 normal, float* matrix)
{
    // Returns a 3x3 matrix stored in row-major order (9 floats)
    // M = I - 2 * n * n^T
    matrix[0] = 1.0f - 2.0f * normal.x * normal.x;
    matrix[1] = -2.0f * normal.x * normal.y;
    matrix[2] = -2.0f * normal.x * normal.z;

    matrix[3] = -2.0f * normal.y * normal.x;
    matrix[4] = 1.0f - 2.0f * normal.y * normal.y;
    matrix[5] = -2.0f * normal.y * normal.z;

    matrix[6] = -2.0f * normal.z * normal.x;
    matrix[7] = -2.0f * normal.z * normal.y;
    matrix[8] = 1.0f - 2.0f * normal.z * normal.z;
}

// Apply a 3x3 matrix to a vector
static __forceinline__ __device__ float3 applyMatrix3x3(const float* matrix, float3 v)
{
    return make_float3(
        matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z, matrix[3] * v.x + matrix[4] * v.y + matrix[5] * v.z,
        matrix[6] * v.x + matrix[7] * v.y + matrix[8] * v.z
    );
}

// Multiply two 3x3 matrices: result = a * b
static __forceinline__ __device__ void multiplyMatrix3x3(const float* a, const float* b, float* result)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i * 3 + j] = a[i * 3 + 0] * b[0 * 3 + j] + a[i * 3 + 1] * b[1 * 3 + j] + a[i * 3 + 2] * b[2 * 3 + j];
        }
    }
}

// Initialize identity matrix
static __forceinline__ __device__ void identityMatrix3x3(float* matrix)
{
    matrix[0] = 1.0f;
    matrix[1] = 0.0f;
    matrix[2] = 0.0f;
    matrix[3] = 0.0f;
    matrix[4] = 1.0f;
    matrix[5] = 0.0f;
    matrix[6] = 0.0f;
    matrix[7] = 0.0f;
    matrix[8] = 1.0f;
}

// Reinhard max tonemapping (preserves [0,1] output)
static __forceinline__ __device__ float3 reinhardMax(float3 color)
{
    float lum = fmaxf(1e-7f, fmaxf(fmaxf(color.x, color.y), color.z));
    float reinhard = lum / (lum + 1.0f);
    return color * (reinhard / lum);
}

#endif  // RAY_COMMON_H

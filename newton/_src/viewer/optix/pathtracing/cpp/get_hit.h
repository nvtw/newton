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

// Hit state computation - CUDA/OptiX version
// Converted from Vulkan GLSL get_hit.glsl
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include scene_common.h content before this file.
//       OptiX device headers should be included before this file.

#ifndef GET_HIT_H
#define GET_HIT_H

//-----------------------------------------------------------------------
// Hit state information
//-----------------------------------------------------------------------
struct HitState {
    float3 pos;
    float3 nrm;
    float3 geonrm;
    float2 uv;
    float3 tangent;
    float3 bitangent;
    float bitangentSign;
};

//--------------------------------------------------------------
// Flipping Back-face
// Note: In OptiX, use optixGetWorldRayDirection() instead of gl_WorldRayDirectionEXT
//--------------------------------------------------------------
static __forceinline__ __device__ float3 adjustShadingNormalToRayDir(float3& N, float3& G, float3 rayDir)
{
    const float3 V = -rayDir;

    if (dot(G, V) < 0)  // Flip if back facing
        G = -G;

    if (dot(G, N) < 0)  // Make Normal and GeoNormal on the same side
        N = -N;

    return N;
}

//-----------------------------------------------------------------------
// Get texture coordinates for a triangle
//-----------------------------------------------------------------------
static __forceinline__ __device__ void getTexCoords0(const RenderPrimitive& renderPrim, uint3 idx, float2 uv[3])
{
    if (!hasVertexTexCoord0(renderPrim)) {
        uv[0] = make_float2(0.0f, 0.0f);
        uv[1] = make_float2(0.0f, 0.0f);
        uv[2] = make_float2(0.0f, 0.0f);
        return;
    }

    const float2* texcoords = reinterpret_cast<const float2*>(renderPrim.vertexBuffer.texCoord0Address);
    uv[0] = texcoords[idx.x];
    uv[1] = texcoords[idx.y];
    uv[2] = texcoords[idx.z];
}

static __forceinline__ __device__ float2
getInterpolatedVertexTexCoords(const RenderPrimitive& renderPrim, uint3 idx, float3 barycentrics)
{
    if (!hasVertexTexCoord0(renderPrim))
        return make_float2(0.0f, 0.0f);

    float2 uv[3];
    getTexCoords0(renderPrim, idx, uv);

    return uv[0] * barycentrics.x + uv[1] * barycentrics.y + uv[2] * barycentrics.z;
}

//-----------------------------------------------------------------------
// Compute tangent space from UV coordinates
//-----------------------------------------------------------------------
static __forceinline__ __device__ void
computeTangentSpace(const RenderPrimitive& renderPrim, uint3 idx, HitState& hit, const float* worldToObject)
{
    float2 uv[3];
    getTexCoords0(renderPrim, idx, uv);

    float2 u = uv[1] - uv[0];
    float2 v = uv[2] - uv[0];

    float d = u.x * v.y - u.y * v.x;
    if (d == 0.0f) {
        float4 t = makeFastTangent(hit.nrm);
        hit.tangent = make_float3(t.x, t.y, t.z);
        hit.bitangent = cross(hit.nrm, hit.tangent) * t.w;
        hit.bitangentSign = t.w;
    } else {
        u = u / d;
        v = v / d;

        float3 v0 = getVertexPosition(renderPrim, idx.x);
        float3 v1 = getVertexPosition(renderPrim, idx.y);
        float3 v2 = getVertexPosition(renderPrim, idx.z);

        float3 p = v1 - v0;
        float3 q = v2 - v0;

        float3 t;
        t.x = v.y * p.x - u.y * q.x;
        t.y = v.y * p.y - u.y * q.y;
        t.z = v.y * p.z - u.y * q.z;

        // Transform to world space using worldToObject (3x4 matrix, row-major)
        // t = t * worldToObject (vector * matrix)
        float3 t_world = make_float3(
            t.x * worldToObject[0] + t.y * worldToObject[4] + t.z * worldToObject[8],
            t.x * worldToObject[1] + t.y * worldToObject[5] + t.z * worldToObject[9],
            t.x * worldToObject[2] + t.y * worldToObject[6] + t.z * worldToObject[10]
        );

        float3 b;
        b.x = u.x * q.x - v.x * p.x;
        b.y = u.x * q.y - v.x * p.y;
        b.z = u.x * q.z - v.x * p.z;

        float3 b_world = make_float3(
            b.x * worldToObject[0] + b.y * worldToObject[4] + b.z * worldToObject[8],
            b.x * worldToObject[1] + b.y * worldToObject[5] + b.z * worldToObject[9],
            b.x * worldToObject[2] + b.y * worldToObject[6] + b.z * worldToObject[10]
        );

        // Orthogonalize T and B to N
        t_world = t_world - hit.nrm * dot(t_world, hit.nrm);
        b_world = b_world - hit.nrm * dot(b_world, hit.nrm);

        hit.tangent = normalize(t_world);
        hit.bitangent = normalize(b_world);

        hit.bitangentSign = dot(cross(hit.nrm, hit.tangent), hit.bitangent) > 0 ? -1.0f : 1.0f;
    }
}

//-----------------------------------------------------------------------
// Get hit state from OptiX built-in functions
// Note: This function should be called from closest-hit or any-hit shaders
//-----------------------------------------------------------------------
static __forceinline__ __device__ HitState GetHitState(
    const RenderPrimitive& renderPrim,
    float bitangentFlip,
    float2 attribs,
    unsigned int primitiveIndex,
    const float* objectToWorld,  // 3x4 matrix, 12 floats
    const float* worldToObject,  // 3x4 matrix, 12 floats
    float3 rayDirection
)
{
    HitState hit;

    // Barycentric coordinate on the triangle
    float3 barycentrics = make_float3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);

    // Getting the 3 indices of the triangle (local)
    uint3 triangleIndex = getTriangleIndices(renderPrim, primitiveIndex);

    // Position
    const float3 pos0 = getVertexPosition(renderPrim, triangleIndex.x);
    const float3 pos1 = getVertexPosition(renderPrim, triangleIndex.y);
    const float3 pos2 = getVertexPosition(renderPrim, triangleIndex.z);
    const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;

    // Transform position to world space: objectToWorld * position
    hit.pos = make_float3(
        objectToWorld[0] * position.x + objectToWorld[1] * position.y + objectToWorld[2] * position.z
            + objectToWorld[3],
        objectToWorld[4] * position.x + objectToWorld[5] * position.y + objectToWorld[6] * position.z
            + objectToWorld[7],
        objectToWorld[8] * position.x + objectToWorld[9] * position.y + objectToWorld[10] * position.z
            + objectToWorld[11]
    );

    // Geometric normal
    const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));

    // Transform normal to world space: geoNormal * worldToObject (for normals)
    float3 worldGeoNormal = make_float3(
        geoNormal.x * worldToObject[0] + geoNormal.y * worldToObject[4] + geoNormal.z * worldToObject[8],
        geoNormal.x * worldToObject[1] + geoNormal.y * worldToObject[5] + geoNormal.z * worldToObject[9],
        geoNormal.x * worldToObject[2] + geoNormal.y * worldToObject[6] + geoNormal.z * worldToObject[10]
    );
    worldGeoNormal = normalize(worldGeoNormal);
    hit.geonrm = worldGeoNormal;

    hit.nrm = worldGeoNormal;
    if (hasVertexNormal(renderPrim)) {
        const float3 normal = getInterpolatedVertexNormal(renderPrim, triangleIndex, barycentrics);
        float3 worldNormal = make_float3(
            normal.x * worldToObject[0] + normal.y * worldToObject[4] + normal.z * worldToObject[8],
            normal.x * worldToObject[1] + normal.y * worldToObject[5] + normal.z * worldToObject[9],
            normal.x * worldToObject[2] + normal.y * worldToObject[6] + normal.z * worldToObject[10]
        );
        worldNormal = normalize(worldNormal);
        adjustShadingNormalToRayDir(worldNormal, worldGeoNormal, rayDirection);
        hit.nrm = worldNormal;
    }

    // TexCoord
    hit.uv = getInterpolatedVertexTexCoords(renderPrim, triangleIndex, barycentrics);

    // Tangent - Bitangent
    if (hasVertexTangent(renderPrim)) {
        float4 tng[3];
        tng[0] = getVertexTangent(renderPrim, triangleIndex.x);
        tng[1] = getVertexTangent(renderPrim, triangleIndex.y);
        tng[2] = getVertexTangent(renderPrim, triangleIndex.z);

        float4 interpolatedTangent = tng[0] * barycentrics.x + tng[1] * barycentrics.y + tng[2] * barycentrics.z;
        hit.tangent = normalize(make_float3(interpolatedTangent.x, interpolatedTangent.y, interpolatedTangent.z));

        // Transform to world space
        float3 tangent_world = make_float3(
            hit.tangent.x * worldToObject[0] + hit.tangent.y * worldToObject[4] + hit.tangent.z * worldToObject[8],
            hit.tangent.x * worldToObject[1] + hit.tangent.y * worldToObject[5] + hit.tangent.z * worldToObject[9],
            hit.tangent.x * worldToObject[2] + hit.tangent.y * worldToObject[6] + hit.tangent.z * worldToObject[10]
        );

        // Orthogonalize to N and normalize
        hit.tangent = normalize(tangent_world - hit.nrm * dot(hit.nrm, tangent_world));
        hit.bitangent = cross(hit.nrm, hit.tangent) * tng[0].w;
        hit.bitangentSign = tng[0].w;
    } else {
        computeTangentSpace(renderPrim, triangleIndex, hit, worldToObject);
    }

    hit.bitangentSign *= bitangentFlip;
    hit.bitangent = hit.bitangent * bitangentFlip;

    return hit;
}

//-----------------------------------------------------------------------
// Simplified version using OptiX built-in functions
// Call this from closest-hit shader
//-----------------------------------------------------------------------
static __forceinline__ __device__ HitState GetHitStateOptiX(const RenderPrimitive& renderPrim, float bitangentFlip)
{
    // Get OptiX built-in values
    float2 attribs = optixGetTriangleBarycentrics();
    unsigned int primitiveIndex = optixGetPrimitiveIndex();

    // Get transform matrices (OptiX provides these as row-major 3x4)
    float objectToWorld[12];
    float worldToObject[12];
    optixGetObjectToWorldTransformMatrix(objectToWorld);
    optixGetWorldToObjectTransformMatrix(worldToObject);

    float3 rayDirection = optixGetWorldRayDirection();

    return GetHitState(renderPrim, bitangentFlip, attribs, primitiveIndex, objectToWorld, worldToObject, rayDirection);
}

#endif  // GET_HIT_H

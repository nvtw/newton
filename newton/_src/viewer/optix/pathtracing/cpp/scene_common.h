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

// Shared scene description and vertex accessor code - CUDA/OptiX version
// Converted from Vulkan GLSL scene_common.glsl
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include func_common.h content before this file.

#ifndef SCENE_COMMON_H
#define SCENE_COMMON_H

//-----------------------------------------------------------------------------
// Scene data structures (inlined from nvvkhl/shaders/dh_scn_desc.h)
//-----------------------------------------------------------------------------
#define DH_SCN_DESC_H 1

struct RenderNode {
    float4x4 objectToWorld;
    float4x4 worldToObject;
    int materialID;
    int renderPrimID;
};

struct VertexBuffers {
    unsigned long long positionAddress;
    unsigned long long normalAddress;
    unsigned long long colorAddress;
    unsigned long long tangentAddress;
    unsigned long long texCoord0Address;
    unsigned long long texCoord1Address;
    unsigned long long prevPositionAddress;  // Previous frame positions for deformable meshes (0 for rigid)
};

struct RenderPrimitive {
    unsigned long long indexAddress;
    unsigned long long materialIdAddress;  // Per-mesh material ID buffer
    VertexBuffers vertexBuffer;
};

struct SceneDescription {
    unsigned long long materialAddress;
    unsigned long long renderNodeAddress;
    unsigned long long renderPrimitiveAddress;
    unsigned long long lightAddress;
    int numLights;
};

//-----------------------------------------------------------------------------
// Texture info structure
//-----------------------------------------------------------------------------
struct GltfTextureInfo {
    float2x3 uvTransform;  // 2x3 matrix for UV transform
    int index;
    int texCoord;
};

// Helper to create identity UV transform
static __forceinline__ __device__ void identityUvTransform(float* uvTransform)
{
    // 2x3 matrix: row-major
    // [1, 0, 0]
    // [0, 1, 0]
    uvTransform[0] = 1.0f;
    uvTransform[1] = 0.0f;
    uvTransform[2] = 0.0f;
    uvTransform[3] = 0.0f;
    uvTransform[4] = 1.0f;
    uvTransform[5] = 0.0f;
}

//-----------------------------------------------------------------------------
// Material structure (glTF PBR)
//-----------------------------------------------------------------------------
struct GltfShadeMaterial {
    float4 pbrBaseColorFactor;
    float3 emissiveFactor;
    float normalTextureScale;
    float pbrRoughnessFactor;
    float pbrMetallicFactor;
    int alphaMode;
    float alphaCutoff;
    float transmissionFactor;
    float ior;
    float3 attenuationColor;
    float thicknessFactor;
    float attenuationDistance;
    float clearcoatFactor;
    float clearcoatRoughness;
    float3 specularColorFactor;
    float specularFactor;
    int unlit;
    float iridescenceFactor;
    float iridescenceThicknessMaximum;
    float iridescenceThicknessMinimum;
    float iridescenceIor;
    float anisotropyStrength;
    float2 anisotropyRotation;
    float sheenRoughnessFactor;
    float3 sheenColorFactor;
    float occlusionStrength;
    float dispersion;
    float4 pbrDiffuseFactor;
    float3 pbrSpecularFactor;
    int usePbrSpecularGlossiness;
    float pbrGlossinessFactor;
    float3 diffuseTransmissionColor;
    float diffuseTransmissionFactor;
    int pad;

    GltfTextureInfo pbrBaseColorTexture;
    GltfTextureInfo normalTexture;
    GltfTextureInfo pbrMetallicRoughnessTexture;
    GltfTextureInfo emissiveTexture;
    GltfTextureInfo transmissionTexture;
    GltfTextureInfo thicknessTexture;
    GltfTextureInfo clearcoatTexture;
    GltfTextureInfo clearcoatRoughnessTexture;
    GltfTextureInfo clearcoatNormalTexture;
    GltfTextureInfo specularTexture;
    GltfTextureInfo specularColorTexture;
    GltfTextureInfo iridescenceTexture;
    GltfTextureInfo iridescenceThicknessTexture;
    GltfTextureInfo anisotropyTexture;
    GltfTextureInfo sheenColorTexture;
    GltfTextureInfo sheenRoughnessTexture;
    GltfTextureInfo occlusionTexture;
    GltfTextureInfo pbrDiffuseTexture;
    GltfTextureInfo pbrSpecularGlossinessTexture;
    GltfTextureInfo diffuseTransmissionTexture;
    GltfTextureInfo diffuseTransmissionColorTexture;
};

//-----------------------------------------------------------------------------
// Vertex accessor functions (inlined from nvvkhl/shaders/vertex_accessor.h)
// In CUDA/OptiX, we use raw pointers instead of buffer references
//-----------------------------------------------------------------------------
#define VERTEX_ACCESSOR_H 1

static __forceinline__ __device__ uint3 getTriangleIndices(const RenderPrimitive& renderPrim, unsigned int idx)
{
    const uint3* indices = reinterpret_cast<const uint3*>(renderPrim.indexAddress);
    return indices[idx];
}

static __forceinline__ __device__ float3 getVertexPosition(const RenderPrimitive& renderPrim, unsigned int idx)
{
    const float3* positions = reinterpret_cast<const float3*>(renderPrim.vertexBuffer.positionAddress);
    return positions[idx];
}

static __forceinline__ __device__ float3
getInterpolatedVertexPosition(const RenderPrimitive& renderPrim, uint3 idx, float3 barycentrics)
{
    const float3* positions = reinterpret_cast<const float3*>(renderPrim.vertexBuffer.positionAddress);
    float3 pos[3];
    pos[0] = positions[idx.x];
    pos[1] = positions[idx.y];
    pos[2] = positions[idx.z];
    return pos[0] * barycentrics.x + pos[1] * barycentrics.y + pos[2] * barycentrics.z;
}

static __forceinline__ __device__ bool hasVertexPrevPosition(const RenderPrimitive& renderPrim)
{
    return renderPrim.vertexBuffer.prevPositionAddress != 0;
}

static __forceinline__ __device__ float3 getVertexPrevPosition(const RenderPrimitive& renderPrim, unsigned int idx)
{
    if (!hasVertexPrevPosition(renderPrim))
        return getVertexPosition(renderPrim, idx);
    const float3* positions = reinterpret_cast<const float3*>(renderPrim.vertexBuffer.prevPositionAddress);
    return positions[idx];
}

static __forceinline__ __device__ float3
getInterpolatedVertexPrevPosition(const RenderPrimitive& renderPrim, uint3 idx, float3 barycentrics)
{
    if (!hasVertexPrevPosition(renderPrim))
        return getInterpolatedVertexPosition(renderPrim, idx, barycentrics);
    const float3* positions = reinterpret_cast<const float3*>(renderPrim.vertexBuffer.prevPositionAddress);
    float3 pos[3];
    pos[0] = positions[idx.x];
    pos[1] = positions[idx.y];
    pos[2] = positions[idx.z];
    return pos[0] * barycentrics.x + pos[1] * barycentrics.y + pos[2] * barycentrics.z;
}

static __forceinline__ __device__ bool hasVertexNormal(const RenderPrimitive& renderPrim)
{
    return renderPrim.vertexBuffer.normalAddress != 0;
}

static __forceinline__ __device__ float3 getVertexNormal(const RenderPrimitive& renderPrim, unsigned int idx)
{
    if (!hasVertexNormal(renderPrim))
        return make_float3(0.0f, 0.0f, 1.0f);
    const float3* normals = reinterpret_cast<const float3*>(renderPrim.vertexBuffer.normalAddress);
    return normals[idx];
}

static __forceinline__ __device__ float3
getInterpolatedVertexNormal(const RenderPrimitive& renderPrim, uint3 idx, float3 barycentrics)
{
    if (!hasVertexNormal(renderPrim))
        return make_float3(0.0f, 0.0f, 1.0f);
    const float3* normals = reinterpret_cast<const float3*>(renderPrim.vertexBuffer.normalAddress);
    float3 nrm[3];
    nrm[0] = normals[idx.x];
    nrm[1] = normals[idx.y];
    nrm[2] = normals[idx.z];
    return nrm[0] * barycentrics.x + nrm[1] * barycentrics.y + nrm[2] * barycentrics.z;
}

static __forceinline__ __device__ bool hasVertexTexCoord0(const RenderPrimitive& renderPrim)
{
    return renderPrim.vertexBuffer.texCoord0Address != 0;
}

static __forceinline__ __device__ float2 getVertexTexCoord0(const RenderPrimitive& renderPrim, unsigned int idx)
{
    if (!hasVertexTexCoord0(renderPrim))
        return make_float2(0.0f, 0.0f);
    const float2* texcoords = reinterpret_cast<const float2*>(renderPrim.vertexBuffer.texCoord0Address);
    return texcoords[idx];
}

static __forceinline__ __device__ float2
getInterpolatedVertexTexCoord0(const RenderPrimitive& renderPrim, uint3 idx, float3 barycentrics)
{
    if (!hasVertexTexCoord0(renderPrim))
        return make_float2(0.0f, 0.0f);
    const float2* texcoords = reinterpret_cast<const float2*>(renderPrim.vertexBuffer.texCoord0Address);
    float2 uv[3];
    uv[0] = texcoords[idx.x];
    uv[1] = texcoords[idx.y];
    uv[2] = texcoords[idx.z];
    return uv[0] * barycentrics.x + uv[1] * barycentrics.y + uv[2] * barycentrics.z;
}

static __forceinline__ __device__ bool hasVertexTexCoord1(const RenderPrimitive& renderPrim)
{
    return renderPrim.vertexBuffer.texCoord1Address != 0;
}

static __forceinline__ __device__ float2 getVertexTexCoord1(const RenderPrimitive& renderPrim, unsigned int idx)
{
    if (!hasVertexTexCoord1(renderPrim))
        return make_float2(0.0f, 0.0f);
    const float2* texcoords = reinterpret_cast<const float2*>(renderPrim.vertexBuffer.texCoord1Address);
    return texcoords[idx];
}

static __forceinline__ __device__ float2
getInterpolatedVertexTexCoord1(const RenderPrimitive& renderPrim, uint3 idx, float3 barycentrics)
{
    if (!hasVertexTexCoord1(renderPrim))
        return make_float2(0.0f, 0.0f);
    const float2* texcoords = reinterpret_cast<const float2*>(renderPrim.vertexBuffer.texCoord1Address);
    float2 uv[3];
    uv[0] = texcoords[idx.x];
    uv[1] = texcoords[idx.y];
    uv[2] = texcoords[idx.z];
    return uv[0] * barycentrics.x + uv[1] * barycentrics.y + uv[2] * barycentrics.z;
}

static __forceinline__ __device__ bool hasVertexTangent(const RenderPrimitive& renderPrim)
{
    return renderPrim.vertexBuffer.tangentAddress != 0;
}

static __forceinline__ __device__ float4 getVertexTangent(const RenderPrimitive& renderPrim, unsigned int idx)
{
    if (!hasVertexTangent(renderPrim))
        return make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    const float4* tangents = reinterpret_cast<const float4*>(renderPrim.vertexBuffer.tangentAddress);
    return tangents[idx];
}

static __forceinline__ __device__ float4
getInterpolatedVertexTangent(const RenderPrimitive& renderPrim, uint3 idx, float3 barycentrics)
{
    if (!hasVertexTangent(renderPrim))
        return make_float4(1.0f, 0.0f, 0.0f, 1.0f);

    const float4* tangents = reinterpret_cast<const float4*>(renderPrim.vertexBuffer.tangentAddress);
    float4 tng[3];
    tng[0] = tangents[idx.x];
    tng[1] = tangents[idx.y];
    tng[2] = tangents[idx.z];
    return tng[0] * barycentrics.x + tng[1] * barycentrics.y + tng[2] * barycentrics.z;
}

static __forceinline__ __device__ bool hasVertexColor(const RenderPrimitive& renderPrim)
{
    return renderPrim.vertexBuffer.colorAddress != 0;
}

// Unpack RGBA8 to float4
static __forceinline__ __device__ float4 unpackUnorm4x8(unsigned int packed)
{
    return make_float4(
        (packed & 0xFF) / 255.0f, ((packed >> 8) & 0xFF) / 255.0f, ((packed >> 16) & 0xFF) / 255.0f,
        ((packed >> 24) & 0xFF) / 255.0f
    );
}

static __forceinline__ __device__ float4 getVertexColor(const RenderPrimitive& renderPrim, unsigned int idx)
{
    if (!hasVertexColor(renderPrim))
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const unsigned int* colors = reinterpret_cast<const unsigned int*>(renderPrim.vertexBuffer.colorAddress);
    return unpackUnorm4x8(colors[idx]);
}

static __forceinline__ __device__ float4
getInterpolatedVertexColor(const RenderPrimitive& renderPrim, uint3 idx, float3 barycentrics)
{
    if (!hasVertexColor(renderPrim))
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    const unsigned int* colors = reinterpret_cast<const unsigned int*>(renderPrim.vertexBuffer.colorAddress);
    float4 col[3];
    col[0] = unpackUnorm4x8(colors[idx.x]);
    col[1] = unpackUnorm4x8(colors[idx.y]);
    col[2] = unpackUnorm4x8(colors[idx.z]);
    return col[0] * barycentrics.x + col[1] * barycentrics.y + col[2] * barycentrics.z;
}

#endif  // SCENE_COMMON_H

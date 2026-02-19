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

// Launch parameters for OptiX path tracing
// This file defines the data structures passed from Python to OptiX shaders

#ifndef LAUNCH_PARAMS_H
#define LAUNCH_PARAMS_H

#include <optix.h>

//-----------------------------------------------------------------------------
// Output buffer modes for debug visualization
//-----------------------------------------------------------------------------
#define OUTPUT_MODE_FINAL       0
#define OUTPUT_MODE_RADIANCE    1
#define OUTPUT_MODE_DEPTH       2
#define OUTPUT_MODE_MOTION      3
#define OUTPUT_MODE_NORMAL      4
#define OUTPUT_MODE_ROUGHNESS   5
#define OUTPUT_MODE_DIFFUSE     6
#define OUTPUT_MODE_SPECULAR    7

//-----------------------------------------------------------------------------
// Feature flags
//-----------------------------------------------------------------------------
#define FLAGS_ENVMAP_SKY              (1 << 0)
#define FLAGS_USE_PSR                 (1 << 1)
#define FLAGS_USE_PATH_REGULARIZATION (1 << 2)

//-----------------------------------------------------------------------------
// SBT offsets and payload indices
//-----------------------------------------------------------------------------
#define PAYLOAD_PRIMARY     0
#define PAYLOAD_SECONDARY   1
#define SBTOFFSET_PRIMARY   0
#define SBTOFFSET_SECONDARY 1
#define MISSINDEX_PRIMARY   0
#define MISSINDEX_SECONDARY 1

//-----------------------------------------------------------------------------
// Alpha modes
//-----------------------------------------------------------------------------
#define ALPHA_OPAQUE 0
#define ALPHA_MASK   1
#define ALPHA_BLEND  2

//-----------------------------------------------------------------------------
// Texture info structure (32 bytes)
//-----------------------------------------------------------------------------
struct GpuTextureInfo {
    // UV transform (mat2x3 stored as 6 floats)
    float uvTransform00, uvTransform01, uvTransform02;  // Column 0
    float uvTransform10, uvTransform11, uvTransform12;  // Column 1
    int index;  // Texture index (-1 if no texture)
    int texCoord;  // Which UV set to use (0 or 1)
};

//-----------------------------------------------------------------------------
// GPU Material structure - matches GltfShadeMaterial
//-----------------------------------------------------------------------------
struct GpuMaterial {
    // Base PBR properties
    float4 pbrBaseColorFactor;
    float3 emissiveFactor;
    float normalTextureScale;
    float pbrRoughnessFactor;
    float pbrMetallicFactor;
    int alphaMode;
    float alphaCutoff;

    // Transmission/Volume
    float transmissionFactor;
    float ior;
    float3 attenuationColor;
    float thicknessFactor;
    float attenuationDistance;

    // Clearcoat
    float clearcoatFactor;
    float clearcoatRoughness;

    // Specular
    float3 specularColorFactor;
    float specularFactor;

    // Misc
    int unlit;

    // Iridescence
    float iridescenceFactor;
    float iridescenceThicknessMaximum;
    float iridescenceThicknessMinimum;
    float iridescenceIor;

    // Anisotropy
    float anisotropyStrength;
    float2 anisotropyRotation;

    // Sheen
    float sheenRoughnessFactor;
    float3 sheenColorFactor;

    // Occlusion & Dispersion
    float occlusionStrength;
    float dispersion;

    // Specular-Glossiness workflow
    float4 pbrDiffuseFactor;
    float3 pbrSpecularFactor;
    int usePbrSpecularGlossiness;
    float pbrGlossinessFactor;

    // Diffuse transmission
    float3 diffuseTransmissionColor;
    float diffuseTransmissionFactor;

    // Padding
    int pad;

    // Texture infos (21 textures)
    GpuTextureInfo pbrBaseColorTexture;
    GpuTextureInfo normalTexture;
    GpuTextureInfo pbrMetallicRoughnessTexture;
    GpuTextureInfo emissiveTexture;
    GpuTextureInfo transmissionTexture;
    GpuTextureInfo thicknessTexture;
    GpuTextureInfo clearcoatTexture;
    GpuTextureInfo clearcoatRoughnessTexture;
    GpuTextureInfo clearcoatNormalTexture;
    GpuTextureInfo specularTexture;
    GpuTextureInfo specularColorTexture;
    GpuTextureInfo iridescenceTexture;
    GpuTextureInfo iridescenceThicknessTexture;
    GpuTextureInfo anisotropyTexture;
    GpuTextureInfo sheenColorTexture;
    GpuTextureInfo sheenRoughnessTexture;
    GpuTextureInfo occlusionTexture;
    GpuTextureInfo pbrDiffuseTexture;
    GpuTextureInfo pbrSpecularGlossinessTexture;
    GpuTextureInfo diffuseTransmissionTexture;
    GpuTextureInfo diffuseTransmissionColorTexture;
};

//-----------------------------------------------------------------------------
// Vertex buffer addresses
//-----------------------------------------------------------------------------
struct VertexBuffers {
    unsigned long long positionAddress;
    unsigned long long normalAddress;
    unsigned long long colorAddress;
    unsigned long long tangentAddress;
    unsigned long long texCoord0Address;
    unsigned long long texCoord1Address;
    unsigned long long prevPositionAddress;
};

//-----------------------------------------------------------------------------
// Per-mesh render primitive
//-----------------------------------------------------------------------------
struct RenderPrimitive {
    unsigned long long indexAddress;
    unsigned long long materialIdAddress;
    VertexBuffers vertexBuffer;
};

//-----------------------------------------------------------------------------
// Per-instance render node
//-----------------------------------------------------------------------------
struct RenderNode {
    float4x4 objectToWorld;
    float4x4 worldToObject;
    int materialID;
    int renderPrimID;
};

//-----------------------------------------------------------------------------
// Scene description
//-----------------------------------------------------------------------------
struct SceneDescription {
    unsigned long long materialAddress;
    unsigned long long renderNodeAddress;
    unsigned long long renderPrimitiveAddress;
    unsigned long long lightAddress;
    int numLights;
    int _padding;
};

//-----------------------------------------------------------------------------
// Light structure
//-----------------------------------------------------------------------------
struct Light {
    float3 position;
    float intensity;
    float3 color;
    int type;  // 0=point, 1=directional, 2=spot
};

//-----------------------------------------------------------------------------
// Frame info - camera and rendering parameters
//-----------------------------------------------------------------------------
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
};

//-----------------------------------------------------------------------------
// Push constants for ray tracing
//-----------------------------------------------------------------------------
struct RtxPushConstant {
    int frame;
    float maxLuminance;
    unsigned int maxDepth;
    float meterToUnitsMultiplier;
    float overrideRoughness;
    float overrideMetallic;
    int mouseCoordX;
    int mouseCoordY;
    float bitangentFlip;
};

//-----------------------------------------------------------------------------
// Environment acceleration for importance sampling
//-----------------------------------------------------------------------------
struct EnvAccel {
    unsigned int alias;
    float q;
    float pdf;
    float _padding;
};

//-----------------------------------------------------------------------------
// Physical sky parameters
//-----------------------------------------------------------------------------
struct PhysicalSkyParameters {
    float3 sunDirection;
    float multiplier;
    float3 rgbUnitConversion;
    float turbidity;
    float3 groundAlbedo;
    int yIsUp;
    float sunAngularRadius;
    float sunIntensity;
    float _pad0;
    float _pad1;
};

//-----------------------------------------------------------------------------
// Main launch parameters - passed to all OptiX programs
//-----------------------------------------------------------------------------
struct LaunchParams {
    // Acceleration structure
    OptixTraversableHandle tlas;

    // Frame info and scene description
    FrameInfo frameInfo;
    SceneDescription sceneDesc;
    RtxPushConstant pushConstants;

    // Output surfaces (write via surf2Dwrite)
    cudaSurfaceObject_t colorOutput;  // Final radiance (HDR)
    cudaSurfaceObject_t normalRoughness;  // Normal (xyz) + roughness (w)
    cudaSurfaceObject_t motionVectors;  // Motion vectors (xy)
    cudaSurfaceObject_t linearDepth;  // Linear depth (view-space Z)
    cudaSurfaceObject_t diffuseAlbedo;  // Diffuse albedo (rgb) + metallicity (a)
    cudaSurfaceObject_t specularAlbedo;  // Specular albedo
    cudaSurfaceObject_t specHitDist;  // Specular hit distance

    // Environment map (optional)
    cudaTextureObject_t envMapTexture;
    unsigned long long envAccelAddress;  // EnvAccel[] for importance sampling
    int envMapWidth;
    int envMapHeight;
    int hasEnvMap;

    // Physical sky parameters
    PhysicalSkyParameters skyParams;
    int useProceduralSky;

    // Scene textures (array of texture objects)
    cudaTextureObject_t* textures;
    int numTextures;

    // Debug output mode
    int outputMode;

    // Render dimensions
    unsigned int width;
    unsigned int height;
};

// Global launch params (defined in raygen shader)
extern "C" __constant__ LaunchParams params;

#endif  // LAUNCH_PARAMS_H

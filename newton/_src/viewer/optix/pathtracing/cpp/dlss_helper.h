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

// DLSS helper functions - CUDA/OptiX version
// Converted from Vulkan GLSL dlss_helper.glsl
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include func_common.h content before this file.

#ifndef DLSS_HELPER_H
#define DLSS_HELPER_H

// DLSS_INF_DISTANCE is defined in shared_types.h, provide fallback for standalone use
#ifndef DLSS_INF_DISTANCE
#define DLSS_INF_DISTANCE 65504.0f  // FP16 max number
#endif

#define FLT_MIN_DLSS 1e-15f

static __forceinline__ __device__ float PositiveRcp(float x) { return 1.0f / fmaxf(x, FLT_MIN_DLSS); }

// "Ray Tracing Gems", Chapter 32, Equation 4 - the approximation assumes GGX VNDF and Schlick's approximation
static __forceinline__ __device__ float3 EnvironmentTerm_Rtg(float3 Rf0, float NoV, float alphaRoughness)
{
    float4 X;
    X.x = 1.0f;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;

    float4 Y;
    Y.x = 1.0f;
    Y.y = alphaRoughness;
    Y.z = alphaRoughness * alphaRoughness;
    Y.w = alphaRoughness * Y.z;

    // GLSL mat2/mat3 constructors fill column-major:
    //   mat2(a,b,c,d)       -> col0=(a,b), col1=(c,d)
    //   mat3(a,b,c,d,e,f,g,h,i) -> col0=(a,b,c), col1=(d,e,f), col2=(g,h,i)
    // mul(M, v) = M * v  (column-major matrix times column vector)
    //
    // M1 = mat2(0.99044, -1.28514, 1.29678, -0.755907)
    //   col0=(0.99044, -1.28514)  col1=(1.29678, -0.755907)
    // M2 = mat3(1.0, 2.92338, 59.4188, 20.3225, -27.0302, 222.592, 121.563, 626.13, 316.627)
    //   col0=(1.0, 2.92338, 59.4188)  col1=(20.3225, -27.0302, 222.592)  col2=(121.563, 626.13, 316.627)
    // M3 = mat2(0.0365463, 3.32707, 9.0632, -9.04756)
    //   col0=(0.0365463, 3.32707)  col1=(9.0632, -9.04756)
    // M4 = mat3(1.0, 3.59685, -1.36772, 9.04401, -16.3174, 9.22949, 5.56589, 19.7886, -20.2123)
    //   col0=(1.0, 3.59685, -1.36772)  col1=(9.04401, -16.3174, 9.22949)  col2=(5.56589, 19.7886, -20.2123)

    // M1 * X.xy: result[i] = col0[i]*X.x + col1[i]*X.y
    float2 M1_X;
    M1_X.x =  0.99044f * X.x +  1.29678f * X.y;
    M1_X.y = -1.28514f * X.x + (-0.755907f) * X.y;

    // M2 * X.xyw: result[i] = col0[i]*X.x + col1[i]*X.y + col2[i]*X.w
    float3 M2_X;
    M2_X.x =  1.0f     * X.x +  20.3225f * X.y + 121.563f * X.w;
    M2_X.y =  2.92338f * X.x + (-27.0302f) * X.y + 626.13f * X.w;
    M2_X.z = 59.4188f  * X.x + 222.592f  * X.y + 316.627f * X.w;

    // M3 * X.xy: result[i] = col0[i]*X.x + col1[i]*X.y
    float2 M3_X;
    M3_X.x = 0.0365463f * X.x +  9.0632f * X.y;
    M3_X.y = 3.32707f   * X.x + (-9.04756f) * X.y;

    // M4 * X.xzw: result[i] = col0[i]*X.x + col1[i]*X.z + col2[i]*X.w
    float3 M4_X;
    M4_X.x =  1.0f      * X.x +   9.04401f * X.z +  5.56589f * X.w;
    M4_X.y =  3.59685f  * X.x + (-16.3174f) * X.z + 19.7886f  * X.w;
    M4_X.z = -1.36772f  * X.x +   9.22949f * X.z + (-20.2123f) * X.w;

    // dot(M1_X, Y.xy)
    float dot_M1_Y = M1_X.x * Y.x + M1_X.y * Y.y;
    // dot(M2_X, Y.xyw)
    float dot_M2_Y = M2_X.x * Y.x + M2_X.y * Y.y + M2_X.z * Y.w;
    // dot(M3_X, Y.xy)
    float dot_M3_Y = M3_X.x * Y.x + M3_X.y * Y.y;
    // dot(M4_X, Y.xyw)
    float dot_M4_Y = M4_X.x * Y.x + M4_X.y * Y.y + M4_X.z * Y.w;

    float bias = dot_M1_Y * PositiveRcp(dot_M2_Y);
    float scale = dot_M3_Y * PositiveRcp(dot_M4_Y);

    return saturate(Rf0 * scale + make_float3(bias, bias, bias));
}

#endif  // DLSS_HELPER_H

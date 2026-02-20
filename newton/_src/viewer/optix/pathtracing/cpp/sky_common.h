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

// Shared sky/environment code - CUDA/OptiX version
// Converted from Vulkan GLSL sky_common.glsl
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include func_common.h content before this file.

#ifndef SKY_COMMON_H
#define SKY_COMMON_H

//-----------------------------------------------------------------------------
// Physical sky parameters
//-----------------------------------------------------------------------------

struct PhysicalSkyParameters {
    float3 rgbUnitConversion;
    float multiplier;

    float haze;
    float redblueshift;
    float saturation;
    float horizonHeight;

    float3 groundColor;
    float horizonBlur;

    float3 nightColor;
    float sunDiskIntensity;

    float3 sunDirection;
    float sunDiskScale;

    float sunGlowIntensity;
    int yIsUp;
};

struct SkySamplingResult {
    float3 direction;  // Direction to the sampled light
    float pdf;  // Probability Density Function value
    float3 radiance;  // Light intensity
};

//-----------------------------------------------------------------------------
// Helper functions for physical sky
//-----------------------------------------------------------------------------

static __forceinline__ __device__ float rgbLuminance(float3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

static __forceinline__ __device__ float3 localCoordsToDir(float3 mainVec, float x, float y, float z)
{
    float3 u = normalize(
        fabsf(mainVec.x) < fabsf(mainVec.y) ? make_float3(0.0f, -mainVec.z, mainVec.y)
                                            : make_float3(mainVec.z, 0.0f, -mainVec.x)
    );
    float3 v = cross(mainVec, u);
    return x * u + y * v + z * mainVec;
}

static __forceinline__ __device__ float2 squareToDisk(float inX, float inY)
{
    float localX = 2.0f * inX - 1.0f;
    float localY = 2.0f * inY - 1.0f;
    if (localX == 0.0f && localY == 0.0f)
        return make_float2(0.0f, 0.0f);

    float r, phi;
    if (localX > -localY) {
        if (localX > localY) {
            r = localX;
            phi = (M_PIf / 4.0f) * (1.0f + localY / localX);
        } else {
            r = localY;
            phi = (M_PIf / 4.0f) * (3.0f - localX / localY);
        }
    } else {
        if (localX < localY) {
            r = -localX;
            phi = (M_PIf / 4.0f) * (5.0f + localY / localX);
        } else {
            r = -localY;
            phi = (M_PIf / 4.0f) * (7.0f - localX / localY);
        }
    }
    return make_float2(r, phi);
}

static __forceinline__ __device__ float3 reflectionDirDiffuseX(float3 inNormal, float2 inSample)
{
    float2 rPhi = squareToDisk(inSample.x, inSample.y);
    float x = rPhi.x * cosf(rPhi.y);
    float y = rPhi.x * sinf(rPhi.y);
    float z = sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y));
    return localCoordsToDir(inNormal, x, y, z);
}

static __forceinline__ __device__ float3 calcSunColor(float3 sunDir, float turbidity)
{
    if (sunDir.z <= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);

    float3 ko = make_float3(12.0f, 8.5f, 0.9f);
    float3 wavelength = make_float3(0.610f, 0.550f, 0.470f);
    float3 solRad = make_float3(1.0f, 0.992f, 0.911f) * (127500.0f / 0.9878f);

    float m = 1.0f / (sunDir.z + 0.15f * powf(93.885f - (acosf(sunDir.z) * 180.0f / M_PIf), -1.253f));
    float beta = 0.04608f * turbidity - 0.04586f;

    float3 ta = make_float3(
        expf(-m * beta * powf(wavelength.x, -1.3f)), expf(-m * beta * powf(wavelength.y, -1.3f)),
        expf(-m * beta * powf(wavelength.z, -1.3f))
    );
    float3 to = make_float3(expf(-m * ko.x * 0.0035f), expf(-m * ko.y * 0.0035f), expf(-m * ko.z * 0.0035f));
    float3 tr = make_float3(
        expf(-m * 0.008735f * powf(wavelength.x, -4.08f)), expf(-m * 0.008735f * powf(wavelength.y, -4.08f)),
        expf(-m * 0.008735f * powf(wavelength.z, -4.08f))
    );

    return tr * ta * to * solRad;
}

static __forceinline__ __device__ float3
skyColorXyz(float3 inDir, float3 inSunPos, float inTurbidity, float inLuminance)
{
    float3 xyz;
    float A, B, C, D, E;
    float cosGamma = dot(inSunPos, inDir);
    if (cosGamma > 1.0f) {
        cosGamma = 2.0f - cosGamma;
    }
    float gamma = acosf(cosGamma);
    float cosTheta = inDir.z;
    float cosThetaSun = inSunPos.z;
    float thetaSun = acosf(cosThetaSun);
    float t2 = inTurbidity * inTurbidity;
    float ts2 = thetaSun * thetaSun;
    float ts3 = ts2 * thetaSun;

    float zenithX
        = ((+0.001650f * ts3 - 0.003742f * ts2 + 0.002088f * thetaSun + 0) * t2
           + (-0.029028f * ts3 + 0.063773f * ts2 - 0.032020f * thetaSun + 0.003948f) * inTurbidity
           + (+0.116936f * ts3 - 0.211960f * ts2 + 0.060523f * thetaSun + 0.258852f));
    float zenithY
        = ((+0.002759f * ts3 - 0.006105f * ts2 + 0.003162f * thetaSun + 0) * t2
           + (-0.042149f * ts3 + 0.089701f * ts2 - 0.041536f * thetaSun + 0.005158f) * inTurbidity
           + (+0.153467f * ts3 - 0.267568f * ts2 + 0.066698f * thetaSun + 0.266881f));
    xyz.y = inLuminance;

    A = -0.019257f * inTurbidity - (0.29f - powf(cosThetaSun, 0.5f) * 0.09f);
    B = -0.066513f * inTurbidity + 0.000818f;
    C = -0.000417f * inTurbidity + 0.212479f;
    D = -0.064097f * inTurbidity - 0.898875f;
    E = -0.003251f * inTurbidity + 0.045178f;

    float x
        = (((1.f + A * expf(B / cosTheta)) * (1.f + C * expf(D * gamma) + E * cosGamma * cosGamma))
           / ((1 + A * expf(B / 1.0f)) * (1 + C * expf(D * thetaSun) + E * cosThetaSun * cosThetaSun)));

    A = -0.016698f * inTurbidity - 0.260787f;
    B = -0.094958f * inTurbidity + 0.009213f;
    C = -0.007928f * inTurbidity + 0.210230f;
    D = -0.044050f * inTurbidity - 1.653694f;
    E = -0.010922f * inTurbidity + 0.052919f;

    float y
        = (((1 + A * expf(B / cosTheta)) * (1 + C * expf(D * gamma) + E * cosGamma * cosGamma))
           / ((1 + A * expf(B / 1.0f)) * (1 + C * expf(D * thetaSun) + E * cosThetaSun * cosThetaSun)));

    float localSaturation = 1.0f;

    x = zenithX * ((x * localSaturation) + (1.0f - localSaturation));
    y = zenithY * ((y * localSaturation) + (1.0f - localSaturation));

    xyz.x = (x / y) * xyz.y;
    xyz.z = ((1.0f - x - y) / y) * xyz.y;
    return xyz;
}

static __forceinline__ __device__ float skyLuminance(float3 inDir, float3 inSunPos, float inTurbidity)
{
    float cosGamma = fminf(fmaxf(dot(inSunPos, inDir), 0.0f), 1.0f);
    float gamma = acosf(cosGamma);
    float cosTheta = inDir.z;
    float cosThetaSun = inSunPos.z;
    float thetaSun = acosf(cosThetaSun);

    float A = 0.178721f * inTurbidity - 1.463037f;
    float B = -0.355402f * inTurbidity + 0.427494f;
    float C = -0.022669f * inTurbidity + 5.325056f;
    float D = 0.120647f * inTurbidity - 2.577052f;
    float E = -0.066967f * inTurbidity + 0.370275f;

    return (
        ((1 + A * expf(B / cosTheta)) * (1 + C * expf(D * gamma) + E * cosGamma * cosGamma))
        / ((1 + A * expf(B / 1.0f)) * (1 + C * expf(D * thetaSun) + E * cosThetaSun * cosThetaSun))
    );
}

static __forceinline__ __device__ float3 calcSkyColor(float3 inSunDir, float3 inDir, float inTurbidity)
{
    float thetaSun = acosf(inSunDir.z);
    float chi = (4.0f / 9.0f - inTurbidity / 120.0f) * (M_PIf - 2.0f * thetaSun);
    float luminance = 1000.0f * ((4.0453f * inTurbidity - 4.9710f) * tanf(chi) - 0.2155f * inTurbidity + 2.4192f);
    luminance *= skyLuminance(inDir, inSunDir, inTurbidity);

    float3 xyz = skyColorXyz(inDir, inSunDir, inTurbidity, luminance);
    float3 envColor = make_float3(
        3.241f * xyz.x - 1.537f * xyz.y - 0.499f * xyz.z, -0.969f * xyz.x + 1.876f * xyz.y + 0.042f * xyz.z,
        0.056f * xyz.x - 0.204f * xyz.y + 1.057f * xyz.z
    );
    return envColor * M_PIf;
}

static __forceinline__ __device__ float3 calcSkyIrradiance(float3 inDataSunDir, float inDataSunDirHaze)
{
    float3 colSum = make_float3(0.0f, 0.0f, 0.0f);
    float3 nuStateNormal = make_float3(0.0f, 0.0f, 1.0f);

    for (float u = 0.1f; u < 1.0f; u += 0.2f) {
        for (float v = 0.1f; v < 1.0f; v += 0.2f) {
            float3 diff = reflectionDirDiffuseX(nuStateNormal, make_float2(u, v));
            colSum = colSum + calcSkyColor(inDataSunDir, diff, inDataSunDirHaze);
        }
    }
    return colSum / 25.0f;
}

static __forceinline__ __device__ float tweakSaturation(float inSaturation, float inHaze)
{
    if (inSaturation > 1.0f)
        return 1.0f;

    float lowSat = inSaturation * inSaturation * inSaturation;
    float localHaze = fminf(fmaxf((inHaze - 2.0f) / 15.0f, 0.0f), 1.0f);
    localHaze *= localHaze * localHaze;
    return mix(inSaturation, lowSat, localHaze);
}

static __forceinline__ __device__ float3 tweakVector(float3 dir, int yIsUp, float horizHeight)
{
    float3 outDir = dir;
    if (yIsUp == 1) {
        outDir = make_float3(dir.x, dir.z, dir.y);
    }
    if (horizHeight != 0) {
        outDir.z -= horizHeight;
        outDir = normalize(outDir);
    }
    return outDir;
}

static __forceinline__ __device__ float3 tweakColor(float3 tint, float saturation, float redness)
{
    float intensity = rgbLuminance(tint);
    float3 outTint = (saturation <= 0.0f) ? make_float3(intensity, intensity, intensity)
                                          : mix(make_float3(intensity, intensity, intensity), tint, saturation);
    outTint = outTint * make_float3(1.0f + redness, 1.0f, 1.0f - redness);
    return fmaxf(outTint, make_float3(0.0f, 0.0f, 0.0f));
}

static __forceinline__ __device__ float nightBrightnessAdjustment(float3 sunDir)
{
    float lmt = 0.30901699437494742410229341718282f;
    if (sunDir.z <= -lmt)
        return 0.0f;
    float factor = (sunDir.z + lmt) / lmt;
    factor *= factor;
    factor *= factor;
    return factor;
}

static __forceinline__ __device__ float2 calcPhysicalScale(float sunDiskScale, float sunGlowIntensity, float sunDiskIntensity)
{
    float sunAngularRadius = 0.00465f;
    float sunDiskRadius = sunAngularRadius * sunDiskScale;
    float sunGlowRadius = sunDiskRadius * 10.0f;

    float glowFuncIntegral = sunGlowIntensity
        * ((4.0f * M_PIf) - (24.0f * M_PIf) / (sunGlowRadius * sunGlowRadius)
           + (24.0f * M_PIf) * sinf(sunGlowRadius) / (sunGlowRadius * sunGlowRadius * sunGlowRadius));

    float targetSundiskIntegral = sunDiskIntensity * M_PIf;

    float skySunglowScale = 1.0f;
    float maxGlowIntegral = 0.5f * targetSundiskIntegral;
    if (glowFuncIntegral > maxGlowIntegral) {
        skySunglowScale *= maxGlowIntegral / glowFuncIntegral;
        targetSundiskIntegral -= maxGlowIntegral;
    } else {
        targetSundiskIntegral -= glowFuncIntegral;
    }

    float sundiskArea = 2.0f * M_PIf * (1.0f - cosf(sunDiskRadius));
    float targetSundiskIntensity = targetSundiskIntegral / sundiskArea;

    float actualSundiskIntegral = 1.0f * sundiskArea;
    float actualSundiskIntensity = sunDiskIntensity * 100.0f * actualSundiskIntegral / sundiskArea;
    return make_float2(
        (targetSundiskIntensity == 0.0f) ? 0.0f : targetSundiskIntensity / actualSundiskIntensity,
        skySunglowScale
    );
}

static __forceinline__ __device__ float3 evalPhysicalSky(const PhysicalSkyParameters& ss, float3 inDirection)
{
    if (ss.multiplier <= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);

    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    float factor = 1.0f;
    float nightFactor = 1.0f;
    float3 outColor = make_float3(0.0f, 0.0f, 0.0f);
    float3 rgbScale = ss.rgbUnitConversion * ss.multiplier;
    float heightAdjusted = (ss.horizonHeight + ss.horizonBlur) / 10.0f;
    float3 dir = tweakVector(inDirection, ss.yIsUp, heightAdjusted);
    float localHaze = fmaxf(2.0f, 2.0f + ss.haze);
    float localSaturation = tweakSaturation(ss.saturation, localHaze);

    float downness = dir.z;
    float3 realDir = dir;
    if (dir.z < 0.001f) {
        dir.z = 0.001f;
        dir = normalize(dir);
    }

    float3 sunDir = ss.sunDirection;
    sunDir = tweakVector(sunDir, ss.yIsUp, heightAdjusted);
    float3 realSunDir = sunDir;
    if (sunDir.z < 0.001f) {
        factor = nightBrightnessAdjustment(sunDir);
        sunDir.z = 0.001f;
        sunDir = normalize(sunDir);
    }

    float3 tint = (factor > 0.0f) ? calcSkyColor(sunDir, dir, localHaze) * factor : make_float3(0.0f, 0.0f, 0.0f);
    float3 dataSunColor = calcSunColor(sunDir, (downness > 0) ? localHaze : 2.0f);

    if (ss.sunDiskIntensity > 0.0f && ss.sunDiskScale > 0.0f) {
        float sunAngle = acosf(dot(realDir, realSunDir));
        float glowRadius = 0.00465f * ss.sunDiskScale * 10.0f;
        if (sunAngle < glowRadius) {
            float2 scales = calcPhysicalScale(ss.sunDiskScale, ss.sunGlowIntensity, ss.sunDiskIntensity);
            float centerProximity = (1.0f - sunAngle / glowRadius);
            float glowFactor = powf(centerProximity, 3.0f) * 2.0f * ss.sunGlowIntensity * scales.y;
            float smoothEdge = 0.95f + (localHaze / 500.0f);
            float t_ss = fminf(fmaxf((centerProximity - 0.85f) / (smoothEdge - 0.85f), 0.0f), 1.0f);
            float diskFactor = (t_ss * t_ss * (3.0f - 2.0f * t_ss)) * 100.0f * ss.sunDiskIntensity * scales.x;
            tint = tint + dataSunColor * (glowFactor + diskFactor);
        }
    }
    outColor = tint * rgbScale;

    if (downness <= 0.0f) {
        float3 irrad = calcSkyIrradiance(sunDir, 2.0f);
        float3 downColor = ss.groundColor * (irrad + dataSunColor * sunDir.z) * rgbScale;
        downColor = downColor * factor;
        float horBlur = ss.horizonBlur / 10.0f;
        if (horBlur > 0.0f) {
            float dness = fminf(fmaxf(-downness / horBlur, 0.0f), 1.0f);
            dness = dness * dness * (3.0f - 2.0f * dness);  // smoothstep
            outColor = mix(outColor, downColor, dness);
            nightFactor = 1.0f - dness;
        } else {
            outColor = downColor;
            nightFactor = 0.0f;
        }
    }

    outColor = tweakColor(outColor, localSaturation, ss.redblueshift);
    result = outColor * M_PIf;

    if (nightFactor > 0.0f) {
        float3 night = ss.nightColor * nightFactor;
        result = fmaxf(result, night);
    }

    return result;
}

// Probability that samplePhysicalSky samples the sun.
static __forceinline__ __device__ float physicalSkySunProbability(const PhysicalSkyParameters& ss)
{
    float sunElevation = ss.sunDirection.z;
    return (ss.sunDiskScale > 1e-5f) ? fminf(fmaxf(ss.sunDiskIntensity * sunElevation * 0.5f + 0.5f, 0.1f), 0.9f)
                                     : 0.0f;
}

// Returns the probability that samplePhysicalSky samples a certain direction.
static __forceinline__ __device__ float samplePhysicalSkyPDF(const PhysicalSkyParameters& ss, float3 inDirection)
{
    const float sunAngularRadius = 0.00465f * ss.sunDiskScale;
    const float skyPdf = 1.0f / (2.0f * M_PIf);
    const float sunSampleAngularRadius = 1.5f * sunAngularRadius;
    const float sunSampleSolidAngle = (sunSampleAngularRadius < 0.001f)
        ? M_PIf * sunSampleAngularRadius * sunSampleAngularRadius
        : 2.0f * M_PIf * (1.0f - cosf(sunSampleAngularRadius));
    const float sunPdf
        = (dot(inDirection, ss.sunDirection) >= cosf(sunSampleAngularRadius)) ? 1.0f / sunSampleSolidAngle : 0.0f;
    return mix(skyPdf, sunPdf, physicalSkySunProbability(ss));
}

// Uniformly samples a spherical cap
static __forceinline__ __device__ float3 sampleSphericalCap(float z_min, float2 xi)
{
    float z = mix(1.0f, z_min, xi.y);
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * M_PIf * xi.x;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return make_float3(x, y, z);
}

// Samples the physical sky model
static __forceinline__ __device__ SkySamplingResult
samplePhysicalSky(const PhysicalSkyParameters& ss, float2 randomSample)
{
    SkySamplingResult result;

    // Decide whether to sample sun or sky
    float sunProb = physicalSkySunProbability(ss);
    float z_min = 0.0f;
    bool sampleSun = randomSample.x < sunProb;
    if (sampleSun) {
        randomSample.x = randomSample.x / sunProb;
        const float sunSampleAngularRadius = 1.5f * 0.00465f * ss.sunDiskScale;
        z_min = cosf(sunSampleAngularRadius);
    } else {
        randomSample.x = (randomSample.x - sunProb) / (1.0f - sunProb);
    }

    result.direction = sampleSphericalCap(z_min, randomSample);

    if (sampleSun) {
        float3 up = make_float3(0.0f, 0.0f, 1.0f);
        float3 right = normalize(cross(up, ss.sunDirection));
        up = cross(ss.sunDirection, right);
        result.direction = result.direction.x * right + result.direction.y * up + result.direction.z * ss.sunDirection;
    }

    result.radiance = evalPhysicalSky(ss, result.direction);
    result.pdf = samplePhysicalSkyPDF(ss, result.direction);
    return result;
}

#endif  // SKY_COMMON_H

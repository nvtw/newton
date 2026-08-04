// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <cuda_runtime.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd_cuda.h>
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_params_dlssd.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) checkCuda((call), #call)
#define NGX_CHECK(call) checkNgx((call), #call)

void checkCuda(cudaError_t result, const char* call)
{
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
}

void checkCudaDriver(CUresult result, const char* call)
{
    if (result == CUDA_SUCCESS)
        return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(call) + ": " + (message ? message : "unknown CUDA driver error"));
}

void checkNgx(NVSDK_NGX_Result result, const char* call)
{
    if (NVSDK_NGX_FAILED(result))
        throw std::runtime_error(std::string(call) + ": NGX result " + std::to_string(static_cast<int>(result)));
}

__global__ void fillFloat(cudaSurfaceObject_t surface, int width, int height, float value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        surf2Dwrite(value, surface, x * sizeof(float), y);
}

__global__ void fillFloat2(cudaSurfaceObject_t surface, int width, int height, float2 value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        surf2Dwrite(value, surface, x * sizeof(float2), y);
}

__global__ void fillFloat4(cudaSurfaceObject_t surface, int width, int height, float4 value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        surf2Dwrite(value, surface, x * sizeof(float4), y);
}

__global__ void spinKernel(unsigned long long cycles)
{
    const unsigned long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

struct Texture
{
    cudaArray_t array = nullptr;
    cudaTextureObject_t object = 0;
    cudaSurfaceObject_t surface = 0;

    Texture(int width, int height, const cudaChannelFormatDesc& format)
    {
        CUDA_CHECK(cudaMallocArray(&array, &format, width, height, cudaArraySurfaceLoadStore));

        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = array;

        cudaTextureDesc texture{};
        texture.addressMode[0] = cudaAddressModeClamp;
        texture.addressMode[1] = cudaAddressModeClamp;
        texture.filterMode = cudaFilterModePoint;
        texture.readMode = cudaReadModeElementType;
        texture.normalizedCoords = 0;
        CUDA_CHECK(cudaCreateTextureObject(&object, &resource, &texture, nullptr));
        CUDA_CHECK(cudaCreateSurfaceObject(&surface, &resource));
    }

    ~Texture()
    {
        if (surface)
            cudaDestroySurfaceObject(surface);
        if (object)
            cudaDestroyTextureObject(object);
        if (array)
            cudaFreeArray(array);
    }

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
};

struct OutputSurface
{
    cudaArray_t array = nullptr;
    cudaSurfaceObject_t object = 0;

    OutputSurface(int width, int height)
    {
        const auto format = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaMallocArray(&array, &format, width, height, cudaArraySurfaceLoadStore));
        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = array;
        CUDA_CHECK(cudaCreateSurfaceObject(&object, &resource));
    }

    ~OutputSurface()
    {
        if (object)
            cudaDestroySurfaceObject(object);
        if (array)
            cudaFreeArray(array);
    }

    OutputSurface(const OutputSurface&) = delete;
    OutputSurface& operator=(const OutputSurface&) = delete;
};

int main(int argc, char** argv)
try {
    constexpr int inputWidth = 640;
    constexpr int inputHeight = 360;
    constexpr int outputWidth = 1280;
    constexpr int outputHeight = 720;
    constexpr int iterations = 10;
    constexpr int spinMilliseconds = 50;
    const unsigned long long applicationId = argc > 1 ? std::stoull(argv[1]) : 1;
    const int upscalingPreset = argc > 2 ? std::stoi(argv[2]) : 11;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));
    int clockRateKilohertz = 0;
    checkCudaDriver(cuDeviceGetAttribute(
        &clockRateKilohertz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 0),
        "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE)");

    cudaStream_t dlssStream = nullptr;
    cudaStream_t workStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&dlssStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&workStream, cudaStreamNonBlocking));

    CUcontext cudaContext = nullptr;
    checkCudaDriver(cuCtxGetCurrent(&cudaContext), "cuCtxGetCurrent");
    NVSDK_NGX_CUDADevice ngxDevice{};
    ngxDevice.cudaContext = cudaContext;
    ngxDevice.cudaStream = dlssStream;

    const std::filesystem::path executableDir = std::filesystem::absolute(argv[0]).parent_path();
    const std::wstring featurePath = executableDir.wstring();
    const wchar_t* featurePathPointer = featurePath.c_str();
    NVSDK_NGX_FeatureCommonInfo featureInfo{};
    featureInfo.PathListInfo.Path = &featurePathPointer;
    featureInfo.PathListInfo.Length = 1;

    std::cout << "DLSS Ray Reconstruction preset: E\n";
    std::cout << "DLSS upscaling preset value: " << upscalingPreset << "\n";
    std::cout << "NGX application ID: " << applicationId << "\n";
    std::cout << "Independent CUDA workload: " << spinMilliseconds << " ms\n";

    NGX_CHECK(NVSDK_NGX_CUDA_Init1(
        applicationId, executableDir.wstring().c_str(), &ngxDevice, &featureInfo));

    NVSDK_NGX_Parameter* parameters = nullptr;
    NGX_CHECK(NVSDK_NGX_CUDA_GetCapabilityParameters(&parameters));
    int available = 0;
    NGX_CHECK(parameters->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &available));
    if (!available)
        throw std::runtime_error("DLSS Ray Reconstruction is unavailable for this application ID");

    parameters->Set(
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
        NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E);
    if (upscalingPreset >= 0)
        parameters->Set(
            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
            static_cast<NVSDK_NGX_DLSS_Hint_Render_Preset>(upscalingPreset));

    NVSDK_NGX_CUDA_DLSSD_Create_Params create{};
    create.Feature.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    create.Feature.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
    create.Feature.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
    create.Feature.InWidth = inputWidth;
    create.Feature.InHeight = inputHeight;
    create.Feature.InTargetWidth = outputWidth;
    create.Feature.InTargetHeight = outputHeight;
    create.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    create.Feature.InFeatureCreateFlags =
        NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    create.InCUContext = cudaContext;
    create.InCUStream = dlssStream;

    NVSDK_NGX_Handle* feature = nullptr;
    NGX_CHECK(NGX_CUDA_CREATE_DLSSD_EXT1(&ngxDevice, &feature, parameters, &create));

    Texture color(inputWidth, inputHeight, cudaCreateChannelDesc<float4>());
    Texture diffuse(inputWidth, inputHeight, cudaCreateChannelDesc<float4>());
    Texture specular(inputWidth, inputHeight, cudaCreateChannelDesc<float4>());
    Texture normalRoughness(inputWidth, inputHeight, cudaCreateChannelDesc<float4>());
    Texture motion(inputWidth, inputHeight, cudaCreateChannelDesc<float2>());
    Texture depth(inputWidth, inputHeight, cudaCreateChannelDesc<float>());
    Texture specularHitDistance(inputWidth, inputHeight, cudaCreateChannelDesc<float>());
    OutputSurface output(outputWidth, outputHeight);

    const dim3 block(16, 16);
    const dim3 grid((inputWidth + 15) / 16, (inputHeight + 15) / 16);
    fillFloat4<<<grid, block, 0, dlssStream>>>(
        color.surface, inputWidth, inputHeight, make_float4(0.2f, 0.3f, 0.4f, 1.0f));
    fillFloat4<<<grid, block, 0, dlssStream>>>(
        diffuse.surface, inputWidth, inputHeight, make_float4(0.5f, 0.4f, 0.3f, 1.0f));
    fillFloat4<<<grid, block, 0, dlssStream>>>(
        specular.surface, inputWidth, inputHeight, make_float4(0.04f, 0.04f, 0.04f, 1.0f));
    fillFloat4<<<grid, block, 0, dlssStream>>>(
        normalRoughness.surface, inputWidth, inputHeight,
        make_float4(0.0f, 0.0f, 1.0f, 0.5f));
    fillFloat2<<<grid, block, 0, dlssStream>>>(
        motion.surface, inputWidth, inputHeight, make_float2(0.0f, 0.0f));
    fillFloat<<<grid, block, 0, dlssStream>>>(
        depth.surface, inputWidth, inputHeight, 1.0f);
    fillFloat<<<grid, block, 0, dlssStream>>>(
        specularHitDistance.surface, inputWidth, inputHeight, 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(dlssStream));

    std::array<float, 16> identity{1.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 1.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 1.0f};

    CUtexObject colorObject = color.object;
    CUtexObject diffuseObject = diffuse.object;
    CUtexObject specularObject = specular.object;
    CUtexObject normalRoughnessObject = normalRoughness.object;
    CUtexObject motionObject = motion.object;
    CUtexObject depthObject = depth.object;
    CUtexObject specularHitDistanceObject = specularHitDistance.object;
    CUsurfObject outputObject = output.object;

    NVSDK_NGX_CUDA_DLSSD_Eval_Params evaluate{};
    evaluate.pInColor = &colorObject;
    evaluate.pInOutput = &outputObject;
    evaluate.pInDiffuseAlbedo = &diffuseObject;
    evaluate.pInSpecularAlbedo = &specularObject;
    evaluate.pInNormals = &normalRoughnessObject;
    evaluate.pInRoughness = &normalRoughnessObject;
    evaluate.pInDepth = &depthObject;
    evaluate.pInMotionVectors = &motionObject;
    evaluate.pInSpecularHitDistance = &specularHitDistanceObject;
    evaluate.pInWorldToViewMatrix = identity.data();
    evaluate.pInViewToClipMatrix = identity.data();
    evaluate.InMVScaleX = 1.0f;
    evaluate.InMVScaleY = 1.0f;
    evaluate.InPreExposure = 1.0f;
    evaluate.InExposureScale = 1.0f;
    evaluate.InRenderSubrectDimensions = {inputWidth, inputHeight};

    NGX_CHECK(NGX_CUDA_EVALUATE_DLSSD_EXT(feature, parameters, &evaluate));
    CUDA_CHECK(cudaStreamSynchronize(dlssStream));

    const unsigned long long spinCycles =
        static_cast<unsigned long long>(clockRateKilohertz) * spinMilliseconds;
    double totalMilliseconds = 0.0;
    for (int frame = 0; frame < iterations; ++frame) {
        spinKernel<<<1, 1, 0, workStream>>>(spinCycles);
        CUDA_CHECK(cudaGetLastError());

        evaluate.InReset = frame == 0 ? 1 : 0;
        const auto begin = std::chrono::steady_clock::now();
        NGX_CHECK(NGX_CUDA_EVALUATE_DLSSD_EXT(feature, parameters, &evaluate));
        const auto end = std::chrono::steady_clock::now();
        const double milliseconds = std::chrono::duration<double, std::milli>(end - begin).count();
        totalMilliseconds += milliseconds;
        std::cout << "evaluate " << frame << ": " << milliseconds << " ms\n";

        CUDA_CHECK(cudaStreamSynchronize(dlssStream));
        CUDA_CHECK(cudaStreamSynchronize(workStream));
    }
    std::cout << "Mean evaluate call: " << totalMilliseconds / iterations << " ms\n";
    std::cout << "A stream-local evaluate should return far sooner than " << spinMilliseconds << " ms.\n";

    NGX_CHECK(NVSDK_NGX_CUDA_ReleaseFeature(feature));
    NGX_CHECK(NVSDK_NGX_CUDA_DestroyParameters(parameters));
    NGX_CHECK(NVSDK_NGX_CUDA_Shutdown1(&ngxDevice));
    CUDA_CHECK(cudaStreamDestroy(workStream));
    CUDA_CHECK(cudaStreamDestroy(dlssStream));
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
}

# DLSS CUDA context-sync repro

This standalone program reproduces context-wide CUDA synchronization during
DLSS Ray Reconstruction evaluation. It has no Newton, Warp, OptiX, OpenGL, or
Python dependency. It uses CUDA and NVIDIA's NGX/DLSS SDK only.

The repro initializes NGX with a persistent `NVSDK_NGX_CUDADevice` containing
the DLSS stream and uses `NVSDK_NGX_CUDA_Init1`,
`NVSDK_NGX_CUDA_CreateFeature1` (through `NGX_CUDA_CREATE_DLSSD_EXT1`), and
`NVSDK_NGX_CUDA_Shutdown1`.

It launches a 50 ms kernel on an independent CUDA stream immediately before
each DLSS evaluation. A stream-local evaluation should enqueue promptly. A
context synchronization makes the evaluation call block for approximately 50
ms.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/dlss_cuda_context_sync
```

CMake downloads DLSS SDK 310.5.3 automatically. To use an existing SDK:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DDLSS_ROOT=/path/to/DLSS-310.5.3
```

The optional first argument is a numeric NVIDIA NGX application ID. The
default `1` is suitable only for reproducing the behavior on systems where NGX
accepts it. Production applications must use an ID issued by NVIDIA.

The optional second argument selects the DLSS upscaling preset by its SDK enum
value. Use `-1` for the OTA/default model, `6` for deprecated F, `10` for J,
or `11` for K. Ray Reconstruction remains on preset E in every case:

```bash
./build/dlss_cuda_context_sync 1 11
```

## Nsight Systems

```bash
nsys profile --trace=cuda --sample=none --output=dlss-sync \
  ./build/dlss_cuda_context_sync
nsys stats --report=cuda_api_sum dlss-sync.nsys-rep
```

On the affected Linux CUDA backend, the summary contains per-evaluation
`cuCtxSynchronize` calls. A control program without DLSS contains none.

The repro combines Ray Reconstruction preset E with DLSS upscaling preset K.
These are separate NGX parameter namespaces; Ray Reconstruction's own preset K
is documented as reverting to its default behavior.

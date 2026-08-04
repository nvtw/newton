cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # DLSS requires Turing or newer. PTX avoids host CMake needing to recognize
  # newer architecture numbers such as Blackwell's sm_120.
  set(CMAKE_CUDA_ARCHITECTURES "75-virtual" CACHE STRING "CUDA architectures")
endif()

project(dlss_cuda_context_sync LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(DLSS_VERSION "310.5.3" CACHE STRING "NVIDIA DLSS SDK version")
set(DLSS_ROOT "" CACHE PATH "Existing NVIDIA DLSS SDK root")

if(NOT DLSS_ROOT)
  include(FetchContent)
  FetchContent_Declare(
    dlss
    URL "https://github.com/NVIDIA/DLSS/archive/refs/tags/v${DLSS_VERSION}.tar.gz"
    URL_HASH "SHA256=6b54a684b5b31e819a51742ad534abb4e8cdada76572f061a5d3149c7432a0a1"
  )
  FetchContent_MakeAvailable(dlss)
  set(DLSS_ROOT "${dlss_SOURCE_DIR}")
endif()

set(DLSS_INCLUDE_DIR "${DLSS_ROOT}/include")
set(DLSS_LIBRARY "${DLSS_ROOT}/lib/Linux_x86_64/libnvsdk_ngx.a")
file(GLOB DLSSD_RUNTIME "${DLSS_ROOT}/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so*")

if(NOT EXISTS "${DLSS_INCLUDE_DIR}/nvsdk_ngx.h" OR NOT EXISTS "${DLSS_LIBRARY}")
  message(FATAL_ERROR "Invalid DLSS_ROOT: ${DLSS_ROOT}")
endif()
if(NOT DLSSD_RUNTIME)
  message(FATAL_ERROR "DLSSD Linux runtime not found below ${DLSS_ROOT}")
endif()
list(GET DLSSD_RUNTIME 0 DLSSD_RUNTIME)

add_executable(dlss_cuda_context_sync main.cu)
target_compile_features(dlss_cuda_context_sync PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(dlss_cuda_context_sync PRIVATE "${DLSS_INCLUDE_DIR}")
target_link_libraries(
  dlss_cuda_context_sync
  PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    "${DLSS_LIBRARY}"
    ${CMAKE_DL_LIBS}
)

add_custom_command(
  TARGET dlss_cuda_context_sync POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${DLSSD_RUNTIME}"
          "$<TARGET_FILE_DIR:dlss_cuda_context_sync>/libnvidia-ngx-dlssd.so"
)

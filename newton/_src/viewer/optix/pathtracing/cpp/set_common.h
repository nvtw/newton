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

/*
 * Shader Execution Reordering (SET) Helper - CUDA/OptiX version
 * Converted from Vulkan GLSL set_common.glsl
 * 
 * This header provides macros to optionally enable SET when available.
 * Define USE_SET before including this header to enable SET functionality.
 * 
 * OptiX 7.7+ supports Shader Execution Reordering (SET) through:
 * - optixReorder() - reorders threads by hit object
 * - optixInvoke() - executes hit/miss shaders after reordering
 * 
 * SET improves ray tracing performance by reordering shader invocations
 * to improve memory coherence. This is especially beneficial when:
 * - Rays hit diverse materials scattered across memory
 * - Secondary/bounce rays diverge significantly
 * 
 * Usage:
 *   #define USE_SET 1
 *   #include "set_common.h"
 *   
 *   // In your raygen shader:
 *   optixReorder();  // Reorder by hit
 *   optixInvoke();   // Execute shaders
 */

#ifndef SET_COMMON_H
#define SET_COMMON_H

// NOTE: This file is designed to be inlined into a single CUDA source.
//       OptiX device headers should be included before this file.

#ifdef USE_SET

// ============================================================
// OptiX SET support (requires OptiX 7.7+)
// ============================================================

// Check if OptiX version supports SET
#if OPTIX_VERSION >= 70700

    #define SET_AVAILABLE 1
    
    // In OptiX, SET is implemented through:
    // - optixReorder() - reorders threads based on hit information
    // - optixInvoke() - invokes the appropriate shader after reordering
    //
    // These are called after optixTrace() returns but before processing the payload.
    // The typical pattern is:
    //
    //   optixTrace(...);
    //   optixReorder();  // Reorder threads by hit
    //   optixInvoke();   // Execute hit/miss shaders
    //
    // Note: In OptiX, unlike Vulkan, there's no separate "hit object" type.
    // The reordering is based on the trace result stored internally.

    // Macro to perform SET reordering after trace
    // Call this after optixTrace() but before processing payload
    #define SET_REORDER() optixReorder()
    
    // Macro to invoke shaders after reordering
    #define SET_INVOKE() optixInvoke()
    
    // Combined reorder and invoke
    #define SET_REORDER_AND_INVOKE() do { optixReorder(); optixInvoke(); } while(0)
    
    // Hint-based reordering (for custom coherence hints)
    #define SET_REORDER_BY_HINT(hint, hintBits) optixReorder(hint, hintBits)

#else
    // OptiX version doesn't support SET
    #define SET_AVAILABLE 0
    #define SET_REORDER()
    #define SET_INVOKE()
    #define SET_REORDER_AND_INVOKE()
    #define SET_REORDER_BY_HINT(hint, hintBits)
#endif

#else
// ============================================================
// When SET is disabled, provide dummy macros
// ============================================================

#define SET_AVAILABLE 0
#define SET_REORDER()
#define SET_INVOKE()
#define SET_REORDER_AND_INVOKE()
#define SET_REORDER_BY_HINT(hint, hintBits)

#endif // USE_SET

#endif // SET_COMMON_H

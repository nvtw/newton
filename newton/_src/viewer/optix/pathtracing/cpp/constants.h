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

// Shared constants for all shaders - CUDA/OptiX version
// Converted from Vulkan GLSL constants.glsl

#ifndef CONSTANTS_H
#define CONSTANTS_H

// Force float constants (avoid accidental double-precision math from host headers).
#ifdef M_PI
#undef M_PI
#endif
#ifdef M_TWO_PI
#undef M_TWO_PI
#endif
#ifdef M_PI_2
#undef M_PI_2
#endif
#ifdef M_PI_4
#undef M_PI_4
#endif
#ifdef M_1_OVER_PI
#undef M_1_OVER_PI
#endif
#ifdef M_1_PI
#undef M_1_PI
#endif
#ifdef M_2_OVER_PI
#undef M_2_OVER_PI
#endif
#define M_PI        3.14159265358979323846f
#define M_TWO_PI    6.28318530717958648f
#define M_PI_2      1.57079632679489661923f
#define M_PI_4      0.785398163397448309616f
#define M_1_OVER_PI 0.318309886183790671538f
#define M_1_PI      0.318309886183790671538f
#define M_2_OVER_PI 0.6366197723675f

#ifndef INFINITE
#define INFINITE 1e32f
#endif

#ifndef M_PIf
#define M_PIf 3.1415926535f
#endif

#endif  // CONSTANTS_H

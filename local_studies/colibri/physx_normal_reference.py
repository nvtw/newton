# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
# Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
# Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

"""Experimental FP32 translation of the PhysX GPU TGS hard-normal row.

Source: solverBlockTGS.cuh, solveContactBlockTGS, separation through deltaF;
PhysX checkout ed6e5ca2474c9c80ad4f4826591b88476779c6ef. This is the scalar
row update only, not a complete PhysX contact solver. Inputs use PhysX's
normal/velocity convention. The caller owns preparation, accumulated motion,
row ordering and equal/opposite physical impulse application.
"""

import warp as wp


@wp.func
def normal_impulse_tgs(
    separation: wp.float32,
    linear_motion: wp.float32,
    angular_motion0: wp.float32,
    angular_motion1: wp.float32,
    min_penetration: wp.float32,
    target_velocity: wp.float32,
    elapsed_time: wp.float32,
    reciprocal_response: wp.float32,
    bias_coefficient: wp.float32,
    max_penetration_bias: wp.float32,
    normal_velocity: wp.float32,
    applied_impulse: wp.float32,
    max_impulse: wp.float32,
) -> wp.vec2f:
    """Return (new accumulated impulse, impulse increment), without relaxation."""
    sep = wp.max(min_penetration, separation + linear_motion + angular_motion0 - angular_motion1)
    target_velocity_error = target_velocity * elapsed_time
    biased_error = reciprocal_response * wp.min(-max_penetration_bias, bias_coefficient * (sep - target_velocity_error))
    # Hard contact: velMultiplier == recipResponse in the original source.
    trial_delta = biased_error - (normal_velocity - target_velocity) * reciprocal_response
    bounded_delta = wp.max(trial_delta, -applied_impulse)
    new_impulse = wp.min(applied_impulse + bounded_delta, max_impulse)
    return wp.vec2f(new_impulse, new_impulse - applied_impulse)

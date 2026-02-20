# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small reusable helper for creating an OptiX device context."""

import logging

logger = logging.getLogger(__name__)


class _Logger:
    def __init__(self):
        self.num_messages = 0

    def __call__(self, level, tag, message):
        logger.info("[%2s][%12s]: %s", level, tag, message)
        self.num_messages += 1


def _create_optix_context(optix, cuda_context):
    """Create an OptiX device context and logger."""
    logger = _Logger()
    ctx_options = optix.DeviceContextOptions(logCallbackFunction=logger, logCallbackLevel=4)
    if optix.version()[1] >= 2:
        ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
    return optix.deviceContextCreate(cuda_context, ctx_options), logger

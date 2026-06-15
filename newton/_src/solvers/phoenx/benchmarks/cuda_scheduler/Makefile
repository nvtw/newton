# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: all run clean

all: scheduler_microbench

scheduler_microbench: scheduler_microbench.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $< -o $@

run: scheduler_microbench
	./scheduler_microbench

clean:
	rm -f scheduler_microbench

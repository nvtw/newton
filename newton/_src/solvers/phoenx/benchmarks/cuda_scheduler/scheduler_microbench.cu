// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#define CUDA_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        cudaError_t err__ = (expr);                                                                                    \
        if (err__ != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__));             \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

struct SceneConfig {
    const char* name;
    std::vector<int> rows_per_color;
};

struct DeviceGraph {
    int worlds = 0;
    int colors = 0;
    int rows = 0;
    int chunks = 0;
    int epochs = 0;
    int queue_capacity = 0;

    int* world_color_row_starts = nullptr;
    int* world_color_chunk_starts = nullptr;
    int* chunk_row_start = nullptr;
    int* chunk_row_count = nullptr;
    int* chunk_world = nullptr;
    int* chunk_color = nullptr;
    int* remaining = nullptr;
    int* queue = nullptr;
    int* queue_ready = nullptr;
    int* queue_head = nullptr;
    int* queue_tail = nullptr;
    int* done = nullptr;
    int* failed = nullptr;
    float* out = nullptr;
};

static int parse_int_arg(int argc, char** argv, const char* flag, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return fallback;
}

static std::string parse_string_arg(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return argv[i + 1];
        }
    }
    return fallback;
}

static std::vector<std::string> split_csv(std::string_view value) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= value.size()) {
        size_t comma = value.find(',', start);
        if (comma == std::string_view::npos) {
            comma = value.size();
        }
        if (comma > start) {
            out.emplace_back(value.substr(start, comma - start));
        }
        start = comma + 1;
        if (comma == value.size()) {
            break;
        }
    }
    return out;
}

static SceneConfig scene_config(const std::string& name) {
    if (name == "h1") {
        return {"h1", {11, 6, 5, 2}};
    }
    if (name == "g1") {
        return {"g1", {19, 14, 8, 2}};
    }
    if (name == "dr_legs") {
        return {"dr_legs", {14, 13, 9, 5, 1}};
    }
    if (name == "tower") {
        return {"tower", {64, 61, 58, 55, 52, 49, 46, 44, 42, 40, 38, 36, 34}};
    }
    std::fprintf(stderr, "unknown scene: %s\n", name.c_str());
    std::exit(2);
}

static void copy_to_device(int*& dst, const std::vector<int>& src) {
    CUDA_CHECK(cudaMalloc(&dst, sizeof(int) * src.size()));
    CUDA_CHECK(cudaMemcpy(dst, src.data(), sizeof(int) * src.size(), cudaMemcpyHostToDevice));
}

static int varied_rows(int base_rows, int world, int color, int imbalance_percent) {
    if (imbalance_percent <= 0) {
        return base_rows;
    }
    unsigned hash = static_cast<unsigned>((world + 1) * 1103515245u ^ (color + 17) * 1664525u);
    float unit = static_cast<float>(hash & 65535u) / 65535.0f;
    float centered = 2.0f * unit - 1.0f;
    float scale = 1.0f + centered * static_cast<float>(imbalance_percent) * 0.01f;
    return std::max(1, static_cast<int>(std::lround(static_cast<float>(base_rows) * scale)));
}

static DeviceGraph make_graph(const SceneConfig& scene, int worlds, int chunk_rows, int epochs, int imbalance_percent) {
    DeviceGraph graph;
    graph.worlds = worlds;
    graph.colors = static_cast<int>(scene.rows_per_color.size());
    graph.epochs = epochs;

    std::vector<int> row_starts(worlds * (graph.colors + 1), 0);
    std::vector<int> chunk_starts(worlds * (graph.colors + 1), 0);
    std::vector<int> chunk_row_start;
    std::vector<int> chunk_row_count;
    std::vector<int> chunk_world;
    std::vector<int> chunk_color;

    int row_cursor = 0;
    int chunk_cursor = 0;
    for (int w = 0; w < worlds; ++w) {
        for (int c = 0; c < graph.colors; ++c) {
            row_starts[w * (graph.colors + 1) + c] = row_cursor;
            chunk_starts[w * (graph.colors + 1) + c] = chunk_cursor;
            int rows = varied_rows(scene.rows_per_color[c], w, c, imbalance_percent);
            for (int off = 0; off < rows; off += chunk_rows) {
                int count = std::min(chunk_rows, rows - off);
                chunk_row_start.push_back(row_cursor + off);
                chunk_row_count.push_back(count);
                chunk_world.push_back(w);
                chunk_color.push_back(c);
                ++chunk_cursor;
            }
            row_cursor += rows;
        }
        row_starts[w * (graph.colors + 1) + graph.colors] = row_cursor;
        chunk_starts[w * (graph.colors + 1) + graph.colors] = chunk_cursor;
    }

    graph.rows = row_cursor;
    graph.chunks = chunk_cursor;
    graph.queue_capacity = std::max(1, graph.chunks * epochs);

    copy_to_device(graph.world_color_row_starts, row_starts);
    copy_to_device(graph.world_color_chunk_starts, chunk_starts);
    copy_to_device(graph.chunk_row_start, chunk_row_start);
    copy_to_device(graph.chunk_row_count, chunk_row_count);
    copy_to_device(graph.chunk_world, chunk_world);
    copy_to_device(graph.chunk_color, chunk_color);

    CUDA_CHECK(cudaMalloc(&graph.remaining, sizeof(int) * worlds * graph.colors * epochs));
    CUDA_CHECK(cudaMalloc(&graph.queue, sizeof(int) * graph.queue_capacity));
    CUDA_CHECK(cudaMalloc(&graph.queue_ready, sizeof(int) * graph.queue_capacity));
    CUDA_CHECK(cudaMalloc(&graph.queue_head, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&graph.queue_tail, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&graph.done, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&graph.failed, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&graph.out, sizeof(float) * graph.rows * epochs));
    return graph;
}

static void free_graph(DeviceGraph& graph) {
    cudaFree(graph.world_color_row_starts);
    cudaFree(graph.world_color_chunk_starts);
    cudaFree(graph.chunk_row_start);
    cudaFree(graph.chunk_row_count);
    cudaFree(graph.chunk_world);
    cudaFree(graph.chunk_color);
    cudaFree(graph.remaining);
    cudaFree(graph.queue);
    cudaFree(graph.queue_ready);
    cudaFree(graph.queue_head);
    cudaFree(graph.queue_tail);
    cudaFree(graph.done);
    cudaFree(graph.failed);
    cudaFree(graph.out);
}

__device__ __forceinline__ float payload(int row, int epoch, int work_iters) {
    float x = 1.0f + static_cast<float>((row * 1103515245u + epoch * 12345u) & 1023u) * 0.001f;
    float y = 2.0f + static_cast<float>(((row + 17) * 1664525u) & 1023u) * 0.001f;
    for (int k = 0; k < work_iters; ++k) {
        x = fmaf(x, 1.000001f, y * 0.000001f);
        y = fmaf(y, 0.999999f, x * 0.000001f);
    }
    return x + y;
}

__global__ void fast_tail_kernel(
    const int* __restrict__ row_starts,
    int worlds,
    int colors,
    int epochs,
    int threads_per_world,
    int work_iters,
    float* __restrict__ out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local = tid % threads_per_world;
    int world = tid / threads_per_world;
    if (world >= worlds) {
        return;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int color = 0; color < colors; ++color) {
            int start = row_starts[world * (colors + 1) + color];
            int end = row_starts[world * (colors + 1) + color + 1];
            for (int row = start + local; row < end; row += threads_per_world) {
                out[epoch * row_starts[worlds * (colors + 1) - 1] + row] = payload(row, epoch, work_iters);
            }
            __syncwarp();
        }
    }
}

__global__ void world_tile_kernel(
    const int* __restrict__ row_starts,
    int worlds,
    int colors,
    int epochs,
    int worlds_per_block,
    int rows,
    int work_iters,
    float* __restrict__ out) {
    extern __shared__ int prefix[];
    int tile_start = blockIdx.x * worlds_per_block;
    if (tile_start >= worlds) {
        return;
    }
    int tile_worlds = min(worlds_per_block, worlds - tile_start);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int color = 0; color < colors; ++color) {
            if (threadIdx.x == 0) {
                prefix[0] = 0;
                for (int i = 0; i < tile_worlds; ++i) {
                    int world = tile_start + i;
                    int start = row_starts[world * (colors + 1) + color];
                    int end = row_starts[world * (colors + 1) + color + 1];
                    prefix[i + 1] = prefix[i] + (end - start);
                }
            }
            __syncthreads();
            int total = prefix[tile_worlds];
            for (int local_row = threadIdx.x; local_row < total; local_row += blockDim.x) {
                int lane_world = 0;
                while (lane_world + 1 < tile_worlds && local_row >= prefix[lane_world + 1]) {
                    ++lane_world;
                }
                int world = tile_start + lane_world;
                int row_offset = local_row - prefix[lane_world];
                int row = row_starts[world * (colors + 1) + color] + row_offset;
                out[epoch * rows + row] = payload(row, epoch, work_iters);
            }
            __syncthreads();
        }
    }
}

__global__ void init_remaining_kernel(
    const int* __restrict__ chunk_starts,
    int* __restrict__ remaining,
    int worlds,
    int colors,
    int epochs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = worlds * colors * epochs;
    if (tid >= total) {
        return;
    }
    int color = tid % colors;
    int tmp = tid / colors;
    int world = tmp % worlds;
    remaining[tid] = chunk_starts[world * (colors + 1) + color + 1] - chunk_starts[world * (colors + 1) + color];
}

__device__ __forceinline__ void publish_task(
    int* __restrict__ queue,
    int* __restrict__ queue_ready,
    int* __restrict__ queue_tail,
    int* __restrict__ failed,
    int queue_capacity,
    int task) {
    int pos = atomicAdd(queue_tail, 1);
    if (pos < queue_capacity) {
        queue[pos] = task;
        __threadfence();
        queue_ready[pos] = 1;
    } else {
        *failed = 2;
    }
}

__global__ void seed_queue_kernel(
    const int* __restrict__ chunk_starts,
    int* __restrict__ queue,
    int* __restrict__ queue_ready,
    int* __restrict__ queue_tail,
    int* __restrict__ failed,
    int worlds,
    int colors,
    int chunks,
    int queue_capacity) {
    int world = blockIdx.x * blockDim.x + threadIdx.x;
    if (world >= worlds) {
        return;
    }
    for (int color = 0; color < colors; ++color) {
        int start = chunk_starts[world * (colors + 1) + color];
        int end = chunk_starts[world * (colors + 1) + color + 1];
        if (start < end) {
            for (int chunk = start; chunk < end; ++chunk) {
                publish_task(queue, queue_ready, queue_tail, failed, queue_capacity, chunk);
            }
            return;
        }
    }
}

__device__ __forceinline__ void notify_next(
    const int* __restrict__ chunk_starts,
    const int* __restrict__ chunk_world,
    const int* __restrict__ chunk_color,
    int* __restrict__ remaining,
    int* __restrict__ queue,
    int* __restrict__ queue_ready,
    int* __restrict__ queue_tail,
    int* __restrict__ done,
    int* __restrict__ failed,
    int worlds,
    int colors,
    int chunks,
    int epochs,
    int queue_capacity,
    int task) {
    int epoch = task / chunks;
    int chunk = task - epoch * chunks;
    int world = chunk_world[chunk];
    int color = chunk_color[chunk];
    int remaining_index = (epoch * worlds + world) * colors + color;
    int old = atomicSub(&remaining[remaining_index], 1);
    if (old == 1) {
        int next_color = color + 1;
        int next_epoch = epoch;
        while (next_epoch < epochs) {
            if (next_color >= colors) {
                next_color = 0;
                ++next_epoch;
                continue;
            }
            int start = chunk_starts[world * (colors + 1) + next_color];
            int end = chunk_starts[world * (colors + 1) + next_color + 1];
            if (start < end) {
                for (int next_chunk = start; next_chunk < end; ++next_chunk) {
                    publish_task(
                        queue,
                        queue_ready,
                        queue_tail,
                        failed,
                        queue_capacity,
                        next_epoch * chunks + next_chunk);
                }
                break;
            }
            ++next_color;
        }
    }
    atomicAdd(done, 1);
}

__global__ void chunk_chain_kernel(
    const int* __restrict__ chunk_starts,
    const int* __restrict__ chunk_row_start,
    const int* __restrict__ chunk_row_count,
    const int* __restrict__ chunk_world,
    const int* __restrict__ chunk_color,
    int* __restrict__ remaining,
    int* __restrict__ queue,
    int* __restrict__ queue_ready,
    int* __restrict__ queue_head,
    int* __restrict__ queue_tail,
    int* __restrict__ done,
    int* __restrict__ failed,
    int worlds,
    int colors,
    int chunks,
    int rows,
    int epochs,
    int queue_capacity,
    int work_iters,
    int max_spins,
    float* __restrict__ out) {
    __shared__ int shared_task;
    __shared__ int shared_stop;
    int local = threadIdx.x;
    int target = chunks * epochs;

    while (true) {
        if (local == 0) {
            shared_task = -1;
            shared_stop = 0;
            int spins = 0;
            while (shared_task < 0) {
                int observed_done = atomicAdd(done, 0);
                if (observed_done >= target) {
                    shared_stop = 1;
                    break;
                }
                int head = atomicAdd(queue_head, 0);
                int tail = atomicAdd(queue_tail, 0);
                if (head >= tail) {
                    if (++spins > max_spins) {
                        *failed = 1;
                        shared_stop = 1;
                        break;
                    }
                    continue;
                }
                int old = atomicCAS(queue_head, head, head + 1);
                if (old != head) {
                    continue;
                }
                while (atomicAdd(&queue_ready[head], 0) == 0) {
                    if (++spins > max_spins) {
                        *failed = 3;
                        shared_stop = 1;
                        break;
                    }
                }
                if (shared_stop == 0) {
                    shared_task = queue[head];
                }
            }
        }
        __syncthreads();
        if (shared_stop != 0) {
            return;
        }

        int task = shared_task;
        int epoch = task / chunks;
        int chunk = task - epoch * chunks;
        int start = chunk_row_start[chunk];
        int count = chunk_row_count[chunk];
        for (int i = local; i < count; i += blockDim.x) {
            int row = start + i;
            out[epoch * rows + row] = payload(row, epoch, work_iters);
        }
        __syncthreads();
        if (local == 0) {
            notify_next(
                chunk_starts,
                chunk_world,
                chunk_color,
                remaining,
                queue,
                queue_ready,
                queue_tail,
                done,
                failed,
                worlds,
                colors,
                chunks,
                epochs,
                queue_capacity,
                task);
        }
        __syncthreads();
    }
}

static void reset_chain(const DeviceGraph& graph) {
    CUDA_CHECK(cudaMemset(graph.queue_ready, 0, sizeof(int) * graph.queue_capacity));
    CUDA_CHECK(cudaMemset(graph.queue_head, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(graph.queue_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(graph.done, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(graph.failed, 0, sizeof(int)));
    int total_remaining = graph.worlds * graph.colors * graph.epochs;
    init_remaining_kernel<<<(total_remaining + 255) / 256, 256>>>(
        graph.world_color_chunk_starts, graph.remaining, graph.worlds, graph.colors, graph.epochs);
    CUDA_CHECK(cudaGetLastError());
    seed_queue_kernel<<<(graph.worlds + 127) / 128, 128>>>(
        graph.world_color_chunk_starts,
        graph.queue,
        graph.queue_ready,
        graph.queue_tail,
        graph.failed,
        graph.worlds,
        graph.colors,
        graph.chunks,
        graph.queue_capacity);
    CUDA_CHECK(cudaGetLastError());
}

static float time_world_tile(
    const DeviceGraph& graph,
    int repeats,
    int warmup,
    int worlds_per_block,
    int block_dim,
    int work_iters) {
    int blocks = (graph.worlds + worlds_per_block - 1) / worlds_per_block;
    size_t shared_bytes = sizeof(int) * (worlds_per_block + 1);
    for (int i = 0; i < warmup; ++i) {
        world_tile_kernel<<<blocks, block_dim, shared_bytes>>>(
            graph.world_color_row_starts,
            graph.worlds,
            graph.colors,
            graph.epochs,
            worlds_per_block,
            graph.rows,
            work_iters,
            graph.out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float best = std::numeric_limits<float>::max();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        world_tile_kernel<<<blocks, block_dim, shared_bytes>>>(
            graph.world_color_row_starts,
            graph.worlds,
            graph.colors,
            graph.epochs,
            worlds_per_block,
            graph.rows,
            work_iters,
            graph.out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        best = std::min(best, ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}

static float time_fast_tail(const DeviceGraph& graph, int repeats, int warmup, int tpw, int block_dim, int work_iters) {
    int threads = graph.worlds * tpw;
    int blocks = (threads + block_dim - 1) / block_dim;
    for (int i = 0; i < warmup; ++i) {
        fast_tail_kernel<<<blocks, block_dim>>>(
            graph.world_color_row_starts, graph.worlds, graph.colors, graph.epochs, tpw, work_iters, graph.out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float best = std::numeric_limits<float>::max();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fast_tail_kernel<<<blocks, block_dim>>>(
            graph.world_color_row_starts, graph.worlds, graph.colors, graph.epochs, tpw, work_iters, graph.out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        best = std::min(best, ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}

static float time_chunk_chain(
    const DeviceGraph& graph,
    int repeats,
    int warmup,
    int worker_blocks,
    int chunk_threads,
    int work_iters,
    int max_spins) {
    for (int i = 0; i < warmup; ++i) {
        reset_chain(graph);
        chunk_chain_kernel<<<worker_blocks, chunk_threads>>>(
            graph.world_color_chunk_starts,
            graph.chunk_row_start,
            graph.chunk_row_count,
            graph.chunk_world,
            graph.chunk_color,
            graph.remaining,
            graph.queue,
            graph.queue_ready,
            graph.queue_head,
            graph.queue_tail,
            graph.done,
            graph.failed,
            graph.worlds,
            graph.colors,
            graph.chunks,
            graph.rows,
            graph.epochs,
            graph.queue_capacity,
            work_iters,
            max_spins,
            graph.out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float best = std::numeric_limits<float>::max();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < repeats; ++i) {
        reset_chain(graph);
        CUDA_CHECK(cudaEventRecord(start));
        chunk_chain_kernel<<<worker_blocks, chunk_threads>>>(
            graph.world_color_chunk_starts,
            graph.chunk_row_start,
            graph.chunk_row_count,
            graph.chunk_world,
            graph.chunk_color,
            graph.remaining,
            graph.queue,
            graph.queue_ready,
            graph.queue_head,
            graph.queue_tail,
            graph.done,
            graph.failed,
            graph.worlds,
            graph.colors,
            graph.chunks,
            graph.rows,
            graph.epochs,
            graph.queue_capacity,
            work_iters,
            max_spins,
            graph.out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        int done = 0;
        int failed = 0;
        CUDA_CHECK(cudaMemcpy(&done, graph.done, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&failed, graph.failed, sizeof(int), cudaMemcpyDeviceToHost));
        if (done != graph.chunks * graph.epochs || failed != 0) {
            std::fprintf(stderr, "chunk_chain failed: done=%d expected=%d failed=%d\n", done, graph.chunks * graph.epochs, failed);
            std::exit(3);
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        best = std::min(best, ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}

int main(int argc, char** argv) {
    int worlds = parse_int_arg(argc, argv, "--worlds", 32);
    int epochs = parse_int_arg(argc, argv, "--epochs", 4);
    int chunk_rows = parse_int_arg(argc, argv, "--chunk-rows", 32);
    int chunk_threads = parse_int_arg(argc, argv, "--chunk-threads", 32);
    int worker_blocks = parse_int_arg(argc, argv, "--worker-blocks", 64);
    int tpw = parse_int_arg(argc, argv, "--tpw", 32);
    int block_dim = parse_int_arg(argc, argv, "--block-dim", 128);
    int worlds_per_block = parse_int_arg(argc, argv, "--worlds-per-block", 8);
    int work_iters = parse_int_arg(argc, argv, "--work-iters", 32);
    int imbalance = parse_int_arg(argc, argv, "--imbalance", 0);
    int repeats = parse_int_arg(argc, argv, "--repeats", 20);
    int warmup = parse_int_arg(argc, argv, "--warmup", 3);
    int max_spins = parse_int_arg(argc, argv, "--max-spins", 100000000);
    std::string scenes_arg = parse_string_arg(argc, argv, "--scenes", "h1,g1,dr_legs,tower");

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf(
        "device=%s sm=%d worlds=%d epochs=%d tpw=%d wpb=%d chunk_rows=%d chunk_threads=%d worker_blocks=%d work_iters=%d imbalance=%d\n",
        prop.name,
        prop.multiProcessorCount,
        worlds,
        epochs,
        tpw,
        worlds_per_block,
        chunk_rows,
        chunk_threads,
        worker_blocks,
        work_iters,
        imbalance);
    std::printf("scene rows chunks colors fast_tail_ms tile_ms tile_speedup chunk_chain_ms chain_speedup rows_per_world chunks_per_world_color\n");

    for (const std::string& scene_name : split_csv(scenes_arg)) {
        SceneConfig scene = scene_config(scene_name);
        DeviceGraph graph = make_graph(scene, worlds, chunk_rows, epochs, imbalance);
        float fast_ms = time_fast_tail(graph, repeats, warmup, tpw, block_dim, work_iters);
        float tile_ms = time_world_tile(graph, repeats, warmup, worlds_per_block, block_dim, work_iters);
        float chain_ms = time_chunk_chain(graph, repeats, warmup, worker_blocks, chunk_threads, work_iters, max_spins);
        float tile_speedup = fast_ms / tile_ms;
        float chain_speedup = fast_ms / chain_ms;
        float rows_per_world = static_cast<float>(graph.rows) / static_cast<float>(worlds);
        float chunks_per_world_color = static_cast<float>(graph.chunks) / static_cast<float>(worlds * graph.colors);
        std::printf(
            "%8s %7d %6d %6d %12.4f %8.4f %11.3fx %14.4f %12.3fx %14.1f %22.2f\n",
            scene.name,
            graph.rows,
            graph.chunks,
            graph.colors,
            fast_ms,
            tile_ms,
            tile_speedup,
            chain_ms,
            chain_speedup,
            rows_per_world,
            chunks_per_world_color);
        free_graph(graph);
    }
    return 0;
}

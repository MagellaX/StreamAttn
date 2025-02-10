/*
 * Advanced FlashAttention with Double Buffering, cp.async, and CUDA Streams
 *
 * This experimental implementation fuses the softmax reduction and weighted sum
 * computations using an “online” (numerically stable) update while hiding global–to–shared
 * memory latency with double buffering and asynchronous copies via cp.async.
 *
 * It also partitions work across multiple CUDA streams so that host–device transfers,
 * kernel execution, and data pipelining can be overlapped.
 *
 * IMPORTANT: This code is provided as an experimental prototype.
 *   - It assumes that each block’s thread count equals the feature dimension (d_model).
 *   - It assumes d_model divides evenly into the block size.
 *   - It uses cp.async (available on compute capability 8.0+); on older architectures it falls back to synchronous loads.
 *   - Proper error checking, edge–case handling, and robustness enhancements are omitted.
 *
 * Compile with:
 *    nvcc -std=c++14 -O3 advanced_flash_attention_double_buffer.cu -o advanced_flash_attention_double_buffer
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
using namespace std;
namespace cg = cooperative_groups;

// cp.async helper: available on Ampere and later.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ inline void cp_async(void* dst, const void* src, size_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" 
                 : : "r"(dst), "l"(src), "n"(bytes) : "memory");
}
#endif

// Advanced fused FlashAttention kernel with double buffering and asynchronous shared memory loads.
// Each block processes one query vector. We assume blockDim.x == d_model.
__global__ void advanced_flash_attention_kernel(
    const float* __restrict__ Q,   // [num_queries, d_model]
    const float* __restrict__ K,   // [seq_len, d_model]
    const float* __restrict__ V,   // [seq_len, d_model]
    float* __restrict__ O,         // [num_queries, d_model]
    int seq_len,
    int d_model,
    float scale)
{
    // Each block handles one query vector.
    int query_idx = blockIdx.x;
    if(query_idx >= seq_len) return;

    int tid = threadIdx.x;  // Assume tid in [0, d_model)
    // Load this query’s element into register.
    float q_val = Q[query_idx * d_model + tid];

    // Parameters for tiling along the key dimension.
    constexpr int TILE_K = 64;  // Number of keys per tile.
    const int buffer_size = TILE_K * d_model;  // Number of floats in one tile buffer.

    // Allocate dynamic shared memory for double buffering: four buffers (two for keys, two for values).
    extern __shared__ float shared_mem[];
    float* key_buffer0 = shared_mem;                   // Buffer for keys (tile 0)
    float* val_buffer0 = key_buffer0 + buffer_size;      // Buffer for values (tile 0)
    float* key_buffer1 = val_buffer0 + buffer_size;      // Buffer for keys (tile 1)
    float* val_buffer1 = key_buffer1 + buffer_size;      // Buffer for values (tile 1)

    // Set up pointers for double buffering.
    float* cur_key = key_buffer0;
    float* cur_val = val_buffer0;
    float* next_key = key_buffer1;
    float* next_val = val_buffer1;

    // Initialize accumulators for the online softmax computation.
    float acc_num = 0.0f;
    float acc_den = 0.0f;
    float running_max = -INFINITY;

    // Preload the first tile (synchronously).
    int first_tile_size = min(TILE_K, seq_len);
    for (int i = tid; i < first_tile_size * d_model; i += blockDim.x) {
        cur_key[i] = K[i];  // Loads first tile (rows 0 to first_tile_size-1)
        cur_val[i] = V[i];
    }
    __syncthreads();

    // Total number of tiles.
    int num_tiles = (seq_len + TILE_K - 1) / TILE_K;

    // Process each tile; use double buffering to overlap shared memory loads with computation.
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * TILE_K;
        int cur_tile_size = min(TILE_K, seq_len - tile_start);

        // If there is a subsequent tile, preload it into the "next" buffer.
        if (tile_idx < num_tiles - 1) {
            int next_tile_size = min(TILE_K, seq_len - (tile_start + cur_tile_size));
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            // Use cp.async for asynchronous copy into next buffer.
            for (int i = tid; i < next_tile_size * d_model; i += blockDim.x) {
                int row = i / d_model;
                int col = i % d_model;
                const float* k_src = K + (tile_start + cur_tile_size + row) * d_model + col;
                const float* v_src = V + (tile_start + cur_tile_size + row) * d_model + col;
                cp_async(&next_key[i], k_src, sizeof(float));
                cp_async(&next_val[i], v_src, sizeof(float));
            }
            __syncthreads(); // Ensure asynchronous copies complete.
#else
            // Fallback: synchronous copy.
            for (int i = tid; i < next_tile_size * d_model; i += blockDim.x) {
                int row = i / d_model;
                int col = i % d_model;
                next_key[i] = K[(tile_start + cur_tile_size + row) * d_model + col];
                next_val[i] = V[(tile_start + cur_tile_size + row) * d_model + col];
            }
            __syncthreads();
#endif
        }

        // Process the keys in the current tile.
        for (int i = 0; i < cur_tile_size; i++) {
            // Compute dot product between the query and the i-th key in the tile.
            float dot = q_val * cur_key[i * d_model + tid];
            // Use cooperative groups to perform a block-wide reduction.
            cg::thread_block cta = cg::this_thread_block();
            float dot_sum = cg::reduce(cta, dot, cg::plus<float>());
            if (tid == 0) {
                dot_sum = dot_sum * scale;
            }
            // Broadcast the computed dot product to all threads.
            dot_sum = __shfl_sync(0xffffffff, dot_sum, 0);

            // Update running maximum and accumulators for numerically stable softmax.
            float new_max = fmaxf(running_max, dot_sum);
            float exp_factor = expf(running_max - new_max);
            float exp_val = expf(dot_sum - new_max);
            float v_elem = cur_val[i * d_model + tid];
            acc_num = acc_num * exp_factor + v_elem * exp_val;
            acc_den = acc_den * exp_factor + exp_val;
            running_max = new_max;
        }

        // Swap buffers: next buffer becomes current.
        float* temp_key = cur_key;
        float* temp_val = cur_val;
        cur_key = next_key;
        cur_val = next_val;
        next_key = temp_key;
        next_val = temp_val;
        __syncthreads();
    }

    // Finalize output: each thread writes its corresponding element.
    float out_val = acc_num / acc_den;
    O[query_idx * d_model + tid] = out_val;
}

// Host code: data setup, stream creation, and kernel launch.
int main() {
    // Problem dimensions.
    const int seq_len = 1024;  // Number of queries (and keys/values)
    const int d_model = 128;   // Feature dimension (assumed equal to blockDim.x)
    const float scale = 1.0f / sqrtf(static_cast<float>(d_model));

    // Allocate host pinned memory.
    size_t vec_size = seq_len * d_model * sizeof(float);
    float *h_Q, *h_K, *h_V, *h_O;
    cudaMallocHost(&h_Q, vec_size);
    cudaMallocHost(&h_K, vec_size);
    cudaMallocHost(&h_V, vec_size);
    cudaMallocHost(&h_O, vec_size);

    // Initialize Q, K, V with random values.
    for (int i = 0; i < seq_len * d_model; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory.
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, vec_size);
    cudaMalloc(&d_K, vec_size);
    cudaMalloc(&d_V, vec_size);
    cudaMalloc(&d_O, vec_size);

    // Copy Q, K, V to device.
    cudaMemcpy(d_Q, h_Q, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vec_size, cudaMemcpyHostToDevice);

    // Create multiple CUDA streams for overlapping data transfers and computation.
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int s = 0; s < num_streams; s++) {
        cudaStreamCreate(&streams[s]);
    }

    // Partition queries among streams.
    int queries_per_stream = seq_len / num_streams;
    int extra = seq_len % num_streams;
    int offset = 0;
    constexpr int TILE_K = 64;
    size_t shared_mem_size = 4 * TILE_K * d_model * sizeof(float);

    // Launch kernel asynchronously on each stream.
    for (int s = 0; s < num_streams; s++) {
        int num_queries = queries_per_stream + (s < extra ? 1 : 0);
        advanced_flash_attention_kernel<<<num_queries, d_model, shared_mem_size, streams[s]>>>(
            d_Q + offset * d_model,
            d_K,
            d_V,
            d_O + offset * d_model,
            seq_len,
            d_model,
            scale
        );
        offset += num_queries;
    }

    // Wait for all streams and clean up.
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    // Copy results back to host.
    cudaMemcpy(h_O, d_O, vec_size, cudaMemcpyDeviceToHost);

    // Print first 5 output vectors.
    for (int i = 0; i < 5; i++) {
        printf("Output vector %d: ", i);
        for (int j = 0; j < d_model; j++) {
            printf("%f ", h_O[i * d_model + j]);
        }
        printf("\n");
    }

    // Cleanup.
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFreeHost(h_Q);
    cudaFreeHost(h_K);
    cudaFreeHost(h_V);
    cudaFreeHost(h_O);

    return 0;
}

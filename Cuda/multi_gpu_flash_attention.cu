/*
 * Multi-GPU Advanced FlashAttention with NCCL and CUDA Streams
 *
 * This experimental prototype extends the advanced FlashAttention kernel to a multi-GPU setup.
 * It partitions the overall batch of queries among available GPUs, each of which launches the
 * fused FlashAttention kernel (with asynchronous double buffering, cp.async, and cooperative groups)
 * and then copies its output back. NCCL is used to initialize a communicator across devices,
 * so that the design can later be extended to perform inter-device collectives (e.g., for gradient
 * synchronization in training).
 *
 * NOTE:
 *  - This code is experimental and intended for research and educational purposes.
 *  - It targets NVIDIA GPUs with Compute Capability 8.0+ for cp.async (with fallback for older GPUs).
 *  - The multi-GPU partitioning is done via host threads (one per GPU). In a production setting,
 *    further integration with frameworks like PyTorch DDP or TensorFlow's distribution strategies is needed.
 *
 * Compile with:
 *    nvcc -std=c++14 -O3 multi_gpu_flash_attention.cu -lnccl -lpthread -o multi_gpu_flash_attention
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <pthread.h>
using namespace std;
namespace cg = cooperative_groups;

// cp.async helper: available on Ampere and later.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ inline void cp_async(void* dst, const void* src, size_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" 
                 : : "r"(dst), "l"(src), "n"(bytes) : "memory");
}
#endif

// Advanced FlashAttention Kernel (per GPU).
// Each CUDA block processes one query vector.
__global__ void advanced_flash_attention_kernel(
    const float* __restrict__ Q,   // [local_seq_len, d_model]
    const float* __restrict__ K,   // [global_seq_len, d_model]
    const float* __restrict__ V,   // [global_seq_len, d_model]
    float* __restrict__ O,         // [local_seq_len, d_model]
    int global_seq_len,
    int d_model,
    float scale)
{
    // Each block processes one query vector.
    int query_idx = blockIdx.x;
    if (query_idx >= gridDim.x) return;

    int tid = threadIdx.x;  // Assumed: blockDim.x == d_model
    float q_val = Q[query_idx * d_model + tid];

    constexpr int TILE_K = 64;
    const int buffer_size = TILE_K * d_model;

    extern __shared__ float shared_mem[];
    float* key_buffer0 = shared_mem;
    float* val_buffer0 = key_buffer0 + buffer_size;
    float* key_buffer1 = val_buffer0 + buffer_size;
    float* val_buffer1 = key_buffer1 + buffer_size;

    float* cur_key = key_buffer0;
    float* cur_val = val_buffer0;
    float* next_key = key_buffer1;
    float* next_val = val_buffer1;

    float acc_num = 0.0f;
    float acc_den = 0.0f;
    float running_max = -INFINITY;

    int first_tile_size = min(TILE_K, global_seq_len);
    for (int i = tid; i < first_tile_size * d_model; i += blockDim.x) {
        cur_key[i] = K[i];
        cur_val[i] = V[i];
    }
    __syncthreads();

    int num_tiles = (global_seq_len + TILE_K - 1) / TILE_K;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * TILE_K;
        int cur_tile_size = min(TILE_K, global_seq_len - tile_start);

        if (tile_idx < num_tiles - 1) {
            int next_tile_size = min(TILE_K, global_seq_len - (tile_start + cur_tile_size));
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            for (int i = tid; i < next_tile_size * d_model; i += blockDim.x) {
                int row = i / d_model;
                int col = i % d_model;
                const float* k_src = K + (tile_start + cur_tile_size + row) * d_model + col;
                const float* v_src = V + (tile_start + cur_tile_size + row) * d_model + col;
                cp_async(&next_key[i], k_src, sizeof(float));
                cp_async(&next_val[i], v_src, sizeof(float));
            }
            __syncthreads();
#else
            for (int i = tid; i < next_tile_size * d_model; i += blockDim.x) {
                int row = i / d_model;
                int col = i % d_model;
                next_key[i] = K[(tile_start + cur_tile_size + row) * d_model + col];
                next_val[i] = V[(tile_start + cur_tile_size + row) * d_model + col];
            }
            __syncthreads();
#endif
        }

        for (int i = 0; i < cur_tile_size; i++) {
            float dot = q_val * cur_key[i * d_model + tid];
            cg::thread_block cta = cg::this_thread_block();
            float dot_sum = cg::reduce(cta, dot, cg::plus<float>());
            if (tid == 0) {
                dot_sum = dot_sum * scale;
            }
            dot_sum = __shfl_sync(0xffffffff, dot_sum, 0);
            float new_max = fmaxf(running_max, dot_sum);
            float exp_factor = expf(running_max - new_max);
            float exp_val = expf(dot_sum - new_max);
            float v_elem = cur_val[i * d_model + tid];
            acc_num = acc_num * exp_factor + v_elem * exp_val;
            acc_den = acc_den * exp_factor + exp_val;
            running_max = new_max;
        }

        float* temp_key = cur_key;
        float* temp_val = cur_val;
        cur_key = next_key;
        cur_val = next_val;
        next_key = temp_key;
        next_val = temp_val;
        __syncthreads();
    }

    O[query_idx * d_model + tid] = acc_num / acc_den;
}

// Structure to hold per-GPU data.
struct ThreadData {
    int device;          // GPU device id
    int gpu_id;          // Same as device id
    int num_gpus;        // Total number of GPUs
    int global_seq_len;  // Total number of keys (common to all GPUs)
    int local_seq_len;   // Number of queries assigned to this GPU
    int d_model;         // Feature dimension
    float scale;         // Scaling factor for dot products
    float *h_Q;          // Host pointer for queries (local partition)
    float *h_K;          // Host pointer for keys (global)
    float *h_V;          // Host pointer for values (global)
    float *h_O;          // Host pointer for output (local partition)
    ncclComm_t comm;     // NCCL communicator for this GPU
};

// GPU thread function: sets device, copies data, launches kernel, copies results back.
void* gpuThreadFunc(void* arg) {
    ThreadData* data = (ThreadData*) arg;
    cudaSetDevice(data->device);

    size_t q_size = data->local_seq_len * data->d_model * sizeof(float);
    size_t kv_size = data->global_seq_len * data->d_model * sizeof(float);
    float *d_Q, *d_O, *d_K, *d_V;
    cudaMalloc(&d_Q, q_size);
    cudaMalloc(&d_O, q_size);
    cudaMalloc(&d_K, kv_size);
    cudaMalloc(&d_V, kv_size);

    cudaMemcpy(d_Q, data->h_Q, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, data->h_K, kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, data->h_V, kv_size, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threads_per_block = data->d_model; // Block size equals d_model.
    int blocks = data->local_seq_len;
    size_t shared_mem_size = 4 * 64 * data->d_model * sizeof(float); // TILE_K = 64
    advanced_flash_attention_kernel<<<blocks, threads_per_block, shared_mem_size, stream>>>(
        d_Q, d_K, d_V, d_O, data->global_seq_len, data->d_model, data->scale);

    cudaStreamSynchronize(stream);
    cudaMemcpy(data->h_O, d_O, q_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_O);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaStreamDestroy(stream);
    return nullptr;
}

int main() {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("Found %d GPUs\n", num_gpus);

    int global_seq_len = 1024; // Total number of keys/queries.
    int d_model = 128;         // Feature dimension.
    float scale = 1.0f / sqrtf((float)d_model);

    int queries_per_gpu = global_seq_len / num_gpus;
    int remainder = global_seq_len % num_gpus;

    size_t kv_size = global_seq_len * d_model * sizeof(float);
    float *h_K, *h_V;
    cudaMallocHost(&h_K, kv_size);
    cudaMallocHost(&h_V, kv_size);

    for (int i = 0; i < global_seq_len * d_model; i++) {
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float **h_Q_parts = new float*[num_gpus];
    float **h_O_parts = new float*[num_gpus];
    int* local_seq_lens = new int[num_gpus];
    int offset = 0;
    for (int i = 0; i < num_gpus; i++) {
        int local_seq = queries_per_gpu + (i < remainder ? 1 : 0);
        local_seq_lens[i] = local_seq;
        size_t q_size = local_seq * d_model * sizeof(float);
        cudaMallocHost(&h_Q_parts[i], q_size);
        cudaMallocHost(&h_O_parts[i], q_size);
        for (int j = 0; j < local_seq * d_model; j++) {
            h_Q_parts[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
        offset += local_seq;
    }

    ncclComm_t* comms = new ncclComm_t[num_gpus];
    int* devices = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) {
        devices[i] = i;
    }
    ncclCommInitAll(comms, num_gpus, devices);

    pthread_t* threads = new pthread_t[num_gpus];
    ThreadData* threadData = new ThreadData[num_gpus];
    for (int i = 0; i < num_gpus; i++) {
        threadData[i].device = i;
        threadData[i].gpu_id = i;
        threadData[i].num_gpus = num_gpus;
        threadData[i].global_seq_len = global_seq_len;
        threadData[i].local_seq_len = local_seq_lens[i];
        threadData[i].d_model = d_model;
        threadData[i].scale = scale;
        threadData[i].h_Q = h_Q_parts[i];
        threadData[i].h_K = h_K;
        threadData[i].h_V = h_V;
        threadData[i].h_O = h_O_parts[i];
        threadData[i].comm = comms[i];
        pthread_create(&threads[i], nullptr, gpuThreadFunc, &threadData[i]);
    }

    for (int i = 0; i < num_gpus; i++) {
        pthread_join(threads[i], nullptr);
    }

    for (int i = 0; i < num_gpus; i++) {
        printf("GPU %d output (first 5 values):\n", i);
        int local_seq = local_seq_lens[i];
        for (int j = 0; j < min(5, local_seq * d_model); j++) {
            printf("%f ", h_O_parts[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < num_gpus; i++) {
        ncclCommDestroy(comms[i]);
        cudaFreeHost(h_Q_parts[i]);
        cudaFreeHost(h_O_parts[i]);
    }
    delete[] h_Q_parts;
    delete[] h_O_parts;
    delete[] local_seq_lens;
    delete[] comms;
    delete[] devices;
    delete[] threads;
    delete[] threadData;
    cudaFreeHost(h_K);
    cudaFreeHost(h_V);

    return 0;
}

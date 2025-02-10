import os
import math
import torch
import torch.distributed as dist
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton Kernel: FlashAttention with Tiling and Online Softmax
#
# Each kernel instance processes one query vector. The kernel iterates over the
# key/value matrices in tiles (of size TILE_K) and computes the scaled dot product
# for each key. It then updates online accumulators for the weighted sum of values
# and the softmax normalization in a numerically stable manner.
#
# Note: Triton does not yet support cp.async directly, so this version uses tl.load
# with appropriate masking and relies on the Triton scheduler to hide memory latency.
# ---------------------------------------------------------------------------
@triton.jit
def flash_attention_kernel(
    Q,       # pointer to queries [local_seq_len, d_model]
    K,       # pointer to keys [global_seq_len, d_model]
    V,       # pointer to values [global_seq_len, d_model]
    O,       # pointer to output [local_seq_len, d_model]
    global_seq_len: tl.constexpr,  # number of keys (global)
    d_model: tl.constexpr,         # feature dimension
    scale: tl.constexpr,           # scaling factor for dot product
):
    # Each program instance handles one query vector.
    pid = tl.program_id(0)
    q_ptr = Q + pid * d_model
    o_ptr = O + pid * d_model
    # Load the query vector.
    q = tl.load(q_ptr, mask=tl.arange(0, d_model) < d_model)

    # Initialize accumulators.
    acc_num = tl.zeros([d_model], dtype=tl.float32)
    acc_den = 0.0
    running_max = -1e9  # Very small initial value.

    TILE_K = 64
    num_tiles = tl.cdiv(global_seq_len, TILE_K)

    for tile in range(num_tiles):
        start = tile * TILE_K
        cur_tile_size = tl.min(TILE_K, global_seq_len - start)
        offset = start * d_model

        # Load keys and values as flattened tensors.
        keys_flat = tl.load(K + offset, mask=tl.arange(0, cur_tile_size * d_model) < cur_tile_size * d_model, other=0.0)
        values_flat = tl.load(V + offset, mask=tl.arange(0, cur_tile_size * d_model) < cur_tile_size * d_model, other=0.0)
        key_tile = tl.reshape(keys_flat, (cur_tile_size, d_model))
        value_tile = tl.reshape(values_flat, (cur_tile_size, d_model))

        for i in range(cur_tile_size):
            key_vec = key_tile[i, :]
            dot = tl.dot(q, key_vec) * scale
            new_max = tl.maximum(running_max, dot)
            exp_factor = tl.exp(running_max - new_max)
            exp_val = tl.exp(dot - new_max)
            val_vec = value_tile[i, :]
            acc_num = acc_num * exp_factor + val_vec * exp_val
            acc_den = acc_den * exp_factor + exp_val
            running_max = new_max

    result = acc_num / acc_den
    tl.store(o_ptr, result)

def flash_attention(Q, K, V):
    """
    Q: [local_seq_len, d_model] tensor on device.
    K: [global_seq_len, d_model] tensor on device (replicated on all GPUs).
    V: [global_seq_len, d_model] tensor on device (replicated on all GPUs).
    Returns:
       O: [local_seq_len, d_model] tensor on device.
    """
    local_seq_len, d_model = Q.shape
    global_seq_len, _ = K.shape
    O = torch.empty_like(Q)
    grid = (local_seq_len,)  # One kernel instance per query.
    flash_attention_kernel[grid](
        Q, K, V, O,
        global_seq_len, d_model, 1.0/math.sqrt(d_model),
        BLOCK_SIZE = d_model
    )
    return O

def main():
    # Distributed initialization.
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    device = torch.device('cuda', rank)

    global_seq_len = 1024   # Total keys/values.
    d_model = 128           # Feature dimension.
    total_queries = 1024    # Total number of queries.

    local_queries = total_queries // world_size
    if rank < total_queries % world_size:
        local_queries += 1

    Q_local = torch.randn(local_queries, d_model, device=device, dtype=torch.float32)
    K = torch.randn(global_seq_len, d_model, device=device, dtype=torch.float32)
    V = torch.randn(global_seq_len, d_model, device=device, dtype=torch.float32)

    O_local = flash_attention(Q_local, K, V)

    if world_size > 1:
        output_list = [torch.empty_like(O_local) for _ in range(world_size)]
        dist.all_gather(output_list, O_local)
        O_total = torch.cat(output_list, dim=0)
        if rank == 0:
            print("Gathered output shape:", O_total.shape)
    else:
        print("Output shape:", O_local.shape)

if __name__ == '__main__':
    main()

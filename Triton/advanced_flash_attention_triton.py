"""
This implementation builds upon a previous version by adding:
  - Parameterization: Allows dynamic adjustment of the tile size.
  - Dynamic Tiling: Processes keys/values in adjustable tiles (tile_k) to better match hardware capabilities.
  - Improved Numerical Robustness: Uses Python’s negative infinity (float('-inf')) for initializing running_max to ensure stability.
  - Extra Logging for Debugging: Optional debug prints controlled by the environment variable DEBUG_FLASHATTN.
  
This file is meant as a research prototype that pushes the envelope in efficiency and robustness for attention computations,
and it integrates seamlessly with PyTorch Distributed for multi-GPU setups.
"""

import os
import math
import torch
import torch.distributed as dist
import triton
import triton.language as tl

# Debug flag; set environment variable DEBUG_FLASHATTN=1 to enable debug prints.
DEBUG = os.environ.get("DEBUG_FLASHATTN", "0") == "1"

@triton.jit
def advanced_flash_attention_kernel(
    Q,       # pointer to queries [local_seq_len, d_model]
    K,       # pointer to keys [global_seq_len, d_model]
    V,       # pointer to values [global_seq_len, d_model]
    O,       # pointer to output [local_seq_len, d_model]
    global_seq_len: tl.constexpr,  # total number of keys (global)
    d_model: tl.constexpr,         # feature dimension
    scale: tl.constexpr,           # scaling factor for dot product
    TILE_K: tl.constexpr,          # tile size (number of keys processed per tile)
):
    """
    Advanced Triton kernel for FlashAttention with dynamic tiling and enhanced numerical robustness.
    This kernel computes a fused attention result by performing an online softmax update.
    """
    pid = tl.program_id(0)
    q_ptr = Q + pid * d_model
    o_ptr = O + pid * d_model

    # Load query vector.
    q = tl.load(q_ptr, mask=tl.arange(0, d_model) < d_model)
    
    # Initialize accumulators.
    acc_num = tl.zeros([d_model], dtype=tl.float32)
    acc_den = 0.0
    # Use negative infinity for robust initialization.
    running_max = float('-inf')

    # Compute number of tiles.
    num_tiles = tl.cdiv(global_seq_len, TILE_K)

    # Optional debug print.
    if DEBUG:
        tl.printf("PID %d: global_seq_len=%d, d_model=%d, TILE_K=%d, num_tiles=%d\n", pid, global_seq_len, d_model, TILE_K, num_tiles)

    for tile in range(num_tiles):
        start = tile * TILE_K
        cur_tile_size = tl.min(TILE_K, global_seq_len - start)
        offset = start * d_model

        # Load key and value tile as flattened tensors.
        keys_flat = tl.load(K + offset, mask=tl.arange(0, cur_tile_size * d_model) < cur_tile_size * d_model, other=0.0)
        values_flat = tl.load(V + offset, mask=tl.arange(0, cur_tile_size * d_model) < cur_tile_size * d_model, other=0.0)
        key_tile = tl.reshape(keys_flat, (cur_tile_size, d_model))
        value_tile = tl.reshape(values_flat, (cur_tile_size, d_model))

        for i in range(cur_tile_size):
            key_vec = key_tile[i, :]
            # Compute scaled dot product.
            dot = tl.dot(q, key_vec) * scale
            # Robust online softmax update.
            new_max = tl.maximum(running_max, dot)
            exp_factor = tl.exp(running_max - new_max)
            exp_val = tl.exp(dot - new_max)
            val_vec = value_tile[i, :]
            acc_num = acc_num * exp_factor + val_vec * exp_val
            acc_den = acc_den * exp_factor + exp_val
            running_max = new_max

            if DEBUG and i == 0:
                tl.printf("Tile %d, i=%d: dot=%f, new_max=%f, exp_val=%f\n", tile, i, dot, new_max, exp_val)

    result = acc_num / acc_den
    tl.store(o_ptr, result)

def advanced_flash_attention(Q, K, V, tile_k=64):
    """
    Wrapper for the advanced FlashAttention kernel.
    
    Parameters:
      Q: [local_seq_len, d_model] tensor on device.
      K: [global_seq_len, d_model] tensor on device (replicated on all GPUs).
      V: [global_seq_len, d_model] tensor on device.
      tile_k: Tile size for processing keys (default: 64).
    
    Returns:
      O: [local_seq_len, d_model] tensor on device containing the attention outputs.
    """
    local_seq_len, d_model = Q.shape
    global_seq_len, _ = K.shape
    O = torch.empty_like(Q)
    grid = (local_seq_len,)  # One kernel instance per query.
    advanced_flash_attention_kernel[grid](
        Q, K, V, O,
        global_seq_len, d_model, 1.0/math.sqrt(d_model),
        tile_k,
        BLOCK_SIZE = d_model
    )
    return O

def main():
    # Distributed initialization if in a multi-GPU setting.
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    device = torch.device('cuda', rank)
    
    # Define dimensions.
    global_seq_len = 1024   # Total number of keys/values.
    d_model = 128           # Feature dimension.
    total_queries = 1024    # Total number of queries.

    # Partition queries among GPUs.
    local_queries = total_queries // world_size
    if rank < total_queries % world_size:
        local_queries += 1

    # Create random tensors on the device.
    Q_local = torch.randn(local_queries, d_model, device=device, dtype=torch.float32)
    K = torch.randn(global_seq_len, d_model, device=device, dtype=torch.float32)
    V = torch.randn(global_seq_len, d_model, device=device, dtype=torch.float32)

    # Compute the advanced flash attention output.
    O_local = advanced_flash_attention(Q_local, K, V, tile_k=64)

    # Optionally, gather outputs from all GPUs.
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

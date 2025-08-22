"""
Ring Attention Implementation for Distributed Long-Context Processing

Implements Ring Attention for near-infinite context by distributing sequences
across devices with overlapping communication and computation. Achieves linear
memory scaling with the number of devices.

References:
- Ring Attention with Blockwise Transformers for Near-Infinite Context
- https://arxiv.org/abs/2310.01889
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import torch.nn.functional as F
from types import SimpleNamespace

from .config import StreamAttentionConfig
from .flashattention_v3 import FlashAttentionV3

logger = logging.getLogger(__name__)


@dataclass
class RingAttentionState:
    """State for Ring Attention computation"""

    # Current block indices
    query_block_idx: int = 0
    key_value_block_idx: int = 0

    # Accumulated attention outputs
    output: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None  # Log-sum-exp for numerical stability

    # Communication buffers
    send_buffer: Optional[torch.Tensor] = None
    recv_buffer: Optional[torch.Tensor] = None

    # Statistics
    total_seq_len: int = 0
    blocks_processed: int = 0
    communication_time: float = 0.0
    computation_time: float = 0.0


class RingCommunicator:
    """Handles ring communication pattern for distributed attention"""

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None):
        self.process_group = process_group or dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        # Ring topology
        self.next_rank = (self.rank + 1) % self.world_size
        self.prev_rank = (self.rank - 1 + self.world_size) % self.world_size

        # Communication handles
        self.send_handle = None
        self.recv_handle = None

    @contextmanager
    def communication_context(self):
        """Context manager for overlapped communication"""
        try:
            yield self
        finally:
            # Ensure all communications complete
            if self.send_handle is not None:
                self.send_handle.wait()
                self.send_handle = None
            if self.recv_handle is not None:
                self.recv_handle.wait()
                self.recv_handle = None

    def isend(self, tensor: torch.Tensor, dst: Optional[int] = None) -> dist.Work:
        """Non-blocking send to next rank in ring"""
        dst = dst or self.next_rank
        self.send_handle = dist.isend(tensor, dst, group=self.process_group)
        return self.send_handle

    def irecv(self, tensor: torch.Tensor, src: Optional[int] = None) -> dist.Work:
        """Non-blocking receive from previous rank in ring"""
        src = src or self.prev_rank
        self.recv_handle = dist.irecv(tensor, src, group=self.process_group)
        return self.recv_handle

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
        send_dst: Optional[int] = None,
        recv_src: Optional[int] = None,
    ) -> Tuple[dist.Work, dist.Work]:
        """Simultaneous send and receive for ring communication"""
        send_handle = self.isend(send_tensor, send_dst)
        recv_handle = self.irecv(recv_tensor, recv_src)
        return send_handle, recv_handle

    def wait_all(self):
        """Wait for all pending communications"""
        if self.send_handle:
            self.send_handle.wait()
        if self.recv_handle:
            self.recv_handle.wait()


class RingAttention(nn.Module):
    """
    Ring Attention for distributed long-context processing

    Features:
    - Distributes sequence across multiple GPUs/nodes
    - Overlaps communication with computation
    - Supports sequences of virtually unlimited length
    - Compatible with FlashAttention optimizations
    - Gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        config: StreamAttentionConfig,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.config = config
        self.process_group = process_group

        # Initialize ring communicator
        if dist.is_initialized():
            self.ring_comm = RingCommunicator(process_group)
            self.world_size = self.ring_comm.world_size
            self.rank = self.ring_comm.rank
        else:
            self.ring_comm = None
            self.world_size = 1
            self.rank = 0
            logger.warning(
                "Distributed not initialized, Ring Attention running in single-device mode"
            )

        # Local attention module (FlashAttention V3 for efficiency)
        self.local_attention = FlashAttentionV3(config)

        # Block size for sequence partitioning
        self.block_size = config.ring_attention_block_size
        self.overlap_size = config.ring_attention_overlap_size

        # Gradient checkpointing
        self.use_gradient_checkpointing = config.gradient_checkpointing

        logger.info(f"Ring Attention initialized: rank {self.rank}/{self.world_size}")

    def _partition_sequence(
        self, tensor: torch.Tensor, scatter: bool = True
    ) -> torch.Tensor:
        """Partition sequence across devices"""
        batch_size, seq_len, num_heads, head_dim = tensor.shape

        # Calculate local sequence length
        local_seq_len = seq_len // self.world_size
        remainder = seq_len % self.world_size

        if self.rank < remainder:
            local_seq_len += 1
            start_idx = self.rank * local_seq_len
        else:
            start_idx = self.rank * local_seq_len + remainder

        end_idx = start_idx + local_seq_len

        if scatter and self.world_size > 1:
            # Use scatter operation for efficient distribution
            scattered = torch.empty(
                batch_size,
                local_seq_len,
                num_heads,
                head_dim,
                dtype=tensor.dtype,
                device=tensor.device,
            )

            # Prepare chunks for scatter
            if self.rank == 0:
                chunks = tensor.chunk(self.world_size, dim=1)
                # Pad last chunk if necessary
                if chunks[-1].shape[1] < local_seq_len:
                    pad_size = local_seq_len - chunks[-1].shape[1]
                    chunks[-1] = F.pad(chunks[-1], (0, 0, 0, 0, 0, pad_size))
            else:
                chunks = None

            dist.scatter(scattered, chunks, src=0, group=self.process_group)
            return scattered
        else:
            # Direct slicing for single device or gather operations
            return tensor[:, start_idx:end_idx]

    def _compute_block_attention(
        self,
        query_block: torch.Tensor,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
        state: RingAttentionState,
        causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention for a single block pair"""

        batch_size, q_len, num_heads, head_dim = query_block.shape
        _, kv_len, _, _ = key_block.shape

        # Compute local attention scores
        scores = torch.matmul(
            query_block.transpose(1, 2), key_block.transpose(1, 2).transpose(-2, -1)
        ) / math.sqrt(head_dim)

        # Apply causal mask if needed
        if causal_mask:
            # Calculate global positions
            q_pos = state.query_block_idx * self.block_size + torch.arange(
                q_len, device=query_block.device
            )
            kv_pos = state.key_value_block_idx * self.block_size + torch.arange(
                kv_len, device=key_block.device
            )

            mask = q_pos.unsqueeze(1) >= kv_pos.unsqueeze(0)
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Compute log-sum-exp for numerical stability
        max_scores = scores.max(dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(scores - max_scores)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)

        # Compute attention weights
        attn_weights = exp_scores / sum_exp

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_block.transpose(1, 2))
        attn_output = attn_output.transpose(1, 2)

        # Compute log-sum-exp for combining with other blocks
        lse = max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1))

        return attn_output, lse

    def _combine_block_outputs(
        self,
        output1: torch.Tensor,
        lse1: torch.Tensor,
        output2: torch.Tensor,
        lse2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine outputs from different blocks using log-sum-exp trick"""

        # Compute combined log-sum-exp
        max_lse = torch.maximum(lse1, lse2)
        exp_diff1 = torch.exp(lse1 - max_lse).unsqueeze(-1)
        exp_diff2 = torch.exp(lse2 - max_lse).unsqueeze(-1)

        # Weighted combination
        combined_output = output1 * exp_diff1 + output2 * exp_diff2
        combined_lse = max_lse + torch.log(
            exp_diff1.squeeze(-1) + exp_diff2.squeeze(-1)
        )

        return combined_output, combined_lse

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Ring Attention forward pass

        Args:
            query: [batch_size, seq_len, num_heads, head_dim] - distributed across devices
            key: [batch_size, seq_len, num_heads, head_dim] - distributed across devices
            value: [batch_size, seq_len, num_heads, head_dim] - distributed across devices
            attention_mask: Optional attention mask
            causal: Whether to use causal masking

        Returns:
            output: [batch_size, local_seq_len, num_heads, head_dim]
        """

        if self.world_size == 1:
            # Single device - use local attention directly
            return self.local_attention(query, key, value, attention_mask, causal)

        batch_size, local_seq_len, num_heads, head_dim = query.shape
        device = query.device

        # Initialize state
        state = RingAttentionState(total_seq_len=local_seq_len * self.world_size)

        # Initialize output and LSE
        output = torch.zeros_like(query)
        lse = torch.full(
            (batch_size, num_heads, local_seq_len),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )

        # Prepare communication buffers
        kv_buffer_shape = (batch_size, local_seq_len, num_heads, head_dim)
        send_buffer = torch.stack([key.clone(), value.clone()], dim=0).contiguous()
        recv_buffer = torch.empty_like(send_buffer)

        # Current KV blocks
        curr_k = key
        curr_v = value

        with self.ring_comm.communication_context():
            # Ring iterations
            for step in range(self.world_size):
                # Determine which KV block we're processing
                kv_block_idx = (self.rank - step + self.world_size) % self.world_size
                state.key_value_block_idx = kv_block_idx

                # Start async communication for next iteration (except last)
                if step < self.world_size - 1:
                    import time

                    comm_start = time.time()

                    # Send current KV to next rank, receive from previous
                    send_handle, recv_handle = self.ring_comm.send_recv(
                        send_buffer, recv_buffer
                    )

                # Compute attention for current block
                comp_start = time.time()

                # Capture immutable indices for checkpoint to avoid mutable state issues
                local_state = SimpleNamespace(
                    query_block_idx=state.query_block_idx,
                    key_value_block_idx=state.key_value_block_idx,
                )

                if self.use_gradient_checkpointing and self.training:
                    # Use gradient checkpointing to save memory
                    block_output, block_lse = torch.utils.checkpoint.checkpoint(
                        self._compute_block_attention,
                        query,
                        curr_k,
                        curr_v,
                        local_state,
                        causal,
                    )
                else:
                    block_output, block_lse = self._compute_block_attention(
                        query, curr_k, curr_v, local_state, causal
                    )

                state.computation_time += time.time() - comp_start

                # Combine with previous results
                if step == 0:
                    output = block_output
                    lse = block_lse
                else:
                    output, lse = self._combine_block_outputs(
                        output, lse, block_output, block_lse
                    )

                # Wait for communication and swap buffers
                if step < self.world_size - 1:
                    send_handle.wait()
                    recv_handle.wait()
                    state.communication_time += time.time() - comm_start

                    # Unstack received tensors
                    curr_k, curr_v = recv_buffer[0], recv_buffer[1]

                    # Swap buffers for next iteration
                    send_buffer, recv_buffer = recv_buffer, send_buffer

                state.blocks_processed += 1

        # Log performance statistics
        if state.blocks_processed > 0:
            logger.debug(
                f"Ring Attention stats - Rank {self.rank}: "
                f"Blocks: {state.blocks_processed}, "
                f"Comp time: {state.computation_time:.3f}s, "
                f"Comm time: {state.communication_time:.3f}s, "
                f"Overlap: {state.communication_time / state.computation_time:.2%}"
            )

        return output

    def forward_with_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with KV caching support for generation"""

        # Concatenate with past KV if provided
        if past_key_values is not None:
            past_key, past_value = past_key_values
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)

        # Run ring attention
        output = self.forward(query, key, value, causal=causal)

        # Return updated cache if requested
        if use_cache:
            return output, (key, value)
        else:
            return output, None


def create_ring_attention(
    config: StreamAttentionConfig, process_group: Optional[dist.ProcessGroup] = None
) -> RingAttention:
    """Factory function to create Ring Attention instance"""
    return RingAttention(config, process_group)

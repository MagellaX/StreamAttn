"""
Star Attention Implementation with 2-Phase Approach

Implements Star Attention for efficient long-context inference achieving up to
11x speedup while maintaining 95-100% accuracy. Uses a two-phase approach:
Phase 1: Local context encoding with anchor blocks
Phase 2: Query encoding with global attention

References:
- Star Attention: Efficient LLM Inference over Long Sequences
- https://arxiv.org/abs/2411.17116
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, NamedTuple
import logging
from dataclasses import dataclass
import time

from .config import StreamAttentionConfig
from .flashattention_v3 import FlashAttentionV3

logger = logging.getLogger(__name__)


class AttentionStats(NamedTuple):
    """Statistics from attention computation"""

    attention_weights: torch.Tensor
    max_scores: torch.Tensor
    sum_exp: torch.Tensor


@dataclass
class StarAttentionState:
    """State for Star Attention computation"""

    # Phase tracking
    current_phase: int = 1

    # Context blocks
    context_blocks: List[torch.Tensor] = None
    anchor_block: Optional[torch.Tensor] = None

    # KV caches per host
    kv_caches: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = None

    # Query host info
    is_query_host: bool = False
    query_host_id: int = 0

    # Performance stats
    phase1_time: float = 0.0
    phase2_time: float = 0.0
    communication_time: float = 0.0
    speedup_achieved: float = 1.0


class StarAttention(nn.Module):
    """
    Star Attention for efficient long-context inference

    Key innovations:
    - Two-phase attention: local context encoding + global query processing
    - Anchor blocks to maintain attention distribution
    - Distributed KV cache with minimal communication
    - Up to 11x speedup with 95-100% accuracy
    """

    def __init__(self, config: StreamAttentionConfig):
        super().__init__()
        self.config = config

        # Star Attention specific parameters
        self.block_size = config.star_attention_block_size
        self.anchor_block_size = config.star_attention_anchor_size
        self.num_hosts = config.star_attention_num_hosts

        # Determine if distributed mode
        self.distributed = torch.distributed.is_initialized() and self.num_hosts > 1
        if self.distributed:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            # Designate host 0 as query host by default
            self.query_host_id = 0
            self.is_query_host = self.rank == self.query_host_id
        else:
            self.rank = 0
            self.world_size = 1
            self.query_host_id = 0
            self.is_query_host = True

        # Local attention module (use FlashAttention V3 for efficiency)
        self.local_attention = FlashAttentionV3(config)

        # Attention aggregation parameters
        self.use_log_sum_exp = True  # For numerical stability
        self.eps = 1e-6

        # Performance optimization flags
        self.enable_kv_compression = config.enable_kv_compression
        self.compression_ratio = config.kv_compression_ratio

        logger.info(
            f"Star Attention initialized: "
            f"Block size: {self.block_size}, "
            f"Anchor size: {self.anchor_block_size}, "
            f"Hosts: {self.num_hosts}, "
            f"Query host: {self.is_query_host}"
        )

    def _create_context_blocks(
        self, context: torch.Tensor, include_anchor: bool = True
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Partition context into blocks with optional anchor block

        Args:
            context: [batch_size, seq_len, num_heads, head_dim]
            include_anchor: Whether to prepend anchor block to each block

        Returns:
            blocks: List of context blocks
            anchor_block: The first block used as anchor
        """
        batch_size, seq_len, num_heads, head_dim = context.shape

        # Split context into blocks
        num_blocks = math.ceil(seq_len / self.block_size)
        blocks = []

        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, seq_len)
            block = context[:, start_idx:end_idx]

            # Pad last block if necessary
            if block.shape[1] < self.block_size and i < num_blocks - 1:
                pad_size = self.block_size - block.shape[1]
                block = F.pad(block, (0, 0, 0, 0, 0, pad_size))

            blocks.append(block)

        # Extract anchor block (first block)
        anchor_block = (
            blocks[0][:, : self.anchor_block_size] if include_anchor else None
        )

        return blocks, anchor_block

    def _distribute_blocks_to_hosts(
        self, blocks: List[torch.Tensor], anchor_block: Optional[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Distribute context blocks across hosts"""

        if not self.distributed:
            return blocks

        # Calculate blocks per host
        blocks_per_host = len(blocks) // self.num_hosts
        remainder = len(blocks) % self.num_hosts

        # Determine which blocks this host processes
        if self.rank < remainder:
            start_idx = self.rank * (blocks_per_host + 1)
            end_idx = start_idx + blocks_per_host + 1
        else:
            start_idx = self.rank * blocks_per_host + remainder
            end_idx = start_idx + blocks_per_host

        host_blocks = blocks[start_idx:end_idx]

        # Add anchor block to all blocks except the first
        if anchor_block is not None and start_idx > 0:
            augmented_blocks = []
            for block in host_blocks:
                # Concatenate anchor block with current block
                augmented_block = torch.cat([anchor_block, block], dim=1)
                augmented_blocks.append(augmented_block)
            return augmented_blocks
        else:
            return host_blocks

    def phase1_context_encoding(
        self, context: torch.Tensor, state: StarAttentionState
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Phase 1: Blockwise local attention for context encoding

        Each host processes its assigned blocks with local attention,
        using anchor blocks to maintain proper attention distribution.
        """

        start_time = time.time()

        # Create and distribute context blocks
        blocks, anchor_block = self._create_context_blocks(context, include_anchor=True)
        state.anchor_block = anchor_block

        # Distribute blocks to hosts
        if self.distributed:
            host_blocks = self._distribute_blocks_to_hosts(blocks, anchor_block)
        else:
            # Single host processes all blocks
            host_blocks = blocks
            if anchor_block is not None:
                # Add anchor to all blocks except first
                augmented_blocks = [blocks[0]]  # First block as-is
                for block in blocks[1:]:
                    augmented_block = torch.cat([anchor_block, block], dim=1)
                    augmented_blocks.append(augmented_block)
                host_blocks = augmented_blocks

        # Process blocks with local attention
        kv_cache = {}

        for block_idx, block in enumerate(host_blocks):
            batch_size, block_len, num_heads, head_dim = block.shape

            # Self-attention on block
            # Using causal=False for within-block attention as per Star Attention
            block_output = self.local_attention(
                query=block,
                key=block,
                value=block,
                causal=False,  # Local blocks use full attention
            )

            # Extract KV pairs (exclude anchor block KVs)
            if anchor_block is not None and block_idx > 0:
                # Remove anchor block portion from KV cache
                start_idx = self.anchor_block_size
                key_cache = block[:, start_idx:]
                value_cache = block[:, start_idx:]
            else:
                key_cache = block
                value_cache = block

            # Store in KV cache
            block_id = self.rank * len(host_blocks) + block_idx
            kv_cache[block_id] = (key_cache, value_cache)

            # Optional KV compression
            if self.enable_kv_compression:
                kv_cache[block_id] = self._compress_kv_cache(
                    kv_cache[block_id], self.compression_ratio
                )

        state.kv_caches = kv_cache
        state.phase1_time = time.time() - start_time

        logger.debug(
            f"Phase 1 complete on host {self.rank}: "
            f"Processed {len(host_blocks)} blocks in {state.phase1_time:.3f}s"
        )

        return kv_cache

    def phase2_query_processing(
        self, query: torch.Tensor, state: StarAttentionState
    ) -> torch.Tensor:
        """
        Phase 2: Global attention for query processing

        Query attends to all KV caches across hosts with efficient
        aggregation using distributed softmax.
        """

        start_time = time.time()
        batch_size, query_len, num_heads, head_dim = query.shape

        # Initialize output
        output = torch.zeros_like(query)

        if self.distributed:
            # Broadcast query to all hosts
            if not self.is_query_host:
                query = torch.empty_like(query)
            torch.distributed.broadcast(query, src=self.query_host_id)

        # Compute local attention with cached KVs
        local_outputs = []
        local_stats = []

        for block_id, (key_cache, value_cache) in state.kv_caches.items():
            # Compute attention between query and this KV block
            scores = torch.matmul(
                query.transpose(1, 2), key_cache.transpose(1, 2).transpose(-2, -1)
            ) / math.sqrt(head_dim)

            # For numerical stability, compute max and exp
            max_scores = scores.max(dim=-1, keepdim=True)[0]
            exp_scores = torch.exp(scores - max_scores)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)

            # Compute local attention output
            attn_weights = exp_scores / (sum_exp + self.eps)
            local_output = torch.matmul(attn_weights, value_cache.transpose(1, 2))
            local_output = local_output.transpose(1, 2)

            local_outputs.append(local_output)
            local_stats.append(
                AttentionStats(
                    attention_weights=attn_weights,
                    max_scores=max_scores,
                    sum_exp=sum_exp,
                )
            )

        # Aggregate attention across all hosts
        if self.distributed:
            comm_start = time.time()

            # Gather statistics from all hosts to query host
            all_outputs = []
            all_max_scores = []
            all_sum_exps = []

            for local_output, stats in zip(local_outputs, local_stats):
                # Each host sends its output and statistics to query host
                if self.is_query_host:
                    gathered_outputs = [
                        torch.zeros_like(local_output) for _ in range(self.world_size)
                    ]
                    gathered_max_scores = [
                        torch.zeros_like(stats.max_scores)
                        for _ in range(self.world_size)
                    ]
                    gathered_sum_exps = [
                        torch.zeros_like(stats.sum_exp) for _ in range(self.world_size)
                    ]
                else:
                    gathered_outputs = None
                    gathered_max_scores = None
                    gathered_sum_exps = None

                # Gather to query host
                torch.distributed.gather(
                    local_output * stats.sum_exp,  # Pre-scaled output
                    gathered_outputs,
                    dst=self.query_host_id,
                )
                torch.distributed.gather(
                    stats.max_scores, gathered_max_scores, dst=self.query_host_id
                )
                torch.distributed.gather(
                    stats.sum_exp * torch.exp(stats.max_scores),
                    gathered_sum_exps,
                    dst=self.query_host_id,
                )

                if self.is_query_host:
                    all_outputs.extend(gathered_outputs)
                    all_max_scores.extend(gathered_max_scores)
                    all_sum_exps.extend(gathered_sum_exps)

            state.communication_time = time.time() - comm_start

            # Aggregate on query host
            if self.is_query_host:
                output = self._aggregate_distributed_attention(
                    all_outputs, all_max_scores, all_sum_exps
                )
        else:
            # Single host - aggregate local results
            output = self._aggregate_local_attention(local_outputs, local_stats)

        state.phase2_time = time.time() - start_time

        # Calculate speedup
        baseline_time = (
            batch_size
            * query_len
            * num_heads
            * head_dim
            * next(iter(state.kv_caches.values()))[0].shape[1]
            * len(state.kv_caches)
        ) / 1e9
        actual_time = state.phase1_time + state.phase2_time
        state.speedup_achieved = baseline_time / actual_time

        logger.debug(
            f"Phase 2 complete on host {self.rank}: "
            f"Time: {state.phase2_time:.3f}s, "
            f"Speedup: {state.speedup_achieved:.1f}x"
        )

        return output

    def _aggregate_local_attention(
        self, local_outputs: List[torch.Tensor], local_stats: List[AttentionStats]
    ) -> torch.Tensor:
        """Aggregate attention outputs using log-sum-exp trick"""

        if len(local_outputs) == 1:
            return local_outputs[0]

        # Stack for efficient computation
        outputs = torch.stack(
            local_outputs, dim=0
        )  # [n_blocks, batch, seq, heads, dim]
        max_scores = torch.stack(
            [s.max_scores for s in local_stats], dim=0
        )  # [n_blocks, batch, heads, seq, 1]
        sum_exps = torch.stack(
            [s.sum_exp for s in local_stats], dim=0
        )  # [n_blocks, batch, heads, seq, 1]

        # Global max for numerical stability
        global_max = max_scores.max(dim=0, keepdim=True)[0]  # [1, batch, heads, seq, 1]

        # Reweight each output -> weights shape to [n_blocks, batch, seq, heads, 1]
        weights = sum_exps * torch.exp(
            max_scores - global_max
        )  # [n_blocks, batch, heads, seq, 1]
        weights = weights.permute(0, 1, 3, 2, 4)  # [n_blocks, batch, seq, heads, 1]
        weights = weights / (weights.sum(dim=0, keepdim=True) + self.eps)

        # Weighted sum
        output = (outputs * weights).sum(dim=0)

        return output

    def _aggregate_distributed_attention(
        self,
        all_outputs: List[torch.Tensor],
        all_max_scores: List[torch.Tensor],
        all_sum_exps: List[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate attention from multiple hosts"""

        # Convert lists to tensors
        outputs = torch.stack(all_outputs, dim=0)  # [n_hosts, batch, seq, heads, dim]
        max_scores = torch.stack(
            all_max_scores, dim=0
        )  # [n_hosts, batch, heads, seq, 1]
        sum_exp_weighted = torch.stack(
            all_sum_exps, dim=0
        )  # [n_hosts, batch, heads, seq, 1] = s_h * exp(m_h)

        # Find global max
        global_max = max_scores.max(dim=0, keepdim=True)[0]  # [1, batch, heads, seq, 1]

        # Numerator across hosts: sum_h exp(m_h - m) * scaled_output_h
        host_weights_num = torch.exp(
            max_scores - global_max
        )  # [n_hosts, batch, heads, seq, 1]
        host_weights_num = host_weights_num.permute(
            0, 1, 3, 2, 4
        )  # [n_hosts, batch, seq, heads, 1]
        numer = (outputs * host_weights_num).sum(dim=0)  # [batch, seq, heads, dim]

        # Denominator across hosts: sum_h exp(m_h - m) * s_h,
        # where s_h * exp(m_h) was gathered as sum_exp_weighted
        denom = (sum_exp_weighted * torch.exp(-global_max)).sum(
            dim=0
        )  # [batch, heads, seq, 1]
        denom = denom.permute(0, 2, 1, 3)  # [batch, seq, heads, 1]

        # Avoid divide by zero
        denom = denom + self.eps

        # Final output
        output = numer / denom

        return output

    def _compress_kv_cache(
        self, kv_pair: Tuple[torch.Tensor, torch.Tensor], compression_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache using importance-based selection

        This is a simple implementation - can be replaced with more
        sophisticated methods like ChunkKV or KVCompress
        """
        key, value = kv_pair
        batch_size, seq_len, num_heads, head_dim = key.shape

        # Keep only top-k most important positions
        keep_len = max(1, int(seq_len * (1 / compression_ratio)))

        # Simple importance scoring based on key norm
        importance = key.norm(dim=-1).mean(dim=-1)  # [batch, seq]
        _, indices = importance.topk(keep_len, dim=-1)

        # Gather compressed KV
        indices = (
            indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_heads, head_dim)
        )
        compressed_key = key.gather(1, indices)
        compressed_value = value.gather(1, indices)

        return compressed_key, compressed_value

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_context: bool = True,
    ) -> torch.Tensor:
        """
        Star Attention forward pass

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor (same shape as query)
            value: Value tensor (same shape as query)
            attention_mask: Optional attention mask
            is_context: Whether this is context (Phase 1) or query (Phase 2)

        Returns:
            output: Attention output
        """

        # For standard transformer layer compatibility, we handle full sequences
        # In practice, users would separate context and query explicitly

        if is_context:
            # Phase 1: Context encoding
            state = StarAttentionState(current_phase=1)
            self.phase1_context_encoding(query, state)
            # Return identity for context phase (KV cache is stored)
            return query
        else:
            # Phase 2: Query processing
            # Assume state is maintained (in practice, would be passed or stored)
            if not hasattr(self, "_state"):
                # Fallback to regular attention if no context was encoded
                return self.local_attention(query, key, value, attention_mask)

            state = self._state
            state.current_phase = 2
            return self.phase2_query_processing(query, state)

    def encode_context(self, context: torch.Tensor) -> StarAttentionState:
        """
        Explicitly encode context (Phase 1)

        This is the recommended API for using Star Attention
        """
        state = StarAttentionState(current_phase=1)
        self.phase1_context_encoding(context, state)
        self._state = state  # Store for later query processing
        return state

    def process_query(
        self, query: torch.Tensor, state: Optional[StarAttentionState] = None
    ) -> torch.Tensor:
        """
        Process query with encoded context (Phase 2)

        This is the recommended API for using Star Attention
        """
        if state is None:
            state = getattr(self, "_state", None)
        if state is None:
            raise ValueError("No context has been encoded. Call encode_context first.")

        state.current_phase = 2
        return self.phase2_query_processing(query, state)


def create_star_attention(config: StreamAttentionConfig) -> StarAttention:
    """Factory function to create Star Attention instance"""
    return StarAttention(config)

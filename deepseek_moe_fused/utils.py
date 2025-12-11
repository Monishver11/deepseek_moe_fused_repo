"""
Utility functions for the Fused DeepSeek-MoE implementation.

This module implements:
1. compute_routing_metadata: Generates the "Metadata Map" for virtual gathering
2. get_grid_config: Calculates the MegaBlocks-style dynamic grid topology

The key insight is that we avoid physically sorting/copying tokens. Instead,
we create index arrays that allow the kernel to read tokens from their
original positions via indirect addressing.
"""

import torch
from typing import Tuple, NamedTuple


class RoutingMetadata(NamedTuple):
    """
    Container for routing metadata passed to the Triton kernel.
    
    Attributes:
        sorted_token_indices: [total_assignments] - Row indices into X for each assignment
            This is the CRITICAL array for "Virtual Gathering". 
            Example: If token 5 goes to expert 0 and token 12 goes to expert 0,
            the first segment might be [5, 12, ...] telling the kernel to read
            rows 5 and 12 from X when processing expert 0.
        
        sorted_expert_indices: [total_assignments] - Expert ID for each assignment
            Sorted so all tokens for expert 0 come first, then expert 1, etc.
            Example: [0, 0, 0, ..., 1, 1, 1, ..., 7, 7, 7, ...]
        
        sorted_weights: [total_assignments] - Router probability scores
            The gating weights for each token-expert pair (used externally for weighting)
        
        tokens_per_expert: [num_experts] - Count of tokens assigned to each expert
            Example: [120, 95, 110, ...] means expert 0 has 120 tokens, expert 1 has 95, etc.
        
        num_tokens: int - Original batch size (N)
        
        total_assignments: int - N * top_k (total token-expert pairs)
    """
    sorted_token_indices: torch.Tensor
    sorted_expert_indices: torch.Tensor  
    sorted_weights: torch.Tensor
    tokens_per_expert: torch.Tensor
    num_tokens: int
    total_assignments: int


def compute_routing_metadata(
    x: torch.Tensor,
    gate: torch.nn.Linear,
    top_k: int,
    num_experts: int,
) -> RoutingMetadata:
    """
    Compute routing metadata from input tokens and gate layer.
    
    This implements the "MegaBlocks topology calculation":
    1. Apply router to get expert logits
    2. Select top-k experts per token
    3. Sort by expert ID to create contiguous workloads
    4. Generate the index map for virtual gathering
    
    Args:
        x: Input tensor of shape [N, H] where N is batch*seq_len, H is hidden_dim
        gate: Linear layer mapping H -> num_experts (the router)
        top_k: Number of experts selected per token
        num_experts: Total number of routed experts
    
    Returns:
        RoutingMetadata containing all arrays needed for kernel dispatch
    
    Memory Layout After Sorting:
    ---------------------------
    Original X:     [token_0, token_1, token_2, ..., token_N-1]
    
    After routing, suppose:
    - token_5 -> experts [0, 3] with weights [0.6, 0.4]
    - token_12 -> experts [0, 7] with weights [0.5, 0.5]
    - token_99 -> experts [3, 0] with weights [0.7, 0.3]
    
    sorted_token_indices might look like:
    [5, 12, 99, ...(more expert 0), 5, 99, ...(expert 3), 12, ...(expert 7)]
    
    The kernel uses these indices to READ from X without copying.
    """
    N, H = x.shape
    device = x.device
    
    # =========================================================================
    # Step 1: Router Forward Pass
    # =========================================================================
    # Apply the gating layer to get logits for each expert
    # gate_logits: [N, num_experts]
    gate_logits = gate(x)
    
    # =========================================================================
    # Step 2: Top-K Selection
    # =========================================================================
    # Select the top-k experts for each token
    # topk_weights: [N, top_k] - The softmax probabilities for selected experts
    # topk_indices: [N, top_k] - Which experts were selected (0 to num_experts-1)
    topk_weights, topk_indices = torch.topk(
        torch.softmax(gate_logits, dim=-1),
        k=top_k,
        dim=-1,
    )
    
    # Normalize weights so they sum to 1 for each token
    # This ensures proper weighted combination of expert outputs
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    # =========================================================================
    # Step 3: Flatten to 1D Assignment Lists
    # =========================================================================
    # Convert from [N, top_k] to flat [N * top_k] arrays
    # This creates one entry per token-expert assignment
    
    # flat_expert_indices: [N * top_k] - Which expert for each assignment
    flat_expert_indices = topk_indices.view(-1)  # [N * top_k]
    
    # flat_token_indices: [N * top_k] - Which token for each assignment
    # For token i with top_k=2, we get entries [i, i] (token i appears twice)
    flat_token_indices = torch.arange(N, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    
    # flat_weights: [N * top_k] - Router probability for each assignment
    flat_weights = topk_weights.view(-1)
    
    total_assignments = N * top_k
    
    # =========================================================================
    # Step 4: Sort by Expert ID (The Metadata Map Generation)
    # =========================================================================
    # This is the CRITICAL step for the kernel's work scheduling.
    # By sorting, we ensure all tokens for expert 0 are contiguous,
    # then all tokens for expert 1, etc.
    #
    # Why sort instead of just using masks?
    # - Contiguous workloads enable coalesced memory access
    # - Each thread block knows exactly which slice of the sorted array to process
    # - No warp divergence from sparse conditional checks
    
    # Get the permutation that sorts by expert ID
    sort_indices = torch.argsort(flat_expert_indices, stable=True)
    
    # Apply the permutation to all arrays
    sorted_expert_indices = flat_expert_indices[sort_indices]
    sorted_token_indices = flat_token_indices[sort_indices]
    sorted_weights = flat_weights[sort_indices]
    
    # =========================================================================
    # Step 5: Compute Histogram (Tokens per Expert)
    # =========================================================================
    # Count how many assignments each expert received
    # This is used for grid topology calculation
    tokens_per_expert = torch.bincount(
        sorted_expert_indices,
        minlength=num_experts,
    ).to(torch.int32)
    
    return RoutingMetadata(
        sorted_token_indices=sorted_token_indices.to(torch.int32),
        sorted_expert_indices=sorted_expert_indices.to(torch.int32),
        sorted_weights=sorted_weights,
        tokens_per_expert=tokens_per_expert,
        num_tokens=N,
        total_assignments=total_assignments,
    )


class GridConfig(NamedTuple):
    """
    Configuration for the Triton kernel launch grid.
    
    Attributes:
        total_blocks: Total number of thread blocks to launch (grid size)
        expert_block_offsets: [num_experts + 1] - Cumulative sum of blocks per expert
            expert_block_offsets[i] gives the starting block index for expert i
            Example: [0, 4, 7, 15, ...] means expert 0 gets blocks 0-3,
            expert 1 gets blocks 4-6, expert 2 gets blocks 7-14, etc.
        expert_token_offsets: [num_experts + 1] - Cumulative sum of tokens per expert
            expert_token_offsets[i] gives the starting index in sorted_token_indices
            for expert i's tokens
    """
    total_blocks: int
    expert_block_offsets: torch.Tensor
    expert_token_offsets: torch.Tensor


def get_grid_config(
    tokens_per_expert: torch.Tensor,
    block_m: int,
    num_experts: int,
) -> GridConfig:
    """
    Calculate the MegaBlocks-style dynamic grid topology.
    
    This maps the "Logical Work Queue" to a "Physical Grid" that Triton executes.
    The key insight is that different experts may have vastly different numbers
    of tokens (load imbalance). Instead of padding to the maximum, we allocate
    exactly the blocks needed for each expert.
    
    Args:
        tokens_per_expert: [num_experts] - Count of tokens assigned to each expert
        block_m: Number of tokens processed per thread block
        num_experts: Total number of experts
    
    Returns:
        GridConfig containing total blocks and offset lookup tensors
    
    Example:
    --------
    Suppose tokens_per_expert = [120, 95, 200] with block_m = 32:
    
    Expert 0: ceil(120/32) = 4 blocks
    Expert 1: ceil(95/32) = 3 blocks  
    Expert 2: ceil(200/32) = 7 blocks
    
    total_blocks = 4 + 3 + 7 = 14
    
    expert_block_offsets = [0, 4, 7, 14]
    
    When kernel block #5 launches, it can determine:
    - Block 5 >= expert_block_offsets[1]=4, so expert >= 1
    - Block 5 < expert_block_offsets[2]=7, so expert = 1
    - Local block index within expert 1 = 5 - 4 = 1
    - This block processes tokens 32-63 of expert 1's token list
    """
    device = tokens_per_expert.device
    
    # =========================================================================
    # Step 1: Calculate Blocks per Expert
    # =========================================================================
    # Each expert needs ceil(num_tokens / block_m) blocks
    # Using integer math: (n + block_m - 1) // block_m
    blocks_per_expert = (tokens_per_expert + block_m - 1) // block_m
    
    # =========================================================================
    # Step 2: Compute Cumulative Sum (Prefix Sum) for Block Offsets
    # =========================================================================
    # This creates the lookup table for the kernel
    # expert_block_offsets[i] = sum of blocks for experts 0, 1, ..., i-1
    
    # Prepend 0 for the cumsum (so expert 0 starts at block 0)
    expert_block_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    expert_block_offsets[1:] = torch.cumsum(blocks_per_expert, dim=0)
    
    # Total blocks is the last element of cumsum
    total_blocks = int(expert_block_offsets[-1].item())
    
    # =========================================================================
    # Step 3: Compute Cumulative Sum for Token Offsets
    # =========================================================================
    # This tells the kernel where each expert's tokens start in sorted_token_indices
    expert_token_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    expert_token_offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)
    
    return GridConfig(
        total_blocks=total_blocks,
        expert_block_offsets=expert_block_offsets,
        expert_token_offsets=expert_token_offsets,
    )


def find_expert_for_block(
    block_id: int,
    expert_block_offsets: torch.Tensor,
) -> Tuple[int, int]:
    """
    Given a block ID, find which expert it belongs to and its local index.
    
    This is the host-side equivalent of the binary search done in the kernel.
    Useful for debugging and verification.
    
    Args:
        block_id: The global block index (0 to total_blocks-1)
        expert_block_offsets: [num_experts + 1] cumulative sum array
    
    Returns:
        (expert_id, local_block_index) tuple
    """
    # Binary search: find largest i such that expert_block_offsets[i] <= block_id
    expert_id = torch.searchsorted(expert_block_offsets, block_id, right=True) - 1
    expert_id = int(expert_id.item())
    local_block_index = block_id - int(expert_block_offsets[expert_id].item())
    return expert_id, local_block_index

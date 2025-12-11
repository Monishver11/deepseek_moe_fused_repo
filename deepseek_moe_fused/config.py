"""
Configuration dataclasses for the Fused DeepSeek-MoE implementation.

These define the architectural hyperparameters for the MoE layer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MoEConfig:
    """
    Configuration for the DeepSeek-style MoE layer.
    
    Architecture Notes:
    - This implements the "up-projection" GEMM only (X @ W_gate_up)
    - Activation and down-projection are handled externally in PyTorch
    - Uses Top-K routing with K=2 by default (standard for DeepSeek/Mixtral)
    
    Memory Considerations (RTX 4070 - 12GB VRAM):
    - With E=8, H=2048, D=2048: ~67MB for routed weights
    - With E=64, H=2048, D=2048: ~536MB for routed weights
    - Use E=8 for development, scale to 64 for production
    """
    
    # Hidden dimension (embedding size)
    hidden_dim: int = 2048
    
    # Expert intermediate dimension
    # Standard dense FFN uses 4*H, but MoE experts are "thinner"
    # D = H keeps per-expert params manageable
    intermediate_dim: int = 2048
    
    # Number of routed experts
    num_experts: int = 8
    
    # Top-K experts selected per token
    top_k: int = 2
    
    # Number of shared experts (treated as single dense matrix)
    num_shared_experts: int = 1
    
    # Triton kernel block sizes
    # BLOCK_M: tokens processed per thread block
    # BLOCK_N: output dimension chunk size  
    # BLOCK_K: hidden dimension chunk size (for K-loop tiling)
    block_m: int = 32
    block_n: int = 64
    block_k: int = 32
    
    # Precision settings
    # Storage in BF16, compute accumulation in FP32
    use_bfloat16: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.intermediate_dim > 0, "intermediate_dim must be positive"
        assert self.num_experts > 0, "num_experts must be positive"
        assert 1 <= self.top_k <= self.num_experts, "top_k must be in [1, num_experts]"
        assert self.block_m > 0 and (self.block_m & (self.block_m - 1)) == 0, \
            "block_m must be a power of 2"
        assert self.block_n > 0 and (self.block_n & (self.block_n - 1)) == 0, \
            "block_n must be a power of 2"
        assert self.block_k > 0 and (self.block_k & (self.block_k - 1)) == 0, \
            "block_k must be a power of 2"
    
    @property
    def routed_weight_size_mb(self) -> float:
        """Estimate routed weights memory in MB (BF16)."""
        # Shape: [num_experts, hidden_dim, intermediate_dim]
        elements = self.num_experts * self.hidden_dim * self.intermediate_dim
        bytes_per_element = 2  # BF16
        return (elements * bytes_per_element) / (1024 * 1024)
    
    @property
    def shared_weight_size_mb(self) -> float:
        """Estimate shared weights memory in MB (BF16)."""
        # Shape: [hidden_dim, intermediate_dim]
        elements = self.hidden_dim * self.intermediate_dim
        bytes_per_element = 2  # BF16
        return (elements * bytes_per_element) / (1024 * 1024)


# Default configurations for different use cases
DEV_CONFIG = MoEConfig(
    hidden_dim=1024,
    intermediate_dim=1024,
    num_experts=8,
    top_k=2,
)

SMALL_CONFIG = MoEConfig(
    hidden_dim=2048,
    intermediate_dim=2048,
    num_experts=8,
    top_k=2,
)

LARGE_CONFIG = MoEConfig(
    hidden_dim=2048,
    intermediate_dim=2048,
    num_experts=64,
    top_k=2,
)

"""
DeepSeek-Style Fused MoE Kernel Implementation

This package implements a fused Grouped GEMM kernel that combines
Routed Expert and Shared Expert computations into a single kernel launch,
reducing memory bandwidth by loading input activations only once.

Key Components:
- FusedDeepSeekMoEMLP: Drop-in replacement for DeepSeekMoEMLP
- fused_moe_forward: Low-level Triton kernel wrapper
- compute_routing_metadata, get_grid_config: Routing utilities

Quick Start:
    from deepseek_moe_fused import FusedDeepSeekMoEMLP
    
    # Drop-in replacement for DeepSeekMoEMLP
    moe_layer = FusedDeepSeekMoEMLP(
        dim=768,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2,
    )
    output, aux_loss = moe_layer(x)
"""

from .adapter import FusedDeepSeekMoEMLP, FusedMoEMLP
from .moe_layer import DeepSeekMoELayer
from .kernels import fused_moe_forward
from .utils import compute_routing_metadata, get_grid_config
from .config import MoEConfig

__all__ = [
    # Main adapter (drop-in replacement)
    'FusedDeepSeekMoEMLP',
    'FusedMoEMLP',
    # Lower-level components
    'DeepSeekMoELayer',
    'fused_moe_forward',
    'compute_routing_metadata',
    'get_grid_config',
    'MoEConfig',
]

__version__ = '0.1.0'
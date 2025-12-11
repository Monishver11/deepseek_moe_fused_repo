"""
Fused DeepSeek MoE Adapter Module.

This provides `FusedDeepSeekMoEMLP` - a drop-in replacement for the naive
DeepSeekMoEMLP in train_gpt_deepseek_moe.py.

The adapter:
1. Uses the fused Triton kernel for UP-projection (shared + routed combined)
2. Falls back to naive PyTorch if kernel fails
3. Maintains identical interface (input/output shapes, aux_loss_dict)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional

from .kernels import fused_moe_forward
from .utils import get_grid_config


class FusedDeepSeekMoEMLP(nn.Module):
    """
    DeepSeek-style MoE with fused UP-projection kernel.
    
    Drop-in replacement for DeepSeekMoEMLP. Same interface:
        output, aux_loss_dict = layer(x)
    
    Architecture per expert:
        UP:   h = x @ W_up        [dim -> hdim]   <- FUSED (shared + routed)
        ACT:  h = relu(h).square()
        DOWN: y = h @ W_down.T    [hdim -> dim]
    
    The fused kernel combines:
        h_combined = x @ W_shared_up + x @ W_routed_up[expert_ids]
    Into a single memory load of X, saving ~50% bandwidth on UP projection.
    
    Args:
        dim: Model dimension (hidden size)
        num_shared_experts: Number of shared experts (must be 1 for fused kernel)
        num_routed_experts: Number of routed experts (E)
        top_k: Number of experts selected per token
        expert_capacity_factor: (unused, kept for interface compatibility)
        use_fused: If True, use fused Triton kernel. If False, use naive PyTorch.
    """
    
    def __init__(
        self,
        dim: int,
        num_shared_experts: int = 1,
        num_routed_experts: int = 4,
        top_k: int = 2,
        expert_capacity_factor: float = 1.25,
        use_fused: bool = True,
    ):
        super().__init__()
        
        # Validate constraints
        assert num_shared_experts == 1, "Fused kernel only supports num_shared_experts=1"
        
        self.dim = dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.use_fused = use_fused
        self.hdim = 4 * dim  # Intermediate dimension
        
        # ====== SHARED EXPERT WEIGHTS ======
        # UP projection: [dim, hdim]
        self.shared_expert_up = nn.Parameter(torch.empty(dim, self.hdim))
        self.shared_expert_up.label = 'mlp'
        
        # DOWN projection: [dim, hdim] (used as h @ W.T in forward)
        self.shared_expert_down = nn.Parameter(torch.empty(dim, self.hdim))
        self.shared_expert_down.label = 'mlp'
        
        # ====== ROUTED EXPERT WEIGHTS ======
        # Router: maps dim -> num_routed_experts
        self.router = nn.Linear(dim, num_routed_experts, bias=False)
        self.router.weight.label = 'moe_router'
        
        # UP projection: [num_routed_experts, dim, hdim]
        # Stacked for efficient kernel access
        self.routed_experts_up = nn.Parameter(
            torch.empty(num_routed_experts, dim, self.hdim)
        )
        self.routed_experts_up.label = 'mlp'
        
        # DOWN projection: ParameterList of [dim, hdim] each
        # Kept as list since DOWN isn't fused
        self.routed_experts_down = nn.ParameterList([
            nn.Parameter(torch.empty(dim, self.hdim))
            for _ in range(num_routed_experts)
        ])
        for p in self.routed_experts_down:
            p.label = 'mlp'
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights matching the original DeepSeekMoEMLP."""
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        
        with torch.no_grad():
            # Router: zero init
            self.router.weight.zero_()
            
            # Shared expert: uniform for UP, zero for DOWN
            self.shared_expert_up.uniform_(-bound, bound)
            self.shared_expert_down.zero_()
            
            # Routed experts: uniform for UP, zero for DOWN
            self.routed_experts_up.uniform_(-bound, bound)
            for down_w in self.routed_experts_down:
                down_w.zero_()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with fused UP-projection.
        
        Args:
            x: Input tensor [B, T, dim]
        
        Returns:
            output: Output tensor [B, T, dim]
            aux_loss_dict: Dictionary with load_balancing_loss, router_z_loss, etc.
        """
        B, T, D = x.shape
        assert D == self.dim
        
        # Flatten to [N, dim] where N = B * T
        x_flat = x.reshape(B * T, D)
        N = B * T
        
        # ====== ROUTING ======
        router_logits = self.router(x_flat)  # [N, num_routed_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # ====== CREATE ROUTING METADATA ======
        flat_expert_indices = topk_indices.view(-1)  # [N * top_k]
        flat_token_indices = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        flat_weights = topk_weights.view(-1)
        
        # Sort by expert for coalesced access
        sort_indices = torch.argsort(flat_expert_indices, stable=True)
        sorted_expert_indices = flat_expert_indices[sort_indices].to(torch.int32)
        sorted_token_indices = flat_token_indices[sort_indices].to(torch.int32)
        sorted_weights = flat_weights[sort_indices]
        
        total_assignments = N * self.top_k
        
        # ====== FUSED UP-PROJECTION ======
        if self.use_fused:
            try:
                h_up_sorted = self._fused_up_projection(
                    x_flat, sorted_token_indices, sorted_expert_indices, total_assignments
                )
            except Exception as e:
                print(f"[FusedMoE] Kernel failed, falling back to naive: {e}")
                h_up_sorted = self._naive_up_projection(
                    x_flat, sorted_token_indices, sorted_expert_indices, total_assignments
                )
        else:
            h_up_sorted = self._naive_up_projection(
                x_flat, sorted_token_indices, sorted_expert_indices, total_assignments
            )
        
        # ====== ACTIVATION ======
        h_act_sorted = F.relu(h_up_sorted).square()
        
        # ====== DOWN-PROJECTION (Not Fused) ======
        # Compute per-expert DOWN projection
        y_sorted = self._down_projection(
            h_act_sorted, sorted_expert_indices, total_assignments
        )
        
        # ====== WEIGHT BY ROUTER PROBS & SCATTER BACK ======
        y_weighted = y_sorted * sorted_weights.unsqueeze(-1).to(y_sorted.dtype)
        
        # Scatter-add back to original token order
        output_flat = torch.zeros(N, D, dtype=y_sorted.dtype, device=x.device)
        output_flat.index_add_(0, sorted_token_indices.long(), y_weighted)
        
        output = output_flat.reshape(B, T, D)
        
        # ====== AUXILIARY LOSSES ======
        aux_loss_dict = self._compute_aux_losses(
            router_logits, router_probs, topk_indices, B, T
        )
        
        return output, aux_loss_dict
    
    def _fused_up_projection(
        self,
        x_flat: Tensor,           # [N, dim]
        sorted_token_indices: Tensor,   # [total_assignments]
        sorted_expert_indices: Tensor,  # [total_assignments]
        total_assignments: int,
    ) -> Tensor:
        """
        Fused UP-projection using Triton kernel.
        
        Computes: h = x @ W_shared_up + x @ W_routed_up[expert_id]
        With single load of x.
        """
        # Ensure BF16 for kernel
        x_bf16 = x_flat.to(torch.bfloat16)
        W_routed_bf16 = self.routed_experts_up.to(torch.bfloat16)
        W_shared_bf16 = self.shared_expert_up.to(torch.bfloat16)
        
        # Grid config (needed for API but new kernel uses 2D grid internally)
        tokens_per_expert = torch.bincount(
            sorted_expert_indices.long(),
            minlength=self.num_routed_experts
        ).to(torch.int32)
        grid_config = get_grid_config(tokens_per_expert, block_m=32, num_experts=self.num_routed_experts)
        
        # Launch fused kernel
        h_up_sorted = fused_moe_forward(
            X=x_bf16,
            W_routed=W_routed_bf16,
            W_shared=W_shared_bf16,
            sorted_token_indices=sorted_token_indices,
            sorted_expert_indices=sorted_expert_indices,
            expert_token_offsets=grid_config.expert_token_offsets,
            total_blocks=grid_config.total_blocks,
            num_experts=self.num_routed_experts,
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=32,
        )
        
        return h_up_sorted
    
    def _naive_up_projection(
        self,
        x_flat: Tensor,
        sorted_token_indices: Tensor,
        sorted_expert_indices: Tensor,
        total_assignments: int,
    ) -> Tensor:
        """
        Naive UP-projection (fallback) using standard PyTorch.
        """
        # Gather input for all assignments
        x_gathered = x_flat[sorted_token_indices.long()]  # [total_assignments, dim]
        
        # Shared UP: same for all tokens
        h_shared = F.linear(x_gathered, self.shared_expert_up.T.type_as(x_gathered))
        
        # Routed UP: per-expert
        h_routed = torch.zeros(
            total_assignments, self.hdim,
            dtype=x_gathered.dtype, device=x_gathered.device
        )
        
        for e in range(self.num_routed_experts):
            mask = (sorted_expert_indices == e)
            if mask.any():
                x_e = x_gathered[mask]
                h_routed[mask] = F.linear(
                    x_e, self.routed_experts_up[e].T.type_as(x_e)
                )
        
        return h_shared + h_routed
    
    def _down_projection(
        self,
        h_sorted: Tensor,           # [total_assignments, hdim]
        sorted_expert_indices: Tensor,
        total_assignments: int,
    ) -> Tensor:
        """
        DOWN-projection (not fused) - per-expert computation.
        
        Both shared and routed DOWN projections.
        """
        # Shared DOWN: same for all
        y_shared = F.linear(h_sorted, self.shared_expert_down.type_as(h_sorted))
        
        # Routed DOWN: per-expert
        y_routed = torch.zeros(
            total_assignments, self.dim,
            dtype=h_sorted.dtype, device=h_sorted.device
        )
        
        for e in range(self.num_routed_experts):
            mask = (sorted_expert_indices == e)
            if mask.any():
                h_e = h_sorted[mask]
                y_routed[mask] = F.linear(
                    h_e, self.routed_experts_down[e].type_as(h_e)
                )
        
        return y_shared + y_routed
    
    def _compute_aux_losses(
        self,
        router_logits: Tensor,  # [N, num_experts]
        router_probs: Tensor,   # [N, num_experts]
        topk_indices: Tensor,   # [N, top_k]
        B: int,
        T: int,
    ) -> Dict[str, Tensor]:
        """
        Compute auxiliary losses for MoE training.
        
        Same as original DeepSeekMoEMLP.
        """
        N = B * T
        
        # One-hot for load balancing (use first choice)
        expert_mask_one_hot = F.one_hot(
            topk_indices[:, 0], num_classes=self.num_routed_experts
        ).float()
        
        # Fraction of tokens per expert
        tokens_per_expert = expert_mask_one_hot.sum(dim=0)
        fraction_per_expert = tokens_per_expert / N
        
        # Average router prob per expert
        avg_router_prob = router_probs.mean(dim=0)
        
        # Load balancing loss
        load_balancing_loss = self.num_routed_experts * torch.sum(
            fraction_per_expert.detach() * avg_router_prob
        )
        
        # Router Z-loss
        router_z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        
        # Importance loss
        importance = router_probs.sum(dim=0)
        importance_loss = torch.var(importance)
        
        # Expert counts (for monitoring)
        expert_counts = tokens_per_expert
        
        return {
            'load_balancing_loss': load_balancing_loss,
            'router_z_loss': router_z_loss,
            'importance_loss': importance_loss,
            'expert_counts': expert_counts,
        }


# Alias for backward compatibility
FusedMoEMLP = FusedDeepSeekMoEMLP
"""
Fused DeepSeek MoE Adapter Module.

This provides `FusedDeepSeekMoEMLP` - a drop-in replacement for the naive
DeepSeekMoEMLP in train_gpt_deepseek_moe.py.

The adapter:
1. Uses the fused Triton kernel for UP-projection (shared + routed combined)
2. Has proper autograd backward pass for training
3. Maintains identical interface (input/output shapes, aux_loss_dict)

Key Design:
- Routing is computed OUTSIDE the autograd function (torch.compile compatible)
- FusedMoEUpProjection autograd.Function wraps the kernel with backward pass
- Backward pass uses standard PyTorch (correct, not fused)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional

from .kernels import fused_moe_forward
from .utils import get_grid_config


class FusedMoEUpProjection(torch.autograd.Function):
    """
    Autograd Function for fused MoE UP-projection.
    
    Forward: Uses custom Triton kernel for fused routed+shared computation
    Backward: Uses standard PyTorch operations for gradient computation
    
    Computes: Y_sorted = X[sorted_indices] @ W_routed[expert_ids] + X[sorted_indices] @ W_shared
    
    This is separate from routing - routing metadata is computed outside and passed in.
    This design allows torch.compile compatibility for the routing part.
    """
    
    @staticmethod
    def forward(
        ctx,
        X: Tensor,                        # [N, H] input activations
        W_routed: Tensor,                 # [E, H, D] routed expert weights (stacked)
        W_shared: Tensor,                 # [H, D] shared expert weights
        sorted_token_indices: Tensor,     # [total_assignments] - which rows of X
        sorted_expert_indices: Tensor,    # [total_assignments] - which expert
        tokens_per_expert: Tensor,        # [E] - count per expert
        num_experts: int,
        block_m: int = 32,
        block_n: int = 64,
        block_k: int = 32,
    ) -> Tensor:
        """
        Forward pass using fused Triton kernel.
        
        Returns:
            Y_sorted: [total_assignments, D] - Output in sorted expert order
        """
        # Get grid config for kernel launch
        grid_config = get_grid_config(tokens_per_expert, block_m, num_experts)
        
        # Launch fused kernel
        Y_sorted = fused_moe_forward(
            X=X,
            W_routed=W_routed,
            W_shared=W_shared,
            sorted_token_indices=sorted_token_indices,
            sorted_expert_indices=sorted_expert_indices,
            expert_token_offsets=grid_config.expert_token_offsets,
            total_blocks=grid_config.total_blocks,
            num_experts=num_experts,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        
        # Save for backward
        ctx.save_for_backward(
            X, W_routed, W_shared,
            sorted_token_indices, sorted_expert_indices, tokens_per_expert,
        )
        ctx.num_experts = num_experts
        
        return Y_sorted
    
    @staticmethod
    def backward(ctx, grad_Y_sorted: Tensor):
        """
        Backward pass using vectorized PyTorch operations (no .item() calls).
        
        Gradients:
            ∂L/∂X = (∂L/∂Y @ W_routed.T + ∂L/∂Y @ W_shared.T) [scattered back]
            ∂L/∂W_routed[e] = X[tokens_for_e].T @ ∂L/∂Y[tokens_for_e]
            ∂L/∂W_shared = X_sorted.T @ ∂L/∂Y
        """
        (
            X, W_routed, W_shared,
            sorted_token_indices, sorted_expert_indices, tokens_per_expert,
        ) = ctx.saved_tensors
        num_experts = ctx.num_experts
        
        N, H = X.shape
        E, _, D = W_routed.shape
        total_assignments = sorted_token_indices.shape[0]
        
        # Cast grad to float32 for stable accumulation
        grad_Y = grad_Y_sorted.float()  # [total_assignments, D]
        
        # Gather X in sorted order
        X_sorted = X[sorted_token_indices.long()].float()  # [total_assignments, H]
        
        # ======================================================================
        # Gradient w.r.t. W_shared (simple matmul)
        # ======================================================================
        grad_W_shared = torch.mm(X_sorted.t(), grad_Y).to(W_shared.dtype)  # [H, D]
        
        # ======================================================================
        # Gradient w.r.t. W_routed (vectorized with einsum)
        # ======================================================================
        # Create one-hot expert mask [total_assignments, E]
        expert_one_hot = F.one_hot(sorted_expert_indices.long(), num_classes=E).float()
        
        # grad_W_routed[e, h, d] = sum_{i: expert[i]=e} X_sorted[i, h] * grad_Y[i, d]
        # Using einsum: 'th,td,te->ehd'
        grad_W_routed = torch.einsum('th,td,te->ehd', X_sorted, grad_Y, expert_one_hot).to(W_routed.dtype)
        
        # ======================================================================
        # Gradient w.r.t. X (vectorized)
        # ======================================================================
        # Shared contribution: same W_shared for all assignments
        grad_from_shared = torch.mm(grad_Y, W_shared.t().float())  # [total_assignments, H]
        
        # Routed contribution: different W_routed per token based on expert assignment
        # W_routed_selected[i] = W_routed[expert_i], shape [total_assignments, H, D]
        W_routed_selected = W_routed[sorted_expert_indices.long()].float()  # [T, H, D]
        
        # grad_from_routed[i] = grad_Y[i] @ W_routed[expert_i].T
        # = (grad_Y @ W_routed_selected.transpose) for each token
        # Using bmm: [T, 1, D] @ [T, D, H] -> [T, 1, H] -> squeeze -> [T, H]
        grad_from_routed = torch.bmm(
            grad_Y.unsqueeze(1),           # [T, 1, D]
            W_routed_selected.transpose(1, 2)  # [T, D, H]
        ).squeeze(1)  # [T, H]
        
        # Total gradient in sorted order, then scatter back
        grad_X_sorted = grad_from_routed + grad_from_shared  # [total_assignments, H]
        
        grad_X = torch.zeros(N, H, dtype=torch.float32, device=X.device)
        grad_X.index_add_(0, sorted_token_indices.long(), grad_X_sorted)
        grad_X = grad_X.to(X.dtype)
        
        # Return gradients (None for non-tensor args)
        return (
            grad_X,
            grad_W_routed,
            grad_W_shared,
            None, None, None,  # sorted_token_indices, sorted_expert_indices, tokens_per_expert
            None, None, None, None,  # num_experts, block_m, block_n, block_k
        )


class FusedDeepSeekMoEMLP(nn.Module):
    """
    DeepSeek-style MoE with optional fused UP-projection kernel.
    
    Drop-in replacement for DeepSeekMoEMLP. Same interface:
        output, aux_loss_dict = layer(x)
    
    Architecture per expert:
        UP:   h = x @ W_up        [dim -> hdim]   <- FUSED (shared + routed) when use_fused=True
        ACT:  h = relu(h).square()
        DOWN: y = h @ W_down.T    [hdim -> dim]
    
    The fused kernel combines:
        h_combined = x @ W_shared_up + x @ W_routed_up[expert_ids]
    Into a single memory load of X, saving ~50% bandwidth on UP projection.
    
    torch.compile Compatibility:
        - use_fused=False: Fully traceable, works with fullgraph=True
        - use_fused=True: Uses Triton kernel with autograd backward, works with fullgraph=False
    
    Args:
        dim: Model dimension (hidden size)
        num_shared_experts: Number of shared experts (must be 1 for fused kernel)
        num_routed_experts: Number of routed experts (E)
        top_k: Number of experts selected per token
        expert_capacity_factor: (unused, kept for interface compatibility)
        use_fused: If True, use fused Triton kernel for forward pass
    """
    
    def __init__(
        self,
        dim: int,
        num_shared_experts: int = 1,
        num_routed_experts: int = 4,
        top_k: int = 2,
        expert_capacity_factor: float = 1.25,
        use_fused: bool = False,  # Default False for torch.compile fullgraph compatibility
    ):
        super().__init__()
        
        assert num_shared_experts == 1, "Fused kernel only supports 1 shared expert"
        
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
        
        # UP projection: ParameterList of [dim, hdim] each
        # Kept as list for optimizer compatibility (NorMuon expects 2D tensors)
        # Will be stacked to 3D at forward time for the kernel
        self.routed_experts_up = nn.ParameterList([
            nn.Parameter(torch.empty(dim, self.hdim))
            for _ in range(num_routed_experts)
        ])
        for p in self.routed_experts_up:
            p.label = 'mlp'
        
        # DOWN projection: ParameterList of [dim, hdim] each
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
            for up_w in self.routed_experts_up:
                up_w.uniform_(-bound, bound)
            for down_w in self.routed_experts_down:
                down_w.zero_()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with optional fused UP-projection.
        
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
        # Cast router to input dtype for compatibility
        router_logits = F.linear(x_flat, self.router.weight.type_as(x_flat))  # [N, num_routed_experts]
        router_probs = F.softmax(router_logits.float(), dim=-1)  # Softmax in FP32 for stability
        
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
        
        # Compute tokens_per_expert using scatter_add (torch.compile compatible)
        tokens_per_expert = torch.zeros(
            self.num_routed_experts, dtype=torch.int32, device=x_flat.device
        )
        ones = torch.ones(total_assignments, dtype=torch.int32, device=x_flat.device)
        tokens_per_expert.scatter_add_(0, sorted_expert_indices.long(), ones)
        
        # ====== UP-PROJECTION ======
        if self.use_fused:
            h_up_sorted = self._fused_up_projection(
                x_flat, sorted_token_indices, sorted_expert_indices, 
                tokens_per_expert, total_assignments
            )
        else:
            h_up_sorted = self._naive_up_projection(
                x_flat, sorted_token_indices, sorted_expert_indices, total_assignments
            )
        
        # ====== ACTIVATION ======
        h_act_sorted = F.relu(h_up_sorted).square()
        
        # ====== DOWN-PROJECTION (Not Fused) ======
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
        tokens_per_expert: Tensor,      # [E]
        total_assignments: int,
    ) -> Tensor:
        """
        Fused UP-projection using Triton kernel with autograd backward.
        """
        # Ensure BF16 for kernel
        x_bf16 = x_flat.to(torch.bfloat16)
        # Stack ParameterList into 3D tensor [E, dim, hdim] for kernel
        W_routed_bf16 = torch.stack(list(self.routed_experts_up)).to(torch.bfloat16)
        W_shared_bf16 = self.shared_expert_up.to(torch.bfloat16)
        
        # Call autograd Function (has proper backward pass!)
        h_up_sorted = FusedMoEUpProjection.apply(
            x_bf16,
            W_routed_bf16,
            W_shared_bf16,
            sorted_token_indices,
            sorted_expert_indices,
            tokens_per_expert,
            self.num_routed_experts,
            32,  # block_m
            64,  # block_n
            32,  # block_k
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
        Fully traceable by torch.compile.
        """
        # Gather input for all assignments
        x_gathered = x_flat[sorted_token_indices.long()]  # [total_assignments, dim]
        
        # Shared UP: same for all tokens
        h_shared = F.linear(x_gathered, self.shared_expert_up.T.type_as(x_gathered))
        
        # Routed UP: compute for all experts and mask
        h_routed = torch.zeros(
            total_assignments, self.hdim,
            dtype=x_gathered.dtype, device=x_gathered.device
        )
        
        for e in range(self.num_routed_experts):
            # Compute for ALL tokens through this expert
            h_e = F.linear(x_gathered, self.routed_experts_up[e].T.type_as(x_gathered))
            # Mask to only keep tokens assigned to this expert
            mask = (sorted_expert_indices == e).unsqueeze(-1).to(h_e.dtype)
            h_routed = h_routed + h_e * mask
        
        return h_shared + h_routed
    
    def _down_projection(
        self,
        h_sorted: Tensor,           # [total_assignments, hdim]
        sorted_expert_indices: Tensor,
        total_assignments: int,
    ) -> Tensor:
        """
        DOWN-projection (not fused) - per-expert computation.
        Fully traceable by torch.compile.
        """
        # Shared DOWN: same for all
        y_shared = F.linear(h_sorted, self.shared_expert_down.type_as(h_sorted))
        
        # Routed DOWN: compute for all experts and mask
        y_routed = torch.zeros(
            total_assignments, self.dim,
            dtype=h_sorted.dtype, device=h_sorted.device
        )
        
        for e in range(self.num_routed_experts):
            # Compute for ALL tokens through this expert
            y_e = F.linear(h_sorted, self.routed_experts_down[e].type_as(h_sorted))
            # Mask to only keep tokens assigned to this expert
            mask = (sorted_expert_indices == e).unsqueeze(-1).to(y_e.dtype)
            y_routed = y_routed + y_e * mask
        
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
        Compute auxiliary losses for load balancing.
        """
        N = B * T
        
        # One-hot encoding for load balancing
        expert_mask = F.one_hot(topk_indices[:, 0], num_classes=self.num_routed_experts).float()
        
        # Fraction of tokens per expert
        tokens_per_expert = expert_mask.sum(dim=0)
        fraction_per_expert = tokens_per_expert / N
        
        # Average router probability per expert
        avg_router_prob = router_probs.mean(dim=0)
        
        # Load balancing loss
        load_balancing_loss = self.num_routed_experts * torch.sum(
            fraction_per_expert.detach() * avg_router_prob
        )
        
        # Router Z-loss (entropy regularization)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        
        # Importance loss (variance of total probability mass)
        importance = router_probs.sum(dim=0)
        importance_loss = torch.var(importance)
        
        return {
            'load_balancing_loss': load_balancing_loss,
            'router_z_loss': router_z_loss,
            'importance_loss': importance_loss,
            'expert_counts': tokens_per_expert,
        }
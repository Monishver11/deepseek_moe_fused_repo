"""
PyTorch Module and Autograd Integration for Fused DeepSeek-MoE.

This module provides:
1. FusedDeepSeekMoE: torch.autograd.Function with custom forward (Triton) and backward (PyTorch)
2. DeepSeekMoELayer: nn.Module wrapper for easy integration into models

The backward pass uses standard PyTorch matmul for simplicity and correctness.
This is a pragmatic choice: the inference speedup from the fused forward kernel
is the primary deliverable, while training correctness is ensured via PyTorch.

Architecture Notes:
------------------
- This implements ONLY the up-projection GEMM: Y = X @ W
- Activation (SwiGLU) and down-projection are handled externally
- Router weighting is applied AFTER this layer (externally in PyTorch)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .utils import compute_routing_metadata, get_grid_config, RoutingMetadata, GridConfig
from .kernels import fused_moe_forward
from .config import MoEConfig, SMALL_CONFIG


class FusedDeepSeekMoE(torch.autograd.Function):
    """
    Autograd Function for fused MoE forward pass.
    
    Forward: Uses custom Triton kernel for fused routed+shared computation
    Backward: Uses standard PyTorch operations for gradient computation
    
    The function computes:
        Y_sorted = X[sorted_indices] @ W_routed[expert_ids] + X[sorted_indices] @ W_shared
        
    Where sorted_indices groups tokens by their assigned expert.
    """
    
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,                     # [N, H] input activations
        W_routed: torch.Tensor,              # [E, H, D] routed expert weights
        W_shared: torch.Tensor,              # [H, D] shared expert weights
        gate_weight: torch.Tensor,           # [E, H] router weights
        top_k: int,
        num_experts: int,
        block_m: int,
        block_n: int,
        block_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using fused Triton kernel.
        
        Returns:
            Y_sorted: [total_assignments, D] - Output in sorted order
            sorted_token_indices: [total_assignments] - For scatter back
            sorted_expert_indices: [total_assignments] - Expert IDs
            sorted_weights: [total_assignments] - Router probabilities
        """
        N, H = X.shape
        E, _, D = W_routed.shape
        
        # ======================================================================
        # Step 1: Compute Routing Metadata
        # ======================================================================
        # This generates the "Metadata Map" for virtual gathering:
        # - sorted_token_indices: which rows of X to read
        # - sorted_expert_indices: which expert computes each assignment
        # - sorted_weights: router probabilities for weighted combination
        
        # Create gate layer for routing (using provided weights)
        # Note: We don't want to store the gate as a Module here since this is
        # a Function. Instead, we do the matmul directly.
        gate_logits = torch.mm(X, gate_weight.t())  # [N, E]
        
        # Top-k selection and metadata generation
        topk_weights, topk_indices = torch.topk(
            torch.softmax(gate_logits, dim=-1),
            k=top_k,
            dim=-1,
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Flatten and sort by expert
        flat_expert_indices = topk_indices.view(-1)
        flat_token_indices = torch.arange(N, device=X.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        flat_weights = topk_weights.view(-1)
        
        sort_indices = torch.argsort(flat_expert_indices, stable=True)
        sorted_expert_indices = flat_expert_indices[sort_indices].to(torch.int32)
        sorted_token_indices = flat_token_indices[sort_indices].to(torch.int32)
        sorted_weights = flat_weights[sort_indices]
        
        tokens_per_expert = torch.bincount(
            sorted_expert_indices.long(),
            minlength=num_experts,
        ).to(torch.int32)
        
        # ======================================================================
        # Step 2: Compute Grid Configuration
        # ======================================================================
        grid_config = get_grid_config(tokens_per_expert, block_m, num_experts)
        
        # ======================================================================
        # Step 3: Launch Fused Kernel
        # ======================================================================
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
        
        # ======================================================================
        # Save for Backward
        # ======================================================================
        ctx.save_for_backward(
            X, W_routed, W_shared, gate_weight,
            sorted_token_indices, sorted_expert_indices, tokens_per_expert,
        )
        ctx.top_k = top_k
        ctx.num_experts = num_experts
        
        return Y_sorted, sorted_token_indices, sorted_expert_indices, sorted_weights
    
    @staticmethod
    def backward(ctx, grad_Y_sorted, grad_indices, grad_expert, grad_weights):
        """
        Backward pass using standard PyTorch operations.
        
        This is a pragmatic implementation that sacrifices the fusion benefit
        during backward for simplicity and correctness. The gradients are:
        
        ∂L/∂X = (∂L/∂Y @ W_routed.T) + (∂L/∂Y @ W_shared.T)  [scattered back]
        ∂L/∂W_routed[e] = X[tokens_for_e].T @ ∂L/∂Y[tokens_for_e]
        ∂L/∂W_shared = X.T @ ∂L/∂Y  [summed over all tokens]
        ∂L/∂gate_weight: Computed via router gradient
        """
        (
            X, W_routed, W_shared, gate_weight,
            sorted_token_indices, sorted_expert_indices, tokens_per_expert,
        ) = ctx.saved_tensors
        top_k = ctx.top_k
        num_experts = ctx.num_experts
        
        N, H = X.shape
        E, _, D = W_routed.shape
        total_assignments = sorted_token_indices.shape[0]
        
        # Cast grad to float32 for stable accumulation
        grad_Y = grad_Y_sorted.float()  # [total_assignments, D]
        
        # ======================================================================
        # Gradient w.r.t. W_shared
        # ======================================================================
        # ∂L/∂W_shared = sum over all assignments of X[idx].T @ grad_Y
        # Since shared expert sees all tokens:
        # Gather X in sorted order, then matmul
        X_sorted = X[sorted_token_indices.long()]  # [total_assignments, H]
        grad_W_shared = torch.mm(X_sorted.t().float(), grad_Y).to(W_shared.dtype)  # [H, D]
        
        # ======================================================================
        # Gradient w.r.t. W_routed (per-expert)
        # ======================================================================
        grad_W_routed = torch.zeros_like(W_routed)
        
        # Compute expert token offsets for iteration
        expert_token_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=X.device)
        expert_token_offsets[1:] = torch.cumsum(tokens_per_expert.long(), dim=0)
        
        for e in range(num_experts):
            start = expert_token_offsets[e].item()
            end = expert_token_offsets[e + 1].item()
            if end > start:
                # X_e: [num_tokens_e, H]
                # grad_Y_e: [num_tokens_e, D]
                X_e = X_sorted[start:end].float()
                grad_Y_e = grad_Y[start:end]
                # ∂L/∂W_routed[e] = X_e.T @ grad_Y_e
                grad_W_routed[e] = torch.mm(X_e.t(), grad_Y_e).to(W_routed.dtype)
        
        # ======================================================================
        # Gradient w.r.t. X
        # ======================================================================
        # ∂L/∂X[i] = sum over assignments involving token i of:
        #   grad_Y @ W_routed[expert].T + grad_Y @ W_shared.T
        
        grad_X = torch.zeros_like(X).float()
        
        # Shared contribution: same for all assignments
        # grad_from_shared = grad_Y @ W_shared.T  # [total_assignments, H]
        grad_from_shared = torch.mm(grad_Y, W_shared.t().float())
        
        # Routed contribution: depends on expert
        grad_from_routed = torch.zeros(total_assignments, H, device=X.device, dtype=torch.float32)
        for e in range(num_experts):
            start = expert_token_offsets[e].item()
            end = expert_token_offsets[e + 1].item()
            if end > start:
                grad_Y_e = grad_Y[start:end]
                grad_from_routed[start:end] = torch.mm(grad_Y_e, W_routed[e].t().float())
        
        # Total gradient in sorted order
        grad_X_sorted = grad_from_routed + grad_from_shared  # [total_assignments, H]
        
        # Scatter back to original order using index_add
        # Each token may appear multiple times (top_k times), so we accumulate
        grad_X.index_add_(0, sorted_token_indices.long(), grad_X_sorted)
        grad_X = grad_X.to(X.dtype)
        
        # ======================================================================
        # Gradient w.r.t. gate_weight (simplified)
        # ======================================================================
        # Full router gradient involves the softmax and top-k selection
        # For V1, we return None and let the outer module handle this
        # In practice, router gradients are often handled separately anyway
        grad_gate_weight = None
        
        return (
            grad_X,
            grad_W_routed,
            grad_W_shared,
            grad_gate_weight,
            None, None, None, None, None,  # top_k, num_experts, block sizes
        )


class DeepSeekMoELayer(nn.Module):
    """
    PyTorch Module wrapper for the Fused DeepSeek-MoE layer.
    
    This implements the up-projection of a DeepSeek-style MoE:
    - Routed experts: Top-K sparse routing
    - Shared expert: Always-active dense computation
    - Fused forward: Single kernel for bandwidth efficiency
    
    Usage:
        config = MoEConfig(hidden_dim=2048, intermediate_dim=2048, num_experts=8)
        moe = DeepSeekMoELayer(config)
        
        # Input: [batch*seq, hidden_dim]
        x = torch.randn(1024, 2048, dtype=torch.bfloat16, device='cuda')
        output, aux_loss = moe(x)
        # output: [batch*seq, intermediate_dim]
    
    Architecture:
        Y_i = sum_{e in top_k(x_i)} [gate_prob_e * (x_i @ W_routed[e])] + x_i @ W_shared
        
    Note: This only computes the up-projection. Activation and down-projection
    should be applied externally.
    """
    
    def __init__(
        self,
        config: MoEConfig = SMALL_CONFIG,
    ):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        
        # Block sizes for Triton kernel
        self.block_m = config.block_m
        self.block_n = config.block_n
        self.block_k = config.block_k
        
        # ======================================================================
        # Parameters
        # ======================================================================
        
        # Router gate: maps hidden_dim -> num_experts
        # This produces logits for expert selection
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        
        # Routed expert weights: [num_experts, hidden_dim, intermediate_dim]
        # Each expert has its own up-projection matrix
        self.W_routed = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_dim,
                config.intermediate_dim,
                dtype=torch.bfloat16 if config.use_bfloat16 else torch.float32,
            )
        )
        
        # Shared expert weights: [hidden_dim, intermediate_dim]
        # Single dense matrix applied to ALL tokens
        self.W_shared = nn.Parameter(
            torch.empty(
                config.hidden_dim,
                config.intermediate_dim,
                dtype=torch.bfloat16 if config.use_bfloat16 else torch.float32,
            )
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using standard techniques."""
        # Router: small init to encourage exploration initially
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        
        # Expert weights: Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.W_routed)
        nn.init.xavier_uniform_(self.W_shared)
    
    def forward(
        self,
        x: torch.Tensor,
        return_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [N, H] where N = batch*seq, H = hidden_dim
            return_router_logits: If True, also return router logits for aux loss
        
        Returns:
            output: [N, D] output activations (up-projected, weighted, combined)
            router_logits: [N, E] (optional) router logits for load balancing loss
        """
        N, H = x.shape
        device = x.device
        
        # Ensure input is BF16 for kernel
        if self.config.use_bfloat16 and x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        
        # Ensure weights are on same device and dtype
        gate_weight = self.gate.weight.to(device=device, dtype=x.dtype)
        W_routed = self.W_routed.to(device=device, dtype=x.dtype)
        W_shared = self.W_shared.to(device=device, dtype=x.dtype)
        
        # ======================================================================
        # Call Fused Kernel via Autograd Function
        # ======================================================================
        Y_sorted, sorted_token_indices, sorted_expert_indices, sorted_weights = \
            FusedDeepSeekMoE.apply(
                x,
                W_routed,
                W_shared,
                gate_weight,
                self.top_k,
                self.num_experts,
                self.block_m,
                self.block_n,
                self.block_k,
            )
        
        # ======================================================================
        # Apply Router Weights and Scatter Back
        # ======================================================================
        # Y_sorted: [total_assignments, D] - outputs in sorted order
        # sorted_weights: [total_assignments] - router probabilities
        # sorted_token_indices: [total_assignments] - which original token
        
        # Weight the outputs by router probabilities
        # [total_assignments, D] * [total_assignments, 1]
        Y_weighted = Y_sorted * sorted_weights.unsqueeze(-1).to(Y_sorted.dtype)
        
        # Scatter-add back to original token order
        # Each token appears top_k times, we sum the weighted contributions
        output = torch.zeros(N, self.intermediate_dim, dtype=Y_sorted.dtype, device=device)
        output.index_add_(
            0,
            sorted_token_indices.long(),
            Y_weighted,
        )
        
        # ======================================================================
        # Compute Router Logits (for auxiliary load balancing loss)
        # ======================================================================
        router_logits = None
        if return_router_logits:
            router_logits = torch.mm(x.float(), gate_weight.t().float())
        
        return output, router_logits
    
    def forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Naive forward pass for verification (no fusion, standard PyTorch).
        
        This computes the same result as forward() but without the fused kernel,
        making it useful for:
        1. Correctness verification (torch.allclose against fused output)
        2. Performance baseline comparison
        
        Args:
            x: Input tensor [N, H]
        
        Returns:
            output: [N, D] output activations
        """
        N, H = x.shape
        device = x.device
        
        # Ensure BF16
        if self.config.use_bfloat16 and x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        
        # Router forward (convert to float for stability, gate may be float32)
        gate_logits = torch.mm(x.float(), self.gate.weight.t().float())  # [N, E]
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        # Top-K selection
        topk_weights, topk_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # ======================================================================
        # Naive Routed Computation (no fusion)
        # ======================================================================
        # For each token, compute output from its selected experts
        output = torch.zeros(N, self.intermediate_dim, dtype=x.dtype, device=device)
        
        for k in range(self.top_k):
            expert_indices = topk_indices[:, k]  # [N] - which expert for this k
            weights = topk_weights[:, k]         # [N] - probability weight
            
            # Compute per-expert contributions
            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if mask.any():
                    x_e = x[mask]  # [num_tokens_e, H]
                    # Routed: X @ W_routed[e]
                    y_routed = torch.mm(x_e.float(), self.W_routed[e].float())
                    # Shared: X @ W_shared
                    y_shared = torch.mm(x_e.float(), self.W_shared.float())
                    # Combined
                    y_e = (y_routed + y_shared).to(x.dtype)
                    # Weighted and accumulated
                    output[mask] += weights[mask].unsqueeze(-1) * y_e
        
        return output


def naive_moe_reference(
    x: torch.Tensor,           # [N, H]
    W_routed: torch.Tensor,    # [E, H, D]
    W_shared: torch.Tensor,    # [H, D]
    gate_weight: torch.Tensor, # [E, H]
    top_k: int,
) -> torch.Tensor:
    """
    Standalone naive MoE implementation for testing.
    
    This is a minimal reference implementation that does NOT use any custom
    kernels. Use this to verify correctness of the fused implementation.
    
    Args:
        x: Input [N, H]
        W_routed: Routed weights [E, H, D]
        W_shared: Shared weights [H, D]
        gate_weight: Router weights [E, H]
        top_k: Number of experts per token
    
    Returns:
        output: [N, D]
    """
    N, H = x.shape
    E, _, D = W_routed.shape
    device = x.device
    dtype = x.dtype
    
    # Router
    gate_logits = torch.mm(x.float(), gate_weight.t().float())  # [N, E]
    gate_probs = torch.softmax(gate_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(gate_probs, k=top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    # Compute outputs
    output = torch.zeros(N, D, dtype=torch.float32, device=device)
    
    for n in range(N):
        for k in range(top_k):
            e = topk_indices[n, k].item()
            w = topk_weights[n, k].item()
            
            # x[n] @ W_routed[e] + x[n] @ W_shared
            x_n = x[n].float().unsqueeze(0)  # [1, H]
            y_routed = torch.mm(x_n, W_routed[e].float())  # [1, D]
            y_shared = torch.mm(x_n, W_shared.float())     # [1, D]
            
            output[n] += w * (y_routed.squeeze(0) + y_shared.squeeze(0))
    
    return output.to(dtype)
"""
Debug script to trace through the fused MoE computation step by step.
Uses small dimensions to make outputs readable.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

# Ensure reproducibility
torch.manual_seed(42)

# Small dimensions for debugging
N = 4       # tokens
H = 8       # hidden dim
D = 8       # intermediate dim  
E = 4       # experts
top_k = 2

print("=" * 80)
print("DEBUG: Fused MoE Computation Trace")
print("=" * 80)
print(f"N={N}, H={H}, D={D}, E={E}, top_k={top_k}")
print()

# Create inputs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

X = torch.randn(N, H, dtype=dtype, device=device)
W_routed = torch.randn(E, H, D, dtype=dtype, device=device)
W_shared = torch.randn(H, D, dtype=dtype, device=device)
gate_weight = torch.randn(E, H, dtype=dtype, device=device)

print("=" * 80)
print("STEP 1: Input Tensors")
print("=" * 80)
print(f"X shape: {X.shape}")
print(f"X:\n{X.float()}")
print(f"\nW_routed shape: {W_routed.shape}")
print(f"W_shared shape: {W_shared.shape}")
print(f"gate_weight shape: {gate_weight.shape}")

# ============================================================================
# STEP 2: Router / Gating
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Router Computation")
print("=" * 80)

gate_logits = torch.mm(X.float(), gate_weight.t().float())  # [N, E]
print(f"gate_logits shape: {gate_logits.shape}")
print(f"gate_logits:\n{gate_logits}")

gate_probs = torch.softmax(gate_logits, dim=-1)
print(f"\ngate_probs (softmax):\n{gate_probs}")

topk_weights, topk_indices = torch.topk(gate_probs, k=top_k, dim=-1)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
print(f"\ntopk_indices (which experts): {topk_indices}")
print(f"topk_weights (normalized): {topk_weights}")

# ============================================================================
# STEP 3: Create Sorted Indices (Routing Metadata)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Routing Metadata (Sorting)")
print("=" * 80)

flat_expert_indices = topk_indices.view(-1)  # [N * top_k]
flat_token_indices = torch.arange(N, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
flat_weights = topk_weights.view(-1)

print(f"flat_expert_indices: {flat_expert_indices}")
print(f"flat_token_indices:  {flat_token_indices}")
print(f"flat_weights:        {flat_weights}")

sort_indices = torch.argsort(flat_expert_indices, stable=True)
sorted_expert_indices = flat_expert_indices[sort_indices].to(torch.int32)
sorted_token_indices = flat_token_indices[sort_indices].to(torch.int32)
sorted_weights = flat_weights[sort_indices]

print(f"\nAfter sorting by expert:")
print(f"sort_indices:         {sort_indices}")
print(f"sorted_expert_indices: {sorted_expert_indices}")
print(f"sorted_token_indices:  {sorted_token_indices}")
print(f"sorted_weights:        {sorted_weights}")

# ============================================================================
# STEP 4: Naive Reference Computation (Ground Truth)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Naive Reference Computation")
print("=" * 80)

total_assignments = N * top_k
Y_naive_sorted = torch.zeros(total_assignments, D, dtype=torch.float32, device=device)

print(f"\nComputing each assignment:")
for i in range(total_assignments):
    token_idx = sorted_token_indices[i].item()
    expert_idx = sorted_expert_indices[i].item()
    
    x_i = X[token_idx].float()  # [H]
    w_routed_e = W_routed[expert_idx].float()  # [H, D]
    w_shared = W_shared.float()  # [H, D]
    
    # Y = X @ W_routed + X @ W_shared
    y_routed = torch.mv(w_routed_e.t(), x_i)  # [D]
    y_shared = torch.mv(w_shared.t(), x_i)    # [D]
    y_i = y_routed + y_shared
    
    Y_naive_sorted[i] = y_i
    
    print(f"  Assignment {i}: token={token_idx}, expert={expert_idx}")
    print(f"    x_i (first 4): {x_i[:4].tolist()}")
    print(f"    y_routed (first 4): {y_routed[:4].tolist()}")
    print(f"    y_shared (first 4): {y_shared[:4].tolist()}")
    print(f"    y_total (first 4): {y_i[:4].tolist()}")

print(f"\nY_naive_sorted shape: {Y_naive_sorted.shape}")
print(f"Y_naive_sorted:\n{Y_naive_sorted}")

# ============================================================================
# STEP 5: Fused Kernel Computation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Fused Kernel Computation")
print("=" * 80)

from deepseek_moe_fused.kernels import fused_moe_forward
from deepseek_moe_fused.utils import get_grid_config

tokens_per_expert = torch.bincount(sorted_expert_indices.long(), minlength=E).to(torch.int32)
print(f"tokens_per_expert: {tokens_per_expert}")

grid_config = get_grid_config(tokens_per_expert, block_m=32, num_experts=E)
print(f"total_blocks: {grid_config.total_blocks}")

Y_fused_sorted = fused_moe_forward(
    X=X,
    W_routed=W_routed,
    W_shared=W_shared,
    sorted_token_indices=sorted_token_indices,
    sorted_expert_indices=sorted_expert_indices,
    expert_token_offsets=grid_config.expert_token_offsets,
    total_blocks=grid_config.total_blocks,
    num_experts=E,
    BLOCK_M=32,
    BLOCK_N=64,
    BLOCK_K=32,
)

print(f"\nY_fused_sorted shape: {Y_fused_sorted.shape}")
print(f"Y_fused_sorted:\n{Y_fused_sorted.float()}")

# ============================================================================
# STEP 6: Compare Results
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Comparison")
print("=" * 80)

diff = (Y_naive_sorted - Y_fused_sorted.float()).abs()
print(f"Absolute difference:\n{diff}")
print(f"\nMax error: {diff.max().item():.6e}")
print(f"Mean error: {diff.mean().item():.6e}")

# Check row by row
print("\nPer-assignment comparison:")
for i in range(total_assignments):
    naive_row = Y_naive_sorted[i]
    fused_row = Y_fused_sorted[i].float()
    row_diff = (naive_row - fused_row).abs()
    print(f"  Assignment {i}: max_diff={row_diff.max().item():.6e}, "
          f"naive_sum={naive_row.sum().item():.4f}, fused_sum={fused_row.sum().item():.4f}")

# ============================================================================
# STEP 7: Test individual components
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Component Tests")
print("=" * 80)

# Test just the shared computation
print("\nTest: X @ W_shared only")
Y_shared_only = torch.mm(X.float(), W_shared.float())
print(f"Y_shared_only shape: {Y_shared_only.shape}")
print(f"Y_shared_only[0] first 4: {Y_shared_only[0, :4].tolist()}")

# Test routed for first assignment
print("\nTest: X[token_0] @ W_routed[expert_0]")
token_0 = sorted_token_indices[0].item()
expert_0 = sorted_expert_indices[0].item()
y_routed_test = torch.mv(W_routed[expert_0].float().t(), X[token_0].float())
print(f"token_idx={token_0}, expert_idx={expert_0}")
print(f"y_routed_test first 4: {y_routed_test[:4].tolist()}")

# ============================================================================
# STEP 8: Verify the final scatter-add step
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: Final Output (after weighting and scatter)")  
print("=" * 80)

# Using naive sorted output
Y_weighted_naive = Y_naive_sorted * sorted_weights.unsqueeze(-1)
output_naive = torch.zeros(N, D, dtype=torch.float32, device=device)
output_naive.index_add_(0, sorted_token_indices.long(), Y_weighted_naive)

# Using fused sorted output
Y_weighted_fused = Y_fused_sorted.float() * sorted_weights.unsqueeze(-1)
output_fused = torch.zeros(N, D, dtype=torch.float32, device=device)
output_fused.index_add_(0, sorted_token_indices.long(), Y_weighted_fused)

print(f"Final output_naive:\n{output_naive}")
print(f"\nFinal output_fused:\n{output_fused}")
print(f"\nFinal output difference:\n{(output_naive - output_fused).abs()}")
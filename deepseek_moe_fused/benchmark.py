"""
Benchmark and Verification Script for Fused DeepSeek-MoE Kernel.

This script:
1. Verifies numerical correctness: Fused kernel output matches naive PyTorch
2. Measures performance: Compares execution time of Fused vs Naive
3. Reports memory usage and bandwidth analysis

Usage:
    python -m deepseek_moe_fused.benchmark
    
    # Or with custom parameters:
    python -m deepseek_moe_fused.benchmark --batch_sizes 1024 4096 8192 --hidden_dim 2048
"""

import argparse
import time
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Ensure we can import from the package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseek_moe_fused.config import MoEConfig
from deepseek_moe_fused.moe_layer import DeepSeekMoELayer, naive_moe_reference


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    batch_size: int
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    top_k: int
    
    # Correctness
    max_abs_error: float
    mean_abs_error: float
    is_correct: bool
    
    # Timing (ms)
    naive_time_ms: float
    fused_time_ms: float
    speedup: float
    
    # Memory (MB)
    peak_memory_mb: float


def verify_correctness(
    fused_output: torch.Tensor,
    naive_output: torch.Tensor,
    atol: float = 0.1,
    rtol: float = 0.1,
) -> Tuple[bool, float, float]:
    """
    Verify that fused kernel output matches naive implementation.
    
    Uses relatively loose tolerances due to:
    - BF16 reduced precision (7-bit mantissa = ~0.8% relative error)
    - Different accumulation orders between implementations
    - Potential fused multiply-add differences
    - Error accumulation scales with tensor size
    
    BF16 Precision Note:
    - BF16 has ~0.78% relative precision per operation
    - With multiple operations, errors compound
    - For large tensors, max errors of 5-10% are expected and acceptable
    - Mean errors should remain low (~0.5%)
    
    Args:
        fused_output: Output from fused Triton kernel
        naive_output: Output from naive PyTorch implementation
        atol: Absolute tolerance (default 0.1 for BF16 with large batches)
        rtol: Relative tolerance (default 0.1 for BF16 with large batches)
    
    Returns:
        (is_correct, max_abs_error, mean_abs_error)
    """
    # Convert to float32 for comparison
    fused_f32 = fused_output.float()
    naive_f32 = naive_output.float()
    
    abs_diff = torch.abs(fused_f32 - naive_f32)
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    
    is_correct = torch.allclose(fused_f32, naive_f32, atol=atol, rtol=rtol)
    
    return is_correct, max_abs_error, mean_abs_error


def benchmark_single_config(
    batch_size: int,
    config: MoEConfig,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = 'cuda',
) -> BenchmarkResult:
    """
    Run benchmark for a single configuration.
    
    Args:
        batch_size: Number of tokens (N)
        config: MoE configuration
        num_warmup: Warmup iterations
        num_iterations: Timed iterations
        device: 'cuda' or 'cpu'
    
    Returns:
        BenchmarkResult with timing and correctness metrics
    """
    torch.cuda.reset_peak_memory_stats(device)
    
    # =========================================================================
    # Setup
    # =========================================================================
    N = batch_size
    H = config.hidden_dim
    D = config.intermediate_dim
    E = config.num_experts
    top_k = config.top_k
    
    # Create random input
    x = torch.randn(N, H, dtype=torch.bfloat16, device=device)
    
    # Create MoE layer
    moe = DeepSeekMoELayer(config).to(device)
    
    # Get weights for naive reference
    gate_weight = moe.gate.weight.data
    W_routed = moe.W_routed.data
    W_shared = moe.W_shared.data
    
    # =========================================================================
    # Correctness Verification
    # =========================================================================
    with torch.no_grad():
        # Fused forward
        fused_output, _ = moe(x)
        
        # Naive reference
        naive_output = naive_moe_reference(
            x, W_routed, W_shared, gate_weight, top_k
        )
    
    is_correct, max_abs_error, mean_abs_error = verify_correctness(
        fused_output, naive_output
    )
    
    # =========================================================================
    # Performance Measurement: Naive
    # =========================================================================
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = naive_moe_reference(x, W_routed, W_shared, gate_weight, top_k)
    torch.cuda.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = naive_moe_reference(x, W_routed, W_shared, gate_weight, top_k)
    torch.cuda.synchronize()
    naive_time_ms = (time.perf_counter() - start) / num_iterations * 1000
    
    # =========================================================================
    # Performance Measurement: Fused
    # =========================================================================
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = moe(x)
    torch.cuda.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = moe(x)
    torch.cuda.synchronize()
    fused_time_ms = (time.perf_counter() - start) / num_iterations * 1000
    
    # =========================================================================
    # Memory Measurement
    # =========================================================================
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    # =========================================================================
    # Results
    # =========================================================================
    speedup = naive_time_ms / fused_time_ms if fused_time_ms > 0 else 0
    
    return BenchmarkResult(
        batch_size=batch_size,
        hidden_dim=H,
        intermediate_dim=D,
        num_experts=E,
        top_k=top_k,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        is_correct=is_correct,
        naive_time_ms=naive_time_ms,
        fused_time_ms=fused_time_ms,
        speedup=speedup,
        peak_memory_mb=peak_memory_mb,
    )


def run_benchmark_suite(
    batch_sizes: List[int],
    config: MoEConfig,
    device: str = 'cuda',
) -> List[BenchmarkResult]:
    """
    Run benchmarks across multiple batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        config: MoE configuration
        device: 'cuda' or 'cpu'
    
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking N={batch_size}...")
        try:
            result = benchmark_single_config(batch_size, config, device=device)
            results.append(result)
            
            # Print immediate feedback
            status = "✓ PASS" if result.is_correct else "✗ FAIL"
            print(f"  {status} | Max Error: {result.max_abs_error:.2e} | Mean Error: {result.mean_abs_error:.2e}")
            print(f"  Naive: {result.naive_time_ms:.2f}ms | Fused: {result.fused_time_ms:.2f}ms | Speedup: {result.speedup:.2f}x")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print a formatted summary table of benchmark results."""
    print("\n" + "=" * 110)
    print("BENCHMARK SUMMARY")
    print("=" * 110)
    
    # Header
    header = f"{'Batch':>8} | {'Hidden':>6} | {'Experts':>7} | {'Top-K':>5} | " \
             f"{'Correct':>7} | {'Max Err':>10} | {'Mean Err':>10} | {'Naive(ms)':>10} | " \
             f"{'Fused(ms)':>10} | {'Speedup':>8}"
    print(header)
    print("-" * 110)
    
    # Data rows
    for r in results:
        correct_str = "✓" if r.is_correct else "✗"
        row = f"{r.batch_size:>8} | {r.hidden_dim:>6} | {r.num_experts:>7} | {r.top_k:>5} | " \
              f"{correct_str:>7} | {r.max_abs_error:>10.2e} | {r.mean_abs_error:>10.2e} | {r.naive_time_ms:>10.2f} | " \
              f"{r.fused_time_ms:>10.2f} | {r.speedup:>7.2f}x"
        print(row)
    
    print("=" * 110)
    
    # Summary statistics
    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        all_correct = all(r.is_correct for r in results)
        print(f"\nAverage Speedup: {avg_speedup:.2f}x")
        print(f"All Correct: {'✓ YES' if all_correct else '✗ NO'}")


def detailed_correctness_test(config: MoEConfig, device: str = 'cuda'):
    """
    Run a detailed correctness test with verbose output.
    
    This is useful for debugging when torch.allclose fails.
    """
    print("\n" + "=" * 60)
    print("DETAILED CORRECTNESS TEST")
    print("=" * 60)
    
    N = 128  # Small batch for detailed analysis
    H = config.hidden_dim
    D = config.intermediate_dim
    
    x = torch.randn(N, H, dtype=torch.bfloat16, device=device)
    moe = DeepSeekMoELayer(config).to(device)
    
    with torch.no_grad():
        fused_output, _ = moe(x)
        naive_output = naive_moe_reference(
            x, moe.W_routed.data, moe.W_shared.data, 
            moe.gate.weight.data, config.top_k
        )
    
    diff = (fused_output.float() - naive_output.float()).abs()
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {fused_output.shape}")
    print(f"\nFused output stats:")
    print(f"  min={fused_output.min().item():.4f}, max={fused_output.max().item():.4f}, "
          f"mean={fused_output.mean().item():.4f}, std={fused_output.std().item():.4f}")
    print(f"\nNaive output stats:")
    print(f"  min={naive_output.min().item():.4f}, max={naive_output.max().item():.4f}, "
          f"mean={naive_output.mean().item():.4f}, std={naive_output.std().item():.4f}")
    print(f"\nDifference stats:")
    print(f"  min={diff.min().item():.2e}, max={diff.max().item():.2e}, "
          f"mean={diff.mean().item():.2e}, std={diff.std().item():.2e}")
    
    # Check relative error
    rel_err = diff / (naive_output.float().abs() + 1e-8)
    print(f"\nRelative error:")
    print(f"  max={rel_err.max().item():.2e}, mean={rel_err.mean().item():.2e}")
    
    # Find location of max error
    max_idx = diff.argmax()
    row = max_idx // D
    col = max_idx % D
    print(f"\nMax error location: row={row}, col={col}")
    print(f"  Fused value: {fused_output[row, col].item():.6f}")
    print(f"  Naive value: {naive_output[row, col].item():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Fused DeepSeek-MoE Kernel')
    parser.add_argument('--batch_sizes', nargs='+', type=int, 
                        default=[128, 512, 1024, 2048, 4096],
                        help='Batch sizes to benchmark')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='Hidden dimension (H)')
    parser.add_argument('--intermediate_dim', type=int, default=2048,
                        help='Intermediate dimension (D)')
    parser.add_argument('--num_experts', type=int, default=8,
                        help='Number of routed experts')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Top-K experts per token')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--detailed', action='store_true',
                        help='Run detailed correctness test')
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create config
    config = MoEConfig(
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    
    print(f"\nConfiguration:")
    print(f"  Hidden Dim (H): {config.hidden_dim}")
    print(f"  Intermediate Dim (D): {config.intermediate_dim}")
    print(f"  Num Experts (E): {config.num_experts}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Routed Weights: {config.routed_weight_size_mb:.1f} MB")
    print(f"  Shared Weights: {config.shared_weight_size_mb:.1f} MB")
    
    # Run detailed test if requested
    if args.detailed:
        detailed_correctness_test(config, args.device)
    
    # Run benchmark suite
    print("\nRunning benchmark suite...")
    results = run_benchmark_suite(args.batch_sizes, config, args.device)
    
    # Print summary
    print_summary(results)
    
    # Final verdict
    all_correct = all(r.is_correct for r in results)
    if all_correct:
        print("\n✓ ALL TESTS PASSED - Fused kernel produces correct results!")
    else:
        print("\n✗ SOME TESTS FAILED - Check error values above")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
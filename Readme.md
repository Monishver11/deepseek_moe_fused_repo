# Fused DeepSeek-MoE Kernel

A high-performance Triton kernel implementing DeepSeek-style Mixture of Experts with fused Routed + Shared expert computation.

## Key Optimization

Standard MoE loads input activations **twice** (once for routed experts, once for shared expert). This implementation fuses both computations into a single kernel, loading `X` **once** into SRAM and performing both GEMMs in registers.

```
Naive:   Load X → Compute Routed → Load X → Compute Shared  (2 HBM reads)
Fused:   Load X → Compute Routed + Shared simultaneously    (1 HBM read)
```

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate deepseek_moe_fused

# Verify installation
python -c "import torch; import triton; print(f'PyTorch: {torch.__version__}, Triton: {triton.__version__}')"
```

### Option 2: Pip with venv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch with CUDA first (for RTX 4070 / CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import triton; print(f'PyTorch: {torch.__version__}, Triton: {triton.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Run Correctness Test + Benchmark

```bash
# Activate environment first
conda activate deepseek_moe_fused  # or source venv/bin/activate

# Basic benchmark (default: E=8, H=2048, D=2048, top_k=2)
python -m benchmark

# With detailed correctness analysis
python -m benchmark --detailed

# Custom configuration
python -m benchmark \
    --batch_sizes 512 1024 2048 4096 8192 \
    --hidden_dim 2048 \
    --intermediate_dim 2048 \
    --num_experts 8 \
    --top_k 2
```

### Use in Your Code

```python
import torch
from deepseek_moe_fused import DeepSeekMoELayer
from deepseek_moe_fused.config import MoEConfig

# Configure
config = MoEConfig(
    hidden_dim=2048,
    intermediate_dim=2048,
    num_experts=8,
    top_k=2,
)

# Create layer
moe = DeepSeekMoELayer(config).cuda()

# Forward pass
x = torch.randn(1024, 2048, dtype=torch.bfloat16, device='cuda')
output, router_logits = moe(x, return_router_logits=True)

print(f"Input: {x.shape}")       # [1024, 2048]
print(f"Output: {output.shape}") # [1024, 2048]
```

## Project Structure

```
deepseek_moe_fused/
├── __init__.py          # Package exports
├── config.py            # MoEConfig dataclass
├── utils.py             # Routing metadata & grid topology
├── kernels.py           # Triton fused forward kernel
├── moe_layer.py         # PyTorch Module & Autograd Function
├── benchmark.py         # Verification & performance testing
├── environment.yml      # Conda environment spec
├── requirements.txt     # Pip requirements
└── README.md            # This file
```

## Expected Output

```
Using GPU: NVIDIA GeForce RTX 4070
CUDA Version: 12.1

Configuration:
  Hidden Dim (H): 2048
  Intermediate Dim (D): 2048
  Num Experts (E): 8
  Top-K: 2
  Routed Weights: 64.0 MB
  Shared Weights: 8.0 MB

Running benchmark suite...

Benchmarking N=512...
  ✓ PASS | Max Error: 3.91e-03
  Naive: 12.45ms | Fused: 8.32ms | Speedup: 1.50x

Benchmarking N=1024...
  ✓ PASS | Max Error: 4.12e-03
  Naive: 24.18ms | Fused: 14.56ms | Speedup: 1.66x

Benchmarking N=4096...
  ✓ PASS | Max Error: 5.23e-03
  Naive: 95.32ms | Fused: 52.14ms | Speedup: 1.83x
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with Compute Capability >= 7.0 (RTX 20xx or newer)
- **VRAM**: >= 8GB recommended (12GB for larger configs)
- **CUDA**: 12.1+ recommended
- **Driver**: 525.60+ recommended

## Troubleshooting

### "Triton compilation failed"
```bash
# Clear Triton cache and retry
rm -rf ~/.triton/cache
python -m benchmark
```

### "CUDA out of memory"
```bash
# Use smaller batch sizes
python -m benchmark --batch_sizes 128 256 512

# Or reduce expert count
python -m benchmark --num_experts 4
```

### "torch.allclose failed"
```bash
# Run detailed test to diagnose
python -m benchmark --detailed --batch_sizes 128
```

## Architecture Notes

This implements **only the up-projection GEMM**:
```
Y = X @ W_routed[selected_experts] + X @ W_shared
```

Activation (SwiGLU) and down-projection should be applied externally. Router weighting is handled in PyTorch after the kernel returns.
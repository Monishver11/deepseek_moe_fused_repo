"""
Setup script for deepseek_moe_fused package.

Installation:
    pip install -e .

This makes the package importable from anywhere:
    from deepseek_moe_fused import FusedDeepSeekMoEMLP
"""

from setuptools import setup, find_packages

setup(
    name="deepseek_moe_fused",
    version="0.1.0",
    description="Fused Triton kernel for DeepSeek-style Mixture of Experts",
    author="Abdur",
    packages=[""],  # Treat current directory as package
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "triton>=2.2.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "einops>=0.6.0",
        ],
    },
)
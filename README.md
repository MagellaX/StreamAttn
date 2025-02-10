# StreamAttention: Multi-GPU FlashAttention with Triton & PyTorch Distributed

StreamAttention is an experimental, high-performance implementation of the FlashAttention mechanism using Triton and PyTorch Distributed. It fuses advanced techniques—tiled online softmax, efficient memory access, and multi-GPU workload partitioning—into a single kernel to accelerate attention operations in transformer models.

> **Disclaimer:**  
> This project is a research prototype. It is not yet production-ready and may contain bugs, numerical instability issues, or challenges when scaling. Use at your own risk, and feel free to report bugs (or share memes). Contributions and feedback are highly welcome!

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Single-GPU Usage](#single-gpu-usage)
  - [Multi-GPU Usage](#multi-gpu-usage)
- [Technical Details](#technical-details)
  - [Parameters Explained](#parameters-explained)
  - [Tiling and Online Softmax](#tiling-and-online-softmax)
- [Known Issues & Torchrun Problems](#known-issues--torchrun-problems)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

StreamAttention is designed to:
- **Accelerate Attention Computation:**  
  Fuse the scaled dot-product and online softmax updates into a single Triton kernel.
- **Scale Across GPUs:**  
  Partition query workloads using PyTorch Distributed and NCCL for multi-GPU execution.
- **Serve as a Research Prototype:**  
  Provide a modular codebase that can be extended and optimized for production, with detailed documentation of design choices and parameters.

The primary objective is to push the envelope on GPU optimization for attention mechanisms—helping researchers and developers reduce memory overhead, lower latency, and scale transformer models effectively. Whether you're aiming for academic breakthroughs or commercial-grade performance, StreamAttention offers a cutting-edge foundation.

---


---

## Installation

### Prerequisites

- **Hardware:** NVIDIA GPU(s) 
- **Software:**  
  - Python 3.7+  
  - CUDA Toolkit 11.0+  
  - PyTorch (with CUDA support)  
  - Triton (install via pip)  
- **Optional:** Docker for containerized builds.

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/stream_attention.git
   cd stream_attention
   ```

Usage
Single-GPU Usage
Use the Python API to run the Triton-based attention kernel on a single GPU. For example:
```bash
import torch
from flash_attention.kernels import flash_attention

# Define dimensions (example values for prototype)
d_model = 128
global_seq_len = 1024  # Total number of keys/values
local_seq_len = 16     # Number of queries processed on this GPU

# Create random tensors on the GPU
Q = torch.randn(local_seq_len, d_model, device="cuda")
K = torch.randn(global_seq_len, d_model, device="cuda")
V = torch.randn(global_seq_len, d_model, device="cuda")

# Compute attention output
O = flash_attention(Q, K, V)
print("Output shape:", O.shape)
```

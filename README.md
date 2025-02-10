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
Multi-GPU Usage
StreamAttention supports multi-GPU execution using PyTorch Distributed (NCCL backend). To run a multi-GPU demo:

Set Environment Variables:

When running in distributed mode, set RANK and WORLD_SIZE (or use a launcher like torchrun). For example:
```bash
export RANK=0
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```
Run the Multi-GPU Demo:

The demo script examples/demo_multi_gpu.py initializes the distributed environment, partitions queries, executes the kernel on each GPU, and optionally gathers outputs:
```bash
python examples/demo_multi_gpu.py
```
Note on torchrun:
Many developers on X/Twitter have noted that torchrun can be finicky—ensure that environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are set correctly. If you encounter errors like "NCCL error" or "Connection timeout," double-check your network configuration, GPU visibility, and port availability. Sometimes it feels like herding cats (or GPUs), but hang in there!


# Contributing

## How to Contribute

Contributions are welcome! To contribute:

1. Fork the repository and create a feature branch
2. Submit pull requests with clear descriptions of your changes
3. Report issues or suggest enhancements by opening an issue on GitHub

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions, suggestions, or collaboration opportunities, please contact [alphacr792@gmail.com] or open an issue on GitHub.

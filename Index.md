# Technical Documentation

## Parameters Explained

The following parameters are crucial for understanding and configuring the system:

`d_model = 128` represents the dimensionality of the feature vectors. While 128 is used here for prototyping, production models might use larger dimensions (e.g., 512, 768, or 1024).

`global_seq_len = 1024` defines the total number of keys (and values). This parameter determines how many tiles the kernel processes.

`local_seq_len` / `total_queries` specifies the number of query vectors processed on a single GPU. In multi-GPU setups, this value is partitioned among the available GPUs.

### Kernel Constants

- `TILE_K = 64`: Number of keys processed per tile, reducing global memory accesses by reusing shared memory
- `running_max = -1e9`: Initial value for the online softmax's running maximum, ensuring any computed dot product will update the maximum
- `scale = 1.0 / sqrt(d_model)`: A scaling factor for normalizing dot products in the attention computation

## Tiling and Online Softmax

The Triton kernel divides the key/value matrices into tiles of size `TILE_K`. For each tile, the process involves:

### Loading
Keys and values are loaded with proper masking (to handle tail cases).

### Computation
The dot product between the query and each key is computed and scaled.

### Online Softmax
The kernel maintains several key variables:
- `acc_num`: Accumulated weighted sum of value vectors
- `acc_den`: Accumulated sum of exponentials
- `running_max`: Running maximum for numerical stability

### Output
The final attention output is computed as `acc_num / acc_den`.

## Known Issues & Torchrun Problems

### Torchrun Configuration

Proper configuration of environment variables is essential:
- Ensure that environment variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) are correctly set. Misconfiguration is a common source of errors—like dialing the wrong number before making a call.
- NCCL errors may occur if GPUs are not visible or if there are network issues. If you see NCCL timeouts or connectivity errors, double-check your setup.
- Some users report that torchrun can conflict with Docker network settings. If you run inside a container, ensure it has proper GPU and port access.

### Numerical Stability

With very large sequence lengths or higher `d_model` values, the online softmax may suffer from precision issues. Consider using higher-precision arithmetic or initializing `running_max` with `float('-inf')` if supported.

### Performance Tuning

The chosen tile size (`TILE_K = 64`) is a starting point. Optimal performance may require experimenting with different tile sizes based on your hardware. If performance is suboptimal, use profiling tools (e.g., NVIDIA Nsight Systems) to identify and resolve bottlenecks.

### Triton Limitations

Currently, Triton does not support the `cp.async` instruction. This implementation relies on `tl.load` with masking and the Triton scheduler to hide memory latency. Future Triton updates might offer enhanced functionality.

### Distributed Setup Issues

When scaling to many GPUs, inter-device communication overhead can become significant. Be prepared to tune NCCL settings and experiment with partitioning strategies. Common errors include timeouts and "NCCL communication failed" messages—remember, even GPUs sometimes need a little extra coaxing!

## Troubleshooting

### Kernel Launch Failures

Verify that your GPU supports Triton (preferably NVIDIA Ampere or later) and that your CUDA drivers are current.

### Numerical Issues

If outputs contain NaN or Inf values, review your scaling factor and `running_max` initialization. Using `float('-inf')` might improve stability.

### Distributed Errors with torchrun

- Double-check that all required environment variables are correctly set
- Confirm that NCCL is properly installed and your network settings permit inter-GPU communication
- If running inside Docker, ensure the container has full GPU and network port access

### Performance Bottlenecks

- Profile your application (using tools like NVIDIA Nsight Systems) to pinpoint bottlenecks
- Experiment with different values for `TILE_K` and adjust kernel launch configurations accordingly

And remember, if torchrun gives you a headache, just take a deep breath, double-check your environment variables, and maybe grab a cup of coffee—after all, even our GPUs need a break sometimes!

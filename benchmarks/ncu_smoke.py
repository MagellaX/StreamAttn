"""Tiny CUDA workload for checking whether Nsight Compute can profile Python."""

import json

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    a = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    b = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        out = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "matmul_ms": start.elapsed_time(end) / 10,
                "checksum": float(out[0, 0].detach().cpu()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

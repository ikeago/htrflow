"""Minimal container smoke check for Blackwell-capable PyTorch builds."""

import json
import time

import torch
import torch.nn.functional as functional


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = torch.device("cuda:0")
capability = torch.cuda.get_device_capability(device)
architectures = torch.cuda.get_arch_list()
expected_architecture = f"sm_{capability[0]}{capability[1]}"
if capability[0] >= 10 and expected_architecture not in architectures:
    raise RuntimeError(f"PyTorch does not advertise {expected_architecture} support: {architectures}")

started = time.perf_counter()
left = torch.randn((2048, 2048), device=device)
right = torch.randn((2048, 2048), device=device)
matrix_checksum = float((left @ right)[0, 0].item())
image = torch.randn((1, 3, 256, 256), device=device)
filters = torch.randn((32, 3, 3, 3), device=device)
convolution_checksum = float(functional.conv2d(image, filters)[0, 0, 0, 0].item())
torch.cuda.synchronize(device)

print(
    json.dumps(
        {
            "status": "passed",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": list(capability),
            "architectures": architectures,
            "elapsed_seconds": time.perf_counter() - started,
            "matrix_checksum": matrix_checksum,
            "convolution_checksum": convolution_checksum,
        },
        sort_keys=True,
    )
)

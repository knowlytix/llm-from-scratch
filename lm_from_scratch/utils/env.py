"""Environment inspection and reproducibility helpers.

These two utilities are used by every chapter notebook to (1) confirm the
runtime is what the book assumes and (2) seed the RNGs deterministically.
"""

from __future__ import annotations

import platform
import random
from typing import Any

import numpy as np
import torch


def check_environment() -> dict[str, Any]:
    """Return a snapshot of the runtime relevant for reproducing the book.

    The returned dict has stable keys so notebooks can render it as a small
    table without conditional logic.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    cuda_version = torch.version.cuda if cuda_available else None
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "num_cuda_devices": torch.cuda.device_count() if cuda_available else 0,
    }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (including CUDA) for reproducibility.

    The book guarantees reproducibility to within tolerance, not bit-equal,
    when mixed precision or fused kernels are enabled (see Ch. 18).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""Small runtime helpers used by the offline and online inference entrypoints."""

import random

import numpy as np
import torch
import psutil


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)

def get_free_ram(
    device: torch.device
) -> int:
    """Get the amount of RAM/VRAM available on the device."""
    if device.type == "cuda":
        # count free VRAM using cuda apis
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        return r-a  # free inside reserved
    else:
        # get free RAM using psutil
        return psutil.virtual_memory().available
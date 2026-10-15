"""Optional CUDA helpers.

The reproducible baseline remains CPU-only. This module makes GPU capability
observable and provides a safe device selector for optional torch extensions.
"""

from __future__ import annotations


def cuda_status() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "PyTorch is not installed"}
    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "device": torch.cuda.get_device_name(0) if available else "cpu",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "reason": None if available else "CUDA runtime is unavailable",
    }


def select_device(requested: str = "auto") -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("requested device must be auto, cpu, or cuda")
    status = cuda_status()
    if requested == "cuda" and not status["available"]:
        raise RuntimeError(f"CUDA requested but unavailable: {status['reason']}")
    if requested == "auto":
        return "cuda" if status["available"] else "cpu"
    return requested

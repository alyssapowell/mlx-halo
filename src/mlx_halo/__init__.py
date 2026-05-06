"""
mlx-halo — Pre-flight safety checks for MLX models on Apple Silicon.

Prevents kernel panics from overlapping Metal GPU allocations during model
loading and swapping. Named after F1's Halo cockpit protection device.

Quick start::

    from mlx_halo import preflight

    result = preflight(model_size_gb=8.0)
    # Raises MemoryError if unsafe to load

Full control::

    from mlx_halo import HaloCheck

    halo = HaloCheck(
        total_vram_gb=32,
        pain_threshold=0.7,
        conflict_check=lambda: pytorch_model is not None,
    )
    result = halo.check_all(estimated_model_gb=18.0)
"""

from .safety import HaloCheck, preflight
from .monitor import SystemMonitor, get_monitor
from .pain import PainCalculator, get_pain_calculator, get_current_pain
from .memory import get_gpu_memory_status, clear_gpu_cache, wait_for_memory_drain
from .types import SystemMetrics, PainProfile, HaloResult, HealthStatus, MemoryStatus

__version__ = "1.0.0"

__all__ = [
    "preflight",
    "HaloCheck",
    "SystemMonitor",
    "get_monitor",
    "PainCalculator",
    "get_pain_calculator",
    "get_current_pain",
    "get_gpu_memory_status",
    "clear_gpu_cache",
    "wait_for_memory_drain",
    "SystemMetrics",
    "PainProfile",
    "HaloResult",
    "HealthStatus",
    "MemoryStatus",
]

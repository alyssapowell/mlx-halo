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

import os as _os

# ── Apple GPU driver workaround ──────────────────────────────────────────
# Relaxes the Metal command buffer context store timeout to reduce kernel
# panics on long-running GPU workloads. Zero-cost env var hint to the
# IOGPUFamily driver — safe to set unconditionally.
#
# Suggested by @zcbenz (MLX maintainer) in ml-explore/mlx#3267.
# Surfaced by Harperbot/metal-guard (runtime MLX safety layer).
if "AGX_RELAX_CDM_CTXSTORE_TIMEOUT" not in _os.environ:
    _os.environ["AGX_RELAX_CDM_CTXSTORE_TIMEOUT"] = "1"

from .safety import HaloCheck, preflight, preflight_generation
from .monitor import SystemMonitor, get_monitor
from .pain import PainCalculator, get_pain_calculator, get_current_pain
from .memory import get_gpu_memory_status, clear_gpu_cache, wait_for_memory_drain
from .types import SystemMetrics, PainProfile, HaloResult, HealthStatus, MemoryStatus

__version__ = "1.0.2"

__all__ = [
    "preflight",
    "preflight_generation",
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

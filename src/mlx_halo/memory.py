"""
GPU memory management for MLX on Apple Silicon.

Handles the critical Metal GPU memory quirks:
- Metal lazily frees memory after model unload
- Overlapping allocations during model swaps can cause kernel panics
- Thermal state affects how long Metal takes to release memory
- PyTorch and MLX cannot share the Metal GPU simultaneously
"""

import gc
import logging
import subprocess
import threading
import time
from typing import Optional

from .types import MemoryStatus

log = logging.getLogger("mlx_halo.memory")

# Serialize all GPU operations to prevent concurrent Metal access corruption
GPU_LOCK = threading.Lock()


def _detect_total_memory_gb() -> float:
    """Auto-detect total unified memory via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024**3)
    except Exception:
        pass
    return 16.0  # safe fallback


def get_gpu_memory_status(
    total_vram_gb: Optional[float] = None,
    buffer_gb: float = 6.0,
) -> MemoryStatus:
    """
    Current GPU memory usage via MLX native APIs.

    Args:
        total_vram_gb: Total unified memory in GB. Auto-detected if None.
        buffer_gb: Reserved buffer for OS + other processes.

    Returns:
        MemoryStatus with active, cache, peak, total_used, and available.
    """
    if total_vram_gb is None:
        total_vram_gb = _detect_total_memory_gb()

    try:
        import mlx.core as mx
        active_gb = mx.get_active_memory() / (1024**3)
        cache_gb = mx.get_cache_memory() / (1024**3)
        peak_gb = mx.get_peak_memory() / (1024**3)
        total_used_gb = active_gb + cache_gb
        available_gb = total_vram_gb - total_used_gb - buffer_gb

        return MemoryStatus(
            active_gb=active_gb,
            cache_gb=cache_gb,
            peak_gb=peak_gb,
            total_used_gb=total_used_gb,
            available_gb=available_gb,
        )
    except ImportError:
        raise ImportError("MLX is required for GPU memory checks. Install with: pip install mlx")
    except Exception as e:
        log.warning(f"Memory check failed: {e}")
        return MemoryStatus(
            active_gb=0.0, cache_gb=0.0, peak_gb=0.0,
            total_used_gb=0.0, available_gb=0.0,
        )


def clear_gpu_cache() -> None:
    """
    Serialized GPU cache clear.

    Always use this instead of mx.clear_cache() directly —
    the lock prevents concurrent Metal operations from corrupting the heap.
    """
    import mlx.core as mx
    acquired = GPU_LOCK.acquire(timeout=10.0)
    try:
        gc.collect()
        mx.clear_cache()
    finally:
        if acquired:
            GPU_LOCK.release()


def wait_for_memory_drain(
    baseline_gb: float = 2.0,
    max_wait: float = 30.0,
    poll_interval: float = 0.5,
    settling_time: Optional[float] = None,
    thermal_pain: float = 0.0,
    total_vram_gb: Optional[float] = None,
    buffer_gb: float = 6.0,
    verbose: bool = False,
) -> bool:
    """
    Poll Metal GPU memory until it drops to baseline and stabilizes.

    CRITICAL: Metal lazily frees memory after model unload. Loading the next
    model while the GPU is still draining the previous one causes overlapping
    allocations that corrupt the Metal heap -> kernel panic.

    After memory drops below baseline, an additional settling period ensures
    Metal is truly finished. Settling time adapts to thermal state — hot
    silicon is slower to release memory.

    Args:
        baseline_gb: Target memory level to consider "drained".
        max_wait: Maximum seconds to wait before giving up.
        poll_interval: Seconds between memory checks.
        settling_time: Seconds to hold below baseline. Auto-calculated from
                       thermal_pain if None.
        thermal_pain: Thermal stress 0.0-1.0. Higher = longer settling.
        total_vram_gb: Total unified memory (auto-detected if None).
        buffer_gb: OS/system memory buffer.
        verbose: Print progress to stdout.

    Returns:
        True if memory drained and settled, False if timeout.
    """
    # Thermal-adaptive settling time
    if settling_time is None:
        if thermal_pain > 0.8:      # Thermal crisis (>95°C)
            settling_time = 10.0
        elif thermal_pain > 0.5:    # High heat (>80°C)
            settling_time = 8.0
        else:
            settling_time = 5.0     # Normal

    start = time.time()
    baseline_reached_at = None

    if verbose:
        print(f"[mlx-halo] Waiting for memory drain (target <{baseline_gb}GB, "
              f"settling {settling_time}s)...", end="", flush=True)

    while True:
        elapsed = time.time() - start
        if elapsed > max_wait:
            status = get_gpu_memory_status(total_vram_gb, buffer_gb)
            if verbose:
                print(f"\n[mlx-halo] Drain timeout after {elapsed:.1f}s. "
                      f"Memory: {status.total_used_gb:.2f}GB")
            return False

        status = get_gpu_memory_status(total_vram_gb, buffer_gb)

        if status.total_used_gb < baseline_gb:
            if baseline_reached_at is None:
                baseline_reached_at = time.time()
                if verbose:
                    print(f"\n[mlx-halo] Baseline reached ({status.total_used_gb:.2f}GB), "
                          f"settling...", end="", flush=True)
            else:
                settling_elapsed = time.time() - baseline_reached_at
                if settling_elapsed >= settling_time:
                    if verbose:
                        print(f"\n[mlx-halo] Drained to {status.total_used_gb:.2f}GB "
                              f"in {elapsed:.1f}s (+{settling_elapsed:.1f}s settling)")
                    return True
        else:
            if baseline_reached_at is not None:
                if verbose:
                    print(f"\n[mlx-halo] Memory rose during settling "
                          f"({status.total_used_gb:.2f}GB), restarting...",
                          end="", flush=True)
                baseline_reached_at = None

        time.sleep(poll_interval)

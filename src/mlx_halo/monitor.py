"""
System monitoring for Apple Silicon — CPU, RAM, GPU VRAM, thermals.

Provides the raw metrics that feed the pain calculator and safety checks.
macOS-specific: uses powermetrics for thermal data, MLX for VRAM.
"""

import logging
import subprocess
from typing import Dict, Optional

import psutil

from .types import HealthStatus, SystemMetrics

log = logging.getLogger("mlx_halo.monitor")


class SystemMonitor:
    """Hardware resource monitor for macOS / Apple Silicon."""

    def __init__(self):
        self._last_metrics: Optional[SystemMetrics] = None
        self._thermal_available = self._check_thermal_access()
        if not self._thermal_available:
            log.info(
                "Thermal monitoring requires sudo. Temperature will be estimated "
                "from CPU load. For accurate temps, configure passwordless sudo "
                "for powermetrics."
            )

    def _check_thermal_access(self) -> bool:
        """Check if powermetrics is accessible without password."""
        try:
            result = subprocess.run(
                ["sudo", "-n", "powermetrics", "--samplers", "smc", "-n", "1"],
                capture_output=True, timeout=1, text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return False

    def get_cpu_usage(self) -> float:
        """CPU usage percentage (all cores average)."""
        return psutil.cpu_percent(interval=0.1)

    def get_cpu_temperature(self) -> Optional[float]:
        """
        CPU temperature in Celsius.

        Uses powermetrics if available (requires sudo), otherwise estimates
        from CPU load. M-series chips idle ~40°C, peak ~85°C under load.
        """
        if self._thermal_available:
            try:
                result = subprocess.run(
                    ["sudo", "-n", "powermetrics", "--samplers", "smc", "-n", "1"],
                    capture_output=True, timeout=2, text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "CPU die temperature" in line or "CPU temperature" in line:
                            parts = line.split(":")
                            if len(parts) > 1:
                                temp_str = parts[1].strip().split()[0]
                                return float(temp_str)
            except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
                log.debug(f"powermetrics failed: {e}")

        # Fallback: estimate from CPU usage
        cpu_pct = self.get_cpu_usage()
        estimated = 40.0 + (cpu_pct / 100.0) * 45.0
        log.debug(f"Estimating temp from CPU {cpu_pct:.0f}% -> {estimated:.0f}°C")
        return estimated

    def get_ram_usage(self) -> Dict[str, float]:
        """RAM usage: used_gb, total_gb, percent."""
        mem = psutil.virtual_memory()
        return {
            "used_gb": mem.used / (1024**3),
            "total_gb": mem.total / (1024**3),
            "percent": mem.percent,
        }

    def get_gpu_vram(self) -> Optional[float]:
        """Active GPU VRAM in GB via MLX. Returns None if MLX unavailable."""
        try:
            import mlx.core as mx
            return mx.get_active_memory() / (1024**3)
        except ImportError:
            log.debug("MLX not available, GPU VRAM monitoring disabled")
            return None
        except Exception as e:
            log.debug(f"Failed to read GPU VRAM: {e}")
            return None

    def is_thermal_throttling(self) -> bool:
        """True if CPU is likely thermally throttling (>95°C)."""
        temp = self.get_cpu_temperature()
        return bool(temp and temp > 95.0)

    def get_health_status(self, pain_score: float) -> HealthStatus:
        """Map pain score to health zone."""
        if pain_score < 0.3:
            return HealthStatus.GREEN
        elif pain_score < 0.7:
            return HealthStatus.YELLOW
        return HealthStatus.RED

    def get_current_metrics(self, pain_score: float = 0.0) -> SystemMetrics:
        """Complete snapshot of system state."""
        ram = self.get_ram_usage()
        temp = self.get_cpu_temperature()
        vram = self.get_gpu_vram()

        metrics = SystemMetrics(
            cpu_percent=self.get_cpu_usage(),
            cpu_temp_celsius=temp,
            ram_used_gb=ram["used_gb"],
            ram_total_gb=ram["total_gb"],
            ram_percent=ram["percent"],
            gpu_vram_gb=vram,
            is_throttling=self.is_thermal_throttling(),
            health_status=self.get_health_status(pain_score),
        )
        self._last_metrics = metrics
        return metrics


_monitor: Optional[SystemMonitor] = None


def get_monitor() -> SystemMonitor:
    """Get or create the singleton SystemMonitor."""
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor

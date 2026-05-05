"""
Resource pain calculator — quantifies system stress as a single 0.0-1.0 score.

Combines thermal, RAM, and VRAM pressure into a weighted metric that drives
model loading decisions. Higher pain = less safe to load large models.

Pain thresholds:
  0.0-0.3  GREEN   Comfortable — safe for large models
  0.3-0.7  YELLOW  Stressed — use medium models, monitor closely
  0.7-1.0  RED     Crisis — refuse local loads, use cloud/API models
"""

import logging
from typing import Optional

from .monitor import get_monitor
from .types import PainProfile

log = logging.getLogger("mlx_halo.pain")


class PainCalculator:
    """
    Calculates resource pressure from hardware metrics.

    All thresholds are configurable via constructor kwargs. Defaults are
    tuned for Apple Silicon MacBooks and Mac Studios (M1-M4).
    """

    def __init__(
        self,
        thermal_comfort: float = 70.0,
        thermal_max: float = 100.0,
        thermal_crisis: float = 95.0,
        ram_comfort: float = 70.0,
        ram_max: float = 100.0,
        ram_crisis: float = 85.0,
        vram_comfort_gb: float = 12.0,
        vram_max_gb: float = 20.0,
        vram_crisis_gb: float = 18.0,
        thermal_weight: float = 0.40,
        ram_weight: float = 0.30,
        vram_weight: float = 0.30,
    ):
        self.thermal_comfort = thermal_comfort
        self.thermal_max = thermal_max
        self.thermal_crisis = thermal_crisis
        self.ram_comfort = ram_comfort
        self.ram_max = ram_max
        self.ram_crisis = ram_crisis
        self.vram_comfort_gb = vram_comfort_gb
        self.vram_max_gb = vram_max_gb
        self.vram_crisis_gb = vram_crisis_gb
        self.thermal_weight = thermal_weight
        self.ram_weight = ram_weight
        self.vram_weight = vram_weight
        self.monitor = get_monitor()

    def _linear_pain(self, value: float, comfort: float, maximum: float) -> float:
        """Linear pain curve: comfort=0.0, maximum=1.0, clamped."""
        return max(0.0, min(1.0, (value - comfort) / (maximum - comfort)))

    def calculate_thermal_pain(self, temp_celsius: Optional[float]) -> float:
        if temp_celsius is None:
            return 0.3  # Unknown, assume moderate
        return self._linear_pain(temp_celsius, self.thermal_comfort, self.thermal_max)

    def calculate_ram_pain(self, ram_percent: float) -> float:
        return self._linear_pain(ram_percent, self.ram_comfort, self.ram_max)

    def calculate_vram_pain(self, vram_gb: Optional[float]) -> float:
        if vram_gb is None:
            return 0.0  # No VRAM data, assume comfortable
        return self._linear_pain(vram_gb, self.vram_comfort_gb, self.vram_max_gb)

    def calculate(self) -> PainProfile:
        """
        Calculate current pain profile from live system metrics.

        Returns:
            PainProfile with overall score, component scores, and crisis flags.
        """
        metrics = self.monitor.get_current_metrics()

        thermal = self.calculate_thermal_pain(metrics.cpu_temp_celsius)
        ram = self.calculate_ram_pain(metrics.ram_percent)
        vram = self.calculate_vram_pain(metrics.gpu_vram_gb)

        pain_score = (
            thermal * self.thermal_weight
            + ram * self.ram_weight
            + vram * self.vram_weight
        )

        return PainProfile(
            pain_score=round(pain_score, 3),
            thermal_pain=round(thermal, 3),
            ram_pain=round(ram, 3),
            vram_pain=round(vram, 3),
            thermal_crisis=bool(
                metrics.cpu_temp_celsius
                and metrics.cpu_temp_celsius > self.thermal_crisis
            ),
            ram_crisis=metrics.ram_percent > self.ram_crisis,
            vram_crisis=bool(
                metrics.gpu_vram_gb
                and metrics.gpu_vram_gb > self.vram_crisis_gb
            ),
        )


_calculator: Optional[PainCalculator] = None


def get_pain_calculator(**kwargs) -> PainCalculator:
    """Get or create the singleton PainCalculator."""
    global _calculator
    if _calculator is None:
        _calculator = PainCalculator(**kwargs)
    return _calculator


def get_current_pain(**kwargs) -> PainProfile:
    """Convenience: calculate current pain in one call."""
    return get_pain_calculator(**kwargs).calculate()

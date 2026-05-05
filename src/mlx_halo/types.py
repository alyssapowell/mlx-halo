"""Core data types for mlx-halo."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HealthStatus(Enum):
    """System health zones based on pain score."""
    GREEN = "GREEN"    # 0.0-0.3: Comfortable, full capacity
    YELLOW = "YELLOW"  # 0.3-0.7: Moderate stress, adaptive
    RED = "RED"        # 0.7-1.0: Crisis, emergency measures


@dataclass
class SystemMetrics:
    """Snapshot of current hardware state."""
    cpu_percent: float
    cpu_temp_celsius: Optional[float]
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_vram_gb: Optional[float]
    is_throttling: bool
    health_status: HealthStatus


@dataclass
class PainProfile:
    """Resource pressure state."""
    pain_score: float           # Overall (0.0-1.0)
    thermal_pain: float         # Thermal component (0.0-1.0)
    ram_pain: float             # RAM component (0.0-1.0)
    vram_pain: float            # VRAM component (0.0-1.0)
    thermal_crisis: bool        # Temp > 95°C
    ram_crisis: bool            # RAM > 85%
    vram_crisis: bool           # VRAM near limit


@dataclass
class MemoryStatus:
    """GPU memory snapshot."""
    active_gb: float
    cache_gb: float
    peak_gb: float
    total_used_gb: float
    available_gb: float


@dataclass
class HaloResult:
    """Result of a pre-flight safety check."""
    safe: bool
    pain_score: float
    memory: MemoryStatus
    checks_passed: list[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    failure_check: Optional[str] = None

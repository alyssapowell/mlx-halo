"""Tests for mlx-halo data types."""

from mlx_halo.types import (
    HealthStatus, SystemMetrics, PainProfile, MemoryStatus, HaloResult,
)


def test_health_status_values():
    assert HealthStatus.GREEN.value == "GREEN"
    assert HealthStatus.YELLOW.value == "YELLOW"
    assert HealthStatus.RED.value == "RED"


def test_pain_profile_creation():
    p = PainProfile(
        pain_score=0.45,
        thermal_pain=0.3,
        ram_pain=0.5,
        vram_pain=0.6,
        thermal_crisis=False,
        ram_crisis=False,
        vram_crisis=False,
    )
    assert p.pain_score == 0.45
    assert not p.thermal_crisis


def test_memory_status_creation():
    m = MemoryStatus(
        active_gb=2.0, cache_gb=1.0, peak_gb=8.0,
        total_used_gb=3.0, available_gb=15.0,
    )
    assert m.total_used_gb == 3.0
    assert m.available_gb == 15.0


def test_halo_result_safe():
    m = MemoryStatus(0, 0, 0, 0, 20)
    r = HaloResult(safe=True, pain_score=0.2, memory=m, checks_passed=["conflict", "vram_drain"])
    assert r.safe
    assert r.failure_reason is None
    assert len(r.checks_passed) == 2


def test_halo_result_failed():
    m = MemoryStatus(0, 0, 0, 0, 0)
    r = HaloResult(
        safe=False, pain_score=0.8, memory=m,
        failure_reason="Too hot", failure_check="pain",
    )
    assert not r.safe
    assert r.failure_check == "pain"

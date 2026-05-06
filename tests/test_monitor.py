"""Tests for system monitor."""

from mlx_halo.monitor import SystemMonitor, get_monitor
from mlx_halo.types import HealthStatus


def test_singleton():
    m1 = get_monitor()
    m2 = get_monitor()
    assert m1 is m2


def test_cpu_usage_range():
    m = SystemMonitor()
    usage = m.get_cpu_usage()
    assert 0.0 <= usage <= 100.0


def test_cpu_temperature_returns_float_or_none():
    m = SystemMonitor()
    temp = m.get_cpu_temperature()
    assert temp is None or (20.0 <= temp <= 120.0)


def test_ram_usage_keys():
    m = SystemMonitor()
    ram = m.get_ram_usage()
    assert "used_gb" in ram
    assert "total_gb" in ram
    assert "percent" in ram
    assert ram["total_gb"] > 0
    assert 0 <= ram["percent"] <= 100


def test_gpu_vram_returns_float_or_none():
    m = SystemMonitor()
    vram = m.get_gpu_vram()
    # None if MLX not installed, float otherwise
    assert vram is None or vram >= 0.0


def test_health_status_mapping():
    m = SystemMonitor()
    assert m.get_health_status(0.1) == HealthStatus.GREEN
    assert m.get_health_status(0.5) == HealthStatus.YELLOW
    assert m.get_health_status(0.9) == HealthStatus.RED


def test_get_current_metrics():
    m = SystemMonitor()
    metrics = m.get_current_metrics(pain_score=0.0)
    assert metrics.cpu_percent >= 0
    assert metrics.ram_total_gb > 0
    assert metrics.health_status == HealthStatus.GREEN

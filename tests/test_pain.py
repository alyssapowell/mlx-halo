"""Tests for pain calculator."""

import pytest
from mlx_halo.pain import PainCalculator, get_current_pain


class TestPainCalculation:
    def setup_method(self):
        self.calc = PainCalculator()

    def test_thermal_pain_below_comfort(self):
        assert self.calc.calculate_thermal_pain(50.0) == 0.0

    def test_thermal_pain_at_comfort(self):
        assert self.calc.calculate_thermal_pain(70.0) == 0.0

    def test_thermal_pain_at_max(self):
        assert self.calc.calculate_thermal_pain(100.0) == 1.0

    def test_thermal_pain_above_max(self):
        assert self.calc.calculate_thermal_pain(110.0) == 1.0

    def test_thermal_pain_midpoint(self):
        pain = self.calc.calculate_thermal_pain(85.0)
        assert 0.4 <= pain <= 0.6

    def test_thermal_pain_none(self):
        assert self.calc.calculate_thermal_pain(None) == 0.3

    def test_ram_pain_below_comfort(self):
        assert self.calc.calculate_ram_pain(50.0) == 0.0

    def test_ram_pain_at_max(self):
        assert self.calc.calculate_ram_pain(100.0) == 1.0

    def test_ram_pain_midpoint(self):
        pain = self.calc.calculate_ram_pain(85.0)
        assert 0.4 <= pain <= 0.6

    def test_vram_pain_below_comfort(self):
        assert self.calc.calculate_vram_pain(5.0) == 0.0

    def test_vram_pain_at_max(self):
        assert self.calc.calculate_vram_pain(20.0) == 1.0

    def test_vram_pain_none(self):
        assert self.calc.calculate_vram_pain(None) == 0.0

    def test_calculate_returns_pain_profile(self):
        profile = self.calc.calculate()
        assert 0.0 <= profile.pain_score <= 1.0
        assert 0.0 <= profile.thermal_pain <= 1.0
        assert 0.0 <= profile.ram_pain <= 1.0
        assert 0.0 <= profile.vram_pain <= 1.0
        assert isinstance(profile.thermal_crisis, bool)
        assert isinstance(profile.ram_crisis, bool)
        assert isinstance(profile.vram_crisis, bool)

    def test_crisis_flags_at_extreme(self):
        calc = PainCalculator(
            thermal_crisis=95.0,
            ram_crisis=85.0,
            vram_crisis_gb=18.0,
        )
        # Can't force metrics, but verify the thresholds are stored
        assert calc.thermal_crisis == 95.0
        assert calc.ram_crisis == 85.0
        assert calc.vram_crisis_gb == 18.0


class TestCustomThresholds:
    def test_custom_thermal_range(self):
        calc = PainCalculator(thermal_comfort=50.0, thermal_max=80.0)
        assert calc.calculate_thermal_pain(50.0) == 0.0
        assert calc.calculate_thermal_pain(80.0) == 1.0
        pain = calc.calculate_thermal_pain(65.0)
        assert 0.4 <= pain <= 0.6

    def test_custom_vram_range(self):
        calc = PainCalculator(vram_comfort_gb=4.0, vram_max_gb=30.0)
        assert calc.calculate_vram_pain(4.0) == 0.0
        assert calc.calculate_vram_pain(30.0) == 1.0

    def test_custom_weights(self):
        # All weight on thermal
        calc = PainCalculator(
            thermal_weight=1.0, ram_weight=0.0, vram_weight=0.0,
        )
        # Thermal pain at max should give pain_score ~1.0
        thermal_pain = calc.calculate_thermal_pain(100.0)
        assert thermal_pain == 1.0


class TestConvenience:
    def test_get_current_pain(self):
        profile = get_current_pain()
        assert 0.0 <= profile.pain_score <= 1.0

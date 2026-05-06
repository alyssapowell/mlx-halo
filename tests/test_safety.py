"""Tests for Halo safety checks."""

import pytest
from mlx_halo.safety import HaloCheck, preflight
from mlx_halo.types import HaloResult

try:
    import mlx.core  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (Apple Silicon only)")


@requires_mlx
class TestHaloCheck:
    def test_conflict_check_blocks(self):
        """Conflict check should raise MemoryError when conflict detected."""
        halo = HaloCheck(
            conflict_check=lambda: True,  # Always conflicting
            verbose=False,
        )
        with pytest.raises(MemoryError, match="Conflicting framework"):
            halo.check_all(8.0)

    def test_conflict_check_passes_when_clear(self):
        """Conflict check should pass when no conflict."""
        halo = HaloCheck(
            conflict_check=lambda: False,
            verbose=False,
        )
        # May still fail on other checks (VRAM, pain, headroom)
        # depending on system state — that's fine, we're testing
        # that the conflict check itself passes
        try:
            result = halo.check_all(0.1)  # Tiny model to minimize other failures
            assert "conflict" in result.checks_passed
        except MemoryError as e:
            # Failed on a later check, not conflict
            assert "Conflicting" not in str(e)

    def test_zombie_check_blocks(self):
        """Zombie check should raise MemoryError when zombies detected."""
        halo = HaloCheck(
            zombie_check=lambda: True,  # Always zombie
            verbose=False,
        )
        with pytest.raises(MemoryError, match="Zombie"):
            halo.check_all(0.1)

    def test_zombie_check_passes_when_clear(self):
        """Zombie check passes when no zombies."""
        halo = HaloCheck(
            zombie_check=lambda: False,
            verbose=False,
        )
        try:
            result = halo.check_all(0.1)
            assert "zombie" in result.checks_passed
        except MemoryError as e:
            assert "Zombie" not in str(e)

    def test_pain_threshold_configurable(self):
        """Pain threshold should be configurable."""
        halo = HaloCheck(pain_threshold=0.99, verbose=False)
        # Very high threshold — pain check should almost always pass
        try:
            result = halo.check_all(0.1)
            assert "pain" in result.checks_passed
        except MemoryError as e:
            # Only headroom or VRAM drain could fail
            assert "Pain" not in str(e)

    def test_headroom_check_blocks_huge_model(self):
        """Requesting more VRAM than exists should fail."""
        halo = HaloCheck(verbose=False)
        with pytest.raises(MemoryError, match="Need.*have"):
            halo.check_all(999.0)  # 999GB model

    def test_result_structure_on_success(self):
        """Successful check returns proper HaloResult."""
        halo = HaloCheck(
            pain_threshold=0.99,
            verbose=False,
        )
        try:
            result = halo.check_all(0.01)
            assert isinstance(result, HaloResult)
            assert result.safe is True
            assert result.pain_score >= 0.0
            assert result.memory is not None
            assert len(result.checks_passed) == 5
            assert result.failure_reason is None
        except MemoryError:
            pytest.skip("System state prevented all checks from passing")

    def test_all_five_checks_present(self):
        """All 5 safety checks should be in the passed list on success."""
        halo = HaloCheck(pain_threshold=0.99, verbose=False)
        try:
            result = halo.check_all(0.01)
            expected = {"conflict", "vram_drain", "zombie", "pain", "headroom"}
            assert set(result.checks_passed) == expected
        except MemoryError:
            pytest.skip("System state prevented all checks from passing")


@requires_mlx
class TestPreflight:
    def test_preflight_convenience(self):
        """preflight() should work as a one-call check."""
        try:
            result = preflight(0.01, pain_threshold=0.99, verbose=False)
            assert result.safe
        except MemoryError:
            pytest.skip("System state prevented preflight from passing")

    def test_preflight_raises_on_impossible(self):
        """preflight() should raise MemoryError for impossible requests."""
        with pytest.raises(MemoryError):
            preflight(999.0, verbose=False)

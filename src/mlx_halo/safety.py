"""
Halo pre-flight safety checks — prevents kernel panics on Apple Silicon.

Named after F1's Halo cockpit protection device: invisible in normal
operation, life-saving when things go wrong.

Five sequential checks before any MLX model load:
1. Conflicting framework unloaded (PyTorch/MLX Metal conflict)
2. VRAM drained from previous model (Metal lazy deallocation)
3. No zombie model references (consistency)
4. Pain score within safe bounds (resource pressure)
5. Sufficient VRAM headroom (with safety margin)
"""

import logging
from typing import Callable, Optional

from .memory import clear_gpu_cache, get_gpu_memory_status, wait_for_memory_drain
from .pain import get_current_pain
from .types import HaloResult

log = logging.getLogger("mlx_halo.safety")


class HaloCheck:
    """
    Pre-flight safety gate for MLX model loading.

    Args:
        total_vram_gb: Total unified memory. Auto-detected if None.
        buffer_gb: Reserved for OS/system. Default 6GB.
        safety_margin_gb: Extra headroom required beyond model size. Default 2GB.
        baseline_vram_gb: VRAM level considered "drained". Default 2GB.
        pain_threshold: Maximum pain score to allow loading. Default 0.65.
        conflict_check: Callable returning True if a conflicting framework
                        (e.g. PyTorch) has models loaded. Default: None (skip).
        zombie_check: Callable returning True if zombie model references exist.
                      Default: None (skip).
        verbose: Print check progress to stdout.
    """

    def __init__(
        self,
        total_vram_gb: Optional[float] = None,
        buffer_gb: float = 6.0,
        safety_margin_gb: float = 2.0,
        baseline_vram_gb: float = 2.0,
        pain_threshold: float = 0.65,
        conflict_check: Optional[Callable[[], bool]] = None,
        zombie_check: Optional[Callable[[], bool]] = None,
        verbose: bool = True,
    ):
        from .memory import _detect_total_memory_gb
        self.total_vram_gb = total_vram_gb or _detect_total_memory_gb()
        self.buffer_gb = buffer_gb
        self.safety_margin_gb = safety_margin_gb
        self.baseline_vram_gb = baseline_vram_gb
        self.pain_threshold = pain_threshold
        self.conflict_check = conflict_check
        self.zombie_check = zombie_check
        self.verbose = verbose

    def _fail(self, check_name: str, reason: str, memory, pain_score: float) -> HaloResult:
        if self.verbose:
            print(f"[HALO BREACH] {check_name}: {reason}")
        return HaloResult(
            safe=False,
            pain_score=pain_score,
            memory=memory,
            failure_reason=reason,
            failure_check=check_name,
        )

    def check_for_generation(self) -> HaloResult:
        """
        Run only the safety gates that are valid for inference on an
        already-loaded model: conflict, zombie, pain.

        ``check_all`` was designed for cold model loads — its drain gate
        forces VRAM down to baseline (which is impossible while the model
        is resident) and its headroom gate requires space for a fresh
        allocation we don't need. Both fail by definition on a warm-path
        generation call, even though the system may be perfectly safe.

        This method runs the gates that *are* applicable in both contexts:
        Metal framework conflict, zombie references, and resource pain.
        VRAM is reported in the result for telemetry but no drain or
        headroom enforcement is applied.

        Use this before each inference call on a loaded model so every
        MLX op stays gated (defense in depth) — the cold-load path keeps
        using ``check_all``.

        Returns:
            HaloResult with safe=True and checks_passed=["conflict",
            "zombie", "pain"] if all applicable gates pass.

        Raises:
            MemoryError: If any applicable gate fails.
        """
        passed = []
        memory = get_gpu_memory_status(self.total_vram_gb, self.buffer_gb)
        pain = get_current_pain()
        pain_score = pain.pain_score

        # Check 1: Conflicting framework unloaded (e.g. PyTorch/Metal still
        # resident). Always relevant — a coexisting Metal user can crash an
        # in-progress MLX kernel just as easily as a fresh load.
        if self.conflict_check is not None:
            if self.conflict_check():
                result = self._fail(
                    "conflict",
                    "Conflicting framework still loaded. "
                    "PyTorch/Metal and MLX/Metal cannot coexist — unload first.",
                    memory, pain_score,
                )
                raise MemoryError(result.failure_reason)
        passed.append("conflict")

        # Check 2: Zombie references. Always relevant — out-of-sync model
        # state can corrupt subsequent ops regardless of fresh-load vs warm.
        if self.zombie_check is not None:
            if self.zombie_check():
                result = self._fail(
                    "zombie",
                    "Zombie model references detected. Clean up before generating.",
                    memory, pain_score,
                )
                raise MemoryError(result.failure_reason)
        passed.append("zombie")

        # Check 3: Pain score within bounds. Always relevant — thermal or
        # memory pressure can stall or panic mid-generation just as it
        # would mid-load.
        if pain_score > self.pain_threshold:
            result = self._fail(
                "pain",
                f"Pain {pain_score:.2f} > {self.pain_threshold}. "
                f"System too stressed. Use API models.",
                memory, pain_score,
            )
            raise MemoryError(result.failure_reason)
        passed.append("pain")

        if self.verbose:
            print(f"[HALO] Pre-generate passed (pain={pain_score:.2f})")

        return HaloResult(
            safe=True,
            pain_score=pain_score,
            memory=memory,
            checks_passed=passed,
        )

    def check_all(self, estimated_model_gb: float) -> HaloResult:
        """
        Run all 5 safety checks in sequence. Fail-fast on first breach.

        Args:
            estimated_model_gb: Expected VRAM usage for the model to be loaded.

        Returns:
            HaloResult with safe=True if all checks pass.

        Raises:
            MemoryError: If any check fails (prevents kernel panic).
        """
        passed = []
        memory = get_gpu_memory_status(self.total_vram_gb, self.buffer_gb)
        pain = get_current_pain()
        pain_score = pain.pain_score

        # Check 1: Conflicting framework unloaded
        if self.conflict_check is not None:
            if self.conflict_check():
                result = self._fail(
                    "conflict",
                    "Conflicting framework still loaded. "
                    "PyTorch/Metal and MLX/Metal cannot coexist — unload first.",
                    memory, pain_score,
                )
                raise MemoryError(result.failure_reason)
        passed.append("conflict")

        # Check 2: VRAM drained from previous model
        memory = get_gpu_memory_status(self.total_vram_gb, self.buffer_gb)
        if memory.total_used_gb > self.baseline_vram_gb:
            if self.verbose:
                print(f"[HALO] VRAM residual ({memory.total_used_gb:.2f}GB) — force-draining...")
            clear_gpu_cache()
            drained = wait_for_memory_drain(
                baseline_gb=self.baseline_vram_gb,
                thermal_pain=pain.thermal_pain,
                total_vram_gb=self.total_vram_gb,
                buffer_gb=self.buffer_gb,
                verbose=self.verbose,
            )
            memory = get_gpu_memory_status(self.total_vram_gb, self.buffer_gb)
            if not drained or memory.total_used_gb > self.baseline_vram_gb:
                result = self._fail(
                    "vram_drain",
                    f"VRAM not drained ({memory.total_used_gb:.2f}GB). "
                    f"Force-drain failed. Use API models.",
                    memory, pain_score,
                )
                raise MemoryError(result.failure_reason)
            if self.verbose:
                print(f"[HALO] Force-drain succeeded ({memory.total_used_gb:.2f}GB)")
        passed.append("vram_drain")

        # Check 3: No zombie references
        if self.zombie_check is not None:
            if self.zombie_check():
                result = self._fail(
                    "zombie",
                    "Zombie model references detected. Clean up before loading.",
                    memory, pain_score,
                )
                raise MemoryError(result.failure_reason)
        passed.append("zombie")

        # Check 4: Pain score within bounds
        if pain_score > self.pain_threshold:
            result = self._fail(
                "pain",
                f"Pain {pain_score:.2f} > {self.pain_threshold}. "
                f"System too stressed. Use API models.",
                memory, pain_score,
            )
            raise MemoryError(result.failure_reason)
        passed.append("pain")

        # Check 5: Sufficient headroom
        required = estimated_model_gb + self.safety_margin_gb
        if memory.available_gb < required:
            result = self._fail(
                "headroom",
                f"Need {required:.2f}GB, have {memory.available_gb:.2f}GB. "
                f"Use smaller model or API.",
                memory, pain_score,
            )
            raise MemoryError(result.failure_reason)
        passed.append("headroom")

        if self.verbose:
            print(f"[HALO] Pre-flight passed (pain={pain_score:.2f})")

        return HaloResult(
            safe=True,
            pain_score=pain_score,
            memory=memory,
            checks_passed=passed,
        )


def preflight(estimated_model_gb: float, **kwargs) -> HaloResult:
    """
    One-call pre-flight check. Raises MemoryError if unsafe.

    Args:
        estimated_model_gb: Expected model VRAM in GB.
        **kwargs: Passed to HaloCheck constructor.

    Returns:
        HaloResult if safe.

    Raises:
        MemoryError: If any safety check fails.

    Example::

        from mlx_halo import preflight

        result = preflight(8.0)  # Check before loading an 8GB model
        print(f"Safe to load, pain={result.pain_score}")
    """
    return HaloCheck(**kwargs).check_all(estimated_model_gb)


def preflight_generation(**kwargs) -> HaloResult:
    """
    One-call pre-flight for inference on an already-loaded model.

    Runs only the safety gates valid in both cold-load and warm-generate
    contexts (conflict, zombie, pain). Skips the drain and headroom gates,
    which assume a fresh allocation.

    Use before every inference call so warm paths stay gated by halo
    without false-failing on drain (impossible while model is resident)
    or headroom (no fresh allocation needed).

    Args:
        **kwargs: Passed to HaloCheck constructor.

    Returns:
        HaloResult if safe.

    Raises:
        MemoryError: If any applicable gate fails.

    Example::

        from mlx_halo import preflight_generation

        # Before each inference call on a loaded model
        preflight_generation(conflict_check=lambda: pytorch_loaded())
    """
    return HaloCheck(**kwargs).check_for_generation()

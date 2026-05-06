# mlx-halo

Pre-flight safety checks for MLX models on Apple Silicon. Prevents kernel panics.

Named after F1's Halo cockpit protection device: invisible in normal operation, life-saving when things go wrong.

## The Problem

Loading MLX models on Apple Silicon can cause kernel panics. The failure modes are undocumented:

- **PyTorch and MLX cannot share Metal GPU simultaneously.** Loading a sentence-transformers model (PyTorch/Metal) while an MLX model is active corrupts the Metal heap.
- **Metal lazily frees memory.** After unloading a model, GPU memory takes 5-10 seconds to actually release. Loading the next model during this window causes overlapping allocations.
- **Thermal state affects drain time.** Hot silicon is slower to release memory. A settling time that works at 60°C fails at 90°C.
- **No error — just a kernel panic.** There's no exception, no warning. The machine reboots.

mlx-halo catches these conditions before they crash your system.

## Install

```bash
pip install mlx-halo
```

MLX is an optional dependency (for GPU memory checks):

```bash
pip install "mlx-halo[mlx]"
```

## Quick Start

```python
from mlx_halo import preflight

# Before loading any MLX model:
result = preflight(model_size_gb=8.0)
# Returns HaloResult if safe
# Raises MemoryError if unsafe
```

## What It Checks

Five sequential safety gates, fail-fast:

| Check | What | Why |
|-------|------|-----|
| **Conflict** | Is a conflicting framework (e.g. PyTorch) holding the Metal GPU? | PyTorch/Metal and MLX/Metal corrupt each other's heap |
| **VRAM Drain** | Has the previous model's memory fully released? | Metal lazy deallocation — overlapping allocations panic |
| **Zombie** | Are there stale model references keeping memory pinned? | Prevents ghost allocations that block new loads |
| **Pain** | Is the system under thermal/memory pressure? | High pressure + model load = panic territory |
| **Headroom** | Is there enough free VRAM for this model + safety margin? | Loading into insufficient space corrupts the allocator |

## Usage

### Basic — One Call

```python
from mlx_halo import preflight

try:
    result = preflight(model_size_gb=8.0)
    print(f"Safe to load (pain={result.pain_score:.2f})")
    # proceed with mlx_lm.load(...)
except MemoryError as e:
    print(f"Unsafe: {e}")
    # fall back to API model
```

### Configurable

```python
from mlx_halo import HaloCheck

halo = HaloCheck(
    total_vram_gb=32,              # Auto-detected if omitted
    safety_margin_gb=3.0,          # Extra headroom beyond model size
    pain_threshold=0.7,            # Max acceptable pain score
    conflict_check=lambda: pytorch_model is not None,
    zombie_check=lambda: stale_ref is not None,
)

result = halo.check_all(estimated_model_gb=18.0)
```

### Pain Score

The pain calculator quantifies system stress as a single 0.0-1.0 score:

```python
from mlx_halo import get_current_pain

pain = get_current_pain()
print(f"Pain: {pain.pain_score:.2f}")
print(f"  Thermal: {pain.thermal_pain:.2f}  Crisis: {pain.thermal_crisis}")
print(f"  RAM:     {pain.ram_pain:.2f}  Crisis: {pain.ram_crisis}")
print(f"  VRAM:    {pain.vram_pain:.2f}  Crisis: {pain.vram_crisis}")
```

| Range | Status | Recommendation |
|-------|--------|----------------|
| 0.0-0.3 | GREEN | Safe for large models |
| 0.3-0.7 | YELLOW | Use medium models, monitor closely |
| 0.7-1.0 | RED | Refuse local loads, use API/cloud |

Custom thresholds for your hardware:

```python
from mlx_halo import PainCalculator

calc = PainCalculator(
    thermal_comfort=60.0,    # °C where pain starts (default 70)
    thermal_max=95.0,        # °C at max pain (default 100)
    vram_comfort_gb=8.0,     # GB where VRAM pain starts (default 12)
    vram_max_gb=28.0,        # GB at max VRAM pain (default 20)
    thermal_weight=0.5,      # Weight in overall score (default 0.4)
    ram_weight=0.3,          # (default 0.3)
    vram_weight=0.2,         # (default 0.3)
)
```

### System Monitor

Raw hardware metrics without the pain abstraction:

```python
from mlx_halo import get_monitor

monitor = get_monitor()
print(f"CPU: {monitor.get_cpu_usage():.1f}%")
print(f"Temp: {monitor.get_cpu_temperature():.1f}°C")
print(f"RAM: {monitor.get_ram_usage()['percent']:.1f}%")
print(f"VRAM: {monitor.get_gpu_vram():.2f} GB")
print(f"Throttling: {monitor.is_thermal_throttling()}")
```

### GPU Memory Management

Direct control over Metal GPU memory:

```python
from mlx_halo import get_gpu_memory_status, clear_gpu_cache, wait_for_memory_drain

# Check current state
status = get_gpu_memory_status()
print(f"Active: {status.active_gb:.2f} GB")
print(f"Cache:  {status.cache_gb:.2f} GB")
print(f"Available: {status.available_gb:.2f} GB")

# After unloading a model — wait for Metal to actually free the memory
clear_gpu_cache()
drained = wait_for_memory_drain(
    baseline_gb=2.0,       # Target memory level
    settling_time=5.0,     # Seconds to hold below baseline
    verbose=True,
)
```

## Examples

### Model Swap (Unload → Drain → Check → Load)

```python
import mlx.core as mx
from mlx_lm import load
from mlx_halo import preflight, clear_gpu_cache, wait_for_memory_drain

# Unload current model
del model
del tokenizer
clear_gpu_cache()

# Wait for Metal to release memory (thermal-adaptive)
from mlx_halo import get_current_pain
pain = get_current_pain()
wait_for_memory_drain(thermal_pain=pain.thermal_pain, verbose=True)

# Safety check before loading next model
preflight(model_size_gb=8.0)

# Safe to load
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
```

### Adaptive Model Selection

```python
from mlx_halo import get_current_pain, get_gpu_memory_status

def select_model():
    pain = get_current_pain()
    mem = get_gpu_memory_status()

    if pain.pain_score > 0.7 or pain.thermal_crisis:
        return None  # Use API, system too stressed

    if mem.available_gb > 20:
        return "mlx-community/Qwen2.5-32B-Instruct-4bit"  # ~18GB
    elif mem.available_gb > 10:
        return "mlx-community/Qwen2.5-7B-Instruct-4bit"   # ~5GB
    elif mem.available_gb > 5:
        return "mlx-community/Phi-4-mini-instruct-4bit"    # ~3GB
    else:
        return None  # Not enough room
```

### Embedding Model Conflict Guard

```python
from sentence_transformers import SentenceTransformer
from mlx_halo import HaloCheck

# Track whether PyTorch embeddings are loaded
embedder = None

def load_embedder():
    global embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

def unload_embedder():
    global embedder
    del embedder
    embedder = None

# Halo knows to check for the conflict
halo = HaloCheck(
    conflict_check=lambda: embedder is not None,
)

# This will raise MemoryError if embedder is still loaded:
halo.check_all(estimated_model_gb=8.0)
```

### Continuous Monitoring Loop

```python
import time
from mlx_halo import get_current_pain, HealthStatus

while True:
    pain = get_current_pain()
    status = "OK" if pain.pain_score < 0.3 else "WARN" if pain.pain_score < 0.7 else "CRIT"
    print(f"[{status}] pain={pain.pain_score:.2f} "
          f"thermal={pain.thermal_pain:.2f} "
          f"ram={pain.ram_pain:.2f} "
          f"vram={pain.vram_pain:.2f}")

    if pain.thermal_crisis:
        print("  THERMAL CRISIS — unload models immediately")
    time.sleep(10)
```

## Hardware Compatibility

Tested on Apple Silicon M1-M4 (MacBook Air, MacBook Pro, Mac Mini, Mac Studio). The thermal monitoring uses `powermetrics` which requires sudo — without it, temperature is estimated from CPU load (less accurate but functional).

For accurate thermal monitoring, configure passwordless sudo for powermetrics:

```bash
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/powermetrics
```

## License

[Liberation License v1.0](LICENSE.md)

"""
core/gpu_guard.py

Pre-flight GPU-availability gate (added Sept 2026, user request): before the
pipeline starts its two GPU-heavy stages (embeddings in deduplication.py,
LLM calls in summarizer.py), check whether the GPU is already busy with
something else -- a game, a training run, anything outside this pipeline --
and, if so, stand by rather than piling qwen2.5:7b-instruct + nomic-embed-text
on top of it. The RTX 5060 only has 8GB VRAM: loading another ~5GB of models
alongside an already-heavy workload risks an out-of-memory crash in whichever
process loses the allocation race, not just a slower morning for both.

Checks BOTH GPU compute utilization and VRAM usage against the same
threshold (default 50%) and waits if EITHER is over it. Utilization is the
literal "is the GPU in use" signal the threshold was framed around; VRAM
headroom is what actually determines whether loading our models will crash
something. Checking only one would miss half the risk this exists to avoid.

This is a single pre-flight gate, checked once before the pipeline's
GPU-dependent stages start (main.py calls it before ingest_all() even
runs) -- it does not continuously monitor mid-run. The pipeline only ever
has one Ollama call in flight at a time, so once running there's nothing
to interrupt.

Degrade-don't-crash, same philosophy as every other core/ module: if
nvidia-smi isn't installed, times out, or returns something unparseable,
that's logged once and treated as "can't tell -- proceed." A broken
diagnostic tool must never be able to silently break the "runs every
morning with no manual intervention" goal (news_pipeline_v2.md, section 1).
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

NVIDIA_SMI_TIMEOUT = 10  # seconds -- a local hardware query, should be near-instant

DEFAULT_GPU_GUARD_CONFIG = {
    "enabled": True,
    "threshold_pct": 50,
    "poll_interval_seconds": 90,
    "max_wait_seconds": 1200,  # 20 minutes
}


def load_gpu_guard_config(path: Path = CONFIG_PATH) -> dict:
    """Reads the `gpu_guard` section of config.yaml. Missing file, missing
    section, or missing individual keys all degrade to
    DEFAULT_GPU_GUARD_CONFIG (same pattern as delivery.load_config)."""
    if not path.exists():
        return dict(DEFAULT_GPU_GUARD_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.error("Could not read/parse config.yaml for gpu_guard settings (%s) -- using defaults.", exc)
        loaded = {}
    section = loaded.get("gpu_guard") or {}
    return {**DEFAULT_GPU_GUARD_CONFIG, **section}


def get_gpu_stats() -> dict | None:
    """Returns {"utilization_pct": float, "memory_pct": float} for GPU 0,
    or None if nvidia-smi is unavailable/unparseable. None is always
    treated by the caller as "can't tell" -- never as "definitely busy" or
    "definitely free" -- so a missing/broken nvidia-smi can't get the
    pipeline permanently stuck."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT,
        )
    except FileNotFoundError:
        logger.warning("nvidia-smi not found -- cannot check GPU load, proceeding without the pre-flight check.")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out -- cannot check GPU load, proceeding without the pre-flight check.")
        return None

    if result.returncode != 0:
        logger.warning(
            "nvidia-smi exited with an error (%s) -- cannot check GPU load, proceeding without the "
            "pre-flight check.",
            result.stderr.strip() or f"exit code {result.returncode}",
        )
        return None

    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) != 3:
        logger.warning(
            "Could not parse nvidia-smi output (%r) -- proceeding without the pre-flight check.", first_line
        )
        return None

    try:
        utilization_pct, mem_used, mem_total = (float(p) for p in parts)
    except ValueError:
        logger.warning(
            "Could not parse nvidia-smi output (%r) -- proceeding without the pre-flight check.", first_line
        )
        return None

    memory_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0.0
    return {"utilization_pct": utilization_pct, "memory_pct": memory_pct}


def wait_for_gpu_availability(
    threshold_pct: float | None = None,
    poll_interval_seconds: float | None = None,
    max_wait_seconds: float | None = None,
    config_path: Path = CONFIG_PATH,
) -> bool:
    """Blocks until GPU utilization AND memory usage are both at or below
    threshold_pct, or until max_wait_seconds have elapsed.

    Returns True if it's safe to proceed -- the GPU was already free,
    became free during the wait, or the check couldn't be run at all (see
    get_gpu_stats()). Returns False only when the GPU was still genuinely
    busy at the end of the wait budget; the caller (main.py) treats False
    as "skip this run," not as something to retry harder on its own --
    tomorrow's scheduled run is the retry.

    Any argument passed explicitly overrides config.yaml's `gpu_guard`
    section; leaving it None means "use config.yaml, or the built-in
    default if config.yaml doesn't set it."
    """
    cfg = load_gpu_guard_config(config_path)
    if not cfg.get("enabled", True):
        logger.info("GPU guard disabled (gpu_guard.enabled: false in config.yaml) -- skipping pre-flight check.")
        return True

    threshold = threshold_pct if threshold_pct is not None else cfg["threshold_pct"]
    poll_interval = poll_interval_seconds if poll_interval_seconds is not None else cfg["poll_interval_seconds"]
    max_wait = max_wait_seconds if max_wait_seconds is not None else cfg["max_wait_seconds"]

    waited = 0.0
    while True:
        stats = get_gpu_stats()
        if stats is None:
            return True

        busy = stats["utilization_pct"] > threshold or stats["memory_pct"] > threshold
        if not busy:
            if waited > 0:
                logger.info(
                    "GPU now below %.0f%% (utilization=%.0f%%, memory=%.0f%%) after waiting %.0fs -- proceeding.",
                    threshold, stats["utilization_pct"], stats["memory_pct"], waited,
                )
            return True

        if waited >= max_wait:
            logger.warning(
                "GPU still busy after waiting %.0fs (utilization=%.0f%%, memory=%.0f%%, threshold=%.0f%%) -- "
                "giving up for this run rather than contending with whatever's using it. Nothing was "
                "ingested/summarized/delivered; will try again at tomorrow's scheduled run.",
                waited, stats["utilization_pct"], stats["memory_pct"], threshold,
            )
            return False

        logger.info(
            "GPU busy (utilization=%.0f%%, memory=%.0f%%, threshold=%.0f%%) -- standing by "
            "(%.0fs/%.0fs of max wait elapsed so far).",
            stats["utilization_pct"], stats["memory_pct"], threshold, waited, max_wait,
        )
        time.sleep(max(0.0, min(poll_interval, max_wait - waited)))
        waited += poll_interval


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(f"Current GPU stats: {get_gpu_stats()}")
    print("Running wait_for_gpu_availability() with config.yaml settings (Ctrl+C to abort)...")
    ok = wait_for_gpu_availability()
    print(f"Result: {'proceed' if ok else 'GIVE UP -- still busy after max wait'}")

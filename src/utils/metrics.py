from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    tour: list[int]
    cost: float
    history: list[float] = field(default_factory=list)
    history_xlabel: str = "Iteration"
    elapsed_sec: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


def _plots_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "output" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _worker_script() -> Path:
    return Path(__file__).resolve().parent / "plot_worker.py"


def save_performance_plot_or_fallback(
    result: RunResult,
    algorithm: str,
    dataset: str,
) -> Path:
    """Save plot via subprocess so matplotlib never loads in the Qt process."""
    safe_algo = "".join(c if c.isalnum() else "_" for c in algorithm)[:40]
    safe_ds = "".join(c if c.isalnum() else "_" for c in dataset)[:40]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "summary" if len(result.history) < 2 else "curve"
    out_path = _plots_dir() / f"{stamp}_{safe_algo}_{safe_ds}_{suffix}.png"

    payload = {
        **asdict(result),
        "algorithm": algorithm,
        "dataset": dataset,
    }
    payload_path = _plots_dir() / f"_payload_{stamp}.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, str(_worker_script()), str(payload_path), str(out_path)],
            capture_output=True,
            text=True,
            cwd=str(_plots_dir().parents[1]),
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "unknown error").strip()
            raise RuntimeError(f"plot_worker failed: {err}")
        if not out_path.is_file():
            raise RuntimeError(f"plot file missing: {out_path}")
    finally:
        payload_path.unlink(missing_ok=True)

    return out_path.resolve()


# Backwards-compatible alias
def save_performance_plot(
    result: RunResult,
    algorithm: str,
    dataset: str,
) -> Path | None:
    if len(result.history) < 2:
        return None
    return save_performance_plot_or_fallback(result, algorithm, dataset)


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_sec = time.perf_counter() - self._start

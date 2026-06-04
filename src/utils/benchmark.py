from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkEntry:
    run_id: int
    algorithm: str
    dataset: str
    cost: float
    elapsed_sec: float
    params: dict[str, Any] = field(default_factory=dict)

    def params_summary(self) -> str:
        if not self.params:
            return "—"
        return ", ".join(f"{k}={v}" for k, v in self.params.items())

    def algorithm_short(self) -> str:
        for tag in ("BKT", "NN", "HC", "SA", "GA"):
            if tag in self.algorithm:
                return tag
        return self.algorithm[:12]


class SessionBenchmark:
    """Lista rulărilor din sesiunea curentă (golită la pornirea aplicației)."""

    def __init__(self) -> None:
        self.runs: list[BenchmarkEntry] = []
        self._next_id = 0

    def clear(self) -> None:
        self.runs.clear()
        self._next_id = 0

    def add_run(
        self,
        algorithm: str,
        dataset: str,
        cost: float,
        elapsed_sec: float,
        params: dict[str, Any] | None = None,
    ) -> BenchmarkEntry:
        self._next_id += 1
        entry = BenchmarkEntry(
            run_id=self._next_id,
            algorithm=algorithm,
            dataset=dataset,
            cost=float(cost),
            elapsed_sec=float(elapsed_sec),
            params=dict(params or {}),
        )
        self.runs.append(entry)
        return entry


def _worker_script() -> Path:
    return Path(__file__).resolve().parent / "plot_worker.py"


def render_benchmark_chart_png(runs: list[BenchmarkEntry]) -> Path | None:
    """Generează PNG temporar (nu se persistă în output/)."""
    if not runs:
        return None

    payload = {"mode": "benchmark", "runs": [asdict(r) for r in runs]}
    payload_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    out_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    payload_path = Path(payload_file.name)
    out_path = Path(out_file.name)
    payload_file.close()
    out_file.close()

    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]

    try:
        proc = subprocess.run(
            [sys.executable, str(_worker_script()), str(payload_path), str(out_path)],
            capture_output=True,
            text=True,
            cwd=str(root),
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "unknown error").strip()
            raise RuntimeError(err)
        if not out_path.is_file():
            return None
        return out_path
    finally:
        payload_path.unlink(missing_ok=True)

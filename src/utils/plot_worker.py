"""Generează PNG pentru graficul de performanță (fără PySide6)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plots_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "output" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


_ALGO_COLORS = {
    "BKT": "#e63946",
    "NN": "#f4a261",
    "HC": "#2a9d8f",
    "SA": "#2a82da",
    "GA": "#9b5de5",
}


def _algo_tag(algorithm: str) -> str:
    for tag in _ALGO_COLORS:
        if tag in algorithm:
            return tag
    return "OTHER"


def _algo_color(algorithm: str) -> str:
    return _ALGO_COLORS.get(_algo_tag(algorithm), "#666666")


def render_benchmark(payload: dict, out_path: Path) -> Path:
    runs = payload.get("runs") or []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
    if not runs:
        ax.text(
            0.5,
            0.5,
            "Nicio rulare încă",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        seen: set[str] = set()
        for run in runs:
            algo = run.get("algorithm", "")
            tag = _algo_tag(algo)
            t = float(run.get("elapsed_sec", 0))
            c = float(run.get("cost", 0))
            rid = run.get("run_id", "?")
            color = _algo_color(algo)
            label = tag if tag not in seen else None
            seen.add(tag)
            ax.scatter(t, c, c=color, s=90, zorder=3, label=label, edgecolors="white")
            ax.annotate(
                str(rid),
                (t, c),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax.set_xlabel("Timp consumat (s)")
        ax.set_ylabel("Cost final")
        ax.set_title("Benchmark sesiune — cost vs timp")
        ax.grid(True, alpha=0.35)
        if seen:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path


def render(payload: dict, out_path: Path) -> Path:
    if payload.get("mode") == "benchmark":
        return render_benchmark(payload, out_path)

    history = payload.get("history") or []
    algorithm = payload.get("algorithm", "TSP")
    dataset = payload.get("dataset", "")
    cost = float(payload.get("cost", 0))
    elapsed = float(payload.get("elapsed_sec", 0))
    params = payload.get("params") or {}
    xlabel = payload.get("history_xlabel", "Iteration")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(history) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        xs = list(range(1, len(history) + 1))
        ax.plot(xs, history, color="#2a82da", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Best cost")
        ax.set_title(f"Convergență — {algorithm} ({dataset})")
        ax.grid(True, alpha=0.3)
        subtitle = f"Cost final: {cost:.2f} | Timp: {elapsed:.2f}s"
        if params:
            subtitle += " | " + ", ".join(f"{k}={v}" for k, v in params.items())
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor="white")
        ax.bar([algorithm[:24]], [cost], color="#2a82da")
        ax.set_ylabel("Cost")
        ax.set_title(f"{dataset} — {algorithm}")
        ax.text(
            0.02,
            0.98,
            f"Cost: {cost:.2f} | Timp: {elapsed:.2f}s",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: plot_worker.py <payload.json> <output.png>", file=sys.stderr)
        return 1

    payload_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    render(payload, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

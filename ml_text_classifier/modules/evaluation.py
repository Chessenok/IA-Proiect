"""Evaluare: grafice acuratețe, heatmap confuzie, classification_report."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from modules.training import ModelResult


def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": r.name,
                "Acuratețe Test": round(r.accuracy, 4),
                "Timp antrenare (s)": round(r.train_time_sec, 3),
            }
            for r in results
        ]
    ).sort_values("Acuratețe Test", ascending=False)


def plot_accuracy_bar(results: list[ModelResult]) -> plt.Figure:
    df = results_to_dataframe(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("viridis", len(df))
    ax.barh(df["Model"], df["Acuratețe Test"], color=colors)
    ax.set_xlabel("Acuratețe Test")
    ax.set_title("Comparare modele")
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(df["Acuratețe Test"]):
        ax.text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_heatmap(
    y_true: list[str],
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicție")
    ax.set_ylabel("Adevăr")
    ax.set_title(f"Matrice de confuzie — {title}")
    fig.tight_layout()
    return fig


def classification_report_df(
    y_true: list[str],
    y_pred: np.ndarray,
) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True)
    rows = []
    for key, vals in report.items():
        if isinstance(vals, dict):
            rows.append(
                {
                    "Clasă": key,
                    "Precision": round(vals["precision"], 3),
                    "Recall": round(vals["recall"], 3),
                    "F1-Score": round(vals["f1-score"], 3),
                    "Support": int(vals["support"]),
                }
            )
    return pd.DataFrame(rows)


def best_model(results: list[ModelResult]) -> ModelResult | None:
    if not results:
        return None
    return max(results, key=lambda r: r.accuracy)

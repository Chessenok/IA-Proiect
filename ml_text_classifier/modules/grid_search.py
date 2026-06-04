"""
Grid Search pe Pipeline (Tfidf + LinearSVC) — doar pe date Train.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from modules.vectorization import build_tfidf_vectorizer


@dataclass
class GridSearchResult:
    best_params: dict
    best_score: float
    cv_results: pd.DataFrame
    best_pipeline: Pipeline
    elapsed_sec: float


def run_tfidf_svc_grid(
    x_train: list[str],
    y_train: list[str],
    ngram_options: list[tuple[int, int]],
    max_features_options: list[int | None],
    sublinear_tf: bool,
    stop_words: str = "english",
    cv_folds: int = 3,
) -> GridSearchResult:
    base_vec = build_tfidf_vectorizer(
        ngram_choice="Unigrame (1,1)",
        max_features=None,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
    )

    pipeline = Pipeline(
        [
            ("tfidf", base_vec),
            ("clf", LinearSVC(max_iter=2000, dual="auto", random_state=42)),
        ]
    )

    param_grid = {
        "tfidf__ngram_range": ngram_options,
        "tfidf__max_features": max_features_options,
    }

    t0 = time.perf_counter()
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=min(cv_folds, max(2, len(set(y_train)))),
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    elapsed = time.perf_counter() - t0

    cv_df = pd.DataFrame(search.cv_results_)
    return GridSearchResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        cv_results=cv_df,
        best_pipeline=search.best_estimator_,
        elapsed_sec=elapsed,
    )


def plot_grid_heatmap(
    grid_result: GridSearchResult,
    ngram_labels: list[str],
    max_feat_labels: list[str],
) -> plt.Figure | None:
    """Heatmap acuratețe mean_test_score pentru combinații ngram × max_features."""
    rows = []
    for _, row in grid_result.cv_results.iterrows():
        ng = str(row["param_tfidf__ngram_range"])
        mf = row["param_tfidf__max_features"]
        mf_label = "Toate" if mf is None else str(int(mf))
        rows.append(
            {
                "ngram": ng,
                "max_features": mf_label,
                "accuracy": row["mean_test_score"],
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="ngram", columns="max_features", values="accuracy")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Grid Search — acuratețe CV (LinearSVC)")
    fig.tight_layout()
    return fig

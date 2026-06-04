"""
Antrenare modele în Pipeline sklearn — TF-IDF + clasificator.
Datele de test nu intră niciodată în .fit().
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from modules.vectorization import build_tfidf_vectorizer


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    accuracy: float
    train_time_sec: float
    y_pred: np.ndarray
    supports_proba: bool


MODEL_REGISTRY = {
    "Naive Bayes (MultinomialNB)": lambda: MultinomialNB(),
    "Linear SVC": lambda: LinearSVC(max_iter=2000, dual="auto", random_state=42),
    "Regresie Logistică": lambda: LogisticRegression(
        max_iter=2000, random_state=42, n_jobs=-1
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}


def _wrap_svc_for_proba(clf, n_samples: int = 0) -> Any:
    """
    LinearSVC nu are predict_proba.
    Calibrare doar dacă avem suficiente exemple; altfel decision_function la inferență.
    """
    if isinstance(clf, LinearSVC) and n_samples >= 30:
        folds = min(3, max(2, n_samples // 10))
        return CalibratedClassifierCV(clf, cv=folds, method="sigmoid")
    return clf


def build_pipeline(
    model_name: str,
    ngram_choice: str,
    max_features: int | None,
    sublinear_tf: bool,
    stop_words: str = "english",
) -> Pipeline:
    vectorizer = build_tfidf_vectorizer(
        ngram_choice=ngram_choice,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
    )
    factory = MODEL_REGISTRY.get(model_name)
    if factory is None:
        raise ValueError(f"Model necunoscut: {model_name}")

    return Pipeline([("tfidf", vectorizer), ("clf", factory())])


def finalize_pipeline(pipeline: Pipeline, n_train: int) -> Pipeline:
    """Aplică calibrare SVC după ce știm dimensiunea setului de antrenare."""
    clf = pipeline.named_steps["clf"]
    pipeline.named_steps["clf"] = _wrap_svc_for_proba(clf, n_train)
    return pipeline


def supports_predict_proba(pipeline: Pipeline) -> bool:
    clf = pipeline.named_steps["clf"]
    return hasattr(clf, "predict_proba") or hasattr(clf, "decision_function")


def train_selected_models(
    x_train: list[str],
    y_train: list[str],
    x_test: list[str],
    y_test: list[str],
    selected_models: list[str],
    ngram_choice: str,
    max_features: int | None,
    sublinear_tf: bool,
    stop_words: str = "english",
) -> list[ModelResult]:
    results: list[ModelResult] = []

    for name in selected_models:
        if name not in MODEL_REGISTRY:
            continue

        pipe = build_pipeline(
            name, ngram_choice, max_features, sublinear_tf, stop_words
        )
        pipe = finalize_pipeline(pipe, len(x_train))
        t0 = time.perf_counter()
        pipe.fit(x_train, y_train)
        elapsed = time.perf_counter() - t0

        y_pred = pipe.predict(x_test)
        acc = float(np.mean(y_pred == np.array(y_test)))

        results.append(
            ModelResult(
                name=name,
                pipeline=pipe,
                accuracy=acc,
                train_time_sec=elapsed,
                y_pred=y_pred,
                supports_proba=supports_predict_proba(pipe),
            )
        )

    return results


def get_class_labels(pipeline: Pipeline, y_train: list[str]) -> list[str]:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "classes_"):
        return list(clf.classes_)
    return sorted(set(y_train))


def predict_with_proba(
    pipeline: Pipeline,
    text: str,
    class_labels: list[str],
) -> tuple[str, dict[str, float]]:
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(pipeline.named_steps["tfidf"].transform([text]))[0]
        probs = {lbl: float(p) for lbl, p in zip(class_labels, proba)}
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(
            pipeline.named_steps["tfidf"].transform([text])
        )[0]
        if scores.ndim == 0:
            scores = np.array([scores, -scores])
        exp = np.exp(scores - scores.max())
        norm = exp / exp.sum()
        probs = {lbl: float(p) for lbl, p in zip(class_labels, norm)}
    else:
        pred = pipeline.predict([text])[0]
        return pred, {pred: 1.0}

    pred = max(probs, key=probs.get)
    return pred, probs

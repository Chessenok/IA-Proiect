from __future__ import annotations

import csv
import io
import json
import math
import string
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nlp.dictionaries import get_dictionary, tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DATASET = PROJECT_ROOT / "ml_text_classifier" / "sample_data" / "demo_categories.json"


@dataclass
class LoadedDataset:
    texts: list[str]
    labels: list[str]
    label_counts: dict[str, int]
    n_classes: int
    source_name: str


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    accuracy: float
    train_time_sec: float
    y_pred: np.ndarray
    supports_proba: bool


class DataLoadError(Exception):
    pass


MODEL_REGISTRY = {
    "Naive Bayes (MultinomialNB)": lambda: MultinomialNB(),
    "Linear SVC": lambda: LinearSVC(max_iter=2000, dual="auto", random_state=42),
    "Regresie Logistică": lambda: LogisticRegression(
        max_iter=2000, random_state=42
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}


def enrich_text_with_domain_dictionary(text: str, dictionary_name: str | None) -> str:
    if not dictionary_name or dictionary_name == "none":
        return text

    domain = get_dictionary(dictionary_name) or {}
    categories: dict[str, list[str]] = domain.get("categories", {})
    if not categories:
        return text

    tokens = tokenize(text)
    feature_tokens: list[str] = []

    def matches(term: str) -> bool:
        term = term.lower()
        return any(tok == term or tok.startswith(term) or term in tok for tok in tokens)

    for category, terms in categories.items():
        hits = [term for term in terms if matches(term)]
        if hits:
            safe_category = category.lower().replace("-", "_").replace(" ", "_")
            feature_tokens.append(f"dict_category_{safe_category}")
            feature_tokens.extend(
                f"dict_term_{hit.lower().replace('-', '_').replace(' ', '_')}"
                for hit in hits[:8]
            )

    if not feature_tokens:
        return text
    return f"{text} {' '.join(feature_tokens)}"


def _romanian_stopwords() -> set[str]:
    path = PROJECT_ROOT / "data" / "nlp" / "stopwords_ro.json"
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return set(data.get("words", []))


def get_stopwords(language: str) -> set[str]:
    if language == "romanian":
        return {_strip_accents(word) for word in _romanian_stopwords()}
    if language == "english":
        return set(ENGLISH_STOP_WORDS)
    return set()


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def preprocess_steps(raw_text: str, stopword_language: str) -> dict[str, str | list[str] | int]:
    lower = raw_text.lower()
    table = str.maketrans("", "", string.punctuation + "„”«»…")
    clean = lower.translate(table)
    tokens = clean.split()
    stopwords = get_stopwords(stopword_language)
    filtered = [token for token in tokens if token not in stopwords]
    return {
        "Text brut": raw_text,
        "Lowercase": lower,
        "Fără punctuație": clean,
        "Tokenizare": tokens,
        "Fără stop words": filtered,
        "Stop words": len(stopwords),
    }


def _flatten_category_dict(raw: dict[str, Any]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    for raw_label, value in raw.items():
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise DataLoadError(f"Cheie categorie invalidă: {raw_label!r}")

        label = raw_label.strip()
        if isinstance(value, str):
            docs = [value]
        elif isinstance(value, list):
            docs = value
        else:
            raise DataLoadError(
                f"Categoria '{label}' trebuie să fie listă de texte sau un string."
            )

        for doc in docs:
            if doc is None:
                continue
            text = str(doc).strip()
            if text:
                texts.append(text)
                labels.append(label)

    if not texts:
        raise DataLoadError("Nu există documente non-goale în fișier.")
    return texts, labels


def parse_json_bytes(content: bytes, source_name: str = "JSON") -> LoadedDataset:
    try:
        raw = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"JSON invalid: {exc}") from exc

    if not isinstance(raw, dict):
        raise DataLoadError("JSON-ul trebuie să fie un obiect categorie -> texte.")

    texts, labels = _flatten_category_dict(raw)
    counts = {label: labels.count(label) for label in set(labels)}
    return LoadedDataset(texts, labels, counts, len(counts), source_name)


def parse_csv_bytes(content: bytes, source_name: str = "CSV") -> LoadedDataset:
    try:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8"))))
    except UnicodeDecodeError as exc:
        raise DataLoadError(f"Encoding CSV invalid: {exc}") from exc

    if len(rows) < 2:
        raise DataLoadError("CSV-ul trebuie să conțină antet + cel puțin un rând.")

    header = [col.strip().lower() for col in rows[0]]
    if len(header) < 2:
        raise DataLoadError("CSV-ul necesită minim 2 coloane: etichetă și text.")

    label_idx, text_idx = 0, 1
    for idx, col in enumerate(header):
        if col in ("label", "category", "class", "eticheta", "categorie"):
            label_idx = idx
        if col in ("text", "document", "content", "mesaj"):
            text_idx = idx

    grouped: dict[str, list[str]] = {}
    for row in rows[1:]:
        if len(row) <= max(label_idx, text_idx):
            continue
        label = row[label_idx].strip()
        text = row[text_idx].strip()
        if label and text:
            grouped.setdefault(label, []).append(text)

    if not grouped:
        raise DataLoadError("Niciun rând valid în CSV.")

    texts, labels = _flatten_category_dict(grouped)
    counts = {label: labels.count(label) for label in set(labels)}
    return LoadedDataset(texts, labels, counts, len(counts), source_name)


def load_dataset(path: Path | str) -> LoadedDataset:
    path = Path(path)
    content = path.read_bytes()
    if path.suffix.lower() == ".json":
        return parse_json_bytes(content, path.name)
    if path.suffix.lower() == ".csv":
        return parse_csv_bytes(content, path.name)
    raise DataLoadError("Format acceptat: .json sau .csv")


def split_train_test(
    dataset: LoadedDataset,
    test_size: float = 0.2,
) -> tuple[list[str], list[str], list[str], list[str]]:
    min_per_class = min(dataset.label_counts.values())
    stratify = dataset.labels if min_per_class >= 2 else None
    effective_test_size = test_size
    if stratify is not None:
        min_test_items = dataset.n_classes
        requested_test_items = math.ceil(len(dataset.texts) * test_size)
        if requested_test_items < min_test_items:
            effective_test_size = min_test_items / len(dataset.texts)
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.texts,
            dataset.labels,
            test_size=effective_test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        raise DataLoadError(f"Split Train/Test eșuat: {exc}") from exc
    return list(x_train), list(x_test), list(y_train), list(y_test)


def build_tfidf_vectorizer(
    ngram_choice: str,
    max_features: int | None,
    sublinear_tf: bool,
    stop_words: str,
) -> TfidfVectorizer:
    ngram_map = {
        "Unigrame (1,1)": (1, 1),
        "Bigrame (1,2)": (1, 2),
        "Trigrame (1,3)": (1, 3),
    }
    kwargs: dict[str, Any] = {
        "ngram_range": ngram_map.get(ngram_choice, (1, 1)),
        "sublinear_tf": sublinear_tf,
        "min_df": 1,
        "strip_accents": "unicode",
        "lowercase": True,
    }
    if max_features is not None and max_features > 0:
        kwargs["max_features"] = max_features
    if stop_words == "english":
        kwargs["stop_words"] = "english"
    elif stop_words == "romanian":
        kwargs["stop_words"] = list(get_stopwords("romanian"))
    return TfidfVectorizer(**kwargs)


def _wrap_svc_for_proba(clf, n_samples: int):
    if isinstance(clf, LinearSVC) and n_samples >= 30:
        folds = min(3, max(2, n_samples // 10))
        return CalibratedClassifierCV(clf, cv=folds, method="sigmoid")
    return clf


def train_selected_models(
    x_train: list[str],
    y_train: list[str],
    x_test: list[str],
    y_test: list[str],
    selected_models: list[str],
    ngram_choice: str,
    max_features: int | None,
    sublinear_tf: bool,
    stop_words: str,
) -> list[ModelResult]:
    results: list[ModelResult] = []
    for name in selected_models:
        factory = MODEL_REGISTRY.get(name)
        if factory is None:
            continue

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    build_tfidf_vectorizer(
                        ngram_choice, max_features, sublinear_tf, stop_words
                    ),
                ),
                ("clf", _wrap_svc_for_proba(factory(), len(x_train))),
            ]
        )
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        y_pred = pipeline.predict(x_test)
        results.append(
            ModelResult(
                name=name,
                pipeline=pipeline,
                accuracy=float(np.mean(y_pred == np.array(y_test))),
                train_time_sec=elapsed,
                y_pred=y_pred,
                supports_proba=hasattr(pipeline.named_steps["clf"], "predict_proba")
                or hasattr(pipeline.named_steps["clf"], "decision_function"),
            )
        )
    return results


def best_model(results: list[ModelResult]) -> ModelResult | None:
    if not results:
        return None
    preference = {
        "Linear SVC": 3,
        "Regresie Logistică": 2,
        "Naive Bayes (MultinomialNB)": 1,
        "Random Forest": 0,
    }
    return max(
        results,
        key=lambda result: (result.accuracy, preference.get(result.name, -1)),
    )


def predict_with_scores(
    pipeline: Pipeline,
    text: str,
    class_labels: list[str],
) -> tuple[str, dict[str, float]]:
    clf = pipeline.named_steps["clf"]
    matrix = pipeline.named_steps["tfidf"].transform([text])

    if hasattr(clf, "predict_proba"):
        values = clf.predict_proba(matrix)[0]
    elif hasattr(clf, "decision_function"):
        values = clf.decision_function(matrix)[0]
        if np.ndim(values) == 0:
            values = np.array([-float(values), float(values)])
        values = np.exp(values - np.max(values))
        values = values / values.sum()
    else:
        pred = str(pipeline.predict([text])[0])
        return pred, {pred: 1.0}

    scores = {label: float(score) for label, score in zip(class_labels, values)}
    return max(scores, key=scores.get), scores


class TextClassifierRunner:
    def __init__(self) -> None:
        self.dataset: LoadedDataset | None = None
        self.x_train: list[str] = []
        self.x_test: list[str] = []
        self.y_train: list[str] = []
        self.y_test: list[str] = []
        self.results: list[ModelResult] = []
        self.best_pipeline: Pipeline | None = None

    def load(self, path: Path | str = DEMO_DATASET, test_size: float = 0.2) -> str:
        dataset = load_dataset(path)
        self.dataset = dataset
        self.x_train, self.x_test, self.y_train, self.y_test = split_train_test(
            dataset, test_size
        )
        self.results = []
        self.best_pipeline = None
        return (
            f"Dataset încărcat: {dataset.source_name}\n"
            f"Documente: {len(dataset.texts)} | Clase: {dataset.n_classes}\n"
            f"Train: {len(self.x_train)} | Test: {len(self.x_test)}\n"
            f"Clase: {', '.join(sorted(dataset.label_counts))}"
        )

    def train(
        self,
        selected_models: list[str],
        ngram_choice: str,
        max_features: int | None,
        sublinear_tf: bool,
        stop_words: str,
        domain_dictionary: str | None = None,
    ) -> str:
        if self.dataset is None:
            self.load(DEMO_DATASET)

        x_train = [
            enrich_text_with_domain_dictionary(text, domain_dictionary)
            for text in self.x_train
        ]
        x_test = [
            enrich_text_with_domain_dictionary(text, domain_dictionary)
            for text in self.x_test
        ]

        self.results = train_selected_models(
            x_train,
            self.y_train,
            x_test,
            self.y_test,
            selected_models,
            ngram_choice,
            max_features,
            sublinear_tf,
            stop_words,
        )
        best = best_model(self.results)
        if best is not None and self.dataset is not None:
            self.best_pipeline = clone(best.pipeline)
            full_texts = [
                enrich_text_with_domain_dictionary(text, domain_dictionary)
                for text in self.dataset.texts
            ]
            self.best_pipeline.fit(full_texts, self.dataset.labels)
        else:
            self.best_pipeline = None
        return format_training_results(self.results, self.y_test)

    def classify(self, text: str, domain_dictionary: str | None = None) -> str:
        text = text.strip()
        if not text:
            return "Introduceți text pentru clasificare."
        if self.best_pipeline is None:
            return "Antrenați mai întâi cel puțin un model."

        labels = list(self.best_pipeline.named_steps["clf"].classes_)
        enriched_text = enrich_text_with_domain_dictionary(text, domain_dictionary)
        predicted, scores = predict_with_scores(self.best_pipeline, enriched_text, labels)
        lines = [f"Clasă prezisă: {predicted}", "", "Scoruri:"]
        for label, score in sorted(scores.items(), key=lambda item: -item[1]):
            lines.append(f"  {label}: {score:.4f}")
        return "\n".join(lines)


def format_training_results(results: list[ModelResult], y_test: list[str]) -> str:
    if not results:
        return "Nu a fost antrenat niciun model."

    lines = ["Rezultate benchmarking:", ""]
    for result in sorted(results, key=lambda item: -item.accuracy):
        lines.append(
            f"{result.name}: acuratețe={result.accuracy:.4f}, "
            f"timp={result.train_time_sec:.3f}s"
        )

    best = best_model(results)
    if best is not None:
        lines.extend(
            [
                "",
                f"Cel mai bun model: {best.name}",
                "",
                "Raport clasificare:",
                classification_report(y_test, best.y_pred, zero_division=0),
            ]
        )
    return "\n".join(lines)

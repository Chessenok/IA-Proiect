"""
Încărcare date din dicționare externe (JSON / CSV).

Format JSON așteptat:
    {"politica": ["text1", "text2"], "sport": ["text3", ...]}

Pipeline-ul sklearn se antrenează DOAR pe split-ul Train; Test rămâne izolat.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class LoadedDataset:
    """Reprezentare internă după parsare fișier."""

    texts: list[str]
    labels: list[str]
    label_counts: dict[str, int]
    n_classes: int
    source_name: str


class DataLoadError(Exception):
    """Fișier invalid sau structură neacceptată."""


def _flatten_category_dict(raw: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    Transformă {etichetă: [texte]} în liste paralele texts / labels.
    Acceptă și valori string unice (convertite la listă).
    """
    texts: list[str] = []
    labels: list[str] = []

    for label, value in raw.items():
        if not isinstance(label, str) or not label.strip():
            raise DataLoadError(f"Cheie categorie invalidă: {label!r}")

        label = label.strip()
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            raise DataLoadError(
                f"Categoria '{label}' trebuie să fie listă de texte sau un string."
            )

        for doc in items:
            if doc is None:
                continue
            text = str(doc).strip()
            if text:
                texts.append(text)
                labels.append(label)

    if not texts:
        raise DataLoadError("Nu există documente non-goale în fișier.")

    return texts, labels


def parse_json_bytes(content: bytes) -> LoadedDataset:
    try:
        raw = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"JSON invalid: {exc}") from exc

    if not isinstance(raw, dict):
        raise DataLoadError("JSON-ul trebuie să fie un obiect (dicționar) categorie -> texte.")

    texts, labels = _flatten_category_dict(raw)
    counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    return LoadedDataset(
        texts=texts,
        labels=labels,
        label_counts=counts,
        n_classes=len(counts),
        source_name="JSON",
    )


def parse_csv_bytes(content: bytes) -> LoadedDataset:
    """
    CSV cu antet: label,text  sau  category,document
    Alternativ: prima coloană = etichetă, a doua = text.
    """
    try:
        text_io = io.StringIO(content.decode("utf-8"))
        reader = csv.reader(text_io)
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise DataLoadError(f"Encoding CSV invalid: {exc}") from exc

    if len(rows) < 2:
        raise DataLoadError("CSV-ul trebuie să conțină antet + cel puțin un rând de date.")

    header = [c.strip().lower() for c in rows[0]]
    data_rows = rows[1:]

    label_idx, text_idx = 0, 1
    for i, col in enumerate(header):
        if col in ("label", "category", "class", "eticheta", "categorie"):
            label_idx = i
        if col in ("text", "document", "content", "mesaj"):
            text_idx = i

    if len(header) < 2:
        raise DataLoadError("CSV-ul necesită minim 2 coloane (etichetă, text).")

    grouped: dict[str, list[str]] = {}
    for row in data_rows:
        if len(row) <= max(label_idx, text_idx):
            continue
        lbl = row[label_idx].strip()
        doc = row[text_idx].strip()
        if lbl and doc:
            grouped.setdefault(lbl, []).append(doc)

    if not grouped:
        raise DataLoadError("Niciun rând valid în CSV.")

    texts, labels = _flatten_category_dict(grouped)
    counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    return LoadedDataset(
        texts=texts,
        labels=labels,
        label_counts=counts,
        n_classes=len(counts),
        source_name="CSV",
    )


def load_from_uploaded_file(
    uploaded_file,
) -> LoadedDataset:
    """Punct de intrare pentru st.file_uploader."""
    if uploaded_file is None:
        raise DataLoadError("Niciun fișier selectat.")

    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".json"):
        return parse_json_bytes(content)
    if name.endswith(".csv"):
        return parse_csv_bytes(content)

    raise DataLoadError("Format acceptat: .json sau .csv")


def split_train_test(
    dataset: LoadedDataset,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Stratificare când fiecare clasă are >= 2 exemple;
    altfel split simplu fără stratify.
    """
    min_per_class = min(dataset.label_counts.values())
    stratify = dataset.labels if min_per_class >= 2 else None

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.texts,
            dataset.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise DataLoadError(
            f"Split Train/Test eșuat (prea puține date per clasă?): {exc}"
        ) from exc

    return list(x_train), list(x_test), list(y_train), list(y_test)


def class_statistics_table(dataset: LoadedDataset) -> list[dict[str, Any]]:
    return [
        {
            "Clasă": lbl,
            "Documente": cnt,
            "Procent (%)": round(100 * cnt / len(dataset.texts), 1),
        }
        for lbl, cnt in sorted(dataset.label_counts.items(), key=lambda x: -x[1])
    ]

"""
Configurare TfidfVectorizer — extragere caracteristici fără leakage.
Vectorizatorul este primul pas din Pipeline; se fit DOAR pe X_train.
"""
from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    ngram_choice: str,
    max_features: int | None,
    sublinear_tf: bool,
    stop_words: str | None = "english",
) -> TfidfVectorizer:
    ngram_map = {
        "Unigrame (1,1)": (1, 1),
        "Bigrame (1,2)": (1, 2),
        "Trigrame (1,3)": (1, 3),
    }
    ngram_range = ngram_map.get(ngram_choice, (1, 1))

    kwargs: dict[str, Any] = {
        "ngram_range": ngram_range,
        "sublinear_tf": sublinear_tf,
        "min_df": 1,
        "strip_accents": "unicode",
        "lowercase": True,
    }

    if max_features is not None and max_features > 0:
        kwargs["max_features"] = max_features

    if stop_words and stop_words != "none":
        if stop_words == "romanian":
            from modules.preprocessing import get_stopwords

            kwargs["stop_words"] = list(get_stopwords("romanian"))
        else:
            kwargs["stop_words"] = "english"

    return TfidfVectorizer(**kwargs)


def matrix_shape_description(n_docs: int, vectorizer: TfidfVectorizer, x_matrix) -> dict:
    return {
        "Documente": n_docs,
        "Features (coloane)": x_matrix.shape[1],
        "Dimensiune matrice": f"{x_matrix.shape[0]} × {x_matrix.shape[1]}",
        "Sparsitate": f"{100 * (1 - x_matrix.nnz / max(x_matrix.size, 1)):.1f}%",
        "ngram_range": vectorizer.ngram_range,
        "max_features": vectorizer.max_features,
    }

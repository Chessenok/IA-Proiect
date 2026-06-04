"""
NLP Playground — pași de preprocesare vizibili în UI.
Reflectă fluxul din laborator: brut -> lowercase -> fără punctuație -> tokens -> fără stopwords.
"""
from __future__ import annotations

import re
import string
from typing import Callable

# Stopwords românești din proiect (opțional)
_ROM_STOP: set[str] | None = None


def _load_romanian_stopwords() -> set[str]:
    global _ROM_STOP
    if _ROM_STOP is not None:
        return _ROM_STOP
    try:
        from pathlib import Path
        import json

        path = Path(__file__).resolve().parents[2] / "data" / "nlp" / "stopwords_ro.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            _ROM_STOP = set(data.get("words", []))
            return _ROM_STOP
    except Exception:
        pass
    _ROM_STOP = set()
    return _ROM_STOP


def get_stopwords(language: str) -> set[str]:
    lang = language.lower()
    if lang in ("romanian", "romana", "ro"):
        words = _load_romanian_stopwords()
        if words:
            return words
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        if lang in ("english", "en", "engleza"):
            return set(ENGLISH_STOP_WORDS)
    except ImportError:
        pass
    return _load_romanian_stopwords() or set()


def step_lowercase(text: str) -> str:
    return text.lower()


def step_remove_punctuation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation + "„”«»…")
    return text.translate(table)


def step_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ0-9]+", text)


def step_remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    return [t for t in tokens if t not in stopwords]


def preprocess_playground(
    raw_text: str,
    stopword_language: str = "english",
) -> dict[str, str | list[str]]:
    """Returnează fiecare etapă pentru afișare în Streamlit."""
    if not raw_text.strip():
        return {"error": "Introduceți o frază."}

    sw = get_stopwords(stopword_language)
    s1 = raw_text
    s2 = step_lowercase(s1)
    s3 = step_remove_punctuation(s2)
    tokens = step_tokenize(s3)
    filtered = step_remove_stopwords(tokens, sw)

    return {
        "1. Text brut": s1,
        "2. Lowercase": s2,
        "3. Fără punctuație": s3,
        "4. Tokenizare": tokens,
        "5. Fără stop words": filtered,
        "stopwords_used": len(sw),
    }


def sklearn_analyzer_from_language(language: str) -> Callable[[str], list[str]]:
    """
    Analyzer custom pentru TfidfVectorizer (opțional).
    Implicit folosim preprocessor='lower' în vectorizer; acest analyzer e pentru demo avansat.
    """
    sw = get_stopwords(language)

    def analyze(doc: str) -> list[str]:
        doc = step_remove_punctuation(step_lowercase(doc))
        return step_remove_stopwords(step_tokenize(doc), sw)

    return analyze

from __future__ import annotations

from collections import Counter

from nlp.dictionaries import get_dictionary, load_all_dictionaries, tokenize


def analyze_sentiment(text: str, dictionary_name: str = "sentiment_ro") -> dict:
    lex = get_dictionary(dictionary_name) or {}
    positive = set(lex.get("positive", []))
    negative = set(lex.get("negative", []))

    tokens = tokenize(text)
    pos_hits = [t for t in tokens if t in positive]
    neg_hits = [t for t in tokens if t in negative]

    score = len(pos_hits) - len(neg_hits)
    if score > 0:
        label = "pozitiv"
    elif score < 0:
        label = "negativ"
    else:
        label = "neutru"

    return {
        "task": f"Sentiment ({dictionary_name})",
        "label": label,
        "score": score,
        "positive_hits": pos_hits,
        "negative_hits": neg_hits,
        "tokens_total": len(tokens),
    }


def extract_keywords(
    text: str,
    top_n: int = 10,
    stopwords_name: str = "stopwords_ro",
) -> dict:
    sw = get_dictionary(stopwords_name) or {}
    stopwords = set(sw.get("words", []))

    tokens = [t for t in tokenize(text) if t not in stopwords and len(t) > 2]
    freq = Counter(tokens).most_common(top_n)

    return {
        "task": f"Cuvinte cheie ({stopwords_name})",
        "keywords": [{"word": w, "count": c} for w, c in freq],
        "tokens_after_filter": len(tokens),
    }


def detect_domain_terms(text: str, dictionary_name: str = "logistics_drone") -> dict:
    domain = get_dictionary(dictionary_name) or {}
    categories: dict[str, list[str]] = domain.get("categories", {})

    tokens = tokenize(text)
    found: dict[str, list[str]] = {}

    def _matches(term: str) -> bool:
        term = term.lower()
        return any(
            tok == term or tok.startswith(term) or term in tok for tok in tokens
        )

    for category, words in categories.items():
        matches = sorted(w for w in words if _matches(w))
        if matches:
            found[category] = matches

    return {
        "task": f"Termeni domeniu ({dictionary_name})",
        "categories_found": found,
        "categories_total": len(categories),
    }


def run_nlp_task(
    task_name: str,
    text: str,
    *,
    sentiment_dictionary: str = "sentiment_ro",
    stopwords_dictionary: str = "stopwords_ro",
    domain_dictionary: str = "logistics_drone",
) -> dict:
    text = text.strip()
    if not text:
        return {"error": "Textul este gol."}

    if "Sentiment" in task_name:
        return analyze_sentiment(text, sentiment_dictionary)
    if "Cuvinte cheie" in task_name or "cheie" in task_name.lower():
        return extract_keywords(text, stopwords_name=stopwords_dictionary)
    if "domeniu" in task_name.lower() or "Termeni" in task_name:
        return detect_domain_terms(text, domain_dictionary)

    return {"error": f"Sarcină necunoscută: {task_name}"}


def format_nlp_result(result: dict) -> str:
    if "error" in result:
        return f"Eroare: {result['error']}"

    lines = [f"=== {result.get('task', 'NLP')} ===", ""]

    if "label" in result:
        lines.append(f"Sentiment: {result['label']} (scor: {result['score']})")
        lines.append(f"Tokeni analizați: {result['tokens_total']}")
        if result["positive_hits"]:
            lines.append(f"  + {', '.join(result['positive_hits'])}")
        if result["negative_hits"]:
            lines.append(f"  - {', '.join(result['negative_hits'])}")

    if "keywords" in result:
        lines.append(f"Tokeni utili: {result['tokens_after_filter']}")
        if result["keywords"]:
            lines.append("Top cuvinte:")
            for item in result["keywords"]:
                lines.append(f"  • {item['word']}: {item['count']}")
        else:
            lines.append("(niciun cuvânt cheie)")

    if "categories_found" in result:
        found = result["categories_found"]
        lines.append(
            f"Categorii detectate: {len(found)} / {result['categories_total']}"
        )
        if found:
            for cat, words in found.items():
                lines.append(f"  [{cat}]: {', '.join(words)}")
        else:
            lines.append("(niciun termen de domeniu)")

    return "\n".join(lines)


def dictionary_stats() -> str:
    dicts = load_all_dictionaries()
    lines = ["Dicționare încărcate:"]
    for name, data in dicts.items():
        desc = data.get("description", "")
        if "positive" in data:
            n = len(data["positive"]) + len(data["negative"])
            lines.append(f"  • {name}: {n} intrări sentiment")
        elif "words" in data:
            lines.append(f"  • {name}: {len(data['words'])} stopwords")
        elif "categories" in data:
            total = sum(len(v) for v in data["categories"].values())
            lines.append(f"  • {name}: {total} termeni în {len(data['categories'])} categorii")
        else:
            lines.append(f"  • {name}: {desc}")
    return "\n".join(lines)

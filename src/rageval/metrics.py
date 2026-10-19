from __future__ import annotations

from .text import content_tokens


def hit_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    return float(any(d in gold for d in ranked[:k]))


def reciprocal_rank(ranked: list[str], gold: list[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in gold:
            return 1 / i
    return 0.0


def answer_correct(text: str, must_contain: list[str]) -> bool:
    low = text.lower()
    return all(m.lower() in low for m in must_contain)


def faithfulness(text: str, cited_docs: list[str]) -> float:
    """Share of the answer's content tokens that appear in the cited documents."""
    toks = content_tokens(text)
    if not toks or not cited_docs:
        return 0.0
    ctx = set(content_tokens(" ".join(cited_docs)))
    return sum(t in ctx for t in toks) / len(toks)

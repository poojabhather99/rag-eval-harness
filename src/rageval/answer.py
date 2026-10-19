"""Extractive answerer: returns the best-supported sentence with a citation, or abstains."""

from __future__ import annotations

from dataclasses import dataclass, field

from .text import content_tokens, sentences


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False


def answer(query: str, hits: list[tuple[str, float]], docs: dict[str, str],
           min_overlap: float = 0.5) -> Answer:
    q = set(content_tokens(query))
    best: tuple[float, str, str] = (0.0, "", "")
    for doc_id, _ in hits:
        for s in sentences(docs[doc_id]):
            score = len(q & set(content_tokens(s))) / (len(q) or 1)
            if score > best[0]:
                best = (score, s, doc_id)
    if best[0] < min_overlap:
        return Answer("I don't know based on the provided documents.", [], True)
    return Answer(best[1], [best[2]])

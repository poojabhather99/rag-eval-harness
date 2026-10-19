"""Two retrievers to compare: plain term overlap and BM25."""

from __future__ import annotations

import math
from collections import Counter

from .text import content_tokens


class Overlap:
    name = "overlap"

    def __init__(self, docs: dict[str, str]):
        self.docs = {k: set(content_tokens(v)) for k, v in docs.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = set(content_tokens(query))
        scored = [(d, len(q & t) / (len(q) or 1)) for d, t in self.docs.items()]
        scored = [s for s in scored if s[1] > 0]
        return sorted(scored, key=lambda x: (-x[1], x[0]))[:k]


class BM25:
    name = "bm25"

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {d: Counter(content_tokens(t)) for d, t in docs.items()}
        self.len = {d: sum(c.values()) for d, c in self.tf.items()}
        self.avg = sum(self.len.values()) / len(docs)
        df: Counter[str] = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = content_tokens(query)
        out = []
        for d, tf in self.tf.items():
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    norm = f + self.k1 * (1 - self.b + self.b * self.len[d] / self.avg)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            if s > 0:
                out.append((d, s))
        return sorted(out, key=lambda x: (-x[1], x[0]))[:k]

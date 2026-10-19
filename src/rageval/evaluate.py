"""Run the harness: python -m rageval.evaluate"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .answer import answer
from .metrics import answer_correct, faithfulness, hit_at_k, reciprocal_rank
from .retrieval import BM25, Overlap

DATA = Path(__file__).resolve().parents[2] / "evals"


def load() -> tuple[dict[str, str], list[dict[str, Any]]]:
    docs = json.loads((DATA / "corpus.json").read_text())
    qs = [json.loads(line) for line in (DATA / "questions.jsonl").read_text().splitlines() if line]
    return docs, qs


def evaluate(retriever_cls, k: int = 3) -> dict[str, Any]:
    docs, qs = load()
    r = retriever_cls(docs)
    ans_qs = [q for q in qs if q["gold_docs"]]
    unans = [q for q in qs if not q["gold_docs"]]
    hit = mrr = correct = faith = 0.0
    for q in ans_qs:
        hits = r.search(q["question"], k)
        ranked = [d for d, _ in hits]
        hit += hit_at_k(ranked, q["gold_docs"], k)
        mrr += reciprocal_rank(ranked, q["gold_docs"])
        a = answer(q["question"], hits, docs)
        correct += (not a.abstained) and answer_correct(a.text, q["must_contain"])
        faith += faithfulness(a.text, [docs[c] for c in a.citations])
    abstain_ok = sum(answer(q["question"], r.search(q["question"], k), docs).abstained
                     for q in unans)
    n = len(ans_qs)
    return {
        "retriever": r.name, "k": k, "answerable": n, "unanswerable": len(unans),
        f"hit@{k}": round(hit / n, 3), "mrr": round(mrr / n, 3),
        "answer_accuracy": round(correct / n, 3), "faithfulness": round(faith / n, 3),
        "abstain_accuracy": round(abstain_ok / len(unans), 3) if unans else None,
    }


def main() -> None:
    for k in (1, 3):
        for cls in (Overlap, BM25):
            print(json.dumps(evaluate(cls, k)))


if __name__ == "__main__":
    main()

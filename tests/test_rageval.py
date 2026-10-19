import pytest

from rageval.answer import answer
from rageval.evaluate import evaluate, load
from rageval.metrics import answer_correct, faithfulness, hit_at_k, reciprocal_rank
from rageval.retrieval import BM25, Overlap
from rageval.text import content_tokens, sentences

DOCS = {"a": "Cats sleep sixteen hours daily.", "b": "Dogs bark at strangers loudly."}


def test_tokenisation_drops_stopwords():
    assert content_tokens("What is the capital of France?") == ["capital", "france"]
    assert sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


@pytest.mark.parametrize("cls", [Overlap, BM25])
def test_retrievers_rank_relevant_first(cls):
    assert cls(DOCS).search("how long do cats sleep")[0][0] == "a"


@pytest.mark.parametrize("cls", [Overlap, BM25])
def test_no_match_returns_empty(cls):
    assert cls(DOCS).search("quantum chromodynamics") == []


def test_bm25_prefers_rarer_terms():
    docs = {"x": "alpha beta", "y": "alpha gamma", "z": "alpha delta"}
    assert BM25(docs).search("alpha gamma")[0][0] == "y"


def test_metrics():
    assert hit_at_k(["a", "b"], ["b"], 1) == 0.0 and hit_at_k(["a", "b"], ["b"], 2) == 1.0
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0
    assert answer_correct("Given at 9 months", ["9 months"])
    assert not answer_correct("Given at 9 months", ["16 months"])


def test_faithfulness_bounds():
    assert faithfulness("cats sleep daily", [DOCS["a"]]) == 1.0
    assert faithfulness("dogs fly", [DOCS["a"]]) == 0.0
    assert faithfulness("anything", []) == 0.0


def test_answer_cites_and_abstains():
    r = BM25(DOCS)
    q1 = "how many hours do cats sleep daily"
    a = answer(q1, r.search(q1), DOCS)
    assert not a.abstained and a.citations == ["a"]
    q = "what is the tallest mountain"
    assert answer(q, r.search(q), DOCS).abstained


def test_dataset_is_well_formed():
    docs, qs = load()
    assert len(docs) >= 10 and len(qs) >= 20
    for q in qs:
        assert all(g in docs for g in q["gold_docs"])
        if q["gold_docs"]:
            gold_text = " ".join(docs[g] for g in q["gold_docs"]).lower()
            assert all(m.lower() in gold_text for m in q["must_contain"])


def test_evaluation_is_deterministic_and_bounded():
    a, b = evaluate(BM25, 3), evaluate(BM25, 3)
    assert a == b
    for key in ("hit@3", "mrr", "answer_accuracy", "faithfulness", "abstain_accuracy"):
        assert 0.0 <= a[key] <= 1.0


def test_bm25_not_worse_than_overlap_at_k1():
    assert evaluate(BM25, 1)["hit@1"] >= evaluate(Overlap, 1)["hit@1"]

# rag-eval-harness

[![CI](https://github.com/poojabhather99/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/poojabhather99/rag-eval-harness/actions/workflows/ci.yml)

A small, dependency-free harness for measuring a RAG pipeline end to end: **retrieval quality, answer accuracy, faithfulness to cited sources, and whether the system abstains when the documents don't contain the answer.** It runs offline and is deterministic, so it works as a regression test in CI.

It complements [bot-gpt](https://github.com/poojabhather99/bot-gpt): bot-gpt is the RAG application, this repo is how you would measure changes to one.

## What is measured

| Metric | Meaning |
|---|---|
| `hit@k` | A gold document appears in the top k results |
| `mrr` | Reciprocal rank of the first gold document |
| `answer_accuracy` | Answer is not an abstention and contains every required phrase |
| `faithfulness` | Share of the answer's content tokens found in the cited documents |
| `abstain_accuracy` | On unanswerable questions, the system says it doesn't know |

The dataset (`evals/`) is 10 short documents about a fictional community health programme and 21 questions: 18 answerable (some paraphrased so they don't share exact words with the source) and 3 unanswerable. Dataset tests check that every gold document exists and that each required phrase really appears in its gold document.

## Components

- `retrieval.py`: term-overlap baseline and BM25, behind the same `search(query, k)` interface
- `answer.py`: extractive answerer that returns the best-supported sentence plus a citation, or abstains below an overlap threshold
- `metrics.py`, `evaluate.py`: metrics and the runner

## Run

```bash
pip install -e ".[dev]"
python -m rageval.evaluate
pytest -q
```

## Current results

| Retriever | k | hit@k | MRR | Answer acc. | Faithfulness | Abstain acc. |
|---|---|---|---|---|---|---|
| overlap | 1 | 1.00 | 1.00 | 0.889 | 0.889 | 0.667 |
| bm25 | 1 | 1.00 | 1.00 | 0.889 | 0.889 | 0.667 |
| overlap | 3 | 1.00 | 1.00 | 0.889 | 0.889 | 0.667 |
| bm25 | 3 | 1.00 | 1.00 | 0.889 | 0.889 | 0.667 |

**The two retrievers tie.** With only 10 short documents the benchmark is too easy to separate BM25 from term overlap, so these numbers do not show that BM25 is better. Separating them needs a larger corpus with distractor documents; that is the first thing to extend.

The remaining errors come from the answerer, not retrieval, and they are informative:

- "At what time does the dashboard refresh?" is answered with an abstention: the source sentence says "refreshes every night at 2 AM" and shares too few content words with the question, so the fixed threshold rejects it.
- "What is the salary of the district coordinator?" is answered with an unrelated sentence that mentions the district coordinator. Word overlap cannot tell that the question asks about something the documents never state.
- Faithfulness equals accuracy here partly by coincidence: both failures score 0 (an abstention has no citation to ground).

## Limitations

- Extractive answerer only; no LLM generation is included. The metrics are model-agnostic, so a generative answerer can be dropped in by returning the same `Answer` structure.
- Faithfulness is token overlap with the cited text, a coarse proxy. It cannot detect a fluent answer that contradicts its source.
- Small synthetic corpus, and the abstention threshold (0.5) was set once and not tuned.

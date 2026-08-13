# Failure analysis

The evaluation layer should classify errors instead of treating every bad answer as an LLM problem:

- `retrieval_failure`: the relevant chunk was not returned.
- `ranking_failure`: the relevant chunk was returned but ranked below the useful top-k.
- `missing_context`: the corpus does not contain evidence for the question.
- `generation_failure`: evidence was present but the answer was incorrect.
- `hallucination`: the answer asserted information absent from the retrieved context.
- `citation_failure`: the answer was useful but did not cite the supporting chunk.

The current smoke dataset is intentionally small. The next evaluation pass should add 50–80 questions across direct, paraphrase, numeric, temporal, multi-hop, distractor and negative cases, then compare vector, lexical and hybrid retrieval at k=1, 2, 3, 5 and 10.

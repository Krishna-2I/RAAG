# Evaluating RAG

Without measurements, changes to a RAG system are guesses. Build a golden set of 30 to 50 questions with the evidence text that answers each one.

Retrieval metrics: hit rate at k asks whether any relevant chunk appears in the top k. Recall at k measures how much of the needed evidence was retrieved. Mean reciprocal rank, or MRR, rewards putting the first relevant chunk near the top: it is the average of 1 divided by the rank of the first relevant result.

Generation metrics: faithfulness checks that every claim in the answer is supported by the retrieved context. Answer relevance checks that the answer addresses the question. When an answer is wrong, first decide whether retrieval or generation failed.

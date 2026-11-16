# Learn notes: Worlds 1 and 2

## f4: embeddings, cosine similarity, HNSW

- **Embedding**: a model maps text to a vector; similar meaning gives nearby vectors.
- **Cosine similarity**: the angle between vectors. With unit-length vectors it equals the dot product, so search is one matrix multiply (`vectors @ query`). That is exactly what `Index.search_dense` does.
- **One model per index**: vectors from different models are not comparable. Changing the embedder means re-indexing.
- **Exact vs approximate search**: exact is fine up to a few thousand chunks; HNSW (a layered graph) trades a little recall for speed at hundreds of thousands. Knobs: `M`, `efConstruction`, `efSearch`.
- **Chunking**: small chunks match precisely but lack context; large chunks carry context but blur the embedding. Start at 800 chars with 120 overlap, then measure.

## naive: why it fails (keep these examples)

Run `ragforge eval --show-misses` and read the misses. Typical causes: the answer is split across a chunk boundary, the query uses an exact identifier (error code, name) that embeddings blur, or the right chunk is retrieved but ranked 6th.

## hybrid: sparse vs dense, RRF

- **BM25** scores by matching query words (rare words weigh more, repetition has diminishing returns, long chunks are normalized: `k1=1.5`, `b=0.75`). Great for exact identifiers, blind to synonyms.
- **Dense** finds meaning but blurs rare tokens.
- **RRF**: `score(chunk) = sum over lists of 1 / (60 + rank)`. Uses ranks only, so no score scaling between the two retrievers.

## rerank: bi-encoder vs cross-encoder

- **Bi-encoder**: embeds query and chunk separately. Fast, approximate, scales to millions.
- **Cross-encoder**: reads query and chunk together. Slow, accurate, so use it only on ~30 candidates.
- Pattern: cheap retrieve 30 -> rerank -> keep 5. Usually the highest-impact upgrade after naive RAG.

## Metrics you now track

Hit@k (any relevant chunk in top k), Recall@k (share of evidence found), MRR (1 / rank of the first relevant chunk), Precision@k, latency, and with `--judge` faithfulness and answer relevance. For every wrong answer decide first: did retrieval fail or generation?

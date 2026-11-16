# Troubleshooting

Error E-4012 means the index schema does not match the embedding model. Fix it by re-indexing the collection with the same embedder that will be used for queries.

Error E-5003 means the vector store ran out of memory during ingestion. Reduce the batch size or switch to a disk-backed index.

Slow answers are usually caused by a large candidate pool in the reranking step. Lower the candidate count from 30 to 20 and measure again.

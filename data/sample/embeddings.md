# Embeddings

An embedding model turns a piece of text into a vector of numbers so that texts with similar meaning land close together. Retrieval works by embedding the query and finding the nearest chunk vectors.

Cosine similarity measures the angle between two vectors. When vectors are normalized to unit length, cosine similarity equals the dot product, which is why most vector stores normalize embeddings first.

The model used at indexing time must be the same model used at query time. Vectors from different models live in different spaces and cannot be compared. The bge-small model produces 384-dimensional vectors, while larger models produce 768 or 1024 dimensions. If a client changes embedding model, all of their documents must be re-indexed.

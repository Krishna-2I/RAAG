# HNSW indexes

Hierarchical Navigable Small World, or HNSW, is a graph index for approximate nearest neighbor search. Vectors are nodes in a layered graph. A search starts at the sparse top layer, jumps greedily toward the query, and descends to denser layers until it reaches the bottom layer.

Three settings matter most. M is the number of links per node. efConstruction controls how carefully the graph is built. efSearch controls how many candidates are explored at query time. Raising efSearch improves recall but makes each query slower.

Exact search compares the query with every vector and is fine for a few thousand chunks. HNSW becomes worthwhile once a collection reaches hundreds of thousands of vectors.

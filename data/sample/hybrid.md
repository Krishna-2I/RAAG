# Hybrid retrieval and rank fusion

Hybrid retrieval runs dense search and BM25 search separately and merges the two ranked lists. Reciprocal Rank Fusion, or RRF, scores each chunk as the sum of 1 divided by (60 + rank) over every list it appears in. The constant 60 is the standard default. RRF uses only ranks, so it needs no score normalization between the two retrievers.

A chunk ranked well by both retrievers rises to the top, while a chunk found by only one still has a chance.

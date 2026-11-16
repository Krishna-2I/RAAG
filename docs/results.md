# Results

Index embedder: `hash:512`, reranker: `overlap`

Eval set: 15 questions, k=3

| Strategy | Hit@3 | Recall@3 | MRR | Precision@3 | Retrieval ms | Recall vs naive | MRR vs naive |
|---|---|---|---|---|---|---|---|
| naive | 0.60 | 0.60 | 0.47 | 0.22 | 0.1 | baseline | baseline |
| hybrid | 0.87 | 0.87 | 0.61 | 0.31 | 0.2 | +44.4% | +31.0% |
| rerank | 1.00 | 1.00 | 0.84 | 0.38 | 0.6 | +66.7% | +81.0% |

> Offline sanity check: `hash:512` embedder and `overlap` reranker on the 8-document sample. See README, "About the numbers".

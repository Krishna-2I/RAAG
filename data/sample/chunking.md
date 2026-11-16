# Chunking

Documents are split into chunks before embedding because an embedding of a whole book blurs every topic together. Small chunks match a question precisely but may lack the surrounding context needed to answer it. Large chunks carry more context but dilute the embedding and waste prompt space.

A common starting point is 800 characters with an overlap of 120 characters. Overlap repeats the end of one chunk at the start of the next so that a sentence cut at a boundary is still found whole in one of them.

Recursive splitting tries paragraph breaks first, then line breaks, then sentences, then words, so chunks end at natural boundaries instead of in the middle of a sentence.

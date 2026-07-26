# Supplementary ASR Evaluation Scripts

This folder contains portable entry points for the two requested supplementary evaluations.

- `reranking/`: fixed TinyBERT-L2 reranking evaluation. It reports `ND-ASR` and `RD-ASR`.
- `robustrag/`: fixed Mistral-7B RobustRAG evaluation. It reports `ND-ASR`, `RR-ASR`, and `RD+RR-ASR`.
- `MODELS_AND_DATA.md`: models and external data/model caches that must be downloaded or copied on another server.

Both evaluations use NQ single-hop full-corpus retrieval with `facebook/contriever`. RAGDefender is fixed to `paraphrase-MiniLM-L6-v2`.

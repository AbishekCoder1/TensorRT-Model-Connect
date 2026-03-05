# Encoder and Ranking Backends

Non-autoregressive encoder-style runtime paths.

Key files:
- `encoder_backend.*`: encoder-only hidden-state inference.
- `embedding_backend.*`: embedding extraction (text/image and pooling).
- `reranking_backend.*`: reranker score inference.

How to understand:
1. Start with the backend entry method (`encode`, `embed`, `rerank`).
2. Follow tensor binding/execution in each backend.
3. Cross-reference multimodal preprocessing when image-aware embedding is enabled.

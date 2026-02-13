# Transformers Coverage Analysis (API-first TRT)

## Snapshot source
Model-family counts are derived from:
`/home/yifeif/repos/transformers/src/transformers/models/auto/modeling_auto.py`

Observed counts in this checkout:
- Causal LM mappings: 144
- Seq2Seq causal LM mappings: 30
- Image classification mappings: 44
- Token classification mappings: 87
- Masked LM mappings: 42
- Speech seq2seq mappings: 12

## Mainstream families and expected implementation difficulty

| Family | Examples | Difficulty | Why |
|---|---|---|---|
| Decoder-only dense LM | Llama, Qwen, Mistral, Gemma, Phi | Medium | Core attention/MLP stack is regular; main work is KV cache + rotary + fused norms |
| Decoder-only MoE LM | Mixtral, Qwen-MoE, DeepSeek-MoE | High | Router/top-k experts + sparse dispatch + memory movement |
| Encoder-only text | BERT, RoBERTa, DeBERTa | Medium | Simpler than autoregressive decode; still needs pooling/head variants |
| Seq2Seq text generation | T5, BART, FLAN-T5 | High | Dual-stack + cross-attention + generation bookkeeping |
| Vision transformer | ViT, DeiT | Medium | Mostly standard MHA/MLP; preprocessing + heads add integration effort |
| Multimodal VLM | Qwen-VL, LLaVA-like | Very high | Coordinating vision encoder + projector + text decoder + tokenizer/processors |
| Speech seq2seq | Whisper, SpeechT5 | Very high | Frontend feature extraction + alignment + decode path complexity |

## TensorRT API compatibility outlook

### Mostly covered natively (10.15+ style APIs)
- Matrix multiply, elementwise ops, reductions, reshape/shuffle, slice, softmax.
- Attention layer entrypoints.
- Rotary embedding and KV cache update APIs in newer TensorRT headers.

### Likely plugin-heavy areas
- Custom normalization or fused activation patterns not directly exposed.
- Exotic positional encodings and model-specific blocks.
- Sparse MoE dispatch/combine patterns.
- Some tokenizer-adjacent pre/post operations if pushed into engine.

## Effort estimate (rough, for API parity direction)
- First robust mainstream decoder-only family: 6-10 weeks.
- 4-6 mainstream decoder families with shared infra: 3-5 months.
- Add seq2seq + strong encoder coverage: +2-3 months.
- Add multimodal and speech with acceptable quality/perf: +4-8 months.

These estimates assume 1-2 experienced engineers, plugin path allowed, and iterative correctness/performance tuning.

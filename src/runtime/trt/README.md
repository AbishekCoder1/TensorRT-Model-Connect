# TensorRT Runtime Modules

This folder hosts TensorRT runtime implementation code, split by responsibility:

- `core/`: shared TensorRT/CUDA primitives used by all strategies.
- `recurrent/`: Mamba/RWKV/hybrid recurrent decode paths.
- `audio/`: speech/audio pipelines (Whisper, Bark, Magpie, Omni, speech-to-speech).
- `multimodal/`: VL front-end pieces (image preprocessing, vision encoder, VL decode).
- `encoder/`: encoder-only, embedding, and reranking backends.
- `perception/`: segmentation, SAM, detection, neural operators.
- `diffusion/`: diffusion backends (base, Wan, FLUX, Z-Image).

How to read this code:
1. Start at `src/cabi/api/trtf_c.cpp` and strategy factories in `src/cabi/factories/factories_*.cpp`.
2. Follow the selected backend class into one of these subfolders.
3. Look in `core/` for shared decode/runtime helpers used by multiple backends.

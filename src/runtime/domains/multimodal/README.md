# Multimodal Runtime (Vision-Language)

Vision-language preprocessing and cross-modal runtime components.

Key files:
- `image_preprocessor.*`: model-specific preprocessing strategies.
- `vision_engine.*`: vision encoder execution helpers.
- `vl_backend.*`: VL generation path combining vision features and text decode.

How to understand:
1. Inspect `image_preprocessor` strategy selection and config parsing.
2. Follow `vision_engine` tensor bindings and output extraction.
3. Read `vl_backend::generate_vl` for end-to-end multimodal decode orchestration.

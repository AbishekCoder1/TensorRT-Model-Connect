# Perception Backends

Vision task runtimes for segmentation, prompted segmentation, detection, and neural operators.

Key files:
- `segmentation_backend.*`: segmentation mask inference.
- `sam_backend.*`: prompted segmentation (SAM).
- `detection_backend.*`: object detection output path.
- `neural_operator_backend.*`: DeepONet/FNO runtime support.

How to understand:
1. Start from backend public method (`segment`, `detect`, `solve`).
2. Trace preprocess -> TRT inference -> output postprocess helpers.
3. Use `multimodal/image_preprocessor.*` when image normalization/cropping behavior matters.

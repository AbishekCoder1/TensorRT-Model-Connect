# Strategy Factories

This folder contains strategy-specific pipeline construction modules.

Files:
- `factories_text.*`: decoder/recurrent text pipelines
- `factories_multimodal.*`: vision-language and composite pipelines
- `factories_audio.*`: speech/audio pipelines
- `factories_vision.*`: segmentation/detection/perception pipelines
- `factories_encoder.*`: encoder/embedding/rerank pipelines
- `factories_diffusion.*`: diffusion pipelines
- `factory_decls.h`: shared factory declarations used across pipeline and registry code

Add new strategy construction logic in the closest domain factory module.

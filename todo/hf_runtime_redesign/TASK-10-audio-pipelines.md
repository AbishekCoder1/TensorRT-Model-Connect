# TASK-10: WhisperPipeline + BarkPipeline + MagpiePipeline + SpeechPipeline + OmniPipeline

## Status: blocked (needs TASK-01, TASK-02, TASK-04)
## Phase: 3 (Audio)
## Risk: high — audio backends are the most complex (60KB each), careful porting needed

## Goal

Implement audio pipelines. Each is a distinct pipeline class because the
architectures differ significantly:

| Pipeline | Architecture | Engines | State |
|----------|-------------|---------|-------|
| WhisperPipeline | Encoder-decoder | encoder + decoder | KvCache (decoder) + CrossKv |
| BarkPipeline | 3-stage cascade | semantic + coarse + fine + codec | KvCache×2 |
| MagpiePipeline | Multi-codebook decoder | text_enc + decoder + codec | KvCache + codebook state |
| SpeechPipeline | Audio-to-audio | depth_enc + decoder + codec | RecurrentState + delay cache |
| OmniPipeline | Text+audio MoE | text_enc + talker + thinker | KvCache + audio state |

## WhisperPipeline (simplest audio)

```cpp
class WhisperPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> encoder_;    // mel → encoder_output
    std::unique_ptr<TrtModule> decoder_;    // autoregressive token decoder
    std::unique_ptr<KvCache> decoder_cache_;
    std::vector<DeviceTensor> cross_k_;     // [num_layers] cross-attention K
    std::vector<DeviceTensor> cross_v_;     // [num_layers] cross-attention V
    WhisperConfig config_;

    TranscriptionResult transcribe(const float* audio, int32_t len, int32_t max_tokens) override {
        // 1. Mel spectrogram (CPU preprocessing)
        // 2. encoder_.forward_device({mel}) → encoder_output (stays on GPU)
        // 3. Compute cross-K/V from encoder output (one-time, stays on GPU)
        // 4. Bind cross-K/V to decoder module
        // 5. Standard autoregressive decode loop with decoder_cache_
        // 6. Return token IDs → text
    }
};
```

## BarkPipeline

```cpp
class BarkPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> semantic_model_;   // text → semantic tokens
    std::unique_ptr<TrtModule> coarse_model_;     // semantic → coarse tokens
    std::unique_ptr<TrtModule> fine_model_;        // coarse → fine tokens (optional)
    std::unique_ptr<TrtModule> codec_decoder_;     // fine tokens → waveform
    std::unique_ptr<KvCache> semantic_cache_;
    std::unique_ptr<KvCache> coarse_cache_;
    std::unique_ptr<ITokenizer> tokenizer_;

    AudioResult generate_audio(const std::string& prompt, const AudioConfig& cfg) override {
        // Stage 1: text → semantic tokens (autoregressive with semantic_cache_)
        // Stage 2: semantic → coarse tokens (autoregressive with coarse_cache_)
        // Stage 3: coarse → fine tokens (optional fine model)
        // Stage 4: fine tokens → waveform (codec decoder single pass)
    }
};
```

## MagpiePipeline, SpeechPipeline, OmniPipeline

These are the most complex and should be ported last within this task.
Each has unique multi-engine orchestration with codebook delay patterns,
CFG support, and custom state management.

Recommendation: port one at a time, validate E2E after each.

## What it replaces

- `WhisperBackend` (11KB)
- `BarkBackend` (35KB)
- `MagpieTtsBackend` (60KB)
- `SpeechBackend` (59KB)
- `OmniBackend` (17KB)
- Audio strategy builder (1.3KB)
- Audio adapter/port/service wrappers

Total replaced: ~183KB → estimated ~30KB

## Suggested sub-task ordering

1. WhisperPipeline (simplest, good pattern validation)
2. BarkPipeline (multi-stage cascade, tests KvCache reuse)
3. OmniPipeline (MoE routing)
4. MagpiePipeline (most complex multi-codebook)
5. SpeechPipeline (audio-to-audio with depth encoder)

## Files to create

- `src/runtime/pipelines/whisper_pipeline.h` + `.cpp`
- `src/runtime/pipelines/bark_pipeline.h` + `.cpp`
- `src/runtime/pipelines/magpie_pipeline.h` + `.cpp`
- `src/runtime/pipelines/speech_pipeline.h` + `.cpp`
- `src/runtime/pipelines/omni_pipeline.h` + `.cpp`
- Tests for each

## Dependencies

TASK-01, TASK-02, TASK-03 (ITokenizer), TASK-04 (Factory)

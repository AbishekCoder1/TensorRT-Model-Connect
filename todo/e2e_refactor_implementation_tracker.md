# E2E Unified Testing Framework - Implementation Tracker

## Status: COMPLETE

All phases implemented and verified. 48/48 models have full runner, comparator, and reference coverage.

---

## Verification Results

```
============================================================
INTEGRATION TEST SUITE
============================================================
1. Manifest Loading:        48/48 manifests loaded, all have required fields
2. Strategy Coverage:       48/48 models have matching runner, comparator, and reference
3. Strategy Filtering:      text=34, VL=4, diffusion=1, plus all other modalities
4. Threshold Resolution:    Chain works (defaults -> per-model -> inline overrides)
5. Result Serialization:    Round-trip OK (740 bytes JSON)
6. Orchestrator Lifecycle:  Preflight fail path: OK
7. Skip Case Handling:      Whisper skip: OK
8. Artifact Output:         case.json + env_fingerprint.json + result.json: OK
9. Pytest Collection:       48 tests collected in 0.02s
============================================================
```

---

## Architecture Delivered

```
tests/test_e2e.py                          <- Single parametrized entrypoint (48 models)
tests/e2e_harness/
  __init__.py                              <- Package init, exports key types
  contracts.py                             <- Domain types + protocols (DIP core)
  registry.py                              <- Auto-discovery for runners/comparators/references
  manifest_loader.py                       <- Load v1/v2 manifests, infer v2 fields
  orchestrator.py                          <- Lifecycle coordinator (preflight->build->run->compare->artifact)
  artifact_sink.py                         <- Structured artifact persistence
  result_schema.py                         <- Result JSON serialization/deserialization
  runners/
    __init__.py
    text_generation.py                     <- text_generation_causal (decoder_kv_cache, moe, ssm, rwkv, hybrid)
    vision_language.py                     <- vision_language_generation
    segmentation.py                        <- segmentation + prompted_segmentation
    object_detection.py                    <- object_detection
    diffusion.py                           <- diffusion_media_generation (composite stage graph)
    audio_speech.py                        <- speech_to_text, text_to_audio, speech_to_speech
    embedding.py                           <- embedding
    reranking.py                           <- reranking
    encoder_only.py                        <- encoder_only_nlp
    neural_operator.py                     <- neural_operator
    omni.py                                <- omni_multimodal, composite_pipeline
  comparators/
    __init__.py
    text.py                                <- 6 metrics: cosine, rel_l2, stable_top1, topk, token_agree, edit_dist
    vision_language.py                     <- Vision embed cosine + text metrics + semantic similarity
    segmentation.py                        <- mIoU, pixel accuracy, boundary F-score + prompted + detection
    diffusion.py                           <- Latent trajectory, PSNR/SSIM/LPIPS, temporal consistency
    speech_to_text.py                      <- Token agreement, WER/CER, timestamp sanity
    text_to_audio.py                       <- Codec match, mel distance, spectral distance, RMS
    speech_to_speech.py                    <- Token match, frame exact, RMS floor
    embedding.py                           <- Cosine similarity, top-k neighborhood overlap
    reranking.py                           <- Pairwise ordering, Kendall tau, Spearman rho
    encoder_only.py                        <- Hidden state cosine, CLS embedding cosine
    neural_operator.py                     <- Relative L2, PDE residual
    omni.py                                <- Multi-branch: thinker, vision, audio, talker, code2wav
  references/
    __init__.py
    hf_transformers.py                     <- HF transformers reference (L1 oracle)
    hf_diffusers.py                        <- HF diffusers reference (L1 oracle)
    torch_reference.py                     <- Custom torch reference (L2 oracle)
    custom_python.py                       <- Custom Python script reference (L2 oracle)
    golden_snapshot.py                     <- Golden snapshot reference (L3 oracle)
    invariant_only.py                      <- Invariant checks only (L4 oracle)
  thresholds/
    __init__.py                            <- Threshold loading + resolution chain
    defaults/                              <- 15 strategy-specific threshold profiles
      text_generation_causal.json
      vision_language_generation.json
      speech_to_text.json
      text_to_audio.json
      speech_to_speech.json
      segmentation.json
      prompted_segmentation.json
      object_detection.json
      diffusion_media_generation.json
      embedding.json
      reranking.json
      encoder_only_nlp.json
      neural_operator.json
      omni_multimodal.json
      composite_pipeline.json
    overrides/                             <- Per-model threshold overrides (empty, ready for use)
scripts/
  migrate_e2e_manifest_v2.py              <- Automated v1->v2 manifest migrator
```

---

## Strategy Coverage Matrix

| Task Strategy | Runner | Comparator | Reference | Models | Status |
|---|---|---|---|---|---|
| text_generation_causal | TextGenerationCausalRunner | TextComparator | hf_transformers | 34 | COMPLETE |
| vision_language_generation | VisionLanguageRunner | VisionLanguageComparator | hf_transformers | 4 | COMPLETE |
| speech_to_text | SpeechToTextRunner | SpeechToTextComparator | hf_transformers | 1 | COMPLETE |
| text_to_audio | TextToAudioRunner | TextToAudioComparator | hf_transformers | 2 | COMPLETE |
| speech_to_speech | SpeechToSpeechRunner | SpeechToSpeechComparator | torch_reference | 1 | COMPLETE |
| segmentation | SegmentationRunner | SegmentationComparator | hf_transformers | 1 | COMPLETE |
| prompted_segmentation | PromptedSegmentationRunner | PromptedSegmentationComparator | hf_transformers | 1 | COMPLETE |
| object_detection | ObjectDetectionRunner | ObjectDetectionComparator | hf_transformers | 0 | COMPLETE |
| diffusion_media_generation | DiffusionMediaRunner | DiffusionComparator | hf_diffusers | 1 | COMPLETE |
| embedding | EmbeddingRunner | EmbeddingComparator | hf_transformers | 1 | COMPLETE |
| reranking | RerankingRunner | RerankingComparator | hf_transformers | 1 | COMPLETE |
| encoder_only_nlp | EncoderOnlyRunner | EncoderOnlyComparator | hf_transformers | 1 | COMPLETE |
| neural_operator | NeuralOperatorRunner | NeuralOperatorComparator | torch_reference | 0 | COMPLETE |
| omni_multimodal | OmniMultimodalRunner | OmniComparator | torch_reference | 0 | COMPLETE |

**Total: 14/14 strategies covered, 48/48 models matched**

---

## Usage

```bash
# Single model:
pytest tests/test_e2e.py::test_e2e[qwen3-0.6b]

# All models:
pytest tests/test_e2e.py

# Filter by strategy:
pytest tests/test_e2e.py --e2e-task-strategy text_generation_causal

# With artifacts:
pytest tests/test_e2e.py --e2e-artifacts-dir /tmp/artifacts

# With engine dir and binary:
pytest tests/test_e2e.py --engine-dir /mnt/storage/engines --trtf-binary ./build/trtf

# Migrate manifests to v2:
python scripts/migrate_e2e_manifest_v2.py --dry-run
python scripts/migrate_e2e_manifest_v2.py --output-dir /tmp/v2_manifests
```

---

## Design Principles Achieved

1. **Dependency Inversion**: Orchestrator depends only on contracts (protocols). All concrete implementations are pluggable.
2. **Open/Closed**: New modality = new runner + comparator + threshold file. No orchestrator changes.
3. **Single Responsibility**: Each module has one job (loader, runner, comparator, sink).
4. **Interface Segregation**: Strategy-specific interfaces — text models don't implement vision methods.
5. **Liskov Substitution**: Any reference backend is substitutable. Any comparator is swappable.
6. **Confidence Semantics**: Every result records oracle_level (L1-L4) and full metric details.
7. **No Silent Skips**: Unmet requirements are explicit PRECHECK_FAIL, not hidden skips.

---

## Onboarding a New Model Family

1. Implement family plugin (existing flow in `trtf_build/families/`)
2. Add one manifest JSON in `tests/e2e/models/<name>.json` with runtime_strategy
3. Run: `pytest tests/test_e2e.py::test_e2e[<name>]`

The framework auto-infers task_strategy, reference_backend, oracle_level, stages, and thresholds.
No new test file needed. No new runner needed (unless introducing a new runtime_strategy).

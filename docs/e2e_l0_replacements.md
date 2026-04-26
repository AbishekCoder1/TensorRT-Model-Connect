# E2E L0 Replacements

MR L0 may replace a large model with a smaller representative only when the
replacement preserves the execution contract:

- the source model takes at least about 10 minutes in the nightly E2E report
- same manifest `family`
- same `runtime_strategy`
- same precision and quantization fields
- same E2E runner, comparator, and artifact contract through the derived task
  strategy
- the large model mainly adds scale, memory-pressure, or long-build coverage
- when no true smaller checkpoint exists, an MR-only representative may keep the
  same checkpoint and reduce cache length, max generated tokens, frame count, or
  inference step count, as long as it preserves the same plugin path and
  artifact type

Nightly E2E still runs every model. MR L0 applies the configured replacement
even for direct changes to a nightly-only model's manifest or model-specific
E2E data; nightly remains the validation lane for the large model's scale.

## Nightly Run Findings

The 2026-04-23 nightly job 303847181 wrote an HTML report with 88 model
results. The current manifest set has 107 manifests after adding MR-only L0
representatives. Nightly CI excludes `ci_tier: "l0_only"` representatives, so
the scheduled suite keeps the full source models without paying extra nightly
time for the smaller repros. Per-model timings below use unique phase time from
each `result.json`, excluding the duplicate `build_s` entry because the report
records both `build_s` and `bundle_build_s`.

| Model | Unique phase time | Decision |
| --- | ---: | --- |
| `qwen3-moe-30b-a3b` | 62.8 min | Replace with `qwen3-moe-tiny-random` in L0. |
| `flux-2-dev-fp8` | 46.6 min | Replace with `flux-2-dev-fp8-l0` in L0 at the same 1024px size with an end-to-end-only stage, fewer denoising steps, and shorter text shape. |
| `flux-2-dev` | 40.7 min | Replace with `flux-2-dev-l0` in L0 at the same 1024px size with an end-to-end-only stage, fewer denoising steps, and shorter text shape. |
| `wan21-t2v-1.3b` | 39.5 min | Replace with `wan21-t2v-1.3b-l0` in L0 at the same 480x832 size with fewer frames, fewer denoising steps, and shorter text cache. |
| `flux-schnell` | 28.4 min | Replace with `flux-schnell-l0` in L0 at the same 1024px size with fewer denoising steps and shorter text cache; the earlier 512px repro was removed. |
| `deepseek-v2-lite` | 25.2 min | Replace with `deepseek-v2-tiny` in L0. |
| `personaplex-7b` | 24.1 min | Replace with `personaplex-7b-l0` in L0 using the same checkpoint with shorter cache and speech frame budgets. |
| `gpt-oss-20b` | 20.2 min | Replace with `gpt-oss-20b-l0` in L0 using the same checkpoint with shorter cache and decode budgets. |
| `z-image-turbo` | 20.0 min | Replace with `z-image-turbo-l0` in L0 at the same 1024px size with fewer denoising steps and shorter text cache. |
| `glm-4-9b` | 15.4 min | Replace with `glm-4-9b-l0` in L0 using the same checkpoint with shorter cache and decode budgets. |
| `internvl3-8b` | 12.1 min | Replace with `internvl3-2b` in L0. |
| `phi-moe` | 12.1 min | Replace with `phi-moe-l0` in L0 using the same tiny MoE checkpoint with shorter cache and decode budgets. |
| `mistral-7b` | 11.4 min | Replace with `mistral-7b-l0` in L0 using the same checkpoint with shorter cache and decode budgets. |
| `pixart-sigma-1024-torchtrt` | 11.0 min | Keep in L0 when affected; it uniquely covers the Torch-TRT PixArt path. |
| `qwen25vl-3b` | 10.8 min | Keep in L0 when affected; `qwen3-vl-2b` is a different Qwen-VL generation and not a strict replacement. |
| `bark-large` | 10.1 min | Replace with `bark-small` in L0. |
| `deepseek-ocr` | 10.1 min | Replace with `deepseek-ocr-l0` in L0 using the same checkpoint with shorter cache and decode budgets. |
| `magpie-tts-357m` | 10.0 min | Keep in L0 when affected; unique Magpie TTS path and already small. |

## L0 Replacement Set

If a broad MR change selects every E2E model, these replacements reduce the MR
L0 list from 107 manifests to 90 test IDs. Nightly excludes the MR-only
representatives and still runs the large nightly-only source models.

| Nightly-only model | MR L0 replacement | Why this is acceptable |
| --- | --- | --- |
| `bark-large` | `bark-small` | Same `bark` family and `text_to_audio_bark` runtime; large model adds TTS scale coverage and took about 10 minutes in the nightly report. |
| `deepseek-ocr` | `deepseek-ocr-l0` | Same checkpoint, `deepseek_ocr` family, `vision_language` runtime, image input, invariant comparator, and OCR text artifact contract; the L0 repro reduces cache length from 4096 to 2048 and max generated tokens from 300 to 80. |
| `deepseek-v2-lite` | `deepseek-v2-tiny` | Same `deepseek_v2` family and `decoder_kv_cache` runtime; tiny manifest is already the CI repro and the lite model took about 25 minutes in the nightly report. |
| `flux-2-dev` | `flux-2-dev-l0` | Same checkpoint, `flux.2` model type, Mistral 3 text encoder, Flux2 DiT, 32-channel VAE, `diffusion_flux` runtime, runner, comparator, and image artifact contract; the L0 repro keeps 1024px output, compiles the text encoder and DiT for a 24-token prompt shape, runs only the required end-to-end stage, and reduces denoising from 28 to 4 steps. |
| `flux-2-dev-fp8` | `flux-2-dev-fp8-l0` | Same checkpoint, FP8 scale file, `flux.2` model type, Mistral 3 text encoder, Flux2 DiT, 32-channel VAE, `diffusion_flux` runtime, runner, comparator, and image artifact contract; the L0 repro keeps 1024px output, compiles the text encoder and DiT for a 24-token prompt shape, runs only the required end-to-end stage, and reduces denoising from 28 to 4 steps. |
| `flux-schnell` | `flux-schnell-l0` | Same checkpoint, `flux` model type, CLIP+T5 encoders, FluxDiT, VAE, `diffusion_flux` runtime, runner, comparator, and image artifact contract; the L0 repro keeps 1024px output, runs only the required end-to-end stage, reduces text cache from 256 to 128, and reduces denoising from 4 to 2 steps. |
| `glm-4-9b` | `glm-4-9b-l0` | Same checkpoint, `glm` family, `decoder_kv_cache` runtime, runner, comparator, and continuation artifact contract; the L0 repro reduces cache length from 256 to 128 and max generated tokens from 20 to 8. |
| `gpt-oss-20b` | `gpt-oss-20b-l0` | Same checkpoint, `gpt_oss` family, `decoder_moe` runtime, runner, comparator, and chat artifact contract; the L0 repro reduces cache length from 256 to 128 and max generated tokens from 20 to 8. |
| `internvl3-8b` | `internvl3-2b` | Same `internvl` family, `vision_language` runtime, InternVL model type, Qwen2 text decoder, InternViT vision path, comparator, and image/text artifact contract; 8B mainly adds scale coverage and took about 12 minutes in the nightly report. |
| `minitron-4b-width` | `minitron-4b-width-l0` | Same checkpoint, Llama-family Minitron width-pruned architecture, `decoder_kv_cache` runtime, runner, comparator, and continuation artifact contract; the L0 repro reduces cache length from 256 to 128 and max generated tokens from 20 to 8. |
| `mistral-7b` | `mistral-7b-l0` | Same checkpoint, `mistral` family, `decoder_kv_cache` runtime, runner, comparator, and chat artifact contract; the L0 repro reduces cache length from 256 to 128 and max generated tokens from 10 to 5. |
| `personaplex-7b` | `personaplex-7b-l0` | Same checkpoint, `personaplex` family, `speech_to_speech` runtime, official token snapshot, runner, comparator, and speech artifact contract; the L0 repro reduces cache length from 512 to 256, speech tail frames from 300 to 100, and max token budget from 1000 to 300. |
| `phi-moe` | `phi-moe-l0` | Same checkpoint, `phi_moe` family, `decoder_kv_cache` runtime, runner, comparator, and chat artifact contract; the L0 repro reduces cache length from 256 to 128 and max generated tokens from 10 to 5. |
| `pixart-sigma-1024` | `pixart-sigma-1024-l0` | Same checkpoint, `pixart` model type, T5-XXL text encoder, DiT/VAE path, `diffusion_pixart` runtime, runner, comparator, and image artifact contract; the L0 repro keeps 1024px output, runs only the required end-to-end stage, reduces text cache from 256 to 128, and reduces denoising from 20 to 4 steps. |
| `qwen3-moe-30b-a3b` | `qwen3-moe-tiny-random` | Same `qwen_moe` family, `decoder_moe` runtime, `qwen3_moe` model type, per-head Q/K norm path, router/expert mapping, normalized top-k routing, runner, comparator, and text artifact contract using `amd-quark/tiny-random-qwen3_moe`. The source model remains nightly coverage for 128-expert/top-8 scale. |
| `wan21-t2v-1.3b` | `wan21-t2v-1.3b-l0` | Same checkpoint, `wan_t2v` model type, T5 encoder, Wan DiT, causal 3D VAE, `diffusion_wan` runtime, runner, comparator, and video artifact contract; the L0 repro keeps 480x832 output, runs only the required end-to-end stage, reduces text cache from 256 to 128, reduces frames from 17 to 5, and reduces denoising from 30 to 8 steps. |
| `z-image-turbo` | `z-image-turbo-l0` | Same checkpoint, `z_image` model type, Qwen3 text encoder, ZImage DiT/VAE path, `diffusion_zimage` runtime, runner, comparator, and image artifact contract; the L0 repro keeps 1024px output, runs only the required end-to-end stage, reduces text cache from 256 to 128, and reduces denoising from 4 to 2 steps. |

## Models Kept In L0 When Affected

These large models are not replaced because there is no smaller model that
preserves the guideline cleanly:

- `qwen25vl-3b` and `qwen3-vl-2b`: no smaller Qwen VL representative in the
  current manifests.
- `riva-translate-4b`: no smaller translation-chat representative in the current
  manifests.
- `nemotron-mini-4b`, `nemotron-hindi-4b`, `nemotron-h-nano-9b`, `qwen35-9b`,
  `internlm2-1.8b`, `stablelm2-1.6b`, and `starcoder2-3b`: no smaller
  same-family representative is currently defined.
- `minitron-4b-depth`, `nemotron-nano-4b`, `qwen3-4b-instruct-2507`,
  `roberta-large`, and `whisper-large-v3-turbo`:
  each took less than about 10 minutes in unique phase time in the nightly
  report, so replacing them is not worth the loss in model-specific signal.
  The Minitron/Nemotron Llama-family models also should not be collapsed to
  `tinyllama-1.1b` because that loses the architecture/config variant the
  manifest is meant to cover.

## Implementation

The replacement is declared in the large model manifest with:

- `ci_tier: "nightly_only"`
- `l0_replacement: "<smaller-model>"`
- `l0_replacement_reason: "<why the replacement preserves confidence>"`

MR-only representatives use `ci_tier: "l0_only"`. Nightly CI passes
`--exclude-ci-tier l0_only` to both cache warming and full test collection so
the scheduled suite does not spend time on reduced-workload repros.

`tools/test_impact.py` applies these replacements by default for
`--e2e-suite l0`. Use `--e2e-suite nightly` to keep the exact impacted model
list.

For FLUX.2, the L0 `build_args.max_cache_length` is also used as the compiled
text sequence length. This does not change the image resolution or Flux2
architecture; it narrows the fixed Mistral encoder and DiT text tensor shape to
the short L0 prompt while the nightly source manifests continue to exercise the
full text shape and full diagnostic stage suite.

For L0-only diffusion representatives, `stages` is explicitly narrowed to the
required `end_to_end` stage when the default component stages only duplicate the
same component path. The full source manifests remain nightly-only and keep the
diagnostic `t5_encode`, `dit_step`, and `vae_decode` stages for component-level
triage.

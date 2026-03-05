# Audio and Speech Backends

Speech/audio generation and transcription runtimes.

Key files:
- `whisper_backend.*`: speech-to-text.
- `bark_backend.*`: text-to-audio (Bark).
- `magpie_tts_backend.*`: Magpie TTS generation.
- `omni_backend.*`: omni multimodal audio path.
- `speech_backend.*`: speech-to-speech pipeline.
- `mel_spectrogram.*`: mel feature extraction helpers.

How to understand:
1. Enter the backend class `generate_*`/`transcribe`/`speak` API.
2. Trace prefill/decode helper stages and codec/post-processing.
3. Use `core/device_kv_cache.*` and `core/trt_decode_runtime.*` for shared decode behavior.

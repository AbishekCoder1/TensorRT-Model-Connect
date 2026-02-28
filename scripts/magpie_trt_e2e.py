#!/usr/bin/env python3
"""MagpieTTS full E2E test: TRT encoder + TRT decoder + NanoCodec -> WAV.

Steps:
  1. Build TRT encoder and decoder engines from NeMo weights
  2. Run TRT encoder on reference text tokens, verify vs reference
  3. Run TRT decoder autoregressively (context prefill + audio generation)
  4. Decode codec tokens via NanoCodec bridge -> WAV
  5. Compare against reference audio
"""

import json
import os
import struct
import subprocess
import sys
import time
import wave

import numpy as np

# ---- paths ----
NEMO_PATH = "/tmp/magpie_tts/magpie_tts_multilingual_357m.nemo"
CODEC_PATH = "/tmp/nanocodec/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo"
REF_TOKENS = "/tmp/ref_text_tokens.npy"
REF_ENCODER = "/tmp/ref_encoder_output.npy"
REF_CODES = "/tmp/ref_decoder_codes.npy"
REF_CONTEXT = "/tmp/ref_context_embedding.npy"
REF_AUDIO = "/tmp/ref_audio.wav"
OUT_CODES_NPY = "/tmp/trt_decoder_codes.npy"
OUT_AUDIO = "/tmp/trt_audio_fixed.wav"
CODEC_BRIDGE = "/workspace/trt-transformers-cpp/scripts/magpie_codec_bridge.py"
VENV_PYTHON = "/workspace/trt-transformers-cpp/.venv/bin/python"

MAX_CACHE_LENGTH = 512
MAX_AUDIO_FRAMES = 300  # safety limit

# ---- CUDA helpers ----
import tensorrt as trt
try:
    from cuda import cudart
except ImportError:
    from cuda.bindings import runtime as cudart


def cuda_check(ret):
    """Check CUDA return value. Handles both (err,) tuples and plain err."""
    if isinstance(ret, tuple):
        err = ret[0]
    else:
        err = ret
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")


def cuda_malloc(nbytes):
    err, ptr = cudart.cudaMalloc(nbytes)
    cuda_check(err)
    return ptr


def cuda_free(ptr):
    cuda_check(cudart.cudaFree(ptr))


def cuda_memcpy_h2d(dst, src_np):
    """Copy numpy array to device pointer."""
    src = np.ascontiguousarray(src_np)
    cuda_check(cudart.cudaMemcpy(
        dst, src.ctypes.data, src.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


def cuda_memcpy_d2h(dst_np, src, nbytes):
    """Copy device pointer to numpy array."""
    cuda_check(cudart.cudaMemcpy(
        dst_np.ctypes.data, src, nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


def cuda_memset(ptr, value, nbytes):
    cuda_check(cudart.cudaMemset(ptr, value, nbytes))


# ---- Step 1: Build TRT engines ----
ENCODER_PLAN_CACHE = "/tmp/magpie_encoder.plan"
DECODER_PLAN_CACHE = "/tmp/magpie_decoder.plan"


def build_engines():
    """Build encoder and decoder TRT engines from NeMo weights.

    Caches engine plans to /tmp for reuse across runs (avoids TRT
    non-determinism from rebuilding).
    """
    print("=" * 60)
    print("STEP 1: Building TRT engines from NeMo weights")
    print("=" * 60)

    # Add trtf_build to path
    sys.path.insert(0, "/workspace/trt-transformers-cpp/trtf_build")
    from trtf_build.families.magpie_tts import MagpieTTSPlugin
    from trtf_build.config import ModelConfig

    plugin = MagpieTTSPlugin()

    # Create a minimal config
    config = ModelConfig.__new__(ModelConfig)
    config.hidden_size = 768
    config.num_hidden_layers = 12
    config.num_attention_heads = 12

    print(f"Loading weights from {NEMO_PATH} ...")
    t0 = time.time()
    weights = plugin.load_weights(NEMO_PATH, config)
    print(f"  Weights loaded in {time.time()-t0:.1f}s")

    # Print key metadata
    for k in sorted(weights.keys()):
        if k.startswith("_"):
            print(f"  {k} = {weights[k]}")

    # Build or load cached encoder
    if os.path.exists(ENCODER_PLAN_CACHE):
        print(f"\nLoading cached encoder from {ENCODER_PLAN_CACHE}")
        with open(ENCODER_PLAN_CACHE, "rb") as f:
            encoder_plan = f.read()
        print(f"  Encoder loaded, plan size={len(encoder_plan)//1024}KB")
    else:
        print("\nBuilding TRT encoder ...")
        t0 = time.time()
        encoder_plan = plugin.build_vision_engine(NEMO_PATH, config, weights, verbose=True)
        print(f"  Encoder built in {time.time()-t0:.1f}s, plan size={len(encoder_plan)//1024}KB")
        with open(ENCODER_PLAN_CACHE, "wb") as f:
            f.write(encoder_plan)

    # Build or load cached decoder
    if os.path.exists(DECODER_PLAN_CACHE):
        print(f"\nLoading cached decoder from {DECODER_PLAN_CACHE}")
        with open(DECODER_PLAN_CACHE, "rb") as f:
            decoder_plan = f.read()
        print(f"  Decoder loaded, plan size={len(decoder_plan)//1024}KB")
    else:
        print(f"\nBuilding TRT decoder (max_cache_length={MAX_CACHE_LENGTH}) ...")
        t0 = time.time()
        decoder_plan = plugin.build_engine(config, weights, max_cache_length=MAX_CACHE_LENGTH, verbose=True)
        print(f"  Decoder built in {time.time()-t0:.1f}s, plan size={len(decoder_plan)//1024}KB")
        with open(DECODER_PLAN_CACHE, "wb") as f:
            f.write(decoder_plan)

    return encoder_plan, decoder_plan, weights


# ---- Step 2: Run TRT encoder ----
def run_encoder(encoder_plan, ref_tokens):
    """Run TRT encoder and return encoder_output [max_pos, hidden]."""
    print("\n" + "=" * 60)
    print("STEP 2: Running TRT encoder")
    print("=" * 60)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(encoder_plan)
    context = engine.create_execution_context()

    # Get dimensions from engine
    max_pos = engine.get_tensor_shape("input_ids")[0]
    hidden = engine.get_tensor_shape("encoder_output")[-1]
    print(f"  Encoder: max_pos={max_pos}, hidden={hidden}")

    # Prepare input: pad tokens to max_pos
    num_tokens = len(ref_tokens)
    input_ids = np.zeros(max_pos, dtype=np.int32)
    input_ids[:num_tokens] = ref_tokens.astype(np.int32)
    print(f"  Input: {num_tokens} tokens, padded to {max_pos}")

    # Allocate device memory
    input_ids_d = cuda_malloc(input_ids.nbytes)
    out_shape = tuple(engine.get_tensor_shape("encoder_output"))
    encoder_output = np.zeros(out_shape, dtype=np.float32)
    encoder_output_d = cuda_malloc(encoder_output.nbytes)

    # Copy input
    cuda_memcpy_h2d(input_ids_d, input_ids)

    # Set tensor addresses
    context.set_tensor_address("input_ids", input_ids_d)
    context.set_tensor_address("encoder_output", encoder_output_d)

    # Execute
    err, stream = cudart.cudaStreamCreate()
    cuda_check(err)

    t0 = time.time()
    ok = context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    elapsed = time.time() - t0
    print(f"  Encoder executed in {elapsed*1000:.1f}ms, success={ok}")

    # Read output
    cuda_memcpy_d2h(encoder_output, encoder_output_d, encoder_output.nbytes)

    # Cleanup
    cuda_free(input_ids_d)
    cuda_free(encoder_output_d)
    cudart.cudaStreamDestroy(stream)

    # encoder_output shape is [max_pos, hidden]
    print(f"  Output shape: {encoder_output.shape}")
    print(f"  Output stats: mean={encoder_output.mean():.6f}, std={encoder_output.std():.6f}")

    return encoder_output, num_tokens


def verify_encoder(encoder_output, num_tokens):
    """Compare TRT encoder output against NeMo reference."""
    print("\n  Verifying encoder against reference ...")
    ref = np.load(REF_ENCODER)  # [1, 47, 768]
    if ref.ndim == 3:
        ref = ref[0]  # [47, 768]

    trt_slice = encoder_output[:num_tokens]  # [47, 768]
    cos = np.dot(trt_slice.ravel(), ref.ravel()) / (
        np.linalg.norm(trt_slice.ravel()) * np.linalg.norm(ref.ravel()) + 1e-12)
    max_err = np.max(np.abs(trt_slice - ref))
    mean_err = np.mean(np.abs(trt_slice - ref))
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  Max abs error: {max_err:.6e}")
    print(f"  Mean abs error: {mean_err:.6e}")
    if cos < 0.999:
        print("  WARNING: Encoder parity is low!")
    else:
        print("  PASS: Encoder matches reference")
    return cos


# ---- Step 3: Run TRT decoder (autoregressive) ----
def run_decoder(decoder_plan, encoder_output, num_tokens, weights):
    """Run TRT decoder autoregressively with context prefill + audio generation."""
    print("\n" + "=" * 60)
    print("STEP 3: Running TRT decoder (autoregressive)")
    print("=" * 60)

    # Extract metadata
    hidden = int(weights["_hidden_size"])
    dec_layers = int(weights["_dec_layers"])
    num_codebooks = int(weights["_num_codebooks"])
    codebook_size = int(weights["_codebook_size"])
    max_source_positions = int(weights["_max_source_positions"])
    attention_window = MAX_CACHE_LENGTH + 1

    print(f"  hidden={hidden}, dec_layers={dec_layers}, codebooks={num_codebooks}")
    print(f"  codebook_size={codebook_size}, max_source={max_source_positions}")
    print(f"  max_cache_length={MAX_CACHE_LENGTH}, attention_window={attention_window}")

    # Load audio embeddings
    audio_embeds = []
    for cb in range(num_codebooks):
        key = f"audio_embedding_{cb}"
        e = np.asarray(weights[key], dtype=np.float32)
        audio_embeds.append(e)
        if cb == 0:
            print(f"  Audio embedding shape: {e.shape}")

    # Load baked context
    ref_context = np.load(REF_CONTEXT)  # [1, 110, 768]
    if ref_context.ndim == 3:
        ref_context = ref_context[0]  # [110, 768]
    ctx_frames = ref_context.shape[0]
    print(f"  Context frames: {ctx_frames}")

    # Setup TRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(decoder_plan)
    context = engine.create_execution_context()

    # Print engine I/O info
    print(f"\n  Engine I/O tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        mode = engine.get_tensor_mode(name)
        print(f"    {name}: {shape} ({'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'})")

    # ---- Allocate device memory ----
    # Inputs
    input_embed_d = cuda_malloc(1 * hidden * 4)
    position_id_d = cuda_malloc(1 * 4)
    attention_mask_d = cuda_malloc(1 * attention_window * 4)

    # KV cache: [max_cache_length, hidden] per layer
    cache_k_d = []
    cache_v_d = []
    cache_bytes = MAX_CACHE_LENGTH * hidden * 4
    for i in range(dec_layers):
        ck = cuda_malloc(cache_bytes)
        cv = cuda_malloc(cache_bytes)
        cuda_memset(ck, 0, cache_bytes)
        cuda_memset(cv, 0, cache_bytes)
        cache_k_d.append(ck)
        cache_v_d.append(cv)

    # Cross-attention inputs: [max_source_positions, hidden]
    cross_bytes = max_source_positions * hidden * 4
    cross_k_d = []
    cross_v_d = []

    # Prepare encoder output padded to max_source_positions.
    # Only use the first num_tokens rows (real encoder output); zero-pad
    # the rest.  The TRT encoder produces non-zero junk for padded input
    # positions, so we must NOT copy those into cross-attention inputs.
    enc_padded = np.zeros((max_source_positions, hidden), dtype=np.float32)
    enc_padded[:num_tokens] = encoder_output[:num_tokens]

    for i in range(dec_layers):
        ck = cuda_malloc(cross_bytes)
        cv = cuda_malloc(cross_bytes)
        cuda_memcpy_h2d(ck, enc_padded)
        cuda_memcpy_h2d(cv, enc_padded)
        cross_k_d.append(ck)
        cross_v_d.append(cv)

    # Output: logits [1, num_codebooks * codebook_size]
    output_size = num_codebooks * codebook_size
    logits_d = cuda_malloc(1 * output_size * 4)

    # Present KV outputs: [1, hidden]
    present_k_d = []
    present_v_d = []
    present_bytes = 1 * hidden * 4
    for i in range(dec_layers):
        pk = cuda_malloc(present_bytes)
        pv = cuda_malloc(present_bytes)
        present_k_d.append(pk)
        present_v_d.append(pv)

    # Create CUDA stream
    err, stream = cudart.cudaStreamCreate()
    cuda_check(err)

    # ---- Helper: set tensor addresses ----
    def set_io():
        context.set_tensor_address("input_embed", input_embed_d)
        context.set_tensor_address("position_id", position_id_d)
        context.set_tensor_address("attention_mask", attention_mask_d)
        context.set_tensor_address("logits", logits_d)

        for i in range(dec_layers):
            context.set_tensor_address(f"cache_k_{i}", cache_k_d[i])
            context.set_tensor_address(f"cache_v_{i}", cache_v_d[i])
            context.set_tensor_address(f"cross_k_{i}", cross_k_d[i])
            context.set_tensor_address(f"cross_v_{i}", cross_v_d[i])
            context.set_tensor_address(f"present_k_{i}", present_k_d[i])
            context.set_tensor_address(f"present_v_{i}", present_v_d[i])

    set_io()

    # ---- Helper: build attention mask for position pos ----
    # The mask is [1, attention_window] where attention_window = max_cache_length + 1
    # Layout: [cache_0, cache_1, ..., cache_{max_cache-1}, current_token]
    # 0.0 for positions we can attend to, -1e9 for masked positions
    # Cache stores entries starting from index 0, so at step pos we have
    # filled entries at cache[0..min(pos, max_cache)-1].
    def build_mask(pos):
        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        filled = min(pos, MAX_CACHE_LENGTH)
        # Unmask the first `filled` cache entries (indices 0..filled-1)
        mask[0, :filled] = 0.0
        # Unmask the current token (last element of the window)
        mask[0, MAX_CACHE_LENGTH] = 0.0
        return mask

    # ---- Helper: update KV cache (shift + append) ----
    # The present_k/v output is [1, hidden] for the current step
    # We need to shift the cache left by 1 and append the new entry
    def update_kv_cache(pos):
        """Read present_k/v from device, update cache on device."""
        for i in range(dec_layers):
            if pos < MAX_CACHE_LENGTH:
                # Cache not full: just copy present to cache[pos]
                # present_k_d[i] has [1, hidden], copy to cache_k_d[i] at offset pos*hidden*4
                offset = pos * hidden * 4
                cudart.cudaMemcpyAsync(
                    cache_k_d[i] + offset, present_k_d[i], hidden * 4,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)
                cudart.cudaMemcpyAsync(
                    cache_v_d[i] + offset, present_v_d[i], hidden * 4,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)
            else:
                # Cache full: shift left by 1, append at end
                # Copy cache[1:] -> cache[0:]
                shift_bytes = (MAX_CACHE_LENGTH - 1) * hidden * 4
                cudart.cudaMemcpyAsync(
                    cache_k_d[i], cache_k_d[i] + hidden * 4, shift_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)
                cudart.cudaMemcpyAsync(
                    cache_v_d[i], cache_v_d[i] + hidden * 4, shift_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)
                # Append present at the end
                end_offset = (MAX_CACHE_LENGTH - 1) * hidden * 4
                cudart.cudaMemcpyAsync(
                    cache_k_d[i] + end_offset, present_k_d[i], hidden * 4,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)
                cudart.cudaMemcpyAsync(
                    cache_v_d[i] + end_offset, present_v_d[i], hidden * 4,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)

    # ---- Phase 1: Context prefill ----
    print(f"\n  Phase 1: Context prefill ({ctx_frames} frames)")
    t0 = time.time()

    for pos in range(ctx_frames):
        # input_embed = context embedding at this position
        input_embed = ref_context[pos:pos+1].copy()  # [1, 768]
        position_id = np.array([pos], dtype=np.int32)
        mask = build_mask(pos)

        cuda_memcpy_h2d(input_embed_d, input_embed)
        cuda_memcpy_h2d(position_id_d, position_id)
        cuda_memcpy_h2d(attention_mask_d, mask)

        ok = context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)
        if not ok:
            raise RuntimeError(f"Decoder execution failed at context step {pos}")

        # Update KV cache
        update_kv_cache(pos)

    elapsed = time.time() - t0
    print(f"  Context prefill done in {elapsed:.2f}s ({elapsed/ctx_frames*1000:.1f}ms/step)")

    # ---- Phase 2: Autoregressive audio generation ----
    print(f"\n  Phase 2: Autoregressive audio generation")
    BOS_TOKEN = 2016
    EOS_TOKEN = 2017
    AUDIO_RANGE = 2016  # only sample from audio tokens [0..2015]
    # Greedy decoding for deterministic output and reliable EOS detection.
    # Temperature sampling with numpy RNG diverges from torch RNG used by
    # the NeMo reference, causing the autoregressive path to never hit EOS.
    TEMPERATURE = 0.0
    TOP_K = 80
    MIN_GENERATED_FRAMES = 4
    MAX_DURATION_FRAMES = 200  # safety limit: ~9.3s at 21.5 fps

    # Initialize previous codes to BOS for all codebooks
    prev_codes = np.full(num_codebooks, BOS_TOKEN, dtype=np.int64)
    all_codes = []  # list of [num_codebooks] arrays

    t0 = time.time()
    for frame in range(MAX_AUDIO_FRAMES):
        pos = ctx_frames + frame

        # Compute input_embed: average of 8 codebook embeddings for prev_codes
        embed_sum = np.zeros((1, hidden), dtype=np.float32)
        for cb in range(num_codebooks):
            token_id = int(prev_codes[cb])
            embed_sum += audio_embeds[cb][token_id:token_id+1]
        input_embed = embed_sum / num_codebooks  # average

        position_id = np.array([pos], dtype=np.int32)
        mask = build_mask(pos)

        cuda_memcpy_h2d(input_embed_d, input_embed)
        cuda_memcpy_h2d(position_id_d, position_id)
        cuda_memcpy_h2d(attention_mask_d, mask)

        ok = context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)
        if not ok:
            raise RuntimeError(f"Decoder execution failed at audio step {frame}")

        # Read logits
        logits_host = np.zeros((1, output_size), dtype=np.float32)
        cuda_memcpy_d2h(logits_host, logits_d, logits_host.nbytes)

        # Sample codes for each codebook
        codes = np.zeros(num_codebooks, dtype=np.int64)
        eos_detected = False
        for cb in range(num_codebooks):
            cb_logits_full = logits_host[0, cb * codebook_size:(cb + 1) * codebook_size]

            # Check EOS: if argmax over FULL codebook range [0..2024) is EOS
            if np.argmax(cb_logits_full) == EOS_TOKEN:
                eos_detected = True

            # Only sample from audio token range [0..2016)
            cb_logits = cb_logits_full[:AUDIO_RANGE].copy()

            # Temperature + top-k sampling
            if TEMPERATURE > 0:
                scaled = cb_logits / TEMPERATURE
                # Top-k: zero out everything below the k-th largest
                if TOP_K > 0 and TOP_K < AUDIO_RANGE:
                    threshold = np.partition(scaled, -TOP_K)[-TOP_K]
                    scaled[scaled < threshold] = -1e9
                # Softmax
                scaled -= scaled.max()
                probs = np.exp(scaled) / np.exp(scaled).sum()
                codes[cb] = np.random.choice(AUDIO_RANGE, p=probs)
            else:
                codes[cb] = np.argmax(cb_logits)

        all_codes.append(codes.copy())
        prev_codes = codes

        # Update KV cache
        update_kv_cache(pos)

        # Stop on EOS or duration limit
        if eos_detected and frame >= MIN_GENERATED_FRAMES:
            print(f"  EOS detected at frame {frame}")
            break
        if frame >= MAX_DURATION_FRAMES:
            print(f"  Duration limit reached at frame {frame}")
            break

        if frame % 20 == 0:
            print(f"  Frame {frame}: codes[0]={codes[0]}, "
                  f"logits range=[{logits_host.min():.2f}, {logits_host.max():.2f}]")

    elapsed = time.time() - t0
    num_frames = len(all_codes)
    print(f"\n  Generated {num_frames} frames in {elapsed:.2f}s "
          f"({elapsed/max(num_frames,1)*1000:.1f}ms/step)")

    # Convert to [1, 8, T] array
    codes_array = np.stack(all_codes, axis=-1)  # [8, T]
    codes_array = codes_array[np.newaxis, :]  # [1, 8, T]
    print(f"  Output codes shape: {codes_array.shape}, "
          f"range=[{codes_array.min()}, {codes_array.max()}]")

    # Save codes
    np.save(OUT_CODES_NPY, codes_array)
    print(f"  Saved codes to {OUT_CODES_NPY}")

    # Cleanup
    cuda_free(input_embed_d)
    cuda_free(position_id_d)
    cuda_free(attention_mask_d)
    cuda_free(logits_d)
    for i in range(dec_layers):
        cuda_free(cache_k_d[i])
        cuda_free(cache_v_d[i])
        cuda_free(cross_k_d[i])
        cuda_free(cross_v_d[i])
        cuda_free(present_k_d[i])
        cuda_free(present_v_d[i])
    cudart.cudaStreamDestroy(stream)

    return codes_array


# ---- Step 4: Decode with NanoCodec ----
def decode_codec(codes_array):
    """Decode codec tokens to audio via NanoCodec bridge."""
    print("\n" + "=" * 60)
    print("STEP 4: Decoding with NanoCodec")
    print("=" * 60)

    cmd = [
        VENV_PYTHON, CODEC_BRIDGE,
        "--codec-model", CODEC_PATH,
        "--tokens-npy", OUT_CODES_NPY,
        "--output", OUT_AUDIO,
        "--device", "cuda",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(f"  stdout: {result.stdout.strip()}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(f"  stderr: {line}")
    if result.returncode != 0:
        raise RuntimeError(f"Codec bridge failed with rc={result.returncode}")

    print(f"  Output audio: {OUT_AUDIO}")


# ---- Step 5: Compare ----
def compare_audio():
    """Compare TRT audio against reference."""
    print("\n" + "=" * 60)
    print("STEP 5: Comparing against reference")
    print("=" * 60)

    def read_wav_stats(path):
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            duration = nframes / sr
            rms = np.sqrt(np.mean(pcm ** 2))
            return {"path": path, "sr": sr, "samples": nframes,
                    "duration": duration, "rms": rms, "audio": pcm}

    ref = read_wav_stats(REF_AUDIO)
    trt_out = read_wav_stats(OUT_AUDIO)

    print(f"\n  Reference: {ref['duration']:.3f}s, RMS={ref['rms']:.4f}, "
          f"samples={ref['samples']}")
    print(f"  TRT:       {trt_out['duration']:.3f}s, RMS={trt_out['rms']:.4f}, "
          f"samples={trt_out['samples']}")

    duration_ratio = trt_out["duration"] / ref["duration"]
    print(f"\n  Duration ratio: {duration_ratio:.3f}")

    rms_ratio = trt_out["rms"] / ref["rms"] if ref["rms"] > 0 else float("inf")
    print(f"  RMS ratio: {rms_ratio:.3f}")

    # Compare decoder codes against reference
    ref_codes = np.load(REF_CODES)  # [1, 8, 82]
    trt_codes = np.load(OUT_CODES_NPY)

    ref_c = ref_codes[0] if ref_codes.ndim == 3 else ref_codes  # [8, 82]
    trt_c = trt_codes[0] if trt_codes.ndim == 3 else trt_codes  # [8, T]

    min_frames = min(ref_c.shape[1], trt_c.shape[1])
    if min_frames > 0:
        match = (ref_c[:, :min_frames] == trt_c[:, :min_frames]).mean()
        print(f"\n  Code match (first {min_frames} frames): {match*100:.1f}%")
        # Per-codebook match
        for cb in range(min(8, ref_c.shape[0])):
            cb_match = (ref_c[cb, :min_frames] == trt_c[cb, :min_frames]).mean()
            print(f"    Codebook {cb}: {cb_match*100:.1f}%")

    print(f"\n  Reference frames: {ref_c.shape[1]}, TRT frames: {trt_c.shape[1]}")

    # Overall assessment
    print("\n" + "=" * 60)
    if 0.5 < duration_ratio < 2.0 and trt_out["rms"] > 0.005:
        print("RESULT: Audio generated successfully")
        print(f"  Duration: {trt_out['duration']:.3f}s (ref: {ref['duration']:.3f}s)")
        print(f"  RMS: {trt_out['rms']:.4f} (ref: {ref['rms']:.4f})")
        print(f"  Output: {OUT_AUDIO}")
    else:
        print("RESULT: Audio may have issues")
        print(f"  Duration ratio: {duration_ratio:.3f} (expected ~1.0)")
        print(f"  RMS: {trt_out['rms']:.4f} (expected > 0.005)")
    print("=" * 60)


# ---- Main ----
def main():
    np.random.seed(12345)

    # Step 1: Build engines
    encoder_plan, decoder_plan, weights = build_engines()

    # Step 2: Run encoder
    ref_tokens = np.load(REF_TOKENS)
    encoder_output, num_tokens = run_encoder(encoder_plan, ref_tokens)
    verify_encoder(encoder_output, num_tokens)

    # Step 3: Run decoder
    codes_array = run_decoder(decoder_plan, encoder_output, num_tokens, weights)

    # Step 4: Decode with codec
    decode_codec(codes_array)

    # Step 5: Compare
    compare_audio()


if __name__ == "__main__":
    main()

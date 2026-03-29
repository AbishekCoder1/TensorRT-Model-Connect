#include "runtime/domains/audio/magpie_kernels.h"

#include <cfloat>
#include <cstdint>

namespace trtf {

// ---------------------------------------------------------------------------
// Kernel: segmented argmax over num_codebooks segments of codebook_size each.
// Each block handles one codebook. Uses shared-memory parallel reduction.
// Writes:
//   d_codes_out[cb]       = argmax within [0, audio_range)
//   d_full_argmax_out[cb] = argmax within [0, codebook_size) (for EOS check)
// ---------------------------------------------------------------------------

__global__ void segmented_argmax_kernel(
    const float* __restrict__ d_logits,
    int32_t codebook_size,
    int32_t audio_range,
    int32_t* __restrict__ d_codes_out,
    int32_t* __restrict__ d_full_argmax_out)
{
    const int32_t cb = static_cast<int32_t>(blockIdx.x);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t block_size = static_cast<int32_t>(blockDim.x);
    const float* cb_logits = d_logits + cb * codebook_size;

    // Each thread finds local best over its strided range
    float best_val_full = -FLT_MAX;
    int32_t best_idx_full = 0;
    float best_val_audio = -FLT_MAX;
    int32_t best_idx_audio = 0;

    // Full range [0, codebook_size)
    for (int32_t i = tid; i < codebook_size; i += block_size)
    {
        float v = cb_logits[i];
        if (v > best_val_full)
        {
            best_val_full = v;
            best_idx_full = i;
        }
        // Audio range [0, audio_range)
        if (i < audio_range && v > best_val_audio)
        {
            best_val_audio = v;
            best_idx_audio = i;
        }
    }

    // Shared memory reduction
    // Layout: [block_size] floats for full vals, [block_size] ints for full idx,
    //         [block_size] floats for audio vals, [block_size] ints for audio idx
    extern __shared__ char smem[];
    float* s_full_val    = reinterpret_cast<float*>(smem);
    int32_t* s_full_idx  = reinterpret_cast<int32_t*>(s_full_val + block_size);
    float* s_audio_val   = reinterpret_cast<float*>(s_full_idx + block_size);
    int32_t* s_audio_idx = reinterpret_cast<int32_t*>(s_audio_val + block_size);

    s_full_val[tid]  = best_val_full;
    s_full_idx[tid]  = best_idx_full;
    s_audio_val[tid] = best_val_audio;
    s_audio_idx[tid] = best_idx_audio;
    __syncthreads();

    // Tree reduction
    for (int32_t stride = block_size / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (s_full_val[tid + stride] > s_full_val[tid])
            {
                s_full_val[tid] = s_full_val[tid + stride];
                s_full_idx[tid] = s_full_idx[tid + stride];
            }
            if (s_audio_val[tid + stride] > s_audio_val[tid])
            {
                s_audio_val[tid] = s_audio_val[tid + stride];
                s_audio_idx[tid] = s_audio_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_codes_out[cb] = s_audio_idx[0];
        d_full_argmax_out[cb] = s_full_idx[0];
    }
}

void magpie_greedy_sample_device(
    const float* d_logits,
    int32_t num_codebooks,
    int32_t codebook_size,
    int32_t audio_range,
    int32_t* d_codes_out,
    int32_t* d_full_argmax_out,
    cudaStream_t stream)
{
    // Use 256 threads per block (enough for codebook_size=2024)
    constexpr int32_t kBlockSize = 256;
    const int32_t grid = num_codebooks;  // one block per codebook

    // Shared memory: 2*(float + int32_t) per thread
    const std::size_t smem_bytes = static_cast<std::size_t>(kBlockSize) *
        (2 * sizeof(float) + 2 * sizeof(int32_t));

    segmented_argmax_kernel<<<grid, kBlockSize, smem_bytes, stream>>>(
        d_logits, codebook_size, audio_range,
        d_codes_out, d_full_argmax_out);
}

// ---------------------------------------------------------------------------
// Kernel: gather 8 codebook embeddings, average them, write to output.
// Launch with hidden_size threads. Each thread accumulates one hidden dim
// across all codebooks.
// ---------------------------------------------------------------------------

__global__ void gather_average_embed_kernel(
    const float* __restrict__ audio_embed_table,
    const int32_t* __restrict__ prev_codes,
    int32_t num_codebooks,
    int32_t codebook_size,
    int32_t hidden_size,
    float* __restrict__ output)
{
    const int32_t h = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (h >= hidden_size) return;

    float sum = 0.0F;
    for (int32_t cb = 0; cb < num_codebooks; ++cb)
    {
        // Table layout: [num_codebooks, codebook_size, hidden_size]
        const int64_t offset = static_cast<int64_t>(cb) * codebook_size * hidden_size
                             + static_cast<int64_t>(prev_codes[cb]) * hidden_size
                             + h;
        sum += audio_embed_table[offset];
    }
    output[h] = sum / static_cast<float>(num_codebooks);
}

void magpie_gather_average_embed_device(
    const float* d_audio_embed_table,
    const int32_t* d_prev_codes,
    int32_t num_codebooks,
    int32_t codebook_size,
    int32_t hidden_size,
    float* d_output,
    cudaStream_t stream)
{
    constexpr int32_t kBlockSize = 256;
    const int32_t grid = (hidden_size + kBlockSize - 1) / kBlockSize;

    gather_average_embed_kernel<<<grid, kBlockSize, 0, stream>>>(
        d_audio_embed_table, d_prev_codes,
        num_codebooks, codebook_size, hidden_size, d_output);
}

// ---------------------------------------------------------------------------
// Kernel: scatter codes into accumulator, update prev_codes, check EOS.
// Single block of num_codebooks threads (8).
// ---------------------------------------------------------------------------

__global__ void scatter_codes_kernel(
    const int32_t* __restrict__ d_codes,
    int32_t* __restrict__ d_all_codes,
    int32_t* __restrict__ d_prev_codes,
    const int32_t* __restrict__ d_full_argmax,
    int32_t* __restrict__ d_eos_flag,
    int32_t frame_idx,
    int32_t num_codebooks,
    int32_t eos_token)
{
    const int32_t cb = static_cast<int32_t>(threadIdx.x);
    if (cb >= num_codebooks) return;

    const int32_t code = d_codes[cb];

    // Write to accumulator: all_codes[frame_idx * num_codebooks + cb]
    d_all_codes[frame_idx * num_codebooks + cb] = code;

    // Update prev_codes for next iteration
    d_prev_codes[cb] = code;

    // Check EOS (any codebook's full_argmax == eos_token sets the flag)
    if (d_full_argmax[cb] == eos_token)
    {
        atomicOr(d_eos_flag, 1);
    }
}

void magpie_scatter_codes_device(
    const int32_t* d_codes,
    int32_t* d_all_codes,
    int32_t* d_prev_codes,
    const int32_t* d_full_argmax,
    int32_t* d_eos_flag,
    int32_t frame_idx,
    int32_t num_codebooks,
    int32_t eos_token,
    cudaStream_t stream)
{
    scatter_codes_kernel<<<1, num_codebooks, 0, stream>>>(
        d_codes, d_all_codes, d_prev_codes, d_full_argmax, d_eos_flag,
        frame_idx, num_codebooks, eos_token);
}

// ---------------------------------------------------------------------------
// Kernel: CFG interpolation
// out[i] = uncond[i] + scale * (cond[i] - uncond[i])
// ---------------------------------------------------------------------------

__global__ void cfg_interpolate_kernel(
    const float* __restrict__ d_cond,
    const float* __restrict__ d_uncond,
    float* __restrict__ d_out,
    float cfg_scale,
    int32_t num_elements)
{
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_elements) return;
    d_out[i] = d_uncond[i] + cfg_scale * (d_cond[i] - d_uncond[i]);
}

void magpie_cfg_interpolate_device(
    const float* d_cond_logits,
    const float* d_uncond_logits,
    float* d_out_logits,
    float cfg_scale,
    int32_t num_elements,
    cudaStream_t stream)
{
    constexpr int32_t kBlockSize = 256;
    const int32_t grid = (num_elements + kBlockSize - 1) / kBlockSize;

    cfg_interpolate_kernel<<<grid, kBlockSize, 0, stream>>>(
        d_cond_logits, d_uncond_logits, d_out_logits,
        cfg_scale, num_elements);
}

} // namespace trtf

// ---------------------------------------------------------------------------
// DeviceOps: public API wrapping the kernels above.
// ---------------------------------------------------------------------------

namespace trtf {
namespace device_ops {

void greedy_sample_codebooks(
    const float* d_logits, int32_t num_codebooks, int32_t codebook_size,
    int32_t audio_range, int32_t* d_codes, int32_t* d_full_argmax,
    cudaStream_t stream)
{
    trtf::magpie_greedy_sample_device(
        d_logits, num_codebooks, codebook_size, audio_range,
        d_codes, d_full_argmax, stream);
}

void gather_average_embeddings(
    const float* d_embed_table, const int32_t* d_token_ids,
    int32_t num_entries, int32_t vocab_size, int32_t hidden_size,
    float* d_output, cudaStream_t stream)
{
    trtf::magpie_gather_average_embed_device(
        d_embed_table, d_token_ids,
        num_entries, vocab_size, hidden_size,
        d_output, stream);
}

void cfg_interpolate(
    const float* d_cond, const float* d_uncond, float* d_out,
    float scale, int32_t n, cudaStream_t stream)
{
    trtf::magpie_cfg_interpolate_device(
        d_cond, d_uncond, d_out, scale, n, stream);
}

void scatter_codes_check_eos(
    const int32_t* d_codes, int32_t* d_all_codes, int32_t* d_prev_codes,
    const int32_t* d_full_argmax, int32_t* d_eos_flag,
    int32_t frame, int32_t num_codebooks, int32_t eos_token,
    cudaStream_t stream)
{
    trtf::magpie_scatter_codes_device(
        d_codes, d_all_codes, d_prev_codes, d_full_argmax, d_eos_flag,
        frame, num_codebooks, eos_token, stream);
}

} // namespace device_ops
} // namespace trtf

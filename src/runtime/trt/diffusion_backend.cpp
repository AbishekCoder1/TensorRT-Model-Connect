#include "runtime/trt/diffusion_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

namespace trtf {

// ---------------------------------------------------------------------------
// Flow Match Euler Scheduler (C++ implementation)
// ---------------------------------------------------------------------------
namespace {

struct FlowMatchEulerState {
    std::vector<double> sigmas;
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    float shift{1.0F};

    void set_timesteps(int32_t num_steps) {
        // Match HF diffusers FlowMatchEulerDiscreteScheduler:
        // 1. Compute sigma_min = shift * (1/N) / (1 + (shift-1)/N)
        // 2. Linspace in t-space from t_max=1000 to t_min=sigma_min*1000
        //    with num_steps points (NOT num_steps+1)
        // 3. Convert to sigma-space: sigma = t / num_train_timesteps
        // 4. Apply shift: sigma = shift * sigma / (1 + (shift-1) * sigma)
        // 5. Append sigma=0 as terminal value

        const double N = static_cast<double>(num_train_timesteps);
        const double s = static_cast<double>(shift);

        // sigma_min is the shifted 1/N
        const double raw_sigma_min = 1.0 / N;
        const double sigma_min = s * raw_sigma_min /
            (1.0 + (s - 1.0) * raw_sigma_min);

        const double t_max = 1.0 * N;  // sigma_max=1.0 * N
        const double t_min = sigma_min * N;

        // Linspace in t-space: num_steps points from t_max to t_min
        sigmas.resize(static_cast<std::size_t>(num_steps) + 1);
        for (int32_t i = 0; i < num_steps; ++i) {
            const double frac = static_cast<double>(i) /
                static_cast<double>(std::max(num_steps - 1, 1));
            const double t_val = t_max + frac * (t_min - t_max);
            double sigma = t_val / N;

            // Apply shift
            if (std::abs(shift - 1.0F) > 1e-6F) {
                sigma = s * sigma / (1.0 + (s - 1.0) * sigma);
            }
            sigmas[static_cast<std::size_t>(i)] = sigma;
        }
        // Terminal sigma = 0
        sigmas[static_cast<std::size_t>(num_steps)] = 0.0;

        timesteps.resize(static_cast<std::size_t>(num_steps));
        for (int32_t i = 0; i < num_steps; ++i) {
            timesteps[static_cast<std::size_t>(i)] =
                static_cast<float>(sigmas[static_cast<std::size_t>(i)] *
                                   num_train_timesteps);
        }
    }

    // Euler step: x = x + (sigma_next - sigma) * velocity
    void step(const float* velocity, const float* sample, float* output,
              std::size_t count, int32_t step_index) const {
        const double sigma = sigmas[static_cast<std::size_t>(step_index)];
        const double sigma_next = sigmas[static_cast<std::size_t>(step_index) + 1];
        const double dt = sigma_next - sigma;

        for (std::size_t i = 0; i < count; ++i) {
            output[i] = static_cast<float>(
                static_cast<double>(sample[i]) +
                dt * static_cast<double>(velocity[i]));
        }
    }
};

// --- CPU math helpers (small matrices, no BLAS needed) ---

void cpu_matmul_bias(const float* A, const float* B, const float* bias,
                     float* out, int32_t M, int32_t K, int32_t N)
{
    // A: [M, K], B: [K, N] (row-major), out: [M, N]
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int32_t k = 0; k < K; ++k) {
                acc += static_cast<double>(A[i * K + k]) *
                       static_cast<double>(B[k * N + j]);
            }
            if (bias != nullptr) {
                acc += static_cast<double>(bias[j]);
            }
            out[i * N + j] = static_cast<float>(acc);
        }
    }
}

void cpu_silu_inplace(float* data, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) {
        const float x = data[i];
        data[i] = x / (1.0F + std::exp(-x));
    }
}

// GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void cpu_gelu_tanh_inplace(float* data, std::size_t count)
{
    constexpr float kSqrt2OverPi = 0.7978845608F;  // sqrt(2/pi)
    constexpr float kCoeff = 0.044715F;
    for (std::size_t i = 0; i < count; ++i) {
        const float x = data[i];
        const float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
        data[i] = 0.5F * x * (1.0F + std::tanh(inner));
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Preprocessor weight parsing
// ---------------------------------------------------------------------------

PreprocessorWeights parse_preprocessor_weights(const std::vector<char>& data)
{
    PreprocessorWeights w;

    if (data.size() < 4) {
        std::cerr << "[diffusion] preprocessor_weights section too small\n";
        return w;
    }

    // Format: [4-byte index length][index JSON][contiguous float32 data]
    uint32_t index_len = 0;
    std::memcpy(&index_len, data.data(), 4);

    if (4 + index_len > data.size()) {
        std::cerr << "[diffusion] preprocessor_weights index length overflow\n";
        return w;
    }

    const std::string index_json(data.data() + 4, data.data() + 4 + index_len);
    const char* blob = data.data() + 4 + index_len;
    const std::size_t blob_size = data.size() - 4 - index_len;

    // Simple JSON parsing for {"key": {"offset": N, "shape": [...]}, ...}
    // Using a minimal approach since we know the exact format.
    auto find_entry = [&](const std::string& key,
                          std::size_t& offset, std::vector<int32_t>& shape) -> bool {
        const std::string search = "\"" + key + "\"";
        auto pos = index_json.find(search);
        if (pos == std::string::npos) return false;

        // Find "offset": N
        auto off_pos = index_json.find("\"offset\"", pos);
        if (off_pos == std::string::npos) return false;
        auto colon = index_json.find(':', off_pos + 8);
        if (colon == std::string::npos) return false;
        offset = static_cast<std::size_t>(std::stoul(index_json.substr(colon + 1)));

        // Find "shape": [...]
        auto shape_pos = index_json.find("\"shape\"", pos);
        if (shape_pos == std::string::npos) return false;
        auto bracket = index_json.find('[', shape_pos);
        auto end_bracket = index_json.find(']', bracket);
        if (bracket == std::string::npos || end_bracket == std::string::npos) return false;

        std::string shape_str = index_json.substr(bracket + 1, end_bracket - bracket - 1);
        shape.clear();
        std::istringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            auto trimmed = token;
            trimmed.erase(0, trimmed.find_first_not_of(" \t"));
            trimmed.erase(trimmed.find_last_not_of(" \t") + 1);
            if (!trimmed.empty()) {
                shape.push_back(std::stoi(trimmed));
            }
        }
        return true;
    };

    auto load_floats = [&](const std::string& key, std::vector<float>& dst) -> bool {
        std::size_t offset = 0;
        std::vector<int32_t> shape;
        if (!find_entry(key, offset, shape)) return false;

        std::size_t count = 1;
        for (auto s : shape) count *= static_cast<std::size_t>(s);
        std::size_t nbytes = count * sizeof(float);

        if (offset + nbytes > blob_size) {
            std::cerr << "[diffusion] weight " << key << " overflows blob\n";
            return false;
        }

        dst.resize(count);
        std::memcpy(dst.data(), blob + offset, nbytes);
        return true;
    };

    load_floats("patch_embedding.weight", w.patch_embed_weight);
    load_floats("patch_embedding.bias", w.patch_embed_bias);
    load_floats("condition_embedder.time_embedding.0.weight", w.time_emb_0_weight);
    load_floats("condition_embedder.time_embedding.0.bias", w.time_emb_0_bias);
    load_floats("condition_embedder.time_embedding.2.weight", w.time_emb_2_weight);
    load_floats("condition_embedder.time_embedding.2.bias", w.time_emb_2_bias);
    load_floats("condition_embedder.time_proj.weight", w.time_proj_weight);
    load_floats("condition_embedder.time_proj.bias", w.time_proj_bias);
    load_floats("condition_embedder.text_embedding.weight", w.text_proj_weight);
    load_floats("condition_embedder.text_embedding.bias", w.text_proj_bias);
    load_floats("condition_embedder.text_embedding_2.weight", w.text_proj_2_weight);
    load_floats("condition_embedder.text_embedding_2.bias", w.text_proj_2_bias);

    // Compute patch_dim from patch_embed_weight shape
    if (!w.patch_embed_weight.empty() && !w.patch_embed_bias.empty()) {
        const auto dit_dim = static_cast<int32_t>(w.patch_embed_bias.size());
        w.patch_dim = static_cast<int32_t>(w.patch_embed_weight.size()) / dit_dim;
    }

    w.valid = !w.patch_embed_weight.empty() && !w.time_emb_0_weight.empty();

    std::cerr << "[diffusion] Preprocessor weights loaded: "
              << (w.valid ? "OK" : "INCOMPLETE")
              << " (patch_dim=" << w.patch_dim << ")\n";
    return w;
}

// ---------------------------------------------------------------------------
// DiffusionBackend
// ---------------------------------------------------------------------------

DiffusionBackend::DiffusionBackend(
    std::vector<DiffusionEngine> text_encoders,
    DiffusionEngine denoiser,
    DiffusionEngine vae_decoder,
    DiffusionConfig config)
    : mTextEncoders(std::move(text_encoders))
    , mDenoiser(std::move(denoiser))
    , mVaeDecoder(std::move(vae_decoder))
    , mConfig(std::move(config))
    , mD_InputIds(0)
    , mD_AttentionMask(0)
    , mD_TextEmbeddings(0)
    , mD_Hidden(0)
    , mD_Temb(0)
    , mD_TimeEmbed(0)
    , mD_EncoderHidden(0)
    , mD_RotaryCos(0)
    , mD_RotarySin(0)
    , mD_Output(0)
    , mD_VaeInput(0)
    , mD_VaeOutput(0)
{
    mOk = mStream.ok();
    if (!mOk) {
        std::cerr << "[diffusion] CUDA stream creation failed\n";
        return;
    }

    // Compute derived dimensions
    const int32_t seq_len = mConfig.text_seq_len;
    const int32_t te_dim = mConfig.text_encoder_dim;
    const int32_t dit_dim = mConfig.dit_dim;
    const int32_t head_dim = dit_dim / std::max(mConfig.dit_num_heads, 1);

    const int32_t t_lat = (mConfig.video_num_frames - 1) /
                           mConfig.scale_factor_temporal + 1;
    const int32_t h_lat = mConfig.video_height / mConfig.scale_factor_spatial;
    const int32_t w_lat = mConfig.video_width / mConfig.scale_factor_spatial;

    int32_t pt = 1, ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        pt = mConfig.patch_size[0];
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t num_patches = (t_lat / pt) * (h_lat / ph) * (w_lat / pw);
    const int32_t out_dim = mConfig.z_dim * pt * ph * pw;

    // Allocate T5 buffers
    mD_InputIds = CudaBuffer(
        static_cast<std::size_t>(seq_len) * sizeof(int32_t));
    mD_AttentionMask = CudaBuffer(
        static_cast<std::size_t>(seq_len) * sizeof(float));
    mD_TextEmbeddings = CudaBuffer(
        static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(te_dim) * sizeof(float));

    // Allocate DiT buffers
    mD_Hidden = CudaBuffer(
        static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_Temb = CudaBuffer(
        static_cast<std::size_t>(6) * static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_TimeEmbed = CudaBuffer(
        static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_EncoderHidden = CudaBuffer(
        static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_RotaryCos = CudaBuffer(
        static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(head_dim) * sizeof(float));
    mD_RotarySin = CudaBuffer(
        static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(head_dim) * sizeof(float));
    mD_Output = CudaBuffer(
        static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(out_dim) * sizeof(float));

    std::cerr << "[diffusion] Buffers allocated: num_patches=" << num_patches
              << ", dit_dim=" << dit_dim << ", head_dim=" << head_dim
              << ", out_dim=" << out_dim << "\n";

    // Allocate VAE buffers (discover shapes from engine)
    init_vae_buffers();
}

void DiffusionBackend::set_preprocessor_weights(PreprocessorWeights weights)
{
    mWeights = std::move(weights);
}

bool DiffusionBackend::is_available() const
{
    return mOk;
}

const char* DiffusionBackend::name() const
{
    return "trt_diffusion";
}

std::vector<int32_t> DiffusionBackend::generate(
    const std::vector<int32_t>& /*input_ids*/,
    const GenerationConfig& /*config*/)
{
    // Diffusion models don't do text generation.
    // Use generate_video() instead.
    return {};
}

// ---------------------------------------------------------------------------
// T5 encoder execution
// ---------------------------------------------------------------------------

bool DiffusionBackend::run_t5_encoder(
    const std::vector<int32_t>& input_ids,
    std::vector<float>& text_embeddings,
    std::string& error)
{
    if (mTextEncoders.empty()) {
        error = "No text encoder engine";
        return false;
    }

    auto& te = mTextEncoders[0];
    const int32_t seq_len = mConfig.text_seq_len;
    const int32_t te_dim = mConfig.text_encoder_dim;

    // Pad or truncate input_ids to [1, seq_len]
    std::vector<int32_t> padded_ids(static_cast<std::size_t>(seq_len), 0);
    const auto copy_len = std::min(
        static_cast<std::size_t>(seq_len), input_ids.size());
    std::copy_n(input_ids.begin(), copy_len, padded_ids.begin());

    // Build attention mask [1, seq_len]: 0.0 for valid, -1e9 for padding.
    // Mask based on token content: pad_token_id == 0 for T5/UMT5.
    std::vector<float> mask(static_cast<std::size_t>(seq_len), -1e9F);
    for (int32_t i = 0; i < seq_len; ++i) {
        if (padded_ids[static_cast<std::size_t>(i)] != 0) {
            mask[static_cast<std::size_t>(i)] = 0.0F;
        }
    }

    // H2D: input_ids, attention_mask
    cudaMemcpyAsync(mD_InputIds.data(), padded_ids.data(),
        padded_ids.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_AttentionMask.data(), mask.data(),
        mask.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());

    // Set tensor addresses
    te.context->setTensorAddress("input_ids", mD_InputIds.data());
    te.context->setTensorAddress("attention_mask", mD_AttentionMask.data());
    te.context->setTensorAddress("text_embeddings", mD_TextEmbeddings.data());

    // Execute
    if (!te.context->enqueueV3(mStream.get())) {
        error = "T5 enqueueV3 failed";
        return false;
    }

    // D2H: text_embeddings [1, seq_len, te_dim]
    const auto emb_size = static_cast<std::size_t>(seq_len) *
                          static_cast<std::size_t>(te_dim);
    text_embeddings.resize(emb_size);
    cudaMemcpyAsync(text_embeddings.data(), mD_TextEmbeddings.data(),
        emb_size * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    // Zero out padding positions in T5 output.
    // Even with attention masking in T5 self-attention, padding positions
    // produce non-zero output (via residual connections). The DiT cross-
    // attention has no masking, so non-zero padding values dilute the text
    // signal. HF's pipeline avoids this by using variable-length sequences.
    for (int32_t i = 0; i < seq_len; ++i) {
        if (padded_ids[static_cast<std::size_t>(i)] == 0) {
            float* row = text_embeddings.data() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(te_dim);
            std::fill_n(row, static_cast<std::size_t>(te_dim), 0.0F);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// CPU preprocessing helpers
// ---------------------------------------------------------------------------

void DiffusionBackend::compute_timestep_embedding(
    float timestep,
    std::vector<float>& temb_6d,
    std::vector<float>& time_embed) const
{
    const int32_t dim = mConfig.dit_dim;
    const int32_t freq_dim = mConfig.freq_dim;
    const int32_t half = freq_dim / 2;

    // 1. Sinusoidal embedding: [1, freq_dim]
    std::vector<float> sinusoidal(static_cast<std::size_t>(freq_dim));
    for (int32_t i = 0; i < half; ++i) {
        const double freq = std::exp(
            -std::log(10000.0) * static_cast<double>(i) / static_cast<double>(half));
        const double angle = static_cast<double>(timestep) * freq;
        sinusoidal[static_cast<std::size_t>(i)] = static_cast<float>(std::cos(angle));
        sinusoidal[static_cast<std::size_t>(i + half)] = static_cast<float>(std::sin(angle));
    }

    // 2. MLP: Linear(freq_dim, dim) -> SiLU -> Linear(dim, dim) -> time_embed
    //    time_emb_0: [freq_dim, dim], time_emb_2: [dim, dim]
    std::vector<float> hidden_1(static_cast<std::size_t>(dim));
    cpu_matmul_bias(sinusoidal.data(), mWeights.time_emb_0_weight.data(),
                    mWeights.time_emb_0_bias.data(),
                    hidden_1.data(), 1, freq_dim, dim);
    cpu_silu_inplace(hidden_1.data(), static_cast<std::size_t>(dim));

    time_embed.resize(static_cast<std::size_t>(dim));
    cpu_matmul_bias(hidden_1.data(), mWeights.time_emb_2_weight.data(),
                    mWeights.time_emb_2_bias.data(),
                    time_embed.data(), 1, dim, dim);

    // 3. time_proj: SiLU(time_embed) -> Linear(dim, 6*dim) -> temb_6d
    std::vector<float> silu_te(time_embed.begin(), time_embed.end());
    cpu_silu_inplace(silu_te.data(), static_cast<std::size_t>(dim));

    temb_6d.resize(static_cast<std::size_t>(6 * dim));
    cpu_matmul_bias(silu_te.data(), mWeights.time_proj_weight.data(),
                    mWeights.time_proj_bias.data(),
                    temb_6d.data(), 1, dim, 6 * dim);
}

void DiffusionBackend::project_text(
    const std::vector<float>& in, int32_t seq_len,
    std::vector<float>& out) const
{
    const int32_t te_dim = mConfig.text_encoder_dim;
    const int32_t dim = mConfig.dit_dim;

    // Layer 1: Linear(text_dim, dim)
    out.resize(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(dim));
    cpu_matmul_bias(in.data(), mWeights.text_proj_weight.data(),
                    mWeights.text_proj_bias.data(),
                    out.data(), seq_len, te_dim, dim);

    // If 2-layer MLP: GELU(tanh) -> Linear(dim, dim)
    if (!mWeights.text_proj_2_weight.empty()) {
        cpu_gelu_tanh_inplace(out.data(),
            static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(dim));
        std::vector<float> tmp(out.size());
        cpu_matmul_bias(out.data(), mWeights.text_proj_2_weight.data(),
                        mWeights.text_proj_2_bias.data(),
                        tmp.data(), seq_len, dim, dim);
        out = std::move(tmp);
    }
}

void DiffusionBackend::patchify(
    const std::vector<float>& latents,
    int32_t c, int32_t t, int32_t h, int32_t w,
    std::vector<float>& patches) const
{
    int32_t pt = 1, ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        pt = mConfig.patch_size[0];
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t nt = t / pt, nh = h / ph, nw = w / pw;
    const int32_t patch_dim = c * pt * ph * pw;
    const int32_t num_patches = nt * nh * nw;

    patches.resize(static_cast<std::size_t>(num_patches) *
                   static_cast<std::size_t>(patch_dim));

    // latents layout: [c, t, h, w] (no batch dim in internal representation)
    // Reorder to [nt, nh, nw, c, pt, ph, pw] then flatten to [num_patches, patch_dim]
    int32_t patch_idx = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                int32_t elem = 0;
                for (int32_t ci = 0; ci < c; ++ci) {
                    for (int32_t pti = 0; pti < pt; ++pti) {
                        for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                            for (int32_t pwi = 0; pwi < pw; ++pwi) {
                                const int32_t tt = ti * pt + pti;
                                const int32_t hh = hi * ph + phi_;
                                const int32_t ww = wi * pw + pwi;
                                const auto src_idx =
                                    static_cast<std::size_t>(ci) * static_cast<std::size_t>(t * h * w) +
                                    static_cast<std::size_t>(tt) * static_cast<std::size_t>(h * w) +
                                    static_cast<std::size_t>(hh) * static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(ww);
                                patches[static_cast<std::size_t>(patch_idx) *
                                        static_cast<std::size_t>(patch_dim) +
                                        static_cast<std::size_t>(elem)] =
                                    latents[src_idx];
                                ++elem;
                            }
                        }
                    }
                }
                ++patch_idx;
            }
        }
    }
}

void DiffusionBackend::unpatchify(
    const std::vector<float>& patches,
    int32_t c, int32_t t, int32_t h, int32_t w,
    std::vector<float>& output) const
{
    int32_t pt = 1, ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        pt = mConfig.patch_size[0];
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t nt = t / pt, nh = h / ph, nw = w / pw;
    const int32_t patch_dim = c * pt * ph * pw;

    output.resize(static_cast<std::size_t>(c * t * h * w));

    // DiT proj_out produces output in [pt, ph, pw, C] order (C varies fastest),
    // matching HF's WanTransformer3DModel unpatchify convention.
    int32_t patch_idx = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                int32_t elem = 0;
                for (int32_t pti = 0; pti < pt; ++pti) {
                    for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                        for (int32_t pwi = 0; pwi < pw; ++pwi) {
                            for (int32_t ci = 0; ci < c; ++ci) {
                                const int32_t tt = ti * pt + pti;
                                const int32_t hh = hi * ph + phi_;
                                const int32_t ww = wi * pw + pwi;
                                const auto dst_idx =
                                    static_cast<std::size_t>(ci) * static_cast<std::size_t>(t * h * w) +
                                    static_cast<std::size_t>(tt) * static_cast<std::size_t>(h * w) +
                                    static_cast<std::size_t>(hh) * static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(ww);
                                output[dst_idx] =
                                    patches[static_cast<std::size_t>(patch_idx) *
                                            static_cast<std::size_t>(patch_dim) +
                                            static_cast<std::size_t>(elem)];
                                ++elem;
                            }
                        }
                    }
                }
                ++patch_idx;
            }
        }
    }

}

void DiffusionBackend::compute_3d_rope(
    int32_t nt, int32_t nh, int32_t nw,
    std::vector<float>& cos_out,
    std::vector<float>& sin_out) const
{
    const int32_t dim = mConfig.dit_dim;
    const int32_t num_heads = mConfig.dit_num_heads;
    const int32_t head_dim = dim / std::max(num_heads, 1);
    const int32_t num_patches = nt * nh * nw;
    const double theta = 10000.0;

    // Wan uses: h_dim = w_dim = 2*(head_dim//6), t_dim = head_dim - h_dim - w_dim
    const int32_t h_dim = 2 * (head_dim / 6);
    const int32_t w_dim = h_dim;
    const int32_t t_dim = head_dim - h_dim - w_dim;

    auto get_1d_rope = [&](int32_t rdim, int32_t max_len,
                           std::vector<std::vector<float>>& cos_table,
                           std::vector<std::vector<float>>& sin_table) {
        const int32_t half = rdim / 2;
        cos_table.resize(static_cast<std::size_t>(max_len));
        sin_table.resize(static_cast<std::size_t>(max_len));

        for (int32_t pos = 0; pos < max_len; ++pos) {
            auto& c = cos_table[static_cast<std::size_t>(pos)];
            auto& s = sin_table[static_cast<std::size_t>(pos)];
            c.resize(static_cast<std::size_t>(rdim));
            s.resize(static_cast<std::size_t>(rdim));

            for (int32_t i = 0; i < half; ++i) {
                const double freq = 1.0 / std::pow(theta,
                    static_cast<double>(i) / static_cast<double>(half));
                const double angle = static_cast<double>(pos) * freq;
                const auto cv = static_cast<float>(std::cos(angle));
                const auto sv = static_cast<float>(std::sin(angle));
                // repeat_interleave: each freq -> 2 positions
                c[static_cast<std::size_t>(2 * i)] = cv;
                c[static_cast<std::size_t>(2 * i + 1)] = cv;
                s[static_cast<std::size_t>(2 * i)] = sv;
                s[static_cast<std::size_t>(2 * i + 1)] = sv;
            }
        }
    };

    std::vector<std::vector<float>> t_cos, t_sin, h_cos, h_sin, w_cos, w_sin;
    get_1d_rope(t_dim, std::max(nt, 1024), t_cos, t_sin);
    get_1d_rope(h_dim, std::max(nh, 1024), h_cos, h_sin);
    get_1d_rope(w_dim, std::max(nw, 1024), w_cos, w_sin);

    cos_out.resize(static_cast<std::size_t>(num_patches) *
                   static_cast<std::size_t>(head_dim));
    sin_out.resize(static_cast<std::size_t>(num_patches) *
                   static_cast<std::size_t>(head_dim));

    int32_t p = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                float* c_row = cos_out.data() + static_cast<std::size_t>(p) *
                               static_cast<std::size_t>(head_dim);
                float* s_row = sin_out.data() + static_cast<std::size_t>(p) *
                               static_cast<std::size_t>(head_dim);

                int32_t off = 0;
                std::memcpy(c_row + off, t_cos[static_cast<std::size_t>(ti)].data(),
                            static_cast<std::size_t>(t_dim) * sizeof(float));
                std::memcpy(s_row + off, t_sin[static_cast<std::size_t>(ti)].data(),
                            static_cast<std::size_t>(t_dim) * sizeof(float));
                off += t_dim;

                std::memcpy(c_row + off, h_cos[static_cast<std::size_t>(hi)].data(),
                            static_cast<std::size_t>(h_dim) * sizeof(float));
                std::memcpy(s_row + off, h_sin[static_cast<std::size_t>(hi)].data(),
                            static_cast<std::size_t>(h_dim) * sizeof(float));
                off += h_dim;

                std::memcpy(c_row + off, w_cos[static_cast<std::size_t>(wi)].data(),
                            static_cast<std::size_t>(w_dim) * sizeof(float));
                std::memcpy(s_row + off, w_sin[static_cast<std::size_t>(wi)].data(),
                            static_cast<std::size_t>(w_dim) * sizeof(float));

                ++p;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DiT denoiser execution
// ---------------------------------------------------------------------------

bool DiffusionBackend::run_denoiser(
    const std::vector<float>& hidden,
    const std::vector<float>& temb_6d,
    const std::vector<float>& time_embed,
    const std::vector<float>& encoder_hidden,
    const std::vector<float>& cos_vals,
    const std::vector<float>& sin_vals,
    std::vector<float>& output,
    std::string& error)
{
    auto& ctx = mDenoiser.context;

    // H2D all inputs
    cudaMemcpyAsync(mD_Hidden.data(), hidden.data(),
        hidden.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_Temb.data(), temb_6d.data(),
        temb_6d.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_TimeEmbed.data(), time_embed.data(),
        time_embed.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_EncoderHidden.data(), encoder_hidden.data(),
        encoder_hidden.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_RotaryCos.data(), cos_vals.data(),
        cos_vals.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_RotarySin.data(), sin_vals.data(),
        sin_vals.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());

    // Set tensor addresses
    ctx->setTensorAddress("hidden_states", mD_Hidden.data());
    ctx->setTensorAddress("timestep_embedding", mD_Temb.data());
    ctx->setTensorAddress("time_embed", mD_TimeEmbed.data());
    ctx->setTensorAddress("encoder_hidden_states", mD_EncoderHidden.data());
    ctx->setTensorAddress("rotary_cos", mD_RotaryCos.data());
    ctx->setTensorAddress("rotary_sin", mD_RotarySin.data());
    ctx->setTensorAddress("output", mD_Output.data());

    // Execute
    if (!ctx->enqueueV3(mStream.get())) {
        error = "DiT enqueueV3 failed";
        return false;
    }

    // D2H output
    const auto out_size = mD_Output.size() / sizeof(float);
    output.resize(out_size);
    cudaMemcpyAsync(output.data(), mD_Output.data(),
        mD_Output.size(),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    return true;
}

// ---------------------------------------------------------------------------
// VAE decode via Python subprocess
// ---------------------------------------------------------------------------

bool DiffusionBackend::decode_vae_subprocess(
    const std::vector<float>& latents,
    int32_t c, int32_t t, int32_t h, int32_t w,
    VideoResult& result, std::string& error)
{
    if (mHfPython.empty()) {
        error = "No hf_python path set for VAE decode subprocess";
        return false;
    }

    if (mConfig.vae_model_id.empty()) {
        error = "No vae_model_id in config for VAE decode subprocess";
        return false;
    }

    // Write latents to temp file
    const std::string tmp_dir = std::filesystem::temp_directory_path().string();
    const std::string lat_file = tmp_dir + "/trtf_vae_latents.bin";
    const std::string out_file = tmp_dir + "/trtf_vae_output.bin";
    const std::string script = std::string(TRTF_SOURCE_DIR) + "/scripts/vae_decode.py";

    {
        std::ofstream f(lat_file, std::ios::binary | std::ios::trunc);
        if (!f) {
            error = "Failed to write latent file: " + lat_file;
            return false;
        }
        f.write(reinterpret_cast<const char*>(latents.data()),
                static_cast<std::streamsize>(latents.size() * sizeof(float)));
    }

    // Build shape string: "1,C,T,H,W"
    const std::string shape_str = "1," + std::to_string(c) + "," +
        std::to_string(t) + "," + std::to_string(h) + "," + std::to_string(w);

    // Invoke subprocess
    const std::string cmd = mHfPython + " " + script +
        " --model-id " + mConfig.vae_model_id +
        " --latents-file " + lat_file +
        " --output-file " + out_file +
        " --shape " + shape_str +
        " 2>&1";

    std::cerr << "[diffusion] Running VAE decode subprocess ...\n";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        error = "Failed to spawn VAE decode subprocess";
        return false;
    }

    // Read subprocess output
    std::array<char, 256> buf{};
    std::string subprocess_output;
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
        subprocess_output += buf.data();
    }
    const int exit_code = pclose(pipe);

    if (!subprocess_output.empty()) {
        std::cerr << subprocess_output;
    }

    if (exit_code != 0) {
        error = "VAE decode subprocess failed (exit=" +
                std::to_string(exit_code) + ")";
        return false;
    }

    // Read output: float32 [1, 3, T_out, H_out, W_out]
    std::ifstream f(out_file, std::ios::binary | std::ios::ate);
    if (!f) {
        error = "Failed to read VAE output: " + out_file;
        return false;
    }

    const auto file_size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> raw(file_size / sizeof(float));
    f.read(reinterpret_cast<char*>(raw.data()),
           static_cast<std::streamsize>(file_size));

    // Decode output dimensions from the raw data size:
    // Output is [1, 3, T_out, H_out, W_out]
    const int32_t T_out = mConfig.video_num_frames;
    const int32_t H_out = mConfig.video_height;
    const int32_t W_out = mConfig.video_width;
    const std::size_t expected_size =
        static_cast<std::size_t>(3) * static_cast<std::size_t>(T_out) *
        static_cast<std::size_t>(H_out) * static_cast<std::size_t>(W_out);

    if (raw.size() < expected_size) {
        error = "VAE output too small: " + std::to_string(raw.size()) +
                " < " + std::to_string(expected_size);
        return false;
    }

    // Convert [3, T, H, W] -> [T, H, W, 3] and clamp to [0, 1]
    result.num_frames = T_out;
    result.height = H_out;
    result.width = W_out;
    result.frames.resize(expected_size);

    for (int32_t ft = 0; ft < T_out; ++ft) {
        for (int32_t fh = 0; fh < H_out; ++fh) {
            for (int32_t fw = 0; fw < W_out; ++fw) {
                for (int32_t fc = 0; fc < 3; ++fc) {
                    const auto src =
                        static_cast<std::size_t>(fc) * static_cast<std::size_t>(T_out * H_out * W_out) +
                        static_cast<std::size_t>(ft) * static_cast<std::size_t>(H_out * W_out) +
                        static_cast<std::size_t>(fh) * static_cast<std::size_t>(W_out) +
                        static_cast<std::size_t>(fw);
                    const auto dst =
                        static_cast<std::size_t>(ft) * static_cast<std::size_t>(H_out * W_out * 3) +
                        static_cast<std::size_t>(fh) * static_cast<std::size_t>(W_out * 3) +
                        static_cast<std::size_t>(fw) * 3 +
                        static_cast<std::size_t>(fc);

                    // Normalize from [-1,1] -> [0,1]
                    float v = (raw[src] + 1.0F) * 0.5F;
                    v = std::max(0.0F, std::min(1.0F, v));
                    result.frames[dst] = v;
                }
            }
        }
    }

    // Clean up temp files
    std::filesystem::remove(lat_file);
    std::filesystem::remove(out_file);

    return true;
}

// ---------------------------------------------------------------------------
// Native VAE decode
// ---------------------------------------------------------------------------

void DiffusionBackend::init_vae_buffers()
{
    auto& engine = mVaeDecoder.engine;
    if (!engine) return;

    const int32_t num_caches = mConfig.num_vae_caches;
    if (num_caches <= 0) return;

    // Discover tensor shapes from the TRT engine
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto dims = engine->getTensorShape(name);
        const std::string sname(name);

        if (sname == "latent_frame") {
            std::size_t sz = sizeof(float);
            for (int32_t d = 0; d < dims.nbDims; ++d)
                sz *= static_cast<std::size_t>(std::max(static_cast<int32_t>(dims.d[d]), 1));
            mD_VaeInput = CudaBuffer(sz);
        }
        else if (sname == "video_frame") {
            std::size_t sz = sizeof(float);
            for (int32_t d = 0; d < dims.nbDims; ++d)
                sz *= static_cast<std::size_t>(std::max(static_cast<int32_t>(dims.d[d]), 1));
            mD_VaeOutput = CudaBuffer(sz);
            // T dimension is dims.d[2]
            if (dims.nbDims >= 3) {
                mVaeOutputT = std::max(static_cast<int32_t>(dims.d[2]), 1);
            }
        }
    }

    // Allocate cache buffers (CudaBuffer is move-only, can't use fill-resize)
    mD_VaeCacheIn.reserve(static_cast<std::size_t>(num_caches));
    mD_VaeCacheOut.reserve(static_cast<std::size_t>(num_caches));
    for (int32_t ci = 0; ci < num_caches; ++ci) {
        mD_VaeCacheIn.emplace_back(0);
        mD_VaeCacheOut.emplace_back(0);
    }
    mVaeCacheSizes.resize(static_cast<std::size_t>(num_caches), 0);

    for (int32_t ci = 0; ci < num_caches; ++ci) {
        const std::string cin_name = "cache_" + std::to_string(ci);
        const std::string cout_name = "cache_out_" + std::to_string(ci);

        // Find the input cache tensor and compute its size
        for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* tname = engine->getIOTensorName(i);
            if (cin_name == tname) {
                const auto dims = engine->getTensorShape(tname);
                std::size_t sz = sizeof(float);
                for (int32_t d = 0; d < dims.nbDims; ++d)
                    sz *= static_cast<std::size_t>(std::max(static_cast<int32_t>(dims.d[d]), 1));
                mD_VaeCacheIn[static_cast<std::size_t>(ci)] = CudaBuffer(sz);
                mD_VaeCacheOut[static_cast<std::size_t>(ci)] = CudaBuffer(sz);
                mVaeCacheSizes[static_cast<std::size_t>(ci)] = sz;
                break;
            }
        }
    }

    std::cerr << "[diffusion] VAE buffers allocated: "
              << num_caches << " caches, output_T=" << mVaeOutputT << "\n";
}

bool DiffusionBackend::decode_vae_native(
    const std::vector<float>& latents,
    int32_t c, int32_t t_lat, int32_t h_lat, int32_t w_lat,
    VideoResult& result, std::string& error)
{
    const int32_t num_caches = mConfig.num_vae_caches;
    auto& ctx = mVaeDecoder.context;
    if (!ctx) {
        error = "No VAE execution context";
        return false;
    }

    const std::size_t frame_size =
        static_cast<std::size_t>(c) * 1 *
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    const std::size_t frame_bytes = frame_size * sizeof(float);

    // Zero all caches
    for (int32_t ci = 0; ci < num_caches; ++ci) {
        const auto idx = static_cast<std::size_t>(ci);
        if (mVaeCacheSizes[idx] > 0) {
            cudaMemsetAsync(mD_VaeCacheIn[idx].data(), 0,
                            mVaeCacheSizes[idx], mStream.get());
        }
    }

    // Output dims
    const int32_t sft = mConfig.scale_factor_temporal;
    const int32_t H_out = mConfig.video_height;
    const int32_t W_out = mConfig.video_width;
    const int32_t T_out_per_frame = mVaeOutputT;  // typically sft (4)

    // Collect all output frames
    const std::size_t out_frame_floats =
        static_cast<std::size_t>(3) * static_cast<std::size_t>(T_out_per_frame) *
        static_cast<std::size_t>(H_out) * static_cast<std::size_t>(W_out);
    const std::size_t out_frame_bytes = out_frame_floats * sizeof(float);

    std::vector<float> all_raw_frames;
    all_raw_frames.reserve(
        static_cast<std::size_t>(t_lat) * out_frame_floats);

    for (int32_t t = 0; t < t_lat; ++t) {
        // Build contiguous [1, c, 1, h, w] from [c, t_lat, h, w]
        std::vector<float> frame_buf(frame_size);
        const auto spatial = static_cast<std::size_t>(h_lat) *
                             static_cast<std::size_t>(w_lat);
        for (int32_t ci = 0; ci < c; ++ci) {
            const float* ch_src = latents.data() +
                static_cast<std::size_t>(ci) *
                static_cast<std::size_t>(t_lat) * spatial +
                static_cast<std::size_t>(t) * spatial;
            std::memcpy(frame_buf.data() +
                static_cast<std::size_t>(ci) * spatial,
                ch_src, spatial * sizeof(float));
        }

        cudaMemcpyAsync(mD_VaeInput.data(), frame_buf.data(), frame_bytes,
                         cudaMemcpyHostToDevice, mStream.get());

        // Set tensor addresses
        ctx->setTensorAddress("latent_frame", mD_VaeInput.data());
        ctx->setTensorAddress("video_frame", mD_VaeOutput.data());

        for (int32_t ci = 0; ci < num_caches; ++ci) {
            const auto idx = static_cast<std::size_t>(ci);
            const std::string cin = "cache_" + std::to_string(ci);
            const std::string cout = "cache_out_" + std::to_string(ci);
            ctx->setTensorAddress(cin.c_str(), mD_VaeCacheIn[idx].data());
            ctx->setTensorAddress(cout.c_str(), mD_VaeCacheOut[idx].data());
        }

        if (!ctx->enqueueV3(mStream.get())) {
            error = "VAE enqueueV3 failed at frame " + std::to_string(t);
            return false;
        }

        // Download output frame [1, 3, T_out, H, W]
        std::vector<float> out_buf(out_frame_floats);
        cudaMemcpyAsync(out_buf.data(), mD_VaeOutput.data(), out_frame_bytes,
                         cudaMemcpyDeviceToHost, mStream.get());
        cudaStreamSynchronize(mStream.get());

        all_raw_frames.insert(all_raw_frames.end(),
                              out_buf.begin(), out_buf.end());

        // Swap caches: copy cache_out -> cache_in
        for (int32_t ci = 0; ci < num_caches; ++ci) {
            const auto idx = static_cast<std::size_t>(ci);
            if (mVaeCacheSizes[idx] > 0) {
                cudaMemcpyAsync(mD_VaeCacheIn[idx].data(),
                                mD_VaeCacheOut[idx].data(),
                                mVaeCacheSizes[idx],
                                cudaMemcpyDeviceToDevice, mStream.get());
            }
        }

        if (t % 2 == 0) {
            std::cerr << "  VAE frame " << (t + 1) << "/" << t_lat << "\n";
        }
    }

    // Concatenated raw: [t_lat * T_out_per_frame, 3, H, W] in CHW layout
    // Trim first (sft - 1) frames
    const int32_t total_out_frames = t_lat * T_out_per_frame;
    const int32_t trim = sft - 1;
    const int32_t T_final = total_out_frames - trim;

    result.num_frames = std::min(T_final, mConfig.video_num_frames);
    result.height = H_out;
    result.width = W_out;
    const auto final_pixel_count =
        static_cast<std::size_t>(result.num_frames) *
        static_cast<std::size_t>(H_out) * static_cast<std::size_t>(W_out) * 3;
    result.frames.resize(final_pixel_count);

    // Convert [3, T_out_per_frame, H, W] per-input-frame layout to [T, H, W, 3]
    // Each input frame produced T_out_per_frame output frames in layout [1, 3, T_out, H, W]
    const auto per_frame_spatial =
        static_cast<std::size_t>(H_out) * static_cast<std::size_t>(W_out);

    for (int32_t input_t = 0; input_t < t_lat; ++input_t) {
        const float* raw_base = all_raw_frames.data() +
            static_cast<std::size_t>(input_t) * out_frame_floats;

        for (int32_t sub_t = 0; sub_t < T_out_per_frame; ++sub_t) {
            const int32_t global_t = input_t * T_out_per_frame + sub_t;

            // Skip trimmed frames
            if (global_t < trim) continue;
            const int32_t final_t = global_t - trim;
            if (final_t >= result.num_frames) continue;

            for (int32_t fh = 0; fh < H_out; ++fh) {
                for (int32_t fw = 0; fw < W_out; ++fw) {
                    for (int32_t fc = 0; fc < 3; ++fc) {
                        // Source: [1, 3, T_out, H, W] layout
                        const auto s_idx =
                            static_cast<std::size_t>(fc) *
                                static_cast<std::size_t>(T_out_per_frame) *
                                per_frame_spatial +
                            static_cast<std::size_t>(sub_t) * per_frame_spatial +
                            static_cast<std::size_t>(fh) *
                                static_cast<std::size_t>(W_out) +
                            static_cast<std::size_t>(fw);

                        // Dest: [T, H, W, 3] layout
                        const auto d_idx =
                            static_cast<std::size_t>(final_t) *
                                static_cast<std::size_t>(H_out) *
                                static_cast<std::size_t>(W_out) * 3 +
                            static_cast<std::size_t>(fh) *
                                static_cast<std::size_t>(W_out) * 3 +
                            static_cast<std::size_t>(fw) * 3 +
                            static_cast<std::size_t>(fc);

                        // Normalize [-1,1] -> [0,1] and clamp
                        float v = (raw_base[s_idx] + 1.0F) * 0.5F;
                        v = std::max(0.0F, std::min(1.0F, v));
                        result.frames[d_idx] = v;
                    }
                }
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Main generation pipeline
// ---------------------------------------------------------------------------

VideoResult DiffusionBackend::generate_video(
    const std::vector<int32_t>& input_ids,
    int32_t num_inference_steps,
    float guidance_scale)
{
    if (num_inference_steps < 0) {
        num_inference_steps = mConfig.num_inference_steps;
    }
    if (guidance_scale < 0.0F) {
        guidance_scale = mConfig.guidance_scale;
    }

    VideoResult result;
    result.height = mConfig.video_height;
    result.width = mConfig.video_width;

    if (!mWeights.valid) {
        std::cerr << "[diffusion] ERROR: preprocessor weights not loaded\n";
        return result;
    }

    // Compute latent dimensions
    const int32_t t_lat = (mConfig.video_num_frames - 1) /
                           mConfig.scale_factor_temporal + 1;
    const int32_t h_lat = mConfig.video_height / mConfig.scale_factor_spatial;
    const int32_t w_lat = mConfig.video_width / mConfig.scale_factor_spatial;
    const int32_t z_dim = mConfig.z_dim;
    const int32_t dim = mConfig.dit_dim;
    const int32_t seq_len = mConfig.text_seq_len;

    int32_t pt = 1, ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        pt = mConfig.patch_size[0];
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t nt = t_lat / pt;
    const int32_t nh_p = h_lat / ph;
    const int32_t nw_p = w_lat / pw;
    const int32_t num_patches = nt * nh_p * nw_p;
    const int32_t patch_dim = z_dim * pt * ph * pw;

    std::cerr << "[diffusion] Latent shape: "
              << z_dim << "x" << t_lat << "x"
              << h_lat << "x" << w_lat
              << " (patches=" << num_patches << ")\n";

    // 1. Text encoding
    std::cerr << "[diffusion] Encoding text ("
              << input_ids.size() << " tokens) ...\n";
    std::vector<float> text_embeddings;
    std::string error;
    if (!run_t5_encoder(input_ids, text_embeddings, error)) {
        std::cerr << "[diffusion] T5 encoding failed: " << error << "\n";
        return result;
    }
    std::cerr << "[diffusion] T5 encoding done ("
              << text_embeddings.size() << " floats)\n";

    // 2. Project text embeddings to DiT dimension
    std::vector<float> text_projected;
    project_text(text_embeddings, seq_len, text_projected);

    // Null text for CFG: encode empty string through T5, then project.
    // HF uses T5("") for the unconditional embedding, NOT zeros.
    // Using zeros breaks CFG guidance and causes color distortion.
    std::vector<int32_t> empty_ids(static_cast<std::size_t>(seq_len), 0);
    // Empty string tokenizes to just EOS (id=1)
    empty_ids[0] = 1;
    std::vector<float> null_embeddings;
    if (!run_t5_encoder(empty_ids, null_embeddings, error)) {
        std::cerr << "[diffusion] T5 null encoding failed: " << error << "\n";
        return result;
    }
    std::vector<float> null_text;
    project_text(null_embeddings, seq_len, null_text);

    // 3. Compute 3D RoPE (once, not per step)
    std::vector<float> rope_cos, rope_sin;
    compute_3d_rope(nt, nh_p, nw_p, rope_cos, rope_sin);

    // 4. Initialize latents with random noise
    const std::size_t latent_count = static_cast<std::size_t>(z_dim) *
        static_cast<std::size_t>(t_lat) * static_cast<std::size_t>(h_lat) *
        static_cast<std::size_t>(w_lat);
    std::vector<float> latents(latent_count);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (auto& v : latents) {
        v = dist(rng);
    }

    // 5. Denoising loop
    FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = 1000;
    scheduler.shift = mConfig.flow_shift;
    scheduler.set_timesteps(num_inference_steps);

    std::vector<float> noise_pred_spatial(latent_count);

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float timestep = scheduler.timesteps[static_cast<std::size_t>(step)];

        // Compute timestep embedding
        std::vector<float> temb_6d, time_embed;
        compute_timestep_embedding(timestep, temb_6d, time_embed);

        // Patchify latents -> [num_patches, patch_dim]
        std::vector<float> patches;
        patchify(latents, z_dim, t_lat, h_lat, w_lat, patches);

        // Apply patch embedding: patches @ weight + bias -> hidden [num_patches, dim]
        std::vector<float> hidden(
            static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(dim));
        cpu_matmul_bias(patches.data(), mWeights.patch_embed_weight.data(),
                        mWeights.patch_embed_bias.data(),
                        hidden.data(), num_patches, patch_dim, dim);

        std::vector<float> denoiser_output;

        if (guidance_scale > 1.0F) {
            // CFG: run twice (conditional + unconditional)
            std::vector<float> cond_pred, uncond_pred;

            if (!run_denoiser(hidden, temb_6d, time_embed, text_projected,
                              rope_cos, rope_sin, cond_pred, error)) {
                std::cerr << "[diffusion] Denoiser (cond) failed: " << error << "\n";
                return result;
            }
            if (!run_denoiser(hidden, temb_6d, time_embed, null_text,
                              rope_cos, rope_sin, uncond_pred, error)) {
                std::cerr << "[diffusion] Denoiser (uncond) failed: " << error << "\n";
                return result;
            }

            // noise_pred = uncond + guidance_scale * (cond - uncond)
            denoiser_output.resize(cond_pred.size());
            for (std::size_t i = 0; i < cond_pred.size(); ++i) {
                denoiser_output[i] = uncond_pred[i] +
                    guidance_scale * (cond_pred[i] - uncond_pred[i]);
            }
        } else {
            if (!run_denoiser(hidden, temb_6d, time_embed, text_projected,
                              rope_cos, rope_sin, denoiser_output, error)) {
                std::cerr << "[diffusion] Denoiser failed: " << error << "\n";
                return result;
            }
        }

        // Unpatchify: [num_patches, out_dim] -> [z_dim, t_lat, h_lat, w_lat]
        unpatchify(denoiser_output, z_dim, t_lat, h_lat, w_lat, noise_pred_spatial);

        // Scheduler step
        scheduler.step(noise_pred_spatial.data(), latents.data(), latents.data(),
                      latent_count, step);

        if (step % 10 == 0 || step == num_inference_steps - 1) {
            std::cerr << "  Step " << (step + 1) << "/"
                      << num_inference_steps
                      << " (t=" << timestep << ")\n";
        }
    }

    // 6. Denormalize latents
    if (!mConfig.latents_mean.empty() && !mConfig.latents_std.empty()) {
        for (int32_t ci = 0; ci < z_dim; ++ci) {
            const float mean = mConfig.latents_mean[static_cast<std::size_t>(ci)];
            const float std_val = mConfig.latents_std[static_cast<std::size_t>(ci)];
            const auto channel_size = static_cast<std::size_t>(t_lat * h_lat * w_lat);
            float* ch = latents.data() + static_cast<std::size_t>(ci) * channel_size;
            for (std::size_t i = 0; i < channel_size; ++i) {
                ch[i] = ch[i] * std_val + mean;
            }
        }
    }

    // 7. VAE decode (native TRT engine, fallback to subprocess)
    std::cerr << "[diffusion] Decoding video ...\n";
    if (mConfig.num_vae_caches > 0 && !mD_VaeCacheIn.empty()) {
        if (!decode_vae_native(latents, z_dim, t_lat, h_lat, w_lat, result, error)) {
            std::cerr << "[diffusion] Native VAE decode failed: " << error
                      << ", falling back to subprocess\n";
            if (!decode_vae_subprocess(latents, z_dim, t_lat, h_lat, w_lat, result, error)) {
                std::cerr << "[diffusion] VAE subprocess also failed: " << error << "\n";
                return result;
            }
        }
    } else {
        if (!decode_vae_subprocess(latents, z_dim, t_lat, h_lat, w_lat, result, error)) {
            std::cerr << "[diffusion] VAE decode failed: " << error << "\n";
            return result;
        }
    }

    result.num_frames = mConfig.video_num_frames;
    std::cerr << "[diffusion] Video generation complete: "
              << result.num_frames << " frames, "
              << result.height << "x" << result.width << "\n";
    return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<DiffusionBackend> CreateDiffusionBackend(
    std::vector<DiffusionEngine> text_encoders,
    DiffusionEngine denoiser,
    DiffusionEngine vae_decoder,
    const FastPathModelConfig& fp_cfg)
{
    DiffusionConfig config;
    config.scheduler = fp_cfg.scheduler;
    config.num_inference_steps = fp_cfg.num_inference_steps;
    config.guidance_scale = fp_cfg.guidance_scale;
    config.flow_shift = fp_cfg.flow_shift;
    config.video_height = fp_cfg.video_height;
    config.video_width = fp_cfg.video_width;
    config.video_num_frames = fp_cfg.video_num_frames;
    config.z_dim = fp_cfg.z_dim;
    config.scale_factor_temporal = fp_cfg.scale_factor_temporal;
    config.scale_factor_spatial = fp_cfg.scale_factor_spatial;
    config.dit_dim = fp_cfg.dit_dim;
    config.dit_num_heads = fp_cfg.dit_num_heads;
    config.freq_dim = fp_cfg.freq_dim;
    config.text_encoder_dim = fp_cfg.text_encoder_dim;
    config.text_seq_len = fp_cfg.text_seq_len;
    config.num_vae_caches = fp_cfg.num_vae_caches;
    config.latents_mean = fp_cfg.latents_mean;
    config.latents_std = fp_cfg.latents_std;
    config.patch_size = fp_cfg.patch_size;
    config.vae_model_id = fp_cfg.vae_model_id;

    return std::make_unique<DiffusionBackend>(
        std::move(text_encoders),
        std::move(denoiser),
        std::move(vae_decoder),
        std::move(config));
}

} // namespace trtf

#endif // TRTF_HAS_TRT

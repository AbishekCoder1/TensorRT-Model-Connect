#include "runtime/trt/diffusion/z_image_diffusion_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>

namespace trtf {

// ---------------------------------------------------------------------------
// Flow Match Euler Scheduler (mu-based shifting for Z-Image)
// ---------------------------------------------------------------------------
namespace {

struct FlowMatchEulerState {
    std::vector<double> sigmas;
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    float shift{3.0F};

    void set_timesteps(int32_t num_steps) {
        // Match HF FlowMatchEulerDiscreteScheduler exactly:
        // 1. HF pipeline sets scheduler.sigma_min = 0.0 before calling set_timesteps
        // 2. linspace(sigma_max*N, sigma_min*N, num_steps) to get timesteps
        // 3. sigmas = timesteps / N
        // 4. sigmas = shift * sigmas / (1 + (shift-1) * sigmas)
        const double s = static_cast<double>(shift);
        const double N = static_cast<double>(num_train_timesteps);
        const double t_max = 1.0 * N;           // = 1000.0
        const double t_min = 0.0;               // sigma_min = 0.0 (pipeline override)

        sigmas.resize(static_cast<std::size_t>(num_steps) + 1);
        for (int32_t i = 0; i < num_steps; ++i) {
            // linspace from t_max to t_min with num_steps points
            double t = t_max + static_cast<double>(i) /
                static_cast<double>(std::max(num_steps - 1, 1)) * (t_min - t_max);
            double base_sigma = t / N;
            double sigma = s * base_sigma / (1.0 + (s - 1.0) * base_sigma);
            sigmas[static_cast<std::size_t>(i)] = sigma;
        }
        sigmas[static_cast<std::size_t>(num_steps)] = 0.0;

        // Timesteps = sigmas * N
        timesteps.resize(static_cast<std::size_t>(num_steps));
        for (int32_t i = 0; i < num_steps; ++i) {
            timesteps[static_cast<std::size_t>(i)] =
                static_cast<float>(sigmas[static_cast<std::size_t>(i)] *
                                   static_cast<double>(num_train_timesteps));
        }
    }

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

} // anonymous namespace

// ---------------------------------------------------------------------------
// Z-Image preprocessor weight parsing
// ---------------------------------------------------------------------------
namespace {

bool extract_z_preprocessor_index(
    const std::vector<char>& data,
    std::string& json_str,
    const uint8_t*& blob,
    std::size_t& blob_size)
{
    if (data.size() < 4) {
        return false;
    }

    const auto* raw = reinterpret_cast<const uint8_t*>(data.data());
    uint32_t json_len = 0;
    std::memcpy(&json_len, raw, sizeof(uint32_t));
    if (4 + json_len > data.size()) {
        return false;
    }

    json_str.assign(reinterpret_cast<const char*>(raw + 4), json_len);
    blob = raw + 4 + json_len;
    blob_size = data.size() - 4 - json_len;
    return true;
}

bool parse_z_shape_list(
    const std::string& shape_expr,
    std::vector<int32_t>& shape)
{
    shape.clear();
    std::size_t pos = 0;
    while (pos < shape_expr.size()) {
        const auto num_start = shape_expr.find_first_of("0123456789", pos);
        if (num_start == std::string::npos) {
            break;
        }
        try {
            shape.push_back(std::stoi(shape_expr.substr(num_start)));
        } catch (const std::exception&) {
            return false;
        }
        const auto next = shape_expr.find_first_of(",]", num_start);
        if (next == std::string::npos) {
            break;
        }
        pos = next + 1;
    }
    return true;
}

bool parse_z_number_field(
    const std::string& json_str,
    std::size_t key_pos,
    const std::string& field,
    std::size_t& value)
{
    const auto fpos = json_str.find("\"" + field + "\"", key_pos);
    if (fpos == std::string::npos) {
        return false;
    }
    const auto colon = json_str.find(':', fpos + field.size() + 2);
    if (colon == std::string::npos) {
        return false;
    }
    const auto start = json_str.find_first_of("0123456789-", colon + 1);
    if (start == std::string::npos) {
        return false;
    }
    try {
        const auto parsed = std::stoll(json_str.substr(start));
        if (parsed < 0) {
            return false;
        }
        value = static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool find_z_entry(
    const std::string& json_str,
    const std::string& key,
    std::size_t& offset,
    std::vector<int32_t>& shape)
{
    const auto pos = json_str.find("\"" + key + "\"");
    if (pos == std::string::npos) {
        return false;
    }
    if (!parse_z_number_field(json_str, pos, "offset", offset)) {
        return false;
    }

    const auto shape_pos = json_str.find("\"shape\"", pos);
    if (shape_pos == std::string::npos) {
        return false;
    }
    const auto bracket = json_str.find('[', shape_pos);
    const auto bracket_end = json_str.find(']', bracket);
    if (bracket == std::string::npos || bracket_end == std::string::npos) {
        return false;
    }
    return parse_z_shape_list(
        json_str.substr(bracket + 1, bracket_end - bracket - 1),
        shape);
}

bool load_z_floats(
    const std::string& json_str,
    const uint8_t* blob,
    std::size_t blob_size,
    const std::string& key,
    std::vector<float>& dst)
{
    std::size_t offset = 0;
    std::vector<int32_t> shape;
    if (!find_z_entry(json_str, key, offset, shape)) {
        return false;
    }

    std::size_t count = 1;
    for (const auto s : shape) {
        count *= static_cast<std::size_t>(s);
    }
    const std::size_t nbytes = count * sizeof(float);
    if (offset + nbytes > blob_size) {
        std::cerr << "[z-image] weight " << key << " overflows blob\n";
        return false;
    }

    dst.resize(count);
    std::memcpy(dst.data(), blob + offset, nbytes);
    return true;
}

void load_z_preprocessor_tensors(
    const std::string& json_str,
    const uint8_t* blob,
    std::size_t blob_size,
    ZImagePreprocessorWeights& w)
{
    load_z_floats(json_str, blob, blob_size, "t_embedder.mlp.0.weight", w.t_emb_0_weight);
    load_z_floats(json_str, blob, blob_size, "t_embedder.mlp.0.bias", w.t_emb_0_bias);
    load_z_floats(json_str, blob, blob_size, "t_embedder.mlp.2.weight", w.t_emb_2_weight);
    load_z_floats(json_str, blob, blob_size, "t_embedder.mlp.2.bias", w.t_emb_2_bias);

    load_z_floats(json_str, blob, blob_size, "cap_embedder.norm.weight", w.cap_norm_weight);
    load_z_floats(json_str, blob, blob_size, "cap_embedder.proj.weight", w.cap_proj_weight);
    load_z_floats(json_str, blob, blob_size, "cap_embedder.proj.bias", w.cap_proj_bias);

    load_z_floats(json_str, blob, blob_size, "x_embedder.weight", w.x_embed_weight);
    load_z_floats(json_str, blob, blob_size, "x_embedder.bias", w.x_embed_bias);

    load_z_floats(json_str, blob, blob_size, "cap_pad_token", w.cap_pad_token);
    load_z_floats(json_str, blob, blob_size, "x_pad_token", w.x_pad_token);
}

void finalize_z_preprocessor_weights(ZImagePreprocessorWeights& w)
{
    if (!w.x_embed_weight.empty() && !w.x_embed_bias.empty()) {
        const auto dit_dim = static_cast<int32_t>(w.x_embed_bias.size());
        w.patch_dim = static_cast<int32_t>(w.x_embed_weight.size()) / dit_dim;
    }
    w.valid = !w.x_embed_weight.empty() && !w.t_emb_0_weight.empty();
}

ZImagePreprocessorWeights parse_z_image_preprocessor_weights(
    const std::vector<char>& data)
{
    ZImagePreprocessorWeights w;
    std::string json_str;
    const uint8_t* blob = nullptr;
    std::size_t blob_size = 0;
    if (!extract_z_preprocessor_index(data, json_str, blob, blob_size)) {
        return w;
    }

    load_z_preprocessor_tensors(json_str, blob, blob_size, w);
    finalize_z_preprocessor_weights(w);

    std::cerr << "[z-image] Preprocessor weights loaded: "
              << (w.valid ? "OK" : "INCOMPLETE")
              << " (patch_dim=" << w.patch_dim << ")\n";
    return w;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ZImageDiffusionBackend
// ---------------------------------------------------------------------------

ZImageDiffusionBackend::ZImageDiffusionBackend(
    std::vector<DiffusionEngine> text_encoders,
    DiffusionEngine denoiser,
    DiffusionEngine vae_decoder,
    DiffusionConfig config)
    : DiffusionBackendBase(
          std::move(text_encoders),
          std::move(denoiser),
          std::move(vae_decoder),
          std::move(config)),
      mD_DiTHidden(0), mD_DiTEncoder(0), mD_DiTTemb(0),
      mD_DiTCos(0), mD_DiTSin(0), mD_DiTOutput(0),
      mD_VaeInput(0), mD_VaeOutput(0)
{
    const int32_t dit_dim = mConfig.dit_dim;
    const int32_t head_dim = dit_dim / std::max(mConfig.dit_num_heads, 1);

    // Compute h_lat/w_lat: HF does h_lat = 2 * (H // (vae_scale * 2))
    const int32_t vae_scale_init = mConfig.scale_factor_spatial;
    const int32_t h_lat = 2 * (mConfig.video_height / (vae_scale_init * 2));
    const int32_t w_lat = 2 * (mConfig.video_width / (vae_scale_init * 2));

    int32_t ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t num_patches = (h_lat / ph) * (w_lat / pw);
    const int32_t text_seq = mConfig.text_seq_len;
    const int32_t total_seq = num_patches + text_seq;
    const int32_t adaln_dim = mConfig.freq_dim;
    const int32_t out_channels = mConfig.z_dim * ph * pw;

    // Z-Image-specific buffers
    mD_DiTHidden = CudaBuffer(
        static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_DiTEncoder = CudaBuffer(
        static_cast<std::size_t>(text_seq) * static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_DiTTemb = CudaBuffer(
        static_cast<std::size_t>(adaln_dim) * sizeof(float));
    mD_DiTCos = CudaBuffer(
        static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim) * sizeof(float));
    mD_DiTSin = CudaBuffer(
        static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim) * sizeof(float));
    mD_DiTOutput = CudaBuffer(
        static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(out_channels) * sizeof(float));

    init_vae_buffers();

    std::cerr << "[z-image] Z-Image 2D backend initialized"
              << " (height=" << mConfig.video_height
              << ", width=" << mConfig.video_width
              << ", h_lat=" << h_lat << ", w_lat=" << w_lat
              << ", patches=" << num_patches
              << ", steps=" << mConfig.num_inference_steps
              << ", cfg=" << mConfig.guidance_scale << ")\n";
}

void ZImageDiffusionBackend::init_vae_buffers()
{
    if (!mVaeDecoder.engine) return;

    // VAE works on h_lat x w_lat latents, output is h_lat*8 x w_lat*8
    const int32_t vae_scale = mConfig.scale_factor_spatial;
    const int32_t h_lat = 2 * (mConfig.video_height / (vae_scale * 2));
    const int32_t w_lat = 2 * (mConfig.video_width / (vae_scale * 2));
    const int32_t h_out = h_lat * 8;
    const int32_t w_out = w_lat * 8;

    mD_VaeInput = CudaBuffer(
        static_cast<std::size_t>(mConfig.z_dim) *
        static_cast<std::size_t>(h_lat) *
        static_cast<std::size_t>(w_lat) * sizeof(float));

    mD_VaeOutput = CudaBuffer(
        3 * static_cast<std::size_t>(h_out) *
        static_cast<std::size_t>(w_out) * sizeof(float));

    std::cerr << "[z-image] VAE buffers: input=" << mD_VaeInput.size()
              << ", output=" << mD_VaeOutput.size()
              << " (lat=" << h_lat << "x" << w_lat
              << ", out=" << h_out << "x" << w_out << ")\n";
}

// ---------------------------------------------------------------------------
// Timestep embedding
// ---------------------------------------------------------------------------

void ZImageDiffusionBackend::compute_timestep_embedding(
    float timestep,
    std::vector<float>& temb) const
{
    const int32_t freq_dim = mConfig.freq_dim;  // 256
    const int32_t half = freq_dim / 2;  // 128
    const float t = timestep;

    // Sinusoidal embedding: [cos(t*f0), cos(t*f1), ..., sin(t*f0), sin(t*f1), ...]
    std::vector<float> sinusoidal(static_cast<std::size_t>(freq_dim));
    for (int32_t i = 0; i < half; ++i) {
        const float freq = std::exp(
            -std::log(10000.0F) * static_cast<float>(i) /
            static_cast<float>(half));
        sinusoidal[static_cast<std::size_t>(i)] = std::cos(t * freq);
        sinusoidal[static_cast<std::size_t>(i + half)] = std::sin(t * freq);
    }

    // MLP: Linear(256, 1024) -> SiLU -> Linear(1024, 256)
    const auto& pp = mZWeights;
    const int32_t mid_dim = static_cast<int32_t>(pp.t_emb_0_bias.size());

    std::vector<float> h1(static_cast<std::size_t>(mid_dim));
    cpu_matmul_bias(sinusoidal.data(), pp.t_emb_0_weight.data(),
                    pp.t_emb_0_bias.data(), h1.data(), 1, freq_dim, mid_dim);
    cpu_silu_inplace(h1.data(), static_cast<std::size_t>(mid_dim));

    temb.resize(static_cast<std::size_t>(freq_dim));
    cpu_matmul_bias(h1.data(), pp.t_emb_2_weight.data(),
                    pp.t_emb_2_bias.data(), temb.data(), 1, mid_dim, freq_dim);
}

// ---------------------------------------------------------------------------
// Caption projection
// ---------------------------------------------------------------------------

void ZImageDiffusionBackend::project_caption(
    const std::vector<float>& text_embeddings,
    int32_t seq_len,
    std::vector<float>& projected) const
{
    const int32_t te_dim = mConfig.text_encoder_dim;
    const int32_t dit_dim = mConfig.dit_dim;
    const auto& pp = mZWeights;

    // RMSNorm(text_embeddings) using cap_norm_weight
    std::vector<float> normed(text_embeddings.size());
    for (int32_t s = 0; s < seq_len; ++s) {
        const float* row = text_embeddings.data() +
            static_cast<std::size_t>(s) * static_cast<std::size_t>(te_dim);
        float* out_row = normed.data() +
            static_cast<std::size_t>(s) * static_cast<std::size_t>(te_dim);

        double sum_sq = 0.0;
        for (int32_t d = 0; d < te_dim; ++d) {
            sum_sq += static_cast<double>(row[d]) * static_cast<double>(row[d]);
        }
        const float rms = std::sqrt(
            static_cast<float>(sum_sq / static_cast<double>(te_dim)) + 1e-5F);
        const float inv_rms = 1.0F / rms;

        for (int32_t d = 0; d < te_dim; ++d) {
            out_row[d] = row[d] * inv_rms * pp.cap_norm_weight[static_cast<std::size_t>(d)];
        }
    }

    // Linear(2560, 3840) + bias
    projected.resize(
        static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(dit_dim));
    cpu_matmul_bias(normed.data(), pp.cap_proj_weight.data(),
                    pp.cap_proj_bias.data(), projected.data(),
                    seq_len, te_dim, dit_dim);
}

// ---------------------------------------------------------------------------
// Patchify / Unpatchify (2D)
// ---------------------------------------------------------------------------

void ZImageDiffusionBackend::patchify_2d(
    const std::vector<float>& latents,
    int32_t c, int32_t h, int32_t w,
    std::vector<float>& patches) const
{
    int32_t ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t nh = h / ph;
    const int32_t nw = w / pw;
    // HF patchify: "c f pf h ph w pw -> (f h w) (pf ph pw c)"
    // For 2D (f=1, pf=1): "c 1 1 h ph w pw -> (h w) (ph pw c)"
    // So patch order is: for each (hy, wx): for each (dy, dx): for each channel c
    const int32_t patch_dim = ph * pw * c;  // Note: ph*pw*c, NOT c*ph*pw
    const int32_t num_patches = nh * nw;

    patches.resize(
        static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(patch_dim));

    // latents layout: [C, H, W] (CHW)
    for (int32_t hy = 0; hy < nh; ++hy) {
        for (int32_t wx = 0; wx < nw; ++wx) {
            const int32_t patch_idx = hy * nw + wx;
            float* dst = patches.data() +
                static_cast<std::size_t>(patch_idx) *
                static_cast<std::size_t>(patch_dim);

            // HF order: (pf ph pw c) -> iterate dy, dx, channel
            int32_t offset = 0;
            for (int32_t dy = 0; dy < ph; ++dy) {
                for (int32_t dx = 0; dx < pw; ++dx) {
                    for (int32_t ci = 0; ci < c; ++ci) {
                        const int32_t y = hy * ph + dy;
                        const int32_t x = wx * pw + dx;
                        const auto src_idx =
                            static_cast<std::size_t>(ci) *
                            static_cast<std::size_t>(h * w) +
                            static_cast<std::size_t>(y * w + x);
                        dst[offset++] = latents[src_idx];
                    }
                }
            }
        }
    }
}

void ZImageDiffusionBackend::unpatchify_2d(
    const std::vector<float>& patches,
    int32_t c, int32_t h, int32_t w,
    std::vector<float>& output) const
{
    int32_t ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t nh = h / ph;
    const int32_t nw = w / pw;
    // HF unpatchify: "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
    // For 2D (f=1, pf=1): patches are [num_patches, pf*ph*pw*c]
    // viewed as [1, nh, nw, 1, ph, pw, c] then permuted to [c, 1, h, w]
    const int32_t patch_dim = ph * pw * c;

    output.resize(static_cast<std::size_t>(c) *
                  static_cast<std::size_t>(h) *
                  static_cast<std::size_t>(w));

    for (int32_t hy = 0; hy < nh; ++hy) {
        for (int32_t wx = 0; wx < nw; ++wx) {
            const int32_t patch_idx = hy * nw + wx;
            const float* src = patches.data() +
                static_cast<std::size_t>(patch_idx) *
                static_cast<std::size_t>(patch_dim);

            int32_t offset = 0;
            for (int32_t dy = 0; dy < ph; ++dy) {
                for (int32_t dx = 0; dx < pw; ++dx) {
                    for (int32_t ci = 0; ci < c; ++ci) {
                        const int32_t y = hy * ph + dy;
                        const int32_t x = wx * pw + dx;
                        const auto dst_idx =
                            static_cast<std::size_t>(ci) *
                            static_cast<std::size_t>(h * w) +
                            static_cast<std::size_t>(y * w + x);
                        output[dst_idx] = src[offset++];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3-axis RoPE (matching HF complex-number RoPE)
// ---------------------------------------------------------------------------

void ZImageDiffusionBackend::compute_3d_rope(
    int32_t nh, int32_t nw, int32_t text_seq_len,
    int32_t cap_padded_len,
    std::vector<float>& cos_out,
    std::vector<float>& sin_out) const
{
    const int32_t head_dim = mConfig.dit_dim / std::max(mConfig.dit_num_heads, 1);
    const int32_t num_patches = nh * nw;
    const int32_t total_seq = num_patches + text_seq_len;

    // Z-Image: axes_dims = [32, 48, 48], theta = 256
    const float theta = 256.0F;
    const int32_t dim_t = 32;
    const int32_t dim_h = 48;
    const int32_t dim_w = 48;

    // Initialize cos=1, sin=0 (identity RoPE)
    cos_out.assign(static_cast<std::size_t>(total_seq) *
                   static_cast<std::size_t>(head_dim), 1.0F);
    sin_out.assign(static_cast<std::size_t>(total_seq) *
                   static_cast<std::size_t>(head_dim), 0.0F);

    // HF RoPE uses complex numbers: freqs = 1/(theta^(2i/d)) for i in 0..d/2
    // Then freqs_cis = e^(i * pos * freqs) = cos(pos*freqs) + i*sin(pos*freqs)
    // Applied as x_complex * freqs_cis where x is viewed as pairs of reals.
    // This is equivalent to rotate-half with interleaved pairs:
    //   cos_row[2*i] = cos(angle), cos_row[2*i+1] = cos(angle)
    //   sin_row[2*i] = -sin(angle), sin_row[2*i+1] = sin(angle)
    auto encode_pos = [&](float* cos_row, float* sin_row,
                          int32_t t_pos, int32_t h_pos, int32_t w_pos) {
        int32_t offset = 0;

        // Time dimension (dim_t/2 pairs)
        // rotate_half matrix already negates the odd element: [a,b] -> [-b,a]
        // So result = [a*cos + (-b)*sin, b*cos + a*sin] = [a*cos - b*sin, b*cos + a*sin]
        // which matches complex multiplication (a+bi)(cos+i*sin).
        // Both cos and sin entries must be POSITIVE (no negation).
        for (int32_t i = 0; i < dim_t / 2; ++i) {
            const float freq = 1.0F / std::pow(
                theta, 2.0F * static_cast<float>(i) / static_cast<float>(dim_t));
            const float angle = static_cast<float>(t_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
        offset += dim_t;

        // Height dimension (dim_h/2 pairs)
        for (int32_t i = 0; i < dim_h / 2; ++i) {
            const float freq = 1.0F / std::pow(
                theta, 2.0F * static_cast<float>(i) / static_cast<float>(dim_h));
            const float angle = static_cast<float>(h_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
        offset += dim_h;

        // Width dimension (dim_w/2 pairs)
        for (int32_t i = 0; i < dim_w / 2; ++i) {
            const float freq = 1.0F / std::pow(
                theta, 2.0F * static_cast<float>(i) / static_cast<float>(dim_w));
            const float angle = static_cast<float>(w_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
    };

    // HF position IDs for noise tokens:
    //   image_ori_pos_ids = create_coordinate_grid(
    //       size=(1, H_tokens, W_tokens),
    //       start=(cap_padded_len + 1, 0, 0))
    // cap_padded_len is the actual caption tokens padded to next multiple of 32.
    const int32_t noise_t_start = cap_padded_len + 1;

    for (int32_t hy = 0; hy < nh; ++hy) {
        for (int32_t wx = 0; wx < nw; ++wx) {
            const int32_t idx = hy * nw + wx;
            encode_pos(
                cos_out.data() + static_cast<std::size_t>(idx) *
                    static_cast<std::size_t>(head_dim),
                sin_out.data() + static_cast<std::size_t>(idx) *
                    static_cast<std::size_t>(head_dim),
                noise_t_start, hy, wx);
        }
    }

    // HF position IDs for caption tokens:
    //   cap_padded_pos_ids = create_coordinate_grid(
    //       size=(cap_padded_len, 1, 1), start=(1, 0, 0))
    // Only the first cap_padded_len positions get valid RoPE.
    // Remaining positions (cap_padded_len..text_seq_len) keep identity (cos=1, sin=0).
    for (int32_t t = 0; t < cap_padded_len; ++t) {
        const int32_t idx = num_patches + t;
        encode_pos(
            cos_out.data() + static_cast<std::size_t>(idx) *
                static_cast<std::size_t>(head_dim),
            sin_out.data() + static_cast<std::size_t>(idx) *
                static_cast<std::size_t>(head_dim),
            t + 1, 0, 0);  // t starts at 1
    }
}

// ---------------------------------------------------------------------------
// VAE native decode
// ---------------------------------------------------------------------------

namespace {

float clamp_z_image_unit(float value)
{
    return std::max(0.0F, std::min(1.0F, value));
}

void convert_z_vae_output(
    const std::vector<float>& raw,
    int32_t h_out,
    int32_t w_out,
    VideoResult& result)
{
    result.height = h_out;
    result.width = w_out;
    result.num_frames = 1;
    result.frames.resize(
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out) * 3);
    for (int32_t y = 0; y < h_out; ++y) {
        for (int32_t x = 0; x < w_out; ++x) {
            for (int32_t ch = 0; ch < 3; ++ch) {
                const auto src_idx =
                    static_cast<std::size_t>(ch) *
                    static_cast<std::size_t>(h_out * w_out) +
                    static_cast<std::size_t>(y * w_out + x);
                const auto dst_idx =
                    static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(w_out * 3) +
                    static_cast<std::size_t>(x * 3 + ch);
                const float v = (raw[src_idx] + 1.0F) * 0.5F;
                result.frames[dst_idx] = clamp_z_image_unit(v);
            }
        }
    }
}

bool run_z_vae_decode(
    nvinfer1::IExecutionContext& ctx,
    CudaBuffer& d_vae_input,
    CudaBuffer& d_vae_output,
    CudaStream& stream,
    const std::vector<float>& latents,
    int32_t c,
    int32_t h_lat,
    int32_t w_lat,
    VideoResult& result,
    std::string& error)
{
    const int32_t h_out = h_lat * 8;
    const int32_t w_out = w_lat * 8;
    const auto input_size = static_cast<std::size_t>(c) *
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);

    cudaMemcpyAsync(d_vae_input.data(), latents.data(),
        input_size * sizeof(float), cudaMemcpyHostToDevice, stream.get());
    ctx.setInputShape("latent_input", nvinfer1::Dims4(1, c, h_lat, w_lat));
    ctx.setTensorAddress("latent_input", d_vae_input.data());
    ctx.setTensorAddress("decoder_output", d_vae_output.data());
    if (!ctx.enqueueV3(stream.get())) {
        error = "VAE enqueueV3 failed";
        return false;
    }

    const auto output_size = static_cast<std::size_t>(3) *
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    std::vector<float> raw(output_size);
    cudaMemcpyAsync(raw.data(), d_vae_output.data(),
        output_size * sizeof(float), cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());
    convert_z_vae_output(raw, h_out, w_out, result);
    return true;
}

} // namespace

bool ZImageDiffusionBackend::decode_vae_native(
    const std::vector<float>& latents,
    int32_t c, int32_t h_lat, int32_t w_lat,
    VideoResult& result, std::string& error)
{
    if (!mVaeDecoder.engine || !mVaeDecoder.context) {
        error = "No VAE decoder engine";
        return false;
    }

    const auto input_size = static_cast<std::size_t>(c) *
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    if (latents.size() < input_size) {
        error = "Latents too small for VAE";
        return false;
    }

    return run_z_vae_decode(
        *mVaeDecoder.context,
        mD_VaeInput,
        mD_VaeOutput,
        mStream,
        latents,
        c,
        h_lat,
        w_lat,
        result,
        error);
}

// ---------------------------------------------------------------------------
// Preprocessor weight loading
// ---------------------------------------------------------------------------

void ZImageDiffusionBackend::set_preprocessor_weights(PreprocessorWeights /*weights*/)
{
    // Z-Image uses its own weight format; this override is intentionally empty.
}

void ZImageDiffusionBackend::load_z_image_preprocessor_weights(
    const std::vector<char>& raw_data)
{
    mZWeights = parse_z_image_preprocessor_weights(raw_data);
    if (!mZWeights.valid) {
        std::cerr << "[z-image] WARNING: Failed to parse Z-Image preprocessor weights\n";
    }
}

// ---------------------------------------------------------------------------
// Chat template (Qwen3 format for Z-Image text encoder)
// ---------------------------------------------------------------------------

std::string ZImageDiffusionBackend::prepare_prompt(const std::string& prompt) const
{
    // HF ZImagePipeline._encode_prompt wraps the prompt in Qwen3 chat template:
    //   tokenizer.apply_chat_template(
    //       [{"role": "user", "content": prompt}],
    //       tokenize=False, add_generation_prompt=True, enable_thinking=True)
    // Which produces: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

namespace {

constexpr int32_t kZImagePadTokenId = 151643;
constexpr int32_t kZImageSeqMultiple = 32;

int32_t resolve_z_image_steps(int32_t requested, int32_t fallback)
{
    return requested <= 0 ? fallback : requested;
}

float resolve_z_image_guidance(float requested, float fallback)
{
    return requested < 0.0F ? fallback : requested;
}

struct ZImageLayout {
    int32_t dit_dim{0};
    int32_t text_seq{0};
    int32_t z_dim{0};
    int32_t h_lat{0};
    int32_t w_lat{0};
    int32_t ph{2};
    int32_t pw{2};
    int32_t nh{0};
    int32_t nw{0};
    int32_t num_patches{0};
    int32_t patch_dim{0};
};

ZImageLayout make_z_image_layout(const DiffusionConfig& config)
{
    ZImageLayout layout;
    layout.dit_dim = config.dit_dim;
    layout.text_seq = config.text_seq_len;
    layout.z_dim = config.z_dim;

    const int32_t vae_scale = config.scale_factor_spatial;
    layout.h_lat = 2 * (config.video_height / (vae_scale * 2));
    layout.w_lat = 2 * (config.video_width / (vae_scale * 2));

    if (config.patch_size.size() >= 3) {
        layout.ph = config.patch_size[1];
        layout.pw = config.patch_size[2];
    }
    layout.nh = layout.h_lat / layout.ph;
    layout.nw = layout.w_lat / layout.pw;
    layout.num_patches = layout.nh * layout.nw;
    layout.patch_dim = layout.ph * layout.pw * layout.z_dim;
    return layout;
}

void log_z_image_layout(const ZImageLayout& layout)
{
    std::cerr << "[z-image] Latent: " << layout.h_lat << "x" << layout.w_lat
              << ", patches: " << layout.num_patches
              << " (" << layout.nh << "x" << layout.nw << ")\n";
}

int32_t count_z_non_pad_tokens(const std::vector<int32_t>& input_ids)
{
    int32_t count = 0;
    for (const auto id : input_ids) {
        if (id != kZImagePadTokenId) {
            ++count;
        }
    }
    return count;
}

int32_t pad_to_next_multiple(int32_t value, int32_t multiple)
{
    const int32_t rem = value % multiple;
    return rem == 0 ? value : value + (multiple - rem);
}

void apply_z_caption_padding(
    std::vector<float>& caption_projected,
    const std::vector<float>& cap_pad_token,
    int32_t cap_ori_len,
    int32_t text_seq,
    int32_t dit_dim)
{
    for (int32_t t = cap_ori_len; t < text_seq; ++t) {
        float* row = caption_projected.data() +
            static_cast<std::size_t>(t) * static_cast<std::size_t>(dit_dim);
        for (int32_t d = 0; d < dit_dim; ++d) {
            row[d] = cap_pad_token[static_cast<std::size_t>(
                d % static_cast<int32_t>(cap_pad_token.size()))];
        }
    }
}

void initialize_z_image_latents(std::vector<float>& latents)
{
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (auto& v : latents) {
        v = dist(gen);
    }
}

void log_z_scheduler(
    const FlowMatchEulerState& scheduler,
    int32_t num_inference_steps)
{
    std::cerr << "[z-image] Scheduler: shift=" << scheduler.shift
              << ", timesteps=[";
    for (int32_t i = 0; i < num_inference_steps; ++i) {
        if (i > 0) {
            std::cerr << ", ";
        }
        std::cerr << scheduler.timesteps[static_cast<std::size_t>(i)];
    }
    std::cerr << "]\n";
}

void negate_inplace(std::vector<float>& values)
{
    for (auto& v : values) {
        v = -v;
    }
}

void log_z_step_stats(
    int32_t step,
    int32_t num_inference_steps,
    float raw_timestep,
    const std::vector<float>& latents)
{
    float lat_min = latents[0];
    float lat_max = latents[0];
    double lat_sum = 0.0;
    for (const auto v : latents) {
        lat_min = std::min(lat_min, v);
        lat_max = std::max(lat_max, v);
        lat_sum += static_cast<double>(v);
    }
    std::cerr << "  Step " << (step + 1) << "/"
              << num_inference_steps
              << " (t=" << raw_timestep
              << ") lat=[" << lat_min << ", " << lat_max
              << "] mean=" << (lat_sum / static_cast<double>(latents.size())) << "\n";
}

template <typename ComputeTembFn, typename PatchifyFn, typename EmbedHiddenFn,
          typename RunDenoiserFn, typename UnpatchifyFn>
bool run_z_image_denoising_loop(
    FlowMatchEulerState& scheduler,
    int32_t num_inference_steps,
    std::vector<float>& latents,
    std::string& error,
    ComputeTembFn&& compute_temb,
    PatchifyFn&& patchify,
    EmbedHiddenFn&& embed_hidden,
    RunDenoiserFn&& run_denoiser,
    UnpatchifyFn&& unpatchify)
{
    std::vector<float> temb;
    std::vector<float> patches;
    std::vector<float> hidden;
    std::vector<float> denoiser_output;
    std::vector<float> noise_pred;

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float raw_timestep = scheduler.timesteps[static_cast<std::size_t>(step)];
        const float t_for_embedding = 1000.0F - raw_timestep;
        compute_temb(t_for_embedding, temb);
        patchify(latents, patches);
        embed_hidden(patches, hidden);
        if (!run_denoiser(hidden, temb, denoiser_output, error, step)) {
            return false;
        }
        unpatchify(denoiser_output, noise_pred);
        negate_inplace(noise_pred);
        scheduler.step(noise_pred.data(), latents.data(), latents.data(),
            latents.size(), step);
        log_z_step_stats(step, num_inference_steps, raw_timestep, latents);
    }
    return true;
}

void denormalize_z_latents(std::vector<float>& latents)
{
    constexpr float kVaeScalingFactor = 0.3611F;
    constexpr float kVaeShiftFactor = 0.1159F;
    const float inv_scale = 1.0F / kVaeScalingFactor;
    for (auto& v : latents) {
        v = v * inv_scale + kVaeShiftFactor;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Generate image (full pipeline)
// ---------------------------------------------------------------------------

VideoResult ZImageDiffusionBackend::generate_video(
    const std::vector<int32_t>& input_ids,
    int32_t num_inference_steps,
    float guidance_scale)
{
    VideoResult result;
    std::string error;

    num_inference_steps = resolve_z_image_steps(
        num_inference_steps, mConfig.num_inference_steps);
    guidance_scale = resolve_z_image_guidance(
        guidance_scale, mConfig.guidance_scale);
    (void)guidance_scale;

    result.height = mConfig.video_height;
    result.width = mConfig.video_width;
    result.num_frames = 1;

    const ZImageLayout layout = make_z_image_layout(mConfig);
    log_z_image_layout(layout);

    if (!mZWeights.valid) {
        std::cerr << "[z-image] WARNING: Z-Image preprocessor weights not loaded.\n";
        return result;
    }

    std::cerr << "[z-image] Running text encoder ...\n";
    std::vector<float> text_embeddings;
    if (!run_t5_encoder(input_ids, text_embeddings, error)) {
        std::cerr << "[z-image] Text encoder failed: " << error << "\n";
        return result;
    }
    std::cerr << "[z-image] Text encoder done\n";

    const int32_t cap_ori_len = count_z_non_pad_tokens(input_ids);
    const int32_t cap_padded_len =
        pad_to_next_multiple(cap_ori_len, kZImageSeqMultiple);

    std::cerr << "[z-image] Caption: " << cap_ori_len << " actual tokens, "
              << cap_padded_len << " padded (SEQ_MULTI_OF=" << kZImageSeqMultiple << ")\n";

    std::vector<float> caption_projected;
    project_caption(text_embeddings, layout.text_seq, caption_projected);
    if (!mZWeights.cap_pad_token.empty()) {
        apply_z_caption_padding(
            caption_projected,
            mZWeights.cap_pad_token,
            cap_ori_len,
            layout.text_seq,
            layout.dit_dim);
    }

    std::vector<float> rope_cos, rope_sin;
    compute_3d_rope(
        layout.nh, layout.nw, layout.text_seq, cap_padded_len, rope_cos, rope_sin);

    const auto latent_size = static_cast<std::size_t>(layout.z_dim) *
        static_cast<std::size_t>(layout.h_lat) *
        static_cast<std::size_t>(layout.w_lat);
    std::vector<float> latents(latent_size);
    initialize_z_image_latents(latents);

    FlowMatchEulerState scheduler;
    scheduler.shift = mConfig.flow_shift;
    scheduler.set_timesteps(num_inference_steps);
    log_z_scheduler(scheduler, num_inference_steps);

    std::cerr << "[z-image] Starting denoising loop ("
              << num_inference_steps << " steps) ...\n";

    const auto compute_temb = [this](
        float timestep,
        std::vector<float>& temb) {
        compute_timestep_embedding(timestep, temb);
    };
    const auto patchify = [this, &layout](
        const std::vector<float>& latents_in,
        std::vector<float>& patches) {
        patchify_2d(latents_in, layout.z_dim, layout.h_lat, layout.w_lat, patches);
    };
    const auto embed_hidden = [this, &layout](
        const std::vector<float>& patches,
        std::vector<float>& hidden) {
        hidden.resize(
            static_cast<std::size_t>(layout.num_patches) *
            static_cast<std::size_t>(layout.dit_dim));
        cpu_matmul_bias(
            patches.data(),
            mZWeights.x_embed_weight.data(),
            mZWeights.x_embed_bias.data(),
            hidden.data(),
            layout.num_patches,
            layout.patch_dim,
            layout.dit_dim);
    };
    const auto run_denoiser = [this, &layout, &caption_projected, &rope_cos, &rope_sin](
        const std::vector<float>& hidden,
        const std::vector<float>& temb,
        std::vector<float>& denoiser_output,
        std::string& err,
        int32_t step) {
        cudaMemcpyAsync(mD_DiTHidden.data(), hidden.data(),
            hidden.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
        cudaMemcpyAsync(mD_DiTEncoder.data(), caption_projected.data(),
            caption_projected.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
        cudaMemcpyAsync(mD_DiTTemb.data(), temb.data(),
            temb.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
        cudaMemcpyAsync(mD_DiTCos.data(), rope_cos.data(),
            rope_cos.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
        cudaMemcpyAsync(mD_DiTSin.data(), rope_sin.data(),
            rope_sin.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());

        auto& ctx = mDenoiser.context;
        ctx->setTensorAddress("hidden_states", mD_DiTHidden.data());
        ctx->setTensorAddress("encoder_hidden_states", mD_DiTEncoder.data());
        ctx->setTensorAddress("timestep_embedding", mD_DiTTemb.data());
        ctx->setTensorAddress("rotary_cos", mD_DiTCos.data());
        ctx->setTensorAddress("rotary_sin", mD_DiTSin.data());
        ctx->setTensorAddress("output", mD_DiTOutput.data());
        if (!ctx->enqueueV3(mStream.get())) {
            err = "DiT enqueueV3 failed at step " + std::to_string(step);
            return false;
        }

        const auto out_size = static_cast<std::size_t>(layout.num_patches) *
            static_cast<std::size_t>(layout.patch_dim);
        denoiser_output.resize(out_size);
        cudaMemcpyAsync(denoiser_output.data(), mD_DiTOutput.data(),
            out_size * sizeof(float), cudaMemcpyDeviceToHost, mStream.get());
        cudaStreamSynchronize(mStream.get());
        return true;
    };
    const auto unpatchify = [this, &layout](
        const std::vector<float>& denoiser_output,
        std::vector<float>& noise_pred) {
        unpatchify_2d(
            denoiser_output, layout.z_dim, layout.h_lat, layout.w_lat, noise_pred);
    };

    if (!run_z_image_denoising_loop(
            scheduler,
            num_inference_steps,
            latents,
            error,
            compute_temb,
            patchify,
            embed_hidden,
            run_denoiser,
            unpatchify)) {
        std::cerr << "[z-image] " << error << "\n";
        return result;
    }

    denormalize_z_latents(latents);

    std::cerr << "[z-image] Decoding latents via native VAE ...\n";
    if (!decode_vae_native(
            latents, layout.z_dim, layout.h_lat, layout.w_lat, result, error)) {
        std::cerr << "[z-image] VAE decode failed: " << error << "\n";
        return result;
    }

    result.num_frames = 1;
    std::cerr << "[z-image] Image generated: "
              << result.width << "x" << result.height << "\n";
    return result;
}

} // namespace trtf

#endif // TRTF_HAS_TRT

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
#include <sstream>

namespace trtf {

// ---------------------------------------------------------------------------
// CPU math helpers
// ---------------------------------------------------------------------------

void DiffusionBackendBase::cpu_matmul_bias(
    const float* A, const float* B, const float* bias,
    float* out, int32_t M, int32_t K, int32_t N)
{
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

void DiffusionBackendBase::cpu_silu_inplace(float* data, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) {
        const float x = data[i];
        data[i] = x / (1.0F + std::exp(-x));
    }
}

void DiffusionBackendBase::cpu_gelu_tanh_inplace(float* data, std::size_t count)
{
    constexpr float kSqrt2OverPi = 0.7978845608F;
    constexpr float kCoeff = 0.044715F;
    for (std::size_t i = 0; i < count; ++i) {
        const float x = data[i];
        const float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
        data[i] = 0.5F * x * (1.0F + std::tanh(inner));
    }
}

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

    uint32_t index_len = 0;
    std::memcpy(&index_len, data.data(), 4);

    if (4 + index_len > data.size()) {
        std::cerr << "[diffusion] preprocessor_weights index length overflow\n";
        return w;
    }

    const std::string index_json(data.data() + 4, data.data() + 4 + index_len);
    const char* blob = data.data() + 4 + index_len;
    const std::size_t blob_size = data.size() - 4 - index_len;

    auto find_entry = [&](const std::string& key,
                          std::size_t& offset, std::vector<int32_t>& shape) -> bool {
        const std::string search = "\"" + key + "\"";
        auto pos = index_json.find(search);
        if (pos == std::string::npos) return false;

        auto off_pos = index_json.find("\"offset\"", pos);
        if (off_pos == std::string::npos) return false;
        auto colon = index_json.find(':', off_pos + 8);
        if (colon == std::string::npos) return false;
        offset = static_cast<std::size_t>(std::stoul(index_json.substr(colon + 1)));

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
// DiffusionBackendBase
// ---------------------------------------------------------------------------

DiffusionBackendBase::DiffusionBackendBase(
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
{
    mOk = mStream.ok();
    if (!mOk) {
        std::cerr << "[diffusion] CUDA stream creation failed\n";
        return;
    }

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

    mD_InputIds = CudaBuffer(
        static_cast<std::size_t>(seq_len) * sizeof(int32_t));
    mD_AttentionMask = CudaBuffer(
        static_cast<std::size_t>(seq_len) * sizeof(float));
    mD_TextEmbeddings = CudaBuffer(
        static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(te_dim) * sizeof(float));

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
}

void DiffusionBackendBase::set_preprocessor_weights(PreprocessorWeights weights)
{
    mWeights = std::move(weights);
}

bool DiffusionBackendBase::is_available() const
{
    return mOk;
}

const char* DiffusionBackendBase::name() const
{
    return "trt_diffusion";
}

std::vector<int32_t> DiffusionBackendBase::generate(
    const std::vector<int32_t>& /*input_ids*/,
    const GenerationConfig& /*config*/)
{
    return {};
}

// ---------------------------------------------------------------------------
// T5 encoder execution
// ---------------------------------------------------------------------------

bool DiffusionBackendBase::run_t5_encoder(
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

    std::vector<int32_t> padded_ids(static_cast<std::size_t>(seq_len), 0);
    const auto copy_len = std::min(
        static_cast<std::size_t>(seq_len), input_ids.size());
    std::copy_n(input_ids.begin(), copy_len, padded_ids.begin());

    std::vector<float> mask(static_cast<std::size_t>(seq_len), -1e9F);
    for (int32_t i = 0; i < seq_len; ++i) {
        if (padded_ids[static_cast<std::size_t>(i)] != 0) {
            mask[static_cast<std::size_t>(i)] = 0.0F;
        }
    }

    cudaMemcpyAsync(mD_InputIds.data(), padded_ids.data(),
        padded_ids.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_AttentionMask.data(), mask.data(),
        mask.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());

    te.context->setTensorAddress("input_ids", mD_InputIds.data());
    te.context->setTensorAddress("attention_mask", mD_AttentionMask.data());
    te.context->setTensorAddress("text_embeddings", mD_TextEmbeddings.data());

    if (!te.context->enqueueV3(mStream.get())) {
        error = "T5 enqueueV3 failed";
        return false;
    }

    const auto emb_size = static_cast<std::size_t>(seq_len) *
                          static_cast<std::size_t>(te_dim);
    text_embeddings.resize(emb_size);
    cudaMemcpyAsync(text_embeddings.data(), mD_TextEmbeddings.data(),
        emb_size * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

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
// DiT denoiser execution
// ---------------------------------------------------------------------------

bool DiffusionBackendBase::run_denoiser(
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

    ctx->setTensorAddress("hidden_states", mD_Hidden.data());
    ctx->setTensorAddress("timestep_embedding", mD_Temb.data());
    ctx->setTensorAddress("time_embed", mD_TimeEmbed.data());
    ctx->setTensorAddress("encoder_hidden_states", mD_EncoderHidden.data());
    ctx->setTensorAddress("rotary_cos", mD_RotaryCos.data());
    ctx->setTensorAddress("rotary_sin", mD_RotarySin.data());
    ctx->setTensorAddress("output", mD_Output.data());

    if (!ctx->enqueueV3(mStream.get())) {
        error = "DiT enqueueV3 failed";
        return false;
    }

    const auto out_size = mD_Output.size() / sizeof(float);
    output.resize(out_size);
    cudaMemcpyAsync(output.data(), mD_Output.data(),
        mD_Output.size(),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    return true;
}

// ---------------------------------------------------------------------------
// VAE decode via Python subprocess (shared fallback)
// ---------------------------------------------------------------------------

bool DiffusionBackendBase::decode_vae_subprocess(
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

    const std::string shape_str = "1," + std::to_string(c) + "," +
        std::to_string(t) + "," + std::to_string(h) + "," + std::to_string(w);

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

                    float v = (raw[src] + 1.0F) * 0.5F;
                    v = std::max(0.0F, std::min(1.0F, v));
                    result.frames[dst] = v;
                }
            }
        }
    }

    std::filesystem::remove(lat_file);
    std::filesystem::remove(out_file);

    return true;
}

} // namespace trtf

#endif // TRTF_HAS_TRT

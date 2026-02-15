#include "trtf/bundle.h"
#include "trtf/pipeline.h"
#include "bundle/bundle_format.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace trtf {

void BuildBundle(const std::string& model_dir,
    const std::string& output_path,
    int max_cache_length,
    const std::string& hf_python)
{
    TrtfPipelineOptions opts{};
    opts.flags = TRTF_FORCE_TRT;
    opts.max_new_tokens = 0;
    opts.max_cache_length = max_cache_length;
    opts.hf_python = hf_python.empty() ? nullptr : hf_python.c_str();
    opts.engine_cache_dir = nullptr;
    opts.no_engine_cache = 0;

    auto* pipeline = trtf_create_pipeline_ex(model_dir.c_str(), &opts);
    if (pipeline == nullptr)
    {
        throw std::runtime_error(std::string("BuildBundle: failed to create pipeline: ") + trtf_last_error());
    }

    std::unique_ptr<IPipeline> guard(pipeline);
    if (!pipeline->save_bundle(output_path.c_str()))
    {
        throw std::runtime_error("BuildBundle: save_bundle failed (engine serialization not available for this backend)");
    }
}

} // namespace trtf

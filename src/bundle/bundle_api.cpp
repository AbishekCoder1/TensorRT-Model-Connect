#include "trtf/bundle.h"
#include "bundle/bundle_format.h"

#include <stdexcept>
#include <string>

namespace trtf {

void BuildBundle(const std::string& model_dir,
    const std::string& output_path,
    int max_cache_length)
{
    // Full implementation requires TRT engine compilation + serialization.
    // This will be wired up when TRT engine serialization is added to PipelineImpl::save_bundle.
    (void) model_dir;
    (void) output_path;
    (void) max_cache_length;
    throw std::runtime_error("BuildBundle: TRT engine serialization not yet implemented. "
                             "Use trtf_create_pipeline() + save_bundle() when TRT is available.");
}

} // namespace trtf

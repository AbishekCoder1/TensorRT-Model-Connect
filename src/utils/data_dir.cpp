#include "utils/data_dir.h"

#include <string>

#ifndef TRTF_SOURCE_DIR
#define TRTF_SOURCE_DIR "."
#endif

namespace trtf {

// Process-wide source-dir override. Populated by
// set_source_dir_override(...) from pipeline_factory after resolving the
// platform.* registry namespace. Empty (the default) means "use the
// compile-time TRTF_SOURCE_DIR." Replaces the old TRTF_DATA_DIR env var.
static std::string& mutable_source_dir_override() {
    static std::string value;
    return value;
}

void set_source_dir_override(const std::string& value)
{
    mutable_source_dir_override() = value;
}

std::string source_dir()
{
    const auto& override_value = mutable_source_dir_override();
    if (!override_value.empty())
        return override_value;
    return TRTF_SOURCE_DIR;
}

std::string scripts_dir()
{
    return source_dir() + "/scripts";
}

std::string models_dir()
{
    return source_dir() + "/models";
}

std::string script_path(const char* script_name)
{
    return scripts_dir() + "/" + script_name;
}

std::string model_path(const char* relative_path)
{
    return models_dir() + "/" + relative_path;
}

} // namespace trtf

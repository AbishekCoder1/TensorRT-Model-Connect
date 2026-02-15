#include "utils/data_dir.h"

#include <cstdlib>
#include <string>

#ifndef TRTF_SOURCE_DIR
#define TRTF_SOURCE_DIR "."
#endif

namespace trtf {

std::string source_dir()
{
    const char* env = std::getenv("TRTF_DATA_DIR");
    if (env != nullptr && env[0] != '\0')
    {
        return env;
    }
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

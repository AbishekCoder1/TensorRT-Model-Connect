#pragma once

#include <string>

namespace trtf {

// Returns the project source directory, resolved in this order:
// 1. TRTF_DATA_DIR environment variable (if set)
// 2. TRTF_SOURCE_DIR compile-time define
std::string source_dir();

// Returns path to scripts/ directory.
std::string scripts_dir();

// Returns path to models/ directory.
std::string models_dir();

// Resolves a script by name under scripts_dir().
std::string script_path(const char* script_name);

// Resolves a model directory under models_dir().
std::string model_path(const char* relative_path);

} // namespace trtf

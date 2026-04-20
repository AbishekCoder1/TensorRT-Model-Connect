#pragma once

// Schema for the "text_trace" namespace. Mirrors
// trtf_build/trtf_build/runtime_config/schemas/text_trace.py one-for-one.

#include "trtf/config/schema_registry.h"

namespace trtf::config::schemas {

Schema make_text_trace_schema();

} // namespace trtf::config::schemas

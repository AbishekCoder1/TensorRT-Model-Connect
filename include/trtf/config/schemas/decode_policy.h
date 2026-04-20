#pragma once

// Schema for the "decode_policy" namespace. Mirrors
// trtf_build/trtf_build/runtime_config/schemas/decode_policy.py one-for-one.
// The cross-language field-set match test gates on it.

#include "trtf/config/schema_registry.h"

namespace trtf::config::schemas {

Schema make_decode_policy_schema();

} // namespace trtf::config::schemas

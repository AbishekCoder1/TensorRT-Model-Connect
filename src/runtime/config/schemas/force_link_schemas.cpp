// Force-link anchor file for config schemas. Same pattern as
// src/runtime/plugins/force_link_plugins.cpp:
//
//   - Each schema TU declares a ``volatile int kForceLink_<name>`` symbol.
//   - This file references every such symbol via a pointer array so the
//     linker can't discard the schema's TU during static-library link.
//   - schema_registry.cpp calls ``force_link_all_schemas()`` at static init,
//     which touches the array and keeps every schema's static-init alive.
//
// Adding a new schema: add an extern declaration and one entry to the
// array. No other shared file edits. That extra line is the only
// "coupling point" tolerated by the scalability test — the price of the
// static-library link model.

namespace trtf::config::schemas {

extern volatile int kForceLink_triattention;
extern volatile int kForceLink_decode_policy;
extern volatile int kForceLink_text_trace;
extern volatile int kForceLink_runtime;
extern volatile int kForceLink_audio_bark;
extern volatile int kForceLink_audio_magpie;
extern volatile int kForceLink_platform;

// Pointer array referencing every anchor symbol. The array itself has
// external linkage so the linker must keep it; each element inside it
// forces the corresponding schema TU to be pulled in.
volatile int* const kAllSchemaAnchors[] = {
    &kForceLink_triattention,
    &kForceLink_decode_policy,
    &kForceLink_text_trace,
    &kForceLink_runtime,
    &kForceLink_audio_bark,
    &kForceLink_audio_magpie,
    &kForceLink_platform,
};

} // namespace trtf::config::schemas

namespace trtf::config {

// Called from schema_registry.cpp during static init.
volatile int* const* force_link_all_schemas()
{
    return trtf::config::schemas::kAllSchemaAnchors;
}

} // namespace trtf::config

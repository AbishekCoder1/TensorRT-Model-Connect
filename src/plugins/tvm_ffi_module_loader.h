#pragma once

// Load a TVM-FFI module (.so) from disk and register a function globally.
// This enables the C++ runtime to use JIT-compiled FlashInfer kernels
// without Python — pure C++ via the TVM-FFI C ABI.

#if TRTF_HAS_TVM_FFI

#include <string>

namespace trtf {

// Load a TVM-FFI module from a shared library path and register a named
// function from it as a TVM-FFI global function.
//
// Example:
//   load_tvm_ffi_module_func(
//     "/path/to/flashinfer_decode.so",  // JIT-compiled .so
//     "run",                             // function name in the module
//     "flashinfer.decode_f16_d128"       // global name for TVMFFIFunctionGetGlobal
//   );
//
// Returns true on success, false on failure.
bool load_tvm_ffi_module_func(const std::string& so_path, const std::string& func_name,
                              const std::string& global_name);

} // namespace trtf

#endif // TRTF_HAS_TVM_FFI

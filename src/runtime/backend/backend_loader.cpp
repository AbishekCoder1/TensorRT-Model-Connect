#include "runtime/backend/backend_loader.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>

namespace trtf {

namespace {

std::string exe_dir()
{
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "";
    buf[len] = '\0';
    std::string path(buf);
    auto pos = path.rfind('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : "";
}

struct CachedBackend {
    void* dl_handle{nullptr};
    IBackend* backend{nullptr};
};

std::mutex g_mu;
std::unordered_map<std::string, CachedBackend> g_cache;

void cleanup_backends()
{
    for (auto& [name, entry] : g_cache) {
        if (entry.backend) {
            auto destroy = reinterpret_cast<void(*)(IBackend*)>(
                dlsym(entry.dl_handle, "trtf_destroy_backend"));
            if (destroy) destroy(entry.backend);
            entry.backend = nullptr;
        }
        if (entry.dl_handle) {
            dlclose(entry.dl_handle);
            entry.dl_handle = nullptr;
        }
    }
}

} // namespace

IBackend* BackendLoader::load(const std::string& backend_name)
{
    std::lock_guard<std::mutex> lock(g_mu);

    auto it = g_cache.find(backend_name);
    if (it != g_cache.end()) return it->second.backend;

    static bool registered = false;
    if (!registered) {
        std::atexit(cleanup_backends);
        registered = true;
    }

    std::string dso_name = "libtrtf_backend_" + backend_name + ".so";

    void* handle = nullptr;
    std::string tried;

    // 1. Exe directory
    std::string exe_path = exe_dir() + "/" + dso_name;
    handle = dlopen(exe_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) tried += "  " + exe_path + ": " + dlerror() + "\n";

    // 2. TRTF_BACKEND_DIR
    if (!handle) {
        const char* env = std::getenv("TRTF_BACKEND_DIR");
        if (env && env[0] != '\0') {
            std::string env_path = std::string(env) + "/" + dso_name;
            handle = dlopen(env_path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle) tried += "  " + env_path + ": " + dlerror() + "\n";
        }
    }

    // 3. Default search (LD_LIBRARY_PATH, system dirs)
    if (!handle) {
        handle = dlopen(dso_name.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) tried += "  " + dso_name + " (default): " + dlerror() + "\n";
    }

    if (!handle) {
        throw std::runtime_error(
            "Backend \"" + backend_name + "\" not available.\n"
            "Could not load " + dso_name + ":\n" + tried + "\n"
            "To use " + backend_name + " bundles, ensure " + dso_name +
            " is next to the trtf binary,\n"
            "in TRTF_BACKEND_DIR, or in LD_LIBRARY_PATH.");
    }

    auto create_fn = reinterpret_cast<IBackend*(*)()>(
        dlsym(handle, "trtf_create_backend"));
    if (!create_fn) {
        dlclose(handle);
        throw std::runtime_error(
            dso_name + " loaded but missing trtf_create_backend symbol");
    }

    IBackend* backend = create_fn();
    if (!backend) {
        dlclose(handle);
        throw std::runtime_error(
            dso_name + ": trtf_create_backend() returned nullptr");
    }

    g_cache[backend_name] = CachedBackend{handle, backend};
    std::cerr << "[trtf] Backend loaded: " << backend->name()
              << " (" << dso_name << ")" << std::endl;
    return backend;
}

} // namespace trtf

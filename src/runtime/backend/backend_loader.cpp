#include "runtime/backend/backend_loader.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <unistd.h>
#include <vector>

namespace trtf {

namespace {

std::string exe_dir() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0)
        return "";
    buf[len] = '\0';
    std::string path(buf);
    auto pos = path.rfind('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : "";
}

struct CachedBackend {
    void* dl_handle{nullptr};
    IBackend* backend{nullptr};
    BackendLoadMetadata metadata;
};

std::mutex g_mu;
std::unordered_map<std::string, CachedBackend> g_cache;
std::unordered_map<std::string, void*> g_preloaded_dependencies;

void cleanup_backends();

void register_cleanup_once() {
    static bool registered = false;
    if (!registered) {
        std::atexit(cleanup_backends);
        registered = true;
    }
}

void cleanup_backends() {
    for (auto& [name, entry] : g_cache) {
        if (entry.backend) {
            auto destroy = reinterpret_cast<void (*)(IBackend*)>(
                dlsym(entry.dl_handle, "trtf_destroy_backend"));
            if (destroy)
                destroy(entry.backend);
            entry.backend = nullptr;
        }
        if (entry.dl_handle) {
            dlclose(entry.dl_handle);
            entry.dl_handle = nullptr;
        }
    }
    for (auto& [path, handle] : g_preloaded_dependencies) {
        if (handle)
            dlclose(handle);
    }
    g_preloaded_dependencies.clear();
}

void append_load_error(std::string& tried, const std::string& label) {
    const char* error = dlerror();
    tried += "  " + label + ": " + (error ? error : "unknown dlopen error") + "\n";
}

void* try_open_backend_dso(const std::string& path, const std::string& label, std::string& tried) {
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        append_load_error(tried, label);
    }
    return handle;
}

std::string join_path(const std::string& dir, const std::string& dso_name) {
    if (dir.empty()) {
        return dso_name;
    }
    if (dir.back() == '/') {
        return dir + dso_name;
    }
    return dir + "/" + dso_name;
}

void* open_backend_dso(const std::string& dso_name, const std::vector<std::string>& search_dirs,
                       std::string& tried) {
    const std::string exe_path = exe_dir();
    if (!exe_path.empty()) {
        void* handle =
            try_open_backend_dso(exe_path + "/" + dso_name, exe_path + "/" + dso_name, tried);
        if (handle) {
            return handle;
        }
    }

    for (const std::string& dir : search_dirs) {
        if (dir.empty()) {
            continue;
        }
        const std::string path = join_path(dir, dso_name);
        void* handle = try_open_backend_dso(path, path, tried);
        if (handle) {
            return handle;
        }
    }

    return try_open_backend_dso(dso_name, dso_name + " (default)", tried);
}

const char* optional_string_symbol(void* handle, const char* symbol) {
    dlerror();
    auto fn = reinterpret_cast<const char* (*)()>(dlsym(handle, symbol));
    if (!fn) {
        return "";
    }
    const char* value = fn();
    return value ? value : "";
}

CachedBackend create_backend(const std::string& requested_name, const std::string& dso_name,
                             void* handle) {
    auto create_fn = reinterpret_cast<IBackend* (*)()>(dlsym(handle, "trtf_create_backend"));
    if (!create_fn) {
        dlclose(handle);
        throw std::runtime_error(dso_name + " loaded but missing trtf_create_backend symbol");
    }

    IBackend* backend = create_fn();
    if (!backend) {
        dlclose(handle);
        throw std::runtime_error(dso_name + ": trtf_create_backend() returned nullptr");
    }

    BackendLoadMetadata metadata;
    metadata.requested_name = requested_name;
    metadata.dso_name = dso_name;
    metadata.backend_name = backend->name() ? backend->name() : "";
    metadata.trt_abi = optional_string_symbol(handle, "trtf_backend_abi");
    metadata.trt_runtime_version =
        optional_string_symbol(handle, "trtf_backend_runtime_version");

    return CachedBackend{handle, backend, std::move(metadata)};
}

} // namespace

IBackend* BackendLoader::load(const std::string& backend_name) {
    return load(backend_name, {});
}

IBackend* BackendLoader::load(const std::string& backend_name,
                              const std::vector<std::string>& search_dirs) {
    return load_first_available({backend_name}, search_dirs);
}

void BackendLoader::preload_dependency(const std::string& path) {
    if (path.empty())
        return;

    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_preloaded_dependencies.find(path);
    if (it != g_preloaded_dependencies.end())
        return;

    register_cleanup_once();

    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        const char* error = dlerror();
        throw std::runtime_error("Failed to preload dependency " + path + ": " +
                                 (error ? error : "unknown dlopen error"));
    }
    g_preloaded_dependencies[path] = handle;
    std::cerr << "[trtf] Preloaded dependency: " << path << std::endl;
}

IBackend* BackendLoader::load_first_available(const std::vector<std::string>& backend_names,
                                              const std::vector<std::string>& search_dirs,
                                              std::string* loaded_backend_name,
                                              BackendLoadMetadata* metadata) {
    std::lock_guard<std::mutex> lock(g_mu);

    register_cleanup_once();

    std::string all_tried;
    for (const std::string& backend_name : backend_names) {
        auto it = g_cache.find(backend_name);
        if (it != g_cache.end()) {
            if (loaded_backend_name)
                *loaded_backend_name = backend_name;
            if (metadata)
                *metadata = it->second.metadata;
            return it->second.backend;
        }

        std::string dso_name = "libtrtf_backend_" + backend_name + ".so";
        std::string tried;
        void* handle = open_backend_dso(dso_name, search_dirs, tried);
        if (!handle) {
            all_tried += "Candidate \"" + backend_name + "\" (" + dso_name + "):\n" + tried;
            continue;
        }

        CachedBackend entry = create_backend(backend_name, dso_name, handle);
        IBackend* backend = entry.backend;
        g_cache[backend_name] = entry;
        if (loaded_backend_name)
            *loaded_backend_name = backend_name;
        if (metadata)
            *metadata = g_cache[backend_name].metadata;
        std::cerr << "[trtf] Backend loaded: " << backend->name() << " (" << dso_name << ")"
                  << std::endl;
        return backend;
    }

    if (backend_names.size() == 1) {
        const std::string& backend_name = backend_names.front();
        const std::string dso_name = "libtrtf_backend_" + backend_name + ".so";
        throw std::runtime_error("Backend \"" + backend_name +
                                 "\" not available.\n"
                                 "Could not load " +
                                 dso_name + ":\n" + all_tried +
                                 "\n"
                                 "To use " +
                                 backend_name + " bundles, ensure " + dso_name +
                                 " is next to the trtf binary,\n"
                                 "in a LoadOptions::backend_search_paths / --backend-dir "
                                 "directory, or in LD_LIBRARY_PATH.");
    }

    std::string names;
    for (const auto& name : backend_names) {
        if (!names.empty())
            names += ", ";
        names += name;
    }
    throw std::runtime_error("No compatible backend DSO available for candidates: " + names +
                             ".\n" + all_tried +
                             "\nEnsure the matching libtrtf_backend_<backend>.so is next to the "
                             "trtf binary, in a LoadOptions::backend_search_paths / --backend-dir "
                             "directory, or in LD_LIBRARY_PATH.");
}

} // namespace trtf

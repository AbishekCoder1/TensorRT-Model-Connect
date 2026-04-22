#include "runtime/backend/backend_loader.h"

#include <iostream>
#include <stdexcept>
#include <string>

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

int main() {
    // Loading a nonexistent backend should throw
    bool threw = false;
    try {
        trtf::BackendLoader::load("nonexistent_backend_xyz");
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        check(msg.find("nonexistent_backend_xyz") != std::string::npos,
              "error mentions backend name");
        check(msg.find("libtrtf_backend_nonexistent_backend_xyz.so") != std::string::npos,
              "error mentions DSO name");
    }
    check(threw, "missing backend throws runtime_error");

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}

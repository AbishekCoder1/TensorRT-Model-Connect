#pragma once

#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

namespace trtf::runtime::services::common {

class ScopedTempDirOwner {
public:
    ScopedTempDirOwner() = default;

    explicit ScopedTempDirOwner(std::string path)
        : mPath(std::move(path))
    {
    }

    ScopedTempDirOwner(ScopedTempDirOwner&& other) noexcept
        : mPath(std::move(other.mPath))
    {
        other.mPath.clear();
    }

    ScopedTempDirOwner& operator=(ScopedTempDirOwner&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            mPath = std::move(other.mPath);
            other.mPath.clear();
        }
        return *this;
    }

    ScopedTempDirOwner(const ScopedTempDirOwner&) = delete;
    ScopedTempDirOwner& operator=(const ScopedTempDirOwner&) = delete;

    ~ScopedTempDirOwner()
    {
        reset();
    }

    void reset(std::string path = {})
    {
        if (!mPath.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(mPath, ec);
        }
        mPath = std::move(path);
    }

    [[nodiscard]] const std::string& path() const
    {
        return mPath;
    }

private:
    std::string mPath;
};

} // namespace trtf::runtime::services::common

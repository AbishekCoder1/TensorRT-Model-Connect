#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtf {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<int32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int32_t>& ids) const = 0;

    virtual int32_t id_for_token(std::string_view token) const = 0;
    virtual std::string token_for_id(int32_t id) const = 0;
};

std::unique_ptr<ITokenizer> CreateVocabTokenizer(std::vector<std::string> vocab);
std::unique_ptr<ITokenizer> CreateHfPythonTokenizer(std::string model_dir);

} // namespace trtf

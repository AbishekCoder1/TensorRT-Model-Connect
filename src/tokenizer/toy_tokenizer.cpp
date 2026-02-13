#include "trtf/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace trtf {
namespace {

class ToyTokenizer final : public ITokenizer {
public:
    explicit ToyTokenizer(std::vector<std::string> vocab)
    {
        mVocab = std::move(vocab);
        for (std::size_t i = 0; i < mVocab.size(); ++i)
        {
            mTokenToId.emplace(normalize(mVocab[i]), static_cast<int32_t>(i));
        }

        const auto it = mTokenToId.find("<unk>");
        mUnkId = (it == mTokenToId.end()) ? 0 : it->second;
    }

    std::vector<int32_t> encode(const std::string& text) const override
    {
        std::vector<int32_t> ids;
        ids.reserve(text.size() / 3 + 1);

        for (const auto& token : split_tokens(text))
        {
            const auto it = mTokenToId.find(token);
            ids.push_back(it == mTokenToId.end() ? mUnkId : it->second);
        }
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override
    {
        std::ostringstream oss;
        bool first = true;
        for (const int32_t id : ids)
        {
            if (id == id_for_token("<bos>") || id == id_for_token("<eos>") || id == id_for_token("<pad>"))
            {
                continue;
            }

            const auto token = token_for_id(id);
            const bool punctuation = token == "." || token == "," || token == "?" || token == "!";

            if (!first && !punctuation)
            {
                oss << ' ';
            }
            oss << token;
            first = false;
        }
        return oss.str();
    }

    int32_t id_for_token(std::string_view token) const override
    {
        const auto key = normalize(token);
        const auto it = mTokenToId.find(key);
        if (it == mTokenToId.end())
        {
            return mUnkId;
        }
        return it->second;
    }

    std::string token_for_id(int32_t id) const override
    {
        if (id < 0 || static_cast<std::size_t>(id) >= mVocab.size())
        {
            return mVocab[static_cast<std::size_t>(mUnkId)];
        }
        return mVocab[static_cast<std::size_t>(id)];
    }

private:
    static std::string normalize(std::string_view token)
    {
        std::string out(token);
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return out;
    }

    static std::vector<std::string> split_tokens(const std::string& text)
    {
        std::vector<std::string> tokens;
        std::string current;

        auto flush_current = [&]() {
            if (!current.empty())
            {
                tokens.push_back(normalize(current));
                current.clear();
            }
        };

        for (const unsigned char ch : text)
        {
            if (std::isalnum(ch) || ch == '\'')
            {
                current.push_back(static_cast<char>(ch));
                continue;
            }

            flush_current();
            if (ch == '.' || ch == ',' || ch == '?' || ch == '!')
            {
                tokens.emplace_back(1, static_cast<char>(ch));
            }
        }

        flush_current();
        return tokens;
    }

    std::vector<std::string> mVocab;
    std::unordered_map<std::string, int32_t> mTokenToId;
    int32_t mUnkId{0};
};

} // namespace

std::unique_ptr<ITokenizer> CreateToyTokenizer()
{
    return CreateVocabTokenizer({
        "<unk>",
        "<bos>",
        "<eos>",
        "the",
        "secret",
        "to",
        "baking",
        "a",
        "really",
        "good",
        "cake",
        "is",
        "use",
        "fresh",
        "butter",
        "and",
        "measure",
        "carefully",
        "follow",
        "recipe",
        "exactly",
        ".",
        ",",
        "?",
        "!",
        "<pad>",
    });
}

std::unique_ptr<ITokenizer> CreateVocabTokenizer(std::vector<std::string> vocab)
{
    if (vocab.empty())
    {
        throw std::invalid_argument("Vocabulary must not be empty.");
    }
    return std::make_unique<ToyTokenizer>(std::move(vocab));
}

} // namespace trtf

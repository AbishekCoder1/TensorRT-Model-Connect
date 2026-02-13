#include "trtf/backend.h"
#include "trtf/model.h"
#include "trtf/tokenizer.h"

#include <algorithm>
#include <vector>

namespace trtf {
namespace {

class CpuReferenceBackend final : public IGenerationBackend {
public:
    CpuReferenceBackend(const ITokenizer& tokenizer, const DecoderModel& model)
        : mTokenizer(tokenizer)
    {
        mDefaultNextId = mTokenizer.id_for_token(model.default_next_token);
        mBosId = model.architecture.bos_token_id >= 0
            ? model.architecture.bos_token_id
            : mTokenizer.id_for_token("<bos>");
        mEosId = model.architecture.eos_token_id >= 0
            ? model.architecture.eos_token_id
            : mTokenizer.id_for_token("<eos>");

        int32_t max_id = 0;
        for (const auto& token : model.vocab)
        {
            max_id = std::max(max_id, mTokenizer.id_for_token(token));
        }

        mNextTokenById.assign(static_cast<std::size_t>(max_id + 1), mDefaultNextId);
        for (const auto& transition : model.transitions)
        {
            const int32_t from_id = mTokenizer.id_for_token(transition.first);
            const int32_t to_id = mTokenizer.id_for_token(transition.second);
            if (from_id >= 0 && static_cast<std::size_t>(from_id) < mNextTokenById.size())
            {
                mNextTokenById[static_cast<std::size_t>(from_id)] = to_id;
            }
        }
    }

    bool is_available() const override
    {
        return true;
    }

    const char* name() const override
    {
        return "cpu-reference";
    }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const GenerationConfig& config) override
    {
        std::vector<int32_t> output = input_ids;
        int32_t current_token = input_ids.empty() ? mBosId : input_ids.back();
        for (std::size_t step = 0; step < config.max_new_tokens; ++step)
        {
            const int32_t next_token = next_for_token(current_token);
            output.push_back(next_token);
            current_token = next_token;
            if (next_token == mEosId)
            {
                break;
            }
        }
        return output;
    }

private:
    int32_t next_for_token(int32_t token) const
    {
        if (token < 0 || static_cast<std::size_t>(token) >= mNextTokenById.size())
        {
            return mDefaultNextId;
        }
        return mNextTokenById[static_cast<std::size_t>(token)];
    }

    const ITokenizer& mTokenizer;
    std::vector<int32_t> mNextTokenById;
    int32_t mDefaultNextId{0};
    int32_t mBosId{0};
    int32_t mEosId{0};
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateCpuReferenceBackend(const ITokenizer& tokenizer, const DecoderModel& model)
{
    return std::make_unique<CpuReferenceBackend>(tokenizer, model);
}

} // namespace trtf

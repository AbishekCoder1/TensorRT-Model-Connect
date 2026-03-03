// =============================================================================
// ipa_tokenizer.cpp — Native C++ IPA tokenizer for MagpieTTS
// =============================================================================
//
// Reimplements NeMo's IPATokenizer in pure C++, eliminating the Python runtime
// dependency for MagpieTTS text tokenization. The tokenizer is entirely
// dictionary-based: phoneme dict lookup for known words, grapheme fallback
// for OOV, and a heteronym set that forces grapheme mode.
//
// Each pronunciation in the dictionary is a string of IPA characters. Each
// individual character (which may be multi-byte UTF-8) maps to a token ID.
// Graphemes are uppercase letters (NeMo default: no prefix).
//
// Data is loaded from 4 bundle sections baked at build time:
//   - magpie_ipa_phoneme_dict: TSV word→pronunciation string
//   - magpie_ipa_heteronyms: one word per line
//   - magpie_ipa_vocab: one token per line (line index = token ID)
//   - magpie_ipa_config: JSON with grapheme_prefix, eos_id, etc.
// =============================================================================

#include "trtf/tokenizer.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtf {
namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Advance past one UTF-8 character and return it as a string.
// Returns empty string at end of input.
std::string next_utf8_char(const std::string& s, std::size_t& pos)
{
    if (pos >= s.size()) return {};
    const auto ch = static_cast<unsigned char>(s[pos]);
    std::size_t len = 1;
    if (ch >= 0xF0) len = 4;
    else if (ch >= 0xE0) len = 3;
    else if (ch >= 0xC0) len = 2;
    if (pos + len > s.size()) len = s.size() - pos;
    std::string result = s.substr(pos, len);
    pos += len;
    return result;
}

// Split a UTF-8 string into individual characters (each may be multi-byte).
std::vector<std::string> utf8_chars(const std::string& s)
{
    std::vector<std::string> chars;
    std::size_t pos = 0;
    while (pos < s.size())
    {
        chars.push_back(next_utf8_char(s, pos));
    }
    return chars;
}

// ---------------------------------------------------------------------------
// Text preprocessing: curly quote replacement + minimal accent stripping
// ---------------------------------------------------------------------------

std::string preprocess_text(const std::string& text)
{
    std::string out;
    out.reserve(text.size());

    for (std::size_t i = 0; i < text.size(); )
    {
        const auto ch = static_cast<unsigned char>(text[i]);

        if (ch >= 0xC0 && i + 1 < text.size())
        {
            const auto ch2 = static_cast<unsigned char>(text[i + 1]);

            // 2-byte: accent stripping for Latin-1 Supplement
            if (ch >= 0xC0 && ch <= 0xC3)
            {
                const uint32_t cp = (static_cast<uint32_t>(ch & 0x1F) << 6)
                    | static_cast<uint32_t>(ch2 & 0x3F);
                char base = '\0';
                if (cp >= 0xC0 && cp <= 0xC5) base = 'A';
                else if (cp == 0xC7) base = 'C';
                else if (cp >= 0xC8 && cp <= 0xCB) base = 'E';
                else if (cp >= 0xCC && cp <= 0xCF) base = 'I';
                else if (cp == 0xD1) base = 'N';
                else if (cp >= 0xD2 && cp <= 0xD6) base = 'O';
                else if (cp >= 0xD9 && cp <= 0xDC) base = 'U';
                else if (cp == 0xDD) base = 'Y';
                else if (cp >= 0xE0 && cp <= 0xE5) base = 'a';
                else if (cp == 0xE7) base = 'c';
                else if (cp >= 0xE8 && cp <= 0xEB) base = 'e';
                else if (cp >= 0xEC && cp <= 0xEF) base = 'i';
                else if (cp == 0xF1) base = 'n';
                else if (cp >= 0xF2 && cp <= 0xF6) base = 'o';
                else if (cp >= 0xF9 && cp <= 0xFC) base = 'u';
                else if (cp == 0xFD || cp == 0xFF) base = 'y';
                if (base != '\0') { out.push_back(base); i += 2; continue; }
            }

            // 3-byte: curly quotes
            if (ch == 0xE2 && i + 2 < text.size())
            {
                const auto ch3 = static_cast<unsigned char>(text[i + 2]);
                if (ch2 == 0x80 && (ch3 == 0x98 || ch3 == 0x99))
                    { out.push_back('\''); i += 3; continue; }
                if (ch2 == 0x80 && (ch3 == 0x9C || ch3 == 0x9D))
                    { out.push_back('"'); i += 3; continue; }
            }
        }

        out.push_back(static_cast<char>(ch));
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Word tokenization — hand-rolled 3-state machine
// ---------------------------------------------------------------------------

enum class TokenType { WORD, PIPE_DELIMITED, OTHER };

struct TextToken {
    std::string text;
    TokenType type;
};

bool is_alpha(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool is_word_internal(char c)
{
    return is_alpha(c) || c == '-' || c == '\'';
}

std::vector<TextToken> tokenize_text(const std::string& text)
{
    std::vector<TextToken> tokens;
    const std::size_t len = text.size();
    std::size_t i = 0;

    while (i < len)
    {
        if (text[i] == '|')
        {
            std::size_t end = text.find('|', i + 1);
            if (end != std::string::npos)
            {
                tokens.push_back({text.substr(i + 1, end - i - 1), TokenType::PIPE_DELIMITED});
                i = end + 1;
                continue;
            }
            tokens.push_back({std::string(1, text[i]), TokenType::OTHER});
            ++i;
            continue;
        }

        if (is_alpha(text[i]))
        {
            std::size_t start = i;
            ++i;
            while (i < len && is_word_internal(text[i]))
            {
                ++i;
            }
            while (i > start + 1 && !is_alpha(text[i - 1]))
            {
                --i;
            }
            tokens.push_back({text.substr(start, i - start), TokenType::WORD});
            continue;
        }

        tokens.push_back({std::string(1, text[i]), TokenType::OTHER});
        ++i;
    }
    return tokens;
}

// ---------------------------------------------------------------------------
// IPA Tokenizer implementation
// ---------------------------------------------------------------------------

class IpaTokenizer final : public ITokenizer {
public:
    // phoneme_dict: word → list of pronunciation strings (IPA character sequences)
    IpaTokenizer(
        std::unordered_map<std::string, std::vector<std::string>> phoneme_dict,
        std::unordered_set<std::string> heteronyms,
        std::vector<std::string> vocab,
        std::unordered_map<std::string, int32_t> token2id,
        std::unordered_set<std::string> known_tokens,
        std::string grapheme_prefix,
        int32_t eos_id,
        bool ignore_ambiguous_words)
        : mPhonemeDict(std::move(phoneme_dict))
        , mHeteronyms(std::move(heteronyms))
        , mVocab(std::move(vocab))
        , mToken2Id(std::move(token2id))
        , mKnownTokens(std::move(known_tokens))
        , mGraphemePrefix(std::move(grapheme_prefix))
        , mEosId(eos_id)
        , mIgnoreAmbiguous(ignore_ambiguous_words)
    {
    }

    std::vector<int32_t> encode(const std::string& text) const override
    {
        const std::string preprocessed = preprocess_text(text);
        const auto text_tokens = tokenize_text(preprocessed);

        // G2P: convert each text token to IPA character tokens
        std::vector<std::string> ipa_tokens;
        for (const auto& tok : text_tokens)
        {
            if (tok.type == TokenType::WORD)
            {
                g2p_word(tok.text, ipa_tokens);
            }
            else if (tok.type == TokenType::PIPE_DELIMITED)
            {
                // Pipe-delimited: emit each UTF-8 char as a token
                emit_ipa_chars(tok.text, ipa_tokens);
            }
            else
            {
                // OTHER: pass through each character
                for (char c : tok.text)
                {
                    ipa_tokens.emplace_back(1, c);
                }
            }
        }

        // Filter to known tokens
        std::vector<std::string> filtered;
        filtered.reserve(ipa_tokens.size());
        for (const auto& t : ipa_tokens)
        {
            if (mKnownTokens.count(t) != 0)
            {
                filtered.push_back(t);
            }
        }

        // Deduplicate consecutive spaces
        std::vector<std::string> deduped;
        deduped.reserve(filtered.size());
        for (const auto& t : filtered)
        {
            if (t == " " && !deduped.empty() && deduped.back() == " ")
            {
                continue;
            }
            deduped.push_back(t);
        }
        if (!deduped.empty() && deduped.back() == " ")
        {
            deduped.pop_back();
        }

        // Map to IDs + append EOS
        std::vector<int32_t> ids;
        ids.reserve(deduped.size() + 1);
        for (const auto& t : deduped)
        {
            auto it = mToken2Id.find(t);
            if (it != mToken2Id.end())
            {
                ids.push_back(it->second);
            }
        }
        ids.push_back(mEosId);
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override
    {
        std::string out;
        for (const int32_t id : ids)
        {
            if (id == mEosId) break;
            if (id >= 0 && static_cast<std::size_t>(id) < mVocab.size())
            {
                out += mVocab[static_cast<std::size_t>(id)];
            }
        }
        return out;
    }

    int32_t id_for_token(std::string_view token) const override
    {
        auto it = mToken2Id.find(std::string(token));
        return (it != mToken2Id.end()) ? it->second : -1;
    }

    std::string token_for_id(int32_t id) const override
    {
        if (id >= 0 && static_cast<std::size_t>(id) < mVocab.size())
        {
            return mVocab[static_cast<std::size_t>(id)];
        }
        return "";
    }

private:
    // Emit individual UTF-8 characters from an IPA pronunciation string.
    // Each character becomes a separate token (may be multi-byte).
    void emit_ipa_chars(const std::string& pronunciation, std::vector<std::string>& out) const
    {
        auto chars = utf8_chars(pronunciation);
        for (const auto& ch : chars)
        {
            out.push_back(ch);
        }
    }

    // G2P for a word token
    void g2p_word(const std::string& word, std::vector<std::string>& out) const
    {
        std::string lower;
        lower.reserve(word.size());
        for (char c : word)
        {
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }

        // Heteronym → emit as graphemes (uppercase chars)
        if (mHeteronyms.count(lower) != 0)
        {
            emit_graphemes(word, out);
            return;
        }

        // Dictionary lookup
        auto it = mPhonemeDict.find(lower);
        if (it != mPhonemeDict.end())
        {
            const auto& pronunciations = it->second;
            if (pronunciations.size() == 1)
            {
                // Unique pronunciation → emit IPA characters
                emit_ipa_chars(pronunciations[0], out);
                return;
            }

            // Multiple pronunciations (ambiguous)
            if (mIgnoreAmbiguous)
            {
                emit_graphemes(word, out);
                return;
            }

            // Use first pronunciation
            emit_ipa_chars(pronunciations[0], out);
            return;
        }

        // Try possessive stripping: word ending in 's or s
        if (lower.size() > 2 && lower.back() == 's')
        {
            std::string base;
            if (lower.size() > 2 && lower[lower.size() - 2] == '\'')
            {
                base = lower.substr(0, lower.size() - 2);
            }
            else
            {
                base = lower.substr(0, lower.size() - 1);
            }

            auto base_it = mPhonemeDict.find(base);
            if (base_it != mPhonemeDict.end() && !base_it->second.empty())
            {
                emit_ipa_chars(base_it->second[0], out);
                out.emplace_back("z");
                return;
            }
        }

        // OOV → emit as graphemes
        emit_graphemes(word, out);
    }

    // Emit each alpha character of a word as a grapheme token.
    // NeMo uses uppercase letters as grapheme tokens (no prefix by default).
    // If grapheme_prefix is set (e.g. "#"), emits prefix+lowercase.
    void emit_graphemes(const std::string& word, std::vector<std::string>& out) const
    {
        for (char c : word)
        {
            if (!is_alpha(c)) continue;
            if (mGraphemePrefix.empty())
            {
                // NeMo default: uppercase letter as token
                out.emplace_back(1, static_cast<char>(
                    std::toupper(static_cast<unsigned char>(c))));
            }
            else
            {
                out.push_back(mGraphemePrefix + std::string(1,
                    static_cast<char>(std::tolower(static_cast<unsigned char>(c)))));
            }
        }
    }

    // word → list of pronunciation strings (each is an IPA character sequence)
    std::unordered_map<std::string, std::vector<std::string>> mPhonemeDict;
    std::unordered_set<std::string> mHeteronyms;
    std::vector<std::string> mVocab;
    std::unordered_map<std::string, int32_t> mToken2Id;
    std::unordered_set<std::string> mKnownTokens;
    std::string mGraphemePrefix;
    int32_t mEosId;
    bool mIgnoreAmbiguous;
};

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

// Parse phoneme dictionary from TSV text:
// word<TAB>pronunciation_string\n
// Each pronunciation is a single IPA string (characters are individual tokens).
// Multiple lines for same word = multiple pronunciations.
std::unordered_map<std::string, std::vector<std::string>>
parse_phoneme_dict(const char* data, std::size_t size)
{
    std::unordered_map<std::string, std::vector<std::string>> dict;
    const std::string text(data, size);
    std::istringstream iss(text);
    std::string line;

    while (std::getline(iss, line))
    {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        const auto tab_pos = line.find('\t');
        if (tab_pos == std::string::npos) continue;

        std::string word = line.substr(0, tab_pos);
        std::string pronunciation = line.substr(tab_pos + 1);

        if (!pronunciation.empty())
        {
            dict[word].push_back(std::move(pronunciation));
        }
    }
    return dict;
}

// Parse heteronyms: one word per line
std::unordered_set<std::string> parse_heteronyms(const char* data, std::size_t size)
{
    std::unordered_set<std::string> set;
    const std::string text(data, size);
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) set.insert(line);
    }
    return set;
}

// Parse vocabulary: one token per line, line index = token ID
std::vector<std::string> parse_vocab(const char* data, std::size_t size)
{
    std::vector<std::string> vocab;
    const std::string text(data, size);
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        vocab.push_back(line);
    }
    return vocab;
}

} // namespace

std::unique_ptr<ITokenizer> CreateIpaTokenizer(
    const char* phoneme_dict_data, std::size_t phoneme_dict_size,
    const char* heteronyms_data, std::size_t heteronyms_size,
    const char* vocab_data, std::size_t vocab_size,
    const char* config_data, std::size_t config_size)
{
    if (phoneme_dict_data == nullptr || phoneme_dict_size == 0)
    {
        throw std::invalid_argument("IPA phoneme dictionary data must not be empty");
    }
    if (vocab_data == nullptr || vocab_size == 0)
    {
        throw std::invalid_argument("IPA vocabulary data must not be empty");
    }

    auto phoneme_dict = parse_phoneme_dict(phoneme_dict_data, phoneme_dict_size);
    auto heteronyms = (heteronyms_data != nullptr && heteronyms_size > 0)
        ? parse_heteronyms(heteronyms_data, heteronyms_size)
        : std::unordered_set<std::string>{};
    auto vocab = parse_vocab(vocab_data, vocab_size);

    // Parse config JSON
    std::string config_text;
    if (config_data != nullptr && config_size > 0)
    {
        config_text.assign(config_data, config_size);
    }
    std::string grapheme_prefix = extract_json_string(config_text, "grapheme_prefix", "");
    int32_t eos_id = extract_json_int(config_text, "eos_id", -1);
    int32_t ignore_ambiguous = extract_json_int(config_text, "ignore_ambiguous_words", 0);

    if (eos_id < 0)
    {
        eos_id = static_cast<int32_t>(vocab.size()) - 1;
    }

    // Build token2id and known_tokens from vocab
    std::unordered_map<std::string, int32_t> token2id;
    std::unordered_set<std::string> known_tokens;
    for (std::size_t i = 0; i < vocab.size(); ++i)
    {
        token2id.emplace(vocab[i], static_cast<int32_t>(i));
        known_tokens.insert(vocab[i]);
    }

    return std::make_unique<IpaTokenizer>(
        std::move(phoneme_dict),
        std::move(heteronyms),
        std::move(vocab),
        std::move(token2id),
        std::move(known_tokens),
        std::move(grapheme_prefix),
        eos_id,
        ignore_ambiguous != 0);
}

} // namespace trtf

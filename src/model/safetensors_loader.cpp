#include "safetensors_loader.h"
#include "utils/text_parsers.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtf {
namespace {

float bits_to_float(uint32_t bits)
{
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float bfloat16_to_float(uint16_t v)
{
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    return bits_to_float(bits);
}

float float16_to_float(uint16_t v)
{
    const uint32_t sign = static_cast<uint32_t>(v & 0x8000U) << 16;
    const uint32_t exp = (v >> 10U) & 0x1FU;
    const uint32_t frac = static_cast<uint32_t>(v & 0x03FFU);

    if (exp == 0U)
    {
        if (frac == 0U)
        {
            return bits_to_float(sign);
        }

        uint32_t mant = frac;
        uint32_t e = 0U;
        while ((mant & 0x0400U) == 0U)
        {
            mant <<= 1U;
            ++e;
        }
        mant &= 0x03FFU;
        const uint32_t out_exp = 127U - 15U - e;
        const uint32_t out_bits = sign | (out_exp << 23U) | (mant << 13U);
        return bits_to_float(out_bits);
    }

    if (exp == 0x1FU)
    {
        const uint32_t out_bits = sign | 0x7F800000U | (frac << 13U);
        return bits_to_float(out_bits);
    }

    const uint32_t out_exp = exp + (127U - 15U);
    const uint32_t out_bits = sign | (out_exp << 23U) | (frac << 13U);
    return bits_to_float(out_bits);
}

uint64_t read_u64_le(std::istream& in)
{
    std::array<unsigned char, 8> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in)
    {
        throw std::runtime_error("Failed reading little-endian u64.");
    }

    uint64_t value = 0;
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        value |= static_cast<uint64_t>(bytes[i]) << (8U * i);
    }
    return value;
}

class JsonCursor {
public:
    explicit JsonCursor(std::string text)
        : mText(std::move(text))
    {
    }

    bool eof() const
    {
        return mPos >= mText.size();
    }

    void skip_ws()
    {
        while (!eof() && std::isspace(static_cast<unsigned char>(mText[mPos])) != 0)
        {
            ++mPos;
        }
    }

    bool consume(char c)
    {
        skip_ws();
        if (!eof() && mText[mPos] == c)
        {
            ++mPos;
            return true;
        }
        return false;
    }

    void expect(char c)
    {
        if (!consume(c))
        {
            throw std::runtime_error(std::string("Invalid safetensors JSON: expected '") + c + "'.");
        }
    }

    char peek()
    {
        skip_ws();
        if (eof())
        {
            throw std::runtime_error("Invalid safetensors JSON: unexpected EOF.");
        }
        return mText[mPos];
    }

    std::string parse_string()
    {
        expect('"');
        std::string out;
        while (!eof())
        {
            char c = mText[mPos++];
            if (c == '"')
            {
                return out;
            }
            if (c == '\\')
            {
                if (eof())
                {
                    throw std::runtime_error("Invalid safetensors JSON string escape.");
                }
                const char esc = mText[mPos++];
                switch (esc)
                {
                case '"':
                case '\\':
                case '/':
                    out.push_back(esc);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u':
                    if (mPos + 4 > mText.size())
                    {
                        throw std::runtime_error("Invalid safetensors JSON unicode escape.");
                    }
                    mPos += 4;
                    out.push_back('?');
                    break;
                default:
                    throw std::runtime_error("Invalid safetensors JSON escape.");
                }
                continue;
            }
            out.push_back(c);
        }
        throw std::runtime_error("Invalid safetensors JSON string (unterminated).");
    }

    int64_t parse_int64()
    {
        skip_ws();
        std::size_t begin = mPos;
        if (!eof() && (mText[mPos] == '-' || mText[mPos] == '+'))
        {
            ++mPos;
        }
        while (!eof() && std::isdigit(static_cast<unsigned char>(mText[mPos])) != 0)
        {
            ++mPos;
        }
        if (mPos == begin)
        {
            throw std::runtime_error("Invalid safetensors JSON integer.");
        }
        return std::stoll(mText.substr(begin, mPos - begin));
    }

    std::vector<int64_t> parse_int64_array()
    {
        expect('[');
        std::vector<int64_t> out;
        if (consume(']'))
        {
            return out;
        }
        while (true)
        {
            out.push_back(parse_int64());
            if (consume(']'))
            {
                break;
            }
            expect(',');
        }
        return out;
    }

    void skip_value()
    {
        const char c = peek();
        if (c == '"')
        {
            (void) parse_string();
            return;
        }
        if (c == '{')
        {
            expect('{');
            if (consume('}'))
            {
                return;
            }
            while (true)
            {
                (void) parse_string();
                expect(':');
                skip_value();
                if (consume('}'))
                {
                    break;
                }
                expect(',');
            }
            return;
        }
        if (c == '[')
        {
            expect('[');
            if (consume(']'))
            {
                return;
            }
            while (true)
            {
                skip_value();
                if (consume(']'))
                {
                    break;
                }
                expect(',');
            }
            return;
        }

        // Number or literal.
        while (!eof())
        {
            const char v = mText[mPos];
            if (v == ',' || v == '}' || v == ']' || std::isspace(static_cast<unsigned char>(v)) != 0)
            {
                break;
            }
            ++mPos;
        }
    }

private:
    std::string mText;
    std::size_t mPos{0};
};

std::unordered_map<std::string, SafetensorEntry> parse_safetensors_header(std::string header)
{
    JsonCursor cursor(std::move(header));
    std::unordered_map<std::string, SafetensorEntry> entries;

    cursor.expect('{');
    if (cursor.consume('}'))
    {
        return entries;
    }

    while (true)
    {
        const std::string key = cursor.parse_string();
        cursor.expect(':');

        if (key == "__metadata__")
        {
            cursor.skip_value();
        }
        else
        {
            SafetensorEntry entry;
            cursor.expect('{');
            if (!cursor.consume('}'))
            {
                while (true)
                {
                    const std::string field = cursor.parse_string();
                    cursor.expect(':');
                    if (field == "dtype")
                    {
                        entry.dtype = cursor.parse_string();
                    }
                    else if (field == "shape")
                    {
                        entry.shape = cursor.parse_int64_array();
                    }
                    else if (field == "data_offsets")
                    {
                        const std::vector<int64_t> offsets = cursor.parse_int64_array();
                        if (offsets.size() != 2 || offsets[0] < 0 || offsets[1] < offsets[0])
                        {
                            throw std::runtime_error("Invalid safetensors data_offsets.");
                        }
                        entry.data_begin = static_cast<uint64_t>(offsets[0]);
                        entry.data_end = static_cast<uint64_t>(offsets[1]);
                    }
                    else
                    {
                        cursor.skip_value();
                    }

                    if (cursor.consume('}'))
                    {
                        break;
                    }
                    cursor.expect(',');
                }
            }

            if (entry.dtype.empty() || entry.shape.empty())
            {
                throw std::runtime_error("Invalid safetensors tensor entry for key: " + key);
            }
            entries.emplace(key, std::move(entry));
        }

        if (cursor.consume('}'))
        {
            break;
        }
        cursor.expect(',');
    }

    return entries;
}

std::size_t checked_element_count(const std::vector<int64_t>& shape, const std::string& tensor_name)
{
    std::size_t count = 1;
    for (const int64_t dim : shape)
    {
        if (dim <= 0)
        {
            throw std::runtime_error("Invalid safetensors shape for " + tensor_name);
        }
        const std::size_t sdim = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / sdim)
        {
            throw std::runtime_error("Safetensors element count overflow for " + tensor_name);
        }
        count *= sdim;
    }
    return count;
}

} // namespace

SafetensorReader::SafetensorReader(std::filesystem::path path)
    : mPath(std::move(path))
{
    std::ifstream in(mPath, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open safetensors file: " + mPath.string());
    }

    const uint64_t header_size = read_u64_le(in);
    const uint64_t file_size = static_cast<uint64_t>(std::filesystem::file_size(mPath));
    if (header_size > file_size || 8ULL + header_size > file_size)
    {
        throw std::runtime_error("Invalid safetensors header size for file: " + mPath.string());
    }

    std::string header(static_cast<std::size_t>(header_size), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!in)
    {
        throw std::runtime_error("Failed reading safetensors header: " + mPath.string());
    }

    mEntries = parse_safetensors_header(std::move(header));
    mDataBaseOffset = 8ULL + header_size;

    for (const auto& kv : mEntries)
    {
        const SafetensorEntry& entry = kv.second;
        if (entry.data_begin > entry.data_end)
        {
            throw std::runtime_error("Invalid safetensors tensor offsets for: " + kv.first);
        }
        if (entry.data_end > file_size - mDataBaseOffset)
        {
            throw std::runtime_error("Safetensors tensor offset out of range for: " + kv.first);
        }
    }
}

bool SafetensorReader::has(const std::string& name) const
{
    return mEntries.find(name) != mEntries.end();
}

const SafetensorEntry& SafetensorReader::entry(const std::string& name) const
{
    const auto it = mEntries.find(name);
    if (it == mEntries.end())
    {
        throw std::runtime_error("Safetensors key not found: " + name);
    }
    return it->second;
}

std::vector<float> SafetensorReader::load_f32(const std::string& name) const
{
    const SafetensorEntry& meta = entry(name);
    const std::size_t elements = checked_element_count(meta.shape, name);
    const uint64_t bytes64 = meta.data_end - meta.data_begin;
    if (bytes64 > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::runtime_error("Safetensors tensor is too large: " + name);
    }
    const std::size_t bytes = static_cast<std::size_t>(bytes64);

    std::ifstream in(mPath, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open safetensors file for tensor read: " + mPath.string());
    }
    const uint64_t absolute_offset = mDataBaseOffset + meta.data_begin;
    if (absolute_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
    {
        throw std::runtime_error("Safetensors tensor offset is out of streamoff range.");
    }
    in.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    if (!in)
    {
        throw std::runtime_error("Failed seeking safetensors tensor: " + name);
    }

    std::vector<unsigned char> raw(bytes, 0);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (!in)
    {
        throw std::runtime_error("Failed reading safetensors tensor: " + name);
    }

    std::vector<float> out(elements, 0.0F);
    if (meta.dtype == "F32")
    {
        if (bytes != elements * sizeof(float))
        {
            throw std::runtime_error("Invalid F32 byte size for safetensors tensor: " + name);
        }
        std::memcpy(out.data(), raw.data(), bytes);
        return out;
    }
    if (meta.dtype == "F16")
    {
        if (bytes != elements * sizeof(uint16_t))
        {
            throw std::runtime_error("Invalid F16 byte size for safetensors tensor: " + name);
        }
        for (std::size_t i = 0; i < elements; ++i)
        {
            uint16_t v = 0;
            std::memcpy(&v, raw.data() + (i * sizeof(uint16_t)), sizeof(uint16_t));
            out[i] = float16_to_float(v);
        }
        return out;
    }
    if (meta.dtype == "BF16")
    {
        if (bytes != elements * sizeof(uint16_t))
        {
            throw std::runtime_error("Invalid BF16 byte size for safetensors tensor: " + name);
        }
        for (std::size_t i = 0; i < elements; ++i)
        {
            uint16_t v = 0;
            std::memcpy(&v, raw.data() + (i * sizeof(uint16_t)), sizeof(uint16_t));
            out[i] = bfloat16_to_float(v);
        }
        return out;
    }

    throw std::runtime_error("Unsupported safetensors dtype for tensor " + name + ": " + meta.dtype);
}

namespace {

void skip_json_ws(const std::string& text, std::size_t& pos)
{
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0)
    {
        ++pos;
    }
}

std::string parse_json_string_raw(const std::string& text, std::size_t& pos)
{
    if (pos >= text.size() || text[pos] != '"')
    {
        throw std::runtime_error("Invalid JSON string while parsing safetensors index.");
    }

    ++pos;
    std::string out;
    while (pos < text.size())
    {
        const char c = text[pos++];
        if (c == '"')
        {
            return out;
        }
        if (c == '\\')
        {
            if (pos >= text.size())
            {
                throw std::runtime_error("Invalid JSON escape while parsing safetensors index.");
            }
            const char esc = text[pos++];
            switch (esc)
            {
            case '"':
            case '\\':
            case '/':
                out.push_back(esc);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u':
                if (pos + 4 > text.size())
                {
                    throw std::runtime_error("Invalid JSON unicode escape while parsing safetensors index.");
                }
                pos += 4;
                out.push_back('?');
                break;
            default:
                throw std::runtime_error("Invalid JSON escape while parsing safetensors index.");
            }
            continue;
        }
        out.push_back(c);
    }

    throw std::runtime_error("Unterminated JSON string while parsing safetensors index.");
}

std::unordered_map<std::string, std::string> parse_safetensors_index_weight_map(const std::filesystem::path& path)
{
    const std::string text = trtf::read_file(path);
    const std::string needle = "\"weight_map\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        throw std::runtime_error("Missing weight_map in safetensors index: " + path.string());
    }

    std::size_t pos = text.find(':', key_pos);
    if (pos == std::string::npos)
    {
        throw std::runtime_error("Invalid weight_map in safetensors index: " + path.string());
    }
    ++pos;
    skip_json_ws(text, pos);
    if (pos >= text.size() || text[pos] != '{')
    {
        throw std::runtime_error("Invalid weight_map object in safetensors index: " + path.string());
    }
    ++pos;

    std::unordered_map<std::string, std::string> out;
    while (true)
    {
        skip_json_ws(text, pos);
        if (pos >= text.size())
        {
            throw std::runtime_error("Unexpected EOF in safetensors index weight_map: " + path.string());
        }
        if (text[pos] == '}')
        {
            ++pos;
            break;
        }

        const std::string tensor_name = parse_json_string_raw(text, pos);
        skip_json_ws(text, pos);
        if (pos >= text.size() || text[pos] != ':')
        {
            throw std::runtime_error("Expected ':' in safetensors index weight_map: " + path.string());
        }
        ++pos;
        skip_json_ws(text, pos);
        const std::string shard_name = parse_json_string_raw(text, pos);

        out.emplace(tensor_name, shard_name);

        skip_json_ws(text, pos);
        if (pos >= text.size())
        {
            throw std::runtime_error("Unexpected EOF in safetensors index weight_map: " + path.string());
        }
        if (text[pos] == ',')
        {
            ++pos;
            continue;
        }
        if (text[pos] == '}')
        {
            ++pos;
            break;
        }
        throw std::runtime_error("Invalid separator in safetensors index weight_map: " + path.string());
    }

    return out;
}

} // namespace

bool is_safetensors_index_file(const std::filesystem::path& path)
{
    return ends_with(path.filename().string(), ".safetensors.index.json");
}

TensorSource::TensorSource(const std::filesystem::path& path)
{
    if (is_safetensors_index_file(path))
    {
        init_from_index(path);
        return;
    }

    mSinglePath = path;
    mSingleReader = std::make_unique<SafetensorReader>(path);
}

bool TensorSource::has(const std::string& name) const
{
    if (mSingleReader)
    {
        return mSingleReader->has(name);
    }
    return mTensorToShard.find(name) != mTensorToShard.end();
}

const SafetensorEntry& TensorSource::entry(const std::string& name) const
{
    if (mSingleReader)
    {
        return mSingleReader->entry(name);
    }
    return shard_reader_for(name).entry(name);
}

std::vector<float> TensorSource::load_f32(const std::string& name) const
{
    if (mSingleReader)
    {
        return mSingleReader->load_f32(name);
    }
    return shard_reader_for(name).load_f32(name);
}

const SafetensorReader& TensorSource::shard_reader_for(const std::string& name) const
{
    const auto it = mTensorToShard.find(name);
    if (it == mTensorToShard.end())
    {
        throw std::runtime_error("Safetensors key not found in sharded index: " + name);
    }
    return *mShardReaders[it->second];
}

void TensorSource::init_from_index(const std::filesystem::path& index_path)
{
    const auto weight_map = parse_safetensors_index_weight_map(index_path);
    if (weight_map.empty())
    {
        throw std::runtime_error("Safetensors index has empty weight_map: " + index_path.string());
    }

    std::unordered_map<std::string, std::size_t> shard_idx_by_path;
    const std::filesystem::path base_dir = index_path.parent_path();

    for (const auto& kv : weight_map)
    {
        const std::filesystem::path shard_path = (base_dir / kv.second).lexically_normal();
        const std::string shard_key = shard_path.string();

        std::size_t shard_idx = 0;
        const auto shard_it = shard_idx_by_path.find(shard_key);
        if (shard_it == shard_idx_by_path.end())
        {
            shard_idx = mShardReaders.size();
            mShardReaders.push_back(std::make_unique<SafetensorReader>(shard_path));
            mShardPaths.push_back(shard_path);
            shard_idx_by_path.emplace(shard_key, shard_idx);
        }
        else
        {
            shard_idx = shard_it->second;
        }

        if (!mShardReaders[shard_idx]->has(kv.first))
        {
            throw std::runtime_error("Tensor " + kv.first + " is mapped to shard " + shard_path.string()
                + " but was not found in that file.");
        }

        mTensorToShard.emplace(kv.first, shard_idx);
    }
}

} // namespace trtf

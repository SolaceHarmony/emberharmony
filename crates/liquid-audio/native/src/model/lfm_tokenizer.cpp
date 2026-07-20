#include "lfm_tokenizer.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

using Json = nlohmann::ordered_json;

namespace {

class TokenizerError final : public std::runtime_error {
  public:
    TokenizerError(int status, std::string message)
        : std::runtime_error(std::move(message)), status_(status) {}
    int status() const { return status_; }

  private:
    int status_;
};

[[noreturn]] void fail(int status, const std::string &message) {
    throw TokenizerError(status, message);
}

void set_error(char *error, size_t length, const char *message) {
    if (!error || length == 0) return;
    std::snprintf(error, length, "%s", message ? message : "tokenizer error");
}

Json read_json(const char *path) {
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(
        std::fopen(path, "rb"), &std::fclose);
    if (!file) fail(-ENOENT, std::string("cannot open tokenizer '") + path + "'");
    if (std::fseek(file.get(), 0, SEEK_END) != 0) fail(-EIO, "cannot size tokenizer");
    const long end = std::ftell(file.get());
    if (end <= 0 || end > 128 * 1024 * 1024L) fail(-EFBIG, "invalid tokenizer size");
    std::rewind(file.get());
    std::vector<char> bytes((size_t)end);
    if (std::fread(bytes.data(), 1, bytes.size(), file.get()) != bytes.size()) {
        fail(-EIO, "cannot read tokenizer");
    }
    try {
        return Json::parse(bytes.begin(), bytes.end());
    } catch (const std::exception &exception) {
        fail(-EINVAL, std::string("invalid tokenizer JSON: ") + exception.what());
    }
}

void append_utf8(uint32_t cp, std::string *out) {
    if (cp <= 0x7f) {
        out->push_back((char)cp);
    } else if (cp <= 0x7ff) {
        out->push_back((char)(0xc0 | (cp >> 6)));
        out->push_back((char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out->push_back((char)(0xe0 | (cp >> 12)));
        out->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
        out->push_back((char)(0x80 | (cp & 0x3f)));
    } else {
        out->push_back((char)(0xf0 | (cp >> 18)));
        out->push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
        out->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
        out->push_back((char)(0x80 | (cp & 0x3f)));
    }
}

struct Rune {
    uint32_t cp;
    size_t start;
    size_t bytes;
};

bool decode_rune(std::string_view text, size_t offset, Rune *out) {
    if (offset >= text.size()) return false;
    const uint8_t a = (uint8_t)text[offset];
    uint32_t cp = 0;
    size_t n = 0;
    if (a < 0x80) {
        cp = a;
        n = 1;
    } else if ((a & 0xe0) == 0xc0) {
        cp = a & 0x1f;
        n = 2;
    } else if ((a & 0xf0) == 0xe0) {
        cp = a & 0x0f;
        n = 3;
    } else if ((a & 0xf8) == 0xf0) {
        cp = a & 0x07;
        n = 4;
    } else {
        return false;
    }
    if (offset + n > text.size()) return false;
    for (size_t index = 1; index < n; ++index) {
        const uint8_t byte = (uint8_t)text[offset + index];
        if ((byte & 0xc0) != 0x80) return false;
        cp = (cp << 6) | (byte & 0x3f);
    }
    const uint32_t minimum = n == 1 ? 0 : n == 2 ? 0x80 : n == 3 ? 0x800 : 0x10000;
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
    *out = {cp, offset, n};
    return true;
}

bool newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

bool whitespace(uint32_t cp) {
    return cp == 0x09 || cp == 0x0a || cp == 0x0b || cp == 0x0c || cp == 0x0d ||
           cp == 0x20 || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

bool number(uint32_t cp) {
    return (cp >= '0' && cp <= '9') || (cp >= 0x660 && cp <= 0x669) ||
           (cp >= 0x6f0 && cp <= 0x6f9) || (cp >= 0xff10 && cp <= 0xff19);
}

bool ascii_letter(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

char ascii_lower(uint32_t cp) {
    return (char)(cp >= 'A' && cp <= 'Z' ? cp + ('a' - 'A') : cp);
}

bool punctuation(uint32_t cp) {
    if (cp < 0x80) {
        return (cp >= 0x21 && cp <= 0x2f) ||
               (cp >= 0x3a && cp <= 0x40) ||
               (cp >= 0x5b && cp <= 0x60) ||
               (cp >= 0x7b && cp <= 0x7e);
    }
    return (cp >= 0x2000 && cp <= 0x206f) || (cp >= 0x2e00 && cp <= 0x2e7f) ||
           (cp >= 0x3000 && cp <= 0x303f) || (cp >= 0xff00 && cp <= 0xff65);
}

bool letter(uint32_t cp) {
    if (cp < 0x80) return ascii_letter(cp);
    return !whitespace(cp) && !number(cp) && !punctuation(cp) && !newline(cp);
}

std::vector<Rune> runes(std::string_view text) {
    std::vector<Rune> values;
    for (size_t offset = 0; offset < text.size();) {
        Rune rune{};
        if (!decode_rune(text, offset, &rune)) fail(-EINVAL, "tokenizer input is not UTF-8");
        values.push_back(rune);
        offset += rune.bytes;
    }
    return values;
}

std::vector<std::string_view> split_pretokens(std::string_view text) {
    const std::vector<Rune> chars = runes(text);
    std::vector<std::string_view> pieces;
    auto emit = [&](size_t first, size_t last) {
        const size_t begin = chars[first].start;
        const size_t end = last == chars.size() ? text.size() : chars[last].start;
        pieces.push_back(text.substr(begin, end - begin));
    };
    size_t at = 0;
    while (at < chars.size()) {
        const size_t start = at;
        if (chars[at].cp == '\'' && at + 1 < chars.size()) {
            size_t end = at + 1;
            while (end < chars.size() && end - at <= 3 && chars[end].cp < 0x80 &&
                   ascii_letter(chars[end].cp)) {
                ++end;
            }
            std::string suffix;
            for (size_t i = at + 1; i < end; ++i) {
                suffix.push_back(ascii_lower(chars[i].cp));
            }
            if (suffix == "s" || suffix == "t" || suffix == "re" || suffix == "ve" ||
                suffix == "m" || suffix == "ll" || suffix == "d") {
                emit(start, end);
                at = end;
                continue;
            }
        }
        size_t cursor = at;
        if (!newline(chars[cursor].cp) && !letter(chars[cursor].cp) &&
            !number(chars[cursor].cp) && cursor + 1 < chars.size() &&
            letter(chars[cursor + 1].cp)) {
            ++cursor;
        }
        if (letter(chars[cursor].cp)) {
            do {
                ++cursor;
            } while (cursor < chars.size() && letter(chars[cursor].cp));
            emit(start, cursor);
            at = cursor;
            continue;
        }
        if (number(chars[at].cp)) {
            cursor = at;
            do {
                ++cursor;
            } while (cursor < chars.size() && cursor - at < 3 && number(chars[cursor].cp));
            emit(start, cursor);
            at = cursor;
            continue;
        }
        cursor = at;
        if (chars[cursor].cp == 0x20 && cursor + 1 < chars.size() &&
            !whitespace(chars[cursor + 1].cp) && !letter(chars[cursor + 1].cp) &&
            !number(chars[cursor + 1].cp)) {
            ++cursor;
        }
        const size_t symbols = cursor;
        while (cursor < chars.size() && !whitespace(chars[cursor].cp) &&
               !letter(chars[cursor].cp) && !number(chars[cursor].cp)) {
            ++cursor;
        }
        if (cursor > symbols) {
            while (cursor < chars.size() && newline(chars[cursor].cp)) ++cursor;
            emit(start, cursor);
            at = cursor;
            continue;
        }
        cursor = at;
        while (cursor < chars.size() && whitespace(chars[cursor].cp)) ++cursor;
        if (cursor == chars.size() || cursor > at) {
            emit(start, cursor);
            at = cursor;
            continue;
        }
        fail(-EINVAL, "tokenizer pre-tokenizer made no progress");
    }
    return pieces;
}

uint64_t pair_key(uint32_t left, uint32_t right) {
    return ((uint64_t)left << 32) | right;
}

} // namespace

struct ResolvedMerge {
    uint32_t rank;
    uint32_t merged;
};

struct LfmTokenizer {
    std::array<std::string, 256> bytes;
    std::array<uint32_t, 256> byte_ids{};
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte;
    std::unordered_map<std::string, uint32_t> vocab;
    std::unordered_map<uint64_t, ResolvedMerge> merges;
    std::vector<std::string> inverse;
    std::vector<std::string> added;
    std::vector<uint8_t> special;
    std::vector<std::pair<std::string, uint32_t>> special_text;
    LfmTokenizerSpecialV1 control{};
};

/* One allocation owns this header and both trailing planes. The encode path
 * only indexes these fixed spans, so capacity cannot silently grow. */
struct LfmTokenizerWorkspace {
    size_t max_input_bytes = 0;
    size_t storage_bytes = 0;
    uint64_t encode_calls = 0;
    Rune *runes = nullptr;
    uint32_t *symbols = nullptr;
};

namespace {

void build_byte_codec(LfmTokenizer *tokenizer) {
    std::vector<uint32_t> values;
    for (uint32_t value = 33; value <= 126; ++value) values.push_back(value);
    for (uint32_t value = 161; value <= 172; ++value) values.push_back(value);
    for (uint32_t value = 174; value <= 255; ++value) values.push_back(value);
    std::array<uint8_t, 256> present{};
    for (uint32_t value : values) present[value] = 1;
    uint32_t extra = 0;
    for (uint32_t byte = 0; byte < 256; ++byte) {
        const uint32_t cp = present[byte] ? byte : 256 + extra++;
        append_utf8(cp, &tokenizer->bytes[byte]);
        tokenizer->unicode_to_byte.emplace(cp, (uint8_t)byte);
    }
}

uint32_t required_special(const LfmTokenizer &tokenizer, const char *name) {
    const auto found = tokenizer.vocab.find(name);
    if (found == tokenizer.vocab.end() || found->second >= tokenizer.special.size() ||
        !tokenizer.special[found->second]) {
        fail(-EINVAL, std::string("tokenizer is missing special token '") + name + "'");
    }
    return found->second;
}

void encode_bpe(const LfmTokenizer &tokenizer, std::string_view piece,
                std::vector<uint32_t> *output) {
    std::vector<uint32_t> symbols;
    symbols.reserve(piece.size());
    for (uint8_t byte : piece) symbols.push_back(tokenizer.byte_ids[byte]);
    while (symbols.size() > 1) {
        uint32_t best = UINT32_MAX;
        uint32_t best_left = 0;
        uint32_t best_right = 0;
        uint32_t best_merged = 0;
        for (size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto found = tokenizer.merges.find(pair_key(symbols[index], symbols[index + 1]));
            if (found != tokenizer.merges.end() && found->second.rank < best) {
                best = found->second.rank;
                best_left = symbols[index];
                best_right = symbols[index + 1];
                best_merged = found->second.merged;
            }
        }
        if (best == UINT32_MAX) break;
        std::vector<uint32_t> merged;
        merged.reserve(symbols.size());
        for (size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size() && symbols[index] == best_left &&
                symbols[index + 1] == best_right) {
                merged.push_back(best_merged);
                index += 2;
            } else {
                merged.push_back(symbols[index++]);
            }
        }
        symbols = std::move(merged);
    }
    output->insert(output->end(), symbols.begin(), symbols.end());
}

void encode_ordinary(const LfmTokenizer &tokenizer, std::string_view text,
                     std::vector<uint32_t> *output) {
    for (std::string_view piece : split_pretokens(text)) {
        encode_bpe(tokenizer, piece, output);
    }
}

void encode_all(const LfmTokenizer &tokenizer, std::string_view text,
                std::vector<uint32_t> *output) {
    size_t at = 0;
    while (at < text.size()) {
        size_t next = text.size();
        const std::pair<std::string, uint32_t> *matched = nullptr;
        for (const auto &entry : tokenizer.special_text) {
            const size_t found = text.find(entry.first, at);
            if (found < next || (found == next && matched &&
                                 entry.first.size() > matched->first.size())) {
                next = found;
                matched = &entry;
            }
        }
        if (!matched) {
            encode_ordinary(tokenizer, text.substr(at), output);
            return;
        }
        if (next > at) encode_ordinary(tokenizer, text.substr(at, next - at), output);
        output->push_back(matched->second);
        at = next + matched->first.size();
    }
}

struct TokenSink {
    uint32_t *out;
    size_t capacity;
    size_t count;
    bool write;
};

int emit_token(TokenSink *sink, uint32_t token) {
    if (sink->write) {
        if (sink->count >= sink->capacity) return -ENOBUFS;
        sink->out[sink->count] = token;
    }
    ++sink->count;
    return 0;
}

bool contraction(const Rune *chars, size_t at, size_t end) {
    const size_t count = end - at - 1;
    const auto lower = [&](size_t index) {
        return ascii_lower(chars[at + 1 + index].cp);
    };
    if (count == 1) {
        const char value = lower(0);
        return value == 's' || value == 't' || value == 'm' || value == 'd';
    }
    if (count != 2) return false;
    const char first = lower(0);
    const char second = lower(1);
    return (first == 'r' && second == 'e') ||
           (first == 'v' && second == 'e') ||
           (first == 'l' && second == 'l');
}

int encode_bpe_bounded(const LfmTokenizer &tokenizer,
                       LfmTokenizerWorkspace *workspace,
                       std::string_view piece, TokenSink *sink) {
    if (piece.size() > workspace->max_input_bytes) return -ENOBUFS;
    size_t count = piece.size();
    for (size_t index = 0; index < count; ++index) {
        workspace->symbols[index] = tokenizer.byte_ids[(uint8_t)piece[index]];
    }
    while (count > 1) {
        uint32_t best = UINT32_MAX;
        uint32_t best_left = 0;
        uint32_t best_right = 0;
        uint32_t best_merged = 0;
        for (size_t index = 0; index + 1 < count; ++index) {
            const auto found = tokenizer.merges.find(
                pair_key(workspace->symbols[index], workspace->symbols[index + 1]));
            if (found != tokenizer.merges.end() && found->second.rank < best) {
                best = found->second.rank;
                best_left = workspace->symbols[index];
                best_right = workspace->symbols[index + 1];
                best_merged = found->second.merged;
            }
        }
        if (best == UINT32_MAX) break;
        size_t read = 0;
        size_t write = 0;
        while (read < count) {
            if (read + 1 < count && workspace->symbols[read] == best_left &&
                workspace->symbols[read + 1] == best_right) {
                workspace->symbols[write++] = best_merged;
                read += 2;
                continue;
            }
            workspace->symbols[write++] = workspace->symbols[read++];
        }
        count = write;
    }
    for (size_t index = 0; index < count; ++index) {
        const int status = emit_token(sink, workspace->symbols[index]);
        if (status != 0) return status;
    }
    return 0;
}

int encode_ordinary_bounded(const LfmTokenizer &tokenizer,
                            LfmTokenizerWorkspace *workspace,
                            std::string_view text, TokenSink *sink) {
    size_t char_count = 0;
    for (size_t offset = 0; offset < text.size();) {
        Rune rune{};
        if (!decode_rune(text, offset, &rune)) return -EINVAL;
        if (char_count >= workspace->max_input_bytes) return -ENOBUFS;
        workspace->runes[char_count++] = rune;
        offset += rune.bytes;
    }
    const auto emit = [&](size_t first, size_t last) {
        const size_t begin = workspace->runes[first].start;
        const size_t end = last == char_count ? text.size()
                                               : workspace->runes[last].start;
        return encode_bpe_bounded(tokenizer, workspace,
                                  text.substr(begin, end - begin), sink);
    };
    size_t at = 0;
    while (at < char_count) {
        const size_t start = at;
        if (workspace->runes[at].cp == '\'' && at + 1 < char_count) {
            size_t end = at + 1;
            while (end < char_count && end - at <= 3 &&
                   workspace->runes[end].cp < 0x80 &&
                   ascii_letter(workspace->runes[end].cp)) {
                ++end;
            }
            if (contraction(workspace->runes, at, end)) {
                const int status = emit(start, end);
                if (status != 0) return status;
                at = end;
                continue;
            }
        }
        size_t cursor = at;
        if (!newline(workspace->runes[cursor].cp) &&
            !letter(workspace->runes[cursor].cp) &&
            !number(workspace->runes[cursor].cp) && cursor + 1 < char_count &&
            letter(workspace->runes[cursor + 1].cp)) {
            ++cursor;
        }
        if (letter(workspace->runes[cursor].cp)) {
            do {
                ++cursor;
            } while (cursor < char_count && letter(workspace->runes[cursor].cp));
            const int status = emit(start, cursor);
            if (status != 0) return status;
            at = cursor;
            continue;
        }
        if (number(workspace->runes[at].cp)) {
            cursor = at;
            do {
                ++cursor;
            } while (cursor < char_count && cursor - at < 3 &&
                     number(workspace->runes[cursor].cp));
            const int status = emit(start, cursor);
            if (status != 0) return status;
            at = cursor;
            continue;
        }
        cursor = at;
        if (workspace->runes[cursor].cp == 0x20 && cursor + 1 < char_count &&
            !whitespace(workspace->runes[cursor + 1].cp) &&
            !letter(workspace->runes[cursor + 1].cp) &&
            !number(workspace->runes[cursor + 1].cp)) {
            ++cursor;
        }
        const size_t symbols = cursor;
        while (cursor < char_count && !whitespace(workspace->runes[cursor].cp) &&
               !letter(workspace->runes[cursor].cp) &&
               !number(workspace->runes[cursor].cp)) {
            ++cursor;
        }
        if (cursor > symbols) {
            while (cursor < char_count && newline(workspace->runes[cursor].cp)) {
                ++cursor;
            }
            const int status = emit(start, cursor);
            if (status != 0) return status;
            at = cursor;
            continue;
        }
        cursor = at;
        while (cursor < char_count && whitespace(workspace->runes[cursor].cp)) {
            ++cursor;
        }
        if (cursor == char_count || cursor > at) {
            const int status = emit(start, cursor);
            if (status != 0) return status;
            at = cursor;
            continue;
        }
        return -EINVAL;
    }
    return 0;
}

int encode_all_bounded(const LfmTokenizer &tokenizer,
                       LfmTokenizerWorkspace *workspace,
                       std::string_view text, TokenSink *sink) {
    size_t at = 0;
    while (at < text.size()) {
        size_t next = text.size();
        const std::pair<std::string, uint32_t> *matched = nullptr;
        for (const auto &entry : tokenizer.special_text) {
            const size_t found = text.find(entry.first, at);
            if (found < next || (found == next && matched &&
                                 entry.first.size() > matched->first.size())) {
                next = found;
                matched = &entry;
            }
        }
        if (!matched) {
            return encode_ordinary_bounded(tokenizer, workspace,
                                           text.substr(at), sink);
        }
        if (next > at) {
            const int status = encode_ordinary_bounded(
                tokenizer, workspace, text.substr(at, next - at), sink);
            if (status != 0) return status;
        }
        const int status = emit_token(sink, matched->second);
        if (status != 0) return status;
        at = next + matched->first.size();
    }
    return 0;
}

bool align_size(size_t value, size_t alignment, size_t *out) {
    if (value > SIZE_MAX - (alignment - 1)) return false;
    *out = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

int decode_token(const LfmTokenizer &tokenizer, uint32_t token,
                 bool skip_special, uint8_t *out, size_t out_capacity,
                 size_t *out_bytes) {
    if (token >= tokenizer.inverse.size() || tokenizer.inverse[token].empty()) {
        fail(-EINVAL, "token ID is outside the vocabulary");
    }
    if (token < tokenizer.added.size() && !tokenizer.added[token].empty()) {
        if (skip_special && tokenizer.special[token]) {
            *out_bytes = 0;
            return 0;
        }
        const std::string &value = tokenizer.added[token];
        *out_bytes = value.size();
        if (value.size() > out_capacity) return -ENOSPC;
        if (!value.empty() && !out) return -EINVAL;
        std::memcpy(out, value.data(), value.size());
        return 0;
    }
    const std::string &encoded = tokenizer.inverse[token];
    size_t count = 0;
    for (size_t offset = 0; offset < encoded.size();) {
        Rune rune{};
        if (!decode_rune(encoded, offset, &rune)) fail(-EINVAL, "vocabulary token is not UTF-8");
        const auto found = tokenizer.unicode_to_byte.find(rune.cp);
        if (found == tokenizer.unicode_to_byte.end()) {
            fail(-EINVAL, "vocabulary token is not ByteLevel encoded");
        }
        ++count;
        offset += rune.bytes;
    }
    *out_bytes = count;
    if (count > out_capacity) return -ENOSPC;
    if (count != 0 && !out) return -EINVAL;
    size_t index = 0;
    for (size_t offset = 0; offset < encoded.size();) {
        Rune rune{};
        (void)decode_rune(encoded, offset, &rune);
        out[index++] = tokenizer.unicode_to_byte.find(rune.cp)->second;
        offset += rune.bytes;
    }
    return 0;
}

} // namespace

extern "C" int lfm_tokenizer_open(const char *path, LfmTokenizer **out,
                                   char *error, size_t error_length) {
    if (!path || !out) return -EINVAL;
    *out = nullptr;
    try {
        const Json document = read_json(path);
        /* This tokenizer builds BPE merges only; any other model type would be
         * decoded as if it were BPE, so it is refused rather than mis-decoded. */
        if (!document.is_object() || !document.contains("model") ||
            !document.at("model").is_object() ||
            document.at("model").value("type", "") != "BPE") {
            fail(-EOPNOTSUPP, "native tokenizer implements BPE models only");
        }
        auto tokenizer = std::make_unique<LfmTokenizer>();
        build_byte_codec(tokenizer.get());
        const Json &model = document.at("model");
        const Json &vocab = model.at("vocab");
        if (!vocab.is_object() || vocab.empty()) fail(-EINVAL, "tokenizer vocabulary is empty");
        uint64_t maximum = 0;
        for (const auto &entry : vocab.items()) {
            if (!entry.value().is_number_unsigned() && !entry.value().is_number_integer()) {
                fail(-EINVAL, "tokenizer vocabulary ID is not an integer");
            }
            const int64_t id = entry.value().get<int64_t>();
            if (id < 0 || (uint64_t)id > UINT32_MAX) fail(-EOVERFLOW, "tokenizer ID overflow");
            if (!tokenizer->vocab.emplace(entry.key(), (uint32_t)id).second) {
                fail(-EINVAL, "duplicate tokenizer vocabulary entry");
            }
            maximum = std::max(maximum, (uint64_t)id);
        }
        if (maximum >= std::numeric_limits<size_t>::max()) fail(-EOVERFLOW, "vocabulary too large");
        tokenizer->inverse.resize((size_t)maximum + 1);
        tokenizer->added.resize((size_t)maximum + 1);
        tokenizer->special.resize((size_t)maximum + 1);
        for (const auto &entry : tokenizer->vocab) {
            std::string &slot = tokenizer->inverse[entry.second];
            if (!slot.empty()) fail(-EINVAL, "duplicate tokenizer ID");
            slot = entry.first;
        }
        for (size_t byte = 0; byte < tokenizer->bytes.size(); ++byte) {
            const auto found = tokenizer->vocab.find(tokenizer->bytes[byte]);
            if (found == tokenizer->vocab.end()) {
                fail(-EINVAL, "tokenizer vocabulary is missing a ByteLevel symbol");
            }
            tokenizer->byte_ids[byte] = found->second;
        }
        const auto added = document.find("added_tokens");
        if (added != document.end()) {
            if (!added->is_array()) fail(-EINVAL, "added_tokens is not an array");
            for (const Json &entry : *added) {
                if (!entry.is_object() || !entry.contains("id") ||
                    !entry.contains("content") || !entry.at("content").is_string()) {
                    fail(-EINVAL, "malformed added token");
                }
                const uint64_t id = entry.at("id").get<uint64_t>();
                if (id >= tokenizer->inverse.size()) fail(-EINVAL, "added token ID out of range");
                const std::string content = entry.at("content").get<std::string>();
                if (tokenizer->inverse[id] != content) fail(-EINVAL, "added token/vocab mismatch");
                tokenizer->added[id] = content;
                tokenizer->special[id] = entry.value("special", false) ? 1 : 0;
                if (tokenizer->special[id]) tokenizer->special_text.emplace_back(content, (uint32_t)id);
            }
        }
        const Json &merges = model.at("merges");
        if (!merges.is_array()) fail(-EINVAL, "tokenizer merges is not an array");
        if (merges.size() > UINT32_MAX) fail(-EOVERFLOW, "too many BPE merges");
        uint32_t rank = 0;
        for (const Json &merge : merges) {
            if (!merge.is_array() || merge.size() != 2 || !merge[0].is_string() ||
                !merge[1].is_string()) {
                fail(-EINVAL, "malformed BPE merge");
            }
            const std::string &left = merge[0].get_ref<const std::string &>();
            const std::string &right = merge[1].get_ref<const std::string &>();
            const auto left_id = tokenizer->vocab.find(left);
            const auto right_id = tokenizer->vocab.find(right);
            const auto merged_id = tokenizer->vocab.find(left + right);
            if (left_id == tokenizer->vocab.end() ||
                right_id == tokenizer->vocab.end() ||
                merged_id == tokenizer->vocab.end()) {
                fail(-EINVAL, "BPE merge symbol is absent from vocabulary");
            }
            if (!tokenizer->merges
                     .emplace(pair_key(left_id->second, right_id->second),
                              ResolvedMerge{rank++, merged_id->second})
                     .second) {
                fail(-EINVAL, "duplicate BPE merge pair");
            }
        }
        std::sort(tokenizer->special_text.begin(), tokenizer->special_text.end(),
                  [](const auto &left, const auto &right) {
                      return left.first.size() > right.first.size();
                  });
        tokenizer->control = {
            .size = sizeof(LfmTokenizerSpecialV1),
            .abi_version = LFM_TOKENIZER_ABI_VERSION,
            .im_start = required_special(*tokenizer, "<|im_start|>"),
            .im_end = required_special(*tokenizer, "<|im_end|>"),
            .text_end = required_special(*tokenizer, "<|text_end|>"),
            .audio_start = required_special(*tokenizer, "<|audio_start|>"),
            .reserved = {},
        };
        *out = tokenizer.release();
        return 0;
    } catch (const TokenizerError &exception) {
        set_error(error, error_length, exception.what());
        return exception.status();
    } catch (const std::bad_alloc &) {
        set_error(error, error_length, "native tokenizer allocation failed");
        return -ENOMEM;
    } catch (const std::exception &exception) {
        set_error(error, error_length, exception.what());
        return -EINVAL;
    }
}

extern "C" void lfm_tokenizer_close(LfmTokenizer *tokenizer) { delete tokenizer; }

extern "C" int lfm_tokenizer_special(const LfmTokenizer *tokenizer,
                                      LfmTokenizerSpecialV1 *out) {
    if (!tokenizer || !out || out->size < sizeof(*out) ||
        out->abi_version != LFM_TOKENIZER_ABI_VERSION) {
        return -EINVAL;
    }
    *out = tokenizer->control;
    return 0;
}

extern "C" int lfm_tokenizer_workspace_create(
    size_t max_input_bytes, LfmTokenizerWorkspace **out) {
    if (!out || max_input_bytes == 0) return -EINVAL;
    *out = nullptr;
    size_t rune_offset = 0;
    if (!align_size(sizeof(LfmTokenizerWorkspace), alignof(Rune), &rune_offset) ||
        max_input_bytes > SIZE_MAX / sizeof(Rune)) {
        return -EOVERFLOW;
    }
    const size_t rune_bytes = max_input_bytes * sizeof(Rune);
    if (rune_offset > SIZE_MAX - rune_bytes) return -EOVERFLOW;
    size_t symbol_offset = 0;
    if (!align_size(rune_offset + rune_bytes, alignof(uint32_t),
                    &symbol_offset) ||
        max_input_bytes > SIZE_MAX / sizeof(uint32_t)) {
        return -EOVERFLOW;
    }
    const size_t symbol_bytes = max_input_bytes * sizeof(uint32_t);
    if (symbol_offset > SIZE_MAX - symbol_bytes) return -EOVERFLOW;
    const size_t total = symbol_offset + symbol_bytes;
    void *storage = ::operator new(total, std::nothrow);
    if (!storage) return -ENOMEM;
    auto *workspace = new (storage) LfmTokenizerWorkspace();
    auto *bytes = static_cast<uint8_t *>(storage);
    workspace->max_input_bytes = max_input_bytes;
    workspace->storage_bytes = total;
    workspace->runes = reinterpret_cast<Rune *>(bytes + rune_offset);
    workspace->symbols = reinterpret_cast<uint32_t *>(bytes + symbol_offset);
    *out = workspace;
    return 0;
}

extern "C" void lfm_tokenizer_workspace_destroy(
    LfmTokenizerWorkspace *workspace) {
    if (!workspace) return;
    workspace->~LfmTokenizerWorkspace();
    ::operator delete(workspace);
}

extern "C" int lfm_tokenizer_workspace_info(
    const LfmTokenizerWorkspace *workspace,
    LfmTokenizerWorkspaceInfoV1 *out) {
    if (!workspace || !out || out->size < sizeof(*out) ||
        out->abi_version != LFM_TOKENIZER_ABI_VERSION) {
        return -EINVAL;
    }
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_TOKENIZER_ABI_VERSION,
        .max_input_bytes = workspace->max_input_bytes,
        .storage_bytes = workspace->storage_bytes,
        .encode_calls = workspace->encode_calls,
        .reserved = {},
    };
    return 0;
}

extern "C" int lfm_tokenizer_encode_bounded(
    const LfmTokenizer *tokenizer, LfmTokenizerWorkspace *workspace,
    const char *text, size_t text_bytes, uint32_t *out,
    size_t out_capacity, size_t *out_count) {
    if (!tokenizer || !workspace || (!text && text_bytes != 0) || !out_count) {
        return -EINVAL;
    }
    *out_count = 0;
    if (text_bytes > workspace->max_input_bytes) return -ENOBUFS;
    if (workspace->encode_calls != UINT64_MAX) ++workspace->encode_calls;
    const std::string_view input(text ? text : "", text_bytes);
    const bool direct = out && out_capacity >= text_bytes;
    TokenSink first{
        .out = direct ? out : nullptr,
        .capacity = direct ? out_capacity : 0,
        .count = 0,
        .write = direct,
    };
    int status = encode_all_bounded(*tokenizer, workspace, input, &first);
    *out_count = first.count;
    if (status != 0 || direct) return status;
    if (first.count > out_capacity) return -ENOBUFS;
    if (first.count != 0 && !out) return -EINVAL;
    TokenSink second{
        .out = out,
        .capacity = out_capacity,
        .count = 0,
        .write = true,
    };
    status = encode_all_bounded(*tokenizer, workspace, input, &second);
    *out_count = second.count;
    return status;
}

extern "C" int lfm_tokenizer_encode(const LfmTokenizer *tokenizer, const char *text,
                                     size_t text_bytes, uint32_t *out,
                                     size_t out_capacity, size_t *out_count) {
    if (!tokenizer || (!text && text_bytes != 0) || !out_count) return -EINVAL;
    try {
        std::vector<uint32_t> encoded;
        encode_all(*tokenizer, std::string_view(text ? text : "", text_bytes), &encoded);
        *out_count = encoded.size();
        if (encoded.size() > out_capacity) return -ENOSPC;
        if (!encoded.empty() && !out) return -EINVAL;
        std::copy(encoded.begin(), encoded.end(), out);
        return 0;
    } catch (const TokenizerError &exception) {
        return exception.status();
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    } catch (...) {
        return -EINVAL;
    }
}

extern "C" int lfm_tokenizer_decode_piece(const LfmTokenizer *tokenizer,
                                           uint32_t token, uint32_t skip_special,
                                           uint8_t *out, size_t out_capacity,
                                           size_t *out_bytes) {
    if (!tokenizer || !out_bytes) return -EINVAL;
    try {
        return decode_token(*tokenizer, token, skip_special != 0, out,
                            out_capacity, out_bytes);
    } catch (const TokenizerError &exception) {
        return exception.status();
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    } catch (...) {
        return -EINVAL;
    }
}

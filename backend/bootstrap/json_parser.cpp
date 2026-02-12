#include "json_parser.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace brain19 {

static constexpr size_t MAX_DEPTH = 100;

void JsonParser::skip_ws(State& s) {
    while (s.pos < s.input.size()) {
        char c = s.input[s.pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            ++s.pos;
        else
            break;
    }
}

bool JsonParser::peek(State& s, char c) {
    skip_ws(s);
    return s.pos < s.input.size() && s.input[s.pos] == c;
}

bool JsonParser::consume(State& s, char c) {
    skip_ws(s);
    if (s.pos < s.input.size() && s.input[s.pos] == c) {
        ++s.pos;
        return true;
    }
    return false;
}

std::optional<std::string> JsonParser::parse_string(State& s) {
    skip_ws(s);
    if (s.pos >= s.input.size() || s.input[s.pos] != '"') return std::nullopt;
    ++s.pos;

    std::string result;
    while (s.pos < s.input.size()) {
        char c = s.input[s.pos++];
        if (c == '"') return result;
        if (c == '\\') {
            if (s.pos >= s.input.size()) return std::nullopt;
            char esc = s.input[s.pos++];
            switch (esc) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                case 'u':
                    // Skip 4 hex digits (basic handling)
                    if (s.pos + 4 > s.input.size()) return std::nullopt;
                    result += '?';
                    s.pos += 4;
                    break;
                default: return std::nullopt;
            }
        } else {
            result += c;
        }
    }
    return std::nullopt;  // Unterminated string
}

std::optional<JsonValue> JsonParser::parse_number(State& s) {
    skip_ws(s);
    size_t start = s.pos;

    if (s.pos < s.input.size() && s.input[s.pos] == '-') ++s.pos;

    bool has_digits = false;
    while (s.pos < s.input.size() && s.input[s.pos] >= '0' && s.input[s.pos] <= '9') {
        ++s.pos;
        has_digits = true;
    }
    if (!has_digits) { s.pos = start; return std::nullopt; }

    if (s.pos < s.input.size() && s.input[s.pos] == '.') {
        ++s.pos;
        while (s.pos < s.input.size() && s.input[s.pos] >= '0' && s.input[s.pos] <= '9')
            ++s.pos;
    }

    if (s.pos < s.input.size() && (s.input[s.pos] == 'e' || s.input[s.pos] == 'E')) {
        ++s.pos;
        if (s.pos < s.input.size() && (s.input[s.pos] == '+' || s.input[s.pos] == '-'))
            ++s.pos;
        while (s.pos < s.input.size() && s.input[s.pos] >= '0' && s.input[s.pos] <= '9')
            ++s.pos;
    }

    std::string num_str = s.input.substr(start, s.pos - start);
    char* end = nullptr;
    double val = std::strtod(num_str.c_str(), &end);
    if (end == num_str.c_str()) { s.pos = start; return std::nullopt; }

    JsonValue v;
    v.data = val;
    return v;
}

bool JsonParser::parse_literal(State& s, const char* lit, [[maybe_unused]] JsonValue& out) {
    skip_ws(s);
    size_t len = 0;
    while (lit[len]) ++len;
    if (s.pos + len > s.input.size()) return false;
    if (s.input.compare(s.pos, len, lit) != 0) return false;
    s.pos += len;
    return true;
}

std::optional<JsonValue> JsonParser::parse_object(State& s) {
    if (!consume(s, '{')) return std::nullopt;
    if (++s.depth > MAX_DEPTH) return std::nullopt;

    JsonObject obj;

    if (peek(s, '}')) { consume(s, '}'); --s.depth; JsonValue v; v.data = std::move(obj); return v; }

    while (true) {
        auto key = parse_string(s);
        if (!key) return std::nullopt;
        if (!consume(s, ':')) return std::nullopt;
        auto val = parse_value(s);
        if (!val) return std::nullopt;
        obj[*key] = std::move(*val);

        if (consume(s, ',')) continue;
        if (consume(s, '}')) break;
        return std::nullopt;
    }

    --s.depth;
    JsonValue v;
    v.data = std::move(obj);
    return v;
}

std::optional<JsonValue> JsonParser::parse_array(State& s) {
    if (!consume(s, '[')) return std::nullopt;
    if (++s.depth > MAX_DEPTH) return std::nullopt;

    JsonArray arr;

    if (peek(s, ']')) { consume(s, ']'); --s.depth; JsonValue v; v.data = std::move(arr); return v; }

    while (true) {
        auto val = parse_value(s);
        if (!val) return std::nullopt;
        arr.push_back(std::move(*val));

        if (consume(s, ',')) continue;
        if (consume(s, ']')) break;
        return std::nullopt;
    }

    --s.depth;
    JsonValue v;
    v.data = std::move(arr);
    return v;
}

std::optional<JsonValue> JsonParser::parse_value(State& s) {
    skip_ws(s);
    if (s.pos >= s.input.size()) return std::nullopt;

    char c = s.input[s.pos];

    if (c == '"') {
        auto str = parse_string(s);
        if (!str) return std::nullopt;
        JsonValue v; v.data = std::move(*str); return v;
    }
    if (c == '{') return parse_object(s);
    if (c == '[') return parse_array(s);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(s);

    JsonValue v;
    if (parse_literal(s, "true", v))  { v.data = true; return v; }
    if (parse_literal(s, "false", v)) { v.data = false; return v; }
    if (parse_literal(s, "null", v))  { v.data = nullptr; return v; }

    return std::nullopt;
}

std::optional<JsonValue> JsonParser::parse(const std::string& json) {
    State s{json, 0, 0};
    auto result = parse_value(s);
    if (!result) return std::nullopt;
    skip_ws(s);
    if (s.pos != json.size()) return std::nullopt;  // Trailing garbage
    return result;
}

std::optional<JsonValue> JsonParser::parse_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return std::nullopt;
    std::ostringstream oss;
    oss << in.rdbuf();
    return parse(oss.str());
}

} // namespace brain19

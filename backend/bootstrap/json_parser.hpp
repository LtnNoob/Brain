#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <optional>

namespace brain19 {

struct JsonValue;
using JsonObject = std::unordered_map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
    std::variant<
        std::nullptr_t,
        bool,
        double,
        std::string,
        JsonArray,
        JsonObject
    > data;

    JsonValue() : data(nullptr) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_number() const { return std::holds_alternative<double>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<JsonArray>(data); }
    bool is_object() const { return std::holds_alternative<JsonObject>(data); }

    double as_number() const { return std::get<double>(data); }
    bool as_bool() const { return std::get<bool>(data); }
    const std::string& as_string() const { return std::get<std::string>(data); }
    const JsonArray& as_array() const { return std::get<JsonArray>(data); }
    const JsonObject& as_object() const { return std::get<JsonObject>(data); }

    double number_or(double def) const { return is_number() ? as_number() : def; }
    std::string string_or(const std::string& def) const { return is_string() ? as_string() : def; }

    // Convenience: access object field
    const JsonValue* get(const std::string& key) const {
        if (!is_object()) return nullptr;
        auto& obj = as_object();
        auto it = obj.find(key);
        return (it != obj.end()) ? &it->second : nullptr;
    }
};

class JsonParser {
public:
    static std::optional<JsonValue> parse(const std::string& json);
    static std::optional<JsonValue> parse_file(const std::string& path);

private:
    struct State {
        const std::string& input;
        size_t pos = 0;
        size_t depth = 0;
    };

    static std::optional<JsonValue> parse_value(State& s);
    static std::optional<JsonValue> parse_object(State& s);
    static std::optional<JsonValue> parse_array(State& s);
    static std::optional<std::string> parse_string(State& s);
    static std::optional<JsonValue> parse_number(State& s);
    static bool parse_literal(State& s, const char* lit, JsonValue& out);
    static void skip_ws(State& s);
    static bool peek(State& s, char c);
    static bool consume(State& s, char c);
};

} // namespace brain19

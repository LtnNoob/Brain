#include "knowledge_ingestor.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace brain19 {

KnowledgeIngestor::KnowledgeIngestor() {
}

// =============================================================================
// JSON PARSING (minimal, no external library)
// =============================================================================

ParseResult KnowledgeIngestor::parse_json(const std::string& json_str) const {
    if (json_str.empty()) {
        return ParseResult::error("Empty JSON input");
    }

    // Find the outermost object braces
    size_t start = json_str.find('{');
    size_t end = json_str.rfind('}');
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        return ParseResult::error("Invalid JSON: no root object found");
    }

    StructuredInput input;

    // Extract source
    input.source_reference = json_extract_string(json_str, "source");

    // Extract concepts array
    std::string concepts_arr = json_extract_array(json_str, "concepts");
    if (!concepts_arr.empty()) {
        auto concept_objects = json_split_array(concepts_arr);
        for (const auto& obj : concept_objects) {
            StructuredConcept cpt;
            cpt.label = json_extract_string(obj, "label");
            cpt.definition = json_extract_string(obj, "definition");
            cpt.trust_category = json_extract_string(obj, "trust");
            cpt.trust_value = json_extract_number(obj, "trust_value", 0.0);

            if (cpt.label.empty()) continue; // Skip empty labels

            input.concepts.push_back(cpt);
        }
    }

    // Extract relations array
    std::string relations_arr = json_extract_array(json_str, "relations");
    if (!relations_arr.empty()) {
        auto relation_objects = json_split_array(relations_arr);
        for (const auto& obj : relation_objects) {
            StructuredRelation rel;
            rel.source_label = json_extract_string(obj, "source");
            rel.target_label = json_extract_string(obj, "target");
            rel.relation_type = json_extract_string(obj, "type");
            rel.weight = json_extract_number(obj, "weight", 1.0);

            if (rel.source_label.empty() || rel.target_label.empty()) continue;

            input.relations.push_back(rel);
        }
    }

    if (input.concepts.empty() && input.relations.empty()) {
        return ParseResult::error("No concepts or relations found in JSON");
    }

    return ParseResult::ok(input);
}

std::string KnowledgeIngestor::json_extract_string(
    const std::string& json, const std::string& key) const
{
    // Scope-aware: find key only at the outermost nesting level of the input
    std::string search = "\"" + key + "\"";
    size_t key_pos = std::string::npos;
    {
        int depth = 0;
        bool in_str = false;
        for (size_t i = 0; i < json.size(); ++i) {
            char c = json[i];
            if (c == '"' && (i == 0 || json[i - 1] != '\\')) {
                in_str = !in_str;
            }
            if (in_str) continue;
            if (c == '{') ++depth;
            else if (c == '}') --depth;
            // Match key only at depth == 1 (top-level object)
            if (depth == 1 && !in_str && i + search.size() <= json.size() && json.compare(i, search.size(), search) == 0) {
                key_pos = i;
                break;
            }
        }
    }
    if (key_pos == std::string::npos) return "";

    // Find colon after key
    size_t colon = json.find(':', key_pos + search.size());
    if (colon == std::string::npos) return "";

    // Find opening quote of value
    size_t val_start = json.find('"', colon + 1);
    if (val_start == std::string::npos) return "";

    // Find closing quote (handle escaped quotes)
    size_t val_end = val_start + 1;
    while (val_end < json.size()) {
        if (json[val_end] == '"' && json[val_end - 1] != '\\') {
            break;
        }
        ++val_end;
    }

    if (val_end >= json.size()) return "";

    std::string value = json.substr(val_start + 1, val_end - val_start - 1);

    // Unescape basic sequences
    std::string result;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 1 < value.size()) {
            switch (value[i + 1]) {
                case '"': result += '"'; ++i; break;
                case '\\': result += '\\'; ++i; break;
                case 'n': result += '\n'; ++i; break;
                case 't': result += '\t'; ++i; break;
                default: result += value[i]; break;
            }
        } else {
            result += value[i];
        }
    }

    return result;
}

std::string KnowledgeIngestor::json_extract_array(
    const std::string& json, const std::string& key) const
{
    // Scope-aware: find key only at the outermost nesting level of the input
    std::string search = "\"" + key + "\"";
    size_t key_pos = std::string::npos;
    {
        int depth = 0;
        bool in_str = false;
        for (size_t i = 0; i < json.size(); ++i) {
            char c = json[i];
            if (c == '"' && (i == 0 || json[i - 1] != '\\')) {
                in_str = !in_str;
            }
            if (in_str) continue;
            if (c == '{') ++depth;
            else if (c == '}') --depth;
            if (depth == 1 && !in_str && i + search.size() <= json.size() && json.compare(i, search.size(), search) == 0) {
                key_pos = i;
                break;
            }
        }
    }
    if (key_pos == std::string::npos) return "";

    size_t colon = json.find(':', key_pos + search.size());
    if (colon == std::string::npos) return "";

    size_t arr_start = json.find('[', colon + 1);
    if (arr_start == std::string::npos) return "";

    // Find matching closing bracket
    int depth = 1;
    size_t arr_end = arr_start + 1;
    while (arr_end < json.size() && depth > 0) {
        if (json[arr_end] == '[') ++depth;
        else if (json[arr_end] == ']') --depth;
        else if (json[arr_end] == '"') {
            // Skip string content
            ++arr_end;
            while (arr_end < json.size() && !(json[arr_end] == '"' && json[arr_end - 1] != '\\')) {
                ++arr_end;
            }
        }
        ++arr_end;
    }

    if (depth != 0) return "";

    return json.substr(arr_start + 1, arr_end - arr_start - 2);
}

std::vector<std::string> KnowledgeIngestor::json_split_array(
    const std::string& array_str) const
{
    std::vector<std::string> objects;

    int depth = 0;
    size_t obj_start = 0;
    bool in_string = false;

    for (size_t i = 0; i < array_str.size(); ++i) {
        char c = array_str[i];

        if (c == '"' && (i == 0 || array_str[i - 1] != '\\')) {
            in_string = !in_string;
            continue;
        }

        if (in_string) continue;

        if (c == '{') {
            if (depth == 0) obj_start = i;
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                objects.push_back(array_str.substr(obj_start, i - obj_start + 1));
            }
        }
    }

    return objects;
}

double KnowledgeIngestor::json_extract_number(
    const std::string& json, const std::string& key, double default_val) const
{
    // Scope-aware: find key only at the outermost nesting level of the input
    std::string search = "\"" + key + "\"";
    size_t key_pos = std::string::npos;
    {
        int depth = 0;
        bool in_str = false;
        for (size_t i = 0; i < json.size(); ++i) {
            char c = json[i];
            if (c == '"' && (i == 0 || json[i - 1] != '\\')) {
                in_str = !in_str;
            }
            if (in_str) continue;
            if (c == '{') ++depth;
            else if (c == '}') --depth;
            if (depth == 1 && !in_str && i + search.size() <= json.size() && json.compare(i, search.size(), search) == 0) {
                key_pos = i;
                break;
            }
        }
    }
    if (key_pos == std::string::npos) return default_val;

    size_t colon = json.find(':', key_pos + search.size());
    if (colon == std::string::npos) return default_val;

    // Skip whitespace
    size_t val_start = colon + 1;
    while (val_start < json.size() && std::isspace(static_cast<unsigned char>(json[val_start]))) {
        ++val_start;
    }

    if (val_start >= json.size()) return default_val;

    // Read number characters
    size_t val_end = val_start;
    while (val_end < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[val_end])) ||
            json[val_end] == '.' || json[val_end] == '-' || json[val_end] == '+')) {
        ++val_end;
    }

    if (val_end == val_start) return default_val;

    try {
        return std::stod(json.substr(val_start, val_end - val_start));
    } catch (...) {
        return default_val;
    }
}

// =============================================================================
// CSV PARSING
// =============================================================================

ParseResult KnowledgeIngestor::parse_csv_concepts(const std::string& csv_str) const {
    if (csv_str.empty()) {
        return ParseResult::error("Empty CSV input");
    }

    auto lines = split_lines(csv_str);
    if (lines.size() < 2) {
        return ParseResult::error("CSV must have header + at least one data row");
    }

    // Parse header
    auto header = csv_split_line(lines[0]);
    if (header.size() < 2) {
        return ParseResult::error("CSV header must have at least 'label' and 'definition'");
    }

    // Find column indices
    int col_label = -1, col_def = -1, col_trust = -1, col_trust_val = -1;
    for (size_t i = 0; i < header.size(); ++i) {
        std::string h = unquote(header[i]);
        // Normalize
        std::string lower;
        for (char c : h) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (lower == "label" || lower == "name" || lower == "concept") col_label = static_cast<int>(i);
        else if (lower == "definition" || lower == "description" || lower == "def") col_def = static_cast<int>(i);
        else if (lower == "trust" || lower == "trust_category" || lower == "category") col_trust = static_cast<int>(i);
        else if (lower == "trust_value" || lower == "confidence" || lower == "value") col_trust_val = static_cast<int>(i);
    }

    if (col_label < 0) {
        return ParseResult::error("CSV must have a 'label' column");
    }
    if (col_def < 0) {
        return ParseResult::error("CSV must have a 'definition' column");
    }

    StructuredInput input;

    for (size_t row = 1; row < lines.size(); ++row) {
        auto fields = csv_split_line(lines[row]);
        if (fields.empty()) continue;

        StructuredConcept cpt;
        if (col_label >= 0 && static_cast<size_t>(col_label) < fields.size()) {
            cpt.label = unquote(fields[static_cast<size_t>(col_label)]);
        }
        if (col_def >= 0 && static_cast<size_t>(col_def) < fields.size()) {
            cpt.definition = unquote(fields[static_cast<size_t>(col_def)]);
        }
        if (col_trust >= 0 && static_cast<size_t>(col_trust) < fields.size()) {
            cpt.trust_category = unquote(fields[static_cast<size_t>(col_trust)]);
        }
        if (col_trust_val >= 0 && static_cast<size_t>(col_trust_val) < fields.size()) {
            try {
                cpt.trust_value = std::stod(unquote(fields[static_cast<size_t>(col_trust_val)]));
            } catch (...) {
                cpt.trust_value = 0.0;
            }
        }

        if (!cpt.label.empty()) {
            input.concepts.push_back(cpt);
        }
    }

    if (input.concepts.empty()) {
        return ParseResult::error("No valid concepts found in CSV");
    }

    return ParseResult::ok(input);
}

ParseResult KnowledgeIngestor::parse_csv_relations(
    const std::string& csv_str, StructuredInput& existing) const
{
    if (csv_str.empty()) {
        return ParseResult::error("Empty CSV input");
    }

    auto lines = split_lines(csv_str);
    if (lines.size() < 2) {
        return ParseResult::error("CSV must have header + at least one data row");
    }

    auto header = csv_split_line(lines[0]);

    int col_source = -1, col_target = -1, col_type = -1, col_weight = -1;
    for (size_t i = 0; i < header.size(); ++i) {
        std::string h = unquote(header[i]);
        std::string lower;
        for (char c : h) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (lower == "source" || lower == "from" || lower == "subject") col_source = static_cast<int>(i);
        else if (lower == "target" || lower == "to" || lower == "object") col_target = static_cast<int>(i);
        else if (lower == "type" || lower == "relation" || lower == "predicate") col_type = static_cast<int>(i);
        else if (lower == "weight" || lower == "strength") col_weight = static_cast<int>(i);
    }

    if (col_source < 0 || col_target < 0) {
        return ParseResult::error("CSV must have 'source' and 'target' columns");
    }

    for (size_t row = 1; row < lines.size(); ++row) {
        auto fields = csv_split_line(lines[row]);
        if (fields.empty()) continue;

        StructuredRelation rel;
        if (static_cast<size_t>(col_source) < fields.size()) {
            rel.source_label = unquote(fields[static_cast<size_t>(col_source)]);
        }
        if (static_cast<size_t>(col_target) < fields.size()) {
            rel.target_label = unquote(fields[static_cast<size_t>(col_target)]);
        }
        if (col_type >= 0 && static_cast<size_t>(col_type) < fields.size()) {
            rel.relation_type = unquote(fields[static_cast<size_t>(col_type)]);
        }
        if (col_weight >= 0 && static_cast<size_t>(col_weight) < fields.size()) {
            try {
                rel.weight = std::stod(unquote(fields[static_cast<size_t>(col_weight)]));
            } catch (...) {
                rel.weight = 1.0;
            }
        }

        if (!rel.source_label.empty() && !rel.target_label.empty()) {
            existing.relations.push_back(rel);
        }
    }

    return ParseResult::ok(existing);
}

std::vector<IngestProposal> KnowledgeIngestor::to_proposals(
    const StructuredInput& input,
    const TrustTagger& tagger) const
{
    std::vector<IngestProposal> proposals;

    for (const auto& cpt : input.concepts) {
        IngestProposal proposal;
        proposal.concept_label = cpt.label;
        proposal.concept_definition = cpt.definition;
        proposal.source_reference = input.source_reference;

        // Assign trust
        if (!cpt.trust_category.empty()) {
            TrustCategory cat = parse_trust_category(cpt.trust_category);
            if (cpt.trust_value > 0.0) {
                proposal.trust_assignment = tagger.assign_trust_with_value(cat, cpt.trust_value);
            } else {
                proposal.trust_assignment = tagger.assign_trust(cat);
            }
        } else {
            // Default: hypothesis level for unclassified
            proposal.trust_assignment = tagger.assign_trust(TrustCategory::HYPOTHESES);
        }

        // Find relations for this concept
        for (const auto& rel : input.relations) {
            if (rel.source_label == cpt.label || rel.target_label == cpt.label) {
                RelationType rt = parse_relation_type(rel.relation_type);
                proposal.proposed_relations.emplace_back(
                    rel.source_label, rel.target_label, rt,
                    "Structured import: " + rel.relation_type,
                    0.9 // High confidence for structured input
                );
            }
        }

        proposals.push_back(std::move(proposal));
    }

    return proposals;
}

TrustCategory KnowledgeIngestor::parse_trust_category(const std::string& str) {
    std::string upper;
    for (char c : str) {
        upper += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }

    if (upper == "FACT" || upper == "FACTS") return TrustCategory::FACTS;
    if (upper == "DEFINITION" || upper == "DEFINITIONS") return TrustCategory::DEFINITIONS;
    if (upper == "THEORY" || upper == "THEORIES") return TrustCategory::THEORIES;
    if (upper == "HYPOTHESIS" || upper == "HYPOTHESES") return TrustCategory::HYPOTHESES;
    if (upper == "INFERENCE" || upper == "INFERENCES") return TrustCategory::INFERENCES;
    if (upper == "SPECULATION") return TrustCategory::SPECULATION;
    if (upper == "INVALIDATED") return TrustCategory::INVALIDATED;

    return TrustCategory::HYPOTHESES; // Default
}

RelationType KnowledgeIngestor::parse_relation_type(const std::string& str) {
    std::string lower;
    for (char c : str) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (lower == "is-a" || lower == "is_a" || lower == "isa") return RelationType::IS_A;
    if (lower == "has-property" || lower == "has_property" || lower == "has") return RelationType::HAS_PROPERTY;
    if (lower == "causes" || lower == "cause") return RelationType::CAUSES;
    if (lower == "enables" || lower == "enable") return RelationType::ENABLES;
    if (lower == "part-of" || lower == "part_of" || lower == "partof") return RelationType::PART_OF;
    if (lower == "similar-to" || lower == "similar_to" || lower == "similar") return RelationType::SIMILAR_TO;
    if (lower == "contradicts" || lower == "contradict") return RelationType::CONTRADICTS;
    if (lower == "supports" || lower == "support") return RelationType::SUPPORTS;
    if (lower == "temporal-before" || lower == "temporal_before" || lower == "before") return RelationType::TEMPORAL_BEFORE;

    return RelationType::CUSTOM;
}

// CSV helpers

std::vector<std::string> KnowledgeIngestor::csv_split_line(const std::string& line) const {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"'; // Escaped quote
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);

    return fields;
}

std::vector<std::string> KnowledgeIngestor::split_lines(const std::string& text) const {
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        // Remove trailing \r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        // Skip empty lines
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::string KnowledgeIngestor::unquote(const std::string& s) const {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    // Trim whitespace
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

} // namespace brain19

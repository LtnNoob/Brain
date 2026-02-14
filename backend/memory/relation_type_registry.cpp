#include "relation_type_registry.hpp"
#include <algorithm>

namespace brain19 {

// =============================================================================
// Singleton
// =============================================================================

RelationTypeRegistry& RelationTypeRegistry::instance() {
    static RelationTypeRegistry reg;
    return reg;
}

// =============================================================================
// Constructor
// =============================================================================

RelationTypeRegistry::RelationTypeRegistry() {
    // Set up fallback for unknown types
    unknown_fallback_.type = RelationType::CUSTOM;
    unknown_fallback_.name = "UNKNOWN";
    unknown_fallback_.name_de = "steht in Beziehung zu";
    unknown_fallback_.name_en = "is related to";
    unknown_fallback_.slug = "unknown";
    unknown_fallback_.category = RelationCategory::CUSTOM_CATEGORY;
    unknown_fallback_.embedding = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    unknown_fallback_.is_builtin = false;

    register_builtins();
}

// =============================================================================
// Built-in registration
// =============================================================================
//
// Embedding core dimensions (16D):
//   0: hierarchical  1: causal  2: compositional  3: similarity
//   4: temporal       5: support/opposition  6: specificity
//   7: directionality 8: abstractness  9: strength
//   10-15: reserved for learned features (initialized to 0)

void RelationTypeRegistry::register_builtins() {
    // --- Original 10 (0-9) — padded from 10D to 16D core ---
    register_one(RelationType::IS_A, "IS_A",
        "ist ein(e)", "is a", "is-a",
        RelationCategory::HIERARCHICAL,
        {0.9, 0.0, 0.1, 0.3, 0.0, 0.1, 0.7, 0.8, 0.5, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::HAS_PROPERTY, "HAS_PROPERTY",
        "hat die Eigenschaft", "has the property", "has-property",
        RelationCategory::COMPOSITIONAL,
        {0.2, 0.0, 0.8, 0.2, 0.0, 0.1, 0.5, 0.6, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::CAUSES, "CAUSES",
        "verursacht", "causes", "causes",
        RelationCategory::CAUSAL,
        {0.0, 0.9, 0.0, 0.1, 0.7, 0.1, 0.6, 0.9, 0.4, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::ENABLES, "ENABLES",
        "ermoeglicht", "enables", "enables",
        RelationCategory::CAUSAL,
        {0.0, 0.6, 0.1, 0.2, 0.4, 0.3, 0.4, 0.7, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::PART_OF, "PART_OF",
        "ist Teil von", "is part of", "part-of",
        RelationCategory::COMPOSITIONAL,
        {0.6, 0.0, 0.9, 0.2, 0.0, 0.1, 0.6, 0.7, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::SIMILAR_TO, "SIMILAR_TO",
        "ist aehnlich wie", "is similar to", "similar-to",
        RelationCategory::SIMILARITY,
        {0.1, 0.0, 0.1, 0.9, 0.0, 0.2, 0.3, 0.1, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::CONTRADICTS, "CONTRADICTS",
        "widerspricht", "contradicts", "contradicts",
        RelationCategory::OPPOSITION,
        {0.0, 0.1, 0.0, -0.5, 0.0, -0.9, 0.7, 0.5, 0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::SUPPORTS, "SUPPORTS",
        "unterstuetzt", "supports", "supports",
        RelationCategory::EPISTEMIC,
        {0.1, 0.2, 0.1, 0.4, 0.0, 0.9, 0.4, 0.5, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::TEMPORAL_BEFORE, "TEMPORAL_BEFORE",
        "geschieht vor", "occurs before", "temporal-before",
        RelationCategory::TEMPORAL,
        {0.0, 0.3, 0.0, 0.1, 0.9, 0.0, 0.3, 0.8, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::CUSTOM, "CUSTOM",
        "steht in Beziehung zu", "is related to", "custom",
        RelationCategory::CUSTOM_CATEGORY,
        {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    // --- New built-in types (10-19) — padded to 16D core ---
    register_one(RelationType::PRODUCES, "PRODUCES",
        "erzeugt", "produces", "produces",
        RelationCategory::CAUSAL,
        {0.1, 0.8, 0.4, 0.1, 0.3, 0.1, 0.5, 0.8, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::REQUIRES, "REQUIRES",
        "benoetigt", "requires", "requires",
        RelationCategory::FUNCTIONAL,
        {0.1, 0.5, 0.3, 0.1, 0.2, 0.2, 0.5, 0.7, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::USES, "USES",
        "verwendet", "uses", "uses",
        RelationCategory::FUNCTIONAL,
        {0.0, 0.3, 0.4, 0.2, 0.1, 0.2, 0.4, 0.6, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::SOURCE, "SOURCE",
        "stammt von", "originates from", "source",
        RelationCategory::FUNCTIONAL,
        {0.3, 0.2, 0.3, 0.1, 0.4, 0.2, 0.5, 0.7, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::HAS_PART, "HAS_PART",
        "hat als Teil", "has part", "has-part",
        RelationCategory::COMPOSITIONAL,
        {0.6, 0.0, 0.9, 0.2, 0.0, 0.1, 0.6, 0.5, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::TEMPORAL_AFTER, "TEMPORAL_AFTER",
        "geschieht nach", "occurs after", "temporal-after",
        RelationCategory::TEMPORAL,
        {0.0, 0.3, 0.0, 0.1, 0.9, 0.0, 0.3, 0.5, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::INSTANCE_OF, "INSTANCE_OF",
        "ist eine Instanz von", "is an instance of", "instance-of",
        RelationCategory::HIERARCHICAL,
        {0.8, 0.0, 0.1, 0.2, 0.0, 0.1, 0.9, 0.8, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::DERIVED_FROM, "DERIVED_FROM",
        "leitet sich ab von", "is derived from", "derived-from",
        RelationCategory::HIERARCHICAL,
        {0.7, 0.1, 0.2, 0.3, 0.3, 0.1, 0.6, 0.7, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::IMPLIES, "IMPLIES",
        "impliziert", "implies", "implies",
        RelationCategory::CAUSAL,
        {0.3, 0.7, 0.0, 0.2, 0.2, 0.5, 0.6, 0.8, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    register_one(RelationType::ASSOCIATED_WITH, "ASSOCIATED_WITH",
        "ist assoziiert mit", "is associated with", "associated-with",
        RelationCategory::SIMILARITY,
        {0.1, 0.1, 0.1, 0.6, 0.1, 0.2, 0.2, 0.2, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);
}

void RelationTypeRegistry::register_one(
    RelationType type, const std::string& name,
    const std::string& name_de, const std::string& name_en,
    const std::string& slug,
    RelationCategory category, const FlexEmbedding& embedding,
    bool builtin
) {
    RelationTypeInfo info;
    info.type = type;
    info.name = name;
    info.name_de = name_de;
    info.name_en = name_en;
    info.slug = slug;
    info.category = category;
    info.embedding = embedding;
    info.is_builtin = builtin;

    uint16_t key = static_cast<uint16_t>(type);
    types_[key] = std::move(info);
    name_index_[name] = type;   // "IS_A"
    name_index_[slug] = type;   // "is-a"
}

// =============================================================================
// Runtime registration
// =============================================================================

RelationType RelationTypeRegistry::register_type(
    const std::string& name,
    const std::string& name_de,
    RelationCategory category,
    const FlexEmbedding& embedding
) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check for duplicate name
    if (name_index_.count(name)) {
        return name_index_[name];
    }

    auto type = static_cast<RelationType>(next_runtime_id_++);
    register_one(type, name, name_de, name_de, name, category, embedding, false);
    return type;
}

// =============================================================================
// Lookups
// =============================================================================

const RelationTypeInfo& RelationTypeRegistry::get(RelationType type) const {
    auto it = types_.find(static_cast<uint16_t>(type));
    if (it != types_.end()) return it->second;
    return unknown_fallback_;
}

bool RelationTypeRegistry::has(RelationType type) const {
    return types_.count(static_cast<uint16_t>(type)) > 0;
}

std::optional<RelationType> RelationTypeRegistry::find_by_name(const std::string& name) const {
    // Try exact match (covers both name "IS_A" and slug "is-a")
    auto it = name_index_.find(name);
    if (it != name_index_.end()) return it->second;

    // Normalize: uppercase + replace hyphens with underscores
    std::string normalized = name;
    for (auto& ch : normalized) {
        if (ch == '-') ch = '_';
        else ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    it = name_index_.find(normalized);
    if (it != name_index_.end()) return it->second;

    return std::nullopt;
}

std::vector<RelationType> RelationTypeRegistry::all_types() const {
    std::vector<RelationType> result;
    result.reserve(types_.size());
    for (const auto& [key, info] : types_) {
        result.push_back(info.type);
    }
    return result;
}

std::vector<RelationType> RelationTypeRegistry::builtin_types() const {
    std::vector<RelationType> result;
    for (const auto& [key, info] : types_) {
        if (info.is_builtin) result.push_back(info.type);
    }
    return result;
}

size_t RelationTypeRegistry::size() const {
    return types_.size();
}

const FlexEmbedding& RelationTypeRegistry::get_embedding(RelationType type) const {
    return get(type).embedding;
}

const std::string& RelationTypeRegistry::get_name_de(RelationType type) const {
    return get(type).name_de;
}

const std::string& RelationTypeRegistry::get_name_en(RelationType type) const {
    return get(type).name_en;
}

const std::string& RelationTypeRegistry::get_slug(RelationType type) const {
    return get(type).slug;
}

const std::string& RelationTypeRegistry::get_name(RelationType type) const {
    return get(type).name;
}

RelationCategory RelationTypeRegistry::get_category(RelationType type) const {
    return get(type).category;
}

} // namespace brain19

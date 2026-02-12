#pragma once

#include "active_relation.hpp"
#include "../micromodel/micro_model.hpp"  // Vec10, EMBED_DIM
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <mutex>

namespace brain19 {

// Category for grouping relation types (used by domain_manager, template_engine)
enum class RelationCategory : uint8_t {
    HIERARCHICAL,     // IS_A, INSTANCE_OF, DERIVED_FROM
    COMPOSITIONAL,    // PART_OF, HAS_PART, HAS_PROPERTY
    CAUSAL,           // CAUSES, ENABLES, PRODUCES, IMPLIES
    SIMILARITY,       // SIMILAR_TO, ASSOCIATED_WITH
    OPPOSITION,       // CONTRADICTS
    EPISTEMIC,        // SUPPORTS
    TEMPORAL,         // TEMPORAL_BEFORE, TEMPORAL_AFTER
    FUNCTIONAL,       // USES, REQUIRES, SOURCE
    CUSTOM_CATEGORY   // CUSTOM and user-defined
};

struct RelationTypeInfo {
    RelationType type;
    std::string name;       // English identifier: "IS_A", "PRODUCES"
    std::string name_de;    // German natural language: "ist ein(e)", "erzeugt"
    std::string slug;       // Hyphenated: "is-a", "produces"
    RelationCategory category;
    Vec10 embedding;        // 10D embedding for this type
    bool is_builtin;        // true for 0-19, false for >=1000
};

class RelationTypeRegistry {
public:
    // Singleton access
    static RelationTypeRegistry& instance();

    // Lookup by type
    const RelationTypeInfo& get(RelationType type) const;
    bool has(RelationType type) const;

    // Lookup by name
    std::optional<RelationType> find_by_name(const std::string& name) const;

    // Register a new runtime type. Returns the assigned RelationType. Thread-safe.
    RelationType register_type(
        const std::string& name,
        const std::string& name_de,
        RelationCategory category,
        const Vec10& embedding
    );

    // Get all registered types
    std::vector<RelationType> all_types() const;
    std::vector<RelationType> builtin_types() const;
    size_t size() const;

    // Convenience accessors (delegate to get())
    const Vec10& get_embedding(RelationType type) const;
    const std::string& get_name_de(RelationType type) const;
    const std::string& get_slug(RelationType type) const;
    const std::string& get_name(RelationType type) const;
    RelationCategory get_category(RelationType type) const;

private:
    RelationTypeRegistry();
    void register_builtins();
    void register_one(RelationType type, const std::string& name,
                      const std::string& name_de, const std::string& slug,
                      RelationCategory category, const Vec10& embedding,
                      bool builtin);

    std::unordered_map<uint16_t, RelationTypeInfo> types_;
    std::unordered_map<std::string, RelationType> name_index_;
    uint16_t next_runtime_id_ = static_cast<uint16_t>(RelationType::RUNTIME_BASE);
    mutable std::mutex mutex_;

    // Fallback for unknown types
    RelationTypeInfo unknown_fallback_;
};

} // namespace brain19

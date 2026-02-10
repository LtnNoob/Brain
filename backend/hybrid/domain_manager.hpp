#pragma once

#include "../ltm/long_term_memory.hpp"
#include "../kan/kan_module.hpp"
#include "../understanding/mini_llm.hpp"
#include "kan_validator.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <functional>

namespace brain19 {

// =============================================================================
// DOMAIN TYPE
// =============================================================================

enum class DomainType {
    PHYSICAL,    // Physics, mechanics, thermodynamics
    BIOLOGICAL,  // Biology, medicine, ecology
    SOCIAL,      // Psychology, sociology, economics
    ABSTRACT,    // Mathematics, logic, philosophy
    TEMPORAL     // Time-series, scheduling, history
};

inline const char* domain_to_string(DomainType d) {
    switch (d) {
        case DomainType::PHYSICAL: return "PHYSICAL";
        case DomainType::BIOLOGICAL: return "BIOLOGICAL";
        case DomainType::SOCIAL: return "SOCIAL";
        case DomainType::ABSTRACT: return "ABSTRACT";
        case DomainType::TEMPORAL: return "TEMPORAL";
        default: return "UNKNOWN";
    }
}

// Hash for DomainType (needed for unordered_map keys)
struct DomainTypeHash {
    size_t operator()(DomainType d) const noexcept {
        return std::hash<int>{}(static_cast<int>(d));
    }
};

// =============================================================================
// DOMAIN INFO
// =============================================================================

struct DomainInfo {
    DomainType type;
    std::vector<ConceptId> concepts;        // Concepts belonging to this domain
    size_t suggested_num_knots;             // Domain-specific KAN config
    size_t suggested_hidden_dim;

    DomainInfo() = delete;

    explicit DomainInfo(DomainType t)
        : type(t)
        , suggested_num_knots(default_knots(t))
        , suggested_hidden_dim(default_hidden(t))
    {}

private:
    static size_t default_knots(DomainType t) {
        switch (t) {
            case DomainType::PHYSICAL: return 15;   // More precision for physics
            case DomainType::BIOLOGICAL: return 10;
            case DomainType::SOCIAL: return 8;
            case DomainType::ABSTRACT: return 12;
            case DomainType::TEMPORAL: return 10;
            default: return 10;
        }
    }
    static size_t default_hidden(DomainType t) {
        switch (t) {
            case DomainType::PHYSICAL: return 8;
            case DomainType::BIOLOGICAL: return 5;
            case DomainType::SOCIAL: return 5;
            case DomainType::ABSTRACT: return 6;
            case DomainType::TEMPORAL: return 6;
            default: return 5;
        }
    }
};

// =============================================================================
// CROSS-DOMAIN INSIGHT
// =============================================================================

struct CrossDomainInsight {
    DomainType domain_a;
    DomainType domain_b;
    std::vector<ConceptId> bridging_concepts;
    std::string description;
    double novelty_score;  // [0.0, 1.0] — higher = more creative

    CrossDomainInsight() = delete;

    CrossDomainInsight(
        DomainType a, DomainType b,
        std::vector<ConceptId> bridges,
        std::string desc,
        double novelty
    ) : domain_a(a), domain_b(b)
      , bridging_concepts(std::move(bridges))
      , description(std::move(desc))
      , novelty_score(std::max(0.0, std::min(1.0, novelty)))
    {}
};

// =============================================================================
// DOMAIN MANAGER
// =============================================================================
//
// Manages domain-specific KAN-LLM pairs.
// Detects domains from RelationType clustering in LTM.
// Supports cross-domain queries for creative insights.
//
class DomainManager {
public:
    struct Config {
        size_t min_concepts_per_domain = 3;
    };

    DomainManager() : DomainManager(Config{}) {}
    explicit DomainManager(Config config);

    // Detect domain for a concept based on its relations in LTM
    DomainType detect_domain(ConceptId concept_id, const LongTermMemory& ltm) const;

    // Detect domains for multiple concepts and cluster them
    std::unordered_map<DomainType, std::vector<ConceptId>, DomainTypeHash> cluster_by_domain(
        const std::vector<ConceptId>& concepts,
        const LongTermMemory& ltm
    ) const;

    // Get or create domain info (lazy instantiation)
    const DomainInfo& get_domain_info(DomainType type);

    // Get domain-specific KAN validator config
    KanValidator::Config get_domain_validator_config(DomainType type) const;

    // Detect cross-domain opportunities
    std::vector<CrossDomainInsight> find_cross_domain_insights(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm
    ) const;

    // Query active domains
    std::vector<DomainType> get_active_domains() const;
    bool has_domain(DomainType type) const;

    const Config& get_config() const { return config_; }

private:
    Config config_;
    std::unordered_map<DomainType, DomainInfo, DomainTypeHash> domains_;

    // Heuristic: Map RelationType patterns to DomainType
    DomainType classify_relations(
        const std::vector<RelationInfo>& relations,
        const LongTermMemory& ltm
    ) const;
};

} // namespace brain19

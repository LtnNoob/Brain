#include "domain_manager.hpp"
#include <algorithm>
#include <unordered_set>

namespace brain19 {

DomainManager::DomainManager(Config config)
    : config_(std::move(config))
{}

DomainType DomainManager::detect_domain(ConceptId concept_id, const LongTermMemory& ltm) const {
    auto relations = ltm.get_outgoing_relations(concept_id);
    auto incoming = ltm.get_incoming_relations(concept_id);
    relations.insert(relations.end(), incoming.begin(), incoming.end());

    if (relations.empty()) {
        return DomainType::ABSTRACT;  // Default for isolated concepts
    }

    return classify_relations(relations, ltm);
}

DomainType DomainManager::classify_relations(
    const std::vector<RelationInfo>& relations,
    const LongTermMemory& /*ltm*/
) const {
    // Count relation types to determine domain
    int causes_count = 0;
    int temporal_count = 0;
    int is_a_count = 0;
    int has_property_count = 0;
    int similar_count = 0;
    int part_of_count = 0;
    int social_count = 0;  // SUPPORTS, CONTRADICTS

    for (const auto& rel : relations) {
        switch (rel.type) {
            case RelationType::CAUSES:
            case RelationType::ENABLES:
                causes_count++;
                break;
            case RelationType::TEMPORAL_BEFORE:
                temporal_count++;
                break;
            case RelationType::IS_A:
                is_a_count++;
                break;
            case RelationType::HAS_PROPERTY:
                has_property_count++;
                break;
            case RelationType::SIMILAR_TO:
                similar_count++;
                break;
            case RelationType::PART_OF:
                part_of_count++;
                break;
            case RelationType::SUPPORTS:
            case RelationType::CONTRADICTS:
                social_count++;
                break;
            case RelationType::CUSTOM:
                break;
        }
    }

    // Heuristic classification
    if (temporal_count > 0 && temporal_count >= causes_count) {
        return DomainType::TEMPORAL;
    }
    if (causes_count > 0 && has_property_count > 0 && causes_count >= social_count) {
        return DomainType::PHYSICAL;
    }
    if (part_of_count > 0 && is_a_count > 0 && has_property_count > part_of_count) {
        return DomainType::BIOLOGICAL;
    }
    if (social_count > 0 && social_count >= causes_count) {
        return DomainType::SOCIAL;
    }
    if (is_a_count > 0 || similar_count > 0) {
        return DomainType::ABSTRACT;
    }

    return DomainType::ABSTRACT;
}

std::unordered_map<DomainType, std::vector<ConceptId>, DomainTypeHash> DomainManager::cluster_by_domain(
    const std::vector<ConceptId>& concepts,
    const LongTermMemory& ltm
) const {
    std::unordered_map<DomainType, std::vector<ConceptId>, DomainTypeHash> clusters;

    for (auto cid : concepts) {
        DomainType domain = detect_domain(cid, ltm);
        clusters[domain].push_back(cid);
    }

    return clusters;
}

const DomainInfo& DomainManager::get_domain_info(DomainType type) {
    auto it = domains_.find(type);
    if (it == domains_.end()) {
        auto [inserted, _] = domains_.emplace(type, DomainInfo(type));
        return inserted->second;
    }
    return it->second;
}

KanValidator::Config DomainManager::get_domain_validator_config(DomainType type) const {
    KanValidator::Config config;

    // Domain-specific tuning
    switch (type) {
        case DomainType::PHYSICAL:
            config.translator_config.default_num_knots = 15;
            config.translator_config.default_hidden_dim = 8;
            config.max_epochs = 2000;
            config.convergence_threshold = 1e-7;
            break;
        case DomainType::BIOLOGICAL:
            config.translator_config.default_num_knots = 10;
            config.translator_config.default_hidden_dim = 5;
            config.max_epochs = 1000;
            break;
        case DomainType::SOCIAL:
            config.translator_config.default_num_knots = 8;
            config.translator_config.default_hidden_dim = 5;
            config.max_epochs = 800;
            config.bridge_config.hypothesis_mse_threshold = 0.15;  // More lenient
            break;
        case DomainType::ABSTRACT:
            config.translator_config.default_num_knots = 12;
            config.translator_config.default_hidden_dim = 6;
            config.max_epochs = 1500;
            break;
        case DomainType::TEMPORAL:
            config.translator_config.default_num_knots = 10;
            config.translator_config.default_hidden_dim = 6;
            config.max_epochs = 1200;
            break;
    }

    return config;
}

std::vector<CrossDomainInsight> DomainManager::find_cross_domain_insights(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm
) const {
    auto clusters = cluster_by_domain(active_concepts, ltm);
    std::vector<CrossDomainInsight> insights;

    if (clusters.size() < 2) return insights;

    // M3: O(n²) pairwise domain comparison — acceptable for ≤ MAX_DOMAINS domains.
    // If domain count grows beyond this, switch to an index-based approach.
    static constexpr size_t MAX_DOMAINS_FOR_PAIRWISE = 10;
    if (clusters.size() > MAX_DOMAINS_FOR_PAIRWISE) {
        // Guard: skip expensive pairwise scan for too many domains
        return insights;
    }

    // Find concepts that bridge domains (have relations to concepts in other domains)
    std::vector<std::pair<DomainType, std::vector<ConceptId>>> domain_list(
        clusters.begin(), clusters.end()
    );

    for (size_t i = 0; i < domain_list.size(); ++i) {
        for (size_t j = i + 1; j < domain_list.size(); ++j) {
            auto& [domain_a, concepts_a] = domain_list[i];
            auto& [domain_b, concepts_b] = domain_list[j];

            // Find bridging concepts (connected across domains)
            std::vector<ConceptId> bridges;
            std::unordered_set<ConceptId> set_b(concepts_b.begin(), concepts_b.end());

            for (auto ca : concepts_a) {
                auto rels = ltm.get_outgoing_relations(ca);
                for (const auto& rel : rels) {
                    if (set_b.count(rel.target)) {
                        bridges.push_back(ca);
                        bridges.push_back(rel.target);
                    }
                }
            }

            // M4 FIX: Deduplicate bridge concept IDs
            {
                std::sort(bridges.begin(), bridges.end());
                bridges.erase(std::unique(bridges.begin(), bridges.end()), bridges.end());
            }

            if (!bridges.empty()) {
                // L2 FIX: Novelty scores from config (no longer hardcoded)
                double novelty = config_.default_novelty;
                if ((domain_a == DomainType::PHYSICAL && domain_b == DomainType::SOCIAL) ||
                    (domain_a == DomainType::SOCIAL && domain_b == DomainType::PHYSICAL)) {
                    novelty = config_.high_novelty;
                }
                if ((domain_a == DomainType::BIOLOGICAL && domain_b == DomainType::ABSTRACT) ||
                    (domain_a == DomainType::ABSTRACT && domain_b == DomainType::BIOLOGICAL)) {
                    novelty = config_.medium_novelty;
                }

                insights.emplace_back(
                    domain_a, domain_b,
                    std::move(bridges),
                    "Cross-domain connection: " + std::string(domain_to_string(domain_a))
                    + " ↔ " + std::string(domain_to_string(domain_b)),
                    novelty
                );
            }
        }
    }

    return insights;
}

std::vector<DomainType> DomainManager::get_active_domains() const {
    std::vector<DomainType> result;
    for (const auto& [type, _] : domains_) {
        result.push_back(type);
    }
    return result;
}

bool DomainManager::has_domain(DomainType type) const {
    return domains_.count(type) > 0;
}

} // namespace brain19

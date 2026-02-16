#include "property_inheritance.hpp"
#include "../memory/relation_type_registry.hpp"
#include "../core/relation_config.hpp"

#include <algorithm>
#include <iostream>

namespace brain19 {

PropertyInheritance::PropertyInheritance(LongTermMemory& ltm) : ltm_(ltm) {}

// =============================================================================
// Inheritable relation types
// =============================================================================

std::vector<RelationType>
PropertyInheritance::inheritable_types(const Config& cfg) const {
    std::vector<RelationType> types;
    types.push_back(RelationType::HAS_PROPERTY);
    if (cfg.propagate_requires) types.push_back(RelationType::REQUIRES);
    if (cfg.propagate_uses)     types.push_back(RelationType::USES);
    if (cfg.propagate_produces) types.push_back(RelationType::PRODUCES);
    return types;
}

// =============================================================================
// Index builders
// =============================================================================

void PropertyInheritance::build_contradicts_index() {
    contradicts_.clear();
    auto all_ids = ltm_.get_all_concept_ids();
    for (auto cid : all_ids) {
        for (const auto& rel : ltm_.get_outgoing_relations(cid)) {
            if (rel.type == RelationType::CONTRADICTS) {
                contradicts_[cid].insert(rel.target);
            }
        }
        // Also check incoming CONTRADICTS (symmetric blocking)
        for (const auto& rel : ltm_.get_incoming_relations(cid)) {
            if (rel.type == RelationType::CONTRADICTS) {
                contradicts_[cid].insert(rel.source);
            }
        }
    }
}

void PropertyInheritance::build_existing_triples_index(
    const std::vector<RelationType>& types)
{
    existing_triples_.clear();
    auto all_ids = ltm_.get_all_concept_ids();
    for (auto cid : all_ids) {
        for (const auto& rel : ltm_.get_outgoing_relations(cid)) {
            for (auto t : types) {
                if (rel.type == t) {
                    existing_triples_.insert(triple_key(cid, rel.target));
                    break;
                }
            }
        }
    }
}

uint64_t PropertyInheritance::triple_key(ConceptId src, ConceptId tgt) {
    return (uint64_t(src) << 32) | uint64_t(tgt);
}

bool PropertyInheritance::is_contradicted(ConceptId cid,
                                          ConceptId property_target) const {
    auto it = contradicts_.find(cid);
    if (it == contradicts_.end()) return false;
    return it->second.count(property_target) > 0;
}

// =============================================================================
// Single fixpoint iteration
// =============================================================================
//
// For each concept C:
//   For each IS_A parent P of C:
//     For each inheritable property of P:
//       Determine true origin: if P inherited it, use inherited trust/hops
//       Compute trust = source_trust * decay_per_hop
//       If trust >= floor AND not contradicted AND not already better:
//         Record inheritance and add relation to LTM
//
// Key: parent properties in LTM include both original AND previously-inherited
// ones. We use inherited_[parent] to distinguish and get correct hop counts.

size_t PropertyInheritance::iterate_once(
    const Config& cfg, Result& result,
    const std::vector<RelationType>& types)
{
    auto& reg = RelationTypeRegistry::instance();
    auto all_ids = ltm_.get_all_concept_ids();
    size_t added = 0;

    for (auto cid : all_ids) {
        // Find IS_A parents of this concept
        for (const auto& isa_rel : ltm_.get_outgoing_relations(cid)) {
            if (isa_rel.type != RelationType::IS_A) continue;
            ConceptId parent = isa_rel.target;

            // Iterate all inheritable outgoing relations of parent
            for (const auto& prop_rel : ltm_.get_outgoing_relations(parent)) {
                bool is_inheritable = false;
                for (auto t : types) {
                    if (prop_rel.type == t) { is_inheritable = true; break; }
                }
                if (!is_inheritable) continue;

                ConceptId prop_tgt = prop_rel.target;

                // Determine true trust and hop count at parent level.
                // If parent inherited this property, use the inherited record;
                // otherwise it's an original property (hop 0).
                double parent_trust = prop_rel.weight;
                size_t parent_hops = 0;
                ConceptId origin = parent;

                auto parent_inh_it = inherited_.find(parent);
                if (parent_inh_it != inherited_.end()) {
                    auto prop_inh_it = parent_inh_it->second.find(prop_tgt);
                    if (prop_inh_it != parent_inh_it->second.end()) {
                        parent_trust = prop_inh_it->second.inherited_trust;
                        parent_hops = prop_inh_it->second.hop_count;
                        origin = prop_inh_it->second.origin_concept;
                    }
                }

                // Hop depth check
                if (parent_hops + 1 > cfg.max_hop_depth) continue;

                // CONTRADICTS check
                if (is_contradicted(cid, prop_tgt)) {
                    result.contradictions_blocked++;
                    continue;
                }

                // Decay trust one hop from parent — per-relation-type decay (Convergence v2, Audit #4)
                const RelationBehavior& behavior = get_behavior(isa_rel.type);
                double decay = (behavior.trust_decay_per_hop > 0.0)
                    ? behavior.trust_decay_per_hop : cfg.decay_per_hop;
                double decayed = parent_trust * decay;

                if (decayed < cfg.trust_floor) {
                    result.trust_floor_cutoffs++;
                    continue;
                }

                // Dedup: already exists as original in LTM?
                uint64_t tkey = triple_key(cid, prop_tgt);
                if (existing_triples_.count(tkey) > 0) {
                    result.duplicates_skipped++;
                    continue;
                }

                // Already inherited with equal or higher trust?
                auto& cid_inherited = inherited_[cid];
                auto existing_it = cid_inherited.find(prop_tgt);
                if (existing_it != cid_inherited.end()) {
                    if (decayed <= existing_it->second.inherited_trust) {
                        result.duplicates_skipped++;
                        continue;
                    }
                }

                // Record inheritance
                InheritedProperty ip;
                ip.property_target = prop_tgt;
                ip.origin_concept = origin;
                ip.inherited_trust = decayed;
                ip.hop_count = parent_hops + 1;
                cid_inherited[prop_tgt] = ip;

                // Add relation to LTM
                double rel_weight = std::max(decayed, cfg.min_relation_weight);
                ltm_.add_relation(cid, prop_tgt, prop_rel.type, rel_weight);
                existing_triples_.insert(tkey);

                added++;
                result.properties_inherited++;
                result.type_distribution[reg.get_name_en(prop_rel.type)]++;
            }
        }
    }

    return added;
}

// =============================================================================
// Main entry: fixpoint iteration
// =============================================================================

PropertyInheritance::Result
PropertyInheritance::propagate(const Config& config) {
    Result result;

    auto types = inheritable_types(config);
    auto all_ids = ltm_.get_all_concept_ids();
    result.concepts_processed = all_ids.size();

    std::cerr << "[PropertyInheritance] Starting: "
              << result.concepts_processed << " concepts\n";

    // Build indices
    build_contradicts_index();
    build_existing_triples_index(types);

    // Clear previous inheritance state
    inherited_.clear();

    // Fixpoint iteration
    for (size_t iter = 0; iter < config.max_iterations; ++iter) {
        size_t added = iterate_once(config, result, types);
        result.iterations_run = iter + 1;

        std::cerr << "[PropertyInheritance] Iteration " << (iter + 1)
                  << ": +" << added << " properties\n";

        if (added == 0) {
            result.converged = true;
            break;
        }
    }

    std::cerr << "[PropertyInheritance] Done: "
              << result.properties_inherited << " inherited, "
              << result.contradictions_blocked << " blocked, "
              << result.trust_floor_cutoffs << " below floor, "
              << result.iterations_run << " iterations"
              << (result.converged ? " (converged)" : " (max reached)")
              << "\n";

    return result;
}

// =============================================================================
// Query inherited properties
// =============================================================================

std::vector<InheritedProperty>
PropertyInheritance::get_inherited(ConceptId cid) const {
    std::vector<InheritedProperty> out;
    auto it = inherited_.find(cid);
    if (it == inherited_.end()) return out;

    out.reserve(it->second.size());
    for (const auto& [prop_tgt, ip] : it->second) {
        out.push_back(ip);
    }

    // Sort by trust descending for deterministic output
    std::sort(out.begin(), out.end(),
              [](const InheritedProperty& a, const InheritedProperty& b) {
                  return a.inherited_trust > b.inherited_trust;
              });

    return out;
}

} // namespace brain19

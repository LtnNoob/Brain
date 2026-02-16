#include "graph_densifier.hpp"
#include "../memory/relation_type_registry.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

namespace brain19 {

GraphDensifier::GraphDensifier(LongTermMemory& ltm) : ltm_(ltm) {}

uint64_t GraphDensifier::pair_key(ConceptId a, ConceptId b) {
    return (uint64_t(a) << 32) | uint64_t(b);
}

bool GraphDensifier::pair_exists(ConceptId a, ConceptId b) const {
    return existing_pairs_.count(pair_key(a, b)) > 0
        || existing_pairs_.count(pair_key(b, a)) > 0;
}

void GraphDensifier::build_existing_pairs_index() {
    existing_pairs_.clear();
    auto all_ids = ltm_.get_all_concept_ids();
    for (auto cid : all_ids) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (cinfo && cinfo->is_anti_knowledge) continue;
        for (const auto& rel : ltm_.get_outgoing_relations(cid)) {
            existing_pairs_.insert(pair_key(rel.source, rel.target));
        }
    }
}

bool GraphDensifier::try_add(ConceptId src, ConceptId tgt, RelationType type,
                             double weight, size_t cap, Result& result,
                             const std::string& type_name) {
    if (src == tgt) return false;
    if (pair_exists(src, tgt)) { result.duplicates_skipped++; return false; }
    if (new_rel_count_[src] >= cap || new_rel_count_[tgt] >= cap) {
        result.cap_limited++;
        return false;
    }

    auto rid = ltm_.add_relation(src, tgt, type, weight);
    if (rid > 0) {
        generated_ids_.push_back(rid);
        existing_pairs_.insert(pair_key(src, tgt));
        new_rel_count_[src]++;
        new_rel_count_[tgt]++;
        result.relations_added++;
        result.type_distribution[type_name]++;
        return true;
    }
    return false;
}

// =============================================================================
// Main entry point — iterative densification
// =============================================================================

GraphDensifier::Result GraphDensifier::densify(const Config& config) {
    Result result;
    auto t0 = std::chrono::steady_clock::now();

    auto all_ids = ltm_.get_all_concept_ids();
    result.concepts = all_ids.size();

    // Count existing relations
    size_t total_rels_before = 0;
    for (auto cid : all_ids)
        total_rels_before += ltm_.get_outgoing_relations(cid).size();
    result.density_before = result.concepts > 0
        ? double(total_rels_before) / result.concepts : 0.0;

    std::cerr << "[GraphDensifier] Starting: " << result.concepts
              << " concepts, " << total_rels_before << " relations (density "
              << result.density_before << ")\n";

    // Iterative densification: each pass builds on the previous
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        size_t added_this_iter = 0;

        // Rebuild pair index (includes relations added in previous iterations)
        build_existing_pairs_index();

        if (iter == 0) {
            std::cerr << "[GraphDensifier] Pair index: "
                      << existing_pairs_.size() << " existing pairs\n";
        }

        // Phase 1: Property inheritance
        if (config.enable_property_inheritance) {
            size_t n = phase_property_inheritance(config, result);
            result.phase_counts["property_inherit"] += n;
            added_this_iter += n;
        }

        // Phase 2: Transitive IS_A
        if (config.enable_transitive_isa) {
            size_t n = phase_transitive_isa(config, result);
            result.phase_counts["transitive_isa"] += n;
            added_this_iter += n;
        }

        // Phase 3: PART_OF transitivity
        if (config.enable_partof_transitive) {
            size_t n = phase_partof_transitive(config, result);
            result.phase_counts["partof_transitive"] += n;
            added_this_iter += n;
        }

        // Phase 4: Causal transitivity
        if (config.enable_causal_transitive) {
            size_t n = phase_causal_transitive(config, result);
            result.phase_counts["causal_transitive"] += n;
            added_this_iter += n;
        }

        // Phase 5: PART_OF property inheritance
        if (config.enable_partof_property) {
            size_t n = phase_partof_property(config, result);
            result.phase_counts["partof_property"] += n;
            added_this_iter += n;
        }

        // Phase 6: Co-activation (last, benefits from other phases' additions)
        if (config.enable_coactivation) {
            size_t n = phase_coactivation(config, result);
            result.phase_counts["coactivation"] += n;
            added_this_iter += n;
        }

        std::cerr << "[GraphDensifier] Iteration " << (iter + 1)
                  << ": +" << added_this_iter << " relations\n";

        // Stop early if no new relations were added (convergence)
        if (added_this_iter == 0) break;
    }

    size_t total_rels_after = total_rels_before + result.relations_added;
    result.density_after = result.concepts > 0
        ? double(total_rels_after) / result.concepts : 0.0;

    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cerr << "[GraphDensifier] Done: +" << result.relations_added
              << " relations, density " << result.density_before
              << " → " << result.density_after
              << " (" << total_ms << "ms)\n";

    return result;
}

// =============================================================================
// Phase 1: Property Inheritance
// =============================================================================
//
// If A IS_A B, copy B's outgoing HAS_PROPERTY/REQUIRES/USES/PRODUCES to A.
// Iterative: also inherits properties from grandparents (via transitive IS_A).

size_t GraphDensifier::phase_property_inheritance(const Config& cfg, Result& r) {
    static const std::vector<RelationType> inheritable = {
        RelationType::HAS_PROPERTY,
        RelationType::REQUIRES,
        RelationType::USES,
        RelationType::PRODUCES,
    };

    auto& reg = RelationTypeRegistry::instance();
    auto all_ids = ltm_.get_all_concept_ids();
    size_t added = 0;

    for (auto cid : all_ids) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (cinfo && cinfo->is_anti_knowledge) continue;
        for (const auto& isa_rel : ltm_.get_outgoing_relations(cid)) {
            if (isa_rel.type != RelationType::IS_A) continue;
            ConceptId parent = isa_rel.target;

            for (const auto& parent_rel : ltm_.get_outgoing_relations(parent)) {
                bool inheritable_type = false;
                for (auto t : inheritable) {
                    if (parent_rel.type == t) { inheritable_type = true; break; }
                }
                if (!inheritable_type) continue;

                if (try_add(cid, parent_rel.target, parent_rel.type,
                            parent_rel.weight * 0.8,
                            cfg.max_new_rels_per_concept, r,
                            reg.get_name_en(parent_rel.type)))
                    added++;
            }
        }
    }

    return added;
}

// =============================================================================
// Phase 2: Transitive IS_A (up to 3 hops)
// =============================================================================

size_t GraphDensifier::phase_transitive_isa(const Config& cfg, Result& r) {
    auto all_ids = ltm_.get_all_concept_ids();
    size_t added = 0;

    for (auto cid : all_ids) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (cinfo && cinfo->is_anti_knowledge) continue;
        for (const auto& rel1 : ltm_.get_outgoing_relations(cid)) {
            if (rel1.type != RelationType::IS_A) continue;
            ConceptId parent = rel1.target;

            for (const auto& rel2 : ltm_.get_outgoing_relations(parent)) {
                if (rel2.type != RelationType::IS_A) continue;
                ConceptId grandparent = rel2.target;

                if (try_add(cid, grandparent, RelationType::IS_A,
                            cfg.base_weight * 0.8,
                            cfg.max_new_rels_per_concept, r, "IS_A"))
                    added++;

                // 3rd hop: great-grandparent
                for (const auto& rel3 : ltm_.get_outgoing_relations(grandparent)) {
                    if (rel3.type != RelationType::IS_A) continue;
                    if (try_add(cid, rel3.target, RelationType::IS_A,
                                cfg.base_weight * 0.6,
                                cfg.max_new_rels_per_concept, r, "IS_A"))
                        added++;
                }
            }
        }
    }

    return added;
}

// =============================================================================
// Phase 3: Co-activation (shared-neighbor ASSOCIATED_WITH)
// =============================================================================

size_t GraphDensifier::phase_coactivation(const Config& cfg, Result& r) {
    auto& reg = RelationTypeRegistry::instance();
    auto all_ids = ltm_.get_all_concept_ids();

    // Identify broad category concepts (many incoming IS_A)
    std::unordered_map<ConceptId, size_t> isa_in_degree;
    for (auto cid : all_ids) {
        for (const auto& rel : ltm_.get_outgoing_relations(cid)) {
            if (rel.type == RelationType::IS_A)
                isa_in_degree[rel.target]++;
        }
    }

    // Build neighbor sets — only specific typed relations
    // Exclude: IS_A, INSTANCE_OF, SIMILAR_TO (hierarchical/broad)
    // Exclude: LINGUISTIC relations (SUBJECT_OF, VERB_OF, etc.)
    std::unordered_map<ConceptId, std::unordered_set<ConceptId>> neighbors;
    for (auto cid : all_ids) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (cinfo && cinfo->is_anti_knowledge) continue;
        if (isa_in_degree[cid] > 8) continue;  // skip broad categories

        auto& ns = neighbors[cid];
        for (const auto& rel : ltm_.get_outgoing_relations(cid)) {
            if (reg.get_category(rel.type) == RelationCategory::LINGUISTIC) continue;
            if (rel.type != RelationType::IS_A &&
                rel.type != RelationType::INSTANCE_OF &&
                rel.type != RelationType::SIMILAR_TO &&
                rel.type != RelationType::ASSOCIATED_WITH &&
                isa_in_degree[rel.target] <= 8)
                ns.insert(rel.target);
        }
        for (const auto& rel : ltm_.get_incoming_relations(cid)) {
            if (reg.get_category(rel.type) == RelationCategory::LINGUISTIC) continue;
            if (rel.type != RelationType::IS_A &&
                rel.type != RelationType::INSTANCE_OF &&
                rel.type != RelationType::SIMILAR_TO &&
                rel.type != RelationType::ASSOCIATED_WITH &&
                isa_in_degree[rel.source] <= 8)
                ns.insert(rel.source);
        }
    }

    // Count common neighbors
    struct PairInfo { size_t common_count = 0; };
    std::unordered_map<uint64_t, PairInfo> pair_counts;

    for (auto cid : all_ids) {
        if (neighbors.find(cid) == neighbors.end()) continue;
        const auto& ns = neighbors[cid];
        if (ns.size() < 2) continue;

        std::vector<ConceptId> nv(ns.begin(), ns.end());
        std::sort(nv.begin(), nv.end());

        size_t limit = std::min(nv.size(), size_t(150));
        for (size_t i = 0; i < limit; ++i) {
            for (size_t j = i + 1; j < limit; ++j) {
                uint64_t key = pair_key(nv[i], nv[j]);
                pair_counts[key].common_count++;
            }
        }
    }

    size_t added = 0;

    for (const auto& [key, info] : pair_counts) {
        if (info.common_count < cfg.min_common_neighbors) continue;

        ConceptId a = static_cast<ConceptId>(key >> 32);
        ConceptId b = static_cast<ConceptId>(key & 0xFFFFFFFF);

        size_t deg_a = neighbors.count(a) ? neighbors[a].size() : 0;
        size_t deg_b = neighbors.count(b) ? neighbors[b].size() : 0;
        if (deg_a == 0 || deg_b == 0) continue;

        double jaccard = double(info.common_count)
            / double(deg_a + deg_b - info.common_count);
        if (jaccard < cfg.jaccard_threshold) continue;

        double weight = cfg.base_weight * std::min(jaccard * 2.0, 1.0);

        if (try_add(a, b, RelationType::ASSOCIATED_WITH,
                    weight, cfg.max_new_rels_per_concept, r, "ASSOCIATED_WITH"))
            added++;
    }

    return added;
}

// =============================================================================
// Phase 4: PART_OF Transitivity
// =============================================================================
//
// If A PART_OF B and B PART_OF C, then A PART_OF C (with decaying weight).

size_t GraphDensifier::phase_partof_transitive(const Config& cfg, Result& r) {
    auto all_ids = ltm_.get_all_concept_ids();
    size_t added = 0;

    for (auto cid : all_ids) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (cinfo && cinfo->is_anti_knowledge) continue;
        for (const auto& rel1 : ltm_.get_outgoing_relations(cid)) {
            if (rel1.type != RelationType::PART_OF) continue;
            ConceptId whole = rel1.target;

            for (const auto& rel2 : ltm_.get_outgoing_relations(whole)) {
                if (rel2.type != RelationType::PART_OF) continue;
                if (try_add(cid, rel2.target, RelationType::PART_OF,
                            cfg.base_weight * 0.7,
                            cfg.max_new_rels_per_concept, r, "PART_OF"))
                    added++;
            }
        }
    }

    return added;
}

// =============================================================================
// Phase 5: Causal Transitivity
// =============================================================================
//
// A CAUSES B, B CAUSES C → A ENABLES C (2-hop causal chain)
// A CAUSES B, B ENABLES C → A ENABLES C (cause → enablement)
// A ENABLES B, B CAUSES C → A ENABLES C (enablement → cause)

size_t GraphDensifier::phase_causal_transitive(const Config& cfg, Result& r) {
    auto all_ids = ltm_.get_all_concept_ids();
    size_t added = 0;
    size_t chains_found = 0;

    for (auto cid : all_ids) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (cinfo && cinfo->is_anti_knowledge) continue;
        for (const auto& rel1 : ltm_.get_outgoing_relations(cid)) {
            if (rel1.type != RelationType::CAUSES &&
                rel1.type != RelationType::ENABLES) continue;

            ConceptId mid = rel1.target;
            for (const auto& rel2 : ltm_.get_outgoing_relations(mid)) {
                if (rel2.type != RelationType::CAUSES &&
                    rel2.type != RelationType::ENABLES) continue;

                chains_found++;
                // Causal chains produce ENABLES (weaker than direct CAUSES)
                if (try_add(cid, rel2.target, RelationType::ENABLES,
                            cfg.base_weight * 0.7,
                            cfg.max_new_rels_per_concept, r, "ENABLES"))
                    added++;
            }
        }
    }

    return added;
}

// =============================================================================
// Phase 6: PART_OF Property Inheritance
// =============================================================================
//
// FIXED (Convergence v2, Audit #5): Properties flow from PART to WHOLE (REVERSE direction).
// If A PART_OF B and A HAS_PROPERTY P → B HAS_PROPERTY P
// (wholes inherit aggregate physical properties of their parts)

size_t GraphDensifier::phase_partof_property(const Config& cfg, Result& r) {
    auto& reg = RelationTypeRegistry::instance();
    auto all_ids = ltm_.get_all_concept_ids();
    size_t added = 0;
    size_t chains_found = 0;

    for (auto part_id : all_ids) {
        auto cinfo = ltm_.retrieve_concept(part_id);
        if (cinfo && cinfo->is_anti_knowledge) continue;
        for (const auto& partof_rel : ltm_.get_outgoing_relations(part_id)) {
            if (partof_rel.type != RelationType::PART_OF) continue;
            ConceptId whole = partof_rel.target;

            // Properties of the PART propagate to the WHOLE
            for (const auto& prop_rel : ltm_.get_outgoing_relations(part_id)) {
                if (prop_rel.type != RelationType::HAS_PROPERTY) continue;

                chains_found++;
                // COMPOSITIONAL trust_decay = 1.0 (structural facts don't decay)
                if (try_add(whole, prop_rel.target, RelationType::HAS_PROPERTY,
                            prop_rel.weight * 0.8,
                            cfg.max_new_rels_per_concept, r,
                            reg.get_name_en(RelationType::HAS_PROPERTY)))
                    added++;
            }
        }
    }

    return added;
}

// =============================================================================
// Quality Gate: Sample generated relations
// =============================================================================

std::vector<GraphDensifier::SampledRelation>
GraphDensifier::sample_generated(size_t n) const {
    std::vector<SampledRelation> samples;
    if (generated_ids_.empty()) return samples;

    std::vector<size_t> indices(generated_ids_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    size_t count = std::min(n, indices.size());

    auto& reg = RelationTypeRegistry::instance();

    for (size_t i = 0; i < count; ++i) {
        auto rel_opt = ltm_.get_relation(generated_ids_[indices[i]]);
        if (!rel_opt) continue;

        auto src_info = ltm_.retrieve_concept(rel_opt->source);
        auto tgt_info = ltm_.retrieve_concept(rel_opt->target);

        SampledRelation s;
        s.source_label = src_info ? src_info->label : "?";
        s.target_label = tgt_info ? tgt_info->label : "?";
        s.type_name = reg.get_name_en(rel_opt->type);
        s.weight = rel_opt->weight;
        samples.push_back(std::move(s));
    }

    return samples;
}

// =============================================================================
// Linguistic Graph Support (Phase 4)
// =============================================================================

bool GraphDensifier::word_concept_exists(const std::string& surface_form, const std::string& pos) const {
    std::string label = "word:" + surface_form + ":" + pos;
    return !ltm_.find_by_label(label).empty();
}

std::optional<ConceptId> GraphDensifier::find_duplicate_sentence(
    ConceptId subject, ConceptId verb, ConceptId object) const
{
    if (subject == 0 || verb == 0) return std::nullopt;

    // Get candidate sentences from subject's SUBJECT_OF relations
    for (const auto& rel : ltm_.get_outgoing_relations(subject)) {
        if (rel.type != RelationType::SUBJECT_OF) continue;
        ConceptId candidate_sent = rel.target;

        // Check verb has VERB_OF → same sentence
        bool verb_match = false;
        for (const auto& vrel : ltm_.get_outgoing_relations(verb)) {
            if (vrel.type == RelationType::VERB_OF && vrel.target == candidate_sent) {
                verb_match = true;
                break;
            }
        }
        if (!verb_match) continue;

        // Check object if present
        if (object != 0) {
            bool obj_match = false;
            for (const auto& orel : ltm_.get_outgoing_relations(object)) {
                if (orel.type == RelationType::OBJECT_OF && orel.target == candidate_sent) {
                    obj_match = true;
                    break;
                }
            }
            if (!obj_match) continue;
        }

        return candidate_sent;
    }

    return std::nullopt;
}

std::vector<std::pair<ConceptId, double>> GraphDensifier::infer_denotes_relations(
    ConceptId word_cid) const
{
    std::vector<std::pair<ConceptId, double>> results;

    auto word_info = ltm_.retrieve_concept(word_cid);
    if (!word_info) return results;

    // Extract surface form from label "word:surface:POS"
    const auto& label = word_info->label;
    if (label.size() < 6 || label.substr(0, 5) != "word:") return results;

    auto second_colon = label.find(':', 5);
    if (second_colon == std::string::npos) return results;
    std::string surface = label.substr(5, second_colon - 5);

    // Label match via index: find semantic concepts with matching label
    auto candidates = ltm_.find_by_label(surface);
    for (auto cid : candidates) {
        if (cid == word_cid) continue;
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;
        // Skip other linguistic concepts
        if (info->label.size() >= 5 &&
            (info->label.substr(0, 5) == "word:" || info->label.substr(0, 4) == "sent:"))
            continue;
        results.push_back({cid, 0.9});
    }

    return results;
}

} // namespace brain19

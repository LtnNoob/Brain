// Concept quality audit: sample 20 random concepts, show their data
#include "ltm/long_term_memory.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "micromodel/concept_embedding_store.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "memory/relation_type_registry.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <map>
#include <set>

using namespace brain19;

static const char* rel_type_name(RelationType rt) {
    switch (rt) {
        case RelationType::IS_A: return "IS_A";
        case RelationType::HAS_PROPERTY: return "HAS_PROPERTY";
        case RelationType::PART_OF: return "PART_OF";
        case RelationType::HAS_PART: return "HAS_PART";
        case RelationType::CAUSES: return "CAUSES";
        case RelationType::ENABLES: return "ENABLES";
        case RelationType::REQUIRES: return "REQUIRES";
        case RelationType::USES: return "USES";
        case RelationType::PRODUCES: return "PRODUCES";
        case RelationType::IMPLIES: return "IMPLIES";
        case RelationType::INSTANCE_OF: return "INSTANCE_OF";
        case RelationType::DERIVED_FROM: return "DERIVED_FROM";
        default: return "OTHER";
    }
}

int main() {
    LongTermMemory ltm;
    FoundationConcepts::seed_from_file(ltm, "../data/foundation_full.json");

    PropertyInheritance prop(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9; cfg.trust_floor = 0.3; cfg.max_iterations = 50;
    cfg.max_hop_depth = 20;
    cfg.propagate_requires = true; cfg.propagate_uses = true; cfg.propagate_produces = true;
    prop.propagate(cfg);

    auto all_ids = ltm.get_all_concept_ids();
    std::cout << "Total concepts: " << all_ids.size() << "\n";
    std::cout << "Total relations: " << ltm.total_relation_count() << "\n\n";

    // Global stats
    std::map<std::string, int> type_counts;
    int no_rels = 0, no_def = 0, short_label = 0;
    int has_is_a = 0, has_property = 0;
    for (auto cid : all_ids) {
        auto info = ltm.retrieve_concept(cid);
        if (!info) continue;
        type_counts[epistemic_type_to_string(info->epistemic.type)]++;
        auto rels = ltm.get_outgoing_relations(cid);
        if (rels.empty()) no_rels++;
        if (info->definition.empty()) no_def++;
        if (info->label.size() < 3) short_label++;
        for (auto& r : rels) {
            if (r.type == RelationType::IS_A) { has_is_a++; break; }
        }
        for (auto& r : rels) {
            if (r.type == RelationType::HAS_PROPERTY) { has_property++; break; }
        }
    }

    std::cout << "=== GLOBAL STATS ===\n";
    std::cout << "Epistemic types:\n";
    for (auto& [t, c] : type_counts)
        std::cout << "  " << t << ": " << c << " (" << (100.0*c/all_ids.size()) << "%)\n";
    std::cout << "No outgoing relations: " << no_rels << "\n";
    std::cout << "No definition: " << no_def << "\n";
    std::cout << "Label < 3 chars: " << short_label << "\n";
    std::cout << "Has IS_A relation: " << has_is_a << "\n";
    std::cout << "Has HAS_PROPERTY: " << has_property << "\n\n";

    // Relation type distribution
    std::map<int, int> rel_type_dist;
    for (auto cid : all_ids) {
        for (auto& r : ltm.get_outgoing_relations(cid))
            rel_type_dist[(int)r.type]++;
    }
    std::cout << "=== RELATION TYPE DISTRIBUTION ===\n";
    for (auto& [t, c] : rel_type_dist) {
        RelationType rt = static_cast<RelationType>(t);
        std::cout << "  " << rel_type_name(rt) << " (" << t << "): " << c << "\n";
    }
    std::cout << "\n";

    // Sample 20 random concepts
    std::mt19937 rng(12345);
    std::shuffle(all_ids.begin(), all_ids.end(), rng);

    std::cout << "=== SAMPLE 20 CONCEPTS (random) ===\n\n";
    for (size_t i = 0; i < 20 && i < all_ids.size(); ++i) {
        auto cid = all_ids[i];
        auto info = ltm.retrieve_concept(cid);
        if (!info) continue;

        auto rels = ltm.get_outgoing_relations(cid);
        std::cout << "[" << i << "] cid=" << cid << " label=\"" << info->label << "\"\n";
        std::cout << "    type=" << epistemic_type_to_string(info->epistemic.type)
                  << " trust=" << info->epistemic.trust << "\n";
        std::cout << "    definition: " << (info->definition.empty() ? "(none)" :
                     (info->definition.size() > 120 ? info->definition.substr(0, 120) + "..." : info->definition))
                  << "\n";
        std::cout << "    outgoing: " << rels.size() << " relations\n";
        for (size_t j = 0; j < std::min(rels.size(), (size_t)5); ++j) {
            auto tinfo = ltm.retrieve_concept(rels[j].target);
            std::cout << "      -> " << rel_type_name(rels[j].type)
                      << " \"" << (tinfo ? tinfo->label : "?")
                      << "\" (w=" << rels[j].weight << ")\n";
        }
        if (rels.size() > 5) std::cout << "      ... +" << (rels.size()-5) << " more\n";
        std::cout << "\n";
    }

    // Sample 5 test concepts
    std::cout << "=== TEST CONCEPTS (inference queries) ===\n\n";
    for (const auto& q : {"photosynthesis", "gravity", "water", "evolution", "electricity"}) {
        auto ids = ltm.find_by_label(q);
        if (ids.empty()) { std::cout << q << ": NOT FOUND\n\n"; continue; }
        auto cid = ids[0];
        auto info = ltm.retrieve_concept(cid);
        auto rels = ltm.get_outgoing_relations(cid);
        std::cout << "\"" << q << "\" cid=" << cid << " label=\"" << (info ? info->label : "?") << "\"\n";
        std::cout << "    type=" << epistemic_type_to_string(info->epistemic.type)
                  << " trust=" << info->epistemic.trust << "\n";
        std::cout << "    outgoing: " << rels.size() << " relations\n";
        for (size_t j = 0; j < rels.size(); ++j) {
            auto tinfo = ltm.retrieve_concept(rels[j].target);
            std::cout << "      -> " << rel_type_name(rels[j].type)
                      << " \"" << (tinfo ? tinfo->label : "?")
                      << "\" (w=" << rels[j].weight << ")\n";
        }
        std::cout << "\n";
    }

    return 0;
}

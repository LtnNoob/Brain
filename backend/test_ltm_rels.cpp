// Quick check: does the LTM have outgoing relations for Photosynthesis etc?
#include "ltm/long_term_memory.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include <iostream>

using namespace brain19;

int main() {
    LongTermMemory ltm;
    FoundationConcepts::seed_from_file(ltm, "../data/foundation_full.json");
    std::cout << "Concepts: " << ltm.get_all_concept_ids().size()
              << " Relations: " << ltm.total_relation_count() << "\n";

    // Property inheritance
    PropertyInheritance prop(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9; cfg.trust_floor = 0.3; cfg.max_iterations = 50;
    cfg.max_hop_depth = 20;
    cfg.propagate_requires = true; cfg.propagate_uses = true; cfg.propagate_produces = true;
    prop.propagate(cfg);
    std::cout << "After PI: " << ltm.total_relation_count() << " relations\n\n";

    for (const auto& query : {"photosynthesis", "gravity", "water", "evolution", "electricity"}) {
        std::cout << "--- " << query << " ---\n";
        auto ids = ltm.find_by_label(query);
        if (ids.empty()) {
            std::cout << "  NOT FOUND in label index!\n";
            continue;
        }
        for (auto cid : ids) {
            auto cinfo = ltm.retrieve_concept(cid);
            std::cout << "  cid=" << cid << " label=" << (cinfo ? cinfo->label : "?") << "\n";
            auto rels = ltm.get_outgoing_relations(cid);
            std::cout << "  outgoing: " << rels.size() << "\n";
            for (size_t i = 0; i < std::min(rels.size(), (size_t)3); ++i) {
                auto tinfo = ltm.retrieve_concept(rels[i].target);
                std::cout << "    -> " << (tinfo ? tinfo->label : "?")
                          << " (type=" << (int)rels[i].type << ")\n";
            }
        }
        std::cout << "\n";
    }
    return 0;
}

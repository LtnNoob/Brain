// Quick inference debug: loads KB + engine, calls encode_concept_h and checks
#include "ltm/long_term_memory.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "micromodel/concept_embedding_store.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "language/kan_language_engine.hpp"
#include "language/language_training.hpp"
#include "language/language_config.hpp"

#include <iostream>

using namespace brain19;

int main() {
    // Load KB
    LongTermMemory ltm;
    FoundationConcepts::seed_from_file(ltm, "../data/foundation_full.json");
    std::cerr << "Concepts: " << ltm.get_all_concept_ids().size()
              << " Relations: " << ltm.total_relation_count() << "\n";

    // Property inheritance
    PropertyInheritance prop(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9; cfg.trust_floor = 0.3; cfg.max_iterations = 50;
    cfg.max_hop_depth = 20;
    cfg.propagate_requires = true; cfg.propagate_uses = true; cfg.propagate_produces = true;
    prop.propagate(cfg);
    std::cerr << "After PI: " << ltm.total_relation_count() << " relations\n";

    // Embeddings
    EmbeddingManager embeddings;
    embeddings.train_embeddings(ltm, 0.05, 10);

    // ConceptModels
    ConceptModelRegistry registry;
    registry.ensure_models_for(ltm);
    ConceptTrainer trainer;
    trainer.train_all(registry, embeddings, ltm);

    // Engine
    LanguageConfig lang_config;
    KANLanguageEngine engine(lang_config, ltm, registry, embeddings);
    engine.initialize();
    engine.rebuild_dimensional_context();

    // LanguageTraining
    LanguageTraining lang_trainer(engine, ltm, registry);

    // Test concepts
    for (const auto& query : {"Photosynthesis", "Gravity", "Water", "Evolution", "Electricity"}) {
        std::cerr << "\n--- Query: " << query << " ---\n";

        auto seeds = engine.find_seeds(query);
        if (seeds.empty()) {
            std::cerr << "  No seeds found!\n";
            continue;
        }
        auto primary = seeds[0];
        std::cerr << "  find_seeds returned cid=" << primary << "\n";

        // Check concept info
        auto cinfo = ltm.retrieve_concept(primary);
        std::cerr << "  label=" << (cinfo ? cinfo->label : "null") << "\n";

        // Check outgoing relations directly
        auto rels = ltm.get_outgoing_relations(primary);
        std::cerr << "  outgoing_relations: " << rels.size() << "\n";
        for (size_t i = 0; i < std::min(rels.size(), (size_t)5); ++i) {
            auto tinfo = ltm.retrieve_concept(rels[i].target);
            std::cerr << "    [" << i << "] type=" << (int)rels[i].type
                      << " target=" << (tinfo ? tinfo->label : "?") << "\n";
        }

        // Now call encode_concept_h (will trigger the debug log we added)
        auto h = lang_trainer.encode_concept_h(primary);
        std::cerr << "  h.size()=" << h.size() << "\n";

        // Check B3
        double b3_sum = 0;
        for (size_t i = 80; i < 90 && i < h.size(); ++i)
            b3_sum += std::abs(h[i]);
        std::cerr << "  B3 mean_abs=" << b3_sum / 10 << "\n";
    }

    return 0;
}

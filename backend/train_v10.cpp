// =============================================================================
// Brain19 V10 Training Run
// =============================================================================
//
// 1. Load foundation KB (foundation_full.json)
// 2. PropertyInheritance::propagate() over entire KB
// 3. Train embeddings from graph (learn_from_graph)
// 4. Language Training with V9 settings (unified concepts, cosine LR, 150 epochs)
//
// Log: /tmp/brain19_v10.log
// Goal: decoder loss < 2.0 (V9 best: 2.177 at epoch 80)
//

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
#include <fstream>
#include <chrono>
#include <string>

using namespace brain19;

static std::ofstream g_log;

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::string line = std::string("[") + ts + "] " + msg;
    std::cout << line << "\n";
    if (g_log.is_open()) {
        g_log << line << "\n";
        g_log.flush();
    }
}

int main() {
    g_log.open("/tmp/brain19_v10.log");
    log("=== Brain19 V10 Training Run ===");
    log("");

    auto t0 = std::chrono::steady_clock::now();

    // ── Step 1: Load Foundation KB ──────────────────────────────────────────
    log("[1/6] Loading foundation KB...");
    LongTermMemory ltm;

    bool loaded = false;
    for (const auto& path : {"../data/foundation_full.json", "data/foundation_full.json",
                              "../data/foundation.json", "data/foundation.json"}) {
        if (FoundationConcepts::seed_from_file(ltm, path)) {
            log("  Loaded from: " + std::string(path));
            loaded = true;
            break;
        }
    }
    if (!loaded) {
        log("  FALLBACK: using hardcoded seeds");
        FoundationConcepts::seed_all(ltm);
    }

    size_t initial_concepts = ltm.get_all_concept_ids().size();
    size_t initial_relations = ltm.total_relation_count();
    log("  Concepts: " + std::to_string(initial_concepts));
    log("  Relations: " + std::to_string(initial_relations));

    // ── Step 2: Property Inheritance ────────────────────────────────────────
    log("");
    log("[2/6] Running PropertyInheritance::propagate()...");
    {
        PropertyInheritance prop(ltm);
        PropertyInheritance::Config cfg;
        cfg.decay_per_hop = 0.9;
        cfg.trust_floor = 0.3;
        cfg.max_iterations = 50;
        cfg.max_hop_depth = 20;
        cfg.propagate_requires = true;
        cfg.propagate_uses = true;
        cfg.propagate_produces = true;
        auto result = prop.propagate(cfg);

        log("  Iterations: " + std::to_string(result.iterations_run)
            + (result.converged ? " (converged)" : " (max reached)"));
        log("  Properties inherited: " + std::to_string(result.properties_inherited));
        log("  Contradictions blocked: " + std::to_string(result.contradictions_blocked));
        log("  Trust floor cutoffs: " + std::to_string(result.trust_floor_cutoffs));
        log("  Concepts processed: " + std::to_string(result.concepts_processed));

        size_t new_relations = ltm.total_relation_count();
        log("  Relations before: " + std::to_string(initial_relations)
            + " -> after: " + std::to_string(new_relations)
            + " (+" + std::to_string(new_relations - initial_relations) + ")");

        for (const auto& [type, count] : result.type_distribution) {
            log("    " + type + ": " + std::to_string(count));
        }
    }

    // ── Step 3: Train Embeddings from Graph ─────────────────────────────────
    log("");
    log("[3/6] Training embeddings from graph (learn_from_graph)...");
    EmbeddingManager embeddings;
    {
        auto t_emb = std::chrono::steady_clock::now();
        auto emb_result = embeddings.train_embeddings(ltm, 0.05, 10);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_emb).count();
        log("  Iterations: " + std::to_string(emb_result.iterations));
        log("  Concepts updated: " + std::to_string(emb_result.concepts_updated));
        log("  Total neighbors: " + std::to_string(emb_result.total_neighbors));
        log("  Time: " + std::to_string(elapsed) + "ms");
    }

    // ── Step 4: Train ConceptModels ─────────────────────────────────────────
    log("");
    log("[4/6] Training ConceptModels from KG...");
    ConceptModelRegistry registry;
    {
        registry.ensure_models_for(ltm);
        log("  Models created: " + std::to_string(registry.size()));

        ConceptTrainer trainer;
        auto t_cm = std::chrono::steady_clock::now();
        auto stats = trainer.train_all(registry, embeddings, ltm);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_cm).count();
        log("  Trained: " + std::to_string(stats.models_trained)
            + " models (" + std::to_string(stats.models_converged)
            + " converged, avg loss " + std::to_string(stats.avg_final_loss) + ")");
        log("  Time: " + std::to_string(elapsed) + "ms");
    }

    // ── Step 5: Init KAN Language Engine ────────────────────────────────────
    log("");
    log("[5/6] Initializing KAN Language Engine...");
    LanguageConfig lang_config;
    KANLanguageEngine engine(lang_config, ltm, registry, embeddings);
    engine.initialize();
    if (!engine.is_ready()) {
        log("  ERROR: Language engine not ready!");
        return 1;
    }
    log("  Tokenizer: " + std::to_string(engine.tokenizer().vocab_size()) + " tokens");
    engine.rebuild_dimensional_context();
    log("  Dimensional context rebuilt");

    // ── Step 6: Language Training (V9 settings + V10 data) ──────────────────
    log("");
    log("[6/6] Language Training V10...");
    log("  Settings: unified concepts, cosine LR decay, 150 epochs");
    log("  V9 baseline: best loss 2.177 @ epoch 80");
    log("  V10 advantage: " + std::to_string(ltm.total_relation_count() - initial_relations)
        + " inherited properties + 16D detail embeddings + graph-trained embeddings");
    log("");

    {
        LanguageTraining lang_trainer(engine, ltm, registry);
        lang_config.encoder_epochs = 0;   // skip encoder (not needed for decoder quality)
        lang_config.decoder_epochs = 150;
        lang_config.decoder_lr = 2.0;

        auto t_lang = std::chrono::steady_clock::now();
        auto lang_result = lang_trainer.train_stage1(lang_config);
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t_lang).count();

        log("");
        log("=== V10 Training Complete ===");
        log("  Epochs: " + std::to_string(lang_result.epochs_run));
        log("  Final loss: " + std::to_string(lang_result.final_loss));
        log("  Converged: " + std::string(lang_result.converged ? "yes" : "no"));
        log("  Training time: " + std::to_string(elapsed) + "s");
        log("  Target: < 2.0 (V9 best: 2.177)");
        log("  Result: " + std::string(lang_result.final_loss < 2.0 ? "TARGET HIT!" : "target missed"));
    }

    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0).count();
    log("");
    log("Total wall time: " + std::to_string(total_elapsed) + "s");
    log("Log saved to: /tmp/brain19_v10.log");

    return 0;
}

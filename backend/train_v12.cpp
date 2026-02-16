// =============================================================================
// Brain19 V12 Training Run — Deep KAN Decoder
// =============================================================================
//
// Same pipeline as V11 but replaces the shallow transform+quadratic+linear
// decoder with a 2-layer EfficientKAN feature extractor (90→256→128).
//
// Architecture: EfficientKAN(90→256, G=8, k=3) → LN → EfficientKAN(256→128, G=5, k=3) → LN → Linear(128→VA)
// Parameters: ~585K (vs ~20K in V11)
// Target: loss < 1.5 (V11 plateau: 1.85)
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
    g_log.open("/tmp/brain19_v12.log");
    log("=== Brain19 V12 Training Run (Deep KAN) ===");
    log("");

    auto t0 = std::chrono::steady_clock::now();

    // ── Step 1: Load Foundation KB ──
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

    // ── Step 2: Property Inheritance ──
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
        auto pi_result = prop.propagate(cfg);

        log("  Iterations: " + std::to_string(pi_result.iterations_run)
            + (pi_result.converged ? " (converged)" : " (max reached)"));
        log("  Properties inherited: " + std::to_string(pi_result.properties_inherited));

        size_t new_relations = ltm.total_relation_count();
        log("  Relations before: " + std::to_string(initial_relations)
            + " -> after: " + std::to_string(new_relations)
            + " (+" + std::to_string(new_relations - initial_relations) + ")");
    }

    // ── Step 3: Train Embeddings ──
    log("");
    log("[3/6] Training embeddings from graph...");
    EmbeddingManager embeddings;
    {
        auto t_emb = std::chrono::steady_clock::now();
        auto emb_result = embeddings.train_embeddings(ltm, 0.05, 10);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_emb).count();
        log("  Iterations: " + std::to_string(emb_result.iterations));
        log("  Time: " + std::to_string(elapsed) + "ms");
    }

    // ── Step 4: Train ConceptModels ──
    log("");
    log("[4/6] Training ConceptModels...");
    ConceptModelRegistry registry;
    {
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto t_cm = std::chrono::steady_clock::now();
        auto stats = trainer.train_all(registry, embeddings, ltm);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_cm).count();
        log("  Trained: " + std::to_string(stats.models_trained)
            + " models (" + std::to_string(stats.models_converged) + " converged)");
        log("  Time: " + std::to_string(elapsed) + "ms");
    }

    // ── Step 5: Init KAN Language Engine ──
    log("");
    log("[5/6] Initializing KAN Language Engine...");
    LanguageConfig lang_config;
    KANLanguageEngine engine(lang_config, ltm, registry, embeddings);
    engine.initialize();
    if (!engine.is_ready()) {
        std::cerr << "[FATAL] Language engine not ready after initialize()\n";
        return 1;
    }
    log("  Tokenizer: " + std::to_string(engine.tokenizer().vocab_size()) + " tokens");
    engine.rebuild_dimensional_context();
    log("  Dimensional context rebuilt");

    // ── Step 6: V12 GPU Training (per-sample SGD, no mini-batch normalization) ──
    log("");
    log("[6/6] Language Training V12 (GPU per-sample SGD)...");
    log("  Architecture: [h,h²] 180D → W_a → VA (+ transform after warmup)");
    log("  V11 fix: batch=1 per-sample SGD (matches CPU online SGD)");
    log("  V11 baseline: best loss 1.848 @ epoch 105 (CPU)");
    log("  Previous GPU: loss 2.66 @ 150 epochs (batch=256, too slow convergence)");
    log("  V12 target: < 1.5 (match/beat CPU convergence on GPU)");
    log("");

    LanguageTraining lang_trainer(engine, ltm, registry);
    {
        lang_config.encoder_epochs = 0;
        lang_config.decoder_epochs = 200;    // more epochs, early stopping prevents overfitting
        lang_config.decoder_lr = 0.002;      // W_a lr (v2 Adam)
        lang_config.deep_kan_lr = 0.001;     // KAN lr (v2 Adam)
        lang_config.use_deep_kan = true;

        auto t_lang = std::chrono::steady_clock::now();
#ifdef USE_LIBTORCH
        auto lang_result = lang_trainer.train_stage1_deep_kan_v2(lang_config);
#else
        auto lang_result = lang_trainer.train_stage1(lang_config);
#endif
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t_lang).count();

        log("");
        log("=== V12 Training Complete ===");
        log("  Epochs: " + std::to_string(lang_result.epochs_run));
        log("  Final loss: " + std::to_string(lang_result.final_loss));
        log("  Converged: " + std::string(lang_result.converged ? "yes" : "no"));
        log("  Training time: " + std::to_string(elapsed) + "s");
        log("  Target: < 1.5 (beat V11 plateau)");
        log("  Result: " + std::string(lang_result.final_loss < 1.5 ? "TARGET HIT!" : "training in progress"));
    }

    // ── Step 7: Inference Test ──
    if (lang_trainer.has_v2_model()) {
        log("");
        log("[7/7] Inference Test (DeepKAN v2)...");
        for (const auto& query : {"Photosynthesis", "Gravity", "Water", "Evolution", "Electricity"}) {
            auto text = lang_trainer.generate_v2(query, 30);
            log("  " + std::string(query) + ": " + text);
        }
    }

    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0).count();
    log("");
    log("Total wall time: " + std::to_string(total_elapsed) + "s");
    log("Log saved to: /tmp/brain19_v12.log");

    return 0;
}

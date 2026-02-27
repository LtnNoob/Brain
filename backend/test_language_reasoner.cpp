// Test: KANLanguageEngine with ConceptReasoner integration
// Tests the full pipeline: query → encode → seeds → reasoner chains → scoring → fusion → decode
#include "ltm/long_term_memory.hpp"
#include "ltm/relation.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "micromodel/concept_embedding_store.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "language/kan_language_engine.hpp"
#include "language/language_config.hpp"
#include "reasoning/concept_reasoner.hpp"

#include <iostream>
#include <chrono>
#include <string>
#include <cmath>
#include <iomanip>

using namespace brain19;

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::cout << "[" << ts << "] " << msg << "\n";
}

static void print_reasoning_chain(const ReasoningChain& chain, const LongTermMemory& ltm) {
    if (chain.empty()) {
        std::cout << "    (empty chain)\n";
        return;
    }
    for (size_t i = 0; i < chain.steps.size(); ++i) {
        const auto& s = chain.steps[i];
        auto cinfo = ltm.retrieve_concept(s.concept_id);
        std::string label = cinfo ? cinfo->label : ("#" + std::to_string(s.concept_id));

        if (i == 0) {
            std::cout << "    " << label << " [seed]";
        } else {
            std::string dir = s.is_outgoing ? "--" : "<-";
            std::string conf = std::to_string(s.confidence).substr(0, 4);
            std::string coh = std::to_string(s.coherence_score).substr(0, 4);
            std::string sim = std::to_string(s.seed_similarity).substr(0, 4);
            std::string focus = s.focus_shifted ? " [F]" : "";
            std::cout << "    " << dir << relation_type_to_string(s.relation_type)
                      << "(" << conf << ", coh=" << coh << ", sim=" << sim << ")" << dir << "> "
                      << label << focus;
        }
        std::cout << "\n";
    }
}

static void run_language_test(const std::string& query,
                               KANLanguageEngine& engine,
                               const LongTermMemory& ltm) {
    auto t0 = std::chrono::steady_clock::now();
    auto result = engine.generate(query);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    std::cout << "\n";
    log("Query: \"" + query + "\" (" + std::to_string(ms) + "ms)");

    // Seeds
    std::string seed_str = "  Seeds: ";
    for (auto cid : result.activated_concepts) {
        auto info = ltm.retrieve_concept(cid);
        seed_str += (info ? info->label : std::to_string(cid)) + ", ";
    }
    log(seed_str);

    // Reasoning chain (from ConceptReasoner)
    if (!result.reasoning_chain.empty()) {
        log("  Reasoning chain (" + std::to_string(result.reasoning_chain.steps.size())
            + " steps, avg_conf=" + std::to_string(result.reasoning_chain.avg_confidence).substr(0,4)
            + "):");
        print_reasoning_chain(result.reasoning_chain, ltm);
    } else {
        log("  Reasoning chain: (none)");
    }

    // Causal chain (concept IDs that go into fusion)
    std::string chain_str = "  Causal chain: ";
    for (auto cid : result.causal_chain) {
        auto info = ltm.retrieve_concept(cid);
        chain_str += (info ? info->label : std::to_string(cid)) + " → ";
    }
    if (!result.causal_chain.empty()) chain_str.erase(chain_str.size() - 4);
    log(chain_str);

    // Convergence state norm (should be non-zero now)
    if (!result.reasoning_chain.empty()) {
        const auto& last_state = result.reasoning_chain.steps.back().chain_state;
        double norm = 0.0;
        for (double v : last_state) norm += v * v;
        norm = std::sqrt(norm);
        log("  Convergence state |h|=" + std::to_string(norm).substr(0, 5)
            + " (feeds into decoder 32D slot)");
    }

    // Output
    log("  Confidence: " + std::to_string(result.confidence).substr(0, 5)
        + " | Template: " + (result.used_template ? "YES" : "concept-predicted")
        + " | Tokens: " + std::to_string(result.tokens_generated));
    log("  Output:");
    // Print text with indent
    std::string line;
    for (char c : result.text) {
        if (c == '\n') {
            std::cout << "    " << line << "\n";
            line.clear();
        } else {
            line += c;
        }
    }
    if (!line.empty()) std::cout << "    " << line << "\n";
}

int main() {
    log("=== KANLanguageEngine + ConceptReasoner Integration Test ===");
    log("");

    // ── Load KB ──
    log("[1/5] Loading foundation KB...");
    LongTermMemory ltm;
    bool loaded = false;
    for (const auto& path : {"../data/foundation_full.json", "data/foundation_full.json",
                              "../data/foundation.json", "data/foundation.json"}) {
        if (FoundationConcepts::seed_from_file(ltm, path, true)) {
            log("  Loaded from: " + std::string(path));
            loaded = true;
            break;
        }
    }
    if (!loaded) {
        log("  FALLBACK: using hardcoded seeds");
        FoundationConcepts::seed_all(ltm);
    }
    log("  Concepts: " + std::to_string(ltm.get_all_concept_ids().size()));
    log("  Relations: " + std::to_string(ltm.total_relation_count()));

    // ── Property Inheritance ──
    log("");
    log("[2/5] PropertyInheritance...");
    {
        PropertyInheritance prop(ltm);
        PropertyInheritance::Config cfg;
        cfg.max_iterations = 50;
        cfg.max_hop_depth = 20;
        auto r = prop.propagate(cfg);
        log("  Inherited: " + std::to_string(r.properties_inherited));
        log("  Relations now: " + std::to_string(ltm.total_relation_count()));
    }

    // ── Train Embeddings + ConceptModels ──
    log("");
    log("[3/5] Training embeddings & ConceptModels...");
    EmbeddingManager embeddings;
    ConceptModelRegistry registry;
    {
        auto t0 = std::chrono::steady_clock::now();
        ConceptEmbeddingStore::LearnConfig emb_config;
        emb_config.alpha = 0.1;
        emb_config.iterations = 15;
        emb_config.negative_samples = 5;
        emb_config.negative_alpha_ratio = 0.3;
        embeddings.train_embeddings(ltm, emb_config);
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto stats = trainer.train_all(registry, embeddings, ltm);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        log("  Models: " + std::to_string(stats.models_trained)
            + " (" + std::to_string(stats.models_converged) + " converged)"
            + " in " + std::to_string(ms) + "ms");
    }

    // ── Initialize Language Engine FIRST (creates its own ConceptReasoner) ──
    log("");
    log("[4/5] Initializing KANLanguageEngine...");
    LanguageConfig lang_config;
    KANLanguageEngine engine(lang_config, ltm, registry, embeddings);
    {
        auto t0 = std::chrono::steady_clock::now();
        engine.initialize();
        engine.rebuild_dimensional_context();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        log("  Engine initialized in " + std::to_string(ms) + "ms");
    }

    // ── EM Training on the ENGINE'S OWN ConceptReasoner ──
    // This trains both ChainKAN + ConvergencePorts on the same reasoner the engine uses
    log("");
    log("[5/5] EM training on engine's ConceptReasoner (3 rounds)...");
    {
        auto* reasoner = engine.reasoner();
        if (!reasoner) {
            log("  ERROR: engine has no reasoner!");
        } else {
            auto all_ids = ltm.get_all_concept_ids();
            size_t stride = std::max(size_t(1), all_ids.size() / 80);

            for (size_t round = 0; round < 3; ++round) {
                auto t0 = std::chrono::steady_clock::now();
                std::vector<ReasoningChain> chains;

                size_t offset = round * 7;
                for (size_t i = offset; i < all_ids.size(); i += stride) {
                    auto chain = reasoner->reason_from(all_ids[i]);
                    if (chain.steps.size() >= 3)
                        chains.push_back(std::move(chain));
                }

                ChainTrainingConfig tcfg;
                tcfg.learning_rate = 0.01 / (1.0 + round * 0.3);
                tcfg.kan_epochs = 100;
                tcfg.convergence_epochs = 10;
                tcfg.convergence_lr = 0.001 / (1.0 + round * 0.3);

                auto result = reasoner->train_composition(chains, tcfg);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0).count();
                log("  Round " + std::to_string(round + 1) + "/3: "
                    + std::to_string(chains.size()) + " chains, "
                    + std::to_string(result.samples_collected) + " samples, KAN "
                    + std::to_string(result.initial_kan_loss).substr(0,6)
                    + "→" + std::to_string(result.final_kan_loss).substr(0,6)
                    + ", " + std::to_string(result.convergence_ports_updated) + " ports"
                    + " (" + std::to_string(ms) + "ms)");
            }
        }
    }

    // ── Test queries ──
    log("");
    log("=== Language Generation Tests ===");

    const std::vector<std::string> queries = {
        "What is Photosynthesis?",
        "How does Gravity work?",
        "Tell me about Water",
        "What is Evolution?",
        "How does Electricity work?",
        "What is DNA?",
        "Explain Philosophy",
        "What is an Algorithm?",
        "How does the brain work?",
        "What causes Climate Change?"
    };

    for (const auto& query : queries) {
        run_language_test(query, engine, ltm);
    }

    log("");
    log("=== Done ===");
    return 0;
}

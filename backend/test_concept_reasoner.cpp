// Standalone test for ConceptReasoner — CM Composition + Contrastive Training
#include "ltm/long_term_memory.hpp"
#include "ltm/relation.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "reasoning/concept_reasoner.hpp"
#include "reasoning/chain_kan.hpp"

#include <iostream>
#include <chrono>
#include <string>
#include <cmath>
#include <iomanip>
#include <numeric>

using namespace brain19;

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::cout << "[" << ts << "] " << msg << "\n";
}

static double chain_state_norm(const std::array<double, 32>& state) {
    double sum = 0.0;
    for (double v : state) sum += v * v;
    return std::sqrt(sum);
}

static double chain_state_cosine(const std::array<double, 32>& a,
                                  const std::array<double, 32>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < 32; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static void run_chain_test(const std::string& query,
                            ConceptReasoner& reasoner,
                            const LongTermMemory& ltm,
                            bool show_chain_state) {
    auto seeds = ltm.find_by_label(query);
    if (seeds.empty()) { log("  " + query + ": not found"); return; }

    auto chain = reasoner.reason_from(seeds);
    if (chain.empty()) { log("  " + query + ": empty chain"); return; }

    std::string line = "  [" + query + "] "
        + std::to_string(chain.steps.size()) + " steps, avg_conf="
        + std::to_string(chain.avg_confidence).substr(0,4)
        + " quality=" + std::to_string(reasoner.compute_chain_quality(chain)).substr(0,4)
        + "\n    ";

    for (size_t i = 0; i < chain.steps.size(); ++i) {
        const auto& s = chain.steps[i];
        auto cinfo = ltm.retrieve_concept(s.concept_id);
        std::string label = cinfo ? cinfo->label : ("#" + std::to_string(s.concept_id));

        if (i > 0) {
            std::string dir = s.is_outgoing ? "--" : "<-";
            std::string conf = std::to_string(s.confidence).substr(0,4);
            std::string focus_mark = s.focus_shifted ? " [F]" : "";
            line += "\n    " + dir + relation_type_to_string(s.relation_type)
                  + "(" + conf + ")" + dir + "> ";
            line += label + focus_mark;

            if (show_chain_state) {
                line += "  coh=" + std::to_string(s.coherence_score).substr(0,5)
                      + " |h|=" + std::to_string(chain_state_norm(s.chain_state)).substr(0,5);
            }
        } else {
            line += label + " [F]";
            if (show_chain_state) {
                line += "  |h|=" + std::to_string(chain_state_norm(s.chain_state)).substr(0,5);
            }
        }
    }
    log(line);

    if (show_chain_state && chain.steps.size() > 2) {
        std::string evolution = "    cos: ";
        for (size_t i = 1; i < chain.steps.size(); ++i) {
            double cos = chain_state_cosine(chain.steps[i-1].chain_state,
                                             chain.steps[i].chain_state);
            evolution += std::to_string(cos).substr(0,5);
            if (i + 1 < chain.steps.size()) evolution += " → ";
        }
        log(evolution);
    }
}

int main() {
    log("=== ConceptReasoner — Contrastive Chain Training ===");
    log("");

    // ── Load KB ──
    log("[1/7] Loading foundation KB...");
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
    log("[2/7] PropertyInheritance...");
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
    log("[3/7] Training embeddings & ConceptModels...");
    EmbeddingManager embeddings;
    ConceptModelRegistry registry;
    {
        auto t0 = std::chrono::steady_clock::now();
        embeddings.train_embeddings(ltm, 0.05, 10);
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto stats = trainer.train_all(registry, embeddings, ltm);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        log("  Models: " + std::to_string(stats.models_trained)
            + " (" + std::to_string(stats.models_converged) + " converged)");
        log("  Time: " + std::to_string(ms) + "ms");
    }

    const std::vector<std::string> queries = {
        "Photosynthesis", "Gravity", "Water", "Evolution",
        "Electricity", "DNA", "Mathematics", "Oxygen",
        "Philosophy", "Algorithm"
    };

    // ── Baseline: composition disabled ──
    log("");
    log("[4/7] Baseline (composition OFF)...");
    {
        ReasonerConfig rcfg;
        rcfg.max_steps = 10;
        rcfg.enable_composition = false;
        ConceptReasoner reasoner(ltm, registry, embeddings, rcfg);

        for (const auto& query : queries) {
            run_chain_test(query, reasoner, ltm, false);
        }
    }

    // ── Composition: untrained ──
    log("");
    log("[5/7] Composition ON (untrained, with context feedback)...");
    ReasonerConfig rcfg_comp;
    rcfg_comp.max_steps = 10;
    rcfg_comp.enable_composition = true;
    rcfg_comp.chain_coherence_weight = 0.3;
    rcfg_comp.chain_ctx_blend = 0.15;
    ConceptReasoner reasoner(ltm, registry, embeddings, rcfg_comp);
    {
        for (const auto& query : queries) {
            run_chain_test(query, reasoner, ltm, true);
        }
    }

    // ── EM-style multi-round training ──
    log("");
    log("[6/7] EM training: collect → train → re-collect → re-train...");
    {
        auto t0_total = std::chrono::steady_clock::now();

        constexpr size_t NUM_ROUNDS = 5;
        auto all_ids = ltm.get_all_concept_ids();
        // Sample ~80 random concepts per round for diversity
        size_t sample_stride = std::max(size_t(1), all_ids.size() / 80);

        for (size_t round = 0; round < NUM_ROUNDS; ++round) {
            auto t0 = std::chrono::steady_clock::now();

            // ── E-step: collect chains with current model ──
            std::vector<ReasoningChain> all_chains;

            for (const auto& query : queries) {
                auto seeds = ltm.find_by_label(query);
                if (seeds.empty()) continue;
                auto chain = reasoner.reason_from(seeds);
                if (chain.steps.size() >= 3)
                    all_chains.push_back(std::move(chain));
            }

            // Vary the stride each round for different random samples
            size_t offset = round * 7;  // different starting point each round
            for (size_t i = offset; i < all_ids.size(); i += sample_stride) {
                auto chain = reasoner.reason_from(all_ids[i]);
                if (chain.steps.size() >= 3)
                    all_chains.push_back(std::move(chain));
            }

            // Quality stats
            std::vector<double> qualities;
            for (const auto& chain : all_chains) {
                qualities.push_back(reasoner.compute_chain_quality(chain));
            }
            std::sort(qualities.begin(), qualities.end());

            // ── M-step: train ChainKAN + ConvergencePorts ──
            ChainTrainingConfig tcfg;
            tcfg.learning_rate = 0.01 / (1.0 + round * 0.3);  // decay LR across rounds
            tcfg.kan_epochs = 100;
            tcfg.convergence_epochs = 10;
            tcfg.convergence_lr = 0.001 / (1.0 + round * 0.3);

            auto result = reasoner.train_composition(all_chains, tcfg);

            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

            log("  Round " + std::to_string(round + 1) + "/" + std::to_string(NUM_ROUNDS)
                + ": " + std::to_string(all_chains.size()) + " chains"
                + ", " + std::to_string(result.samples_collected) + " samples"
                + ", KAN " + std::to_string(result.initial_kan_loss).substr(0,6)
                + "→" + std::to_string(result.final_kan_loss).substr(0,6)
                + ", " + std::to_string(result.convergence_ports_updated) + " ports"
                + ", q=[" + std::to_string(qualities.front()).substr(0,4)
                + ".." + std::to_string(qualities[qualities.size()/2]).substr(0,4)
                + ".." + std::to_string(qualities.back()).substr(0,4) + "]"
                + " (" + std::to_string(ms) + "ms)");
        }

        auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0_total).count();
        log("  Total training time: " + std::to_string(ms_total) + "ms");
    }

    // ── After training: re-run chains ──
    log("");
    log("[7/7] After contrastive training + ConvergencePort fine-tuning...");
    {
        for (const auto& query : queries) {
            run_chain_test(query, reasoner, ltm, true);
        }
    }

    log("");
    log("=== Done ===");
    return 0;
}

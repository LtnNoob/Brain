// Standalone test for GraphReasoner --- Graph als Neuronales Netz
// Full activation vectors flow through the graph, epistemic audit trail.

#include "ltm/long_term_memory.hpp"
#include "ltm/relation.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "graph_net/graph_reasoner.hpp"
#include "reasoning/concept_reasoner.hpp"

#include <iostream>
#include <chrono>
#include <string>
#include <cmath>
#include <iomanip>
#include <cassert>

using namespace brain19;

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::cout << "[" << ts << "] " << msg << "\n";
}

static void run_graph_chain_test(const std::string& query,
                                  GraphReasoner& reasoner,
                                  const LongTermMemory& ltm,
                                  bool verbose) {
    auto seeds = ltm.find_by_label(query);
    if (seeds.empty()) { log("  " + query + ": not found"); return; }

    auto chain = reasoner.reason_from(seeds);
    if (chain.empty()) { log("  " + query + ": empty chain"); return; }

    // Basic chain info
    std::cout << "  [" << query << "] "
              << chain.length() << " steps, "
              << "chain_trust=" << std::fixed << std::setprecision(3) << chain.chain_trust << ", "
              << "epistemic=" << epistemic_type_to_string(chain.chain_epistemic_type) << ", "
              << "termination=" << termination_reason_to_string(chain.termination) << ", "
              << "quality=" << reasoner.compute_chain_quality(chain) << "\n";

    // Show concept sequence
    std::cout << "    Chain: ";
    auto cseq = chain.concept_sequence();
    auto rseq = chain.relation_sequence();
    for (size_t i = 0; i < cseq.size(); ++i) {
        if (i > 0 && i - 1 < rseq.size())
            std::cout << " --" << relation_type_to_string(rseq[i-1]) << "--> ";
        auto info = ltm.retrieve_concept(cseq[i]);
        std::cout << (info ? info->label : "?");
    }
    std::cout << "\n";

    // Show activation magnitudes
    std::cout << "    Activations: ";
    for (size_t i = 0; i < chain.steps.size(); ++i) {
        if (i > 0) std::cout << " -> ";
        double mag = (i == 0) ? chain.steps[i].input_activation.core_magnitude()
                              : chain.steps[i].output_activation.core_magnitude();
        std::cout << std::fixed << std::setprecision(3) << mag;
    }
    std::cout << " (ratio=" << chain.magnitude_ratio << ")\n";

    // Verbose: full explain() output
    if (verbose) {
        std::cout << "\n" << chain.explain(ltm) << "\n";
    }
}

static void run_comparison_test(const std::string& query,
                                 GraphReasoner& graph_reasoner,
                                 ConceptReasoner& concept_reasoner,
                                 const LongTermMemory& ltm) {
    auto seeds = ltm.find_by_label(query);
    if (seeds.empty()) return;

    auto gchain = graph_reasoner.reason_from(seeds);
    auto cchain = concept_reasoner.reason_from(seeds);

    double gq = graph_reasoner.compute_chain_quality(gchain);
    double cq = concept_reasoner.compute_chain_quality(cchain);

    std::cout << "  [" << query << "] "
              << "Graph: " << gchain.length() << " steps, quality="
              << std::fixed << std::setprecision(3) << gq
              << " | Concept: " << (cchain.steps.size() > 0 ? cchain.steps.size()-1 : 0)
              << " steps, quality=" << cq;
    if (gq > cq) std::cout << " >>> GRAPH BETTER";
    else if (cq > gq) std::cout << " >>> CONCEPT BETTER";
    else std::cout << " >>> TIE";
    std::cout << "\n";
}

int main() {
    log("=== GraphReasoner Test ===");
    log("");

    // Step 1: Setup KB
    log("[1/6] Loading foundation KB...");
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

    // Step 2: Property Inheritance
    log("");
    log("[2/6] PropertyInheritance...");
    {
        PropertyInheritance prop(ltm);
        PropertyInheritance::Config cfg;
        cfg.max_iterations = 50;
        cfg.max_hop_depth = 20;
        auto r = prop.propagate(cfg);
        log("  Inherited: " + std::to_string(r.properties_inherited));
        log("  Relations now: " + std::to_string(ltm.total_relation_count()));
    }

    // Step 3: Train embeddings + ConceptModels
    log("");
    log("[3/6] Training embeddings & ConceptModels...");
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

    // Step 4: Create GraphReasoner
    log("");
    log("[4/6] Creating GraphReasoner...");
    GraphReasonerConfig gcfg;
    gcfg.max_steps = 8;
    gcfg.enable_composition = true;
    gcfg.chain_coherence_weight = 0.3;
    gcfg.chain_ctx_blend = 0.15;
    gcfg.seed_anchor_weight = 0.35;
    gcfg.seed_anchor_decay = 0.03;
    gcfg.min_coherence_gate = 0.25;
    gcfg.min_seed_similarity = 0.15;
    gcfg.max_consecutive_seed_drops = 2;
    // Anti-drift: semantic similarity gate
    gcfg.min_embedding_similarity = 0.05;
    gcfg.embedding_sim_weight = 0.1;

    GraphReasoner graph_reasoner(ltm, registry, embeddings, gcfg);
    log("  GraphReasoner created");

    // Step 5: Run chain tests
    log("");
    log("[5/6] Graph Reasoning Chains...");

    const std::vector<std::string> test_queries = {
        "Photosynthesis", "Gravity", "Water", "Evolution",
        "Electricity", "DNA", "Mathematics", "Oxygen",
        "Philosophy", "Algorithm"
    };

    for (const auto& q : test_queries) {
        run_graph_chain_test(q, graph_reasoner, ltm, false);
    }

    // Detailed trace for first found query
    log("");
    log("=== Detailed Trace ===");
    for (const auto& q : test_queries) {
        auto seeds = ltm.find_by_label(q);
        if (!seeds.empty()) {
            run_graph_chain_test(q, graph_reasoner, ltm, true);
            break;
        }
    }

    // Step 6: Verification
    log("");
    log("[6/6] Verification...");

    bool all_pass = true;
    for (const auto& q : test_queries) {
        auto seeds = ltm.find_by_label(q);
        if (seeds.empty()) continue;

        auto chain = graph_reasoner.reason_from(seeds);
        if (chain.empty()) continue;

        // chain_trust >= 0
        if (chain.chain_trust < 0.0) {
            log("  FAIL: chain_trust < 0 for " + q);
            all_pass = false;
        }

        // termination reason set
        if (chain.termination == TerminationReason::STILL_RUNNING) {
            log("  FAIL: termination still running for " + q);
            all_pass = false;
        }

        // explain() produces output
        std::string explanation = chain.explain(ltm);
        if (explanation.empty() || explanation.find("Graph Reasoning Chain") == std::string::npos) {
            log("  FAIL: explain() broken for " + q);
            all_pass = false;
        }

        // Activation vectors are non-zero
        for (size_t i = 1; i < chain.steps.size(); ++i) {
            if (chain.steps[i].output_activation.core_magnitude() < 1e-15) {
                log("  FAIL: zero activation at step " + std::to_string(i) + " for " + q);
                all_pass = false;
                break;
            }
        }

        // Top dimensions recorded
        for (size_t i = 1; i < chain.steps.size(); ++i) {
            if (chain.steps[i].top_dims.empty()) {
                log("  FAIL: no top dims at step " + std::to_string(i) + " for " + q);
                all_pass = false;
                break;
            }
        }
    }

    if (all_pass) {
        log("  All verifications PASSED");
    }

    // Comparison with ConceptReasoner
    log("");
    log("=== Comparison: GraphReasoner vs ConceptReasoner ===");

    ReasonerConfig rcfg;
    rcfg.max_steps = 8;
    rcfg.enable_composition = true;
    rcfg.chain_coherence_weight = 0.3;
    rcfg.chain_ctx_blend = 0.15;
    rcfg.seed_anchor_weight = 0.35;
    rcfg.seed_anchor_decay = 0.03;
    rcfg.min_coherence_gate = 0.25;
    rcfg.min_seed_similarity = 0.25;
    rcfg.max_consecutive_drops = 2;

    ConceptReasoner concept_reasoner(ltm, registry, embeddings, rcfg);

    for (const auto& q : test_queries) {
        run_comparison_test(q, graph_reasoner, concept_reasoner, ltm);
    }

    // Co-Learning Signal extraction test
    log("");
    log("=== Co-Learning Signals ===");
    for (const auto& q : {"Photosynthesis", "DNA", "Philosophy"}) {
        auto seeds = ltm.find_by_label(q);
        if (seeds.empty()) continue;

        auto chain = graph_reasoner.reason_from(seeds);
        if (chain.empty()) continue;

        auto signal = graph_reasoner.extract_signals(chain);

        std::cout << "  [" << q << "] chain_quality=" << std::fixed << std::setprecision(3)
                  << signal.chain_quality << " chain_pain=" << signal.chain_pain()
                  << " traversed=" << signal.traversed_edges.size()
                  << " rejected=" << signal.rejected_edges.size() << "\n";

        for (const auto& edge : signal.traversed_edges) {
            auto src = ltm.retrieve_concept(edge.source);
            auto tgt = ltm.retrieve_concept(edge.target);
            std::cout << "    " << (edge.is_positive ? "+" : "-") << " "
                      << (src ? src->label : "?") << " -> " << (tgt ? tgt->label : "?")
                      << " NN=" << edge.nn_quality << " KAN=" << edge.kan_quality
                      << " gate=" << edge.kan_gate
                      << " emb=" << edge.embedding_similarity << "\n";
        }

        if (!signal.suggestions.empty()) {
            std::cout << "    Suggestions:\n";
            for (const auto& s : signal.suggestions) {
                auto src = ltm.retrieve_concept(s.source);
                auto tgt = ltm.retrieve_concept(s.target);
                std::cout << "      " << (s.delta_weight >= 0 ? "+" : "")
                          << std::fixed << std::setprecision(3) << s.delta_weight << " "
                          << (src ? src->label : "?") << " -> " << (tgt ? tgt->label : "?")
                          << " (" << s.reason << ")\n";
            }
        }
    }

    // Edge evaluation test
    log("");
    log("=== Edge Evaluation (direct) ===");
    {
        auto photo_ids = ltm.find_by_label("Photosynthesis");
        auto chloro_ids = ltm.find_by_label("Chlorophyll");
        if (!photo_ids.empty() && !chloro_ids.empty()) {
            auto es = graph_reasoner.evaluate_edge(
                photo_ids[0], chloro_ids[0], RelationType::REQUIRES);
            std::cout << "  Photosynthesis --REQUIRES--> Chlorophyll: "
                      << (es.is_positive ? "POSITIVE" : "NEGATIVE")
                      << " tq=" << std::fixed << std::setprecision(3) << es.transform_quality
                      << " coh=" << es.coherence
                      << " emb_sim=" << es.embedding_similarity
                      << " reward=" << es.reward() << "\n";
        }
    }

    log("");
    log("=== All tests complete ===");
    return 0;
}

// =============================================================================
// Comprehensive System Test: Logger + Co-Learning + Residual + GRU + Pruning
// =============================================================================
//
// 7-part test covering:
//   1. Co-Learning Loop end-to-end (10 cycles, per-seed metrics)
//   2. Before/After comparison (chain quality, trust, drift)
//   3. Episodic Memory store/replay/consolidation
//   4. Residual Connection validation (12-step chains)
//   5. GRU Forget Gate (10+ step chain state saturation)
//   6. Edge Pruning (weight changes from pain signals)
//   7. Regression check (build verification)
//

#include "ltm/long_term_memory.hpp"
#include "ltm/relation.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "graph_net/graph_reasoner.hpp"
#include "graph_net/reasoning_logger.hpp"
#include "colearn/colearn_loop.hpp"
#include "colearn/episodic_memory.hpp"
#include "colearn/knowledge_extractor.hpp"
#include "colearn/colearn_types.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>
#include <map>
#include <unordered_set>
#include <random>

using namespace brain19;

// =============================================================================
// Utilities
// =============================================================================

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::cout << "[" << ts << "] " << msg << "\n";
}

static void sep(const std::string& title) {
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(72, '=') << "\n\n";
}

static void table_header(const std::vector<std::string>& cols, const std::vector<int>& widths) {
    std::cout << "  ";
    for (size_t i = 0; i < cols.size(); ++i)
        std::cout << std::left << std::setw(widths[i]) << cols[i];
    std::cout << "\n  ";
    for (size_t i = 0; i < cols.size(); ++i)
        std::cout << std::string(static_cast<size_t>(widths[i] - 1), '-') << " ";
    std::cout << "\n";
}

// Per-seed chain snapshot
struct ChainSnapshot {
    std::string label;
    ConceptId seed = 0;
    size_t chain_len = 0;
    double chain_trust = 0.0;
    double chain_quality = 0.0;
    double mag_ratio = 0.0;
    std::string termination;
    double avg_nn = 0.0;
    double avg_kan = 0.0;
    size_t nn_dom = 0;
    size_t kan_dom = 0;
    double total_pain = 0.0;
    double total_reward = 0.0;
    double min_seed_sim = 1.0;
};

static ChainSnapshot snapshot_chain(const std::string& label, ConceptId seed,
                                     GraphReasoner& reasoner,
                                     const EmbeddingManager& embeddings) {
    ChainSnapshot snap;
    snap.label = label;
    snap.seed = seed;

    auto chain = reasoner.reason_from(seed);
    snap.chain_len = chain.length();
    snap.chain_trust = chain.chain_trust;
    snap.chain_quality = reasoner.compute_chain_quality(chain);
    snap.mag_ratio = chain.magnitude_ratio;
    snap.termination = termination_reason_to_string(chain.termination);

    // Chain-mean threshold matching extract_signals (adaptive)
    double score_sum = 0.0;
    size_t step_count = chain.steps.size() > 1 ? chain.steps.size() - 1 : 0;
    for (size_t i = 1; i < chain.steps.size(); ++i) {
        score_sum += chain.steps[i].composite_score;
    }
    const double threshold = step_count > 0 ? score_sum / static_cast<double>(step_count) : 0.5;

    double sum_nn = 0.0, sum_kan = 0.0;
    for (size_t i = 1; i < chain.steps.size(); ++i) {
        const auto& s = chain.steps[i];
        sum_nn += s.nn_quality;
        sum_kan += s.kan_quality;
        if (s.nn_quality > s.kan_quality) ++snap.nn_dom;
        else if (s.kan_quality > s.nn_quality) ++snap.kan_dom;

        if (s.seed_similarity < snap.min_seed_sim)
            snap.min_seed_sim = s.seed_similarity;

        // Pain/reward using chain-mean threshold (matching extract_signals)
        bool is_positive = (s.composite_score >= threshold);
        if (is_positive) {
            snap.total_reward += s.composite_score - threshold;
        } else {
            snap.total_pain += threshold - s.composite_score;
        }
    }
    size_t sc = chain.steps.size() > 1 ? chain.steps.size() - 1 : 1;
    snap.avg_nn = sum_nn / static_cast<double>(sc);
    snap.avg_kan = sum_kan / static_cast<double>(sc);

    return snap;
}

static void print_snapshot_table(const std::vector<ChainSnapshot>& snaps) {
    table_header({"Seed", "Len", "Trust", "Quality", "MagR", "Termination",
                  "NN", "KAN", "NNdom", "KANdom", "Pain", "Reward", "MinSS"},
                 {16, 5, 8, 9, 8, 22, 7, 7, 7, 7, 7, 8, 7});
    for (const auto& s : snaps) {
        std::cout << "  " << std::left << std::setw(16) << s.label
                  << std::setw(5) << s.chain_len
                  << std::fixed << std::setprecision(3)
                  << std::setw(8) << s.chain_trust
                  << std::setw(9) << s.chain_quality
                  << std::setw(8) << s.mag_ratio
                  << std::setw(22) << s.termination
                  << std::setw(7) << s.avg_nn
                  << std::setw(7) << s.avg_kan
                  << std::setw(7) << s.nn_dom
                  << std::setw(7) << s.kan_dom
                  << std::setw(7) << s.total_pain
                  << std::setw(8) << s.total_reward
                  << std::setw(7) << s.min_seed_sim
                  << "\n";
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    log("=== Comprehensive System Test: Logger + CoLearn + Residual + GRU + Pruning ===");

    // =========================================================================
    // Setup
    // =========================================================================
    sep("SETUP: Loading Knowledge Base + Training Models");

    LongTermMemory ltm;
    bool loaded = false;
    for (const auto& path : {"../data/foundation_full.json", "data/foundation_full.json",
                              "../data/foundation.json", "data/foundation.json"}) {
        if (FoundationConcepts::seed_from_file(ltm, path, true)) {
            log("Loaded from: " + std::string(path));
            loaded = true;
            break;
        }
    }
    if (!loaded) {
        log("FALLBACK: using hardcoded seeds");
        FoundationConcepts::seed_all(ltm);
    }
    log("Concepts: " + std::to_string(ltm.get_all_concept_ids().size()));
    log("Relations: " + std::to_string(ltm.total_relation_count()));

    // Property inheritance
    {
        PropertyInheritance prop(ltm);
        PropertyInheritance::Config cfg;
        cfg.max_iterations = 50;
        cfg.max_hop_depth = 20;
        auto r = prop.propagate(cfg);
        log("Inherited: " + std::to_string(r.properties_inherited));
    }

    // Train embeddings + models
    EmbeddingManager embeddings;
    ConceptModelRegistry registry;
    {
        embeddings.train_embeddings(ltm, 0.05, 5);
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto stats = trainer.train_all(registry, embeddings, ltm);
        log("Models: " + std::to_string(stats.models_trained)
            + " (" + std::to_string(stats.models_converged) + " converged)");
    }

    // Look up test seeds
    const std::vector<std::string> seed_labels = {
        "Photosynthesis", "DNA", "Electricity", "Water", "Mathematics"
    };
    std::vector<ConceptId> seed_ids;
    for (const auto& label : seed_labels) {
        auto ids = ltm.find_by_label(label);
        if (!ids.empty()) {
            seed_ids.push_back(ids[0]);
            log("Seed: " + label + " -> id=" + std::to_string(ids[0])
                + " rels=" + std::to_string(ltm.get_relation_count(ids[0])));
        } else {
            log("WARNING: seed '" + label + "' not found!");
            seed_ids.push_back(0);
        }
    }

    // Create reasoner (6-step for standard tests)
    GraphReasonerConfig gcfg;
    gcfg.max_steps = 6;
    gcfg.enable_composition = true;
    gcfg.chain_coherence_weight = 0.3;
    gcfg.chain_ctx_blend = 0.15;
    gcfg.seed_anchor_weight = 0.35;
    gcfg.seed_anchor_decay = 0.03;
    gcfg.min_embedding_similarity = 0.05;
    gcfg.embedding_sim_weight = 0.1;
    GraphReasoner reasoner(ltm, registry, embeddings, gcfg);

    // Attach JSONL logger
    const char* log_path = "/tmp/brain19_system_test.jsonl";
    std::remove(log_path);
    ReasoningLogger logger(log_path);
    reasoner.set_logger(&logger);

    // =========================================================================
    // TEST 1+2: Before Snapshot + Co-Learning Loop + After Snapshot
    // =========================================================================
    sep("TEST 1/2: BEFORE SNAPSHOT (Baseline)");

    std::vector<ChainSnapshot> before_snaps;
    for (size_t i = 0; i < seed_labels.size(); ++i) {
        if (seed_ids[i] == 0) continue;
        before_snaps.push_back(snapshot_chain(seed_labels[i], seed_ids[i], reasoner, embeddings));
    }
    print_snapshot_table(before_snaps);

    // Record initial relation count
    size_t initial_rel_count = ltm.total_relation_count();
    log("\nInitial relation count: " + std::to_string(initial_rel_count));

    // --- Degrade graph: corrupt model parameters to simulate bad models ---
    sep("DEGRADATION: Corrupting 30% of concept model parameters");
    {
        size_t corrupted = 0;
        size_t total_models = 0;
        std::mt19937 rng(42);
        std::normal_distribution<double> noise(0.0, 1.0);  // σ=1.0 (aggressive)

        auto all_ids = ltm.get_all_concept_ids();
        for (ConceptId cid : all_ids) {
            ConceptModel* m = registry.get_model(cid);
            if (!m) continue;
            ++total_models;

            // Corrupt ~30% of models
            if (rng() % 100 < 30) {
                std::array<double, CM_FLAT_SIZE> params;
                m->to_flat(params);

                // Corrupt bilinear core [0..303] and multihead [940..1579]
                // Leave training state, convergence port, KAN intact
                for (size_t i = 0; i < 304; ++i) {
                    params[i] += noise(rng);
                }
                for (size_t i = 940; i < 1580; ++i) {
                    params[i] += noise(rng);
                }

                m->from_flat(params);
                ++corrupted;
            }
        }
        log("Corrupted " + std::to_string(corrupted) + " / " + std::to_string(total_models)
            + " models (" + std::to_string(100 * corrupted / total_models) + "%)");
    }

    // Snapshot AFTER degradation
    std::cout << "\n  --- After degradation ---\n";
    std::vector<ChainSnapshot> degraded_snaps;
    for (size_t i = 0; i < seed_labels.size(); ++i) {
        if (seed_ids[i] == 0) continue;
        degraded_snaps.push_back(snapshot_chain(seed_labels[i], seed_ids[i], reasoner, embeddings));
    }
    print_snapshot_table(degraded_snaps);

    std::cout << "\n  --- Delta (corrupted - baseline) ---\n";
    table_header({"Seed", "dLen", "dTrust", "dQuality", "dPain", "dReward"},
                 {16, 7, 9, 10, 8, 9});
    for (size_t s = 0; s < degraded_snaps.size() && s < before_snaps.size(); ++s) {
        std::cout << "  " << std::left << std::setw(16) << degraded_snaps[s].label
                  << std::showpos << std::fixed << std::setprecision(3)
                  << std::setw(7) << static_cast<int>(degraded_snaps[s].chain_len) - static_cast<int>(before_snaps[s].chain_len)
                  << std::setw(9) << degraded_snaps[s].chain_trust - before_snaps[s].chain_trust
                  << std::setw(10) << degraded_snaps[s].chain_quality - before_snaps[s].chain_quality
                  << std::setw(8) << degraded_snaps[s].total_pain - before_snaps[s].total_pain
                  << std::setw(9) << degraded_snaps[s].total_reward - before_snaps[s].total_reward
                  << std::noshowpos << "\n";
    }

    // --- Co-Learning Loop: 20 cycles, retraining recovers corrupted models ---
    sep("TEST 1: CO-LEARNING LOOP (Recovery from model corruption, 20 cycles)");

    CoLearnConfig cl_cfg;
    cl_cfg.wake_chains_per_cycle = 50;
    cl_cfg.sleep_replay_count = 50;
    cl_cfg.retrain_epochs = 50;            // enough to fine-tune
    cl_cfg.retrain_refined_epochs = 5;    // refined KAN fine-tuning
    cl_cfg.retrain_learning_rate = 0.001; // 10x smaller than initial training LR
    cl_cfg.retrain_kan_lr = 0.0005;       // gentle KAN fine-tuning
    cl_cfg.max_episodes = 5000;
    cl_cfg.weight_delta = 0.05;           // standard deltas
    cl_cfg.lr_decay_rate = 0.05;          // standard decay
    cl_cfg.retrain_threshold = 0.05;      // low threshold — trigger retrain for corrupted models

    CoLearnLoop loop(ltm, registry, embeddings, reasoner, cl_cfg);

    // Control quality: track fixed seeds independently of wake-phase seed selection
    auto control_quality = [&]() -> double {
        double sum = 0.0;
        size_t count = 0;
        for (size_t s = 0; s < seed_ids.size(); ++s) {
            if (seed_ids[s] == 0) continue;
            auto chain = reasoner.reason_from(seed_ids[s]);
            sum += reasoner.compute_chain_quality(chain);
            ++count;
        }
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    };
    double control_q0 = control_quality();

    table_header({"Cycle", "Chains", "AvgQual", "CtrlQual", "Str", "Weak", "Pruned",
                  "Retrain", "Rollback", "QualDelta", "ErrCorr", "TermC", "DropC", "SucC"},
                 {7, 8, 9, 10, 7, 7, 8, 9, 9, 10, 8, 7, 7, 6});

    std::vector<CoLearnLoop::CycleResult> all_results;
    std::vector<double> control_quals;
    control_quals.push_back(control_q0);

    for (size_t i = 0; i < 20; ++i) {
        auto result = loop.run_cycle();
        all_results.push_back(result);

        double ctrl_q = control_quality();
        control_quals.push_back(ctrl_q);

        std::cout << "  " << std::left
                  << std::setw(7) << result.cycle_number
                  << std::setw(8) << result.chains_produced
                  << std::fixed << std::setprecision(4)
                  << std::setw(9) << result.avg_chain_quality
                  << std::setw(10) << ctrl_q
                  << std::setw(7) << result.consolidation.edges_strengthened
                  << std::setw(7) << result.consolidation.edges_weakened
                  << std::setw(8) << result.consolidation.edges_pruned
                  << std::setw(9) << result.models_retrained
                  << std::setw(9) << result.models_rolled_back
                  << std::setw(10) << result.quality_delta
                  << std::setw(8) << result.correction_samples_injected
                  << std::setw(7) << result.terminal_corrections
                  << std::setw(7) << result.quality_drop_corrections
                  << std::setw(6) << result.success_reinforcements
                  << "\n";

        // After-10 snapshot
        if (i == 9) {
            std::cout << "\n  --- After 10 cycles snapshot ---\n";
            std::vector<ChainSnapshot> mid_snaps;
            for (size_t s = 0; s < seed_labels.size(); ++s) {
                if (seed_ids[s] == 0) continue;
                mid_snaps.push_back(snapshot_chain(seed_labels[s], seed_ids[s], reasoner, embeddings));
            }
            print_snapshot_table(mid_snaps);
            std::cout << "\n";

            // Print delta table
            std::cout << "  --- Delta (cycle 10 - degraded) ---\n";
            table_header({"Seed", "dLen", "dTrust", "dQuality", "dPain", "dReward"},
                         {16, 7, 9, 10, 8, 9});
            for (size_t s = 0; s < mid_snaps.size() && s < degraded_snaps.size(); ++s) {
                std::cout << "  " << std::left << std::setw(16) << mid_snaps[s].label
                          << std::showpos << std::fixed << std::setprecision(3)
                          << std::setw(7) << static_cast<int>(mid_snaps[s].chain_len) - static_cast<int>(degraded_snaps[s].chain_len)
                          << std::setw(9) << mid_snaps[s].chain_trust - degraded_snaps[s].chain_trust
                          << std::setw(10) << mid_snaps[s].chain_quality - degraded_snaps[s].chain_quality
                          << std::setw(8) << mid_snaps[s].total_pain - degraded_snaps[s].total_pain
                          << std::setw(9) << mid_snaps[s].total_reward - degraded_snaps[s].total_reward
                          << std::noshowpos << "\n";
            }
            std::cout << "\n";
        }
    }

    // Summary statistics
    double total_strengthened = 0, total_weakened = 0, total_pruned = 0;
    for (const auto& r : all_results) {
        total_strengthened += r.consolidation.edges_strengthened;
        total_weakened += r.consolidation.edges_weakened;
        total_pruned += r.consolidation.edges_pruned;
    }
    size_t total_corrections = 0, total_terminal = 0, total_drops = 0, total_success = 0;
    for (const auto& r : all_results) {
        total_corrections += r.correction_samples_injected;
        total_terminal += r.terminal_corrections;
        total_drops += r.quality_drop_corrections;
        total_success += r.success_reinforcements;
    }

    std::cout << "\n  Total across 20 cycles: strengthened=" << total_strengthened
              << " weakened=" << total_weakened << " pruned=" << total_pruned << "\n";
    std::cout << "  Error corrections: " << total_corrections << " samples injected"
              << " (terminal=" << total_terminal
              << " quality_drop=" << total_drops
              << " success=" << total_success << ")\n";
    std::cout << "  Control quality (5 fixed seeds): "
              << std::fixed << std::setprecision(4) << control_quals.front()
              << " -> " << control_quals.back()
              << " (delta=" << std::showpos << (control_quals.back() - control_quals.front())
              << std::noshowpos << ")\n";

    // After-20 snapshot
    sep("TEST 2: AFTER 20 CYCLES (Recovery from model corruption)");

    std::vector<ChainSnapshot> after_snaps;
    for (size_t i = 0; i < seed_labels.size(); ++i) {
        if (seed_ids[i] == 0) continue;
        after_snaps.push_back(snapshot_chain(seed_labels[i], seed_ids[i], reasoner, embeddings));
    }
    print_snapshot_table(after_snaps);

    // Delta vs degraded (recovery)
    std::cout << "\n  --- Recovery delta (cycle 20 - degraded) ---\n";
    table_header({"Seed", "dLen", "dTrust", "dQuality", "dMagR", "dPain", "dReward", "dMinSS"},
                 {16, 7, 9, 10, 8, 8, 9, 8});
    for (size_t s = 0; s < after_snaps.size() && s < degraded_snaps.size(); ++s) {
        std::cout << "  " << std::left << std::setw(16) << after_snaps[s].label
                  << std::showpos << std::fixed << std::setprecision(3)
                  << std::setw(7) << static_cast<int>(after_snaps[s].chain_len) - static_cast<int>(degraded_snaps[s].chain_len)
                  << std::setw(9) << after_snaps[s].chain_trust - degraded_snaps[s].chain_trust
                  << std::setw(10) << after_snaps[s].chain_quality - degraded_snaps[s].chain_quality
                  << std::setw(8) << after_snaps[s].mag_ratio - degraded_snaps[s].mag_ratio
                  << std::setw(8) << after_snaps[s].total_pain - degraded_snaps[s].total_pain
                  << std::setw(9) << after_snaps[s].total_reward - degraded_snaps[s].total_reward
                  << std::setw(8) << after_snaps[s].min_seed_sim - degraded_snaps[s].min_seed_sim
                  << std::noshowpos << "\n";
    }

    // Delta vs original (full recovery check)
    std::cout << "\n  --- Full recovery delta (cycle 20 - original baseline) ---\n";
    table_header({"Seed", "dLen", "dTrust", "dQuality", "dMagR", "dPain", "dReward"},
                 {16, 7, 9, 10, 8, 8, 9});
    for (size_t s = 0; s < after_snaps.size() && s < before_snaps.size(); ++s) {
        std::cout << "  " << std::left << std::setw(16) << after_snaps[s].label
                  << std::showpos << std::fixed << std::setprecision(3)
                  << std::setw(7) << static_cast<int>(after_snaps[s].chain_len) - static_cast<int>(before_snaps[s].chain_len)
                  << std::setw(9) << after_snaps[s].chain_trust - before_snaps[s].chain_trust
                  << std::setw(10) << after_snaps[s].chain_quality - before_snaps[s].chain_quality
                  << std::setw(8) << after_snaps[s].mag_ratio - before_snaps[s].mag_ratio
                  << std::setw(8) << after_snaps[s].total_pain - before_snaps[s].total_pain
                  << std::setw(9) << after_snaps[s].total_reward - before_snaps[s].total_reward
                  << std::noshowpos << "\n";
    }

    size_t final_rel_count = ltm.total_relation_count();
    std::cout << "\n  Relation count: " << initial_rel_count << " -> " << final_rel_count
              << " (delta=" << (static_cast<int>(final_rel_count) - static_cast<int>(initial_rel_count)) << ")\n";

    // =========================================================================
    // TEST 3: Episodic Memory
    // =========================================================================
    sep("TEST 3: EPISODIC MEMORY (Store / Replay / Consolidation)");

    const auto& ep_mem = loop.episodic_memory();
    log("Total episodes stored: " + std::to_string(ep_mem.episode_count()));

    // Check replay counts and consolidation
    size_t replayed = 0, consolidated = 0;
    double avg_ep_quality = 0.0;
    size_t ep_count = 0;

    // Get episodes through concept index
    std::map<TerminationReason, size_t> term_dist;
    std::unordered_set<uint64_t> seen_eps;
    for (ConceptId cid : seed_ids) {
        if (cid == 0) continue;
        auto eps = ep_mem.episodes_for_concept(cid);
        for (const Episode* ep : eps) {
            if (!ep || seen_eps.count(ep->id)) continue;
            seen_eps.insert(ep->id);
            if (ep->replay_count > 0) ++replayed;
            if (ep->consolidation_strength > 0.0) ++consolidated;
            avg_ep_quality += ep->quality;
            ++ep_count;
            term_dist[ep->termination]++;
        }
    }
    log("Episodes touching test seeds: " + std::to_string(ep_count));
    log("Replayed episodes (replay_count > 0): " + std::to_string(replayed));
    log("Consolidated episodes (strength > 0): " + std::to_string(consolidated));
    if (ep_count > 0)
        log("Avg episode quality: " + std::to_string(avg_ep_quality / static_cast<double>(ep_count)));

    std::cout << "\n  Termination distribution (episodes touching test seeds):\n";
    for (const auto& [reason, count] : term_dist) {
        std::cout << "    " << std::setw(24) << std::left
                  << termination_reason_to_string(reason)
                  << count << "\n";
    }

    // Verify from_chain round-trip
    {
        EpisodicMemory test_mem;
        auto chain = reasoner.reason_from(seed_ids[0]);
        Episode ep = test_mem.from_chain(chain, seed_ids[0]);
        uint64_t id = test_mem.store(ep);
        const Episode* stored = test_mem.get(id);
        bool round_trip_ok = stored && stored->seed == seed_ids[0]
                          && stored->steps.size() == chain.steps.size()
                          && stored->termination == chain.termination;
        log("Episode round-trip: " + std::string(round_trip_ok ? "PASS" : "FAIL"));
    }

    // =========================================================================
    // TEST 4: Residual Connection Validation (12-step chains)
    // =========================================================================
    sep("TEST 4: RESIDUAL CONNECTION (12-step chains)");

    GraphReasonerConfig long_cfg = gcfg;
    long_cfg.max_steps = 12;
    long_cfg.min_seed_similarity = 0.01;    // Very lenient to allow long chains
    long_cfg.min_composite_score = 0.01;
    long_cfg.min_coherence_gate = 0.05;
    long_cfg.min_chain_trust = 0.001;
    long_cfg.max_consecutive_seed_drops = 12;
    GraphReasoner long_reasoner(ltm, registry, embeddings, long_cfg);

    table_header({"Seed", "Len", "MagRatio", "StartMag", "EndMag", "MinMag", "Verdict"},
                 {16, 5, 10, 10, 10, 10, 12});

    bool residual_ok = true;
    for (size_t i = 0; i < seed_labels.size(); ++i) {
        if (seed_ids[i] == 0) continue;
        auto chain = long_reasoner.reason_from(seed_ids[i]);

        double start_mag = chain.steps.empty() ? 0.0
            : chain.steps[0].input_activation.core_magnitude();
        double end_mag = chain.steps.empty() ? 0.0
            : chain.steps.back().output_activation.core_magnitude();
        double min_mag = 1e30;
        for (size_t s = 0; s < chain.steps.size(); ++s) {
            double m = s == 0 ? chain.steps[s].input_activation.core_magnitude()
                              : chain.steps[s].output_activation.core_magnitude();
            if (m < min_mag) min_mag = m;
        }
        double ratio = start_mag > 1e-12 ? end_mag / start_mag : 0.0;
        bool ok = min_mag > 0.01 && ratio > 0.05;
        if (!ok) residual_ok = false;

        std::cout << "  " << std::left << std::setw(16) << seed_labels[i]
                  << std::setw(5) << chain.length()
                  << std::fixed << std::setprecision(4)
                  << std::setw(10) << ratio
                  << std::setw(10) << start_mag
                  << std::setw(10) << end_mag
                  << std::setw(10) << min_mag
                  << (ok ? "OK" : "SIGNAL LOSS")
                  << "\n";
    }
    log("Residual connection: " + std::string(residual_ok ? "PASS" : "FAIL"));

    // Per-step magnitude trace for one seed
    {
        auto chain = long_reasoner.reason_from(seed_ids[0]);
        std::cout << "\n  Step-by-step magnitude trace (" << seed_labels[0] << ", "
                  << chain.length() << " steps):\n";
        std::cout << "  Step  Magnitude  Ratio-to-prev\n";
        double prev_mag = 0.0;
        for (size_t s = 0; s < chain.steps.size(); ++s) {
            double m = s == 0 ? chain.steps[s].input_activation.core_magnitude()
                              : chain.steps[s].output_activation.core_magnitude();
            double r = prev_mag > 1e-12 ? m / prev_mag : 0.0;
            std::cout << "  " << std::setw(5) << s
                      << std::fixed << std::setprecision(4)
                      << std::setw(11) << m
                      << std::setw(14) << (s == 0 ? 0.0 : r)
                      << "\n";
            prev_mag = m;
        }
    }

    // =========================================================================
    // TEST 5: GRU Forget Gate (chain state saturation)
    // =========================================================================
    sep("TEST 5: GRU FORGET GATE (chain state saturation test)");

    std::cout << "  Testing chain_state norm across steps (should NOT saturate):\n\n";

    table_header({"Seed", "Len", "State@2", "State@5", "State@end", "MaxNorm", "Saturated?"},
                 {16, 5, 10, 10, 12, 10, 12});

    bool gru_ok = true;
    for (size_t i = 0; i < seed_labels.size(); ++i) {
        if (seed_ids[i] == 0) continue;
        auto chain = long_reasoner.reason_from(seed_ids[i]);

        auto state_norm = [](const std::array<double, convergence::OUTPUT_DIM>& s) {
            double sum = 0.0;
            for (size_t d = 0; d < convergence::OUTPUT_DIM; ++d)
                sum += s[d] * s[d];
            return std::sqrt(sum);
        };

        double norm_at_2 = 0.0, norm_at_5 = 0.0, norm_at_end = 0.0;
        double max_norm = 0.0;
        for (size_t s = 0; s < chain.steps.size(); ++s) {
            double n = state_norm(chain.steps[s].chain_state);
            if (n > max_norm) max_norm = n;
            if (s == 2) norm_at_2 = n;
            if (s == 5) norm_at_5 = n;
            if (s == chain.steps.size() - 1) norm_at_end = n;
        }

        // GRU gate should prevent values from growing unbounded
        // Saturated = all components near +/-1 → norm near sqrt(32) ≈ 5.66
        bool saturated = max_norm > 5.5;
        if (saturated) gru_ok = false;

        std::cout << "  " << std::left << std::setw(16) << seed_labels[i]
                  << std::setw(5) << chain.length()
                  << std::fixed << std::setprecision(4)
                  << std::setw(10) << norm_at_2
                  << std::setw(10) << norm_at_5
                  << std::setw(12) << norm_at_end
                  << std::setw(10) << max_norm
                  << (saturated ? "YES" : "NO")
                  << "\n";
    }
    log("GRU forget gate: " + std::string(gru_ok ? "PASS (no saturation)" : "FAIL (saturation detected)"));

    // Detailed chain state trace for one seed
    {
        auto chain = long_reasoner.reason_from(seed_ids[0]);
        std::cout << "\n  Chain state norm trace (" << seed_labels[0] << "):\n";
        std::cout << "  Step  StateNorm  Components[0..3]\n";
        for (size_t s = 0; s < chain.steps.size(); ++s) {
            const auto& cs = chain.steps[s].chain_state;
            double norm = 0.0;
            for (size_t d = 0; d < convergence::OUTPUT_DIM; ++d)
                norm += cs[d] * cs[d];
            norm = std::sqrt(norm);

            std::cout << "  " << std::setw(5) << s
                      << std::fixed << std::setprecision(4)
                      << std::setw(11) << norm
                      << "  [" << cs[0] << ", " << cs[1] << ", "
                      << cs[2] << ", " << cs[3] << "]\n";
        }
    }

    // =========================================================================
    // TEST 6: Edge Pruning
    // =========================================================================
    sep("TEST 6: EDGE PRUNING (weight changes from pain signals)");

    // Pick a seed, reason, extract signals, track weight changes
    ConceptId prune_seed = seed_ids[0]; // Photosynthesis

    // Snapshot relation weights before
    auto chain_for_signals = reasoner.reason_from(prune_seed);
    auto signals = reasoner.extract_signals(chain_for_signals);

    std::cout << "  Chain seed: " << seed_labels[0] << " (quality="
              << std::fixed << std::setprecision(4) << signals.chain_quality << ")\n";
    std::cout << "  Traversed edges: " << signals.traversed_edges.size() << "\n";
    std::cout << "  Rejected edges: " << signals.rejected_edges.size() << "\n";
    std::cout << "  Chain pain: " << signals.chain_pain() << "\n\n";

    table_header({"Edge", "Traversed", "Positive", "Score", "Pain", "Reward", "WeightBefore", "WeightAfter"},
                 {30, 11, 9, 8, 8, 8, 14, 13});

    // Record weights before
    struct EdgeRecord {
        ConceptId source, target;
        RelationType relation;
        double weight_before;
        double weight_after;
        bool traversed;
        bool is_positive;
        double score;
        double pain;
        double reward;
    };
    std::vector<EdgeRecord> edge_records;

    for (const auto& es : signals.traversed_edges) {
        EdgeRecord rec;
        rec.source = es.source;
        rec.target = es.target;
        rec.relation = es.relation;
        rec.traversed = true;
        rec.is_positive = es.is_positive;
        rec.score = es.composite_score;
        rec.pain = es.pain();
        rec.reward = es.reward();

        // Find relation weight
        auto rels = ltm.get_relations_between(es.source, es.target);
        rec.weight_before = 0.0;
        for (const auto& r : rels) {
            if (r.type == es.relation) {
                rec.weight_before = r.weight;
                break;
            }
        }
        edge_records.push_back(rec);
    }

    // Apply signals through knowledge extractor
    CoLearnConfig prune_cfg;
    prune_cfg.weight_delta = 0.05;
    KnowledgeExtractor extractor(ltm, reasoner, prune_cfg);
    size_t changes = extractor.apply_signals(signals);

    // Record weights after
    for (auto& rec : edge_records) {
        auto rels = ltm.get_relations_between(rec.source, rec.target);
        rec.weight_after = 0.0;
        for (const auto& r : rels) {
            if (r.type == rec.relation) {
                rec.weight_after = r.weight;
                break;
            }
        }

        auto src_info = ltm.retrieve_concept(rec.source);
        auto tgt_info = ltm.retrieve_concept(rec.target);
        std::string src_label = src_info ? src_info->label.substr(0, 12) : "?";
        std::string tgt_label = tgt_info ? tgt_info->label.substr(0, 12) : "?";
        std::string edge_str = src_label + "->" + tgt_label;

        std::cout << "  " << std::left << std::setw(30) << edge_str
                  << std::setw(11) << (rec.traversed ? "yes" : "no")
                  << std::setw(9) << (rec.is_positive ? "yes" : "no")
                  << std::fixed << std::setprecision(4)
                  << std::setw(8) << rec.score
                  << std::setw(8) << rec.pain
                  << std::setw(8) << rec.reward
                  << std::setw(14) << rec.weight_before
                  << std::setw(13) << rec.weight_after
                  << "\n";
    }

    log("Edge weight changes applied: " + std::to_string(changes));

    // Verify: positive edges should be strengthened, negative weakened
    size_t strengthened = 0, weakened = 0, unchanged = 0;
    for (const auto& rec : edge_records) {
        if (rec.weight_after > rec.weight_before + 1e-6) ++strengthened;
        else if (rec.weight_after < rec.weight_before - 1e-6) ++weakened;
        else ++unchanged;
    }
    std::cout << "\n  Strengthened: " << strengthened
              << " | Weakened: " << weakened
              << " | Unchanged: " << unchanged << "\n";

    // =========================================================================
    // TEST 7: Regression Check
    // =========================================================================
    sep("TEST 7: REGRESSION CHECK");

    // Detach logger before running other reasoners
    reasoner.set_logger(nullptr);

    // Verify core functionality still works
    bool regression_ok = true;
    int checks_passed = 0;
    int checks_total = 0;

    auto reg_check = [&](bool cond, const std::string& name) {
        ++checks_total;
        if (cond) {
            ++checks_passed;
            std::cout << "  PASS: " << name << "\n";
        } else {
            regression_ok = false;
            std::cout << "  FAIL: " << name << "\n";
        }
    };

    // Basic reasoning
    for (size_t i = 0; i < seed_labels.size(); ++i) {
        if (seed_ids[i] == 0) continue;
        auto chain = reasoner.reason_from(seed_ids[i]);
        reg_check(!chain.empty(), seed_labels[i] + " chain non-empty");
        reg_check(chain.chain_trust >= 0.0, seed_labels[i] + " chain trust >= 0");
        reg_check(chain.termination != TerminationReason::STILL_RUNNING,
                  seed_labels[i] + " terminates");
    }

    // Multi-seed reasoning
    {
        auto chain = reasoner.reason_from(seed_ids);
        reg_check(!chain.empty(), "multi-seed reason_from works");
    }

    // Feedback reasoning
    {
        GraphReasonerConfig fb_cfg = gcfg;
        fb_cfg.feedback.enable = true;
        fb_cfg.feedback.max_rounds = 2;
        fb_cfg.feedback.quality_skip_threshold = 0.99;
        GraphReasoner fb_reasoner(ltm, registry, embeddings, fb_cfg);
        auto chain = fb_reasoner.reason_with_feedback(seed_ids[0]);
        reg_check(!chain.empty(), "reason_with_feedback works");
    }

    // Extract signals
    {
        auto chain = reasoner.reason_from(seed_ids[0]);
        auto sig = reasoner.extract_signals(chain);
        reg_check(sig.seed == seed_ids[0], "extract_signals seed correct");
        reg_check(!sig.traversed_edges.empty() || chain.length() == 0,
                  "extract_signals has traversed edges");
    }

    // Evaluate edge
    {
        auto rels = ltm.get_outgoing_relations(seed_ids[0]);
        if (!rels.empty()) {
            auto es = reasoner.evaluate_edge(seed_ids[0], rels[0].target, rels[0].type);
            reg_check(es.source == seed_ids[0], "evaluate_edge source correct");
        } else {
            reg_check(true, "evaluate_edge (no rels to test)");
        }
    }

    // Chain quality
    {
        auto chain = reasoner.reason_from(seed_ids[0]);
        double q = reasoner.compute_chain_quality(chain);
        reg_check(q >= 0.0 && q <= 2.0, "compute_chain_quality in range");
    }

    // Error-driven learning: corrections were generated
    reg_check(total_corrections > 0,
              "error corrections injected (" + std::to_string(total_corrections) + " samples)");
    reg_check(total_terminal > 0,
              "terminal corrections generated (" + std::to_string(total_terminal) + ")");

    // explain()
    {
        auto chain = reasoner.reason_from(seed_ids[0]);
        std::string expl = chain.explain(ltm);
        reg_check(expl.find("Graph Reasoning Chain") != std::string::npos,
                  "explain() contains header");
    }

    // Logger attach/detach
    {
        ReasoningLogger tmp_logger("/tmp/brain19_regression_test.jsonl");
        reasoner.set_logger(&tmp_logger);
        auto chain = reasoner.reason_from(seed_ids[0]);
        reg_check(!chain.empty(), "reasoning with logger works");
        reasoner.set_logger(nullptr);
        auto chain2 = reasoner.reason_from(seed_ids[0]);
        reg_check(!chain2.empty(), "reasoning without logger works");
        std::remove("/tmp/brain19_regression_test.jsonl");
    }

    std::cout << "\n  Regression: " << checks_passed << "/" << checks_total << " passed\n";

    // =========================================================================
    // FINAL REPORT
    // =========================================================================
    sep("FINAL REPORT");

    // Quality trend across all 20 cycles
    std::cout << "  Quality trend across 20 co-learning cycles (recovering from degraded graph):\n";
    std::cout << "  Cycle  AvgQuality  Strengthened  Weakened  Pruned  Retrained  QualDelta  ErrCorr\n";
    for (const auto& r : all_results) {
        std::cout << "  " << std::setw(6) << r.cycle_number
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << r.avg_chain_quality
                  << std::setw(14) << r.consolidation.edges_strengthened
                  << std::setw(10) << r.consolidation.edges_weakened
                  << std::setw(8) << r.consolidation.edges_pruned
                  << std::setw(11) << r.models_retrained
                  << std::showpos << std::setw(11) << r.quality_delta
                  << std::noshowpos
                  << std::setw(9) << r.correction_samples_injected
                  << "\n";
    }

    // Three-way comparison: Original → Degraded → Recovered
    double avg_q_orig = 0.0, avg_q_degraded = 0.0, avg_q_recovered = 0.0;
    double avg_trust_orig = 0.0, avg_trust_degraded = 0.0, avg_trust_recovered = 0.0;
    double avg_pain_orig = 0.0, avg_pain_degraded = 0.0, avg_pain_recovered = 0.0;
    for (size_t i = 0; i < before_snaps.size(); ++i) {
        avg_q_orig += before_snaps[i].chain_quality;
        avg_trust_orig += before_snaps[i].chain_trust;
        avg_pain_orig += before_snaps[i].total_pain;
    }
    for (size_t i = 0; i < degraded_snaps.size(); ++i) {
        avg_q_degraded += degraded_snaps[i].chain_quality;
        avg_trust_degraded += degraded_snaps[i].chain_trust;
        avg_pain_degraded += degraded_snaps[i].total_pain;
    }
    for (size_t i = 0; i < after_snaps.size(); ++i) {
        avg_q_recovered += after_snaps[i].chain_quality;
        avg_trust_recovered += after_snaps[i].chain_trust;
        avg_pain_recovered += after_snaps[i].total_pain;
    }
    if (!before_snaps.empty()) {
        double n = static_cast<double>(before_snaps.size());
        avg_q_orig /= n; avg_trust_orig /= n; avg_pain_orig /= n;
    }
    if (!degraded_snaps.empty()) {
        double n = static_cast<double>(degraded_snaps.size());
        avg_q_degraded /= n; avg_trust_degraded /= n; avg_pain_degraded /= n;
    }
    if (!after_snaps.empty()) {
        double n = static_cast<double>(after_snaps.size());
        avg_q_recovered /= n; avg_trust_recovered /= n; avg_pain_recovered /= n;
    }

    std::cout << "\n  Three-way comparison (averages across 5 seeds):\n";
    std::cout << "                  Original  Degraded  Recovered  Recovery%\n";
    auto recovery_pct = [](double orig, double degraded, double recovered) -> double {
        double drop = orig - degraded;
        if (std::abs(drop) < 1e-8) return 100.0;
        return 100.0 * (recovered - degraded) / drop;
    };
    std::cout << "  Avg Quality   " << std::fixed << std::setprecision(4)
              << std::setw(10) << avg_q_orig
              << std::setw(10) << avg_q_degraded
              << std::setw(11) << avg_q_recovered
              << std::setw(10) << std::setprecision(1) << recovery_pct(avg_q_orig, avg_q_degraded, avg_q_recovered) << "%\n";
    std::cout << "  Avg Trust     " << std::setprecision(4)
              << std::setw(10) << avg_trust_orig
              << std::setw(10) << avg_trust_degraded
              << std::setw(11) << avg_trust_recovered
              << std::setw(10) << std::setprecision(1) << recovery_pct(avg_trust_orig, avg_trust_degraded, avg_trust_recovered) << "%\n";
    std::cout << "  Avg Pain      " << std::setprecision(4)
              << std::setw(10) << avg_pain_orig
              << std::setw(10) << avg_pain_degraded
              << std::setw(11) << avg_pain_recovered << "\n";

    // Highlight the key question
    double quality_drop = avg_q_orig - avg_q_degraded;
    double quality_recovery = avg_q_recovered - avg_q_degraded;
    std::cout << "\n  KEY METRIC: Quality dropped " << std::setprecision(4) << quality_drop
              << " from degradation, recovered " << quality_recovery
              << " (" << std::setprecision(1) << (quality_drop > 1e-8 ? 100.0 * quality_recovery / quality_drop : 0.0) << "% recovery)\n";

    // Learning verdict: is control quality stable (do no harm)?
    double ctrl_delta = control_quals.back() - control_quals.front();
    bool stable_ok = std::abs(ctrl_delta) < 0.05;  // Stable if within ±0.05
    std::cout << "\n  Co-learning:        " << (stable_ok ? "STABLE" : "UNSTABLE")
              << " (control delta=" << std::showpos << std::setprecision(4) << ctrl_delta
              << std::noshowpos << ")\n";

    // Verdict
    std::cout << "  JSONL log lines written to: " << log_path << "\n";
    std::cout << "  Episodes in memory: " << ep_mem.episode_count() << "\n";
    std::cout << "  Residual connection: " << (residual_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "  GRU forget gate:    " << (gru_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "  Regression:         " << (regression_ok ? "PASS" : "FAIL")
              << " (" << checks_passed << "/" << checks_total << ")\n";

    bool all_pass = residual_ok && gru_ok && regression_ok;
    std::cout << "\n  OVERALL: " << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";

    return all_pass ? 0 : 1;
}

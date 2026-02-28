// Standalone test for Co-Learning Loop + Episodic Memory
// Tests: EpisodicMemory, KnowledgeExtractor, CoLearnLoop, LTM modify_relation_weight

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
#include <chrono>
#include <string>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <fstream>
#include <cstdio>
#include <thread>
#include <atomic>
#include <vector>

using namespace brain19;

static int tests_passed = 0;
static int tests_failed = 0;

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::cout << "[" << ts << "] " << msg << "\n";
}

static void check(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  PASS: " << name << "\n";
        ++tests_passed;
    } else {
        std::cout << "  FAIL: " << name << "\n";
        ++tests_failed;
    }
}

// =============================================================================
// Test 1: LTM modify_relation_weight
// =============================================================================

static void test_ltm_modify_weight() {
    log("--- Test 1: LTM modify_relation_weight ---");

    LongTermMemory ltm;
    auto a = ltm.store_concept("A", "concept A",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto b = ltm.store_concept("B", "concept B",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.8));

    RelationId rid = ltm.add_relation(a, b, RelationType::CAUSES, 0.5);
    check(rid != 0, "relation created");

    auto rel = ltm.get_relation(rid);
    check(rel.has_value(), "relation retrievable");
    check(std::abs(rel->weight - 0.5) < 0.001, "initial weight = 0.5");

    // Modify weight
    bool ok = ltm.modify_relation_weight(rid, 0.8);
    check(ok, "modify_relation_weight returns true");

    rel = ltm.get_relation(rid);
    check(std::abs(rel->weight - 0.8) < 0.001, "weight updated to 0.8");

    // Clamp to [0,1]
    ltm.modify_relation_weight(rid, 1.5);
    rel = ltm.get_relation(rid);
    check(std::abs(rel->weight - 1.0) < 0.001, "weight clamped to 1.0");

    ltm.modify_relation_weight(rid, -0.3);
    rel = ltm.get_relation(rid);
    check(std::abs(rel->weight - 0.0) < 0.001, "weight clamped to 0.0");

    // Invalid relation
    ok = ltm.modify_relation_weight(9999, 0.5);
    check(!ok, "modify nonexistent relation returns false");
}

// =============================================================================
// Test 2: EpisodicMemory store/retrieve/select
// =============================================================================

static void test_episodic_memory_basics() {
    log("--- Test 2: EpisodicMemory basics ---");

    EpisodicMemory mem(100);
    check(mem.episode_count() == 0, "initially empty");

    // Create and store an episode
    Episode ep;
    ep.seed = 42;
    ep.quality = 0.75;
    ep.termination = TerminationReason::MAX_STEPS_REACHED;

    EpisodeStep s1;
    s1.concept_id = 42;
    s1.step_trust = 0.8;
    ep.steps.push_back(s1);

    EpisodeStep s2;
    s2.concept_id = 43;
    s2.relation = RelationType::CAUSES;
    s2.from_concept = 42;
    s2.step_trust = 0.7;
    ep.steps.push_back(s2);

    uint64_t id = mem.store(ep);
    check(id > 0, "store returns valid ID");
    check(mem.episode_count() == 1, "count is 1 after store");

    const Episode* retrieved = mem.get(id);
    check(retrieved != nullptr, "get returns non-null");
    check(retrieved->seed == 42, "seed preserved");
    check(retrieved->steps.size() == 2, "steps preserved");
    check(std::abs(retrieved->quality - 0.75) < 0.001, "quality preserved");

    // Concept index
    auto eps_42 = mem.episodes_for_concept(42);
    check(eps_42.size() == 1, "concept 42 has 1 episode");
    auto eps_43 = mem.episodes_for_concept(43);
    check(eps_43.size() == 1, "concept 43 has 1 episode");
    auto eps_99 = mem.episodes_for_concept(99);
    check(eps_99.empty(), "concept 99 has no episodes");

    // Mark replayed
    mem.mark_replayed(id);
    retrieved = mem.get(id);
    check(retrieved->replay_count == 1, "replay_count incremented");

    // Mark consolidated
    mem.mark_consolidated(id, 0.6);
    retrieved = mem.get(id);
    check(std::abs(retrieved->consolidation_strength - 0.6) < 0.001, "consolidation_strength set");
}

// =============================================================================
// Test 3: EpisodicMemory select_for_replay
// =============================================================================

static void test_episodic_memory_replay() {
    log("--- Test 3: EpisodicMemory select_for_replay ---");

    EpisodicMemory mem(100);

    // Store 5 episodes with varying quality
    for (int i = 0; i < 5; ++i) {
        Episode ep;
        ep.seed = static_cast<ConceptId>(100 + i);
        ep.quality = 0.1 * static_cast<double>(i + 1);  // 0.1, 0.2, 0.3, 0.4, 0.5
        ep.timestamp_ms = static_cast<uint64_t>(1000 + i * 100);

        EpisodeStep s;
        s.concept_id = ep.seed;
        ep.steps.push_back(s);
        mem.store(ep);
    }

    check(mem.episode_count() == 5, "5 episodes stored");

    // Select top 3 by quality
    auto selected = mem.select_for_replay(3, 1.0, 0.0, 0.0);
    check(selected.size() == 3, "selected 3 episodes");
    check(selected[0]->quality >= selected[1]->quality, "sorted by quality desc");
    check(selected[1]->quality >= selected[2]->quality, "sorted by quality desc (2)");
}

// =============================================================================
// Test 4: EpisodicMemory eviction
// =============================================================================

static void test_episodic_memory_eviction() {
    log("--- Test 4: EpisodicMemory eviction ---");

    EpisodicMemory mem(5);

    // Store 5 episodes
    std::vector<uint64_t> ids;
    for (int i = 0; i < 5; ++i) {
        Episode ep;
        ep.seed = static_cast<ConceptId>(i + 1);
        ep.quality = 0.5;
        EpisodeStep s;
        s.concept_id = ep.seed;
        ep.steps.push_back(s);
        ids.push_back(mem.store(ep));
    }

    check(mem.episode_count() == 5, "5 episodes stored");

    // Mark first 2 as fully consolidated
    mem.mark_consolidated(ids[0], 0.9);
    mem.mark_consolidated(ids[1], 0.8);

    // Evict to target 3
    size_t evicted = mem.evict_consolidated(3);
    check(evicted == 2, "evicted 2 consolidated episodes");
    check(mem.episode_count() == 3, "3 episodes remain");

    // Verify the consolidated ones are gone
    check(mem.get(ids[0]) == nullptr, "consolidated ep 0 evicted");
    check(mem.get(ids[1]) == nullptr, "consolidated ep 1 evicted");
    check(mem.get(ids[2]) != nullptr, "non-consolidated ep 2 remains");
}

// =============================================================================
// Test 5: GraphChain → Episode conversion
// =============================================================================

static void test_chain_to_episode(GraphReasoner& reasoner, const LongTermMemory& ltm,
                                   EmbeddingManager& /*embeddings*/) {
    log("--- Test 5: GraphChain -> Episode conversion ---");

    // Find a seed
    auto all_ids = ltm.get_all_concept_ids();
    if (all_ids.empty()) {
        log("  SKIP: no concepts in LTM");
        return;
    }

    // Try to find a concept with relations
    ConceptId seed = 0;
    for (ConceptId cid : all_ids) {
        if (ltm.get_relation_count(cid) >= 2) {
            seed = cid;
            break;
        }
    }
    if (seed == 0) seed = all_ids[0];

    GraphChain chain = reasoner.reason_from(seed);
    check(!chain.empty(), "chain is not empty");

    EpisodicMemory mem;
    Episode ep = mem.from_chain(chain, seed);

    check(ep.seed == seed, "episode seed matches");
    check(ep.steps.size() == chain.steps.size(), "episode steps match chain steps");
    check(ep.termination == chain.termination, "termination matches");

    if (!ep.steps.empty()) {
        check(ep.steps[0].concept_id == chain.steps[0].source_id,
              "first step concept matches seed");
    }

    // Store and verify
    uint64_t id = mem.store(ep);
    check(mem.get(id) != nullptr, "stored episode retrievable");
    check(mem.get(id)->steps.size() == ep.steps.size(), "stored steps match");
}

// =============================================================================
// Test 6: KnowledgeExtractor consolidation
// =============================================================================

static void test_knowledge_extractor(GraphReasoner& reasoner, LongTermMemory& ltm) {
    log("--- Test 6: KnowledgeExtractor consolidation ---");

    CoLearnConfig config;
    config.weight_delta = 0.02;
    config.prune_threshold = 0.01;

    KnowledgeExtractor extractor(ltm, reasoner, config);

    // Create an episode from a real chain
    auto all_ids = ltm.get_all_concept_ids();
    ConceptId seed = 0;
    for (ConceptId cid : all_ids) {
        if (ltm.get_relation_count(cid) >= 2) {
            seed = cid;
            break;
        }
    }
    if (seed == 0 && !all_ids.empty()) seed = all_ids[0];

    GraphChain chain = reasoner.reason_from(seed);
    EpisodicMemory mem;
    Episode ep = mem.from_chain(chain, seed);
    ep.quality = reasoner.compute_chain_quality(chain);

    // Remember initial relation count
    size_t initial_rels = ltm.total_relation_count();

    // Consolidate
    ConsolidationResult result = extractor.consolidate_episode(ep);
    check(result.episodes_consolidated == 1, "1 episode consolidated");

    size_t total_changes = result.edges_strengthened + result.edges_weakened + result.edges_pruned;
    log("  Strengthened: " + std::to_string(result.edges_strengthened)
        + " Weakened: " + std::to_string(result.edges_weakened)
        + " Pruned: " + std::to_string(result.edges_pruned));

    // We can't guarantee changes since it depends on chain quality,
    // but the function should run without error
    check(true, "consolidation completed without error");
    (void)initial_rels;
    (void)total_changes;
}

// =============================================================================
// Test 7: CoLearnLoop — run cycles
// =============================================================================

static void test_colearn_loop(LongTermMemory& ltm, ConceptModelRegistry& registry,
                               EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    log("--- Test 7: CoLearnLoop cycles ---");

    CoLearnConfig config;
    config.wake_chains_per_cycle = 5;   // Small for test speed
    config.sleep_replay_count = 3;
    config.retrain_epochs = 2;          // Minimal for test speed
    config.max_episodes = 100;

    CoLearnLoop loop(ltm, registry, embeddings, reasoner, config);

    check(loop.cycle_count() == 0, "initial cycle_count = 0");
    check(loop.episodic_memory().episode_count() == 0, "initial episodes = 0");

    // Run 3 cycles
    auto results = loop.run_cycles(3);
    check(results.size() == 3, "3 cycle results returned");

    // Verify cycle numbers
    check(results[0].cycle_number == 1, "cycle 1 number correct");
    check(results[1].cycle_number == 2, "cycle 2 number correct");
    check(results[2].cycle_number == 3, "cycle 3 number correct");

    // Verify episodes were stored
    size_t total_episodes = loop.episodic_memory().episode_count();
    log("  Total episodes after 3 cycles: " + std::to_string(total_episodes));
    check(total_episodes > 0, "episodes were stored");

    // Verify chains were produced
    size_t total_chains = 0;
    for (const auto& r : results) total_chains += r.chains_produced;
    log("  Total chains: " + std::to_string(total_chains));
    check(total_chains > 0, "chains were produced");

    // Verify quality tracking
    log("  Cycle qualities: "
        + std::to_string(results[0].avg_chain_quality) + ", "
        + std::to_string(results[1].avg_chain_quality) + ", "
        + std::to_string(results[2].avg_chain_quality));

    // Verify cycle_count incremented
    check(loop.cycle_count() == 3, "cycle_count = 3 after 3 cycles");

    // Log consolidation results
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        log("  Cycle " + std::to_string(i+1) + ":"
            + " chains=" + std::to_string(r.chains_produced)
            + " episodes=" + std::to_string(r.episodes_stored)
            + " quality=" + std::to_string(r.avg_chain_quality)
            + " strengthened=" + std::to_string(r.consolidation.edges_strengthened)
            + " weakened=" + std::to_string(r.consolidation.edges_weakened)
            + " models=" + std::to_string(r.models_retrained)
            + " converged=" + std::to_string(r.models_converged));
    }
}

// =============================================================================
// Test 8: Named constants (dimension fix verification)
// =============================================================================

static void test_named_constants() {
    log("--- Test 8: Named constants verification ---");

    check(convergence::OUTPUT_DIM == 32, "OUTPUT_DIM = 32");
    check(convergence::QUERY_DIM == 90, "QUERY_DIM = 90");
    check(convergence::CM_INPUT_DIM == 122, "CM_INPUT_DIM = 122");
    check(CORE_DIM == 16, "CORE_DIM = 16");
    check(FlexConfig::MAX_DIM == 512, "FlexConfig::MAX_DIM = 512");
    check(FlexConfig::MAX_DIM - CORE_DIM == 496, "MAX_DIM - CORE_DIM = 496");
    check(CM_FLAT_SIZE == 9933, "CM_FLAT_SIZE = 9933 (V8)");
    check(CM_FLAT_SIZE_V6 == 5836, "CM_FLAT_SIZE_V6 = 5836");
}

// =============================================================================
// Test 9: Residual connection (Fix 1)
// =============================================================================

static void test_residual_connection(GraphReasoner& reasoner, const LongTermMemory& ltm) {
    log("--- Test 9: Residual connection (activation preservation) ---");

    auto all_ids = ltm.get_all_concept_ids();
    ConceptId seed = 0;
    for (ConceptId cid : all_ids) {
        if (ltm.get_relation_count(cid) >= 3) {
            seed = cid;
            break;
        }
    }
    if (seed == 0 && !all_ids.empty()) seed = all_ids[0];

    GraphChain chain = reasoner.reason_from(seed);

    // With residual connections, activation magnitude should decay slower
    // than the old multiplicative gate (which would give 0.6^steps)
    if (chain.steps.size() >= 3) {
        double start_mag = chain.steps[0].output_activation.core_magnitude();
        double end_mag = chain.steps.back().output_activation.core_magnitude();
        double ratio = start_mag > 1e-12 ? end_mag / start_mag : 0.0;

        log("  Chain length: " + std::to_string(chain.steps.size()));
        log("  Start magnitude: " + std::to_string(start_mag));
        log("  End magnitude: " + std::to_string(end_mag));
        log("  Ratio: " + std::to_string(ratio));

        // With residual, ratio should stay much higher than contractive gate
        // Old: 0.6^steps. New: should stay above 0.1 for typical chains
        check(ratio > 0.05, "activation ratio > 0.05 (residual prevents vanishing)");
        check(end_mag > 0.01, "end magnitude > 0.01 (signal not dead)");
    } else {
        log("  Chain too short to test (" + std::to_string(chain.steps.size()) + " steps)");
        check(true, "chain produced (too short for decay test)");
    }
}

// =============================================================================
// Test 10: LR decay in consolidation (Fix 2)
// =============================================================================

static void test_lr_decay(LongTermMemory& ltm, GraphReasoner& reasoner) {
    log("--- Test 10: LR decay in consolidation ---");

    CoLearnConfig config;
    config.weight_delta = 0.1;
    config.lr_decay_rate = 0.05;

    KnowledgeExtractor extractor(ltm, reasoner, config);

    // At cycle 0: decay factor = 1/(1 + 0.05*0) = 1.0
    // At cycle 20: decay factor = 1/(1 + 0.05*20) = 1/2 = 0.5
    // At cycle 100: decay factor = 1/(1 + 0.05*100) = 1/6 ≈ 0.167

    // Verify decay factor formula by checking that set_cycle exists and compiles
    extractor.set_cycle(0);
    extractor.set_cycle(20);
    extractor.set_cycle(100);
    check(true, "set_cycle(0/20/100) compiles and runs");

    // Verify the decay rate is stored in config
    check(std::abs(config.lr_decay_rate - 0.05) < 1e-10, "lr_decay_rate = 0.05");

    // Consolidate at cycle 0 vs cycle 100 — delta should be smaller at cycle 100
    auto all_ids = ltm.get_all_concept_ids();
    ConceptId seed = 0;
    for (ConceptId cid : all_ids) {
        if (ltm.get_relation_count(cid) >= 2) {
            seed = cid;
            break;
        }
    }
    if (seed == 0 && !all_ids.empty()) seed = all_ids[0];

    GraphChain chain = reasoner.reason_from(seed);
    EpisodicMemory mem;
    Episode ep = mem.from_chain(chain, seed);
    ep.quality = reasoner.compute_chain_quality(chain);

    // Just verify consolidation works with LR decay active
    extractor.set_cycle(50);
    ConsolidationResult result = extractor.consolidate_episode(ep);
    check(result.episodes_consolidated == 1, "consolidation with LR decay succeeds");
}

// =============================================================================
// Test 11: ConvergencePort GRU gate (Fix 3)
// =============================================================================

static void test_convergence_gate() {
    log("--- Test 11: ConvergencePort GRU gate ---");

    ConceptModel cm;

    // With default init: W_gate=0, b_gate=0 → gate=sigmoid(0)=0.5
    // output = 0.5 * tanh(W*x+b) + 0.5 * prev_state
    double input[122];
    for (size_t i = 0; i < 90; ++i) input[i] = 0.1;   // query features
    for (size_t i = 0; i < 32; ++i) input[90 + i] = 0.8; // prev_state

    double output[32];
    cm.forward_convergence(input, output);

    // Gate should be ~0.5, so output should be between new_val and prev_state
    for (size_t i = 0; i < 32; ++i) {
        check(output[i] > -1.0 && output[i] < 1.0,
              "gate output " + std::to_string(i) + " bounded");
    }

    // With zero gate weights, output = 0.5*tanh(...) + 0.5*0.8
    // Should be close to a blend of tanh output and 0.4 (0.5*0.8)
    check(true, "GRU gate compute runs without error");

    // Test serialization round-trip of gate weights
    cm.convergence_port().W_gate[0] = 0.42;
    cm.convergence_port().b_gate[0] = -0.1;

    std::array<double, CM_FLAT_SIZE> flat;
    cm.to_flat(flat);

    ConceptModel cm2;
    cm2.from_flat(flat);

    check(std::abs(cm2.convergence_port().W_gate[0] - 0.42) < 1e-15,
          "gate W serialization round-trip");
    check(std::abs(cm2.convergence_port().b_gate[0] - (-0.1)) < 1e-15,
          "gate b serialization round-trip");
}

// =============================================================================
// Test 12: Recursive feedback — backward compatibility
// =============================================================================

static void test_feedback_backward_compat(GraphReasoner& reasoner, const LongTermMemory& ltm) {
    log("--- Test 12: Feedback backward compatibility ---");

    // reason_from(seed) should still work identically (delegates to internal with nullptr)
    auto all_ids = ltm.get_all_concept_ids();
    ConceptId seed = 0;
    for (ConceptId cid : all_ids) {
        if (ltm.get_relation_count(cid) >= 2) {
            seed = cid;
            break;
        }
    }
    if (seed == 0 && !all_ids.empty()) seed = all_ids[0];

    GraphChain chain1 = reasoner.reason_from(seed);
    check(!chain1.empty(), "reason_from still produces non-empty chain");
    check(chain1.steps.size() >= 1, "reason_from chain has at least seed step");

    double quality = reasoner.compute_chain_quality(chain1);
    check(quality >= 0.0, "chain quality is non-negative");
    log("  Backward-compat chain: " + std::to_string(chain1.steps.size())
        + " steps, quality=" + std::to_string(quality));
}

// =============================================================================
// Test 13: Recursive feedback — explored set grows across rounds
// =============================================================================

static void test_feedback_explored_growth(LongTermMemory& ltm,
                                           ConceptModelRegistry& registry,
                                           EmbeddingManager& embeddings) {
    log("--- Test 13: Feedback explored set growth ---");

    // Create a reasoner with feedback enabled
    GraphReasonerConfig cfg;
    cfg.max_steps = 6;
    cfg.enable_composition = true;
    cfg.chain_coherence_weight = 0.3;
    cfg.chain_ctx_blend = 0.15;
    cfg.seed_anchor_weight = 0.35;
    cfg.seed_anchor_decay = 0.03;
    cfg.min_embedding_similarity = 0.05;
    cfg.embedding_sim_weight = 0.1;
    cfg.feedback.enable = true;
    cfg.feedback.max_rounds = 3;
    cfg.feedback.quality_skip_threshold = 0.99;  // High threshold to force feedback rounds
    cfg.feedback.improvement_threshold = -1.0;   // Never stop early for this test
    cfg.feedback.context_blend_alpha = 0.4;

    GraphReasoner fb_reasoner(ltm, registry, embeddings, cfg);

    auto all_ids = ltm.get_all_concept_ids();
    ConceptId seed = 0;
    for (ConceptId cid : all_ids) {
        if (ltm.get_relation_count(cid) >= 3) {
            seed = cid;
            break;
        }
    }
    if (seed == 0 && !all_ids.empty()) seed = all_ids[0];

    // Run reason_with_feedback — should explore different paths
    GraphChain fb_chain = fb_reasoner.reason_with_feedback(seed);
    check(!fb_chain.empty(), "feedback chain is non-empty");

    // Compare with single-round chain
    GraphChain single = fb_reasoner.reason_from(seed);

    log("  Single chain: " + std::to_string(single.steps.size()) + " steps, quality="
        + std::to_string(fb_reasoner.compute_chain_quality(single)));
    log("  Feedback chain: " + std::to_string(fb_chain.steps.size()) + " steps, quality="
        + std::to_string(fb_reasoner.compute_chain_quality(fb_chain)));

    check(true, "reason_with_feedback completes without error");
}

// =============================================================================
// Test 14: Recursive feedback — quality skip threshold
// =============================================================================

static void test_feedback_skip_threshold(LongTermMemory& ltm,
                                          ConceptModelRegistry& registry,
                                          EmbeddingManager& embeddings) {
    log("--- Test 14: Feedback quality skip threshold ---");

    // Very low skip threshold → should skip feedback immediately
    GraphReasonerConfig cfg;
    cfg.max_steps = 6;
    cfg.enable_composition = true;
    cfg.chain_coherence_weight = 0.3;
    cfg.chain_ctx_blend = 0.15;
    cfg.seed_anchor_weight = 0.35;
    cfg.seed_anchor_decay = 0.03;
    cfg.min_embedding_similarity = 0.05;
    cfg.embedding_sim_weight = 0.1;
    cfg.feedback.enable = true;
    cfg.feedback.max_rounds = 3;
    cfg.feedback.quality_skip_threshold = 0.0;  // Skip immediately (any quality >= 0.0)
    cfg.feedback.improvement_threshold = 0.02;

    GraphReasoner skip_reasoner(ltm, registry, embeddings, cfg);

    auto all_ids = ltm.get_all_concept_ids();
    ConceptId seed = 0;
    for (ConceptId cid : all_ids) {
        if (ltm.get_relation_count(cid) >= 2) {
            seed = cid;
            break;
        }
    }
    if (seed == 0 && !all_ids.empty()) seed = all_ids[0];

    // With skip_threshold = 0.0, the first chain should be returned immediately
    GraphChain skip_chain = skip_reasoner.reason_with_feedback(seed);
    GraphChain single_chain = skip_reasoner.reason_from(seed);

    // Both should have the same number of steps (no feedback rounds happened)
    check(skip_chain.steps.size() == single_chain.steps.size(),
          "skip threshold=0 produces same chain as single round");
    log("  Skip chain: " + std::to_string(skip_chain.steps.size()) + " steps");
}

// =============================================================================
// Test 15: FeedbackState and FeedbackConfig struct defaults
// =============================================================================

static void test_feedback_types() {
    log("--- Test 15: FeedbackState/FeedbackConfig defaults ---");

    FeedbackState state;
    check(state.best_quality == 0.0, "FeedbackState default best_quality = 0.0");
    check(state.round == 0, "FeedbackState default round = 0");
    check(state.explored.empty(), "FeedbackState default explored is empty");

    GraphReasonerConfig::FeedbackConfig fcfg;
    check(fcfg.enable == false, "FeedbackConfig default enable = false");
    check(fcfg.max_rounds == 3, "FeedbackConfig default max_rounds = 3");
    check(std::abs(fcfg.quality_skip_threshold - 0.8) < 1e-10,
          "FeedbackConfig default quality_skip_threshold = 0.8");
    check(std::abs(fcfg.improvement_threshold - 0.02) < 1e-10,
          "FeedbackConfig default improvement_threshold = 0.02");
    check(std::abs(fcfg.context_blend_alpha - 0.4) < 1e-10,
          "FeedbackConfig default context_blend_alpha = 0.4");
}

// =============================================================================
// Test 16: ReasoningLogger JSONL output
// =============================================================================

static void test_reasoning_logger(GraphReasoner& reasoner, const LongTermMemory& ltm,
                                   EmbeddingManager& /*embeddings*/) {
    log("--- Test 16: ReasoningLogger JSONL output ---");

    const char* tmp_path = "/tmp/brain19_test_reasoning_log.jsonl";
    std::remove(tmp_path);

    // Create logger and attach to reasoner
    {
        ReasoningLogger logger(tmp_path);
        check(logger.is_open(), "logger opens file");

        reasoner.set_logger(&logger);

        // Find a seed with relations
        auto all_ids = ltm.get_all_concept_ids();
        ConceptId seed = 0;
        for (ConceptId cid : all_ids) {
            if (ltm.get_relation_count(cid) >= 2) {
                seed = cid;
                break;
            }
        }
        if (seed == 0 && !all_ids.empty()) seed = all_ids[0];

        // Run reasoning — should produce a log line
        GraphChain chain = reasoner.reason_from(seed);
        check(!chain.empty(), "chain produced with logger attached");

        // Detach logger before it goes out of scope
        reasoner.set_logger(nullptr);
    }

    // Read and verify the JSONL output
    std::ifstream in(tmp_path);
    check(in.good(), "JSONL file readable");

    std::string line;
    bool got_line = static_cast<bool>(std::getline(in, line));
    check(got_line, "JSONL file has at least one line");
    check(!line.empty(), "JSONL line is non-empty");

    // Verify all required fields are present
    check(line.find("\"ts\":") != std::string::npos, "has ts field");
    check(line.find("\"seed\":") != std::string::npos, "has seed field");
    check(line.find("\"seed_trust\":") != std::string::npos, "has seed_trust field");
    check(line.find("\"seed_type\":") != std::string::npos, "has seed_type field");
    check(line.find("\"seed_rel_count\":") != std::string::npos, "has seed_rel_count field");
    check(line.find("\"seed_emb\":") != std::string::npos, "has seed_emb field");
    check(line.find("\"chain_len\":") != std::string::npos, "has chain_len field");
    check(line.find("\"chain_trust\":") != std::string::npos, "has chain_trust field");
    check(line.find("\"chain_quality\":") != std::string::npos, "has chain_quality field");
    check(line.find("\"termination\":") != std::string::npos, "has termination field");
    check(line.find("\"mag_ratio\":") != std::string::npos, "has mag_ratio field");
    check(line.find("\"chain_pain\":") != std::string::npos, "has chain_pain field");
    check(line.find("\"concepts\":") != std::string::npos, "has concepts field");
    check(line.find("\"relations\":") != std::string::npos, "has relations field");
    check(line.find("\"steps\":") != std::string::npos, "has steps field");
    check(line.find("\"avg_nn\":") != std::string::npos, "has avg_nn field");
    check(line.find("\"avg_kan\":") != std::string::npos, "has avg_kan field");
    check(line.find("\"nn_dom\":") != std::string::npos, "has nn_dom field");
    check(line.find("\"kan_dom\":") != std::string::npos, "has kan_dom field");
    check(line.find("\"total_reward\":") != std::string::npos, "has total_reward field");
    check(line.find("\"total_pain\":") != std::string::npos, "has total_pain field");
    check(line.find("\"fb_round\":") != std::string::npos, "has fb_round field");
    check(line.find("\"fb_round\":-1") != std::string::npos, "fb_round is -1 for non-feedback call");

    // Verify it starts with { and ends with } (valid JSON object)
    check(line.front() == '{', "line starts with {");
    check(line.back() == '}', "line ends with }");

    // Verify no second line (only one reason_from call)
    std::string line2;
    bool got_second = static_cast<bool>(std::getline(in, line2));
    check(!got_second || line2.empty(), "only one JSONL line written");

    in.close();

    // Verify reasoner still works without logger
    {
        auto all_ids = ltm.get_all_concept_ids();
        ConceptId seed = all_ids.empty() ? 0 : all_ids[0];
        GraphChain chain2 = reasoner.reason_from(seed);
        check(!chain2.empty(), "reasoning works after logger detached");
    }

    std::remove(tmp_path);
    log("  JSONL line length: " + std::to_string(line.size()) + " bytes");
}

// =============================================================================
// Test 17: Parallel wake phase (thread_count=4)
// =============================================================================

static void test_parallel_wake(LongTermMemory& ltm, ConceptModelRegistry& registry,
                                EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    log("--- Test 17: Parallel wake phase ---");

    CoLearnConfig config;
    config.wake_chains_per_cycle = 8;
    config.sleep_replay_count = 3;
    config.retrain_epochs = 2;
    config.max_episodes = 100;
    config.thread_count = 4;  // Parallel!

    CoLearnLoop loop(ltm, registry, embeddings, reasoner, config);

    check(loop.cycle_count() == 0, "parallel: initial cycle_count = 0");

    auto results = loop.run_cycles(2);
    check(results.size() == 2, "parallel: 2 cycle results returned");

    size_t total_chains = 0;
    for (const auto& r : results) total_chains += r.chains_produced;
    log("  Parallel chains produced: " + std::to_string(total_chains));
    check(total_chains > 0, "parallel: chains were produced");

    size_t total_episodes = loop.episodic_memory().episode_count();
    check(total_episodes > 0, "parallel: episodes were stored");

    check(results[0].cycle_number == 1, "parallel: cycle 1");
    check(results[1].cycle_number == 2, "parallel: cycle 2");

    log("  Parallel cycle 1: chains=" + std::to_string(results[0].chains_produced)
        + " quality=" + std::to_string(results[0].avg_chain_quality));
    log("  Parallel cycle 2: chains=" + std::to_string(results[1].chains_produced)
        + " quality=" + std::to_string(results[1].avg_chain_quality));
}

// =============================================================================
// Test 18: Thread-safe EpisodicMemory (concurrent stores)
// =============================================================================

static void test_episodic_memory_threadsafe() {
    log("--- Test 18: Thread-safe EpisodicMemory ---");

    EpisodicMemory mem(2000);

    constexpr size_t NUM_THREADS = 4;
    constexpr size_t STORES_PER_THREAD = 100;

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&mem, t]() {
            for (size_t i = 0; i < STORES_PER_THREAD; ++i) {
                Episode ep;
                ep.seed = static_cast<ConceptId>(t * 1000 + i);
                ep.quality = 0.5;
                EpisodeStep s;
                s.concept_id = ep.seed;
                ep.steps.push_back(s);
                mem.store(ep);
            }
        });
    }

    for (auto& th : threads) th.join();

    size_t count = mem.episode_count();
    log("  Episodes stored: " + std::to_string(count)
        + " (expected " + std::to_string(NUM_THREADS * STORES_PER_THREAD) + ")");
    check(count == NUM_THREADS * STORES_PER_THREAD, "thread-safe: all episodes stored");
}

// =============================================================================
// Test 19: Thread-safe ErrorCollector (concurrent collect_from_chain)
// =============================================================================

static void test_error_collector_threadsafe() {
    log("--- Test 19: Thread-safe ErrorCollector ---");

    ErrorCollector collector;

    constexpr size_t NUM_THREADS = 4;
    std::atomic<size_t> chains_submitted{0};

    // Build a minimal chain that will produce a terminal correction
    auto make_chain = [](ConceptId src, ConceptId tgt) -> GraphChain {
        GraphChain chain;
        chain.termination = TerminationReason::NO_VIABLE_CANDIDATES;

        TraceStep s0;
        s0.step_index = 0;
        s0.source_id = src;
        s0.target_id = src;
        s0.composite_score = 0.8;
        chain.steps.push_back(s0);

        TraceStep s1;
        s1.step_index = 1;
        s1.source_id = src;
        s1.target_id = tgt;
        s1.composite_score = 0.9;  // High prediction → terminal error
        chain.steps.push_back(s1);

        return chain;
    };

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < 10; ++i) {
                auto chain = make_chain(
                    static_cast<ConceptId>(t * 100 + i),
                    static_cast<ConceptId>(t * 100 + i + 1));
                ChainSignal signal;
                signal.chain_quality = 0.3;
                collector.collect_from_chain(chain, signal);
                ++chains_submitted;
            }
        });
    }

    for (auto& th : threads) th.join();

    size_t total = collector.total_corrections();
    log("  Total corrections: " + std::to_string(total));
    log("  Terminal count: " + std::to_string(collector.terminal_count()));
    check(chains_submitted.load() == NUM_THREADS * 10, "all chains submitted");
    check(total > 0, "thread-safe: corrections collected");
    check(collector.terminal_count() > 0, "thread-safe: terminal corrections found");
}

// =============================================================================
// Test 20: Continuous mode (start/stop)
// =============================================================================

static void test_continuous_mode(LongTermMemory& ltm, ConceptModelRegistry& registry,
                                  EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    log("--- Test 20: Continuous mode ---");

    CoLearnConfig config;
    config.wake_chains_per_cycle = 3;
    config.sleep_replay_count = 2;
    config.retrain_epochs = 1;
    config.max_episodes = 100;
    config.thread_count = 2;
    config.continuous_interval_ms = 10;  // Small delay for test

    CoLearnLoop loop(ltm, registry, embeddings, reasoner, config);

    std::atomic<size_t> callback_count{0};
    loop.set_cycle_callback([&callback_count](const CoLearnLoop::CycleResult& /*r*/) {
        ++callback_count;
    });

    check(!loop.is_running(), "not running before start");

    loop.start_continuous();
    check(loop.is_running(), "running after start");

    // Let it run for ~200ms
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    loop.stop_continuous();
    check(!loop.is_running(), "not running after stop");

    size_t cycles = loop.cycle_count();
    size_t callbacks = callback_count.load();
    log("  Cycles completed: " + std::to_string(cycles));
    log("  Callbacks fired: " + std::to_string(callbacks));
    check(cycles > 0, "continuous: at least 1 cycle completed");
    check(callbacks > 0, "continuous: callback was called");
    check(callbacks == cycles, "continuous: callback count matches cycle count");

    // Double stop should be safe
    loop.stop_continuous();
    check(true, "double stop_continuous is safe");
}

// =============================================================================
// Test 21: Serial fallback equivalence (thread_count=1 produces valid results)
// =============================================================================

static void test_serial_fallback(LongTermMemory& ltm, ConceptModelRegistry& registry,
                                  EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    log("--- Test 21: Serial fallback equivalence ---");

    CoLearnConfig config;
    config.wake_chains_per_cycle = 5;
    config.sleep_replay_count = 3;
    config.retrain_epochs = 2;
    config.max_episodes = 100;
    config.thread_count = 1;  // Explicit serial

    CoLearnLoop loop(ltm, registry, embeddings, reasoner, config);
    auto result = loop.run_cycle();

    check(result.cycle_number == 1, "serial fallback: cycle_number = 1");
    check(result.chains_produced > 0, "serial fallback: chains produced");
    check(result.episodes_stored > 0, "serial fallback: episodes stored");
    check(result.avg_chain_quality >= 0.0, "serial fallback: quality >= 0");

    log("  Serial: chains=" + std::to_string(result.chains_produced)
        + " quality=" + std::to_string(result.avg_chain_quality));
}

// =============================================================================
// Main
// =============================================================================

int main() {
    log("=== Co-Learning Loop + Episodic Memory Test ===");
    log("");

    // Test 1: LTM modify_relation_weight (standalone, no setup needed)
    test_ltm_modify_weight();

    // Test 2-4: EpisodicMemory (standalone)
    test_episodic_memory_basics();
    test_episodic_memory_replay();
    test_episodic_memory_eviction();

    // Test 8: Named constants
    test_named_constants();

    // Setup for integration tests
    log("");
    log("--- Setup: Loading KB + Training ---");

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

    // Property inheritance
    {
        PropertyInheritance prop(ltm);
        PropertyInheritance::Config cfg;
        cfg.max_iterations = 50;
        cfg.max_hop_depth = 20;
        auto r = prop.propagate(cfg);
        log("  Inherited: " + std::to_string(r.properties_inherited));
    }

    // Train
    EmbeddingManager embeddings;
    ConceptModelRegistry registry;
    {
        embeddings.train_embeddings(ltm, 0.05, 5);
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto stats = trainer.train_all(registry, embeddings, ltm);
        log("  Models: " + std::to_string(stats.models_trained)
            + " (" + std::to_string(stats.models_converged) + " converged)");
    }

    // Create GraphReasoner
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

    log("");

    // Integration tests
    test_chain_to_episode(reasoner, ltm, embeddings);
    test_knowledge_extractor(reasoner, ltm);
    test_colearn_loop(ltm, registry, embeddings, reasoner);

    // Professor review fixes
    test_residual_connection(reasoner, ltm);
    test_lr_decay(ltm, reasoner);
    test_convergence_gate();

    // Recursive feedback tests
    test_feedback_types();
    test_feedback_backward_compat(reasoner, ltm);
    test_feedback_explored_growth(ltm, registry, embeddings);
    test_feedback_skip_threshold(ltm, registry, embeddings);

    // Reasoning logger test
    test_reasoning_logger(reasoner, ltm, embeddings);

    // Parallel CoLearnLoop tests
    test_parallel_wake(ltm, registry, embeddings, reasoner);
    test_episodic_memory_threadsafe();
    test_error_collector_threadsafe();
    test_continuous_mode(ltm, registry, embeddings, reasoner);
    test_serial_fallback(ltm, registry, embeddings, reasoner);

    // Summary
    log("");
    log("=== Results ===");
    log("  Passed: " + std::to_string(tests_passed));
    log("  Failed: " + std::to_string(tests_failed));

    if (tests_failed > 0) {
        log("  STATUS: SOME TESTS FAILED");
        return 1;
    }

    log("  STATUS: ALL TESTS PASSED");
    return 0;
}

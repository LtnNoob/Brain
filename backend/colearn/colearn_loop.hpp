#pragma once

#include "colearn_types.hpp"
#include "episodic_memory.hpp"
#include "knowledge_extractor.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../cmodel/concept_trainer.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../graph_net/graph_reasoner.hpp"

#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// CO-LEARNING LOOP — Orchestrator: Wake / Sleep / Train
// =============================================================================
//
// The brain learns through Co-Learning (LDS = Lernen durch Schmerz).
// CMs and the knowledge graph co-evolve through iterative bootstrapping:
//   Seed with minimal (bad) knowledge → train bad CMs → CMs evaluate relations
//   → update graph → retrain → repeat.
//
// Wake Phase:  Reason → store episodes
// Sleep Phase: Replay → consolidate → apply weight deltas
// Train Phase: Retrain CMs on updated graph
//

class CoLearnLoop {
public:
    struct CycleResult {
        size_t chains_produced = 0;
        size_t episodes_stored = 0;
        double avg_chain_quality = 0.0;
        ConsolidationResult consolidation;
        size_t models_retrained = 0;
        size_t models_converged = 0;
        size_t models_rolled_back = 0;
        double avg_loss_before = 0.0;
        double avg_loss_after = 0.0;
        size_t cycle_number = 0;
        double quality_delta = 0.0;
        // Error-driven learning stats
        size_t correction_samples_injected = 0;
        size_t terminal_corrections = 0;
        size_t quality_drop_corrections = 0;
        size_t success_reinforcements = 0;
    };

    CoLearnLoop(LongTermMemory& ltm, ConceptModelRegistry& registry,
                EmbeddingManager& embeddings, GraphReasoner& reasoner,
                const CoLearnConfig& config = {});

    // Run one full wake/sleep/train cycle
    CycleResult run_cycle();

    // Run multiple cycles
    std::vector<CycleResult> run_cycles(size_t n);

    // Individual phases (for fine-grained control)
    void wake_phase();
    void sleep_phase();
    void train_phase();

    // Accessors
    EpisodicMemory& episodic_memory() { return episodic_memory_; }
    const EpisodicMemory& episodic_memory() const { return episodic_memory_; }
    size_t cycle_count() const { return cycle_count_; }
    double last_avg_quality() const { return last_avg_quality_; }

private:
    LongTermMemory& ltm_;
    ConceptModelRegistry& registry_;
    EmbeddingManager& embeddings_;
    GraphReasoner& reasoner_;
    CoLearnConfig config_;
    EpisodicMemory episodic_memory_;
    KnowledgeExtractor extractor_;
    size_t cycle_count_ = 0;
    double last_avg_quality_ = 0.0;

    // Internal state for current cycle
    size_t last_chains_produced_ = 0;
    size_t last_episodes_stored_ = 0;
    double last_quality_sum_ = 0.0;
    ConsolidationResult last_consolidation_;
    ConceptTrainerStats last_train_stats_;
    double pre_train_avg_loss_ = 0.0;

    // Pain tracking for seed selection (EMA per seed)
    std::unordered_map<ConceptId, double> seed_pain_scores_;

    // Error-driven learning: collects prediction errors from wake phase
    ErrorCollector error_collector_;

    // Correction stats from last train phase
    struct CorrectionStats {
        size_t samples_injected = 0;
        size_t terminal = 0;
        size_t quality_drop = 0;
        size_t success = 0;
    };
    CorrectionStats last_correction_stats_;

    // Select diverse seeds for wake phase (4-way: random + hub + low-trust + high-pain)
    std::vector<ConceptId> select_wake_seeds();
};

} // namespace brain19

#pragma once

#include "colearn_types.hpp"
#include "episodic_memory.hpp"
#include "knowledge_extractor.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../cmodel/concept_trainer.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../graph_net/graph_reasoner.hpp"
#include "../curiosity/curiosity_engine.hpp"

#include "../concurrent/thread_pool.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
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
        // Superposition stats
        size_t superposition_enabled_count = 0;   // Concepts newly enabled this cycle
        size_t superposition_trained_count = 0;    // Concepts with superposition training updates
        // FlexEmbedding growth/shrink stats
        size_t embeddings_grown = 0;
        size_t embeddings_shrunk = 0;
    };

    CoLearnLoop(LongTermMemory& ltm, ConceptModelRegistry& registry,
                EmbeddingManager& embeddings, GraphReasoner& reasoner,
                const CoLearnConfig& config = {});

    ~CoLearnLoop();

    // Non-copyable
    CoLearnLoop(const CoLearnLoop&) = delete;
    CoLearnLoop& operator=(const CoLearnLoop&) = delete;

    // Run one full wake/sleep/train cycle
    CycleResult run_cycle();

    // Run multiple cycles
    std::vector<CycleResult> run_cycles(size_t n);

    // Individual phases (for fine-grained control)
    void wake_phase();
    void sleep_phase();
    void train_phase();

    // Continuous mode: runs wake/sleep/train in a loop on a background thread
    using CycleCallback = std::function<void(const CycleResult&)>;
    void start_continuous();
    void stop_continuous();
    bool is_running() const;
    void set_cycle_callback(CycleCallback cb);

    // CuriosityEngine integration
    void set_curiosity_engine(CuriosityEngine* engine) { curiosity_ = engine; }

    // Expose seed pain scores for CuriosityEngine
    const std::unordered_map<ConceptId, double>& seed_pain_scores() const { return seed_pain_scores_; }

    // Accessors
    EpisodicMemory& episodic_memory() { return episodic_memory_; }
    const EpisodicMemory& episodic_memory() const { return episodic_memory_; }
    const ErrorCollector& error_collector() const { return error_collector_; }
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

    // CuriosityEngine: intelligent seed selection (nullable, falls back to 4-way)
    CuriosityEngine* curiosity_ = nullptr;

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

    // Superposition context tracking: per-concept quality grouped by source relation
    struct SuperpositionTracker {
        struct ConceptContextStats {
            std::unordered_map<uint16_t, std::vector<double>> quality_by_relation;
            size_t total_observations = 0;
        };
        std::unordered_map<ConceptId, ConceptContextStats> stats;

        void record(ConceptId cid, RelationType rel, double quality);
        bool should_enable(ConceptId cid, size_t min_observations,
                           double min_std) const;
    };
    SuperpositionTracker superposition_tracker_;
    size_t last_superposition_enabled_ = 0;
    size_t last_superposition_trained_ = 0;

    // FlexEmbedding growth/shrink tracking
    uint32_t tick_ = 0;
    size_t last_embeddings_grown_ = 0;
    size_t last_embeddings_shrunk_ = 0;

    // Select diverse seeds for wake phase (4-way: random + hub + low-trust + high-pain)
    std::vector<ConceptId> select_wake_seeds();

    // Parallel wake implementation
    void wake_phase_serial(const std::vector<ConceptId>& seeds);
    void wake_phase_parallel(const std::vector<ConceptId>& seeds);

    // Thread pool for parallel wake phase (nullptr if thread_count <= 1)
    std::unique_ptr<ThreadPool> thread_pool_;

    // Continuous mode
    std::thread continuous_thread_;
    std::atomic<bool> continuous_running_{false};
    std::atomic<bool> continuous_stop_{false};
    CycleCallback cycle_callback_;
};

} // namespace brain19

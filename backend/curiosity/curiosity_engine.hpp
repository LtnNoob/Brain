#pragma once

#include "curiosity_trigger.hpp"
#include "signal_types.hpp"
#include "curiosity_score.hpp"
#include "trend_tracker.hpp"

#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../colearn/episodic_memory.hpp"
#include "../colearn/error_collector.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace brain19 {

// Forward declarations (nullable subsystems)
class EpistemicPromotion;
class StreamMonitor;

// Backward-compat observation snapshot (used by ThinkingPipeline)
struct SystemObservation {
    ContextId context_id = 0;
    size_t active_concept_count = 0;
    size_t active_relation_count = 0;
};

// =============================================================================
// CURIOSITY CONFIG
// =============================================================================

struct CuriosityConfig {
    // 12 dimension weights (normalized internally)
    double w_pain = 0.20;
    double w_trust = 0.10;
    double w_model = 0.10;
    double w_nn_kan = 0.10;
    double w_topology = 0.08;
    double w_contradiction = 0.12;
    double w_pred_error = 0.10;
    double w_novelty = 0.05;
    double w_episodic = 0.05;
    double w_activation = 0.03;
    double w_edge_weight = 0.02;
    double w_quality_deg = 0.05;

    // Cross-signal consciousness detector
    double cross_signal_bonus = 1.5;
    size_t cross_signal_min_dims = 3;

    // Seed planning
    size_t max_seeds = 50;
    double diversity_weight = 0.3;

    // Trigger generation
    double trigger_threshold = 0.3;  // min score to generate a trigger
    size_t max_triggers = 10;
    size_t min_cluster_size = 2;     // min concepts per trigger cluster
};

// =============================================================================
// CURIOSITY ENGINE — "Das Bewusstsein des Systems"
// =============================================================================
//
// 4-phase pipeline:
//   Observe → Score → Plan → Trigger
//
// Read-access to ALL system metrics, generates intelligent prioritized seeds.
//

class CuriosityEngine {
public:
    explicit CuriosityEngine(CuriosityConfig config = {});
    ~CuriosityEngine();

    // ─── Full refresh: observe all signals, score, generate plan ────────────
    void refresh(
        const LongTermMemory& ltm,
        const ConceptModelRegistry& registry,
        const EpisodicMemory& episodic,
        const ErrorCollector& error_collector,
        const std::unordered_map<ConceptId, double>& seed_pain_scores,
        const EpistemicPromotion* promotion,   // nullable
        const StreamMonitor* monitor           // nullable
    );

    // ─── Seed selection (reads cached plan from last refresh) ───────────────
    std::vector<ConceptId> select_seeds(size_t count) const;
    std::vector<SeedEntry> select_seed_entries(size_t count) const;

    // ─── Trigger generation (reads cached scores) ──────────────────────────
    std::vector<CuriosityTrigger> generate_triggers(ContextId ctx) const;

    // ─── Backward-compat wrapper for ThinkingPipeline ──────────────────────
    std::vector<CuriosityTrigger> observe_and_generate_triggers(
        const std::vector<SystemObservation>& observations);

    // ─── Health & introspection ────────────────────────────────────────────
    double system_health() const;
    const TrendTracker& trends() const { return trends_; }
    const SeedPlan& last_plan() const { return cached_plan_; }
    const std::vector<CuriosityScore>& last_scores() const { return cached_scores_; }

    // Legacy setters (no-ops, kept for compilation)
    void set_shallow_relation_threshold(double) {}
    void set_low_exploration_threshold(size_t) {}

private:
    CuriosityConfig config_;
    TrendTracker trends_;
    SeedPlan cached_plan_;
    std::vector<CuriosityScore> cached_scores_;
    SystemSnapshot cached_snapshot_;

    // Normalized weights (computed once from config)
    std::array<double, 12> weights_{};
    void normalize_weights();

    // ─── Phase 1: Build SystemSnapshot from subsystem refs ─────────────────
    SystemSnapshot observe(
        const LongTermMemory& ltm,
        const ConceptModelRegistry& registry,
        const EpisodicMemory& episodic,
        const ErrorCollector& error_collector,
        const std::unordered_map<ConceptId, double>& seed_pain_scores
    ) const;

    // ─── Phase 2: Score all concepts ───────────────────────────────────────
    std::vector<CuriosityScore> score_concepts(const SystemSnapshot& snap) const;

    // ─── Phase 3: Generate seed plan with diversity ────────────────────────
    SeedPlan generate_seed_plan(
        const std::vector<CuriosityScore>& scores,
        const SystemSnapshot& snap,
        const LongTermMemory& ltm);

    // ─── Per-dimension scoring (all normalize to [0,1]) ────────────────────
    double score_pain(const ConceptSignals& cs) const;
    double score_trust_deficit(const ConceptSignals& cs) const;
    double score_model_uncertainty(const ConceptSignals& cs) const;
    double score_nn_kan_conflict(const ConceptSignals& cs) const;
    double score_topology_gap(const ConceptSignals& cs, const SystemSignals& sys) const;
    double score_contradiction(const ConceptSignals& cs) const;
    double score_prediction_error(const ConceptSignals& cs, const SystemSignals& sys) const;
    double score_novelty(const ConceptSignals& cs) const;
    double score_episodic_revisit(const ConceptSignals& cs) const;
    double score_activation_anomaly(const ConceptSignals& cs) const;
    double score_edge_weight_anomaly(const ConceptSignals& cs) const;
    double score_quality_degradation(const ConceptSignals& cs) const;
    double score_cross_signal(const std::array<double, CURIOSITY_DIM_COUNT>& dims) const;

    // ─── Diversity-aware selection ─────────────────────────────────────────
    void diversify_seeds(std::vector<SeedEntry>& seeds,
                         const LongTermMemory& ltm, size_t target) const;

    // ─── Dimension → GoalType mapping ──────────────────────────────────────
    static GoalType dimension_to_goal(CuriosityDimension dim);

    // ─── Trigger clustering ────────────────────────────────────────────────
    static TriggerType dimension_to_trigger_type(CuriosityDimension dim);
};

} // namespace brain19

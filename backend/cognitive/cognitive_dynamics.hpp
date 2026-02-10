#pragma once

#include "cognitive_config.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/stm.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <atomic>

namespace brain19 {

// =============================================================================
// COGNITIVE DYNAMICS
// =============================================================================
//
// Cognitive Dynamics ist eine ADDITIVE Schicht für Brain19 die modelliert:
// - Spreading Activation (entlang Relations, trust-gewichtet, depth-limited)
// - Salience Computation (Wichtigkeits-Ranking)
// - Focus Management (Arbeitsgedächtnis mit Kapazitätsgrenzen)
// - Thought Path Ranking (Inferenzpfad-Priorisierung)
//
// ARCHITEKTUR-VERTRAG:
// ✅ READ-ONLY Zugriff auf LTM und Trust
// ✅ SCHREIBT nur in STM (Aktivierungen) und eigenen Zustand
// ✅ Deterministisch (gleiche Inputs → gleiche Outputs)
// ✅ Bounded (alle Werte in [0.0, 1.0])
// ✅ Depth-limited (keine Endlosschleifen)
//
// ❌ DARF NICHT:
// - Wissen erzeugen
// - Hypothesen generieren  
// - Trust ändern
// - Epistemische Entscheidungen treffen
//
// INTEGRATION:
// Wird vom BrainController aufgerufen, nicht autonom.
// Parallel rechnen erlaubt, seriell entscheiden.
//
class CognitiveDynamics {
public:
    explicit CognitiveDynamics(CognitiveDynamicsConfig config = CognitiveDynamicsConfig());
    ~CognitiveDynamics();
    
    // No copy (owns state)
    CognitiveDynamics(const CognitiveDynamics&) = delete;
    CognitiveDynamics& operator=(const CognitiveDynamics&) = delete;
    
    // Move allowed
    CognitiveDynamics(CognitiveDynamics&&) = default;
    CognitiveDynamics& operator=(CognitiveDynamics&&) = default;
    
    // =========================================================================
    // SPREADING ACTIVATION
    // =========================================================================
    // Propagate activation from source along relations.
    // Uses trust as weight factor, respects depth limits.
    //
    // FORMULA:
    // activation(B) += activation(A) × relation_weight × trust(A) × damping^depth
    //
    // GUARANTEES:
    // - Deterministic: same inputs → same outputs
    // - Bounded: all activations ∈ [0.0, max_activation]
    // - Depth-limited: no infinite loops
    // - Trust unchanged: LTM is read-only
    
    // Spread activation from single source
    SpreadingStats spread_activation(
        ConceptId source,
        double initial_activation,
        ContextId context,
        const LongTermMemory& ltm,
        ShortTermMemory& stm
    );
    
    // Spread activation from multiple sources
    SpreadingStats spread_activation_multi(
        const std::vector<ConceptId>& sources,
        double initial_activation,
        ContextId context,
        const LongTermMemory& ltm,
        ShortTermMemory& stm
    );
    
    // =========================================================================
    // SALIENCE COMPUTATION
    // =========================================================================
    // Compute importance scores for concepts based on:
    // - Activation level (from STM)
    // - Trust value (from epistemic metadata)
    // - Connectivity (relation count)
    // - Recency (last access time)
    //
    // GUARANTEES:
    // - Read-only: LTM/STM not modified
    // - Deterministic: same inputs → same outputs
    // - Bounded: salience ∈ [0.0, max_salience]
    
    // Compute salience for single cid
    SalienceScore compute_salience(
        ConceptId cid,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        uint64_t current_tick = 0
    ) const;
    
    // Compute salience for multiple concepts
    std::vector<SalienceScore> compute_salience_batch(
        const std::vector<ConceptId>& concepts,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        uint64_t current_tick = 0
    ) const;
    
    // Get top-K most salient concepts
    std::vector<SalienceScore> get_top_k_salient(
        const std::vector<ConceptId>& candidates,
        size_t k,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        uint64_t current_tick = 0
    ) const;
    
    // Compute salience with query boost
    std::vector<SalienceScore> compute_query_salience(
        const std::vector<ConceptId>& query_concepts,
        const std::vector<ConceptId>& candidates,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        uint64_t current_tick = 0
    ) const;
    
    // =========================================================================
    // FOCUS MANAGEMENT
    // =========================================================================
    // Manage attention allocation with capacity limits.
    // Focus set represents current "working memory".
    //
    // GUARANTEES:
    // - Bounded size: max_focus_size
    // - Decay over time
    // - Deterministic updates
    
    // Initialize focus for context
    void init_focus(ContextId context);
    
    // Add cid to focus
    void focus_on(ContextId context, ConceptId cid, double boost = 0.0);
    
    // Apply focus decay
    void decay_focus(ContextId context);
    
    // Get current focus set (sorted by focus score)
    std::vector<FocusEntry> get_focus_set(ContextId context) const;
    
    // Check if cid is in focus
    bool is_focused(ContextId context, ConceptId cid) const;
    
    // Get focus score for cid
    double get_focus_score(ContextId context, ConceptId cid) const;
    
    // Clear focus for context
    void clear_focus(ContextId context);
    
    // =========================================================================
    // THOUGHT PATH RANKING
    // =========================================================================
    // Prioritize inference paths based on salience and trust.
    // Supports beam search for efficient exploration.
    //
    // GUARANTEES:
    // - Bounded paths: max_paths
    // - Depth-limited: max_depth
    // - Deterministic ranking
    
    // Find best thought paths from source to any target
    std::vector<ThoughtPath> find_best_paths(
        ConceptId source,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm
    ) const;
    
    // Find best thought paths to specific target
    std::vector<ThoughtPath> find_paths_to(
        ConceptId source,
        ConceptId target,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm
    ) const;
    
    // Score a given path
    double score_path(
        const ThoughtPath& path,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm
    ) const;
    
    // =========================================================================
    // CONFIGURATION
    // =========================================================================
    
    void set_config(const CognitiveDynamicsConfig& config);
    const CognitiveDynamicsConfig& get_config() const { return config_; }
    
    // =========================================================================
    // STATISTICS
    // =========================================================================
    
    struct Stats {
        std::atomic<uint64_t> total_spreads{0};
        std::atomic<uint64_t> total_salience_computations{0};
        std::atomic<uint64_t> total_focus_updates{0};
        std::atomic<uint64_t> total_path_searches{0};
        SpreadingStats last_spread;  // THREAD-SAFETY: protected by single-writer assumption
        
        Stats() = default;
        Stats(Stats&& o) noexcept
            : total_spreads(o.total_spreads.load(std::memory_order_relaxed))
            , total_salience_computations(o.total_salience_computations.load(std::memory_order_relaxed))
            , total_focus_updates(o.total_focus_updates.load(std::memory_order_relaxed))
            , total_path_searches(o.total_path_searches.load(std::memory_order_relaxed))
            , last_spread(o.last_spread)
        {}
        Stats& operator=(Stats&& o) noexcept {
            total_spreads.store(o.total_spreads.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_salience_computations.store(o.total_salience_computations.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_focus_updates.store(o.total_focus_updates.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_path_searches.store(o.total_path_searches.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_spread = o.last_spread;
            return *this;
        }
        Stats(const Stats&) = delete;
        Stats& operator=(const Stats&) = delete;
    };
    
    Stats get_stats() const {
        Stats copy;
        copy.total_spreads.store(stats_.total_spreads.load(std::memory_order_relaxed));
        copy.total_salience_computations.store(stats_.total_salience_computations.load(std::memory_order_relaxed));
        copy.total_focus_updates.store(stats_.total_focus_updates.load(std::memory_order_relaxed));
        copy.total_path_searches.store(stats_.total_path_searches.load(std::memory_order_relaxed));
        copy.last_spread = stats_.last_spread;
        return copy;
    }
    void reset_stats();
    
private:
    CognitiveDynamicsConfig config_;
    mutable Stats stats_;
    
    // Focus state per context
    std::unordered_map<ContextId, std::vector<FocusEntry>> focus_sets_;
    uint64_t current_tick_ = 0;
    
    // =========================================================================
    // INTERNAL HELPERS
    // =========================================================================
    
    // Spreading activation recursive helper
    void spread_recursive(
        ConceptId current,
        double activation,
        size_t depth,
        ContextId context,
        const LongTermMemory& ltm,
        ShortTermMemory& stm,
        std::unordered_set<ConceptId>& visited,
        SpreadingStats& stats
    );
    
    // Salience factor helpers
    double compute_activation_factor(
        ConceptId cid,
        ContextId context,
        const ShortTermMemory& stm
    ) const;
    
    double compute_trust_factor(
        ConceptId cid,
        const LongTermMemory& ltm
    ) const;
    
    double compute_connectivity_factor(
        ConceptId cid,
        const LongTermMemory& ltm,
        size_t max_connectivity
    ) const;
    
    double compute_recency_factor(
        ConceptId cid,
        ContextId context,
        uint64_t current_tick
    ) const;
    
    double compute_query_boost(
        ConceptId cid,
        const std::vector<ConceptId>& query_concepts,
        const LongTermMemory& ltm
    ) const;
    
    // Clamping helpers
    double clamp_activation(double value) const;
    double clamp_salience(double value) const;
    double clamp_focus(double value) const;
    
    // Path search helpers
    void expand_paths(
        std::vector<ThoughtPath>& paths,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        const ConceptId* target = nullptr  // nullptr = any target
    ) const;
    
    double compute_path_score(
        const ThoughtPath& path,
        ContextId context,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm
    ) const;
    
    // Focus helpers
    void prune_focus_set(ContextId context);
    void update_access_time(ContextId context, ConceptId cid);
};

} // namespace brain19

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

namespace brain19 {

// =============================================================================
// COGNITIVE DYNAMICS CONFIGURATION
// =============================================================================
//
// ARCHITEKTUR-VERTRAG (NICHT VERHANDELBAR):
//
// 1. Cognitive Dynamics ist READ-ONLY bezüglich:
//    - LTM (Long-Term Memory)
//    - Trust-Werte
//    - EpistemicType
//    - EpistemicStatus
//
// 2. Cognitive Dynamics SCHREIBT NUR:
//    - Eigene interne Zustände (Salience, Focus)
//    - STM Aktivierungen (durch Delegation)
//
// 3. Cognitive Dynamics DARF NICHT:
//    - Wissen erzeugen
//    - Hypothesen generieren
//    - Epistemische Entscheidungen treffen
//    - Trust-Werte ändern
//
// 4. Cognitive Dynamics MACHT:
//    - Salience berechnen (Wichtigkeits-Ranking)
//    - Focus verwalten (Aufmerksamkeits-Allokation)
//    - Denkpfade priorisieren (Inferenz-Guidance)
//    - Spreading Activation (Trust-gewichtet, depth-limited)
//
// =============================================================================

using ConceptId = uint64_t;
using ContextId = uint64_t;
using RelationId = uint64_t;

// -----------------------------------------------------------------------------
// ACTIVATION SPREADER CONFIG
// -----------------------------------------------------------------------------

struct ActivationSpreaderConfig {
    // Maximum propagation depth (prevents infinite loops)
    size_t max_depth = 3;
    
    // Damping factor per depth level [0.0, 1.0]
    // Higher = less decay, more propagation
    double damping_factor = 0.8;
    
    // Minimum activation threshold to continue spreading
    double activation_threshold = 0.01;
    
    // Maximum activation value (bounded)
    double max_activation = 1.0;
    
    // Minimum activation value (floor)
    double min_activation = 0.0;
    
    // Whether to use trust as weight factor
    bool trust_weighted = true;
    
    // Whether to use relation weight as factor
    bool relation_weighted = true;
    
    bool is_valid() const {
        return max_depth > 0 &&
               max_depth <= 10 &&
               damping_factor >= 0.0 &&
               damping_factor <= 1.0 &&
               activation_threshold >= 0.0 &&
               activation_threshold <= 1.0 &&
               max_activation > min_activation &&
               max_activation <= 1.0 &&
               min_activation >= 0.0;
    }
};

// -----------------------------------------------------------------------------
// FOCUS MANAGER CONFIG
// -----------------------------------------------------------------------------

struct FocusManagerConfig {
    // Maximum concepts in focus set (cognitive load limit)
    // Miller's magic number: 7 ± 2
    size_t max_focus_size = 7;
    
    // Focus decay rate per tick [0.0, 1.0]
    double decay_rate = 0.1;
    
    // Minimum focus score to stay in focus set
    double focus_threshold = 0.05;
    
    // Maximum focus score (bounded)
    double max_focus = 1.0;
    
    // Focus boost when explicitly attended
    double attention_boost = 0.3;
    
    bool is_valid() const {
        return max_focus_size > 0 &&
               max_focus_size <= 50 &&
               decay_rate >= 0.0 &&
               decay_rate <= 1.0 &&
               focus_threshold >= 0.0 &&
               focus_threshold <= 1.0 &&
               max_focus > 0.0 &&
               max_focus <= 1.0 &&
               attention_boost >= 0.0 &&
               attention_boost <= 1.0;
    }
};

// -----------------------------------------------------------------------------
// SALIENCE COMPUTER CONFIG
// -----------------------------------------------------------------------------

struct SalienceComputerConfig {
    // Weight for activation in salience formula
    double activation_weight = 0.4;
    
    // Weight for trust in salience formula
    double trust_weight = 0.3;
    
    // Weight for connectivity (relation count)
    double connectivity_weight = 0.2;
    
    // Weight for recency (0.0: no per-concept access tracking exists yet)
    double recency_weight = 0.0;
    
    // Maximum salience score
    double max_salience = 1.0;
    
    // Minimum salience threshold (below = irrelevant)
    double salience_threshold = 0.01;
    
    // Query match boost factor
    double query_boost_factor = 0.5;
    
    bool is_valid() const {
        double total = activation_weight + trust_weight + 
                       connectivity_weight + recency_weight;
        return total > 0.0 &&
               total <= 1.0 + 0.001 &&
               max_salience > 0.0 &&
               max_salience <= 1.0 &&
               salience_threshold >= 0.0 &&
               query_boost_factor >= 0.0 &&
               query_boost_factor <= 1.0;
    }
};

// -----------------------------------------------------------------------------
// THOUGHT PATH CONFIG
// -----------------------------------------------------------------------------

struct ThoughtPathConfig {
    // Maximum paths to track (beam width)
    size_t max_paths = 10;
    
    // Minimum path score to keep exploring
    double path_threshold = 0.05;
    
    // Depth penalty factor per level
    double depth_penalty = 0.1;
    
    // Maximum path depth
    size_t max_depth = 5;
    
    // Weight for salience in path score
    double salience_weight = 0.5;
    
    // Weight for trust in path score
    double trust_weight = 0.3;
    
    // Weight for coherence in path score
    double coherence_weight = 0.2;
    
    bool is_valid() const {
        double total = salience_weight + trust_weight + coherence_weight;
        return max_paths > 0 &&
               max_paths <= 100 &&
               path_threshold >= 0.0 &&
               path_threshold <= 1.0 &&
               depth_penalty >= 0.0 &&
               depth_penalty <= 1.0 &&
               max_depth > 0 &&
               max_depth <= 10 &&
               total > 0.0 &&
               total <= 1.0 + 0.001;
    }
};

// -----------------------------------------------------------------------------
// MASTER CONFIG
// -----------------------------------------------------------------------------

struct CognitiveDynamicsConfig {
    ActivationSpreaderConfig spreader;
    FocusManagerConfig focus;
    SalienceComputerConfig salience;
    ThoughtPathConfig thought_path;
    
    // Enable/disable components
    bool enable_spreading = true;
    bool enable_focus_decay = true;
    bool enable_salience = true;
    bool enable_path_ranking = true;
    
    // Debug mode
    bool debug_mode = false;
    
    CognitiveDynamicsConfig() = default;
    
    bool is_valid() const {
        return spreader.is_valid() &&
               focus.is_valid() &&
               salience.is_valid() &&
               thought_path.is_valid();
    }
};

// =============================================================================
// STATE TYPES
// =============================================================================

// Activation entry in spreading
struct ActivationEntry {
    ConceptId concept_id;
    double activation;
    size_t depth;
    
    ActivationEntry() : concept_id(0), activation(0.0), depth(0) {}
    ActivationEntry(ConceptId id, double act, size_t d)
        : concept_id(id), activation(act), depth(d) {}
};

// Focus entry for attention management
struct FocusEntry {
    ConceptId concept_id;
    double focus_score;
    uint64_t last_accessed_tick;
    
    FocusEntry() : concept_id(0), focus_score(0.0), last_accessed_tick(0) {}
    FocusEntry(ConceptId id, double score, uint64_t tick)
        : concept_id(id), focus_score(score), last_accessed_tick(tick) {}
    
    // Comparison for sorting (highest focus first)
    bool operator<(const FocusEntry& other) const {
        return focus_score > other.focus_score;
    }
};

// Salience score with breakdown
struct SalienceScore {
    ConceptId concept_id;
    double salience;
    double activation_contrib;
    double trust_contrib;
    double connectivity_contrib;
    double recency_contrib;
    double query_boost;
    
    SalienceScore() 
        : concept_id(0), salience(0.0), activation_contrib(0.0),
          trust_contrib(0.0), connectivity_contrib(0.0),
          recency_contrib(0.0), query_boost(0.0) {}
    
    explicit SalienceScore(ConceptId id)
        : concept_id(id), salience(0.0), activation_contrib(0.0),
          trust_contrib(0.0), connectivity_contrib(0.0),
          recency_contrib(0.0), query_boost(0.0) {}
    
    // Comparison for sorting (highest salience first)
    bool operator<(const SalienceScore& other) const {
        return salience > other.salience;
    }
};

// Thought path node
struct ThoughtPathNode {
    ConceptId concept_id;
    RelationId relation_id;  // Relation that led here (0 for root)
    double local_score;
    double cumulative_score;
    size_t depth;
    
    ThoughtPathNode()
        : concept_id(0), relation_id(0), local_score(0.0),
          cumulative_score(0.0), depth(0) {}
    
    ThoughtPathNode(ConceptId cid, RelationId rid, double local, double cum, size_t d)
        : concept_id(cid), relation_id(rid), local_score(local),
          cumulative_score(cum), depth(d) {}
};

// Complete thought path
struct ThoughtPath {
    std::vector<ThoughtPathNode> nodes;
    double total_score;
    
    ThoughtPath() : total_score(0.0) {}
    
    size_t length() const { return nodes.size(); }
    bool empty() const { return nodes.empty(); }
    
    // Comparison for sorting (highest score first)
    bool operator<(const ThoughtPath& other) const {
        return total_score > other.total_score;
    }
};

// Spreading activation statistics
struct SpreadingStats {
    size_t concepts_activated;
    size_t max_depth_reached;
    size_t total_propagations;
    double total_activation_added;
    
    SpreadingStats()
        : concepts_activated(0), max_depth_reached(0),
          total_propagations(0), total_activation_added(0.0) {}
    
    void reset() {
        concepts_activated = 0;
        max_depth_reached = 0;
        total_propagations = 0;
        total_activation_added = 0.0;
    }
};

} // namespace brain19

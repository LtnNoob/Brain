#pragma once

#include "../micromodel/flex_embedding.hpp"
#include "../memory/active_relation.hpp"
#include "../common/types.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace brain19 {

// =============================================================================
// GRAPH REASONING TYPES
// =============================================================================
//
// The Knowledge Graph IS the neural network. Traversal = Forward Pass.
// Each concept's ConceptModel W(16x16) transforms activation vectors,
// each edge is a dimensional transformation. Full activation vectors
// flow through the graph --- no collapsing to scalars.
//
// OOP Design:
//   Activation is a proper class (encapsulated core/detail).
//   EdgeResult is a value object carrying transformation results.
//   GraphReasonerConfig is a structured configuration.
//   TerminationReason is a strongly-typed enum with descriptions.
//

// =============================================================================
// TerminationReason --- Why did the chain stop?
// =============================================================================

enum class TerminationReason {
    STILL_RUNNING = 0,     // Chain hasn't terminated
    ACTIVATION_DECAY,      // |activation.core| < threshold (signal too weak)
    TRUST_TOO_LOW,         // Geometric mean of step_trusts < threshold
    NO_VIABLE_CANDIDATES,  // All candidates rejected or score < 0
    COHERENCE_GATE,        // ChainKAN coherence below threshold
    SEED_DRIFT,            // Too many steps with low seed similarity
    MAX_STEPS_REACHED,     // Hit step limit
};

inline const char* termination_reason_to_string(TerminationReason reason) {
    switch (reason) {
        case TerminationReason::STILL_RUNNING:        return "still_running";
        case TerminationReason::ACTIVATION_DECAY:     return "activation_decay";
        case TerminationReason::TRUST_TOO_LOW:        return "trust_too_low";
        case TerminationReason::NO_VIABLE_CANDIDATES: return "no_viable_candidates";
        case TerminationReason::COHERENCE_GATE:       return "coherence_gate";
        case TerminationReason::SEED_DRIFT:           return "seed_drift";
        case TerminationReason::MAX_STEPS_REACHED:    return "max_steps_reached";
    }
    return "unknown";
}

// =============================================================================
// Activation --- The signal vector flowing through the graph
// =============================================================================
//
// Encapsulates the 16D core signal (transformed by ConceptModel W) plus
// variable-dim detail passthrough. Phase 1: detail passes through unchanged.
// Phase 2 (future): low-rank adapters for detail dimensions.
//

class Activation {
public:
    Activation() { core_.fill(0.0); }

    // Factory: create from FlexEmbedding
    static Activation from_embedding(const FlexEmbedding& emb) {
        Activation act;
        act.core_ = emb.core;
        act.detail_ = emb.detail;
        return act;
    }

    // Convert back to FlexEmbedding
    FlexEmbedding to_embedding() const {
        FlexEmbedding emb;
        emb.core = core_;
        emb.detail = detail_;
        return emb;
    }

    // Dimension info
    size_t dim() const { return CORE_DIM + detail_.size(); }
    size_t core_dim() const { return CORE_DIM; }
    size_t detail_dim() const { return detail_.size(); }

    // Core access
    const CoreVec& core() const { return core_; }
    CoreVec& core_mut() { return core_; }

    // Detail access (passthrough in Phase 1)
    const std::vector<double>& detail() const { return detail_; }
    std::vector<double>& detail_mut() { return detail_; }

    // Core magnitude: ||core||
    double core_magnitude() const {
        double sum = 0.0;
        for (size_t i = 0; i < CORE_DIM; ++i)
            sum += core_[i] * core_[i];
        return std::sqrt(sum);
    }

    // Cosine similarity between this activation's core and another's
    double core_cosine(const Activation& other) const {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < CORE_DIM; ++i) {
            dot += core_[i] * other.core_[i];
            na  += core_[i] * core_[i];
            nb  += other.core_[i] * other.core_[i];
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        return denom > 1e-12 ? dot / denom : 0.0;
    }

    // Top-K contributing dimensions (by absolute value)
    std::vector<size_t> top_k_dims(size_t k) const {
        std::vector<std::pair<double, size_t>> dims;
        dims.reserve(CORE_DIM);
        for (size_t i = 0; i < CORE_DIM; ++i)
            dims.push_back({std::abs(core_[i]), i});
        std::sort(dims.begin(), dims.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<size_t> result;
        for (size_t i = 0; i < std::min(k, dims.size()); ++i)
            result.push_back(dims[i].second);
        return result;
    }

    // Element-wise access
    double operator[](size_t i) const {
        return i < CORE_DIM ? core_[i] : detail_[i - CORE_DIM];
    }

private:
    CoreVec core_;
    std::vector<double> detail_;
};

// =============================================================================
// EdgeResult --- Result of forward_edge() transformation
// =============================================================================
//
// Carries the full output of passing an activation through a concept's
// ConceptModel along a specific relation edge.
//

struct EdgeResult {
    Activation output;             // Transformed activation (after KAN gating)
    CoreVec dimensional_contrib{}; // v = W*c + b (which dimensions drive the decision)
    double transform_quality = 0.0; // |output| / |input| (magnitude preservation)
    double coherence = 0.0;         // cosine(output.core, target_embedding)
    double epistemic_alignment = 0.0; // Trust compatibility source -> target
    double composite_score = 0.0;    // Weighted combination of all scores

    // Dual neuron tracking (KAN = reasoning, NN = pattern matching)
    double nn_quality = 0.0;       // NN path: how well W*x+b transformed the signal
    double kan_quality = 0.0;      // KAN path: FlexKAN gate value (approval of transform)
    double kan_gate = 1.0;         // KAN gating factor applied to NN output
};

// =============================================================================
// GraphReasonerConfig --- Configuration for GraphReasoner
// =============================================================================

struct GraphReasonerConfig {
    // Chain limits
    size_t max_steps = 12;
    size_t max_candidates = 50;

    // Activation thresholds
    double min_activation_magnitude = 0.05;  // Signal too weak to continue

    // Scoring weights (must sum to 1.0)
    double weight_transform_quality = 0.35;
    double weight_coherence = 0.35;
    double weight_epistemic_alignment = 0.15;
    double weight_relation = 0.15;

    // Trust thresholds
    double min_chain_trust = 0.01;        // Geometric mean of step_trusts
    double min_composite_score = 0.05;    // Minimum score for a candidate

    // Focus gate (from ConceptReasoner, reused)
    double focus_gate_weight = 0.2;
    double focus_min_gate = 0.1;

    // ChainKAN coherence
    double chain_coherence_weight = 0.3;
    double min_coherence_gate = 0.25;
    bool enable_composition = true;
    double chain_ctx_blend = 0.15;

    // Seed anchor
    double seed_anchor_weight = 0.15;
    double seed_anchor_decay = 0.08;
    double min_seed_similarity = 0.15;
    size_t max_consecutive_seed_drops = 3;

    // Relation scoring
    double relation_weight_power = 0.5;
    double diversity_bonus = 0.05;
    double incoming_discount = 0.7;

    // Context EMA
    double context_alpha = 0.3;

    // Step trust weights
    double step_trust_source_weight = 0.3;
    double step_trust_edge_weight = 0.3;
    double step_trust_target_weight = 0.2;
    double step_trust_transform_weight = 0.2;

    // Semantic gates (anti-drift)
    double min_embedding_similarity = 0.05;   // Min cosine(source_emb, target_emb)
    double min_topic_similarity = 0.0;        // Min cosine(chain_centroid, target_emb)
    double topic_centroid_alpha = 0.3;        // EMA weight for topic centroid update
    double embedding_sim_weight = 0.1;        // Weight of embedding similarity in composite

    // Top-K dimensions to record in trace
    size_t trace_top_k_dims = 5;

    // Recursive feedback configuration
    struct FeedbackConfig {
        bool enable = false;                    // Off by default (backward-compatible)
        size_t max_rounds = 3;                  // Maximum feedback iterations
        double quality_skip_threshold = 0.8;    // Skip feedback if first chain is already great
        double improvement_threshold = 0.02;    // Stop if quality improves less than this
        double context_blend_alpha = 0.4;       // How much prior discoveries influence new chain
    };
    FeedbackConfig feedback;
};

// =============================================================================
// FeedbackState --- Internal state passed between feedback rounds
// =============================================================================
//
// Used by reason_with_feedback() to carry context across adaptive rounds.
// enriched_context: centroid of high-quality step embeddings from prior chain.
// refined_topic: final topic direction from prior chain.
// explored: accumulated visited set forcing different paths each round.
//

struct FeedbackState {
    CoreVec enriched_context{};             // Centroid of high-quality step embeddings
    CoreVec refined_topic{};                // Final topic_centroid from prior chain
    std::unordered_set<ConceptId> explored; // All concepts visited across all rounds
    double best_quality = 0.0;              // Best chain quality so far
    size_t round = 0;                       // Current round number
};

// =============================================================================
// Co-Learning Signal Types --- Feedback for Graph <-> CM loop
// =============================================================================
//
// The GraphReasoner produces learning signals from reasoning chains:
//   - EdgeSignal: per-edge quality signal (positive or negative)
//   - ChainSignal: chain-level feedback with edge signals
//
// Usage in Co-Learning loop:
//   1. Reason through graph -> get GraphChain
//   2. Extract signals -> get ChainSignal
//   3. Positive edges: increase weight in graph
//   4. Negative edges (pain signal): decrease weight or prune
//   5. Retrain CMs on updated graph
//   6. Repeat
//

struct EdgeSignal {
    ConceptId source = 0;
    ConceptId target = 0;
    RelationType relation = RelationType::CUSTOM;

    // Quality metrics from forward_edge
    double transform_quality = 0.0;    // How well W*a preserved the signal
    double coherence = 0.0;            // How well output aligned with target embedding
    double epistemic_alignment = 0.0;  // Trust compatibility
    double composite_score = 0.0;      // Final weighted score

    // Dual neuron quality (separate learning signals)
    double nn_quality = 0.0;           // NN path quality (W*x+b transform)
    double kan_quality = 0.0;          // KAN path quality (reasoning gate)
    double kan_gate = 1.0;             // KAN gating factor applied

    // Embedding similarity (semantic relatedness)
    double embedding_similarity = 0.0;

    // Signal classification
    bool was_traversed = false;        // True if this edge was chosen in the chain
    bool is_positive = false;          // True if quality > threshold
    std::string rejection_reason;      // If rejected, why

    // Pain signal strength: higher = more evidence this edge is wrong
    // 0 = confirmed good, 1 = maximally painful
    double pain() const {
        if (is_positive) return 0.0;
        double quality = 0.5 * transform_quality + 0.3 * coherence + 0.2 * embedding_similarity;
        return std::max(0.0, 1.0 - quality);
    }

    // Separate pain signals for dual neuron training
    double nn_pain() const { return is_positive ? 0.0 : std::max(0.0, 1.0 - nn_quality); }
    double kan_pain() const { return is_positive ? 0.0 : std::max(0.0, 1.0 - kan_quality); }

    // Reward signal: higher = more evidence this edge is good
    double reward() const {
        if (!is_positive) return 0.0;
        return 0.4 * transform_quality + 0.3 * coherence
             + 0.2 * epistemic_alignment + 0.1 * embedding_similarity;
    }
};

struct ChainSignal {
    ConceptId seed = 0;
    TerminationReason termination = TerminationReason::STILL_RUNNING;
    double chain_quality = 0.0;

    // Per-edge signals (traversed + rejected alternatives)
    std::vector<EdgeSignal> traversed_edges;    // Edges actually taken
    std::vector<EdgeSignal> rejected_edges;     // Top rejected alternatives

    // Chain-level pain: high if chain terminated badly
    double chain_pain() const {
        if (termination == TerminationReason::MAX_STEPS_REACHED) return 0.0;
        if (termination == TerminationReason::SEED_DRIFT) return 0.4;
        if (termination == TerminationReason::NO_VIABLE_CANDIDATES) return 0.7;
        if (termination == TerminationReason::ACTIVATION_DECAY) return 0.5;
        if (termination == TerminationReason::TRUST_TOO_LOW) return 0.6;
        if (termination == TerminationReason::COHERENCE_GATE) return 0.3;
        return 0.0;
    }

    // Suggested edge modifications
    struct EdgeSuggestion {
        ConceptId source = 0;
        ConceptId target = 0;
        RelationType relation = RelationType::CUSTOM;
        double delta_weight = 0.0;     // Positive = strengthen, negative = weaken
        std::string reason;
    };
    std::vector<EdgeSuggestion> suggestions;
};

} // namespace brain19

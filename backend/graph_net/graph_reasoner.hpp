#pragma once

#include "types.hpp"
#include "epistemic_trace.hpp"
#include "reasoning_logger.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../reasoning/chain_kan.hpp"
#include "../convergence/convergence_config.hpp"

#include <array>
#include <unordered_set>
#include <vector>

namespace brain19 {

// =============================================================================
// GraphReasoner --- Graph als Neuronales Netz
// =============================================================================
//
// The Knowledge Graph IS the neural network. Traversal = Forward Pass.
//
// Each concept's ConceptModel W(16x16) transforms activation vectors.
// Each edge applies relation-modulated dimensional transformation.
// Full activation vectors flow through the graph.
//
// Key difference from ConceptReasoner:
//   ConceptReasoner collapses to scalars: predict() -> scalar -> pick best.
//   GraphReasoner keeps full 16D activations: forward_edge() -> Activation.
//
// OOP Design:
//   GraphReasoner is a concrete class implementing graph-based reasoning.
//   It has a clear public interface (reason_from, explain) and encapsulated
//   private implementation (forward_edge, evaluate_candidates, build_trace_step).
//
//   Extension point: future MemoryAwareReasoner can inherit and override
//   evaluate_candidates() to incorporate episodic memory matching.
//
// Epistemic Nachvollziehbarkeit:
//   Every step is fully documented in TraceStep. The GraphChain::explain()
//   method produces a human-readable audit trail.
//

class GraphReasoner {
public:
    // --- Nested types (public for extensibility) ---

    // Focus entry: concept + its embedding for focus gate evaluation
    struct FocusEntry {
        ConceptId id = 0;
        FlexEmbedding emb;
    };

    // Scored candidate from evaluate_candidates
    struct ScoredCandidate {
        ConceptId target = 0;
        RelationType relation = RelationType::CUSTOM;
        double composite_score = 0.0;
        bool outgoing = true;
        bool is_causal = false;
        EdgeResult edge_result;
        std::string rejection_reason;
        std::array<double, convergence::OUTPUT_DIM> simulated_state{};
        double chain_coherence = 0.0;
        double embedding_similarity = 0.0;  // Cosine(source_emb, target_emb)
    };

    // --- Construction ---

    GraphReasoner(const LongTermMemory& ltm,
                  ConceptModelRegistry& registry,
                  EmbeddingManager& embeddings,
                  GraphReasonerConfig config = {});

    // --- Public interface ---

    // Reason from a single seed concept
    GraphChain reason_from(ConceptId seed) const;

    // Reason from multiple seeds, return best chain by chain quality
    GraphChain reason_from(const std::vector<ConceptId>& seeds) const;

    // Adaptive multi-round reasoning with recursive feedback
    GraphChain reason_with_feedback(ConceptId seed) const;
    GraphChain reason_with_feedback(const std::vector<ConceptId>& seeds) const;

    // Compute holistic chain quality metric
    double compute_chain_quality(const GraphChain& chain) const;

    // Access ChainKAN (for external training / inspection)
    ChainKAN& chain_kan() { return chain_kan_; }
    const ChainKAN& chain_kan() const { return chain_kan_; }

    // Access configuration
    const GraphReasonerConfig& config() const { return config_; }

    // Optional JSONL logger for orchestrator training data
    void set_logger(ReasoningLogger* logger);

    // --- Co-Learning API (Graph <-> CM feedback loop) ---

    // Extract learning signals from a completed chain.
    // Positive signals for traversed edges, negative for rejected/painful ones.
    ChainSignal extract_signals(const GraphChain& chain) const;

    // Evaluate a specific edge's quality (for graph pruning/strengthening).
    // Returns EdgeSignal with transform_quality, coherence, embedding_similarity.
    EdgeSignal evaluate_edge(ConceptId source, ConceptId target,
                             RelationType relation) const;

protected:
    // --- Core transformation: the heart of graph reasoning ---

    // Forward-pass through one edge: W*activation + b, relation modulation, tanh
    EdgeResult forward_edge(ConceptId source, ConceptId target,
                            RelationType rel_type,
                            const Activation& input,
                            const FlexEmbedding& ctx) const;

    // Evaluate all candidates from current position
    std::vector<ScoredCandidate> evaluate_candidates(
        ConceptId current,
        const Activation& activation,
        const FlexEmbedding& ctx,
        const std::vector<FocusEntry>& focus_stack,
        const std::unordered_set<ConceptId>& visited,
        const std::unordered_set<uint16_t>& used_rels,
        const std::array<double, convergence::OUTPUT_DIM>& chain_state,
        ConceptId seed_id, size_t step_index,
        const FlexEmbedding& topic_centroid = FlexEmbedding{}) const;

    // Build a complete trace step from a scored candidate
    TraceStep build_trace_step(
        ConceptId source, const ScoredCandidate& winner,
        const Activation& input_act,
        const std::vector<ScoredCandidate>& all_candidates,
        size_t step_index,
        const std::array<double, convergence::OUTPUT_DIM>& chain_state) const;

    // --- Helper methods ---

    // Epistemic alignment between source and target trust/type
    static double compute_epistemic_alignment(
        double source_trust, double target_trust,
        EpistemicType source_type, EpistemicType target_type);

    // Edge confidence from ConceptModel training quality
    double edge_confidence_from_model(ConceptId source_id) const;

    // Is this relation causal (shifts focus)?
    static bool is_causal_relation(RelationType type);

    // Relation reasoning weight
    static double relation_reasoning_weight(RelationType type);

    // EMA context update with chain state feedback
    FlexEmbedding update_context(const FlexEmbedding& ctx, ConceptId new_concept,
                                  const std::array<double, convergence::OUTPUT_DIM>& chain_state,
                                  const CoreVec& dim_score = CoreVec{}) const;

    // Epistemic type rank (for alignment computation)
    static int epistemic_type_rank(EpistemicType type);

private:
    // Refactored core: accepts optional feedback priming
    GraphChain reason_from_internal(ConceptId seed,
                                     const FeedbackState* feedback) const;

    // Extract feedback from a completed chain
    FeedbackState extract_feedback(const GraphChain& chain,
                                    const FeedbackState* prior) const;

    const LongTermMemory& ltm_;
    ConceptModelRegistry& registry_;
    EmbeddingManager& embeddings_;
    GraphReasonerConfig config_;
    mutable ChainKAN chain_kan_;
    ReasoningLogger* logger_ = nullptr;

    // Chain state -> context projection (OUTPUT_DIM -> CORE_DIM)
    std::array<double, convergence::OUTPUT_DIM * CORE_DIM> chain_ctx_proj_W_{};
    std::array<double, 16> chain_ctx_proj_b_{};

    void initialize_chain_ctx_projection();
    CoreVec project_chain_to_core(const std::array<double, convergence::OUTPUT_DIM>& chain_state) const;

    // Build 90D features for ConvergencePort
    std::array<double, convergence::QUERY_DIM> build_composition_features(
        ConceptId current, ConceptId target, RelationType rel,
        const FlexEmbedding& ctx, const PredictFeatures& pf) const;

    // Forward chain state through ConvergencePort
    std::array<double, convergence::OUTPUT_DIM> forward_chain_state(
        ConceptId concept_id,
        const std::array<double, convergence::QUERY_DIM>& features,
        const std::array<double, convergence::OUTPUT_DIM>& prev_state) const;
};

} // namespace brain19

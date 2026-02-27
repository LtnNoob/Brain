#pragma once

#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "chain_kan.hpp"
#include <array>
#include <vector>
#include <unordered_set>

namespace brain19 {

// =============================================================================
// CONCEPT REASONER — Focus-Guided Distributed Reasoning + CM Composition
// =============================================================================
//
// Each concept's ConceptModel acts as a local reasoning expert.
// A shifting focus concept guides traversal: the focus CM gates candidates,
// preventing drift while allowing logical cross-domain reasoning.
//
// CM Composition (via ConvergencePort):
//   Each CM has a 122→32 ConvergencePort. By chaining these ports, a 32D hidden
//   state flows through the reasoning chain like an RNN. A shared ChainKAN
//   evaluates chain coherence from consecutive hidden states.
//
//   Chain state feedback: the 32D chain state is projected to 16D and mixed
//   into the EMA context, closing the loop so composition influences local scoring.
//
// Focus shift rule:
//   - Causal edges (CAUSES, ENABLES, REQUIRES, PRODUCES) → focus shifts to target
//   - Non-causal edges → focus stays, focus CM must approve candidate
//

struct ReasonerConfig {
    size_t max_steps = 12;
    double min_confidence = 0.15;       // stop if best score < this
    double context_alpha = 0.3;         // EMA: h = alpha*emb(new) + (1-alpha)*h
    double relation_weight_power = 0.5; // score *= pow(rel.weight, this)
    double diversity_bonus = 0.05;      // bonus for unexplored relation types
    double incoming_discount = 0.7;     // score multiplier for reverse edges
    size_t max_candidates = 50;
    double focus_gate_weight = 0.2;     // blend weight for focus CM score on non-causal edges
    double focus_min_gate = 0.1;        // hard cutoff: focus CM must score above this

    // CM Composition settings
    double chain_coherence_weight = 0.3;  // blend weight for ChainKAN score
    bool enable_composition = true;       // toggle ConvergencePort composition
    double chain_ctx_blend = 0.15;        // blend weight for chain state → context feedback

    // Seed-anchor: penalize candidates far from seed topic
    double seed_anchor_weight = 0.15;     // penalty weight for low seed similarity
    double seed_anchor_decay = 0.08;      // per-step decay of anchor strength

    // Coherence-gated termination
    double min_coherence_gate = 0.25;     // stop chain if coherence drops below this

    // Chain validation: best-prefix truncation
    bool enable_chain_validation = true;
    double min_seed_similarity = 0.15;     // early-stop if seed_sim below this
    size_t max_consecutive_drops = 3;      // early-stop after N steps below threshold
};

struct ReasoningStep {
    ConceptId concept_id = 0;
    RelationType relation_type = RelationType::CUSTOM;
    double confidence = 0.0;
    bool is_outgoing = true;
    bool focus_shifted = false;         // did focus shift at this step?

    // CM Composition state
    std::array<double, 32> chain_state{};   // 32D hidden state at this step
    double coherence_score = 0.0;           // ChainKAN output (0 if composition disabled)
    double seed_similarity = 0.0;           // cosine sim to seed embedding (topic relevance)

    // Hard negative: runner-up candidate's simulated chain state
    std::array<double, 32> runner_up_state{};  // 2nd-best candidate's chain state
    bool has_runner_up = false;
};

struct ReasoningChain {
    std::vector<ReasoningStep> steps;
    double avg_confidence = 0.0;

    std::vector<ConceptId> concept_sequence() const;
    std::vector<RelationType> relation_sequence() const;
    bool empty() const { return steps.empty(); }
};

// =============================================================================
// Chain Training — contrastive + chain-terminal quality signal
// =============================================================================

struct ChainTrainingConfig {
    double learning_rate = 0.01;
    size_t kan_epochs = 50;
    size_t convergence_epochs = 5;  // backward_convergence fine-tuning epochs
    double convergence_lr = 0.001;  // convergence port learning rate (smaller!)
    double good_chain_threshold = 0.6;   // top percentile considered "good"
    double bad_chain_threshold = 0.3;    // bottom percentile considered "bad"
};

struct ChainTrainingResult {
    double initial_kan_loss = 0.0;
    double final_kan_loss = 0.0;
    size_t samples_collected = 0;
    size_t chains_used = 0;
    size_t convergence_ports_updated = 0;
};

class ConceptReasoner {
public:
    ConceptReasoner(const LongTermMemory& ltm,
                    ConceptModelRegistry& registry,
                    EmbeddingManager& embeddings,
                    ReasonerConfig config = {});

    // Reason from a single seed concept
    ReasoningChain reason_from(ConceptId seed) const;

    // Reason from multiple seeds, return best chain by avg_confidence
    ReasoningChain reason_from(const std::vector<ConceptId>& seeds) const;

    // Access ChainKAN for external training
    ChainKAN& chain_kan() { return chain_kan_; }
    const ChainKAN& chain_kan() const { return chain_kan_; }

    // Chain-terminal quality metric for a completed chain
    double compute_chain_quality(const ReasoningChain& chain) const;

    // Train ChainKAN + ConvergencePorts from collected chains
    // Uses chain-terminal quality + contrastive negative sampling
    ChainTrainingResult train_composition(
        const std::vector<ReasoningChain>& chains,
        const ChainTrainingConfig& tcfg = {});

    // Access chain→context projection (for inspection/training)
    const std::array<double, 32 * 16>& chain_ctx_proj_W() const { return chain_ctx_proj_W_; }
    const std::array<double, 16>& chain_ctx_proj_b() const { return chain_ctx_proj_b_; }

private:
    const LongTermMemory& ltm_;
    ConceptModelRegistry& registry_;
    EmbeddingManager& embeddings_;
    ReasonerConfig config_;
    mutable ChainKAN chain_kan_;  // mutable: training happens externally

    // Chain state → context projection (32→16)
    std::array<double, 32 * 16> chain_ctx_proj_W_{};  // 512 params
    std::array<double, 16> chain_ctx_proj_b_{};

    // Score one edge using source concept's ConceptModel
    double score_edge(ConceptId source, ConceptId target,
                      RelationType type, const FlexEmbedding& ctx) const;

    struct ScoredCandidate {
        ConceptId target;
        RelationType rel;
        double score;
        bool outgoing;
        bool is_causal;  // would this step shift focus?
        std::array<double, 32> simulated_state{};  // chain state if this candidate is picked
        double coherence = 0.0;                     // ChainKAN coherence score
        CoreVec dimensional_score{};                // per-dim CM activation for this edge
    };

    // Focus entry: concept + its embedding
    struct FocusEntry {
        ConceptId id;
        FlexEmbedding emb;
    };

    // Score candidates, applying focus gate + seed anchor + composition
    std::vector<ScoredCandidate> score_candidates(
        ConceptId current, const std::vector<FocusEntry>& focus_stack,
        const FlexEmbedding& ctx,
        const std::unordered_set<ConceptId>& visited,
        const std::unordered_set<uint16_t>& used_rels,
        const std::array<double, 32>& chain_state,
        ConceptId seed_id, size_t step_index) const;

    // EMA context update (with chain state feedback when composition enabled)
    FlexEmbedding update_context(const FlexEmbedding& ctx, ConceptId new_concept,
                                  const std::array<double, 32>& chain_state,
                                  const CoreVec& dimensional_score = CoreVec{}) const;

    // Project 32D chain state to 16D core space
    CoreVec project_chain_to_core(const std::array<double, 32>& chain_state) const;

    // Is this a focus-shifting (causal/logical) relation?
    static bool is_causal_relation(RelationType type);

    // Relation type weight for scoring
    static double relation_reasoning_weight(RelationType type);

    // === CM Composition helpers ===

    // Build 90D feature block for ConvergencePort input
    std::array<double, 90> build_composition_features(
        ConceptId current, ConceptId target, RelationType rel,
        const FlexEmbedding& ctx, const PredictFeatures& pf) const;

    // Forward chain state through CM's ConvergencePort
    std::array<double, 32> forward_chain_state(
        ConceptId concept_id,
        const std::array<double, 90>& features,
        const std::array<double, 32>& prev_state) const;

    // Initialize the chain→context projection weights
    void initialize_chain_ctx_projection();
};

} // namespace brain19

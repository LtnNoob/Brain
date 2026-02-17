#pragma once

#include "kan_language_engine.hpp"
#include "language_config.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../cuda/cuda_training.h"
#include <memory>
#include <string>
#include <vector>

namespace brain19 {

// =============================================================================
// TRAINING EXAMPLE — for fusion training (Stage 2)
// =============================================================================

struct LanguageTrainingExample {
    std::string query;
    std::vector<ConceptId> expected_chain;
    std::string expected_answer;
};

// =============================================================================
// TRAINING RESULT
// =============================================================================

struct LanguageTrainingResult {
    size_t stage;
    size_t epochs_run;
    double final_loss;
    bool converged;
    std::string stage_name;
};

// =============================================================================
// LANGUAGE TRAINING — Multi-stage training pipeline
// =============================================================================
//
// Stage 1: Encoder + Decoder (isolated, from LTM data)
//   - Encoder: (text, concept_embedding) pairs → MSE loss
//   - Decoder: (concept_embedding, text) pairs → cross-entropy loss
//
// Stage 2: Fusion + Scorer (ConceptModels frozen)
//   - Semantic scorers + FusionLayer + Decoder fine-tuning
//   - From QA pairs with expected causal chains
//
// Stage 3: Joint fine-tuning (optional, ConceptModels with 1/10 LR)
//   - Epistemic integrity guard: max 10% relative change per model
//

class LanguageTraining {
public:
    explicit LanguageTraining(KANLanguageEngine& engine, LongTermMemory& ltm,
                              ConceptModelRegistry& registry);

    // Run Stage 1: Train encoder and decoder from LTM data
    LanguageTrainingResult train_stage1(const LanguageConfig& config);

    // Run Stage 2: Train fusion from QA pairs
    LanguageTrainingResult train_stage2(
        const std::vector<LanguageTrainingExample>& examples,
        const LanguageConfig& config
    );

    // Run all stages
    std::vector<LanguageTrainingResult> train_all(
        const std::vector<LanguageTrainingExample>& qa_pairs,
        const LanguageConfig& config
    );

    // Generate training data from LTM (for Stage 1)
    struct EncoderPair {
        std::string text;
        std::vector<double> target_embedding;
    };
    std::vector<EncoderPair> generate_encoder_data() const;

    struct DecoderPair {
        std::vector<double> embedding;
        std::string target_text;
    };
    std::vector<DecoderPair> generate_decoder_data() const;

    // Concept decoder training data: {fused_vector, target_concept_ids, trust_weight}
    struct ConceptDecoderPair {
        std::vector<double> embedding;               // extended fused vector
        std::vector<ConceptId> target_concepts;       // ordered target concept sequence
        double trust_weight;                          // epistemic trust as sample weight
        ConceptId source_concept;                     // for debugging
    };
    std::vector<ConceptDecoderPair> generate_concept_decoder_data() const;

    // Build fused embedding vector for a source concept with specific target/relation embeddings
    std::vector<double> build_concept_fused_vector(
        ConceptId source,
        const std::vector<FlexEmbedding>& target_embeddings,
        const std::vector<FlexEmbedding>& rel_type_embeddings) const;

    // Generate relation-based decoder data: one compound paragraph per concept
    // combining ALL outgoing relations into 15-30 token training targets.
    // Input vectors use FusionLayer-projected concept embeddings (R^64).
    std::vector<DecoderPair> generate_relation_decoder_data() const;

    // Deep KAN training path (V12: 2-layer EfficientKAN feature extractor + linear output)
    LanguageTrainingResult train_stage1_deep_kan(const LanguageConfig& config);

    // Deep KAN v2: LibTorch with CM-Feedback-Port (KAN↔CM bidirectional coupling)
    LanguageTrainingResult train_stage1_deep_kan_v2(const LanguageConfig& config);

    // Generate text using trained DeepKAN v2 model (call after train_stage1_deep_kan_v2)
    std::string generate_v2(const std::string& query, size_t max_tokens = 30) const;
    bool has_v2_model() const { return v2_valid_; }

private:
    // Stored state from last DeepKAN v2 training (for inference)
    bool v2_valid_ = false;
    cuda::DeepKANWeights v2_dkw_;
    std::vector<double> v2_emb_table_;
    std::vector<double> v2_flex_table_;
    std::vector<uint16_t> v2_active_tokens_;
    size_t v2_VA_ = 0, v2_V_ = 0;
    size_t v2_FUSED_BASE_ = 64, v2_flex_dim_ = 16;
#ifdef USE_LIBTORCH
    // Forward-declared to avoid libtorch include in header
    struct V2ConvergenceState;
    std::shared_ptr<V2ConvergenceState> v2_cpd_;

    // Concept prediction state (LibTorch)
    struct V2ConceptState;
    std::shared_ptr<V2ConceptState> v2_concept_state_;
#endif
    bool v2_concept_valid_ = false;
    std::vector<double> v2_concept_matrix_;      // [NC * 16]
    std::vector<double> v2_concept_emb_64d_;     // [NC * 64]
    std::vector<double> v2_concept_flex_16d_;    // [NC * 16]
    size_t v2_num_concepts_ = 0;
    std::vector<ConceptId> v2_idx_to_concept_;

    KANLanguageEngine& engine_;
    LongTermMemory& ltm_;
    ConceptModelRegistry& registry_;

    // Concept decoder: closed-form ridge regression for concept projection W
    // Solves: W = (H^T H + λI)^{-1} H^T E where E is [N × 16] target embeddings
    void train_concept_decoder_closedform(const std::vector<ConceptDecoderPair>& data, double lambda);

    // Concept decoder: SGD epoch with trust-weighted cross-entropy
    double train_concept_decoder_epoch(const std::vector<ConceptDecoderPair>& data, double lr);

    // Train encoder on (text → embedding) pairs
    double train_encoder_epoch(const std::vector<EncoderPair>& data, double lr);

    // Closed-form ridge regression for output projection W
    // Solves: W = argmin_W Σ ||W^T · h_ext - one_hot(target)||² + λ||W||²
    // One-shot optimal solution in ~50ms vs iterative SGD.
    void train_decoder_closedform(const std::vector<DecoderPair>& data, double lambda);
};

} // namespace brain19

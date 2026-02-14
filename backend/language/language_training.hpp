#pragma once

#include "kan_language_engine.hpp"
#include "language_config.hpp"
#include "../ltm/long_term_memory.hpp"
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
    explicit LanguageTraining(KANLanguageEngine& engine, LongTermMemory& ltm);

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

    // Generate relation-based decoder data: one compound paragraph per concept
    // combining ALL outgoing relations into 15-30 token training targets.
    // Input vectors use FusionLayer-projected concept embeddings (R^64).
    std::vector<DecoderPair> generate_relation_decoder_data() const;

private:
    KANLanguageEngine& engine_;
    LongTermMemory& ltm_;

    // Train encoder on (text → embedding) pairs
    double train_encoder_epoch(const std::vector<EncoderPair>& data, double lr);

    // Train decoder with Teacher-Forcing + Cross-Entropy + Target Propagation
    //
    // For each (fused_vector, target_token_sequence):
    //   1. Forward pass with teacher forcing → collect hidden states + logits
    //   2. Cross-Entropy loss at each step
    //   3. Gradient dL/dh via output projection → "better" hidden states
    //   4. Output projection: direct gradient descent
    //   5. Init-KAN + Update-KAN: train on (input, target_h) pairs via MSE
    double train_decoder_epoch(const std::vector<DecoderPair>& data, double lr);

    // Closed-form ridge regression for output projection W
    // Solves: W = argmin_W Σ ||W^T · h_ext - one_hot(target)||² + λ||W||²
    // One-shot optimal solution in ~50ms vs iterative SGD.
    void train_decoder_closedform(const std::vector<DecoderPair>& data, double lambda);

    // Softmax helper
    static std::vector<double> softmax(const std::vector<double>& logits);
};

} // namespace brain19

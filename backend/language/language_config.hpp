#pragma once

#include <cstddef>
#include <cstdint>

namespace brain19 {

// =============================================================================
// LANGUAGE ENGINE CONFIGURATION
// =============================================================================
//
// Constants and configuration for the KAN-MiniLLM Hybrid Language Engine.
// ~1M parameters total: 52% token embeddings, 27% ConceptModels, 17% decoder.
//

struct LanguageConfig {
    // ── Tokenizer ──
    static constexpr size_t VOCAB_SIZE         = 8192;
    static constexpr size_t TOKEN_EMBED_DIM    = 64;
    static constexpr size_t MAX_SEQ_LEN        = 128;
    static constexpr uint16_t PAD_TOKEN        = 0;
    static constexpr uint16_t BOS_TOKEN        = 1;
    static constexpr uint16_t EOS_TOKEN        = 2;
    static constexpr uint16_t UNK_TOKEN        = 3;
    static constexpr uint16_t SEP_TOKEN        = 4;
    static constexpr uint16_t CONCEPT_TOKEN_START = 5;
    static constexpr uint16_t CONCEPT_TOKEN_END   = 1004;  // up to 1000 concepts
    static constexpr uint16_t BPE_TOKEN_START     = 1005;

    // ── KAN Encoder ── (64→32→16→16)
    static constexpr size_t ENCODER_QUERY_DIM  = 16;  // output: query embedding

    // ── Semantic Scorer ──
    // Relevance:  (16+16)=32 → 16 → 1
    // Causality:  (16+16+16)=48 → 16 → 1
    // Template:   16 → 8 → 4
    static constexpr size_t NUM_TEMPLATE_TYPES = 4;

    // ── Fusion Layer ──
    static constexpr size_t GATE_INPUT_DIM     = 3;   // [activation_norm, relevance, causality]
    static constexpr size_t FUSED_DIM          = 64;
    static constexpr size_t TOP_K_CONCEPTS     = 3;

    // ── KAN Decoder ──
    static constexpr size_t DECODER_HIDDEN_DIM = 16;
    // Init KAN: 64 → 16
    // Update KAN: (16+64)=80 → 32 → 16
    // Output projection: 16 → VOCAB_SIZE

    // ── Concept Prediction ──
    static constexpr size_t CONCEPT_EMBED_DIM    = 16;   // FlexEmbedding core dimension
    static constexpr size_t MAX_CONCEPT_SEQUENCE  = 10;   // max concepts per prediction
    // Separate temperatures for training vs inference:
    //   Training: T=0.1 produces sharp gradients (cosine sim /0.1 → logits in [-10,10])
    //     which helps the model commit to correct targets during learning.
    //   Inference: T=1.0 uses raw cosine similarity for smoother, more calibrated predictions.
    double concept_train_temperature     = 0.1;
    double concept_inference_temperature = 1.0;

    // ── Generation ──
    size_t max_tokens               = 30;
    double decoder_confidence_threshold = 0.15;  // below this → template fallback
    double concept_token_boost      = 2.0;       // boost for concept tokens in decoding

    // ── Training ──
    double encoder_lr               = 0.001;
    double decoder_lr               = 2.0;
    double fusion_lr                = 0.001;
    size_t encoder_epochs           = 200;
    size_t decoder_epochs           = 150;   // token prediction epochs
    size_t concept_epochs           = 300;   // concept prediction epochs (more classes → needs more)
    size_t fusion_epochs            = 300;

    // ── Interleaved Training (Bidirectional Feedback) ──
    size_t feedback_rounds            = 6;       // interleaved rounds
    size_t concept_epochs_per_round   = 50;      // concept epochs per round (6×50=300)
    size_t token_epochs_per_round     = 25;      // token epochs per round (6×25=150)

    // ── KAN → Graph Feedback ──
    size_t kan_feedback_sample_size   = 200;     // concepts to run inference on per round
    double kan_feedback_min_confidence = 0.7;    // min confidence to propose relation
    double kan_feedback_relation_weight = 0.25;  // weight for proposed ASSOCIATED_WITH relations
    size_t kan_feedback_max_relations = 50;      // max new relations per round

    // ── CM → Graph Feedback (Trust Adjustment) ──
    double cm_trust_adjustment_max    = 0.1;     // max trust delta per concept per round
    size_t cm_min_samples_for_adjust  = 3;       // need ≥3 predictions to judge a relation

    // ── Relation Decoder Training ──
    static constexpr size_t MAX_RELATION_DECODER_PAIRS = 20000;
    static constexpr size_t MAX_DEFINITION_DECODER_PAIRS = 20000;  // unified concept descriptions
    static constexpr size_t MAX_CONCEPT_DECODER_PAIRS = 500000;    // augmented concept prediction data

    // ── KAN Knots ──
    size_t kan_num_knots            = 10;

    // ── Deep KAN Decoder (V12) ──
    bool use_deep_kan               = false;
    double deep_kan_lr              = 0.01;   // LR for KAN spline + residual weights

    // ── Convergence Integration ──
    static constexpr size_t CONVERGENCE_DIM = 32;  // = ConvergencePort::OUTPUT_DIM
};

} // namespace brain19

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

    // ── Generation ──
    size_t max_tokens               = 30;
    double decoder_confidence_threshold = 0.15;  // below this → template fallback
    double concept_token_boost      = 2.0;       // boost for concept tokens in decoding

    // ── Training ──
    double encoder_lr               = 0.001;
    double decoder_lr               = 0.001;
    double fusion_lr                = 0.001;
    size_t encoder_epochs           = 200;
    size_t decoder_epochs           = 200;
    size_t fusion_epochs            = 300;

    // ── KAN Knots ──
    size_t kan_num_knots            = 10;
};

} // namespace brain19

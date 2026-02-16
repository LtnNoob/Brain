#pragma once

#include <cstddef>

namespace brain19 {

// =============================================================================
// CONVERGENCE PIPELINE — Dimension Constants & Hyperparameters
// =============================================================================
//
// Deep KAN ↔ ConceptModel Integration (v2 Architecture)
//
// Pipeline:
//   h ∈ ℝ^QUERY_DIM
//   → KAN Layer 1 → k1 ∈ ℝ^KAN_L1_OUT
//   → KAN Projection → k1_proj ∈ ℝ^KAN_PROJ_OUT (shared, for CM input)
//   → ConceptModels(h ⊕ k1_proj) → cm ∈ ℝ^CM_OUTPUT_DIM
//   → KAN Layer 2(k1 ⊕ cm) → k2 ∈ ℝ^KAN_L2_OUT
//   → KAN Layer 3(k2) → G(h) ∈ ℝ^OUTPUT_DIM
//   → Gated Residual PoE(h, G(h), L(h)) → fused ∈ ℝ^OUTPUT_DIM
//

namespace convergence {

// ─── Dimensions ──────────────────────────────────────────────────────────────

constexpr size_t QUERY_DIM       = 90;    // Input embedding dimension (from DeepKAN encoder)
constexpr size_t KAN_L1_OUT      = 256;   // KAN Layer 1 output
constexpr size_t KAN_PROJ_OUT    = 32;    // Shared projection for CM input (256→32)
constexpr size_t CM_OUTPUT_DIM   = 32;    // ConceptModel output per concept

// CM input = raw query (QUERY_DIM) + projected KAN L1 (KAN_PROJ_OUT)
constexpr size_t CM_INPUT_DIM    = QUERY_DIM + KAN_PROJ_OUT;  // 122

// KAN L2 input = KAN L1 output + CM aggregated output
constexpr size_t KAN_L2_IN       = KAN_L1_OUT + CM_OUTPUT_DIM; // 288
constexpr size_t KAN_L2_OUT      = 128;
constexpr size_t KAN_L3_OUT      = 32;    // Final output dimension

// Output dimension (same as KAN_L3_OUT and CM_OUTPUT_DIM)
constexpr size_t OUTPUT_DIM      = KAN_L3_OUT;  // 32

// ─── Router ──────────────────────────────────────────────────────────────────

constexpr size_t ROUTER_TOP_K    = 4;     // Number of active concepts per query
constexpr size_t ROUTER_DIM      = QUERY_DIM; // Centroids operate in query space

// ─── KAN Grid ────────────────────────────────────────────────────────────────

constexpr size_t KAN_L1_GRID     = 8;     // B-spline grid size for Layer 1
constexpr size_t KAN_L2_GRID     = 5;     // B-spline grid size for Layer 2
constexpr size_t KAN_L3_GRID     = 5;     // B-spline grid size for Layer 3
constexpr size_t SPLINE_ORDER    = 3;     // B-spline order

// ─── Gate ────────────────────────────────────────────────────────────────────

// Convergence gate: γ = σ(W_gate · h + b_gate)
// W_gate: [OUTPUT_DIM × QUERY_DIM] = 32×90 = 2880 params
// b_gate: [OUTPUT_DIM] = 32 params
// Total gate params: 2912
constexpr size_t GATE_PARAMS     = OUTPUT_DIM * QUERY_DIM + OUTPUT_DIM;

// ─── Ignition Thresholds ─────────────────────────────────────────────────────

constexpr float IGNITION_FAST       = 0.85f;  // agreement > 0.85 → skip gate, use G(h)
constexpr float IGNITION_DELIBERATE = 0.40f;  // agreement > 0.40 → 1 iteration with gate
// Below DELIBERATE → conflict: expand neighborhood, iterate up to MAX_CONFLICT_ITERS
constexpr int   MAX_CONFLICT_ITERS  = 2;

// ─── Training ────────────────────────────────────────────────────────────────

constexpr float DEFAULT_LR_KAN_L1    = 1e-5f;
constexpr float DEFAULT_LR_KAN_PROJ  = 5e-5f;
constexpr float DEFAULT_LR_KAN_L2L3  = 1e-5f;
constexpr float DEFAULT_LR_CM        = 5e-5f;
constexpr float DEFAULT_LR_ROUTER    = 1e-4f;
constexpr float DEFAULT_LR_GATE      = 1e-4f;

constexpr float GRAD_CLIP_KAN        = 0.5f;
constexpr float GRAD_CLIP_CM         = 1.0f;
constexpr float GRAD_CLIP_ROUTER     = 2.0f;
constexpr float GRAD_CLIP_GATE       = 2.0f;

// Regularization
constexpr float ENTROPY_REG_COEFF    = 0.1f;   // Anti gate collapse
constexpr float BALANCE_REG_COEFF    = 0.01f;  // Anti router collapse

// ─── VRAM Budget ─────────────────────────────────────────────────────────────

// Total estimated: ~87 MB
// KAN (3 layers + proj): ~3.1 MB
// ConceptBank: ~66 MB (25769 concepts × bilinear params)
// Router centroids: ~9.3 MB (25769 × 90 × 4 bytes)
// Gate: < 1 KB
// Working buffers: ~8 MB

} // namespace convergence
} // namespace brain19

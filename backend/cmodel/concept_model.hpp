#pragma once

#include "../micromodel/micro_model.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace brain19 {

static constexpr size_t CM_FLAT_SIZE = 1900;

// Pattern weights per concept (learned from validation feedback)
struct ConceptPatternWeights {
    double shared_parent = 1.0;
    double transitive_causation = 1.0;
    double missing_link = 1.0;
    double weak_strengthening = 1.0;
    double contradictory_signal = 1.0;
    double chain_hypothesis = 0.85;
    std::array<double, 9> reserved{};  // future patterns — total: 15 doubles
};

// Multi-Head Bilinear: K projection heads over concept embeddings
// s_i = dot(P_i * input_q, Q_i * input_k) for i=0..K-1
struct MultiHeadBilinear {
    static constexpr size_t K = 4;            // number of heads
    static constexpr size_t D_PROJ = 4;       // projection dimension per head
    static constexpr size_t INPUT_DIM = 20;   // 16 core + 4 cyclic-compressed detail
    static constexpr size_t PARAMS_PER_HEAD = 2 * D_PROJ * INPUT_DIM;  // P_i + Q_i = 160
    static constexpr size_t TOTAL_PARAMS = K * PARAMS_PER_HEAD;        // 640

    std::array<double, TOTAL_PARAMS> params{};

    void compute(const FlexEmbedding& e_q, const FlexEmbedding& e_k,
                 std::array<double, K>& scores) const;
    void initialize();  // zeros
};

// FlexKAN: lightweight [6,4,1] B-spline network per concept
// Input: [s_0, s_1, s_2, s_3, bilinear_score, dim_fraction] -> refined_score in (0,1)
struct FlexKAN {
    static constexpr size_t INPUT_DIM = 6;
    static constexpr size_t HIDDEN_DIM = 4;
    static constexpr size_t NUM_KNOTS = 10;
    static constexpr size_t LAYER0_EDGES = INPUT_DIM * HIDDEN_DIM;     // 24
    static constexpr size_t LAYER1_EDGES = HIDDEN_DIM * 1;             // 4
    static constexpr size_t TOTAL_EDGES = LAYER0_EDGES + LAYER1_EDGES; // 28
    static constexpr size_t TOTAL_PARAMS = TOTAL_EDGES * NUM_KNOTS;    // 280

    std::array<double, TOTAL_PARAMS> params{};

    double evaluate(const std::array<double, INPUT_DIM>& input) const;
    void train_step(const std::array<double, INPUT_DIM>& input,
                    double target, double learning_rate);
    void initialize_identity();
};

class ConceptModel {
public:
    ConceptModel();

    // === Edge Prediction ===
    // Fast bilinear (same as MicroModel): w = sigma(e^T * (W*c+b))
    double predict(const FlexEmbedding& e, const FlexEmbedding& c) const;
    // Full prediction: bilinear + multi-head concept features + KAN refinement
    double predict_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                           const FlexEmbedding& concept_from,
                           const FlexEmbedding& concept_to) const;
    // Backward-compat 2-arg (empty concept embeddings -> s_i=0 -> KAN identity -> bilinear)
    double predict_refined(const FlexEmbedding& e, const FlexEmbedding& c) const;

    // === Training ===
    double train_step(const FlexEmbedding& e, const FlexEmbedding& c,
                      double target, const MicroTrainingConfig& config);
    MicroTrainingResult train(const std::vector<TrainingSample>& samples,
                              const MicroTrainingConfig& config);
    // Train refined (multi-head + KAN) end-to-end via numerical gradient
    void train_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                       const FlexEmbedding& concept_from, const FlexEmbedding& concept_to,
                       double target, double learning_rate);
    // Legacy KAN training (backward compat)
    void train_kan(double bilinear_score, double ctx_feature,
                   double validated_target, double learning_rate);

    // === Pattern Weights ===
    ConceptPatternWeights& pattern_weights() { return patterns_; }
    const ConceptPatternWeights& pattern_weights() const { return patterns_; }

    // === Serialization (1900 doubles) ===
    void to_flat(std::array<double, CM_FLAT_SIZE>& out) const;
    void from_flat(const std::array<double, CM_FLAT_SIZE>& in);

    // === Bilinear internals (for compatibility) ===
    const CoreMat& weights() const { return W_; }
    const CoreVec& bias() const { return b_; }
    const CoreVec& e_init() const { return e_init_; }
    const CoreVec& c_init() const { return c_init_; }

private:
    // Bilinear core (940 doubles) — same math as MicroModel
    CoreMat W_;          // 256
    CoreVec b_;          // 16
    CoreVec e_init_;     // 16
    CoreVec c_init_;     // 16
    TrainingState state_; // 636

    // Multi-Head Bilinear projections (640 doubles)
    MultiHeadBilinear multihead_;

    // FlexKAN [6,4,1] (280 doubles)
    FlexKAN kan_;

    // Per-concept pattern weights (15 doubles)
    ConceptPatternWeights patterns_;

    // Reserved (25 doubles)
    std::array<double, 25> reserved_{};
};

} // namespace brain19

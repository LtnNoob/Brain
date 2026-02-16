#pragma once

#include "../micromodel/micro_model.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace brain19 {

static constexpr size_t CM_FLAT_SIZE_V5 = 1900;   // Pre-convergence layout
static constexpr size_t CM_FLAT_SIZE = 5836;       // V6: adds ConvergencePort (3936 params)

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
    void initialize();  // Xavier init (breaks symmetry between heads)
};

// Convergence Input Port: per-concept 122→32 linear+tanh expert
// (Convergence v2, Section 3: Deep KAN ↔ ConceptModel Integration)
// Input: h(90) ⊕ k1_proj(32) = 122 dims from convergence pipeline
// Output: 32 dims for local expert contribution
struct ConvergencePort {
    static constexpr size_t INPUT_DIM = 122;
    static constexpr size_t OUTPUT_DIM = 32;
    static constexpr size_t W_SIZE = OUTPUT_DIM * INPUT_DIM;  // 3904
    static constexpr size_t TOTAL_PARAMS = W_SIZE + OUTPUT_DIM; // 3936

    std::array<double, W_SIZE> W{};
    std::array<double, OUTPUT_DIM> b{};

    // Forward: output[i] = tanh(W[i] · input + b[i])
    void compute(const double* input, double* output) const;
    void initialize();  // Xavier init
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

// Adam optimizer state for refined training (MultiHead + KAN)
// NOT serialized — rebuilt each training cycle, passed externally for memory efficiency
struct RefinedAdamState {
    static constexpr size_t MH_PARAMS = MultiHeadBilinear::TOTAL_PARAMS;  // 640
    static constexpr size_t KAN_PARAMS = FlexKAN::TOTAL_PARAMS;           // 280
    static constexpr size_t TOTAL = MH_PARAMS + KAN_PARAMS;              // 920

    std::array<double, TOTAL> momentum{};
    std::array<double, TOTAL> variance{};
    double timestep = 0.0;

    void reset() {
        momentum.fill(0.0);
        variance.fill(0.0);
        timestep = 0.0;
    }
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
                      double target, const MicroTrainingConfig& config,
                      double sample_weight = 1.0);
    MicroTrainingResult train(const std::vector<TrainingSample>& samples,
                              const MicroTrainingConfig& config);
    // Train refined (multi-head + KAN) end-to-end with analytical gradient
    void train_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                       const FlexEmbedding& concept_from, const FlexEmbedding& concept_to,
                       double target, double learning_rate);
    // Train refined with Adam optimizer (external state for memory efficiency)
    void train_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                       const FlexEmbedding& concept_from, const FlexEmbedding& concept_to,
                       double target, double learning_rate, RefinedAdamState& adam_state);
    // Legacy KAN training (backward compat)
    void train_kan(double bilinear_score, double ctx_feature,
                   double validated_target, double learning_rate);

    // === Convergence Port (122→32) ===
    void forward_convergence(const double* input_122, double* output_32) const;
    // Backward: returns gradient w.r.t. input (122-dim) for backprop to KAN
    void backward_convergence(const double* input_122, const double* cached_output_32,
                               const double* grad_output_32, double learning_rate);
    ConvergencePort& convergence_port() { return conv_port_; }
    const ConvergencePort& convergence_port() const { return conv_port_; }

    // === Pattern Weights ===
    ConceptPatternWeights& pattern_weights() { return patterns_; }
    const ConceptPatternWeights& pattern_weights() const { return patterns_; }

    // === Serialization (1900 doubles) ===
    void to_flat(std::array<double, CM_FLAT_SIZE>& out) const;
    void from_flat(const std::array<double, CM_FLAT_SIZE>& in);

    // === Training quality tracking ===
    bool is_converged() const { return converged_; }
    size_t sample_count() const { return sample_count_; }
    double final_loss() const { return final_loss_; }

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

    // Convergence port (3936 doubles)
    ConvergencePort conv_port_;

    // Per-concept pattern weights (15 doubles)
    ConceptPatternWeights patterns_;

    // Training quality state
    bool converged_ = false;
    size_t sample_count_ = 0;
    double final_loss_ = 1.0;

    // Reserved (25 doubles)
    std::array<double, 25> reserved_{};
};

} // namespace brain19

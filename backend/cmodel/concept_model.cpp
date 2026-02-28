#include "concept_model.hpp"

#include <algorithm>

namespace brain19 {

// =============================================================================
// Static helpers
// =============================================================================

static std::array<double, 4> cyclic_compress(const std::vector<double>& detail) {
    std::array<double, 4> compressed{};
    for (size_t d = 0; d < detail.size(); ++d) {
        compressed[d % 4] += detail[d];
    }
    return compressed;
}

static std::array<double, 20> make_input_vec(const FlexEmbedding& emb) {
    std::array<double, 20> input{};
    for (size_t i = 0; i < CORE_DIM; ++i) {
        input[i] = emb.core[i];
    }
    auto compressed = cyclic_compress(emb.detail);
    for (size_t i = 0; i < 4; ++i) {
        input[CORE_DIM + i] = compressed[i];
    }
    return input;
}

// =============================================================================
// ContextSuperposition::modulate
// =============================================================================
//
// Computes W_eff * x where W_eff = W + Σ αₖ·uₖ·vₖᵀ
// αₖ = softmax(key_k · ctx_query)
//

void ContextSuperposition::modulate(const CoreMat& W, const CoreVec& x,
                                     const std::array<double, KEY_DIM>& ctx_query,
                                     CoreVec& output) const {
    // 1. Compute attention weights: αₖ = softmax(key_k · ctx_query)
    std::array<double, N_MODES> logits{};
    for (size_t k = 0; k < N_MODES; ++k) {
        double dot = 0.0;
        for (size_t d = 0; d < KEY_DIM; ++d)
            dot += keys[k * KEY_DIM + d] * ctx_query[d];
        logits[k] = dot;
    }

    // Stable softmax
    double max_logit = logits[0];
    for (size_t k = 1; k < N_MODES; ++k)
        if (logits[k] > max_logit) max_logit = logits[k];

    std::array<double, N_MODES> alpha{};
    double sum_exp = 0.0;
    for (size_t k = 0; k < N_MODES; ++k) {
        alpha[k] = std::exp(logits[k] - max_logit);
        sum_exp += alpha[k];
    }
    for (size_t k = 0; k < N_MODES; ++k)
        alpha[k] /= sum_exp;

    // 2. Compute vₖᵀ · x for each mode (inner product)
    std::array<double, N_MODES> vTx{};
    for (size_t k = 0; k < N_MODES; ++k) {
        double dot = 0.0;
        for (size_t j = 0; j < CORE_DIM; ++j)
            dot += v[k * CORE_DIM + j] * x[j];
        vTx[k] = dot;
    }

    // 3. output = W*x + Σ αₖ·(vₖᵀ·x)·uₖ
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < CORE_DIM; ++j)
            sum += W[i * CORE_DIM + j] * x[j];

        // Add low-rank modulation
        for (size_t k = 0; k < N_MODES; ++k)
            sum += alpha[k] * vTx[k] * u[k * CORE_DIM + i];

        output[i] = sum;
    }
}

// =============================================================================
// MultiHeadBilinear
// =============================================================================

void MultiHeadBilinear::compute(const FlexEmbedding& e_q, const FlexEmbedding& e_k,
                                 std::array<double, K>& scores) const {
    auto input_q = make_input_vec(e_q);
    auto input_k = make_input_vec(e_k);

    for (size_t h = 0; h < K; ++h) {
        // P_i at offset h * PARAMS_PER_HEAD, size D_PROJ x INPUT_DIM
        // Q_i at offset h * PARAMS_PER_HEAD + D_PROJ * INPUT_DIM
        size_t p_offset = h * PARAMS_PER_HEAD;
        size_t q_offset = p_offset + D_PROJ * INPUT_DIM;

        double dot = 0.0;
        for (size_t d = 0; d < D_PROJ; ++d) {
            double pq = 0.0;
            double pk = 0.0;
            for (size_t j = 0; j < INPUT_DIM; ++j) {
                pq += params[p_offset + d * INPUT_DIM + j] * input_q[j];
                pk += params[q_offset + d * INPUT_DIM + j] * input_k[j];
            }
            dot += pq * pk;
        }
        scores[h] = dot;
    }
}

void MultiHeadBilinear::initialize() {
    // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
    // Breaks symmetry so heads learn different features
    double scale = std::sqrt(2.0 / static_cast<double>(INPUT_DIM + D_PROJ));

    for (size_t i = 0; i < TOTAL_PARAMS; ++i) {
        // Deterministic pseudo-random using sin-hash (no RNG dependency)
        double x = std::sin(static_cast<double>(i * 17 + 31)) * 43758.5453;
        x = x - std::floor(x);  // fractional part in [0,1)
        params[i] = (x * 2.0 - 1.0) * scale;
    }
}

// =============================================================================
// FlexKAN — Lightweight [6,4,1] B-spline network
// =============================================================================
//
// Layer 0: 6 inputs -> 4 hidden  (24 edges, each 10 params = 240)
// Layer 1: 4 hidden -> 1 output  (4 edges, each 10 params = 40)
// Total: 280 params
//
// Each edge function uses a simplified B-spline:
//   f(x) = sum_{k=0}^{9} coeff[k] * basis_k(x)
// where basis_k is a tent function centered at knot k/(NUM_KNOTS-1).
//

static double tent_basis(double x, size_t k, size_t num_knots) {
    double center = static_cast<double>(k) / static_cast<double>(num_knots - 1);
    double width = 1.0 / static_cast<double>(num_knots - 1);
    double dist = std::abs(x - center);
    if (dist >= width) return 0.0;
    return 1.0 - dist / width;
}

// Derivative of tent basis w.r.t. x: +1/width on ascending, -1/width on descending, 0 at peak/outside
static double tent_derivative(double x, size_t k, size_t num_knots) {
    double center = static_cast<double>(k) / static_cast<double>(num_knots - 1);
    double width = 1.0 / static_cast<double>(num_knots - 1);
    double diff = x - center;
    if (std::abs(diff) >= width) return 0.0;
    if (diff > 0.0) return -1.0 / width;
    if (diff < 0.0) return  1.0 / width;
    return 0.0;  // at peak (subgradient = 0)
}

static double kan_edge_forward(const double* coeffs, double x, size_t num_knots) {
    x = std::max(0.0, std::min(1.0, x));
    double result = 0.0;
    for (size_t k = 0; k < num_knots; ++k) {
        result += coeffs[k] * tent_basis(x, k, num_knots);
    }
    return result;
}

double FlexKAN::evaluate(const std::array<double, INPUT_DIM>& input) const {
    // Clamp inputs to [0,1]
    double in[INPUT_DIM];
    for (size_t i = 0; i < INPUT_DIM; ++i) {
        in[i] = std::max(0.0, std::min(1.0, input[i]));
    }

    // Layer 0: INPUT_DIM inputs -> HIDDEN_DIM hidden
    // Edge layout: edge(input_i, hidden_j) at index (i * HIDDEN_DIM + j) * NUM_KNOTS
    double hidden[HIDDEN_DIM] = {};
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        for (size_t i = 0; i < INPUT_DIM; ++i) {
            size_t edge_idx = (i * HIDDEN_DIM + j) * NUM_KNOTS;
            hidden[j] += kan_edge_forward(&params[edge_idx], in[i], NUM_KNOTS);
        }
    }

    // Activate hidden layer with sigmoid
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        hidden[j] = sigmoid(hidden[j]);
    }

    // Layer 1: HIDDEN_DIM hidden -> 1 output
    double output = 0.0;
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        size_t edge_idx = (LAYER0_EDGES + j) * NUM_KNOTS;
        output += kan_edge_forward(&params[edge_idx], hidden[j], NUM_KNOTS);
    }

    return sigmoid(output);
}

void FlexKAN::train_step(const std::array<double, INPUT_DIM>& input,
                          double target, double learning_rate) {
    // Analytical gradient — replaces 280 forward passes with 1 forward + 1 backward

    // Forward with cached intermediates
    double in[INPUT_DIM];
    for (size_t i = 0; i < INPUT_DIM; ++i) {
        in[i] = std::max(0.0, std::min(1.0, input[i]));
    }

    double hidden_raw[HIDDEN_DIM] = {};
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        for (size_t i = 0; i < INPUT_DIM; ++i) {
            size_t edge_idx = (i * HIDDEN_DIM + j) * NUM_KNOTS;
            hidden_raw[j] += kan_edge_forward(&params[edge_idx], in[i], NUM_KNOTS);
        }
    }

    double hidden_act[HIDDEN_DIM];
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        hidden_act[j] = sigmoid(hidden_raw[j]);
    }

    double output_raw = 0.0;
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        size_t edge_idx = (LAYER0_EDGES + j) * NUM_KNOTS;
        output_raw += kan_edge_forward(&params[edge_idx], hidden_act[j], NUM_KNOTS);
    }

    double output = sigmoid(output_raw);

    // Backward pass
    double dL_dout_raw = (output - target) * output * (1.0 - output);

    // Compute dL/d(hidden_raw[j]) using original params before any updates
    double dL_dhidden_raw[HIDDEN_DIM] = {};
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        size_t edge_base = (LAYER0_EDGES + j) * NUM_KNOTS;
        double d_edge_dx = 0.0;
        for (size_t k = 0; k < NUM_KNOTS; ++k) {
            d_edge_dx += params[edge_base + k] * tent_derivative(hidden_act[j], k, NUM_KNOTS);
        }
        dL_dhidden_raw[j] = dL_dout_raw * d_edge_dx * hidden_act[j] * (1.0 - hidden_act[j]);
    }

    // Update layer 1 coefficients
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        size_t edge_base = (LAYER0_EDGES + j) * NUM_KNOTS;
        for (size_t k = 0; k < NUM_KNOTS; ++k) {
            double grad = dL_dout_raw * tent_basis(hidden_act[j], k, NUM_KNOTS);
            params[edge_base + k] -= learning_rate * grad;
        }
    }

    // Update layer 0 coefficients
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        for (size_t i = 0; i < INPUT_DIM; ++i) {
            size_t edge_base = (i * HIDDEN_DIM + j) * NUM_KNOTS;
            for (size_t k = 0; k < NUM_KNOTS; ++k) {
                double grad = dL_dhidden_raw[j] * tent_basis(in[i], k, NUM_KNOTS);
                params[edge_base + k] -= learning_rate * grad;
            }
        }
    }
}

void FlexKAN::initialize_identity() {
    params.fill(0.0);

    // We want: output ~ bilinear_score (which is input[4])
    // Two sigmoids in the path, so use logit-scaled coefficients.
    auto safe_logit = [](double p) -> double {
        p = std::max(0.01, std::min(0.99, p));
        return std::log(p / (1.0 - p));
    };

    // Layer 0: edge from input_4 to hidden_0 with logit identity
    // Edge index: (4 * HIDDEN_DIM + 0) * NUM_KNOTS
    size_t l0_edge = (4 * HIDDEN_DIM + 0) * NUM_KNOTS;
    for (size_t k = 0; k < NUM_KNOTS; ++k) {
        double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
        params[l0_edge + k] = safe_logit(x);
    }

    // Layer 1: edge from hidden_0 to output with logit identity
    size_t l1_edge = (LAYER0_EDGES + 0) * NUM_KNOTS;
    for (size_t k = 0; k < NUM_KNOTS; ++k) {
        double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
        params[l1_edge + k] = safe_logit(x);
    }
    // All other edges are zero — only input_4->hidden_0->output contributes.
}

// =============================================================================
// ConvergencePort — 122→32 linear+tanh per concept
// =============================================================================

void ConvergencePort::compute(const double* input, double* output) const {
    for (size_t i = 0; i < OUTPUT_DIM; ++i) {
        // Gate: σ(W_gate[i] · input + b_gate[i])
        double gate_sum = b_gate[i];
        for (size_t j = 0; j < INPUT_DIM; ++j)
            gate_sum += W_gate[i * INPUT_DIM + j] * input[j];
        double gate = 1.0 / (1.0 + std::exp(-gate_sum));

        // New state: tanh(W[i] · input + b[i])
        double new_sum = b[i];
        for (size_t j = 0; j < INPUT_DIM; ++j)
            new_sum += W[i * INPUT_DIM + j] * input[j];
        double new_val = std::tanh(new_sum);

        // GRU-style: output = gate * new + (1-gate) * prev_state
        double prev = input[PREV_STATE_OFFSET + i];
        output[i] = gate * new_val + (1.0 - gate) * prev;
    }
}

void ConvergencePort::initialize() {
    // Xavier initialization for W: scale = sqrt(2 / (fan_in + fan_out))
    double scale = std::sqrt(2.0 / static_cast<double>(INPUT_DIM + OUTPUT_DIM));
    for (size_t i = 0; i < W_SIZE; ++i) {
        double x = std::sin(static_cast<double>(i * 23 + 47)) * 43758.5453;
        x = x - std::floor(x);
        W[i] = (x * 2.0 - 1.0) * scale;
    }
    b.fill(0.0);
    // Gate: zero init → sigmoid(0) = 0.5 → half update, half retain (neutral)
    W_gate.fill(0.0);
    b_gate.fill(0.0);
}

// =============================================================================
// ConceptModel Construction
// =============================================================================

ConceptModel::ConceptModel() {
    // Initialize W with small values (Xavier-like for 16x16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        for (size_t j = 0; j < CORE_DIM; ++j) {
            double diag = (i == j) ? 0.1 : 0.0;
            double off = 0.01 * std::sin(static_cast<double>(i * 10 + j));
            W_[i * CORE_DIM + j] = diag + off;
        }
    }

    for (size_t i = 0; i < CORE_DIM; ++i) {
        b_[i] = 0.01 * std::cos(static_cast<double>(i));
    }

    for (size_t i = 0; i < CORE_DIM; ++i) {
        e_init_[i] = 0.1 * std::sin(static_cast<double>(i * 3 + 1));
        c_init_[i] = 0.1 * std::cos(static_cast<double>(i * 7 + 2));
    }

    multihead_.initialize();
    kan_.initialize_identity();
    conv_port_.initialize();
    reserved_.fill(0.0);
}

// =============================================================================
// Forward pass — operates on Core dimensions only
// =============================================================================

double ConceptModel::predict(const FlexEmbedding& e, const FlexEmbedding& c,
                              CoreVec* v_out) const {
    CoreVec v;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = b_[i];
        for (size_t j = 0; j < CORE_DIM; ++j) {
            sum += W_[i * CORE_DIM + j] * c.core[j];
        }
        v[i] = sum;
    }

    // Expose intermediate for dimensional flow
    if (v_out) *v_out = v;

    double z = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        z += e.core[i] * v[i];
    }

    return sigmoid(z);
}

double ConceptModel::predict_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                                      const FlexEmbedding& concept_from,
                                      const FlexEmbedding& concept_to) const {
    double bilinear = predict(rel_emb, ctx_emb);

    // Compute multi-head scores from concept embeddings
    std::array<double, MultiHeadBilinear::K> mh_scores;
    multihead_.compute(concept_from, concept_to, mh_scores);

    // dim_fraction: maturity signal from concept detail dimensions
    double dim_fraction = static_cast<double>(
        std::min(concept_from.detail.size(), concept_to.detail.size())) / 496.0;

    // Assemble KAN input: [s_0, s_1, s_2, s_3, bilinear_score, dim_fraction]
    std::array<double, FlexKAN::INPUT_DIM> kan_input;
    for (size_t i = 0; i < MultiHeadBilinear::K; ++i) {
        kan_input[i] = sigmoid(mh_scores[i]);
    }
    kan_input[4] = bilinear;
    kan_input[5] = dim_fraction;

    return kan_.evaluate(kan_input);
}

PredictFeatures ConceptModel::predict_refined_with_features(
    const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
    const FlexEmbedding& concept_from, const FlexEmbedding& concept_to) const
{
    PredictFeatures f;
    CoreVec v;
    f.bilinear_score = predict(rel_emb, ctx_emb, &v);
    f.dimensional_score = v;

    std::array<double, MultiHeadBilinear::K> mh_scores;
    multihead_.compute(concept_from, concept_to, mh_scores);

    for (size_t i = 0; i < MultiHeadBilinear::K; ++i) {
        f.multihead_scores[i] = sigmoid(mh_scores[i]);
    }

    f.dim_fraction = static_cast<double>(
        std::min(concept_from.detail.size(), concept_to.detail.size())) / 496.0;

    std::array<double, FlexKAN::INPUT_DIM> kan_input;
    for (size_t i = 0; i < MultiHeadBilinear::K; ++i) {
        kan_input[i] = f.multihead_scores[i];
    }
    kan_input[4] = f.bilinear_score;
    kan_input[5] = f.dim_fraction;

    f.refined_score = kan_.evaluate(kan_input);
    return f;
}

double ConceptModel::predict_refined(const FlexEmbedding& e, const FlexEmbedding& c) const {
    FlexEmbedding empty;
    return predict_refined(e, c, empty, empty);
}

// =============================================================================
// Training step (Adam optimizer) — identical to MicroModel
// =============================================================================

double ConceptModel::train_step(const FlexEmbedding& e, const FlexEmbedding& c,
                                 double target, const MicroTrainingConfig& config,
                                 double sample_weight) {
    // Forward pass
    CoreVec v;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = b_[i];
        for (size_t j = 0; j < CORE_DIM; ++j) {
            sum += W_[i * CORE_DIM + j] * c.core[j];
        }
        v[i] = sum;
    }

    double z = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        z += e.core[i] * v[i];
    }

    double w = sigmoid(z);
    double error = w - target;
    // Weight scales both loss and gradient — low-trust samples train weaker
    double loss = sample_weight * 0.5 * error * error;
    double delta = sample_weight * error * w * (1.0 - w);

    state_.timestep += 1.0;
    double t = state_.timestep;

    double beta1 = config.adam_beta1;
    double beta2 = config.adam_beta2;
    double eps = config.adam_epsilon;
    double lr = config.learning_rate;

    double bc1 = 1.0 - std::pow(beta1, t);
    double bc2 = 1.0 - std::pow(beta2, t);

    for (size_t i = 0; i < CORE_DIM; ++i) {
        double grad_b = delta * e.core[i];

        state_.db_momentum[i] = beta1 * state_.db_momentum[i] + (1.0 - beta1) * grad_b;
        state_.db_variance[i] = beta2 * state_.db_variance[i] + (1.0 - beta2) * grad_b * grad_b;

        double m_hat = state_.db_momentum[i] / bc1;
        double v_hat = state_.db_variance[i] / bc2;

        b_[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

        for (size_t j = 0; j < CORE_DIM; ++j) {
            double grad_w = delta * e.core[i] * c.core[j];
            size_t idx = i * CORE_DIM + j;

            state_.dW_momentum[idx] = beta1 * state_.dW_momentum[idx] + (1.0 - beta1) * grad_w;
            state_.dW_variance[idx] = beta2 * state_.dW_variance[idx] + (1.0 - beta2) * grad_w * grad_w;

            double mw_hat = state_.dW_momentum[idx] / bc1;
            double vw_hat = state_.dW_variance[idx] / bc2;

            W_[idx] -= lr * mw_hat / (std::sqrt(vw_hat) + eps);
        }
    }

    state_.last_loss = loss;
    state_.total_samples += 1.0;
    if (loss < state_.best_loss) {
        state_.best_loss = loss;
    }

    return loss;
}

// =============================================================================
// Batch training
// =============================================================================

MicroTrainingResult ConceptModel::train(const std::vector<TrainingSample>& samples,
                                         const MicroTrainingConfig& config) {
    MicroTrainingResult result;
    sample_count_ = samples.size();

    if (samples.empty()) {
        result.converged = true;
        converged_ = true;
        final_loss_ = 0.0;
        return result;
    }

    double best_avg_loss = 1e9;
    size_t patience_counter = 0;

    for (size_t epoch = 0; epoch < config.max_epochs; ++epoch) {
        double total_loss = 0.0;

        for (const auto& sample : samples) {
            total_loss += train_step(sample.relation_embedding,
                                     sample.context_embedding,
                                     sample.target, config,
                                     sample.weight);
        }

        double avg_loss = total_loss / static_cast<double>(samples.size());
        result.epochs_run = epoch + 1;
        result.final_loss = avg_loss;

        if (avg_loss < best_avg_loss - config.convergence_threshold) {
            best_avg_loss = avg_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
        }

        // Converged if loss is already very low
        if (avg_loss < 0.01) {
            result.converged = true;
            break;
        }

        // Converged if no improvement for 10 epochs
        if (patience_counter >= 10) {
            result.converged = true;
            break;
        }
    }

    converged_ = result.converged;
    final_loss_ = result.final_loss;
    return result;
}

// =============================================================================
// Refined training (multi-head + KAN, analytical gradient)
// =============================================================================
//
// Replaces 920 forward passes (numerical gradient) with 1 forward + 1 backward.
// Backprop chain: loss -> sigmoid -> KAN(L1) -> sigmoid -> KAN(L0) -> sigmoid -> multihead
//

void ConceptModel::train_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                                  const FlexEmbedding& concept_from,
                                  const FlexEmbedding& concept_to,
                                  double target, double learning_rate) {
    // =================================================================
    // Forward pass with cached intermediates
    // =================================================================

    // 1. Bilinear score (constant w.r.t. refined params)
    double bilinear = predict(rel_emb, ctx_emb);

    // 2. Multi-head bilinear — inline to cache projections
    auto input_q = make_input_vec(concept_from);
    auto input_k = make_input_vec(concept_to);

    constexpr size_t MH_K  = MultiHeadBilinear::K;
    constexpr size_t MH_D  = MultiHeadBilinear::D_PROJ;
    constexpr size_t MH_IN = MultiHeadBilinear::INPUT_DIM;

    std::array<double, MH_K> mh_scores{};
    double p_proj[MH_K][MH_D];
    double q_proj[MH_K][MH_D];

    for (size_t h = 0; h < MH_K; ++h) {
        size_t p_off = h * MultiHeadBilinear::PARAMS_PER_HEAD;
        size_t q_off = p_off + MH_D * MH_IN;

        double dot = 0.0;
        for (size_t d = 0; d < MH_D; ++d) {
            p_proj[h][d] = 0.0;
            q_proj[h][d] = 0.0;
            for (size_t j = 0; j < MH_IN; ++j) {
                p_proj[h][d] += multihead_.params[p_off + d * MH_IN + j] * input_q[j];
                q_proj[h][d] += multihead_.params[q_off + d * MH_IN + j] * input_k[j];
            }
            dot += p_proj[h][d] * q_proj[h][d];
        }
        mh_scores[h] = dot;
    }

    // 3. Assemble KAN input
    constexpr size_t KAN_IN = FlexKAN::INPUT_DIM;
    constexpr size_t KAN_H  = FlexKAN::HIDDEN_DIM;
    constexpr size_t KAN_NK = FlexKAN::NUM_KNOTS;

    std::array<double, KAN_IN> kan_input;
    for (size_t i = 0; i < MH_K; ++i) {
        kan_input[i] = sigmoid(mh_scores[i]);
    }
    kan_input[4] = bilinear;
    kan_input[5] = static_cast<double>(
        std::min(concept_from.detail.size(), concept_to.detail.size())) / 496.0;

    // Clamp for KAN
    double in[KAN_IN];
    for (size_t i = 0; i < KAN_IN; ++i) {
        in[i] = std::max(0.0, std::min(1.0, kan_input[i]));
    }

    // 4. KAN Layer 0 forward
    double hidden_raw[KAN_H] = {};
    for (size_t j = 0; j < KAN_H; ++j) {
        for (size_t i = 0; i < KAN_IN; ++i) {
            size_t edge_idx = (i * KAN_H + j) * KAN_NK;
            hidden_raw[j] += kan_edge_forward(&kan_.params[edge_idx], in[i], KAN_NK);
        }
    }

    double hidden_act[KAN_H];
    for (size_t j = 0; j < KAN_H; ++j) {
        hidden_act[j] = sigmoid(hidden_raw[j]);
    }

    // 5. KAN Layer 1 forward
    double output_raw = 0.0;
    for (size_t j = 0; j < KAN_H; ++j) {
        size_t edge_idx = (FlexKAN::LAYER0_EDGES + j) * KAN_NK;
        output_raw += kan_edge_forward(&kan_.params[edge_idx], hidden_act[j], KAN_NK);
    }

    double output = sigmoid(output_raw);

    // =================================================================
    // Backward pass — compute all gradients before applying updates
    // =================================================================

    double dL_dout = output - target;
    double dL_dout_raw = dL_dout * output * (1.0 - output);

    // --- KAN Layer 1: coefficient gradients + backprop to hidden ---
    double kan_grad[FlexKAN::TOTAL_PARAMS] = {};
    double dL_dhidden_act[KAN_H] = {};

    for (size_t j = 0; j < KAN_H; ++j) {
        size_t edge_base = (FlexKAN::LAYER0_EDGES + j) * KAN_NK;

        // Coefficient gradients: dL/d(coeff) = dL/d(out_raw) * tent(hidden_act[j], k)
        for (size_t k = 0; k < KAN_NK; ++k) {
            kan_grad[edge_base + k] = dL_dout_raw * tent_basis(hidden_act[j], k, KAN_NK);
        }

        // Backprop to hidden_act[j]: d(out_raw)/d(hidden_act[j]) = sum_k coeff * d_tent/dx
        double d_edge_dx = 0.0;
        for (size_t k = 0; k < KAN_NK; ++k) {
            d_edge_dx += kan_.params[edge_base + k] * tent_derivative(hidden_act[j], k, KAN_NK);
        }
        dL_dhidden_act[j] = dL_dout_raw * d_edge_dx;
    }

    // Through sigmoid: dL/d(hidden_raw[j])
    double dL_dhidden_raw[KAN_H];
    for (size_t j = 0; j < KAN_H; ++j) {
        dL_dhidden_raw[j] = dL_dhidden_act[j] * hidden_act[j] * (1.0 - hidden_act[j]);
    }

    // --- KAN Layer 0: coefficient gradients + backprop to input ---
    double dL_din[KAN_IN] = {};

    for (size_t i = 0; i < KAN_IN; ++i) {
        for (size_t j = 0; j < KAN_H; ++j) {
            size_t edge_base = (i * KAN_H + j) * KAN_NK;

            // Coefficient gradients
            for (size_t k = 0; k < KAN_NK; ++k) {
                kan_grad[edge_base + k] = dL_dhidden_raw[j] * tent_basis(in[i], k, KAN_NK);
            }

            // Backprop to input
            double d_edge_dx = 0.0;
            for (size_t k = 0; k < KAN_NK; ++k) {
                d_edge_dx += kan_.params[edge_base + k] * tent_derivative(in[i], k, KAN_NK);
            }
            dL_din[i] += dL_dhidden_raw[j] * d_edge_dx;
        }
    }

    // --- Multi-head gradients ---
    // kan_input[h] = sigmoid(mh_scores[h]), so:
    // dL/d(mh_scores[h]) = dL/d(in[h]) * kan_input[h] * (1 - kan_input[h])
    double mh_grad[MultiHeadBilinear::TOTAL_PARAMS] = {};

    for (size_t h = 0; h < MH_K; ++h) {
        double dL_dmh = dL_din[h] * kan_input[h] * (1.0 - kan_input[h]);

        size_t p_off = h * MultiHeadBilinear::PARAMS_PER_HEAD;
        size_t q_off = p_off + MH_D * MH_IN;

        // score[h] = sum_d dot(P*input_q, Q*input_k)
        // d(score[h])/dP[d][j] = q_proj[d] * input_q[j]
        // d(score[h])/dQ[d][j] = p_proj[d] * input_k[j]
        for (size_t d = 0; d < MH_D; ++d) {
            for (size_t j = 0; j < MH_IN; ++j) {
                mh_grad[p_off + d * MH_IN + j] = dL_dmh * q_proj[h][d] * input_q[j];
                mh_grad[q_off + d * MH_IN + j] = dL_dmh * p_proj[h][d] * input_k[j];
            }
        }
    }

    // =================================================================
    // Apply all updates
    // =================================================================

    for (size_t i = 0; i < MultiHeadBilinear::TOTAL_PARAMS; ++i) {
        multihead_.params[i] -= learning_rate * mh_grad[i];
    }

    for (size_t i = 0; i < FlexKAN::TOTAL_PARAMS; ++i) {
        kan_.params[i] -= learning_rate * kan_grad[i];
    }
}

// =============================================================================
// Refined training with Adam optimizer
// =============================================================================

void ConceptModel::train_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                                  const FlexEmbedding& concept_from,
                                  const FlexEmbedding& concept_to,
                                  double target, double learning_rate,
                                  RefinedAdamState& adam) {
    // Forward pass with cached intermediates (identical to SGD version)
    double bilinear = predict(rel_emb, ctx_emb);

    auto input_q = make_input_vec(concept_from);
    auto input_k = make_input_vec(concept_to);

    constexpr size_t MH_K  = MultiHeadBilinear::K;
    constexpr size_t MH_D  = MultiHeadBilinear::D_PROJ;
    constexpr size_t MH_IN = MultiHeadBilinear::INPUT_DIM;

    std::array<double, MH_K> mh_scores{};
    double p_proj[MH_K][MH_D];
    double q_proj[MH_K][MH_D];

    for (size_t h = 0; h < MH_K; ++h) {
        size_t p_off = h * MultiHeadBilinear::PARAMS_PER_HEAD;
        size_t q_off = p_off + MH_D * MH_IN;

        double dot = 0.0;
        for (size_t d = 0; d < MH_D; ++d) {
            p_proj[h][d] = 0.0;
            q_proj[h][d] = 0.0;
            for (size_t j = 0; j < MH_IN; ++j) {
                p_proj[h][d] += multihead_.params[p_off + d * MH_IN + j] * input_q[j];
                q_proj[h][d] += multihead_.params[q_off + d * MH_IN + j] * input_k[j];
            }
            dot += p_proj[h][d] * q_proj[h][d];
        }
        mh_scores[h] = dot;
    }

    constexpr size_t KAN_IN = FlexKAN::INPUT_DIM;
    constexpr size_t KAN_H  = FlexKAN::HIDDEN_DIM;
    constexpr size_t KAN_NK = FlexKAN::NUM_KNOTS;

    std::array<double, KAN_IN> kan_input;
    for (size_t i = 0; i < MH_K; ++i) {
        kan_input[i] = sigmoid(mh_scores[i]);
    }
    kan_input[4] = bilinear;
    kan_input[5] = static_cast<double>(
        std::min(concept_from.detail.size(), concept_to.detail.size())) / 496.0;

    double in[KAN_IN];
    for (size_t i = 0; i < KAN_IN; ++i) {
        in[i] = std::max(0.0, std::min(1.0, kan_input[i]));
    }

    double hidden_raw[KAN_H] = {};
    for (size_t j = 0; j < KAN_H; ++j) {
        for (size_t i = 0; i < KAN_IN; ++i) {
            size_t edge_idx = (i * KAN_H + j) * KAN_NK;
            hidden_raw[j] += kan_edge_forward(&kan_.params[edge_idx], in[i], KAN_NK);
        }
    }

    double hidden_act[KAN_H];
    for (size_t j = 0; j < KAN_H; ++j) {
        hidden_act[j] = sigmoid(hidden_raw[j]);
    }

    double output_raw = 0.0;
    for (size_t j = 0; j < KAN_H; ++j) {
        size_t edge_idx = (FlexKAN::LAYER0_EDGES + j) * KAN_NK;
        output_raw += kan_edge_forward(&kan_.params[edge_idx], hidden_act[j], KAN_NK);
    }

    double output = sigmoid(output_raw);

    // Backward pass (identical gradient computation)
    double dL_dout = output - target;
    double dL_dout_raw = dL_dout * output * (1.0 - output);

    double kan_grad[FlexKAN::TOTAL_PARAMS] = {};
    double dL_dhidden_act[KAN_H] = {};

    for (size_t j = 0; j < KAN_H; ++j) {
        size_t edge_base = (FlexKAN::LAYER0_EDGES + j) * KAN_NK;

        for (size_t k = 0; k < KAN_NK; ++k) {
            kan_grad[edge_base + k] = dL_dout_raw * tent_basis(hidden_act[j], k, KAN_NK);
        }

        double d_edge_dx = 0.0;
        for (size_t k = 0; k < KAN_NK; ++k) {
            d_edge_dx += kan_.params[edge_base + k] * tent_derivative(hidden_act[j], k, KAN_NK);
        }
        dL_dhidden_act[j] = dL_dout_raw * d_edge_dx;
    }

    double dL_dhidden_raw[KAN_H];
    for (size_t j = 0; j < KAN_H; ++j) {
        dL_dhidden_raw[j] = dL_dhidden_act[j] * hidden_act[j] * (1.0 - hidden_act[j]);
    }

    double dL_din[KAN_IN] = {};
    for (size_t i = 0; i < KAN_IN; ++i) {
        for (size_t j = 0; j < KAN_H; ++j) {
            size_t edge_base = (i * KAN_H + j) * KAN_NK;

            for (size_t k = 0; k < KAN_NK; ++k) {
                kan_grad[edge_base + k] = dL_dhidden_raw[j] * tent_basis(in[i], k, KAN_NK);
            }

            double d_edge_dx = 0.0;
            for (size_t k = 0; k < KAN_NK; ++k) {
                d_edge_dx += kan_.params[edge_base + k] * tent_derivative(in[i], k, KAN_NK);
            }
            dL_din[i] += dL_dhidden_raw[j] * d_edge_dx;
        }
    }

    double mh_grad[MultiHeadBilinear::TOTAL_PARAMS] = {};
    for (size_t h = 0; h < MH_K; ++h) {
        double dL_dmh = dL_din[h] * kan_input[h] * (1.0 - kan_input[h]);

        size_t p_off = h * MultiHeadBilinear::PARAMS_PER_HEAD;
        size_t q_off = p_off + MH_D * MH_IN;

        for (size_t d = 0; d < MH_D; ++d) {
            for (size_t j = 0; j < MH_IN; ++j) {
                mh_grad[p_off + d * MH_IN + j] = dL_dmh * q_proj[h][d] * input_q[j];
                mh_grad[q_off + d * MH_IN + j] = dL_dmh * p_proj[h][d] * input_k[j];
            }
        }
    }

    // Apply updates with Adam optimizer
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr double eps = 1e-8;

    adam.timestep += 1.0;
    double bc1 = 1.0 - std::pow(beta1, adam.timestep);
    double bc2 = 1.0 - std::pow(beta2, adam.timestep);

    // MultiHead params (indices 0..639 in Adam state)
    for (size_t i = 0; i < MultiHeadBilinear::TOTAL_PARAMS; ++i) {
        adam.momentum[i] = beta1 * adam.momentum[i] + (1.0 - beta1) * mh_grad[i];
        adam.variance[i] = beta2 * adam.variance[i] + (1.0 - beta2) * mh_grad[i] * mh_grad[i];
        double m_hat = adam.momentum[i] / bc1;
        double v_hat = adam.variance[i] / bc2;
        multihead_.params[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }

    // KAN params (indices 640..919 in Adam state)
    constexpr size_t KAN_OFF = MultiHeadBilinear::TOTAL_PARAMS;
    for (size_t i = 0; i < FlexKAN::TOTAL_PARAMS; ++i) {
        adam.momentum[KAN_OFF + i] = beta1 * adam.momentum[KAN_OFF + i] + (1.0 - beta1) * kan_grad[i];
        adam.variance[KAN_OFF + i] = beta2 * adam.variance[KAN_OFF + i] + (1.0 - beta2) * kan_grad[i] * kan_grad[i];
        double m_hat = adam.momentum[KAN_OFF + i] / bc1;
        double v_hat = adam.variance[KAN_OFF + i] / bc2;
        kan_.params[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// =============================================================================
// Convergence Port (122→32) — forward and backward
// =============================================================================

void ConceptModel::forward_convergence(const double* input_122, double* output_32) const {
    conv_port_.compute(input_122, output_32);
}

void ConceptModel::backward_convergence(const double* input_122, const double* /*cached_output_32*/,
                                         const double* grad_output_32, double learning_rate) {
    // GRU-style gate backward:
    // output = gate * new_val + (1-gate) * prev
    // gate = σ(W_gate·x + b_gate), new_val = tanh(W·x + b)

    constexpr size_t IN = ConvergencePort::INPUT_DIM;
    constexpr size_t OUT = ConvergencePort::OUTPUT_DIM;
    constexpr size_t PREV_OFF = ConvergencePort::PREV_STATE_OFFSET;

    for (size_t i = 0; i < OUT; ++i) {
        // Recompute gate and new_val from input
        double gate_sum = conv_port_.b_gate[i];
        for (size_t j = 0; j < IN; ++j)
            gate_sum += conv_port_.W_gate[i * IN + j] * input_122[j];
        double gate = 1.0 / (1.0 + std::exp(-gate_sum));

        double new_sum = conv_port_.b[i];
        for (size_t j = 0; j < IN; ++j)
            new_sum += conv_port_.W[i * IN + j] * input_122[j];
        double new_val = std::tanh(new_sum);

        double prev = input_122[PREV_OFF + i];
        double dL_dout = grad_output_32[i];

        // dL/d(new_sum) = dL/dout * gate * (1 - new_val²)
        double dL_dnew_sum = dL_dout * gate * (1.0 - new_val * new_val);

        // dL/d(gate_sum) = dL/dout * (new_val - prev) * gate * (1 - gate)
        double dL_dgate_sum = dL_dout * (new_val - prev) * gate * (1.0 - gate);

        // Update W and b (transform path)
        conv_port_.b[i] -= learning_rate * dL_dnew_sum;
        for (size_t j = 0; j < IN; ++j)
            conv_port_.W[i * IN + j] -= learning_rate * dL_dnew_sum * input_122[j];

        // Update W_gate and b_gate (gate path)
        conv_port_.b_gate[i] -= learning_rate * dL_dgate_sum;
        for (size_t j = 0; j < IN; ++j)
            conv_port_.W_gate[i * IN + j] -= learning_rate * dL_dgate_sum * input_122[j];
    }
}

// =============================================================================
// Legacy KAN training
// =============================================================================

void ConceptModel::train_kan(double bilinear_score, double ctx_feature,
                              double validated_target, double learning_rate) {
    // Legacy: pass through to FlexKAN with only bilinear_score as meaningful input
    std::array<double, FlexKAN::INPUT_DIM> input{};
    input[4] = bilinear_score;
    input[5] = ctx_feature;
    kan_.train_step(input, validated_target, learning_rate);
}

// =============================================================================
// Superposition training — gradient for u, v, keys
// =============================================================================
//
// Forward:  v[i] = Σ_j W[i,j]*c[j] + Σ_k α_k * (v_k·c) * u_k[i] + b[i]
//           z = e · v,   pred = sigmoid(z)
//
// Loss = 0.5 * (pred - target)²
//
// Gradients:
//   delta = (pred - target) * pred * (1 - pred)
//   grad_v[i] = delta * e[i]
//
//   ∂loss/∂u_k[i] = grad_v[i] * α_k * vTx_k
//   ∂loss/∂v_k[j] = α_k * (Σ_i grad_v[i] * u_k[i]) * c[j]
//   ∂loss/∂key_k[d] = ctx_query[d] * α_k * (vTx_k * E_k - F)
//     where E_k = Σ_i grad_v[i] * u_k[i]
//           F   = Σ_m α_m * vTx_m * E_m
//

void ConceptModel::train_superposition_step(
    const FlexEmbedding& e, const FlexEmbedding& c,
    double target, double learning_rate,
    const std::array<double, ContextSuperposition::KEY_DIM>& ctx_query)
{
    if (superposition_.enabled < 0.5) return;

    constexpr size_t N = ContextSuperposition::N_MODES;
    constexpr size_t KD = ContextSuperposition::KEY_DIM;

    // --- Forward pass with superposition ---

    // 1. Attention weights: α_k = softmax(key_k · ctx_query)
    std::array<double, N> logits{};
    for (size_t k = 0; k < N; ++k) {
        double dot = 0.0;
        for (size_t d = 0; d < KD; ++d)
            dot += superposition_.keys[k * KD + d] * ctx_query[d];
        logits[k] = dot;
    }

    double max_logit = logits[0];
    for (size_t k = 1; k < N; ++k)
        if (logits[k] > max_logit) max_logit = logits[k];

    std::array<double, N> alpha{};
    double sum_exp = 0.0;
    for (size_t k = 0; k < N; ++k) {
        alpha[k] = std::exp(logits[k] - max_logit);
        sum_exp += alpha[k];
    }
    for (size_t k = 0; k < N; ++k)
        alpha[k] /= sum_exp;

    // 2. vTx_k = v_k · c.core
    std::array<double, N> vTx{};
    for (size_t k = 0; k < N; ++k) {
        double dot = 0.0;
        for (size_t j = 0; j < CORE_DIM; ++j)
            dot += superposition_.v[k * CORE_DIM + j] * c.core[j];
        vTx[k] = dot;
    }

    // 3. v = W_eff * c + b, z = e · v
    double z = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double vi = b_[i];
        for (size_t j = 0; j < CORE_DIM; ++j)
            vi += W_[i * CORE_DIM + j] * c.core[j];
        for (size_t k = 0; k < N; ++k)
            vi += alpha[k] * vTx[k] * superposition_.u[k * CORE_DIM + i];
        z += e.core[i] * vi;
    }

    double pred = sigmoid(z);
    double error = pred - target;
    double delta = error * pred * (1.0 - pred);

    // --- Backward pass ---

    // grad_v[i] = delta * e.core[i]  (gradient w.r.t. pre-sigmoid output per dim)
    CoreVec grad_v{};
    for (size_t i = 0; i < CORE_DIM; ++i)
        grad_v[i] = delta * e.core[i];

    // E_k = Σ_i grad_v[i] * u_k[i]
    std::array<double, N> E{};
    for (size_t k = 0; k < N; ++k) {
        double sum = 0.0;
        for (size_t i = 0; i < CORE_DIM; ++i)
            sum += grad_v[i] * superposition_.u[k * CORE_DIM + i];
        E[k] = sum;
    }

    // F = Σ_m α_m * vTx_m * E_m
    double F = 0.0;
    for (size_t m = 0; m < N; ++m)
        F += alpha[m] * vTx[m] * E[m];

    // Update u: ∂loss/∂u_k[i] = grad_v[i] * α_k * vTx_k
    for (size_t k = 0; k < N; ++k) {
        double scale = learning_rate * alpha[k] * vTx[k];
        for (size_t i = 0; i < CORE_DIM; ++i)
            superposition_.u[k * CORE_DIM + i] -= scale * grad_v[i];
    }

    // Update v: ∂loss/∂v_k[j] = α_k * E_k * c.core[j]
    for (size_t k = 0; k < N; ++k) {
        double scale = learning_rate * alpha[k] * E[k];
        for (size_t i = 0; i < CORE_DIM; ++i)
            superposition_.v[k * CORE_DIM + i] -= scale * c.core[i];
    }

    // Update keys: ∂loss/∂key_k[d] = ctx_query[d] * α_k * (vTx_k * E_k - F)
    for (size_t k = 0; k < N; ++k) {
        double scale = learning_rate * alpha[k] * (vTx[k] * E[k] - F);
        for (size_t d = 0; d < KD; ++d)
            superposition_.keys[k * KD + d] -= scale * ctx_query[d];
    }
}

// =============================================================================
// Serialization (9933 doubles — V8)
// =============================================================================
// Layout: [0..939]     bilinear core (W, b, e_init, c_init, TrainingState)
//         [940..1579]  MultiHeadBilinear (640)
//         [1580..1859] FlexKAN (280)
//         [1860..1874] ConceptPatternWeights (15)
//         [1875..1899] reserved (25)
//         [1900..5803] ConvergencePort W (3904)
//         [5804..5835] ConvergencePort b (32)
//         [5836..9739] ConvergencePort W_gate (3904)
//         [9740..9771] ConvergencePort b_gate (32)
//         [9772..9835] ContextSuperposition u (64)
//         [9836..9899] ContextSuperposition v (64)
//         [9900..9931] ContextSuperposition keys (32)
//         [9932]       ContextSuperposition enabled (1)

void ConceptModel::to_flat(std::array<double, CM_FLAT_SIZE>& out) const {
    size_t idx = 0;

    // W (256)
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) out[idx++] = W_[i];
    // b (16)
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = b_[i];
    // e_init (16)
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = e_init_[i];
    // c_init (16)
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = c_init_[i];

    // TrainingState
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) out[idx++] = state_.dW_momentum[i];
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = state_.db_momentum[i];
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) out[idx++] = state_.dW_variance[i];
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = state_.db_variance[i];
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = state_.e_grad_accum[i];
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = state_.c_grad_accum[i];
    out[idx++] = state_.timestep;
    out[idx++] = state_.last_loss;
    out[idx++] = state_.best_loss;
    out[idx++] = state_.total_samples;
    out[idx++] = state_.reserved_scalar;
    for (size_t i = 0; i < 55; ++i) out[idx++] = state_.reserved[i];

    // idx = 940

    // MultiHeadBilinear (640)
    for (size_t i = 0; i < MultiHeadBilinear::TOTAL_PARAMS; ++i) out[idx++] = multihead_.params[i];

    // FlexKAN (280)
    for (size_t i = 0; i < FlexKAN::TOTAL_PARAMS; ++i) out[idx++] = kan_.params[i];

    // Pattern weights (15)
    out[idx++] = patterns_.shared_parent;
    out[idx++] = patterns_.transitive_causation;
    out[idx++] = patterns_.missing_link;
    out[idx++] = patterns_.weak_strengthening;
    out[idx++] = patterns_.contradictory_signal;
    out[idx++] = patterns_.chain_hypothesis;
    for (size_t i = 0; i < 9; ++i) out[idx++] = patterns_.reserved[i];

    // Reserved (25)
    for (size_t i = 0; i < 25; ++i) out[idx++] = reserved_[i];

    // idx = 1900

    // ConvergencePort W (3904)
    for (size_t i = 0; i < ConvergencePort::W_SIZE; ++i) out[idx++] = conv_port_.W[i];

    // ConvergencePort b (32)
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) out[idx++] = conv_port_.b[i];

    // ConvergencePort W_gate (3904)
    for (size_t i = 0; i < ConvergencePort::W_GATE_SIZE; ++i) out[idx++] = conv_port_.W_gate[i];

    // ConvergencePort b_gate (32)
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) out[idx++] = conv_port_.b_gate[i];

    // idx = 9772

    // ContextSuperposition u (64)
    for (size_t i = 0; i < ContextSuperposition::N_MODES * CORE_DIM; ++i)
        out[idx++] = superposition_.u[i];
    // ContextSuperposition v (64)
    for (size_t i = 0; i < ContextSuperposition::N_MODES * CORE_DIM; ++i)
        out[idx++] = superposition_.v[i];
    // ContextSuperposition keys (32)
    for (size_t i = 0; i < ContextSuperposition::N_MODES * ContextSuperposition::KEY_DIM; ++i)
        out[idx++] = superposition_.keys[i];
    // ContextSuperposition enabled (1)
    out[idx++] = superposition_.enabled;

    // idx = 9933
}

void ConceptModel::from_flat(const std::array<double, CM_FLAT_SIZE>& in) {
    size_t idx = 0;

    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) W_[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) b_[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) e_init_[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) c_init_[i] = in[idx++];

    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) state_.dW_momentum[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) state_.db_momentum[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) state_.dW_variance[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) state_.db_variance[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) state_.e_grad_accum[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) state_.c_grad_accum[i] = in[idx++];
    state_.timestep = in[idx++];
    state_.last_loss = in[idx++];
    state_.best_loss = in[idx++];
    state_.total_samples = in[idx++];
    state_.reserved_scalar = in[idx++];
    for (size_t i = 0; i < 55; ++i) state_.reserved[i] = in[idx++];

    for (size_t i = 0; i < MultiHeadBilinear::TOTAL_PARAMS; ++i) multihead_.params[i] = in[idx++];

    for (size_t i = 0; i < FlexKAN::TOTAL_PARAMS; ++i) kan_.params[i] = in[idx++];

    patterns_.shared_parent = in[idx++];
    patterns_.transitive_causation = in[idx++];
    patterns_.missing_link = in[idx++];
    patterns_.weak_strengthening = in[idx++];
    patterns_.contradictory_signal = in[idx++];
    patterns_.chain_hypothesis = in[idx++];
    for (size_t i = 0; i < 9; ++i) patterns_.reserved[i] = in[idx++];

    for (size_t i = 0; i < 25; ++i) reserved_[i] = in[idx++];

    // idx = 1900

    // ConvergencePort W (3904)
    for (size_t i = 0; i < ConvergencePort::W_SIZE; ++i) conv_port_.W[i] = in[idx++];

    // ConvergencePort b (32)
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) conv_port_.b[i] = in[idx++];

    // ConvergencePort W_gate (3904)
    for (size_t i = 0; i < ConvergencePort::W_GATE_SIZE; ++i) conv_port_.W_gate[i] = in[idx++];

    // ConvergencePort b_gate (32)
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) conv_port_.b_gate[i] = in[idx++];

    // idx = 9772

    // ContextSuperposition u (64)
    for (size_t i = 0; i < ContextSuperposition::N_MODES * CORE_DIM; ++i)
        superposition_.u[i] = in[idx++];
    // ContextSuperposition v (64)
    for (size_t i = 0; i < ContextSuperposition::N_MODES * CORE_DIM; ++i)
        superposition_.v[i] = in[idx++];
    // ContextSuperposition keys (32)
    for (size_t i = 0; i < ContextSuperposition::N_MODES * ContextSuperposition::KEY_DIM; ++i)
        superposition_.keys[i] = in[idx++];
    // ContextSuperposition enabled (1)
    superposition_.enabled = in[idx++];

    // idx = 9933
}

} // namespace brain19

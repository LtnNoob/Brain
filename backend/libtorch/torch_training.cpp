// libtorch/torch_training.cpp — LibTorch Deep KAN v2 training loop
#include "torch_training.hpp"
#include "torch_kan.hpp"

#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace brain19 {
namespace libtorch {

// =============================================================================
// Helper: sigmoid for LSTM-style gating
// =============================================================================
static inline double gate_sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// =============================================================================
// Helper: precompute hidden states h[0:90] from training data
// =============================================================================
// Blocks 1-3 are deterministic given embedding tables.
// Default (fixed): Block1 h=0.8h+0.2emb, Block2 h=0.9h+0.1flex, Block3 h*=0.95
// LSTM mode: sigmoid-gated mixing based on h·emb similarity
//
// Returns per-sample token ranges so we can split train/val at sample level.

struct PrecomputedData {
    std::vector<float> all_h;         // [N_tok * 90]
    std::vector<int64_t> all_toks;    // [N_tok] original token ids
    std::vector<int64_t> all_targets; // [N_tok] compressed target indices
    // Per-sample: [start_token_idx, num_tokens]
    std::vector<size_t> sample_tok_offsets;
    std::vector<size_t> sample_tok_counts;
    size_t n_tokens = 0;
};

static PrecomputedData precompute_hidden_states(const cuda::TrainingData& data) {
    PrecomputedData pd;
    const size_t FUSED_BASE = data.FUSED_BASE;  // 64
    const size_t fd = data.flex_dim;             // 16
    const size_t H_90 = 90;
    const size_t V = data.V;

    size_t total_toks = 0;
    for (size_t s = 0; s < data.num_samples; ++s)
        total_toks += data.sample_lengths[s];

    pd.all_h.reserve(total_toks * H_90);
    pd.all_toks.reserve(total_toks);
    pd.all_targets.reserve(total_toks);
    pd.sample_tok_offsets.resize(data.num_samples);
    pd.sample_tok_counts.resize(data.num_samples);

    std::vector<double> h(H_90, 0.0);

    for (size_t s = 0; s < data.num_samples; ++s) {
        pd.sample_tok_offsets[s] = pd.n_tokens;

        const double* emb = &data.embeddings[s * data.H];
        for (size_t i = 0; i < H_90 && i < data.H; ++i)
            h[i] = emb[i];
        for (size_t i = data.H; i < H_90; ++i)
            h[i] = 0.0;

        size_t off = data.sample_offsets[s];
        size_t len = data.sample_lengths[s];
        size_t sample_count = 0;

        for (size_t t = 0; t < len; ++t) {
            uint16_t tok = data.all_tokens[off + t];
            if (tok >= V) continue;

            // Next-token prediction: target = compress[tok_{t+1}]
            // Skip last token (no next token to predict)
            if (t + 1 < len) {
                uint16_t next_tok = data.all_tokens[off + t + 1];
                if (next_tok < V) {
                    size_t next_ca = data.compress[next_tok];

                    for (size_t i = 0; i < H_90; ++i)
                        pd.all_h.push_back((float)h[i]);
                    pd.all_toks.push_back((int64_t)tok);
                    pd.all_targets.push_back((int64_t)next_ca);
                    pd.n_tokens++;
                    sample_count++;
                }
            }

            // Evolve hidden state (3 blocks) — always, even for last token
            if (data.use_lstm_gates) {
                // LSTM-style: content-dependent sigmoid gating
                // Gate = σ(h·emb / sqrt(dim)) → modulates mixing coefficient
                // Block 1: base=0.2, range [0.1, 0.3]
                double dot1 = 0;
                for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                    double emb_val = (tok * FUSED_BASE + i < data.emb_table.size())
                        ? data.emb_table[tok * FUSED_BASE + i] : 0.0;
                    dot1 += h[i] * emb_val;
                }
                double gate1 = gate_sigmoid(dot1 / 8.0);  // sqrt(64)=8
                double alpha1 = 0.1 + gate1 * 0.2;        // [0.1, 0.3]
                for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                    double emb_val = (tok * FUSED_BASE + i < data.emb_table.size())
                        ? data.emb_table[tok * FUSED_BASE + i] : 0.0;
                    h[i] = h[i] * (1.0 - alpha1) + emb_val * alpha1;
                }
                // Block 2: base=0.1, range [0.05, 0.15]
                double dot2 = 0;
                for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                    double flex_val = (tok * fd + i < data.flex_table.size())
                        ? data.flex_table[tok * fd + i] : 0.0;
                    dot2 += h[FUSED_BASE + i] * flex_val;
                }
                double gate2 = gate_sigmoid(dot2 / 4.0);  // sqrt(16)=4
                double alpha2 = 0.05 + gate2 * 0.1;       // [0.05, 0.15]
                for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                    double flex_val = (tok * fd + i < data.flex_table.size())
                        ? data.flex_table[tok * fd + i] : 0.0;
                    h[FUSED_BASE + i] = h[FUSED_BASE + i] * (1.0 - alpha2) + flex_val * alpha2;
                }
                // Block 3: content-dependent decay [0.93, 0.97]
                double h3_norm = 0;
                for (size_t i = FUSED_BASE + fd; i < H_90; ++i)
                    h3_norm += h[i] * h[i];
                double decay = 0.93 + gate_sigmoid(-h3_norm + 1.0) * 0.04;
                for (size_t i = FUSED_BASE + fd; i < H_90; ++i)
                    h[i] *= decay;
            } else {
                // Fixed coefficients (original)
                for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                    double emb_val = (tok * FUSED_BASE + i < data.emb_table.size())
                        ? data.emb_table[tok * FUSED_BASE + i] : 0.0;
                    h[i] = h[i] * 0.8 + emb_val * 0.2;
                }
                for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                    double flex_val = (tok * fd + i < data.flex_table.size())
                        ? data.flex_table[tok * fd + i] : 0.0;
                    h[FUSED_BASE + i] = h[FUSED_BASE + i] * 0.9 + flex_val * 0.1;
                }
                for (size_t i = FUSED_BASE + fd; i < H_90; ++i) {
                    h[i] *= 0.95;
                }
            }
        }

        pd.sample_tok_counts[s] = sample_count;
    }

    return pd;
}

// =============================================================================
// Helper: load C++ double weights into torch float32 tensor
// =============================================================================

static torch::Tensor vec_to_tensor(const std::vector<double>& src, std::vector<long> shape) {
    long total = 1;
    for (auto s : shape) total *= s;
    auto t = torch::empty(shape);
    auto acc = t.data_ptr<float>();
    for (long i = 0; i < total && i < (long)src.size(); ++i)
        acc[i] = (float)src[i];
    return t;
}

// =============================================================================
// Helper: copy torch float32 tensor back to C++ double vector
// =============================================================================

static void tensor_to_vec(const torch::Tensor& t, std::vector<double>& dst) {
    auto cpu_t = t.cpu().contiguous();
    auto acc = cpu_t.data_ptr<float>();
    long n = cpu_t.numel();
    dst.resize((size_t)n);
    for (long i = 0; i < n; ++i)
        dst[i] = (double)acc[i];
}

// =============================================================================
// Helper: load existing KAN weights into the model
// =============================================================================

static void load_kan_weights(DeepKANv2Decoder& model,
                             const cuda::DeepKANWeights& dkw,
                             const ConvergencePortData& cpd,
                             size_t V) {
    auto& m = *model;

    // KAN Layer 1: 90→256, G=8, k=3, basis=11
    {
        auto& l = *m.kan_l1;
        l.spline_weights.data().copy_(
            vec_to_tensor(dkw.k1_weights, {256, 90 * 11}));
        l.residual_W.data().copy_(
            vec_to_tensor(dkw.k1_residual, {90, 256}));
        l.ln_gamma.data().copy_(
            vec_to_tensor(dkw.k1_gamma, {256}));
        l.ln_beta.data().copy_(
            vec_to_tensor(dkw.k1_beta, {256}));
        l.knots.copy_(vec_to_tensor(dkw.k1_knots, {(long)dkw.k1_knots.size()}));
    }

    // KAN Layer 2: 288→128, G=5, k=3, basis=8
    {
        auto& l = *m.kan_l2;
        l.spline_weights.data().copy_(
            vec_to_tensor(dkw.k2_weights, {128, 288 * 8}));
        l.residual_W.data().copy_(
            vec_to_tensor(dkw.k2_residual, {288, 128}));
        l.ln_gamma.data().copy_(
            vec_to_tensor(dkw.k2_gamma, {128}));
        l.ln_beta.data().copy_(
            vec_to_tensor(dkw.k2_beta, {128}));
        l.knots.copy_(vec_to_tensor(dkw.k2_knots, {(long)dkw.k2_knots.size()}));
    }

    // KAN Layer 3: 128→128, G=5, k=3, basis=8
    {
        auto& l = *m.kan_l3;
        l.spline_weights.data().copy_(
            vec_to_tensor(dkw.k3_weights, {128, 128 * 8}));
        l.residual_W.data().copy_(
            vec_to_tensor(dkw.k3_residual, {128, 128}));
        l.ln_gamma.data().copy_(
            vec_to_tensor(dkw.k3_gamma, {128}));
        l.ln_beta.data().copy_(
            vec_to_tensor(dkw.k3_beta, {128}));
        l.knots.copy_(vec_to_tensor(dkw.k3_knots, {(long)dkw.k3_knots.size()}));
    }

    // Output projection: dkw.W_a [128 * VA] row-major -> nn::Linear [VA, 128]
    {
        long VA = (long)m.VA_;
        auto wa_tensor = vec_to_tensor(dkw.W_a, {128, VA});
        m.output->weight.data().copy_(wa_tensor.t());
    }

    // Shared ConvergencePort: embedding + linear
    if (!cpd.conv_emb_weights.empty()) {
        auto emb_w = vec_to_tensor(cpd.conv_emb_weights, {(long)V, CONV_EMB_DIM});
        m.conv_emb->weight.data().copy_(emb_w);
    }
    if (!cpd.conv_linear_W.empty()) {
        auto lw = vec_to_tensor(cpd.conv_linear_W,
            {CONV_OUTPUT_DIM, (long)(CONV_INPUT_DIM + CONV_EMB_DIM)});
        m.conv_linear->weight.data().copy_(lw);
    }
    if (!cpd.conv_linear_b.empty()) {
        auto lb = vec_to_tensor(cpd.conv_linear_b, {CONV_OUTPUT_DIM});
        m.conv_linear->bias.data().copy_(lb);
    }
}

// =============================================================================
// Helper: extract trained weights back to C++ structs
// =============================================================================

static void extract_kan_weights(const DeepKANv2Decoder& model,
                                cuda::DeepKANWeights& dkw,
                                ConvergencePortData& cpd) {
    auto& m = *model;

    tensor_to_vec(m.kan_l1->spline_weights, dkw.k1_weights);
    tensor_to_vec(m.kan_l1->residual_W, dkw.k1_residual);
    tensor_to_vec(m.kan_l1->ln_gamma, dkw.k1_gamma);
    tensor_to_vec(m.kan_l1->ln_beta, dkw.k1_beta);

    tensor_to_vec(m.kan_l2->spline_weights, dkw.k2_weights);
    tensor_to_vec(m.kan_l2->residual_W, dkw.k2_residual);
    tensor_to_vec(m.kan_l2->ln_gamma, dkw.k2_gamma);
    tensor_to_vec(m.kan_l2->ln_beta, dkw.k2_beta);

    tensor_to_vec(m.kan_l3->spline_weights, dkw.k3_weights);
    tensor_to_vec(m.kan_l3->residual_W, dkw.k3_residual);
    tensor_to_vec(m.kan_l3->ln_gamma, dkw.k3_gamma);
    tensor_to_vec(m.kan_l3->ln_beta, dkw.k3_beta);

    {
        auto wa = m.output->weight.data().t().contiguous();
        tensor_to_vec(wa, dkw.W_a);
    }

    // Shared ConvergencePort
    tensor_to_vec(m.conv_emb->weight, cpd.conv_emb_weights);
    tensor_to_vec(m.conv_linear->weight, cpd.conv_linear_W);
    tensor_to_vec(m.conv_linear->bias, cpd.conv_linear_b);
}

// =============================================================================
// Per-input-dim LR scaling for KAN L1 spline weights
// =============================================================================

static void scale_kan_l1_gradients(DeepKANv2Decoder& model,
                                   const std::vector<double>& lr_scale) {
    if (lr_scale.empty()) return;
    auto& l = *model->kan_l1;

    if (l.spline_weights.grad().defined()) {
        auto& grad = l.spline_weights.mutable_grad();
        size_t bs = l.basis_size_;
        for (size_t d = 0; d < 90 && d < lr_scale.size(); ++d) {
            if (std::abs(lr_scale[d] - 1.0) > 1e-8) {
                grad.narrow(1, (long)(d * bs), (long)bs).mul_((float)lr_scale[d]);
            }
        }
    }

    if (l.residual_W.grad().defined()) {
        auto& grad = l.residual_W.mutable_grad();
        for (size_t d = 0; d < 90 && d < lr_scale.size(); ++d) {
            if (std::abs(lr_scale[d] - 1.0) > 1e-8) {
                grad.narrow(0, (long)d, 1).mul_((float)lr_scale[d]);
            }
        }
    }
}

// =============================================================================
// Helper: compute loss on a set of token indices (no_grad)
// =============================================================================

static double eval_loss(DeepKANv2Decoder& model,
                        const torch::Tensor& h_tensor,
                        const torch::Tensor& tok_tensor,
                        const torch::Tensor& target_tensor,
                        const std::vector<int64_t>& token_indices,
                        size_t batch_size,
                        torch::Device device) {
    torch::NoGradGuard no_grad;
    model->eval();

    size_t N = token_indices.size();
    if (N == 0) return 0.0;

    auto idx_tensor = torch::from_blob(
        const_cast<int64_t*>(token_indices.data()), {(long)N},
        torch::kInt64).to(device);

    double total_loss = 0.0;
    size_t total_tokens = 0;
    size_t num_batches = (N + batch_size - 1) / batch_size;

    for (size_t b = 0; b < num_batches; ++b) {
        size_t start = b * batch_size;
        size_t end = std::min(start + batch_size, N);
        size_t bs = end - start;

        auto batch_idx = idx_tensor.narrow(0, (long)start, (long)bs);
        auto h_batch = h_tensor.index_select(0, batch_idx);
        auto tok_batch = tok_tensor.index_select(0, batch_idx);
        auto target_batch = target_tensor.index_select(0, batch_idx);

        auto logits = model->forward(h_batch, tok_batch);
        auto loss = torch::nn::functional::cross_entropy(logits, target_batch);

        total_loss += loss.item<double>() * bs;
        total_tokens += bs;
    }

    model->train();
    return total_loss / std::max(total_tokens, (size_t)1);
}

// =============================================================================
// train_deep_kan_v2 — Main training entry point
// =============================================================================

bool train_deep_kan_v2(const cuda::TrainingData& data,
                       cuda::DeepKANWeights& dkw,
                       ConvergencePortData& cpd,
                       const DeepKANv2Config& config,
                       cuda::TrainingResult& result) {
    auto t_start = std::chrono::steady_clock::now();

    // ── Step 1: Precompute hidden states ──
    std::cerr << "[LibTorch] Precomputing hidden states (90D, Blocks 1-3)...\n";
    auto pd = precompute_hidden_states(data);
    size_t N = pd.n_tokens;
    size_t num_samples = data.num_samples;

    if (N == 0) {
        std::cerr << "[LibTorch] No tokens to train on!\n";
        return false;
    }

    auto t_precomp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[LibTorch]   Precomputed " << N << " tokens from "
              << num_samples << " samples (" << t_precomp << "ms)\n";

    // ── Step 1b: Train/Validation split (80/20 at SAMPLE level) ──
    std::vector<size_t> sample_order(num_samples);
    std::iota(sample_order.begin(), sample_order.end(), 0);
    {
        std::mt19937 split_rng(7777);  // fixed seed for reproducible split
        std::shuffle(sample_order.begin(), sample_order.end(), split_rng);
    }
    size_t n_train_samples = (num_samples * 80) / 100;
    size_t n_val_samples = num_samples - n_train_samples;

    // Collect token indices for train and val sets
    std::vector<int64_t> train_token_indices;
    std::vector<int64_t> val_token_indices;
    train_token_indices.reserve(N);
    val_token_indices.reserve(N / 4);

    for (size_t i = 0; i < n_train_samples; ++i) {
        size_t s = sample_order[i];
        size_t tok_off = pd.sample_tok_offsets[s];
        size_t tok_cnt = pd.sample_tok_counts[s];
        for (size_t t = 0; t < tok_cnt; ++t)
            train_token_indices.push_back((int64_t)(tok_off + t));
    }
    for (size_t i = n_train_samples; i < num_samples; ++i) {
        size_t s = sample_order[i];
        size_t tok_off = pd.sample_tok_offsets[s];
        size_t tok_cnt = pd.sample_tok_counts[s];
        for (size_t t = 0; t < tok_cnt; ++t)
            val_token_indices.push_back((int64_t)(tok_off + t));
    }

    size_t N_train = train_token_indices.size();
    size_t N_val = val_token_indices.size();
    std::cerr << "[LibTorch]   Split: " << n_train_samples << " train samples ("
              << N_train << " tokens), " << n_val_samples << " val samples ("
              << N_val << " tokens)\n";

    // ── Step 2: Build model ──
    size_t VA = data.VA;
    size_t V = data.V;

    DeepKANv2Decoder model(VA, V, config.dropout_p);
    load_kan_weights(model, dkw, cpd, V);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, 0);
        std::cerr << "[LibTorch]   Using CUDA\n";
    } else {
        std::cerr << "[LibTorch]   CUDA not available, using CPU\n";
    }
    model->to(device);

    // ── Build tensors (all tokens, indexed by train/val splits) ──
    auto h_tensor = torch::from_blob(pd.all_h.data(), {(long)N, 90},
        torch::kFloat32).clone().to(device);
    auto tok_tensor = torch::from_blob(pd.all_toks.data(), {(long)N},
        torch::kInt64).clone().to(device);
    auto target_tensor = torch::from_blob(pd.all_targets.data(), {(long)N},
        torch::kInt64).clone().to(device);

    // ── Step 3: Setup optimizer with parameter groups ──
    std::vector<torch::Tensor> output_params;
    std::vector<torch::Tensor> kan_params;
    std::vector<torch::Tensor> conv_params;

    for (auto& item : model->named_parameters()) {
        const auto& name = item.key();
        auto& param = item.value();
        if (name.find("conv_emb") != std::string::npos ||
            name.find("conv_linear") != std::string::npos) {
            conv_params.push_back(param);
        } else if (name.find("output.") != std::string::npos) {
            output_params.push_back(param);
        } else {
            kan_params.push_back(param);
        }
    }

    auto make_group = [&config](std::vector<torch::Tensor>& params, double lr) {
        auto opts = std::make_unique<torch::optim::AdamOptions>(lr);
        opts->betas(std::make_tuple(0.9, 0.999));
        opts->weight_decay(config.weight_decay);
        return torch::optim::OptimizerParamGroup(params, std::move(opts));
    };

    torch::optim::Adam optimizer({
        make_group(output_params, config.lr_output),
        make_group(kan_params, config.lr_kan),
        make_group(conv_params, config.lr_conv)
    });

    // ── Step 4: Training loop with train/val monitoring ──
    size_t batch_size = config.batch_size;
    size_t num_train_batches = (N_train + batch_size - 1) / batch_size;

    double best_val_loss = 1e9;
    double best_train_loss = 1e9;
    size_t epochs_without_improvement = 0;
    size_t best_epoch = 0;
    std::mt19937 rng(42);

    // Save best model state for early stopping restore
    std::vector<char> best_model_state;

    std::cerr << "[LibTorch]   Training: " << config.num_epochs << " epochs, "
              << N_train << " train tokens, " << N_val << " val tokens, "
              << "batch=" << batch_size << ", VA=" << VA
              << ", dropout=" << config.dropout_p
              << ", patience=" << config.patience << "\n";

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        // Cosine LR decay
        double progress = (double)epoch / std::max(config.num_epochs - 1, (size_t)1);
        double cos_mult = 0.5 * (1.0 + std::cos(progress * M_PI));
        double lr_mult = 0.1 + 0.9 * cos_mult;

        // Update LRs
        {
            auto& groups = optimizer.param_groups();
            static_cast<torch::optim::AdamOptions&>(groups[0].options()).lr(config.lr_output * lr_mult);

            double kan_lr = (epoch < config.warmup_epochs) ? 0.0 : config.lr_kan * lr_mult;
            double conv_lr = (epoch < config.warmup_epochs) ? 0.0 : config.lr_conv * lr_mult;
            static_cast<torch::optim::AdamOptions&>(groups[1].options()).lr(kan_lr);
            static_cast<torch::optim::AdamOptions&>(groups[2].options()).lr(conv_lr);
        }

        // Shuffle train indices
        std::shuffle(train_token_indices.begin(), train_token_indices.end(), rng);
        auto train_idx_tensor = torch::from_blob(
            train_token_indices.data(), {(long)N_train}, torch::kInt64).to(device);

        double epoch_loss = 0.0;
        size_t epoch_tokens = 0;
        model->train();

        for (size_t b = 0; b < num_train_batches; ++b) {
            size_t start = b * batch_size;
            size_t end = std::min(start + batch_size, N_train);
            size_t bs = end - start;

            auto batch_idx = train_idx_tensor.narrow(0, (long)start, (long)bs);
            auto h_batch = h_tensor.index_select(0, batch_idx);
            auto tok_batch = tok_tensor.index_select(0, batch_idx);
            auto target_batch = target_tensor.index_select(0, batch_idx);

            auto logits = model->forward(h_batch, tok_batch);
            auto loss = torch::nn::functional::cross_entropy(logits, target_batch);

            optimizer.zero_grad();
            loss.backward();

            if (epoch >= config.warmup_epochs && !config.lr_scale.empty()) {
                scale_kan_l1_gradients(model, config.lr_scale);
            }

            optimizer.step();

            epoch_loss += loss.item<double>() * bs;
            epoch_tokens += bs;
        }

        // Evaluate both train and val loss at same model state (post-epoch, no_grad)
        // This gives apples-to-apples comparison — running avg during updates is misleading
        bool do_eval = (epoch % 5 == 0) || (epoch == config.num_epochs - 1);

        auto t_now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();

        if (do_eval) {
            double train_loss = eval_loss(model, h_tensor, tok_tensor, target_tensor,
                                          train_token_indices, batch_size, device);
            double val_loss = eval_loss(model, h_tensor, tok_tensor, target_tensor,
                                        val_token_indices, batch_size, device);
            if (train_loss < best_train_loss) best_train_loss = train_loss;

            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_without_improvement = 0;

                // Save best model state
                std::ostringstream oss;
                torch::serialize::OutputArchive archive;
                model->save(archive);
                archive.save_to(oss);
                auto s = oss.str();
                best_model_state.assign(s.begin(), s.end());
            } else {
                epochs_without_improvement += 5;  // eval every 5 epochs
            }

            double gap = val_loss - train_loss;
            std::cerr << "[LibTorch]   epoch " << epoch << "/" << config.num_epochs
                      << " train=" << train_loss
                      << " val=" << val_loss
                      << " gap=" << gap
                      << " best_val=" << best_val_loss
                      << " (" << t_now << "ms)\n";

            // Early stopping: val-gap overfitting detection
            if (config.max_val_gap > 0 && epoch > 20 && gap > config.max_val_gap) {
                std::cerr << "[LibTorch]   Val-gap early stop: gap=" << gap
                          << " > max_val_gap=" << config.max_val_gap
                          << " at epoch " << epoch << " (best @ " << best_epoch << ")\n";
                break;
            }

            // Early stopping: patience-based
            if (config.patience > 0 && epochs_without_improvement >= config.patience) {
                std::cerr << "[LibTorch]   Early stopping: no improvement for "
                          << epochs_without_improvement << " epochs (best @ epoch "
                          << best_epoch << ")\n";
                break;
            }
        } else {
            // Quick running-avg log (not comparable to val, just progress indicator)
            double running_avg = epoch_loss / std::max(epoch_tokens, (size_t)1);
            std::cerr << "[LibTorch]   epoch " << epoch << "/" << config.num_epochs
                      << " running=" << running_avg
                      << " (" << t_now << "ms)\n";
        }
    }

    // ── Step 5: Restore best model and extract weights ──
    if (!best_model_state.empty()) {
        std::istringstream iss(std::string(best_model_state.begin(), best_model_state.end()));
        torch::serialize::InputArchive archive;
        archive.load_from(iss);
        model->load(archive);
        std::cerr << "[LibTorch]   Restored best model from epoch " << best_epoch << "\n";
    }
    model->to(torch::kCPU);
    extract_kan_weights(model, dkw, cpd);

    result.best_loss = best_val_loss;

    auto t_total = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[LibTorch]   Training complete: best_train=" << best_train_loss
              << " best_val=" << best_val_loss
              << " (" << t_total << "ms total)\n";

    return true;
}

// =============================================================================
// Autoregressive generation with trained DeepKAN v2
// =============================================================================

GenerateResult generate_deep_kan_v2(
    const cuda::DeepKANWeights& dkw,
    const ConvergencePortData& cpd,
    size_t VA, size_t V,
    const std::vector<double>& emb_table,
    const std::vector<double>& flex_table,
    size_t FUSED_BASE, size_t flex_dim,
    const std::vector<uint16_t>& active_tokens,
    const std::vector<float>& initial_h,
    uint16_t start_token,
    size_t max_tokens,
    bool use_lstm_gates)
{
    GenerateResult gr;
    const size_t H_90 = 90;

    // Build model and load trained weights (dropout=0 at inference)
    DeepKANv2Decoder model(VA, V, 0.0);
    load_kan_weights(model, dkw, cpd, V);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA, 0);
    model->to(device);
    model->eval();

    torch::NoGradGuard no_grad;

    // Working hidden state
    std::vector<float> h(initial_h.begin(), initial_h.end());
    h.resize(H_90, 0.0f);
    int64_t tok = (int64_t)start_token;

    for (size_t t = 0; t < max_tokens; ++t) {
        // Forward pass: single sample
        auto h_tensor = torch::from_blob(h.data(), {1, (long)H_90},
            torch::kFloat32).clone().to(device);
        auto tok_tensor = torch::tensor({tok}, torch::kInt64).to(device);

        auto logits = model->forward(h_tensor, tok_tensor);  // [1, VA]
        auto probs = torch::softmax(logits, 1);
        auto best_active = probs.argmax(1).item<int64_t>();
        double best_prob = probs[0][best_active].item<double>();

        // Map active index → real token ID
        if ((size_t)best_active >= VA) break;
        uint16_t real_tok = active_tokens[(size_t)best_active];

        gr.tokens.push_back(real_tok);
        gr.probs.push_back(best_prob);

        // Stop on EOS (token 2) or low confidence after warmup
        if (real_tok == 2) break;
        if (best_prob < 0.005 && t > 5) break;

        // Set tok for next iteration (real token ID for conv_emb)
        tok = (int64_t)real_tok;

        // Evolve hidden state (3-block, matching training)
        if (use_lstm_gates) {
            // LSTM-style gating (must match precompute_hidden_states)
            double dot1 = 0;
            for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                double emb_val = (real_tok * FUSED_BASE + i < emb_table.size())
                    ? emb_table[real_tok * FUSED_BASE + i] : 0.0;
                dot1 += h[i] * emb_val;
            }
            double gate1 = gate_sigmoid(dot1 / 8.0);
            double alpha1 = 0.1 + gate1 * 0.2;
            for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                double emb_val = (real_tok * FUSED_BASE + i < emb_table.size())
                    ? emb_table[real_tok * FUSED_BASE + i] : 0.0;
                h[i] = (float)(h[i] * (1.0 - alpha1) + emb_val * alpha1);
            }
            double dot2 = 0;
            for (size_t i = 0; i < flex_dim && (FUSED_BASE + i) < H_90; ++i) {
                double flex_val = (real_tok * flex_dim + i < flex_table.size())
                    ? flex_table[real_tok * flex_dim + i] : 0.0;
                dot2 += h[FUSED_BASE + i] * flex_val;
            }
            double gate2 = gate_sigmoid(dot2 / 4.0);
            double alpha2 = 0.05 + gate2 * 0.1;
            for (size_t i = 0; i < flex_dim && (FUSED_BASE + i) < H_90; ++i) {
                double flex_val = (real_tok * flex_dim + i < flex_table.size())
                    ? flex_table[real_tok * flex_dim + i] : 0.0;
                h[FUSED_BASE + i] = (float)(h[FUSED_BASE + i] * (1.0 - alpha2) + flex_val * alpha2);
            }
            double h3_norm = 0;
            for (size_t i = FUSED_BASE + flex_dim; i < H_90; ++i)
                h3_norm += h[i] * h[i];
            double decay = 0.93 + gate_sigmoid(-h3_norm + 1.0) * 0.04;
            for (size_t i = FUSED_BASE + flex_dim; i < H_90; ++i)
                h[i] *= (float)decay;
        } else {
            // Block 1: [0, FUSED_BASE) token embedding
            for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                double emb_val = (real_tok * FUSED_BASE + i < emb_table.size())
                    ? emb_table[real_tok * FUSED_BASE + i] : 0.0;
                h[i] = (float)(h[i] * 0.8 + emb_val * 0.2);
            }
            // Block 2: [FUSED_BASE, FUSED_BASE+flex_dim) flex detail
            for (size_t i = 0; i < flex_dim && (FUSED_BASE + i) < H_90; ++i) {
                double flex_val = (real_tok * flex_dim + i < flex_table.size())
                    ? flex_table[real_tok * flex_dim + i] : 0.0;
                h[FUSED_BASE + i] = (float)(h[FUSED_BASE + i] * 0.9 + flex_val * 0.1);
            }
            // Block 3: [FUSED_BASE+flex_dim, H_90) slow decay
            for (size_t i = FUSED_BASE + flex_dim; i < H_90; ++i) {
                h[i] *= 0.95f;
            }
        }
    }

    return gr;
}

// =============================================================================
// Concept prediction: precompute hidden states with concept-based h evolution
// =============================================================================
// Same 3-block evolution as token path, but uses concept embeddings:
//   Block 1 [0:64]:  h = 0.8*h + 0.2*concept_emb_64d[concept]
//   Block 2 [64:80]: h = 0.9*h + 0.1*concept_flex_16d[concept]
//   Block 3 [80:90]: h *= 0.95
//
// Training pair: (h_t, concept_16d[t]) → target = concept[t+1]

struct ConceptPrecomputedData {
    std::vector<float> all_h;           // [N_pairs * 90]
    std::vector<float> all_concept_emb; // [N_pairs * 16] for ConvergencePort input
    std::vector<int64_t> all_targets;   // [N_pairs] target concept index
    std::vector<float> all_trust;       // [N_pairs] trust weight
    std::vector<size_t> sample_offsets;
    std::vector<size_t> sample_counts;
    size_t n_pairs = 0;
};

static ConceptPrecomputedData precompute_concept_hidden_states(
    const ConceptTrainingData& data)
{
    ConceptPrecomputedData pd;
    const size_t H_90 = 90;
    const size_t FUSED_BASE = 64;
    const size_t fd = 16;

    // Count total pairs for reservation
    size_t total = 0;
    for (size_t s = 0; s < data.num_samples; ++s) {
        if (data.seq_lengths[s] > 1)
            total += data.seq_lengths[s] - 1;
    }

    pd.all_h.reserve(total * H_90);
    pd.all_concept_emb.reserve(total * CONV_EMB_DIM);
    pd.all_targets.reserve(total);
    pd.all_trust.reserve(total);
    pd.sample_offsets.resize(data.num_samples);
    pd.sample_counts.resize(data.num_samples);

    std::vector<double> h(H_90, 0.0);

    for (size_t s = 0; s < data.num_samples; ++s) {
        pd.sample_offsets[s] = pd.n_pairs;

        // Initialize h from sample's fused vector
        for (size_t i = 0; i < H_90; ++i)
            h[i] = data.initial_h[s * H_90 + i];

        size_t off = data.seq_offsets[s];
        size_t len = data.seq_lengths[s];
        size_t count = 0;

        for (size_t t = 0; t < len; ++t) {
            int64_t ci = data.concept_seqs[off + t];
            if (ci < 0 || (size_t)ci >= data.num_concepts) continue;

            // Next-concept prediction: target = concept[t+1]
            if (t + 1 < len) {
                int64_t next_ci = data.concept_seqs[off + t + 1];
                if (next_ci >= 0 && (size_t)next_ci < data.num_concepts) {
                    // Store hidden state
                    for (size_t i = 0; i < H_90; ++i)
                        pd.all_h.push_back((float)h[i]);

                    // Store current concept's 16D core embedding for CM input
                    // concept_matrix is 32D (core+detail), but CM only uses first 16D
                    for (size_t d = 0; d < CONV_EMB_DIM; ++d)
                        pd.all_concept_emb.push_back(
                            (float)data.concept_matrix[(size_t)ci * CONCEPT_PROJ_DIM + d]);

                    pd.all_targets.push_back(next_ci);
                    pd.all_trust.push_back((float)data.trust_weights[s]);
                    pd.n_pairs++;
                    count++;
                }
            }

            // Evolve hidden state using concept embeddings
            if (data.use_lstm_gates) {
                // LSTM-style: content-dependent sigmoid gating
                // Block 1: gate from h·emb similarity
                double dot1 = 0;
                for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                    double v = data.concept_emb_64d[(size_t)ci * FUSED_BASE + i];
                    dot1 += h[i] * v;
                }
                double gate1 = gate_sigmoid(dot1 / 8.0);
                double alpha1 = 0.1 + gate1 * 0.2;
                for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                    double v = data.concept_emb_64d[(size_t)ci * FUSED_BASE + i];
                    h[i] = h[i] * (1.0 - alpha1) + v * alpha1;
                }
                // Block 2
                double dot2 = 0;
                for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                    double v = data.concept_flex_16d[(size_t)ci * fd + i];
                    dot2 += h[FUSED_BASE + i] * v;
                }
                double gate2 = gate_sigmoid(dot2 / 4.0);
                double alpha2 = 0.05 + gate2 * 0.1;
                for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                    double v = data.concept_flex_16d[(size_t)ci * fd + i];
                    h[FUSED_BASE + i] = h[FUSED_BASE + i] * (1.0 - alpha2) + v * alpha2;
                }
                // Block 3: content-dependent decay
                double h3_norm = 0;
                for (size_t i = FUSED_BASE + fd; i < H_90; ++i)
                    h3_norm += h[i] * h[i];
                double decay = 0.93 + gate_sigmoid(-h3_norm + 1.0) * 0.04;
                for (size_t i = FUSED_BASE + fd; i < H_90; ++i)
                    h[i] *= decay;
            } else {
                // Fixed coefficients (original)
                for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                    double v = data.concept_emb_64d[(size_t)ci * FUSED_BASE + i];
                    h[i] = h[i] * 0.8 + v * 0.2;
                }
                for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                    double v = data.concept_flex_16d[(size_t)ci * fd + i];
                    h[FUSED_BASE + i] = h[FUSED_BASE + i] * 0.9 + v * 0.1;
                }
                for (size_t i = FUSED_BASE + fd; i < H_90; ++i) {
                    h[i] *= 0.95;
                }
            }
        }

        pd.sample_counts[s] = count;
    }

    return pd;
}

// =============================================================================
// Helper: load KAN core weights (L1/L2/L3 + conv_linear + k1_proj + concept_proj)
// Skips conv_emb and output (unused in concept mode)
// =============================================================================

static void load_concept_model_weights(DeepKANv2Decoder& model,
                                        const cuda::DeepKANWeights& dkw,
                                        const ConvergencePortData& cpd,
                                        const ConceptWeights& cw) {
    auto& m = *model;

    // KAN Layer 1
    m.kan_l1->spline_weights.data().copy_(
        vec_to_tensor(dkw.k1_weights, {256, 90 * 11}));
    m.kan_l1->residual_W.data().copy_(
        vec_to_tensor(dkw.k1_residual, {90, 256}));
    m.kan_l1->ln_gamma.data().copy_(
        vec_to_tensor(dkw.k1_gamma, {256}));
    m.kan_l1->ln_beta.data().copy_(
        vec_to_tensor(dkw.k1_beta, {256}));
    m.kan_l1->knots.copy_(
        vec_to_tensor(dkw.k1_knots, {(long)dkw.k1_knots.size()}));

    // KAN Layer 2
    m.kan_l2->spline_weights.data().copy_(
        vec_to_tensor(dkw.k2_weights, {128, 288 * 8}));
    m.kan_l2->residual_W.data().copy_(
        vec_to_tensor(dkw.k2_residual, {288, 128}));
    m.kan_l2->ln_gamma.data().copy_(
        vec_to_tensor(dkw.k2_gamma, {128}));
    m.kan_l2->ln_beta.data().copy_(
        vec_to_tensor(dkw.k2_beta, {128}));
    m.kan_l2->knots.copy_(
        vec_to_tensor(dkw.k2_knots, {(long)dkw.k2_knots.size()}));

    // KAN Layer 3
    m.kan_l3->spline_weights.data().copy_(
        vec_to_tensor(dkw.k3_weights, {128, 128 * 8}));
    m.kan_l3->residual_W.data().copy_(
        vec_to_tensor(dkw.k3_residual, {128, 128}));
    m.kan_l3->ln_gamma.data().copy_(
        vec_to_tensor(dkw.k3_gamma, {128}));
    m.kan_l3->ln_beta.data().copy_(
        vec_to_tensor(dkw.k3_beta, {128}));
    m.kan_l3->knots.copy_(
        vec_to_tensor(dkw.k3_knots, {(long)dkw.k3_knots.size()}));

    // conv_linear (shared ConvergencePort linear, not conv_emb)
    if (!cpd.conv_linear_W.empty())
        m.conv_linear->weight.data().copy_(
            vec_to_tensor(cpd.conv_linear_W,
                {(long)CONV_OUTPUT_DIM, (long)(CONV_INPUT_DIM + CONV_EMB_DIM)}));
    if (!cpd.conv_linear_b.empty())
        m.conv_linear->bias.data().copy_(
            vec_to_tensor(cpd.conv_linear_b, {(long)CONV_OUTPUT_DIM}));

    // k1_proj
    if (!cw.k1_proj_W.empty())
        m.k1_proj->weight.data().copy_(
            vec_to_tensor(cw.k1_proj_W, {(long)CONV_OUTPUT_DIM, 256}));
    if (!cw.k1_proj_b.empty())
        m.k1_proj->bias.data().copy_(
            vec_to_tensor(cw.k1_proj_b, {(long)CONV_OUTPUT_DIM}));

    // concept_proj (128 → CONCEPT_PROJ_DIM)
    if (!cw.concept_proj_W.empty())
        m.concept_proj->weight.data().copy_(
            vec_to_tensor(cw.concept_proj_W, {(long)CONCEPT_PROJ_DIM, 128}));
}

// =============================================================================
// Helper: extract trained concept model weights
// =============================================================================

static void extract_concept_model_weights(const DeepKANv2Decoder& model,
                                           cuda::DeepKANWeights& dkw,
                                           ConvergencePortData& cpd,
                                           ConceptWeights& cw) {
    auto& m = *model;

    // KAN layers
    tensor_to_vec(m.kan_l1->spline_weights, dkw.k1_weights);
    tensor_to_vec(m.kan_l1->residual_W, dkw.k1_residual);
    tensor_to_vec(m.kan_l1->ln_gamma, dkw.k1_gamma);
    tensor_to_vec(m.kan_l1->ln_beta, dkw.k1_beta);

    tensor_to_vec(m.kan_l2->spline_weights, dkw.k2_weights);
    tensor_to_vec(m.kan_l2->residual_W, dkw.k2_residual);
    tensor_to_vec(m.kan_l2->ln_gamma, dkw.k2_gamma);
    tensor_to_vec(m.kan_l2->ln_beta, dkw.k2_beta);

    tensor_to_vec(m.kan_l3->spline_weights, dkw.k3_weights);
    tensor_to_vec(m.kan_l3->residual_W, dkw.k3_residual);
    tensor_to_vec(m.kan_l3->ln_gamma, dkw.k3_gamma);
    tensor_to_vec(m.kan_l3->ln_beta, dkw.k3_beta);

    // conv_linear
    tensor_to_vec(m.conv_linear->weight, cpd.conv_linear_W);
    tensor_to_vec(m.conv_linear->bias, cpd.conv_linear_b);

    // k1_proj + concept_proj
    tensor_to_vec(m.k1_proj->weight, cw.k1_proj_W);
    tensor_to_vec(m.k1_proj->bias, cw.k1_proj_b);
    tensor_to_vec(m.concept_proj->weight, cw.concept_proj_W);
}

// =============================================================================
// Helper: evaluate concept prediction loss (unweighted CE, for early stopping)
// =============================================================================

static double eval_concept_loss(DeepKANv2Decoder& model,
                                const torch::Tensor& h_tensor,
                                const torch::Tensor& emb_tensor,
                                const torch::Tensor& target_tensor,
                                const std::vector<int64_t>& indices,
                                size_t batch_size,
                                torch::Device device) {
    torch::NoGradGuard no_grad;
    model->eval();

    size_t N = indices.size();
    if (N == 0) return 0.0;

    auto idx_tensor = torch::from_blob(
        const_cast<int64_t*>(indices.data()), {(long)N},
        torch::kInt64).to(device);

    double total_loss = 0.0;
    size_t total = 0;
    size_t num_batches = (N + batch_size - 1) / batch_size;

    for (size_t b = 0; b < num_batches; ++b) {
        size_t start = b * batch_size;
        size_t end = std::min(start + batch_size, N);
        size_t bs = end - start;

        auto batch_idx = idx_tensor.narrow(0, (long)start, (long)bs);
        auto h_batch = h_tensor.index_select(0, batch_idx);
        auto emb_batch = emb_tensor.index_select(0, batch_idx);
        auto target_batch = target_tensor.index_select(0, batch_idx);

        auto logits = model->forward_concepts(h_batch, emb_batch);
        auto loss = torch::nn::functional::cross_entropy(logits, target_batch);

        total_loss += loss.item<double>() * bs;
        total += bs;
    }

    model->train();
    return total_loss / std::max(total, (size_t)1);
}

// =============================================================================
// train_concept_deep_kan_v2 — Concept prediction training
// =============================================================================

bool train_concept_deep_kan_v2(const ConceptTrainingData& data,
                                cuda::DeepKANWeights& dkw,
                                ConvergencePortData& cpd,
                                ConceptWeights& cw,
                                const DeepKANv2Config& config,
                                float concept_temperature,
                                cuda::TrainingResult& result) {
    auto t_start = std::chrono::steady_clock::now();

    // ── Step 1: Precompute hidden states ──
    std::cerr << "[LibTorch/Concept] Precomputing concept hidden states...\n";
    auto pd = precompute_concept_hidden_states(data);
    size_t N = pd.n_pairs;

    if (N == 0) {
        std::cerr << "[LibTorch/Concept] No concept pairs to train on!\n";
        return false;
    }

    auto t_precomp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[LibTorch/Concept]   Precomputed " << N << " concept pairs from "
              << data.num_samples << " samples (" << t_precomp << "ms)\n";

    // ── Step 1b: Train/Val split (80/20 at sample level) ──
    size_t num_samples = data.num_samples;
    std::vector<size_t> sample_order(num_samples);
    std::iota(sample_order.begin(), sample_order.end(), 0);
    { std::mt19937 rng(7777); std::shuffle(sample_order.begin(), sample_order.end(), rng); }

    size_t n_train_samples = (num_samples * 80) / 100;

    std::vector<int64_t> train_indices, val_indices;
    train_indices.reserve(N);
    val_indices.reserve(N / 4);

    for (size_t i = 0; i < n_train_samples; ++i) {
        size_t s = sample_order[i];
        size_t off = pd.sample_offsets[s];
        size_t cnt = pd.sample_counts[s];
        for (size_t t = 0; t < cnt; ++t)
            train_indices.push_back((int64_t)(off + t));
    }
    for (size_t i = n_train_samples; i < num_samples; ++i) {
        size_t s = sample_order[i];
        size_t off = pd.sample_offsets[s];
        size_t cnt = pd.sample_counts[s];
        for (size_t t = 0; t < cnt; ++t)
            val_indices.push_back((int64_t)(off + t));
    }

    size_t N_train = train_indices.size();
    size_t N_val = val_indices.size();
    std::cerr << "[LibTorch/Concept]   Split: " << n_train_samples << " train ("
              << N_train << " pairs), " << (num_samples - n_train_samples) << " val ("
              << N_val << " pairs)\n";

    // ── Step 2: Build model ──
    size_t NC = data.num_concepts;
    // VA=1, V=1: token output head and conv_emb unused in concept mode
    DeepKANv2Decoder model(1, 1, config.dropout_p);

    // Load existing weights if available (warm-start)
    if (!dkw.k1_weights.empty()) {
        load_concept_model_weights(model, dkw, cpd, cw);
    }

    // Build and set concept matrix (32D: core 16D + detail 16D)
    auto cm_tensor = vec_to_tensor(data.concept_matrix,
        {(long)NC, (long)CONCEPT_PROJ_DIM});
    model->set_concept_matrix(cm_tensor, concept_temperature);

    // Freeze unused parameters (conv_emb, output head)
    model->conv_emb->weight.set_requires_grad(false);
    model->output->weight.set_requires_grad(false);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, 0);
        std::cerr << "[LibTorch/Concept]   Using CUDA\n";
    } else {
        std::cerr << "[LibTorch/Concept]   CUDA not available, using CPU\n";
    }
    model->to(device);

    // ── Build tensors ──
    auto h_tensor = torch::from_blob(pd.all_h.data(), {(long)N, 90},
        torch::kFloat32).clone().to(device);
    auto emb_tensor = torch::from_blob(pd.all_concept_emb.data(),
        {(long)N, (long)CONV_EMB_DIM},
        torch::kFloat32).clone().to(device);
    auto target_tensor = torch::from_blob(pd.all_targets.data(), {(long)N},
        torch::kInt64).clone().to(device);
    auto trust_tensor = torch::from_blob(pd.all_trust.data(), {(long)N},
        torch::kFloat32).clone().to(device);

    // ── Step 3: Setup optimizer with parameter groups ──
    std::vector<torch::Tensor> concept_head_params;  // concept_proj (train from epoch 0)
    std::vector<torch::Tensor> kan_params;            // KAN L1/L2/L3 + k1_proj
    std::vector<torch::Tensor> conv_params;           // conv_linear

    for (auto& item : model->named_parameters()) {
        auto& param = item.value();
        if (!param.requires_grad()) continue;
        const auto& name = item.key();

        if (name.find("concept_proj") != std::string::npos) {
            concept_head_params.push_back(param);
        } else if (name.find("conv_linear") != std::string::npos) {
            conv_params.push_back(param);
        } else {
            kan_params.push_back(param);
        }
    }

    auto make_group = [&config](std::vector<torch::Tensor>& params, double lr) {
        auto opts = std::make_unique<torch::optim::AdamOptions>(lr);
        opts->betas(std::make_tuple(0.9, 0.999));
        opts->weight_decay(config.weight_decay);
        return torch::optim::OptimizerParamGroup(params, std::move(opts));
    };

    torch::optim::Adam optimizer({
        make_group(concept_head_params, config.lr_output),
        make_group(kan_params, config.lr_kan),
        make_group(conv_params, config.lr_conv)
    });

    // ── Step 4: Training loop ──
    size_t batch_size = config.batch_size;
    size_t num_train_batches = (N_train + batch_size - 1) / batch_size;

    double best_val_loss = 1e9;
    double best_train_loss = 1e9;
    size_t epochs_without_improvement = 0;
    size_t best_epoch = 0;
    std::mt19937 rng(42);

    std::vector<char> best_model_state;

    std::cerr << "[LibTorch/Concept]   Training: " << config.num_epochs << " epochs, "
              << N_train << " train, " << N_val << " val, "
              << "batch=" << batch_size << ", N_concepts=" << NC
              << ", T=" << concept_temperature
              << ", patience=" << config.patience << "\n";

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        // Cosine LR decay
        double progress = (double)epoch / std::max(config.num_epochs - 1, (size_t)1);
        double cos_mult = 0.5 * (1.0 + std::cos(progress * M_PI));
        double lr_mult = 0.1 + 0.9 * cos_mult;

        {
            auto& groups = optimizer.param_groups();
            static_cast<torch::optim::AdamOptions&>(groups[0].options()).lr(
                config.lr_output * lr_mult);

            double kan_lr = (epoch < config.warmup_epochs) ? 0.0 : config.lr_kan * lr_mult;
            double conv_lr = (epoch < config.warmup_epochs) ? 0.0 : config.lr_conv * lr_mult;
            static_cast<torch::optim::AdamOptions&>(groups[1].options()).lr(kan_lr);
            static_cast<torch::optim::AdamOptions&>(groups[2].options()).lr(conv_lr);
        }

        // Shuffle train indices
        std::shuffle(train_indices.begin(), train_indices.end(), rng);
        auto train_idx_tensor = torch::from_blob(
            train_indices.data(), {(long)N_train}, torch::kInt64).to(device);

        double epoch_loss = 0.0;
        size_t epoch_pairs = 0;
        model->train();

        for (size_t b = 0; b < num_train_batches; ++b) {
            size_t start = b * batch_size;
            size_t end = std::min(start + batch_size, N_train);
            size_t bs = end - start;

            auto batch_idx = train_idx_tensor.narrow(0, (long)start, (long)bs);
            auto h_batch = h_tensor.index_select(0, batch_idx);
            auto emb_batch = emb_tensor.index_select(0, batch_idx);
            auto target_batch = target_tensor.index_select(0, batch_idx);
            auto trust_batch = trust_tensor.index_select(0, batch_idx);

            auto logits = model->forward_concepts(h_batch, emb_batch);

            // Trust-weighted cross-entropy
            auto log_probs = torch::log_softmax(logits, 1);
            auto nll = torch::nll_loss(log_probs, target_batch,
                /*weight=*/{}, torch::Reduction::None);
            auto loss = (nll * trust_batch).mean();

            optimizer.zero_grad();
            loss.backward();

            if (epoch >= config.warmup_epochs && !config.lr_scale.empty()) {
                scale_kan_l1_gradients(model, config.lr_scale);
            }

            optimizer.step();

            epoch_loss += loss.item<double>() * bs;
            epoch_pairs += bs;
        }

        // Evaluate every 5 epochs
        bool do_eval = (epoch % 5 == 0) || (epoch == config.num_epochs - 1);
        auto t_now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();

        if (do_eval) {
            double train_loss = eval_concept_loss(model, h_tensor, emb_tensor,
                target_tensor, train_indices, batch_size, device);
            double val_loss = eval_concept_loss(model, h_tensor, emb_tensor,
                target_tensor, val_indices, batch_size, device);

            if (train_loss < best_train_loss) best_train_loss = train_loss;

            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_without_improvement = 0;

                std::ostringstream oss;
                torch::serialize::OutputArchive archive;
                model->save(archive);
                archive.save_to(oss);
                auto s = oss.str();
                best_model_state.assign(s.begin(), s.end());
            } else {
                epochs_without_improvement += 5;
            }

            double gap = val_loss - train_loss;
            std::cerr << "[LibTorch/Concept]   epoch " << epoch << "/" << config.num_epochs
                      << " train=" << train_loss
                      << " val=" << val_loss
                      << " gap=" << gap
                      << " best_val=" << best_val_loss
                      << " (" << t_now << "ms)\n";

            // Early stopping: val-gap overfitting detection
            if (config.max_val_gap > 0 && epoch > 20 && gap > config.max_val_gap) {
                std::cerr << "[LibTorch/Concept]   Val-gap early stop: gap=" << gap
                          << " > max_val_gap=" << config.max_val_gap
                          << " at epoch " << epoch << " (best @ " << best_epoch << ")\n";
                break;
            }

            if (config.patience > 0 && epochs_without_improvement >= config.patience) {
                std::cerr << "[LibTorch/Concept]   Early stopping at epoch " << epoch
                          << " (best @ " << best_epoch << ")\n";
                break;
            }
        } else {
            double avg = epoch_loss / std::max(epoch_pairs, (size_t)1);
            std::cerr << "[LibTorch/Concept]   epoch " << epoch << "/" << config.num_epochs
                      << " running=" << avg << " (" << t_now << "ms)\n";
        }
    }

    // ── Step 5: Restore best model and extract weights ──
    if (!best_model_state.empty()) {
        std::istringstream iss(std::string(best_model_state.begin(), best_model_state.end()));
        torch::serialize::InputArchive archive;
        archive.load_from(iss);
        model->load(archive);
        std::cerr << "[LibTorch/Concept]   Restored best model from epoch " << best_epoch << "\n";
    }

    model->to(torch::kCPU);
    extract_concept_model_weights(model, dkw, cpd, cw);

    result.best_loss = best_val_loss;

    auto t_total = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[LibTorch/Concept]   Complete: best_train=" << best_train_loss
              << " best_val=" << best_val_loss
              << " (" << t_total << "ms)\n";

    return true;
}

// =============================================================================
// Autoregressive concept generation with trained DeepKAN v2
// =============================================================================

ConceptGenerateResult generate_concept_deep_kan_v2(
    const cuda::DeepKANWeights& dkw,
    const ConvergencePortData& cpd,
    const ConceptWeights& cw,
    const std::vector<double>& concept_matrix,
    const std::vector<double>& concept_emb_64d,
    const std::vector<double>& concept_flex_16d,
    size_t num_concepts,
    const std::vector<float>& initial_h,
    int64_t start_concept_idx,
    size_t max_concepts,
    float temperature,
    bool use_lstm_gates)
{
    ConceptGenerateResult gr;
    const size_t H_90 = 90;
    const size_t FUSED_BASE = 64;
    const size_t fd = 16;
    const size_t NC = num_concepts;

    // Build model (dropout=0 at inference)
    DeepKANv2Decoder model(1, 1, 0.0);

    if (!dkw.k1_weights.empty()) {
        load_concept_model_weights(model, dkw, cpd, cw);
    }

    // Set concept matrix
    auto cm_tensor = vec_to_tensor(concept_matrix, {(long)NC, (long)CONCEPT_PROJ_DIM});
    model->set_concept_matrix(cm_tensor, temperature);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA, 0);
    model->to(device);
    model->eval();

    torch::NoGradGuard no_grad;

    // Working hidden state
    std::vector<float> h(initial_h.begin(), initial_h.end());
    h.resize(H_90, 0.0f);
    int64_t current_idx = start_concept_idx;

    for (size_t t = 0; t < max_concepts; ++t) {
        if (current_idx < 0 || (size_t)current_idx >= NC) break;

        auto h_tensor = torch::from_blob(h.data(), {1, (long)H_90},
            torch::kFloat32).clone().to(device);

        // Look up current concept's 16D core embedding for ConvergencePort
        // concept_matrix is 32D (core+detail), but CM input uses first 16D
        std::vector<float> ce(CONV_EMB_DIM);
        for (size_t d = 0; d < CONV_EMB_DIM; ++d)
            ce[d] = (float)concept_matrix[(size_t)current_idx * CONCEPT_PROJ_DIM + d];
        auto emb_tensor_local = torch::from_blob(ce.data(), {1, (long)CONV_EMB_DIM},
            torch::kFloat32).clone().to(device);

        auto logits = model->forward_concepts(h_tensor, emb_tensor_local);
        auto probs = torch::softmax(logits, 1);
        auto best_idx = probs.argmax(1).item<int64_t>();
        double best_prob = probs[0][best_idx].item<double>();

        gr.concept_indices.push_back(best_idx);
        gr.confidences.push_back(best_prob);

        // Stop on low confidence after warmup
        if (best_prob < 0.01 && t > 2) break;

        // Evolve hidden state using predicted concept's embeddings
        if (use_lstm_gates) {
            double dot1 = 0;
            for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                double v = concept_emb_64d[(size_t)best_idx * FUSED_BASE + i];
                dot1 += h[i] * v;
            }
            double gate1 = gate_sigmoid(dot1 / 8.0);
            double alpha1 = 0.1 + gate1 * 0.2;
            for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                double v = concept_emb_64d[(size_t)best_idx * FUSED_BASE + i];
                h[i] = (float)(h[i] * (1.0 - alpha1) + v * alpha1);
            }
            double dot2 = 0;
            for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                double v = concept_flex_16d[(size_t)best_idx * fd + i];
                dot2 += h[FUSED_BASE + i] * v;
            }
            double gate2 = gate_sigmoid(dot2 / 4.0);
            double alpha2 = 0.05 + gate2 * 0.1;
            for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                double v = concept_flex_16d[(size_t)best_idx * fd + i];
                h[FUSED_BASE + i] = (float)(h[FUSED_BASE + i] * (1.0 - alpha2) + v * alpha2);
            }
            double h3_norm = 0;
            for (size_t i = FUSED_BASE + fd; i < H_90; ++i)
                h3_norm += h[i] * h[i];
            double decay = 0.93 + gate_sigmoid(-h3_norm + 1.0) * 0.04;
            for (size_t i = FUSED_BASE + fd; i < H_90; ++i)
                h[i] *= (float)decay;
        } else {
            for (size_t i = 0; i < FUSED_BASE && i < H_90; ++i) {
                double v = concept_emb_64d[(size_t)best_idx * FUSED_BASE + i];
                h[i] = (float)(h[i] * 0.8 + v * 0.2);
            }
            for (size_t i = 0; i < fd && (FUSED_BASE + i) < H_90; ++i) {
                double v = concept_flex_16d[(size_t)best_idx * fd + i];
                h[FUSED_BASE + i] = (float)(h[FUSED_BASE + i] * 0.9 + v * 0.1);
            }
            for (size_t i = FUSED_BASE + fd; i < H_90; ++i) {
                h[i] *= 0.95f;
            }
        }

        current_idx = best_idx;
    }

    return gr;
}

// =============================================================================
// train_unified_deep_kan_v2 — Token + Concept in shared forward pass
// =============================================================================
// Both heads trained simultaneously: combined loss prevents catastrophic
// interference where token training destroys concept_proj alignment.
// In each training step:
//   1. Token batch → forward() → token_CE
//   2. Concept batch → forward_concepts() → trust-weighted concept_CE
//   3. total_loss = token_CE + weight * concept_CE
//   4. Single backward() → shared KAN backbone gets gradients from both tasks

bool train_unified_deep_kan_v2(
    const cuda::TrainingData& token_data,
    const ConceptTrainingData& concept_data,
    cuda::DeepKANWeights& dkw,
    ConvergencePortData& cpd,
    ConceptWeights& cw,
    const UnifiedTrainingConfig& config,
    UnifiedTrainingResult& result)
{
    auto t_start = std::chrono::steady_clock::now();

    // ── Step 1: Precompute hidden states for both tasks ──
    std::cerr << "[Unified] Precomputing token hidden states...\n";
    auto tok_pd = precompute_hidden_states(token_data);
    std::cerr << "[Unified] Precomputing concept hidden states...\n";
    auto con_pd = precompute_concept_hidden_states(concept_data);

    size_t N_tok = tok_pd.n_tokens;
    size_t N_con = con_pd.n_pairs;

    if (N_tok == 0 || N_con == 0) {
        std::cerr << "[Unified] Need both token and concept data! tok="
                  << N_tok << " con=" << N_con << "\n";
        return false;
    }

    auto t_precomp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[Unified]   " << N_tok << " tokens, " << N_con
              << " concept pairs (" << t_precomp << "ms)\n";

    // ── Step 2: Train/Val splits for both datasets (80/20 at sample level) ──
    // Token split
    size_t tok_num_samples = token_data.num_samples;
    std::vector<size_t> tok_sample_order(tok_num_samples);
    std::iota(tok_sample_order.begin(), tok_sample_order.end(), 0);
    { std::mt19937 rng(7777); std::shuffle(tok_sample_order.begin(), tok_sample_order.end(), rng); }
    size_t tok_n_train = (tok_num_samples * 80) / 100;

    std::vector<int64_t> tok_train_idx, tok_val_idx;
    tok_train_idx.reserve(N_tok);
    tok_val_idx.reserve(N_tok / 4);
    for (size_t i = 0; i < tok_n_train; ++i) {
        size_t s = tok_sample_order[i];
        for (size_t t = 0; t < tok_pd.sample_tok_counts[s]; ++t)
            tok_train_idx.push_back((int64_t)(tok_pd.sample_tok_offsets[s] + t));
    }
    for (size_t i = tok_n_train; i < tok_num_samples; ++i) {
        size_t s = tok_sample_order[i];
        for (size_t t = 0; t < tok_pd.sample_tok_counts[s]; ++t)
            tok_val_idx.push_back((int64_t)(tok_pd.sample_tok_offsets[s] + t));
    }

    // Concept split
    size_t con_num_samples = concept_data.num_samples;
    std::vector<size_t> con_sample_order(con_num_samples);
    std::iota(con_sample_order.begin(), con_sample_order.end(), 0);
    { std::mt19937 rng(8888); std::shuffle(con_sample_order.begin(), con_sample_order.end(), rng); }
    size_t con_n_train = (con_num_samples * 80) / 100;

    std::vector<int64_t> con_train_idx, con_val_idx;
    con_train_idx.reserve(N_con);
    con_val_idx.reserve(N_con / 4);
    for (size_t i = 0; i < con_n_train; ++i) {
        size_t s = con_sample_order[i];
        for (size_t t = 0; t < con_pd.sample_counts[s]; ++t)
            con_train_idx.push_back((int64_t)(con_pd.sample_offsets[s] + t));
    }
    for (size_t i = con_n_train; i < con_num_samples; ++i) {
        size_t s = con_sample_order[i];
        for (size_t t = 0; t < con_pd.sample_counts[s]; ++t)
            con_val_idx.push_back((int64_t)(con_pd.sample_offsets[s] + t));
    }

    size_t N_tok_train = tok_train_idx.size();
    size_t N_tok_val = tok_val_idx.size();
    size_t N_con_train = con_train_idx.size();
    size_t N_con_val = con_val_idx.size();

    std::cerr << "[Unified]   Token split: " << N_tok_train << " train, "
              << N_tok_val << " val\n";
    std::cerr << "[Unified]   Concept split: " << N_con_train << " train, "
              << N_con_val << " val\n";

    // ── Step 3: Build model with BOTH heads active ──
    size_t VA = token_data.VA;
    size_t V = token_data.V;
    size_t NC = concept_data.num_concepts;

    DeepKANv2Decoder model(VA, V, config.dropout_p);

    // Load shared KAN backbone + token output + conv_emb/conv_linear
    load_kan_weights(model, dkw, cpd, V);

    // Load concept-specific weights (k1_proj, concept_proj) if available
    if (!cw.k1_proj_W.empty())
        model->k1_proj->weight.data().copy_(
            vec_to_tensor(cw.k1_proj_W, {(long)CONV_OUTPUT_DIM, 256}));
    if (!cw.k1_proj_b.empty())
        model->k1_proj->bias.data().copy_(
            vec_to_tensor(cw.k1_proj_b, {(long)CONV_OUTPUT_DIM}));
    if (!cw.concept_proj_W.empty())
        model->concept_proj->weight.data().copy_(
            vec_to_tensor(cw.concept_proj_W, {(long)CONCEPT_PROJ_DIM, 128}));

    // Set concept embedding matrix for cosine similarity output
    auto cm_tensor = vec_to_tensor(concept_data.concept_matrix,
        {(long)NC, (long)CONCEPT_PROJ_DIM});
    model->set_concept_matrix(cm_tensor, config.concept_temperature);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, 0);
        std::cerr << "[Unified]   Using CUDA\n";
    } else {
        std::cerr << "[Unified]   CUDA not available, using CPU\n";
    }
    model->to(device);

    // ── Build tensors ──
    // Token tensors
    auto tok_h = torch::from_blob(tok_pd.all_h.data(), {(long)N_tok, 90},
        torch::kFloat32).clone().to(device);
    auto tok_ids_t = torch::from_blob(tok_pd.all_toks.data(), {(long)N_tok},
        torch::kInt64).clone().to(device);
    auto tok_targets = torch::from_blob(tok_pd.all_targets.data(), {(long)N_tok},
        torch::kInt64).clone().to(device);

    // Concept tensors
    auto con_h = torch::from_blob(con_pd.all_h.data(), {(long)N_con, 90},
        torch::kFloat32).clone().to(device);
    auto con_emb = torch::from_blob(con_pd.all_concept_emb.data(),
        {(long)N_con, (long)CONV_EMB_DIM},
        torch::kFloat32).clone().to(device);
    auto con_targets = torch::from_blob(con_pd.all_targets.data(), {(long)N_con},
        torch::kInt64).clone().to(device);
    auto con_trust = torch::from_blob(con_pd.all_trust.data(), {(long)N_con},
        torch::kFloat32).clone().to(device);

    // ── Step 4: Optimizer with 4 parameter groups ──
    std::vector<torch::Tensor> token_head_params;   // output.weight
    std::vector<torch::Tensor> concept_head_params; // concept_proj.weight
    std::vector<torch::Tensor> kan_params;           // KAN L1/L2/L3 + k1_proj
    std::vector<torch::Tensor> conv_params;          // conv_emb + conv_linear

    for (auto& item : model->named_parameters()) {
        const auto& name = item.key();
        auto& param = item.value();

        if (name.find("output.") != std::string::npos) {
            token_head_params.push_back(param);
        } else if (name.find("concept_proj") != std::string::npos) {
            concept_head_params.push_back(param);
        } else if (name.find("conv_emb") != std::string::npos ||
                   name.find("conv_linear") != std::string::npos) {
            conv_params.push_back(param);
        } else {
            kan_params.push_back(param);
        }
    }

    auto make_group = [&config](std::vector<torch::Tensor>& params, double lr) {
        auto opts = std::make_unique<torch::optim::AdamOptions>(lr);
        opts->betas(std::make_tuple(0.9, 0.999));
        opts->weight_decay(config.weight_decay);
        return torch::optim::OptimizerParamGroup(params, std::move(opts));
    };

    torch::optim::Adam optimizer({
        make_group(token_head_params, config.lr_token_head),
        make_group(concept_head_params, config.lr_concept_head),
        make_group(kan_params, config.lr_kan),
        make_group(conv_params, config.lr_conv)
    });

    // ── Step 5: Training loop ──
    size_t batch_size = config.batch_size;
    size_t tok_batches = (N_tok_train + batch_size - 1) / batch_size;
    size_t con_batches = (N_con_train + batch_size - 1) / batch_size;
    size_t num_steps = std::max(tok_batches, con_batches);

    double best_combined = 1e9;
    double best_tok_val = 1e9;
    double best_con_val = 1e9;
    size_t best_epoch = 0;
    size_t epochs_no_improve = 0;
    std::mt19937 rng(42);
    std::vector<char> best_model_state;

    std::cerr << "[Unified]   Training: " << config.num_epochs << " epochs, "
              << N_tok_train << " tok_train, " << N_con_train << " con_train, "
              << "batch=" << batch_size
              << ", concept_weight=" << config.concept_loss_weight
              << ", T=" << config.concept_temperature
              << ", patience=" << config.patience << "\n";

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        // Cosine LR decay
        double progress = (double)epoch / std::max(config.num_epochs - 1, (size_t)1);
        double cos_mult = 0.5 * (1.0 + std::cos(progress * M_PI));
        double lr_mult = 0.1 + 0.9 * cos_mult;

        {
            auto& groups = optimizer.param_groups();
            static_cast<torch::optim::AdamOptions&>(groups[0].options()).lr(
                config.lr_token_head * lr_mult);
            static_cast<torch::optim::AdamOptions&>(groups[1].options()).lr(
                config.lr_concept_head * lr_mult);

            double kan_lr = (epoch < config.warmup_epochs) ? 0.0 : config.lr_kan * lr_mult;
            double conv_lr = (epoch < config.warmup_epochs) ? 0.0 : config.lr_conv * lr_mult;
            static_cast<torch::optim::AdamOptions&>(groups[2].options()).lr(kan_lr);
            static_cast<torch::optim::AdamOptions&>(groups[3].options()).lr(conv_lr);
        }

        // Shuffle train indices each epoch
        std::shuffle(tok_train_idx.begin(), tok_train_idx.end(), rng);
        std::shuffle(con_train_idx.begin(), con_train_idx.end(), rng);

        auto tok_idx_t = torch::from_blob(tok_train_idx.data(), {(long)N_tok_train},
            torch::kInt64).to(device);
        auto con_idx_t = torch::from_blob(con_train_idx.data(), {(long)N_con_train},
            torch::kInt64).to(device);

        model->train();

        for (size_t step = 0; step < num_steps; ++step) {
            optimizer.zero_grad();

            // ── Token batch → forward → token_loss ──
            torch::Tensor total_loss;
            if (step < tok_batches) {
                size_t start = step * batch_size;
                size_t end = std::min(start + batch_size, N_tok_train);
                size_t bs = end - start;

                auto bidx = tok_idx_t.narrow(0, (long)start, (long)bs);
                auto h_b = tok_h.index_select(0, bidx);
                auto t_b = tok_ids_t.index_select(0, bidx);
                auto tgt_b = tok_targets.index_select(0, bidx);

                auto logits = model->forward(h_b, t_b);
                total_loss = torch::nn::functional::cross_entropy(logits, tgt_b);
            } else {
                total_loss = torch::zeros({1}, torch::TensorOptions().device(device));
            }

            // ── Concept batch → forward_concepts → concept_loss ──
            {
                size_t con_step = step % con_batches;  // cycle if fewer concept batches
                size_t start = con_step * batch_size;
                size_t end = std::min(start + batch_size, N_con_train);
                size_t bs = end - start;

                auto bidx = con_idx_t.narrow(0, (long)start, (long)bs);
                auto h_b = con_h.index_select(0, bidx);
                auto e_b = con_emb.index_select(0, bidx);
                auto tgt_b = con_targets.index_select(0, bidx);
                auto trust_b = con_trust.index_select(0, bidx);

                auto logits = model->forward_concepts(h_b, e_b);
                auto log_probs = torch::log_softmax(logits, 1);
                auto nll = torch::nll_loss(log_probs, tgt_b,
                    /*weight=*/{}, torch::Reduction::None);
                auto con_loss = (nll * trust_b).mean();

                total_loss = total_loss + config.concept_loss_weight * con_loss;
            }

            // ── Combined backward through shared KAN backbone ──
            total_loss.backward();

            if (epoch >= config.warmup_epochs && !config.lr_scale.empty()) {
                scale_kan_l1_gradients(model, config.lr_scale);
            }

            optimizer.step();
        }

        // ── Evaluation every 5 epochs ──
        bool do_eval = (epoch % 5 == 0) || (epoch == config.num_epochs - 1);
        auto t_now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();

        if (do_eval) {
            double tok_train = eval_loss(model, tok_h, tok_ids_t, tok_targets,
                tok_train_idx, batch_size, device);
            double tok_val = eval_loss(model, tok_h, tok_ids_t, tok_targets,
                tok_val_idx, batch_size, device);
            double con_train = eval_concept_loss(model, con_h, con_emb,
                con_targets, con_train_idx, batch_size, device);
            double con_val = eval_concept_loss(model, con_h, con_emb,
                con_targets, con_val_idx, batch_size, device);

            double combined = tok_val + config.concept_loss_weight * con_val;

            if (combined < best_combined) {
                best_combined = combined;
                best_tok_val = tok_val;
                best_con_val = con_val;
                best_epoch = epoch;
                epochs_no_improve = 0;

                // Save best model state
                std::ostringstream oss;
                torch::serialize::OutputArchive archive;
                model->save(archive);
                archive.save_to(oss);
                auto s = oss.str();
                best_model_state.assign(s.begin(), s.end());
            } else {
                epochs_no_improve += 5;
            }

            double tok_gap = tok_val - tok_train;
            double con_gap = con_val - con_train;

            std::cerr << "[Unified]   epoch " << epoch << "/" << config.num_epochs
                      << " tok=" << tok_train << "/" << tok_val
                      << " con=" << con_train << "/" << con_val
                      << " combined=" << combined
                      << " best=" << best_combined
                      << " (" << t_now << "ms)\n";

            // Early stopping: val-gap overfitting
            if (config.max_val_gap > 0 && epoch > 20 &&
                (tok_gap > config.max_val_gap || con_gap > config.max_val_gap)) {
                std::cerr << "[Unified]   Val-gap early stop at epoch " << epoch
                          << " (tok_gap=" << tok_gap << " con_gap=" << con_gap
                          << " best @ " << best_epoch << ")\n";
                break;
            }

            // Early stopping: patience
            if (config.patience > 0 && epochs_no_improve >= config.patience) {
                std::cerr << "[Unified]   Patience early stop at epoch " << epoch
                          << " (best @ " << best_epoch << ")\n";
                break;
            }
        } else {
            std::cerr << "[Unified]   epoch " << epoch << "/" << config.num_epochs
                      << " (" << t_now << "ms)\n";
        }
    }

    // ── Step 6: Restore best model and extract ALL weights ──
    if (!best_model_state.empty()) {
        std::istringstream iss(std::string(best_model_state.begin(), best_model_state.end()));
        torch::serialize::InputArchive archive;
        archive.load_from(iss);
        model->load(archive);
        std::cerr << "[Unified]   Restored best model from epoch " << best_epoch << "\n";
    }

    model->to(torch::kCPU);

    // Extract shared KAN backbone + token output + conv weights
    extract_kan_weights(model, dkw, cpd);

    // Extract concept-specific weights
    tensor_to_vec(model->k1_proj->weight, cw.k1_proj_W);
    tensor_to_vec(model->k1_proj->bias, cw.k1_proj_b);
    tensor_to_vec(model->concept_proj->weight, cw.concept_proj_W);

    result.best_token_val = best_tok_val;
    result.best_concept_val = best_con_val;
    result.best_combined_val = best_combined;
    result.best_epoch = best_epoch;

    auto t_total = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[Unified]   Complete: tok_val=" << best_tok_val
              << " con_val=" << best_con_val
              << " combined=" << best_combined
              << " (best @ epoch " << best_epoch << ", " << t_total << "ms)\n";

    return true;
}

} // namespace libtorch
} // namespace brain19

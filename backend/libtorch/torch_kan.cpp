// libtorch/torch_kan.cpp — EfficientKAN + DeepKANv2 Decoder implementation
#include "torch_kan.hpp"
#include <cmath>

namespace brain19 {
namespace libtorch {

// =============================================================================
// EfficientKANLayer
// =============================================================================

EfficientKANLayerImpl::EfficientKANLayerImpl(
    size_t in_dim, size_t out_dim, size_t grid_size, size_t spline_order)
    : in_dim_(in_dim), out_dim_(out_dim)
    , grid_size_(grid_size), spline_order_(spline_order)
    , basis_size_(grid_size + spline_order)
{
    // Spline weights: [out_dim, in_dim * basis_size]
    double scale = 0.01 / std::sqrt((double)(in_dim * basis_size_));
    spline_weights = register_parameter("spline_weights",
        torch::randn({(long)out_dim, (long)(in_dim * basis_size_)}) * scale);

    // Residual projection: [in_dim, out_dim], Xavier init
    double res_scale = std::sqrt(6.0 / (double)(in_dim + out_dim));
    residual_W = register_parameter("residual_W",
        (torch::rand({(long)in_dim, (long)out_dim}) * 2.0 - 1.0) * res_scale);

    // LayerNorm parameters
    ln_gamma = register_parameter("ln_gamma", torch::ones({(long)out_dim}));
    ln_beta = register_parameter("ln_beta", torch::zeros({(long)out_dim}));

    // Knot vector: uniform on [-1, 1] with spline_order extensions on each side
    double h = 2.0 / grid_size;
    size_t n_knots = grid_size + 2 * spline_order + 1;
    auto knots_vec = torch::empty({(long)n_knots});
    auto knots_acc = knots_vec.accessor<float, 1>();
    for (size_t i = 0; i < n_knots; ++i) {
        knots_acc[(long)i] = (float)(-1.0 - (double)spline_order * h + (double)i * h);
    }
    knots = register_buffer("knots", knots_vec);
}

torch::Tensor EfficientKANLayerImpl::compute_basis(torch::Tensor x) {
    // x: [B, in_dim] -> output: [B, in_dim, basis_size]
    // B-spline basis using Cox-de Boor recurrence in tensor ops
    //
    // We build basis functions iteratively from order 0 to spline_order.
    // Order 0: indicator functions (detached, no grad)
    // Orders 1+: polynomial combinations (differentiable)

    size_t k = spline_order_;
    size_t n0 = basis_size_ + k;  // number of order-0 functions

    // x_expanded: [B, D, 1] for broadcasting against knots
    auto x_exp = x.unsqueeze(-1);  // [B, D, 1]

    // Knot spans: knots[i] to knots[i+1], for i = 0..n0-1
    // Order 0: basis_0[i] = (x >= knots[i]) & (x < knots[i+1])
    auto knots_lo = knots.narrow(0, 0, (long)n0);     // [n0]
    auto knots_hi = knots.narrow(0, 1, (long)n0);     // [n0]

    // [B, D, n0]
    auto lo = knots_lo.unsqueeze(0).unsqueeze(0);  // [1, 1, n0]
    auto hi = knots_hi.unsqueeze(0).unsqueeze(0);  // [1, 1, n0]

    // Order 0: indicator functions (not differentiable, detach)
    auto basis_p = ((x_exp >= lo) & (x_exp < hi)).to(x.dtype()).detach();  // [B, D, n0]

    // Right endpoint: last basis function should be 1 when x >= last knot
    // basis_p[:, :, n0-1] should be 1 where x >= knots[n0]
    auto right_mask = (x >= knots[(long)n0].item<float>()).unsqueeze(-1);  // [B, D, 1]
    basis_p = basis_p.clone();
    basis_p.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), (long)(n0 - 1)},
        basis_p.index({torch::indexing::Slice(), torch::indexing::Slice(), (long)(n0 - 1)}) +
        right_mask.squeeze(-1).to(x.dtype()));

    // Build up orders 1..k using Cox-de Boor recurrence
    for (size_t p = 1; p <= k; ++p) {
        size_t np = n0 - p;  // number of basis functions at this order

        // knot differences for left and right terms
        auto t_lo = knots.narrow(0, 0, (long)np);          // [np]
        auto t_hi = knots.narrow(0, (long)p, (long)np);    // [np] = knots[i+p]
        auto t_lo1 = knots.narrow(0, 1, (long)np);         // [np] = knots[i+1]
        auto t_hi1 = knots.narrow(0, (long)(p + 1), (long)np);  // [np] = knots[i+p+1]

        auto d1 = (t_hi - t_lo).unsqueeze(0).unsqueeze(0);    // [1, 1, np]
        auto d2 = (t_hi1 - t_lo1).unsqueeze(0).unsqueeze(0);  // [1, 1, np]

        // Avoid division by zero
        auto safe_d1 = d1.clamp_min(1e-10f);
        auto safe_d2 = d2.clamp_min(1e-10f);

        auto left_num = x_exp - t_lo.unsqueeze(0).unsqueeze(0);     // [B, D, np]
        auto right_num = t_hi1.unsqueeze(0).unsqueeze(0) - x_exp;   // [B, D, np]

        // Mask where denominators are too small
        auto mask1 = (d1.abs() > 1e-10f).to(x.dtype());
        auto mask2 = (d2.abs() > 1e-10f).to(x.dtype());

        auto left_term = mask1 * (left_num / safe_d1) * basis_p.narrow(-1, 0, (long)np);
        auto right_term = mask2 * (right_num / safe_d2) * basis_p.narrow(-1, 1, (long)np);

        basis_p = left_term + right_term;  // [B, D, np]
    }

    // basis_p should now be [B, D, basis_size_]
    return basis_p;
}

torch::Tensor EfficientKANLayerImpl::forward(torch::Tensor x) {
    // x: [B, in_dim]
    auto basis = compute_basis(x);  // [B, in_dim, basis_size]

    // Flatten basis: [B, in_dim * basis_size]
    auto basis_flat = basis.reshape({x.size(0), -1});

    // KAN output: basis_flat @ spline_weights^T → [B, out_dim]
    auto kan_out = torch::mm(basis_flat, spline_weights.t());

    // Residual: SiLU(x @ residual_W) → [B, out_dim]
    auto z_res = torch::mm(x, residual_W);
    auto silu_res = z_res * torch::sigmoid(z_res);

    // Combine
    auto pre_norm = kan_out + silu_res;

    // LayerNorm (manual, since we have custom gamma/beta)
    auto mean = pre_norm.mean(-1, /*keepdim=*/true);
    auto var = pre_norm.var(-1, /*unbiased=*/false, /*keepdim=*/true);
    auto x_hat = (pre_norm - mean) / (var + 1e-5f).sqrt();
    return x_hat * ln_gamma + ln_beta;
}

// =============================================================================
// DeepKANv2Decoder
// =============================================================================

DeepKANv2DecoderImpl::DeepKANv2DecoderImpl(size_t vocab_active, size_t vocab_total,
                                             double dropout_p)
    : VA_(vocab_active), V_total_(vocab_total), dropout_p_(dropout_p)
{
    kan_l1 = register_module("kan_l1", EfficientKANLayer(90, 256, 8, 3));
    drop1 = register_module("drop1", torch::nn::Dropout(
        torch::nn::DropoutOptions(dropout_p)));

    k1_proj = register_module("k1_proj", torch::nn::Linear(
        torch::nn::LinearOptions(256, CONV_OUTPUT_DIM)));

    // Shared ConvergencePort: Embedding(V,16) + Linear(138,32)
    // ~25K params instead of 4.96M per-token weights
    conv_emb = register_module("conv_emb", torch::nn::Embedding(
        torch::nn::EmbeddingOptions((long)vocab_total, CONV_EMB_DIM)));
    conv_linear = register_module("conv_linear", torch::nn::Linear(
        torch::nn::LinearOptions(CONV_INPUT_DIM + CONV_EMB_DIM, CONV_OUTPUT_DIM)));
    drop_cm = register_module("drop_cm", torch::nn::Dropout(
        torch::nn::DropoutOptions(dropout_p)));

    kan_l2 = register_module("kan_l2", EfficientKANLayer(256 + CONV_OUTPUT_DIM, 128, 5, 3));
    drop2 = register_module("drop2", torch::nn::Dropout(
        torch::nn::DropoutOptions(dropout_p)));

    kan_l3 = register_module("kan_l3", EfficientKANLayer(128, 128, 5, 3));
    drop3 = register_module("drop3", torch::nn::Dropout(
        torch::nn::DropoutOptions(dropout_p)));

    output = register_module("output", torch::nn::Linear(
        torch::nn::LinearOptions(128, (long)vocab_active).bias(false)));

    // Concept prediction head: 128 → 16 (concept embedding space)
    concept_proj = register_module("concept_proj", torch::nn::Linear(
        torch::nn::LinearOptions(128, CONV_EMB_DIM).bias(false)));
}

torch::Tensor DeepKANv2DecoderImpl::forward(torch::Tensor h, torch::Tensor tok_ids) {
    // h: [B, 90], tok_ids: [B] (int64)

    auto k1 = drop1(kan_l1->forward(h));    // [B, 256]
    auto k1_p = k1_proj->forward(k1);       // [B, 32]

    // Shared ConvergencePort: cm = tanh(conv_linear(cat(h, k1_proj, conv_emb(tok))))
    auto e = conv_emb->forward(tok_ids);     // [B, 16]
    auto cm_input = torch::cat({h, k1_p, e}, /*dim=*/1);  // [B, 138]
    auto cm = drop_cm(torch::tanh(conv_linear->forward(cm_input))); // [B, 32]

    // KAN Layer 2: input = cat(k1, cm) = [B, 288]
    auto k2_input = torch::cat({k1, cm}, /*dim=*/1);
    auto k2 = drop2(kan_l2->forward(k2_input));   // [B, 128]

    auto features = drop3(kan_l3->forward(k2));    // [B, 128]

    return output->forward(features);        // [B, VA]
}

void DeepKANv2DecoderImpl::set_concept_matrix(torch::Tensor matrix, float temperature) {
    // matrix: [N_concepts, 16] — pre-normalized
    concept_matrix_ = register_buffer("concept_matrix", matrix);
    concept_temperature_ = temperature;
}

torch::Tensor DeepKANv2DecoderImpl::forward_concepts(
    torch::Tensor h, torch::Tensor concept_emb_16d)
{
    // h: [B, 90], concept_emb_16d: [B, 16] — current concept's core embedding
    // Returns: concept_logits [B, N_concepts]
    //
    // Same KAN backbone as token path — gradients flow through all layers:
    //   concept_proj → KAN L3 → KAN L2 → conv_linear → KAN L1

    auto k1 = drop1(kan_l1->forward(h));           // [B, 256]
    auto k1_p = k1_proj->forward(k1);              // [B, 32]

    // ConvergencePort: use concept embedding directly (same 16D as conv_emb output)
    // This bypasses conv_emb lookup but feeds through conv_linear — gradient flows
    auto cm_input = torch::cat({h, k1_p, concept_emb_16d}, /*dim=*/1);  // [B, 138]
    auto cm = drop_cm(torch::tanh(conv_linear->forward(cm_input)));     // [B, 32]

    auto k2_input = torch::cat({k1, cm}, /*dim=*/1);
    auto k2 = drop2(kan_l2->forward(k2_input));    // [B, 128]
    auto features = drop3(kan_l3->forward(k2));     // [B, 128]

    // Project to concept embedding space
    auto proj = concept_proj->forward(features);    // [B, 16]

    // L2-normalize projection
    auto proj_norm = proj / proj.norm(2, /*dim=*/-1, /*keepdim=*/true).clamp_min(1e-8f);

    // Cosine similarity: [B, 16] @ [N, 16]^T = [B, N]
    // concept_matrix_ is already L2-normalized
    auto logits = torch::mm(proj_norm, concept_matrix_.t()) / concept_temperature_;

    return logits;
}

} // namespace libtorch
} // namespace brain19

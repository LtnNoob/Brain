#include "deep_kan.hpp"
#include <cmath>
#include <algorithm>
#include <random>

namespace brain19 {

// =============================================================================
// B-spline basis computation (Cox-de Boor, iterative)
// =============================================================================

void EfficientKANLayer::compute_basis(double x, double* basis, double* deriv) const {
    const size_t nb = basis_size_;       // grid_size + spline_order
    const size_t k = spline_order_;
    const size_t n0 = nb + k;           // number of order-0 basis functions

    // Stack-allocated working buffers (max order=3, max n0=26)
    constexpr size_t MAX_N = 32;
    double B[4][MAX_N] = {};

    // Order 0: indicator functions
    for (size_t i = 0; i < n0 && i < MAX_N; ++i) {
        B[0][i] = (x >= knots_[i] && x < knots_[i + 1]) ? 1.0 : 0.0;
    }
    // Right endpoint
    if (n0 > 0 && x >= knots_[n0]) B[0][n0 - 1] = 1.0;

    // Build up orders 1..k
    for (size_t p = 1; p <= k; ++p) {
        size_t np = n0 - p;
        for (size_t i = 0; i < np && i < MAX_N; ++i) {
            B[p][i] = 0.0;
            double d1 = knots_[i + p] - knots_[i];
            double d2 = knots_[i + p + 1] - knots_[i + 1];
            if (d1 > 1e-10) B[p][i] += (x - knots_[i]) / d1 * B[p - 1][i];
            if (d2 > 1e-10) B[p][i] += (knots_[i + p + 1] - x) / d2 * B[p - 1][i + 1];
        }
    }

    for (size_t i = 0; i < nb; ++i) basis[i] = B[k][i];

    // Derivatives: B'_{i,k}(x) = k * [B_{i,k-1}/(t_{i+k}-t_i) - B_{i+1,k-1}/(t_{i+k+1}-t_{i+1})]
    if (deriv && k > 0) {
        for (size_t i = 0; i < nb; ++i) {
            double d = 0.0;
            double d1 = knots_[i + k] - knots_[i];
            double d2 = knots_[i + k + 1] - knots_[i + 1];
            if (d1 > 1e-10) d += (double)k * B[k - 1][i] / d1;
            if (d2 > 1e-10) d -= (double)k * B[k - 1][i + 1] / d2;
            deriv[i] = d;
        }
    }
}

// =============================================================================
// EfficientKANLayer
// =============================================================================

EfficientKANLayer::EfficientKANLayer(size_t in_dim, size_t out_dim,
                                      size_t grid_size, size_t spline_order,
                                      bool has_residual, bool has_layernorm)
    : in_dim_(in_dim), out_dim_(out_dim)
    , grid_size_(grid_size), spline_order_(spline_order)
    , basis_size_(grid_size + spline_order)
    , has_residual_(has_residual), has_layernorm_(has_layernorm)
{
    // Uniform knot vector on [-1, 1] with k extensions on each side
    double h = 2.0 / grid_size;
    size_t n_knots = grid_size + 2 * spline_order + 1;
    knots_.resize(n_knots);
    for (size_t i = 0; i < n_knots; ++i) {
        knots_[i] = -1.0 - (double)spline_order * h + (double)i * h;
    }

    initialize();
}

void EfficientKANLayer::initialize() {
    // Spline weights: small random (KAN output ≈ 0 initially)
    size_t w_size = out_dim_ * in_dim_ * basis_size_;
    weights_.resize(w_size);
    double scale = 0.01 / std::sqrt((double)(in_dim_ * basis_size_));
    std::mt19937 rng(42 + in_dim_ * 1000 + out_dim_);
    std::normal_distribution<double> dist(0.0, scale);
    for (auto& w : weights_) w = dist(rng);

    // Residual projection: Xavier init
    if (has_residual_) {
        residual_W_.resize(in_dim_ * out_dim_);
        double res_scale = std::sqrt(6.0 / (double)(in_dim_ + out_dim_));
        std::uniform_real_distribution<double> res_dist(-res_scale, res_scale);
        for (auto& w : residual_W_) w = res_dist(rng);
    }

    // LayerNorm: gamma=1, beta=0
    if (has_layernorm_) {
        ln_gamma_.assign(out_dim_, 1.0);
        ln_beta_.assign(out_dim_, 0.0);
    }
}

size_t EfficientKANLayer::num_params() const {
    size_t n = out_dim_ * in_dim_ * basis_size_;
    if (has_residual_) n += in_dim_ * out_dim_;
    if (has_layernorm_) n += 2 * out_dim_;
    return n;
}

// =============================================================================
// Forward Pass
// =============================================================================

std::vector<double> EfficientKANLayer::forward(
    const std::vector<double>& input,
    EfficientKANLayerCache& cache) const
{
    cache.input = input;
    const size_t flat_dim = in_dim_ * basis_size_;

    // Step 1: Compute B-spline basis for each input dimension
    cache.basis_flat.resize(flat_dim);
    cache.basis_deriv.resize(flat_dim);
    for (size_t i = 0; i < in_dim_; ++i) {
        double x = std::clamp(input[i], knots_.front(), knots_.back() - 1e-10);
        compute_basis(x, &cache.basis_flat[i * basis_size_],
                         &cache.basis_deriv[i * basis_size_]);
    }

    // Step 2: KAN matmul: kan_out = W @ basis_flat
    cache.kan_out.assign(out_dim_, 0.0);
    for (size_t o = 0; o < out_dim_; ++o) {
        double sum = 0.0;
        const double* w_row = &weights_[o * flat_dim];
        for (size_t j = 0; j < flat_dim; ++j) {
            sum += w_row[j] * cache.basis_flat[j];
        }
        cache.kan_out[o] = sum;
    }

    // Step 3: Residual = SiLU(input @ W_res)
    if (has_residual_) {
        cache.z_res.resize(out_dim_);
        for (size_t o = 0; o < out_dim_; ++o) {
            double sum = 0.0;
            for (size_t i = 0; i < in_dim_; ++i) {
                sum += input[i] * residual_W_[i * out_dim_ + o];
            }
            cache.z_res[o] = sum;
        }
        cache.pre_norm.resize(out_dim_);
        for (size_t o = 0; o < out_dim_; ++o) {
            double sig = 1.0 / (1.0 + std::exp(-cache.z_res[o]));
            cache.pre_norm[o] = cache.kan_out[o] + cache.z_res[o] * sig;
        }
    } else {
        cache.pre_norm = cache.kan_out;
    }

    // Step 4: LayerNorm
    if (has_layernorm_) {
        double mean = 0.0;
        for (size_t o = 0; o < out_dim_; ++o) mean += cache.pre_norm[o];
        mean /= (double)out_dim_;

        double var = 0.0;
        for (size_t o = 0; o < out_dim_; ++o) {
            double d = cache.pre_norm[o] - mean;
            var += d * d;
        }
        var /= (double)out_dim_;

        cache.ln_inv_std = 1.0 / std::sqrt(var + 1e-5);
        cache.x_hat.resize(out_dim_);
        cache.output.resize(out_dim_);
        for (size_t o = 0; o < out_dim_; ++o) {
            cache.x_hat[o] = (cache.pre_norm[o] - mean) * cache.ln_inv_std;
            cache.output[o] = ln_gamma_[o] * cache.x_hat[o] + ln_beta_[o];
        }
    } else {
        cache.output = cache.pre_norm;
    }

    return cache.output;
}

// =============================================================================
// Backward Pass
// =============================================================================

std::vector<double> EfficientKANLayer::backward(
    const EfficientKANLayerCache& cache,
    const std::vector<double>& d_output,
    double lr,
    const double* lr_input_scale)
{
    const size_t flat_dim = in_dim_ * basis_size_;
    std::vector<double> d_pre_norm(out_dim_);

    // ── LayerNorm backward ──
    if (has_layernorm_) {
        // Update gamma, beta
        for (size_t o = 0; o < out_dim_; ++o) {
            ln_gamma_[o] -= lr * d_output[o] * cache.x_hat[o];
            ln_beta_[o] -= lr * d_output[o];
        }
        // d_pre_norm
        double c1 = 0.0, c2 = 0.0;
        for (size_t o = 0; o < out_dim_; ++o) {
            double dx = d_output[o] * ln_gamma_[o];
            c1 += dx;
            c2 += dx * cache.x_hat[o];
        }
        c1 /= (double)out_dim_;
        c2 /= (double)out_dim_;
        for (size_t o = 0; o < out_dim_; ++o) {
            double dx = d_output[o] * ln_gamma_[o];
            d_pre_norm[o] = cache.ln_inv_std * (dx - c1 - cache.x_hat[o] * c2);
        }
    } else {
        d_pre_norm = d_output;
    }

    // ── KAN weight update: dW[o,j] = d_pre_norm[o] * basis_flat[j] ──
    for (size_t o = 0; o < out_dim_; ++o) {
        double grad_o = d_pre_norm[o];
        double* w_row = &weights_[o * flat_dim];
        for (size_t j = 0; j < flat_dim; ++j) {
            size_t input_idx = j / basis_size_;
            double effective_lr = lr;
            if (lr_input_scale) effective_lr *= lr_input_scale[input_idx];
            w_row[j] -= effective_lr * grad_o * cache.basis_flat[j];
        }
    }

    // ── d_basis_flat = W^T @ d_pre_norm ──
    std::vector<double> d_basis_flat(flat_dim, 0.0);
    for (size_t o = 0; o < out_dim_; ++o) {
        double grad_o = d_pre_norm[o];
        const double* w_row = &weights_[o * flat_dim];
        for (size_t j = 0; j < flat_dim; ++j) {
            d_basis_flat[j] += w_row[j] * grad_o;
        }
    }

    // ── d_input from KAN: propagate through B-spline derivatives ──
    std::vector<double> d_input(in_dim_, 0.0);
    for (size_t i = 0; i < in_dim_; ++i) {
        for (size_t b = 0; b < basis_size_; ++b) {
            d_input[i] += d_basis_flat[i * basis_size_ + b] *
                           cache.basis_deriv[i * basis_size_ + b];
        }
    }

    // ── Residual backward ──
    if (has_residual_) {
        // SiLU backward: d_silu/dz = sig(z) * (1 + z*(1-sig(z)))
        std::vector<double> d_z_res(out_dim_);
        for (size_t o = 0; o < out_dim_; ++o) {
            double sig = 1.0 / (1.0 + std::exp(-cache.z_res[o]));
            double dsilu = sig * (1.0 + cache.z_res[o] * (1.0 - sig));
            d_z_res[o] = d_pre_norm[o] * dsilu;
        }

        // W_res update
        for (size_t i = 0; i < in_dim_; ++i) {
            double inp_i = cache.input[i];
            double effective_lr = lr;
            if (lr_input_scale) effective_lr *= lr_input_scale[i];
            for (size_t o = 0; o < out_dim_; ++o) {
                residual_W_[i * out_dim_ + o] -= effective_lr * inp_i * d_z_res[o];
            }
        }

        // d_input from residual
        for (size_t i = 0; i < in_dim_; ++i) {
            for (size_t o = 0; o < out_dim_; ++o) {
                d_input[i] += d_z_res[o] * residual_W_[i * out_dim_ + o];
            }
        }
    }

    return d_input;
}

// =============================================================================
// DeepKAN
// =============================================================================

DeepKAN::DeepKAN(const std::vector<size_t>& layer_dims,
                 const std::vector<size_t>& grid_sizes,
                 size_t spline_order)
    : spline_order_(spline_order)
{
    size_t n_layers = layer_dims.size() - 1;
    layers_.reserve(n_layers);
    caches_.resize(n_layers);

    for (size_t l = 0; l < n_layers; ++l) {
        layers_.emplace_back(
            layer_dims[l], layer_dims[l + 1],
            grid_sizes[l], spline_order,
            true,   // residual on all layers
            true    // layernorm on all layers
        );
    }
}

std::vector<double> DeepKAN::forward(const std::vector<double>& input) {
    std::vector<double> x = input;
    for (size_t l = 0; l < layers_.size(); ++l) {
        x = layers_[l].forward(x, caches_[l]);
    }
    return x;
}

void DeepKAN::backward(const std::vector<double>& d_output, double lr,
                        const double* lr_input_scale) {
    std::vector<double> d = d_output;
    for (int l = (int)layers_.size() - 1; l >= 0; --l) {
        const double* scale = (l == 0) ? lr_input_scale : nullptr;
        d = layers_[l].backward(caches_[l], d, lr, scale);
    }
}

std::vector<double> DeepKAN::inference(const std::vector<double>& input) const {
    std::vector<double> x = input;
    for (size_t l = 0; l < layers_.size(); ++l) {
        EfficientKANLayerCache cache;
        x = layers_[l].forward(x, cache);
    }
    return x;
}

size_t DeepKAN::input_dim() const {
    return layers_.empty() ? 0 : layers_.front().in_dim();
}

size_t DeepKAN::output_dim() const {
    return layers_.empty() ? 0 : layers_.back().out_dim();
}

size_t DeepKAN::num_params() const {
    size_t total = 0;
    for (const auto& l : layers_) total += l.num_params();
    return total;
}

} // namespace brain19

#pragma once

#include <vector>
#include <cstddef>

namespace brain19 {

// =============================================================================
// EfficientKAN Layer — B-spline basis + matmul formulation
// =============================================================================
//
// Forward:  y = W @ B(x) + SiLU(x @ W_res)  → LayerNorm
// Where B(x) flattens per-input B-spline basis into a single vector,
// making the KAN equivalent to a standard matmul on transformed features.
//
// W shape:     [out_dim, in_dim * basis_size]
// W_res shape: [in_dim, out_dim]  (residual projection)
// basis_size = grid_size + spline_order
//

struct EfficientKANLayerCache {
    std::vector<double> input;          // [in_dim]
    std::vector<double> basis_flat;     // [in_dim * basis_size]
    std::vector<double> basis_deriv;    // [in_dim * basis_size]
    std::vector<double> kan_out;        // [out_dim]
    std::vector<double> z_res;          // [out_dim] (pre-SiLU)
    std::vector<double> pre_norm;       // [out_dim]
    std::vector<double> x_hat;          // [out_dim] (LN normalized)
    std::vector<double> output;         // [out_dim]
    double ln_inv_std;
};

class EfficientKANLayer {
public:
    EfficientKANLayer(size_t in_dim, size_t out_dim,
                      size_t grid_size = 5, size_t spline_order = 3,
                      bool has_residual = true, bool has_layernorm = true);

    // Forward: populates cache, returns output
    std::vector<double> forward(const std::vector<double>& input,
                                 EfficientKANLayerCache& cache) const;

    // Backward: returns d_input, updates weights in-place
    // lr_input_scale: optional per-input-dim LR multiplier (for block-aware LR on first layer)
    std::vector<double> backward(const EfficientKANLayerCache& cache,
                                  const std::vector<double>& d_output,
                                  double lr,
                                  const double* lr_input_scale = nullptr);

    size_t in_dim() const { return in_dim_; }
    size_t out_dim() const { return out_dim_; }
    size_t grid_size() const { return grid_size_; }
    size_t basis_size() const { return basis_size_; }
    size_t num_params() const;

    std::vector<double>& spline_weights() { return weights_; }
    const std::vector<double>& spline_weights() const { return weights_; }
    std::vector<double>& residual_W() { return residual_W_; }
    const std::vector<double>& knots() const { return knots_; }
    std::vector<double>& ln_gamma() { return ln_gamma_; }
    std::vector<double>& ln_beta() { return ln_beta_; }
    const std::vector<double>& ln_gamma() const { return ln_gamma_; }
    const std::vector<double>& ln_beta() const { return ln_beta_; }

private:
    size_t in_dim_, out_dim_, grid_size_, spline_order_, basis_size_;
    std::vector<double> knots_;
    std::vector<double> weights_;      // [out_dim * in_dim * basis_size]
    std::vector<double> residual_W_;   // [in_dim * out_dim]
    std::vector<double> ln_gamma_;     // [out_dim]
    std::vector<double> ln_beta_;      // [out_dim]
    bool has_residual_, has_layernorm_;

    void compute_basis(double x, double* basis, double* deriv) const;
    void initialize();
};

// =============================================================================
// DeepKAN — Stack of EfficientKAN layers
// =============================================================================

class DeepKAN {
public:
    // layer_dims: {input, hidden1, hidden2, ...}  e.g., {90, 256, 128}
    // grid_sizes: one per layer, e.g., {8, 5}
    DeepKAN(const std::vector<size_t>& layer_dims,
            const std::vector<size_t>& grid_sizes,
            size_t spline_order = 3);

    // Forward (caches for backward)
    std::vector<double> forward(const std::vector<double>& input);

    // Backward: updates all weights, returns d_input (usually discarded)
    void backward(const std::vector<double>& d_output, double lr,
                  const double* lr_input_scale = nullptr);

    // Inference only (no caching)
    std::vector<double> inference(const std::vector<double>& input) const;

    size_t input_dim() const;
    size_t output_dim() const;
    size_t num_layers() const { return layers_.size(); }
    size_t num_params() const;

    EfficientKANLayer& layer(size_t i) { return layers_[i]; }
    const EfficientKANLayer& layer(size_t i) const { return layers_[i]; }

private:
    std::vector<EfficientKANLayer> layers_;
    std::vector<EfficientKANLayerCache> caches_;
    size_t spline_order_;
};

} // namespace brain19

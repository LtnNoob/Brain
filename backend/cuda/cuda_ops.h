// cuda/cuda_ops.h — Brain19 CUDA acceleration interface
// CPU-safe header: works with or without USE_CUDA
#pragma once

#include <vector>
#include <cstddef>

namespace brain19 {
namespace cuda {

// ─── Runtime availability ───────────────────────────────────────────────────
// Returns true if compiled with CUDA and a GPU is available
bool gpu_available();

// Returns VRAM in bytes (0 if no GPU)
size_t gpu_vram_bytes();

// ─── Ridge Regression: C = H^T H + λI, B = H^T Y, W = C^{-1} B ────────────
//
// H: N×D matrix (row-major, flattened), Y_indices: N target indices
// Output: W is D×VA (row-major), written into w_out
// lambda: regularization, logit_scale: target scaling
//
// For 8GB VRAM: if H doesn't fit, processes in chunks automatically.
// Falls back to CPU if GPU unavailable or allocation fails.

struct RidgeParams {
    const double* H;          // N × D row-major
    size_t N;                 // number of samples
    size_t D;                 // feature dimension (H_EXT)
    const size_t* targets;    // N target indices (compressed)
    size_t VA;                // active vocab size
    double lambda;
    double logit_scale;
};

// Solve ridge regression. Returns D×VA result in w_out (row-major).
// Returns true if GPU was used, false if CPU fallback.
bool ridge_solve(const RidgeParams& params, double* w_out);

// ─── Batch Matrix Multiply: C = A^T B ───────────────────────────────────────
// A: M×K, B: M×N → C: K×N  (all row-major)
bool matmul_AtB(const double* A, const double* B, double* C,
                size_t M, size_t K, size_t N);

// ─── Gauss-Jordan Inversion ─────────────────────────────────────────────────
// Invert D×D symmetric positive-definite matrix in-place
// inv_out: D×D output
bool matrix_invert(const double* mat, double* inv_out, size_t D);

} // namespace cuda
} // namespace brain19

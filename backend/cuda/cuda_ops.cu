// cuda/cuda_ops.cu — CUDA kernels for Brain19 ridge regression acceleration
// Compiled only with USE_CUDA defined (via nvcc)
#ifdef USE_CUDA

#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// ─── Error handling ─────────────────────────────────────────────────────────

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA] %s failed: %s\n", #call, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

namespace brain19 {
namespace cuda {

// ─── Runtime checks ─────────────────────────────────────────────────────────

static bool s_gpu_checked = false;
static bool s_gpu_ok = false;
static size_t s_vram = 0;

static void check_gpu() {
    if (s_gpu_checked) return;
    s_gpu_checked = true;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return;
    s_vram = prop.totalGlobalMem;
    s_gpu_ok = true;
    fprintf(stderr, "[CUDA] GPU: %s, VRAM: %.1f GB\n",
            prop.name, s_vram / (1024.0*1024.0*1024.0));
}

bool gpu_available() { check_gpu(); return s_gpu_ok; }
size_t gpu_vram_bytes() { check_gpu(); return s_vram; }

// ─── Kernel: Build C = H^T H (symmetric, chunked accumulation) ─────────────
// Each block computes one tile of C[i][j] by iterating over sample chunks
// Grid: (D, D) — but only upper triangle + diagonal

__global__ void kernel_HtH(const double* H, double* C, size_t N, size_t D,
                            size_t chunk_start, size_t chunk_size) {
    size_t i = blockIdx.x;
    size_t j = blockIdx.y;
    if (i > j) return;  // upper triangle only

    // Each thread accumulates a portion of the dot product
    double sum = 0.0;
    for (size_t n = threadIdx.x; n < chunk_size; n += blockDim.x) {
        size_t row = chunk_start + n;
        if (row < N) {
            sum += H[row * D + i] * H[row * D + j];
        }
    }

    // Warp reduction
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block reduction via shared memory
    __shared__ double shared[32];  // max 32 warps = 1024 threads
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        sum = shared[threadIdx.x];
        for (int offset = blockDim.x / warpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(&C[i * D + j], sum);
        if (i != j) atomicAdd(&C[j * D + i], sum);
    }
}

// ─── Kernel: Build B = H^T Y_onehot ────────────────────────────────────────
// Y is sparse (one-hot), so B[:,v] += logit_scale * H[n,:] for target[n]==v

__global__ void kernel_HtY(const double* H, const size_t* targets, double* B,
                           size_t N, size_t D, size_t VA, double logit_scale,
                           size_t chunk_start, size_t chunk_size) {
    size_t n_offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_offset >= chunk_size) return;
    size_t n = chunk_start + n_offset;
    if (n >= N) return;

    size_t v = targets[n];
    if (v >= VA) return;

    for (size_t i = 0; i < D; ++i) {
        atomicAdd(&B[i * VA + v], logit_scale * H[n * D + i]);
    }
}

// ─── Gauss-Jordan on GPU (small matrix, D ~ 128-256) ───────────────────────
// For small D, CPU is fine. We only GPU-accelerate the O(N) parts.

static void cpu_gauss_jordan(const double* C, double* inv, size_t D) {
    // Build augmented matrix [C | I]
    std::vector<double> aug(D * 2 * D, 0.0);
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) aug[i * 2*D + j] = C[i * D + j];
        aug[i * 2*D + D + i] = 1.0;
    }

    for (size_t k = 0; k < D; ++k) {
        // Partial pivot
        double max_val = std::abs(aug[k * 2*D + k]);
        size_t max_row = k;
        for (size_t i = k + 1; i < D; ++i) {
            double v = std::abs(aug[i * 2*D + k]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val < 1e-12) continue;
        if (max_row != k) {
            for (size_t j = 0; j < 2*D; ++j)
                std::swap(aug[k * 2*D + j], aug[max_row * 2*D + j]);
        }

        double pivot = aug[k * 2*D + k];
        for (size_t j = 0; j < 2*D; ++j) aug[k * 2*D + j] /= pivot;

        for (size_t i = 0; i < D; ++i) {
            if (i == k) continue;
            double factor = aug[i * 2*D + k];
            for (size_t j = 0; j < 2*D; ++j)
                aug[i * 2*D + j] -= factor * aug[k * 2*D + j];
        }
    }

    for (size_t i = 0; i < D; ++i)
        for (size_t j = 0; j < D; ++j)
            inv[i * D + j] = aug[i * 2*D + D + j];
}

// ─── Matrix multiply C = A * B (A: D×D, B: D×VA → C: D×VA) on CPU ────────
// (Small enough that GPU overhead isn't worth it for D~128)
static void cpu_matmul(const double* A, const double* B, double* C,
                       size_t D, size_t VA) {
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < VA; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < D; ++k) {
                sum += A[i * D + k] * B[k * VA + j];
            }
            C[i * VA + j] = sum;
        }
    }
}

// ─── Main ridge_solve (GPU-accelerated) ────────────────────────────────────

bool ridge_solve(const RidgeParams& params, double* w_out) {
    check_gpu();
    if (!s_gpu_ok) return false;

    const size_t N = params.N;
    const size_t D = params.D;
    const size_t VA = params.VA;

    // Memory estimates
    // H on GPU: N * D * 8 bytes
    // C on GPU: D * D * 8 bytes
    // B on GPU: D * VA * 8 bytes
    // targets: N * 8 bytes
    size_t H_bytes = N * D * sizeof(double);
    size_t C_bytes = D * D * sizeof(double);
    size_t B_bytes = D * VA * sizeof(double);
    size_t tgt_bytes = N * sizeof(size_t);

    // Determine chunk size for H based on ACTUALLY FREE VRAM (not total!)
    size_t free_mem = 0, total_mem = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaMemGetInfo failed: %s, falling back to CPU\n",
                cudaGetErrorString(mem_err));
        return false;
    }
    fprintf(stderr, "[CUDA] VRAM: %.1fMB free / %.1fMB total\n",
            free_mem / (1024.0*1024.0), total_mem / (1024.0*1024.0));

    // Use at most 20% of FREE VRAM for H chunks — conservative to avoid SIGKILL
    size_t reserved = C_bytes + B_bytes + tgt_bytes + 512ULL * 1024 * 1024;  // 512MB safety margin
    size_t usable = (free_mem > reserved) ? (free_mem - reserved) : 0;
    usable = usable / 5;  // very conservative: use max 20% of remaining free VRAM for H
    size_t max_chunk = usable / (D * sizeof(double));
    if (max_chunk < 1024) {
        fprintf(stderr, "[CUDA] Not enough free VRAM for ridge (free=%zuMB, need C+B+tgt=%zuMB + H chunks), falling back to CPU\n",
                free_mem / (1024*1024), reserved / (1024*1024));
        return false;
    }
    size_t chunk_size = std::min(N, max_chunk);

    fprintf(stderr, "[CUDA] Ridge: N=%zu, D=%zu, VA=%zu, chunk=%zu (%.1fMB/chunk)\n",
            N, D, VA, chunk_size, chunk_size * D * 8.0 / (1024*1024));

    // Allocate GPU memory — with fallback on failure
    double *d_H = nullptr, *d_C = nullptr, *d_B = nullptr;
    size_t *d_targets = nullptr;

    // Helper: free any already-allocated buffers on failure
    auto cleanup = [&]() {
        if (d_H) cudaFree(d_H);
        if (d_C) cudaFree(d_C);
        if (d_B) cudaFree(d_B);
        if (d_targets) cudaFree(d_targets);
    };

    cudaError_t alloc_err;
    alloc_err = cudaMalloc(&d_C, C_bytes);
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaMalloc d_C (%zuMB) failed: %s, falling back to CPU\n",
                C_bytes/(1024*1024), cudaGetErrorString(alloc_err));
        cleanup(); return false;
    }
    CUDA_CHECK(cudaMemset(d_C, 0, C_bytes));

    alloc_err = cudaMalloc(&d_B, B_bytes);
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaMalloc d_B (%zuMB) failed: %s, falling back to CPU\n",
                B_bytes/(1024*1024), cudaGetErrorString(alloc_err));
        cleanup(); return false;
    }
    CUDA_CHECK(cudaMemset(d_B, 0, B_bytes));

    size_t H_chunk_bytes = chunk_size * D * sizeof(double);
    alloc_err = cudaMalloc(&d_H, H_chunk_bytes);
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaMalloc d_H (%zuMB) failed: %s, falling back to CPU\n",
                H_chunk_bytes/(1024*1024), cudaGetErrorString(alloc_err));
        cleanup(); return false;
    }

    alloc_err = cudaMalloc(&d_targets, chunk_size * sizeof(size_t));
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaMalloc d_targets (%zuMB) failed: %s, falling back to CPU\n",
                chunk_size * sizeof(size_t)/(1024*1024), cudaGetErrorString(alloc_err));
        cleanup(); return false;
    }

    // Process in chunks
    for (size_t offset = 0; offset < N; offset += chunk_size) {
        size_t this_chunk = std::min(chunk_size, N - offset);

        CUDA_CHECK(cudaMemcpy(d_H, params.H + offset * D,
                              this_chunk * D * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, params.targets + offset,
                              this_chunk * sizeof(size_t), cudaMemcpyHostToDevice));

        // Build C += chunk_H^T chunk_H
        dim3 grid_C(D, D);
        int threads_C = std::min((size_t)256, this_chunk);
        kernel_HtH<<<grid_C, threads_C>>>(d_H, d_C, this_chunk, D, 0, this_chunk);
        {
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                fprintf(stderr, "[CUDA] kernel_HtH launch failed: %s\n", cudaGetErrorString(kerr));
                cleanup(); return false;
            }
        }

        // Build B += chunk_H^T Y_onehot
        int threads_B = 256;
        int blocks_B = (this_chunk + threads_B - 1) / threads_B;
        kernel_HtY<<<blocks_B, threads_B>>>(d_H, d_targets, d_B,
                                             this_chunk, D, VA, params.logit_scale,
                                             0, this_chunk);
        {
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                fprintf(stderr, "[CUDA] kernel_HtY launch failed: %s\n", cudaGetErrorString(kerr));
                cleanup(); return false;
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy C and B back to host
    std::vector<double> C_host(D * D);
    std::vector<double> B_host(D * VA);
    CUDA_CHECK(cudaMemcpy(C_host.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B_host.data(), d_B, B_bytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    cleanup();

    // Add regularization
    for (size_t i = 0; i < D; ++i) {
        C_host[i * D + i] += params.lambda;
    }

    // Invert C on CPU (D×D is small, ~128-256)
    std::vector<double> C_inv(D * D);
    cpu_gauss_jordan(C_host.data(), C_inv.data(), D);

    // W = C^{-1} * B on CPU
    cpu_matmul(C_inv.data(), B_host.data(), w_out, D, VA);

    fprintf(stderr, "[CUDA] Ridge solve complete\n");
    return true;
}

bool matmul_AtB(const double* A, const double* B, double* C,
                size_t M, size_t K, size_t N) {
    check_gpu();
    if (!s_gpu_ok) return false;
    // For now, only ridge_solve is GPU-accelerated
    // The H^T H computation is the bottleneck; other matmuls are small
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
    return false;
}

bool matrix_invert(const double* mat, double* inv_out, size_t D) {
    check_gpu();
    if (!s_gpu_ok) return false;
    cpu_gauss_jordan(mat, inv_out, D);
    return true;
}

} // namespace cuda
} // namespace brain19

#endif // USE_CUDA

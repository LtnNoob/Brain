// cuda/cuda_training_v2.cu — Block Coordinate Descent CUDA Training
// Sequential samples (like CPU), parallel gradient computation per token.
// No weight averaging — true online SGD with CUDA-accelerated matmuls.
#ifdef USE_CUDA

#include "cuda_training.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA-BCD] %s failed: %s\n", #call, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

namespace brain19 {
namespace cuda {

// ─── Constants ──────────────────────────────────────────────────────────────
// Max dimensions (stack arrays in kernels)
static constexpr int MAX_H = 128;
static constexpr int MAX_H_EXT = 256;
static constexpr int MAX_K = 64;
static constexpr int MAX_VA = 512;

// ─── Kernel: Transform forward ──────────────────────────────────────────────
// Computes h_out = h + tanh(h·W1+b1)·W2+b2, then h_ext = [h_out, h_out²]
// Launch with H threads — each thread computes one output dimension
__global__ void kernel_transform_forward(
    const double* __restrict__ h,       // [H]
    const double* __restrict__ W1,      // [H * K]
    const double* __restrict__ b1,      // [K]
    const double* __restrict__ W2,      // [K * H]
    const double* __restrict__ b2,      // [H]
    double* __restrict__ h_out,         // [H]
    double* __restrict__ h_ext,         // [H_EXT]
    double* __restrict__ a1,            // [K] — saved for backprop
    double* __restrict__ z1,            // [K] — saved for backprop
    size_t H, size_t K
) {
    // Phase 1: compute z1, a1 (K threads needed)
    // We use thread 0..K-1 for this
    size_t tid = threadIdx.x;

    // Step 1: Each of first K threads computes one z1[k], a1[k]
    if (tid < K) {
        double sum = b1[tid];
        for (size_t i = 0; i < H; ++i)
            sum += h[i] * W1[i * K + tid];
        z1[tid] = sum;
        a1[tid] = tanh(sum);
    }
    __syncthreads();

    // Step 2: Each of H threads computes h_out[tid] and h_ext
    if (tid < H) {
        double sum = h[tid] + b2[tid];
        for (size_t k = 0; k < K; ++k)
            sum += a1[k] * W2[k * H + tid];
        h_out[tid] = sum;
        h_ext[tid] = sum;
        h_ext[H + tid] = sum * sum;
    }
}

// ─── Kernel: Compute logits = h_ext · W_a ──────────────────────────────────
// Launch with VA threads — each thread computes one logit
__global__ void kernel_logits(
    const double* __restrict__ h_ext,   // [H_EXT]
    const double* __restrict__ W_a,     // [H_EXT * VA]
    double* __restrict__ logits,        // [VA]
    size_t H_EXT, size_t VA
) {
    size_t a = threadIdx.x;
    if (a >= VA) return;
    double sum = 0.0;
    for (size_t i = 0; i < H_EXT; ++i)
        sum += h_ext[i] * W_a[i * VA + a];
    logits[a] = sum;
}

// ─── Kernel: Softmax + CE loss ──────────────────────────────────────────────
// Single thread — VA is small enough
__global__ void kernel_softmax_loss(
    double* __restrict__ logits,        // [VA] — overwritten with probs
    size_t VA, size_t target_ca,
    double* __restrict__ loss_out       // [1] — atomicAdd
) {
    // Find max
    double max_val = logits[0];
    for (size_t a = 1; a < VA; ++a)
        if (logits[a] > max_val) max_val = logits[a];

    // Exp + sum
    double exp_sum = 0.0;
    for (size_t a = 0; a < VA; ++a) {
        double v = logits[a] - max_val;
        logits[a] = exp(v < 80.0 ? v : 80.0);
        exp_sum += logits[a];
    }

    // Normalize to probs (in-place)
    if (exp_sum > 1e-12) {
        double inv = 1.0 / exp_sum;
        for (size_t a = 0; a < VA; ++a)
            logits[a] *= inv;
    }

    // CE loss
    double p = logits[target_ca] > 1e-12 ? logits[target_ca] : 1e-12;
    atomicAdd(loss_out, -log(p));
}

// ─── Kernel: W_a gradient + update (Block A) ───────────────────────────────
// Launch with H_EXT * VA threads (or grid of blocks)
// grad_W_a[i][a] = h_ext[i] * (probs[a] - (a==ca))
// W_a[i][a] -= lr * grad
__global__ void kernel_update_Wa(
    double* __restrict__ W_a,           // [H_EXT * VA]
    const double* __restrict__ h_ext,   // [H_EXT]
    const double* __restrict__ probs,   // [VA]
    size_t H_EXT, size_t VA,
    size_t target_ca, double lr
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H_EXT * VA) return;

    size_t i = idx / VA;
    size_t a = idx % VA;

    double grad = h_ext[i] * probs[a];
    if (a == target_ca) grad -= h_ext[i];
    W_a[idx] -= lr * grad;
}

// ─── Kernel: Compute d_h_ext for transform backprop ────────────────────────
// d_h_ext[i] = Σ_a probs[a] * W_a[i*VA+a] - W_a[i*VA+ca]
// Launch with H_EXT threads
__global__ void kernel_compute_d_h_ext(
    const double* __restrict__ W_a,     // [H_EXT * VA] — PRE-update values needed!
    const double* __restrict__ probs,   // [VA]
    double* __restrict__ d_h_ext,       // [H_EXT]
    size_t H_EXT, size_t VA, size_t target_ca
) {
    size_t i = threadIdx.x;
    if (i >= H_EXT) return;

    double d = 0.0;
    for (size_t a = 0; a < VA; ++a)
        d += probs[a] * W_a[i * VA + a];
    d -= W_a[i * VA + target_ca];
    d_h_ext[i] = d;
}

// ─── Kernel: Transform backprop + update (Block B+C) ────────────────────────
// Single block, H threads. Updates W1, b1, W2 in-place.
__global__ void kernel_update_transform(
    double* __restrict__ W1,            // [H * K]
    double* __restrict__ b1,            // [K]
    double* __restrict__ W2,            // [K * H]
    const double* __restrict__ h,       // [H]
    const double* __restrict__ h_out,   // [H]
    const double* __restrict__ a1,      // [K]
    const double* __restrict__ d_h_ext, // [H_EXT]
    size_t H, size_t K, double lr_transform
) {
    size_t tid = threadIdx.x;

    // Shared memory for d_h_out, d_a1, d_z1
    extern __shared__ double smem[];  // [H + K + K]
    double* s_d_h_out = smem;         // [H]
    double* s_d_a1 = smem + H;        // [K]
    double* s_d_z1 = smem + H + K;    // [K]

    // Step 1: d_h_out[i] = d_h_ext[i] + 2*h_out[i]*d_h_ext[H+i]
    if (tid < H) {
        s_d_h_out[tid] = d_h_ext[tid] + 2.0 * h_out[tid] * d_h_ext[H + tid];
    }
    __syncthreads();

    // Step 2: d_a1[k] = Σ_j d_h_out[j] * W2[k*H+j]
    if (tid < K) {
        double d = 0.0;
        for (size_t j = 0; j < H; ++j)
            d += s_d_h_out[j] * W2[tid * H + j];
        s_d_a1[tid] = d;
    }
    __syncthreads();

    // Step 3: Update W2[k][j] -= lr * a1[k] * d_h_out[j]
    // Each thread handles one j, loops over k
    if (tid < H) {
        for (size_t k = 0; k < K; ++k)
            W2[k * H + tid] -= lr_transform * a1[k] * s_d_h_out[tid];
    }
    __syncthreads();

    // Step 4: d_z1[k] = d_a1[k] * (1 - a1[k]²)
    if (tid < K) {
        s_d_z1[tid] = s_d_a1[tid] * (1.0 - a1[tid] * a1[tid]);
    }
    __syncthreads();

    // Step 5: Update W1[i][k] -= lr * h[i] * d_z1[k]
    if (tid < H) {
        for (size_t k = 0; k < K; ++k)
            W1[tid * K + k] -= lr_transform * h[tid] * s_d_z1[k];
    }

    // Step 6: Update b1[k] -= lr * d_z1[k] (first K threads)
    if (tid < K) {
        b1[tid] -= lr_transform * s_d_z1[tid];
    }
}

// ─── Kernel: Evolve hidden state ────────────────────────────────────────────
// Launch with H threads
__global__ void kernel_evolve_h(
    double* __restrict__ h,             // [H]
    const double* __restrict__ emb_table, // [V * FUSED_BASE]
    uint16_t target_tok, size_t V,
    size_t FUSED_BASE, size_t H
) {
    size_t i = threadIdx.x;
    if (i >= H) return;

    if (i < FUSED_BASE && target_tok < V) {
        h[i] = h[i] * 0.8 + emb_table[target_tok * FUSED_BASE + i] * 0.2;
    } else if (i >= FUSED_BASE) {
        h[i] *= 0.95;
    }
}

// ─── Host: Block Coordinate Descent training loop ───────────────────────────

bool train_sgd_gpu(const TrainingData& data,
                   TrainingWeights& weights,
                   const TrainingConfig& config,
                   TrainingResult& result) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return false;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "[CUDA-BCD] VRAM: %.1fMB free / %.1fMB total\n",
            free_mem/(1024.0*1024.0), total_mem/(1024.0*1024.0));

    const size_t NS = data.num_samples;
    const size_t H = data.H;
    const size_t H_EXT = data.H_EXT;
    const size_t K = data.K;
    const size_t VA = data.VA;
    const size_t V = data.V;
    const size_t FUSED_BASE = data.FUSED_BASE;

    if (H > MAX_H || H_EXT > MAX_H_EXT || K > MAX_K || VA > MAX_VA) {
        fprintf(stderr, "[CUDA-BCD] Dims too large for kernel stack arrays\n");
        return false;
    }

    // Weight sizes
    size_t wa_size = H_EXT * VA;
    size_t w1_size = H * K;
    size_t b1_size = K;
    size_t w2_size = K * H;
    size_t b2_size = H;

    // ── Allocate device memory ──
    // Static data (read-only during training)
    uint16_t* d_all_tokens = nullptr;
    size_t* d_sample_offsets = nullptr;
    size_t* d_sample_lengths = nullptr;
    double* d_embeddings = nullptr;
    size_t* d_compress = nullptr;
    double* d_emb_table = nullptr;

    // Weights (single copy, updated in-place — true online SGD)
    double* d_W_a = nullptr;
    double* d_W1 = nullptr;
    double* d_b1 = nullptr;
    double* d_W2 = nullptr;
    double* d_b2 = nullptr;

    // Working buffers (single sample at a time)
    double* d_h = nullptr;         // [H]
    double* d_h_out = nullptr;     // [H]
    double* d_h_ext = nullptr;     // [H_EXT]
    double* d_a1 = nullptr;        // [K]
    double* d_z1 = nullptr;        // [K]
    double* d_logits = nullptr;    // [VA] — also used as probs after softmax
    double* d_d_h_ext = nullptr;   // [H_EXT]
    double* d_loss = nullptr;      // [1]

    // Allocate static data
    CUDA_CHECK(cudaMalloc(&d_all_tokens, data.all_tokens.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_sample_offsets, NS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_sample_lengths, NS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_embeddings, data.embeddings.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_compress, V * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_emb_table, data.emb_table.size() * sizeof(double)));

    // Allocate weights
    CUDA_CHECK(cudaMalloc(&d_W_a, wa_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W1, w1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1, b1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2, w2_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2, b2_size * sizeof(double)));

    // Allocate working buffers
    CUDA_CHECK(cudaMalloc(&d_h, H * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_h_out, H * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_h_ext, H_EXT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a1, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z1, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_logits, VA * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d_h_ext, H_EXT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(double)));

    // Copy static data to device
    CUDA_CHECK(cudaMemcpy(d_all_tokens, data.all_tokens.data(), data.all_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_offsets, data.sample_offsets.data(), NS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_lengths, data.sample_lengths.data(), NS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_embeddings, data.embeddings.data(), data.embeddings.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_compress, data.compress.data(), V * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emb_table, data.emb_table.data(), data.emb_table.size() * sizeof(double), cudaMemcpyHostToDevice));

    // Copy weights to device
    CUDA_CHECK(cudaMemcpy(d_W_a, weights.W_a.data(), wa_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, weights.W1.data(), w1_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, weights.b1.data(), b1_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, weights.W2.data(), w2_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, weights.b2.data(), b2_size * sizeof(double), cudaMemcpyHostToDevice));

    // Create two CUDA streams for parallel gradient updates
    cudaStream_t stream_Wa, stream_transform;
    CUDA_CHECK(cudaStreamCreate(&stream_Wa));
    CUDA_CHECK(cudaStreamCreate(&stream_transform));

    // Host-side buffers for sample metadata
    std::vector<size_t> h_offsets(NS), h_lengths(NS);
    std::copy(data.sample_offsets.begin(), data.sample_offsets.end(), h_offsets.begin());
    std::copy(data.sample_lengths.begin(), data.sample_lengths.end(), h_lengths.begin());

    // Host-side compress map for token lookup
    const auto& compress_map = data.compress;

    // Host-side token data for sequential iteration
    const auto& all_tokens = data.all_tokens;

    auto t_start = std::chrono::steady_clock::now();
    result.best_loss = 1e9;

    // Shared memory size for transform backprop kernel
    size_t transform_smem = (H + K + K) * sizeof(double);

    // Thread configs
    size_t threads_H = ((H + 31) / 32) * 32;       // round up to warp
    if (threads_H < 32) threads_H = 32;
    size_t threads_VA = ((VA + 31) / 32) * 32;
    if (threads_VA < 32) threads_VA = 32;
    size_t wa_total = wa_size;
    size_t wa_blocks = (wa_total + 255) / 256;

    fprintf(stderr, "[CUDA-BCD] Starting %zu epochs, %zu samples, H=%zu K=%zu VA=%zu\n",
            config.num_epochs, NS, H, K, VA);
    fprintf(stderr, "[CUDA-BCD] Strategy: sequential samples, parallel gradient blocks (2 streams)\n");
    fprintf(stderr, "[CUDA-BCD] Block A (W_a): %zu elements, Block B+C (transform): %zu elements\n",
            wa_size, w1_size + b1_size + w2_size);

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        bool train_transform = (epoch >= config.transform_warmup);
        double progress = static_cast<double>(epoch) / std::max(config.num_epochs - 1, size_t(1));
        double cos_mult = 0.5 * (1.0 + cos(progress * 3.14159265358979));
        double lr_epoch = config.base_lr * (0.1 + 0.9 * cos_mult);
        double lr_transform_epoch = config.lr_transform_base * (0.1 + 0.9 * cos_mult);

        // Reset loss accumulator
        double zero = 0.0;
        CUDA_CHECK(cudaMemcpy(d_loss, &zero, sizeof(double), cudaMemcpyHostToDevice));

        size_t epoch_tokens = 0;

        for (size_t s = 0; s < NS; ++s) {
            size_t tok_start = h_offsets[s];
            size_t tok_len = h_lengths[s];

            // Initialize h from embedding on device
            CUDA_CHECK(cudaMemcpy(d_h, d_embeddings + s * H, H * sizeof(double), cudaMemcpyDeviceToDevice));

            for (size_t t = 0; t < tok_len; ++t) {
                uint16_t target_tok = all_tokens[tok_start + t];
                if (target_tok >= V) continue;
                size_t ca = compress_map[target_tok];
                if (ca >= VA) continue;

                epoch_tokens++;

                // 1. Transform forward (H threads)
                kernel_transform_forward<<<1, threads_H>>>(
                    d_h, d_W1, d_b1, d_W2, d_b2,
                    d_h_out, d_h_ext, d_a1, d_z1,
                    H, K
                );

                // 2. Compute logits (VA threads)
                kernel_logits<<<1, threads_VA>>>(
                    d_h_ext, d_W_a, d_logits,
                    H_EXT, VA
                );

                // 3. Softmax + loss (1 thread)
                kernel_softmax_loss<<<1, 1>>>(
                    d_logits, VA, ca, d_loss
                );

                // After softmax, d_logits contains probs
                // Now we need d_h_ext BEFORE updating W_a (uses pre-update W_a)

                if (train_transform) {
                    // Compute d_h_ext on default stream (needs pre-update W_a)
                    kernel_compute_d_h_ext<<<1, ((H_EXT + 31) / 32) * 32>>>(
                        d_W_a, d_logits, d_d_h_ext,
                        H_EXT, VA, ca
                    );
                }

                // Synchronize default stream before launching on separate streams
                CUDA_CHECK(cudaDeviceSynchronize());

                // 5. Block A: Update W_a (stream_Wa) — INDEPENDENT
                kernel_update_Wa<<<wa_blocks, 256, 0, stream_Wa>>>(
                    d_W_a, d_h_ext, d_logits,
                    H_EXT, VA, ca, lr_epoch
                );

                // 6. Block B+C: Update transform weights (stream_transform) — INDEPENDENT
                if (train_transform) {
                    kernel_update_transform<<<1, threads_H, transform_smem, stream_transform>>>(
                        d_W1, d_b1, d_W2,
                        d_h, d_h_out, d_a1, d_d_h_ext,
                        H, K, lr_transform_epoch
                    );
                }

                // Synchronize both streams before evolving h (next token depends on updated state)
                CUDA_CHECK(cudaStreamSynchronize(stream_Wa));
                CUDA_CHECK(cudaStreamSynchronize(stream_transform));

                // 7. Evolve hidden state
                kernel_evolve_h<<<1, threads_H>>>(
                    d_h, d_emb_table, target_tok, V, FUSED_BASE, H
                );
            }
        }

        // Read back epoch loss
        double epoch_loss_val = 0.0;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&epoch_loss_val, d_loss, sizeof(double), cudaMemcpyDeviceToHost));

        double loss = (epoch_tokens > 0) ? epoch_loss_val / epoch_tokens : 1e9;
        if (loss < result.best_loss) result.best_loss = loss;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            fprintf(stderr, "[CUDA-BCD]   Epoch %zu/%zu loss=%.5f lr=%.4f toks=%zu (%zums)\n",
                    epoch + 1, config.num_epochs, loss, lr_epoch, epoch_tokens, elapsed);
        }
        if (loss < 0.5) break;
    }

    // Copy weights back
    CUDA_CHECK(cudaMemcpy(weights.W_a.data(), d_W_a, wa_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W1.data(), d_W1, w1_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.b1.data(), d_b1, b1_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W2.data(), d_W2, w2_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.b2.data(), d_b2, b2_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaStreamDestroy(stream_Wa);
    cudaStreamDestroy(stream_transform);
    cudaFree(d_all_tokens); cudaFree(d_sample_offsets); cudaFree(d_sample_lengths);
    cudaFree(d_embeddings); cudaFree(d_compress); cudaFree(d_emb_table);
    cudaFree(d_W_a); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_h); cudaFree(d_h_out); cudaFree(d_h_ext);
    cudaFree(d_a1); cudaFree(d_z1); cudaFree(d_logits);
    cudaFree(d_d_h_ext); cudaFree(d_loss);

    fprintf(stderr, "[CUDA-BCD] Complete, best loss=%.5f\n", result.best_loss);
    return true;
}

} // namespace cuda
} // namespace brain19

#endif // USE_CUDA

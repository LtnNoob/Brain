// cuda/cuda_training.cu — Monolithic CUDA Training Kernel
// One kernel launch per epoch. Sequential samples+tokens on GPU.
// H threads parallelize matmuls within each token step.
// All data in VRAM — no CPU-GPU transfers during training.
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
        fprintf(stderr, "[CUDA-Mono] %s failed: %s\n", #call, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

namespace brain19 {
namespace cuda {

// ─── Monolithic kernel: one block, H threads, all samples sequential ────────
// Thread 0 does scalar work (softmax, loss, control flow)
// All H threads do parallel matmuls
__global__ void kernel_epoch(
    const uint16_t* all_tokens,
    const size_t* sample_offsets,
    const size_t* sample_lengths,
    const double* embeddings,       // [NS * H]
    const size_t* compress,         // [V]
    const double* emb_table,        // [V * FUSED_BASE]
    double* W_a,                    // [H_EXT * VA]
    double* W1,                     // [H * K]
    double* b1,                     // [K]
    double* W2,                     // [K * H]
    double* b2,                     // [H]
    size_t NS, size_t V, size_t VA,
    size_t H, size_t H_EXT,
    size_t K, size_t FUSED_BASE,
    double lr_epoch, double lr_transform_epoch,
    bool train_transform,
    double* loss_out,               // [1]
    size_t* tokens_out              // [1]
) {
    // Shared memory layout:
    // h[H], h_out[H], h_ext[H_EXT], a1[K], z1[K],
    // logits[VA], probs[VA], d_h_ext[H_EXT], d_h_out[H], d_a1[K], d_z1[K]
    extern __shared__ double smem[];
    
    size_t tid = threadIdx.x;
    
    // Partition shared memory
    double* h = smem;                                    // [H]
    double* h_out = h + H;                               // [H]
    double* h_ext = h_out + H;                           // [H_EXT]
    double* a1 = h_ext + H_EXT;                          // [K]
    double* z1_unused = a1 + K;                          // [K] (for alignment)
    double* logits = z1_unused + K;                      // [VA]
    double* d_h_ext = logits + VA;                       // [H_EXT]
    double* d_h_out = d_h_ext + H_EXT;                  // [H]
    double* d_a1 = d_h_out + H;                          // [K]
    double* d_z1 = d_a1 + K;                             // [K]
    // Total: 4H + 2*H_EXT + 4K + VA doubles

    double total_loss = 0.0;
    size_t total_tokens = 0;

    for (size_t s = 0; s < NS; ++s) {
        size_t tok_start = sample_offsets[s];
        size_t tok_len = sample_lengths[s];

        // Init h from embedding (parallel)
        if (tid < H) {
            h[tid] = embeddings[s * H + tid];
        }
        __syncthreads();

        for (size_t t = 0; t < tok_len; ++t) {
            uint16_t target_tok = all_tokens[tok_start + t];
            if (target_tok >= V) continue;
            size_t ca = compress[target_tok];
            if (ca >= VA) continue;

            // ── Forward: Transform ──
            // Step 1: z1[k] = b1[k] + Σ h[i]*W1[i*K+k], a1[k] = tanh(z1[k])
            if (tid < K) {
                double sum = b1[tid];
                for (size_t i = 0; i < H; ++i)
                    sum += h[i] * W1[i * K + tid];
                a1[tid] = tanh(sum);
            }
            __syncthreads();

            // Step 2: h_out[j] = h[j] + b2[j] + Σ a1[k]*W2[k*H+j]
            if (tid < H) {
                double sum = h[tid] + b2[tid];
                for (size_t k = 0; k < K; ++k)
                    sum += a1[k] * W2[k * H + tid];
                h_out[tid] = sum;
                h_ext[tid] = sum;
                h_ext[H + tid] = sum * sum;
            }
            __syncthreads();

            // ── Forward: Logits ──
            // logits[a] = Σ h_ext[i]*W_a[i*VA+a]
            if (tid < VA) {
                double sum = 0.0;
                for (size_t i = 0; i < H_EXT; ++i)
                    sum += h_ext[i] * W_a[i * VA + tid];
                logits[tid] = sum;
            }
            __syncthreads();

            // ── Softmax + CE loss (thread 0 only) ──
            if (tid == 0) {
                double max_val = logits[0];
                for (size_t a = 1; a < VA; ++a)
                    if (logits[a] > max_val) max_val = logits[a];
                
                double exp_sum = 0.0;
                for (size_t a = 0; a < VA; ++a) {
                    double v = logits[a] - max_val;
                    logits[a] = exp(v < 80.0 ? v : 80.0);
                    exp_sum += logits[a];
                }
                if (exp_sum > 1e-12) {
                    double inv = 1.0 / exp_sum;
                    for (size_t a = 0; a < VA; ++a)
                        logits[a] *= inv;
                }

                double p = logits[ca] > 1e-12 ? logits[ca] : 1e-12;
                total_loss += -log(p);
                total_tokens++;
            }
            __syncthreads();
            // Now logits[] contains probs

            // ── Compute d_h_ext (pre-update W_a) ──
            if (train_transform && tid < H_EXT) {
                double d = 0.0;
                for (size_t a = 0; a < VA; ++a)
                    d += logits[a] * W_a[tid * VA + a];
                d -= W_a[tid * VA + ca];
                d_h_ext[tid] = d;
            }
            __syncthreads();

            // ── Update W_a ──
            // Each thread handles a slice of H_EXT rows
            for (size_t i = tid; i < H_EXT; i += blockDim.x) {
                double hi = h_ext[i];
                for (size_t a = 0; a < VA; ++a) {
                    W_a[i * VA + a] -= lr_epoch * hi * logits[a];
                }
                W_a[i * VA + ca] += lr_epoch * hi;
            }
            __syncthreads();

            // ── Backprop through transform ──
            if (train_transform) {
                // d_h_out
                if (tid < H) {
                    d_h_out[tid] = d_h_ext[tid] + 2.0 * h_out[tid] * d_h_ext[H + tid];
                }
                __syncthreads();

                // d_a1
                if (tid < K) {
                    double d = 0.0;
                    for (size_t j = 0; j < H; ++j)
                        d += d_h_out[j] * W2[tid * H + j];
                    d_a1[tid] = d;
                }
                __syncthreads();

                // Update W2
                if (tid < H) {
                    for (size_t k = 0; k < K; ++k)
                        W2[k * H + tid] -= lr_transform_epoch * a1[k] * d_h_out[tid];
                }
                __syncthreads();

                // d_z1
                if (tid < K) {
                    d_z1[tid] = d_a1[tid] * (1.0 - a1[tid] * a1[tid]);
                }
                __syncthreads();

                // Update W1
                if (tid < H) {
                    for (size_t k = 0; k < K; ++k)
                        W1[tid * K + k] -= lr_transform_epoch * h[tid] * d_z1[k];
                }

                // Update b1
                if (tid < K) {
                    b1[tid] -= lr_transform_epoch * d_z1[tid];
                }
                __syncthreads();
            }

            // ── Evolve hidden state ──
            if (tid < FUSED_BASE && target_tok < V) {
                h[tid] = h[tid] * 0.8 + emb_table[target_tok * FUSED_BASE + tid] * 0.2;
            } else if (tid >= FUSED_BASE && tid < H) {
                h[tid] *= 0.95;
            }
            __syncthreads();
        }
    }

    // Write loss/tokens (thread 0)
    if (tid == 0) {
        *loss_out = total_loss;
        *tokens_out = total_tokens;
    }
}

// ─── Host function ──────────────────────────────────────────────────────────

bool train_sgd_gpu(const TrainingData& data,
                   TrainingWeights& weights,
                   const TrainingConfig& config,
                   TrainingResult& result) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return false;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "[CUDA-Mono] VRAM: %.1fMB free / %.1fMB total\n",
            free_mem/(1024.0*1024.0), total_mem/(1024.0*1024.0));

    const size_t NS = data.num_samples;
    const size_t H = data.H;
    const size_t H_EXT = data.H_EXT;
    const size_t K = data.K;
    const size_t VA = data.VA;
    const size_t V = data.V;
    const size_t FUSED_BASE = data.FUSED_BASE;

    size_t wa_size = H_EXT * VA;
    size_t w1_size = H * K;
    size_t b1_size = K;
    size_t w2_size = K * H;

    // Shared memory: 4H + 2*H_EXT + 4K + VA doubles
    size_t smem_size = (4*H + 2*H_EXT + 4*K + VA) * sizeof(double);
    fprintf(stderr, "[CUDA-Mono] Shared memory: %zu bytes (limit 48KB)\n", smem_size);
    if (smem_size > 48 * 1024) {
        fprintf(stderr, "[CUDA-Mono] Shared memory exceeds 48KB limit!\n");
        return false;
    }

    // Thread count: max(H, K, VA), rounded up to warp
    size_t threads = H;
    if (VA > threads) threads = VA;
    if (H_EXT > threads) threads = H_EXT;
    threads = ((threads + 31) / 32) * 32;
    if (threads > 1024) threads = 1024;

    // Allocate device memory
    uint16_t* d_all_tokens; size_t* d_sample_offsets; size_t* d_sample_lengths;
    double* d_embeddings; size_t* d_compress; double* d_emb_table;
    double* d_W_a; double* d_W1; double* d_b1; double* d_W2; double* d_b2;
    double* d_loss; size_t* d_tokens;

    CUDA_CHECK(cudaMalloc(&d_all_tokens, data.all_tokens.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_sample_offsets, NS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_sample_lengths, NS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_embeddings, data.embeddings.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_compress, V * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_emb_table, data.emb_table.size() * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_W_a, wa_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W1, w1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1, b1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2, w2_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2, H * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tokens, sizeof(size_t)));

    // Copy data (once)
    CUDA_CHECK(cudaMemcpy(d_all_tokens, data.all_tokens.data(), data.all_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_offsets, data.sample_offsets.data(), NS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_lengths, data.sample_lengths.data(), NS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_embeddings, data.embeddings.data(), data.embeddings.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_compress, data.compress.data(), V * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emb_table, data.emb_table.data(), data.emb_table.size() * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_W_a, weights.W_a.data(), wa_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, weights.W1.data(), w1_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, weights.b1.data(), b1_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, weights.W2.data(), w2_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, weights.b2.data(), H * sizeof(double), cudaMemcpyHostToDevice));

    auto t_start = std::chrono::steady_clock::now();
    result.best_loss = 1e9;

    fprintf(stderr, "[CUDA-Mono] Starting %zu epochs, %zu samples, %zu threads, smem=%zuB\n",
            config.num_epochs, NS, threads, smem_size);

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        bool train_transform = (epoch >= config.transform_warmup);
        double progress = static_cast<double>(epoch) / std::max(config.num_epochs - 1, size_t(1));
        double cos_mult = 0.5 * (1.0 + cos(progress * 3.14159265358979));
        double lr_epoch = config.base_lr * (0.1 + 0.9 * cos_mult);
        double lr_transform_epoch = config.lr_transform_base * (0.1 + 0.9 * cos_mult);

        // One kernel launch per epoch!
        kernel_epoch<<<1, threads, smem_size>>>(
            d_all_tokens, d_sample_offsets, d_sample_lengths,
            d_embeddings, d_compress, d_emb_table,
            d_W_a, d_W1, d_b1, d_W2, d_b2,
            NS, V, VA, H, H_EXT, K, FUSED_BASE,
            lr_epoch, lr_transform_epoch, train_transform,
            d_loss, d_tokens
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        double epoch_loss;
        size_t epoch_tokens;
        CUDA_CHECK(cudaMemcpy(&epoch_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&epoch_tokens, d_tokens, sizeof(size_t), cudaMemcpyDeviceToHost));

        double loss = (epoch_tokens > 0) ? epoch_loss / epoch_tokens : 1e9;
        if (loss < result.best_loss) result.best_loss = loss;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            fprintf(stderr, "[CUDA-Mono]   Epoch %zu/%zu loss=%.5f lr=%.4f (%zums)\n",
                    epoch + 1, config.num_epochs, loss, lr_epoch, elapsed);
        }
        if (loss < 0.5) break;
    }

    CUDA_CHECK(cudaMemcpy(weights.W_a.data(), d_W_a, wa_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W1.data(), d_W1, w1_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.b1.data(), d_b1, b1_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W2.data(), d_W2, w2_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.b2.data(), d_b2, H * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_all_tokens); cudaFree(d_sample_offsets); cudaFree(d_sample_lengths);
    cudaFree(d_embeddings); cudaFree(d_compress); cudaFree(d_emb_table);
    cudaFree(d_W_a); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_loss); cudaFree(d_tokens);

    fprintf(stderr, "[CUDA-Mono] Complete, best loss=%.5f\n", result.best_loss);
    return true;
}

// =============================================================================
// V11 Fused Per-Sample SGD
// =============================================================================
//
// Architecture: One block per sample, fused forward + weight update.
// No gradient accumulators, no separate update kernels.
// Direct W -= lr * grad in the forward kernel (like CPU online SGD).
//
// Each block: sequential token loop, parallel matmuls via 256 threads.
// Block-aware LR: Token×1.0, Flex×0.3, DimCtx×0.1 applied per-dim.
// Grid: NS blocks (one per sample per kernel launch batch).
//
// =============================================================================

static constexpr size_t V11_TPB = 256;

// ── Monolithic V11 kernel: 1 block, 256 threads, all samples sequential ──
// Like V10 mono but with: flex_dim, flex_table, block-aware LR
// 1 kernel launch per epoch — zero API overhead per sample
__global__ void kernel_epoch_v11(
    const uint16_t* all_tokens,
    const size_t* sample_offsets,
    const size_t* sample_lengths,
    const double* embeddings,       // [NS * H]
    const size_t* compress,         // [V]
    const double* emb_table,        // [V * FUSED_BASE]
    const double* flex_table,       // [V * flex_dim]
    const double* conv_table,       // [V * conv_dim]
    double* W_a,                    // [H_EXT * VA]
    double* W1,                     // [H * K]
    double* b1,                     // [K]
    double* W2,                     // [K * H]
    size_t NS, size_t V, size_t VA,
    size_t H, size_t H_EXT, size_t K,
    size_t FUSED_BASE, size_t flex_dim, size_t conv_dim,
    bool train_transform,
    double lr_A, double lr_B, double lr_C, double lr_D, double lr_transform,
    double* loss_out, unsigned long long* tokens_out
) {
    const int tid = threadIdx.x;
    const int BDX = blockDim.x;

    extern __shared__ double smem[];
    double* h       = smem;
    double* h_out   = h + H;
    double* h_ext   = h_out + H;
    double* a1      = h_ext + H_EXT;
    double* logits  = a1 + K;
    double* d_h_ext = logits + VA;
    double* d_h_out = d_h_ext + H_EXT;
    double* d_a1    = d_h_out + H;
    double* d_z1    = d_a1 + K;

    double total_loss = 0.0;
    unsigned long long total_tokens = 0;

    for (size_t s = 0; s < NS; ++s) {
        size_t tok_start = sample_offsets[s];
        size_t tok_len   = sample_lengths[s];

        // Init h from embedding
        for (size_t i = tid; i < H; i += BDX)
            h[i] = embeddings[s * H + i];
        __syncthreads();

        for (size_t t = 0; t < tok_len; ++t) {
            uint16_t target_tok = all_tokens[tok_start + t];
            if (target_tok >= V) continue;
            size_t ca = compress[target_tok];
            if (ca >= VA) continue;

            // ── Transform forward ──
            if (train_transform) {
                for (size_t k = tid; k < K; k += BDX) {
                    double sum = b1[k];
                    for (size_t i = 0; i < H; ++i)
                        sum += h[i] * W1[i * K + k];
                    a1[k] = tanh(sum);
                }
                __syncthreads();
                for (size_t j = tid; j < H; j += BDX) {
                    double sum = h[j];
                    for (size_t k = 0; k < K; ++k)
                        sum += a1[k] * W2[k * H + j];
                    h_out[j] = sum;
                    h_ext[j] = sum;
                    h_ext[H + j] = sum * sum;
                }
            } else {
                for (size_t i = tid; i < H; i += BDX) {
                    h_ext[i] = h[i];
                    h_ext[H + i] = h[i] * h[i];
                }
            }
            __syncthreads();

            // ── Logits ──
            for (size_t a = tid; a < VA; a += BDX) {
                double sum = 0.0;
                for (size_t i = 0; i < H_EXT; ++i)
                    sum += h_ext[i] * W_a[i * VA + a];
                logits[a] = sum;
            }
            __syncthreads();

            // ── Softmax + CE loss (thread 0) ──
            if (tid == 0) {
                double max_val = logits[0];
                for (size_t a = 1; a < VA; ++a)
                    if (logits[a] > max_val) max_val = logits[a];
                double exp_sum = 0.0;
                for (size_t a = 0; a < VA; ++a) {
                    double v = logits[a] - max_val;
                    logits[a] = exp(v < 80.0 ? v : 80.0);
                    exp_sum += logits[a];
                }
                if (exp_sum > 1e-12) {
                    double inv = 1.0 / exp_sum;
                    for (size_t a = 0; a < VA; ++a) logits[a] *= inv;
                }
                double p = logits[ca] > 1e-12 ? logits[ca] : 1e-12;
                total_loss += -log(p);
                total_tokens++;
            }
            __syncthreads();

            // ── d_h_ext for transform backward ──
            if (train_transform) {
                for (size_t i = tid; i < H_EXT; i += BDX) {
                    double d = 0.0;
                    for (size_t a = 0; a < VA; ++a)
                        d += logits[a] * W_a[i * VA + a];
                    d -= W_a[i * VA + ca];
                    d_h_ext[i] = d;
                }
                __syncthreads();
            }

            // ── W_a update: 4-block-aware LR ──
            {
                size_t conv_start = H - conv_dim;
                for (size_t i = tid; i < H_EXT; i += BDX) {
                    double hi = h_ext[i];
                    size_t dim_idx = (i < H) ? i : (i - H);
                    double lr_dim;
                    if (dim_idx < FUSED_BASE) lr_dim = lr_A;
                    else if (dim_idx < FUSED_BASE + flex_dim) lr_dim = lr_B;
                    else if (dim_idx >= conv_start) lr_dim = lr_D;  // convergence
                    else lr_dim = lr_C;  // dimctx

                    for (size_t a = 0; a < VA; ++a)
                        W_a[i * VA + a] -= lr_dim * hi * logits[a];
                    W_a[i * VA + ca] += lr_dim * hi;
                }
            }
            __syncthreads();

            // ── Transform backward + update ──
            if (train_transform) {
                for (size_t i = tid; i < H; i += BDX)
                    d_h_out[i] = d_h_ext[i] + 2.0 * h_out[i] * d_h_ext[H + i];
                __syncthreads();

                for (size_t k = tid; k < K; k += BDX) {
                    double d = 0.0;
                    for (size_t j = 0; j < H; ++j)
                        d += d_h_out[j] * W2[k * H + j];
                    d_a1[k] = d;
                }
                __syncthreads();

                for (size_t k = tid; k < K; k += BDX)
                    for (size_t j = 0; j < H; ++j)
                        W2[k * H + j] -= lr_transform * a1[k] * d_h_out[j];

                for (size_t k = tid; k < K; k += BDX)
                    d_z1[k] = d_a1[k] * (1.0 - a1[k] * a1[k]);
                __syncthreads();

                for (size_t i = tid; i < H; i += BDX)
                    for (size_t k = 0; k < K; ++k)
                        W1[i * K + k] -= lr_transform * h[i] * d_z1[k];

                for (size_t k = tid; k < K; k += BDX)
                    b1[k] -= lr_transform * d_z1[k];
                __syncthreads();
            }

            // ── Hidden state evolution (4-block) ──
            {
                size_t conv_start_idx = H - conv_dim;
                // Block 1: Token fused
                for (size_t i = tid; i < FUSED_BASE; i += BDX)
                    h[i] = h[i] * 0.8 + emb_table[target_tok * FUSED_BASE + i] * 0.2;
                // Block 2: FlexDetail
                for (size_t d = tid; d < flex_dim; d += BDX)
                    h[FUSED_BASE + d] = h[FUSED_BASE + d] * 0.9 + flex_table[target_tok * flex_dim + d] * 0.1;
                // Block 3: DimCtx (stop before conv)
                for (size_t i = FUSED_BASE + flex_dim + tid; i < conv_start_idx; i += BDX)
                    h[i] *= 0.95;
                // Block 4: Convergence
                if (conv_dim > 0) {
                    for (size_t d = tid; d < conv_dim; d += BDX)
                        h[conv_start_idx + d] = h[conv_start_idx + d] * 0.9
                            + conv_table[target_tok * conv_dim + d] * 0.1;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        *loss_out = total_loss;
        *tokens_out = total_tokens;
    }
}

// ── V11 host: monolithic kernel, 1 block, 256 threads ─────────────────────
// Like V10 mono but with flex_dim, flex_table, block-aware LR.
// 1 kernel launch per epoch = zero API overhead per sample.

bool train_sgd_v11_gpu(const TrainingData& data,
                       TrainingWeights& weights,
                       const TrainingConfig& config,
                       TrainingResult& result) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return false;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "[CUDA-V11] VRAM: %.1fMB free / %.1fMB total\n",
            free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0));

    const size_t NS = data.num_samples;
    const size_t H = data.H;
    const size_t H_EXT = data.H_EXT;
    const size_t K = data.K;
    const size_t VA = data.VA;
    const size_t V = data.V;
    const size_t FUSED_BASE = data.FUSED_BASE;
    const size_t FLEX_DIM = data.flex_dim;
    const size_t CONV_DIM = data.conv_dim;

    size_t wa_size = H_EXT * VA;
    size_t w1_size = H * K;
    size_t w2_size = K * H;

    size_t smem_size = (4 * H + 2 * H_EXT + 4 * K + VA) * sizeof(double);

    fprintf(stderr, "[CUDA-V11] Config: H=%zu (base=%zu + flex=%zu + dimctx=%zu + conv=%zu), K=%zu, VA=%zu\n",
            H, FUSED_BASE, FLEX_DIM, H - FUSED_BASE - FLEX_DIM - CONV_DIM, CONV_DIM, K, VA);
    fprintf(stderr, "[CUDA-V11] Monolithic: 1 block, %zu threads, smem=%zu bytes\n",
            V11_TPB, smem_size);

    if (smem_size > 48 * 1024) {
        fprintf(stderr, "[CUDA-V11] Shared memory %zu > 48KB!\n", smem_size);
        return false;
    }

    // Allocate device memory
    uint16_t* d_all_tokens; size_t* d_sample_offsets; size_t* d_sample_lengths;
    double* d_embeddings; size_t* d_compress; double* d_emb_table; double* d_flex_table; double* d_conv_table;
    CUDA_CHECK(cudaMalloc(&d_all_tokens, data.all_tokens.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_sample_offsets, NS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_sample_lengths, NS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_embeddings, data.embeddings.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_compress, V * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_emb_table, data.emb_table.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flex_table, std::max(data.flex_table.size(), size_t(1)) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_conv_table, std::max(data.conv_table.size(), size_t(1)) * sizeof(double)));

    double* d_W_a; double* d_W1; double* d_b1; double* d_W2;
    CUDA_CHECK(cudaMalloc(&d_W_a, wa_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W1, w1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2, w2_size * sizeof(double)));

    double* d_loss;
    unsigned long long* d_tokens;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tokens, sizeof(unsigned long long)));

    // Copy data to GPU (once)
    CUDA_CHECK(cudaMemcpy(d_all_tokens, data.all_tokens.data(), data.all_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_offsets, data.sample_offsets.data(), NS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_lengths, data.sample_lengths.data(), NS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_embeddings, data.embeddings.data(), data.embeddings.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_compress, data.compress.data(), V * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emb_table, data.emb_table.data(), data.emb_table.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (!data.flex_table.empty())
        CUDA_CHECK(cudaMemcpy(d_flex_table, data.flex_table.data(), data.flex_table.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (!data.conv_table.empty())
        CUDA_CHECK(cudaMemcpy(d_conv_table, data.conv_table.data(), data.conv_table.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_a, weights.W_a.data(), wa_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, weights.W1.data(), w1_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, weights.b1.data(), K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, weights.W2.data(), w2_size * sizeof(double), cudaMemcpyHostToDevice));

    fprintf(stderr, "[CUDA-V11] Starting %zu epochs, %zu samples\n", config.num_epochs, NS);

    auto t_start = std::chrono::steady_clock::now();
    result.best_loss = 1e9;

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        bool train_transform = (epoch >= config.transform_warmup);
        double progress = static_cast<double>(epoch) / std::max(config.num_epochs - 1, size_t(1));
        double cos_mult = 0.5 * (1.0 + cos(progress * 3.14159265358979));
        double lr_epoch = config.base_lr * (0.1 + 0.9 * cos_mult);
        double lr_transform = config.lr_transform_base * (0.1 + 0.9 * cos_mult);

        double lr_A = lr_epoch * 1.0;
        double lr_B = lr_epoch * 0.3;
        double lr_C = lr_epoch * 0.1;
        double lr_D = lr_epoch * 0.3;  // convergence

        // 1 kernel launch per epoch — all samples sequential inside
        kernel_epoch_v11<<<1, V11_TPB, smem_size>>>(
            d_all_tokens, d_sample_offsets, d_sample_lengths,
            d_embeddings, d_compress, d_emb_table, d_flex_table, d_conv_table,
            d_W_a, d_W1, d_b1, d_W2,
            NS, V, VA, H, H_EXT, K, FUSED_BASE, FLEX_DIM, CONV_DIM,
            train_transform,
            lr_A, lr_B, lr_C, lr_D, lr_transform,
            d_loss, d_tokens
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        double epoch_loss;
        unsigned long long epoch_tokens;
        CUDA_CHECK(cudaMemcpy(&epoch_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&epoch_tokens, d_tokens, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        double loss = (epoch_tokens > 0) ? epoch_loss / epoch_tokens : 1e9;
        if (loss < result.best_loss) result.best_loss = loss;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            fprintf(stderr, "[CUDA-V11]   Epoch %zu/%zu loss=%.5f lr=%.4f (%zums)\n",
                    epoch + 1, config.num_epochs, loss, lr_epoch, elapsed);
        }
        if (loss < 0.5) break;
    }

    CUDA_CHECK(cudaMemcpy(weights.W_a.data(), d_W_a, wa_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W1.data(), d_W1, w1_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.b1.data(), d_b1, K * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W2.data(), d_W2, w2_size * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_all_tokens); cudaFree(d_sample_offsets); cudaFree(d_sample_lengths);
    cudaFree(d_embeddings); cudaFree(d_compress); cudaFree(d_emb_table); cudaFree(d_flex_table); cudaFree(d_conv_table);
    cudaFree(d_W_a); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2);
    cudaFree(d_loss); cudaFree(d_tokens);

    fprintf(stderr, "[CUDA-V11] Complete, best loss=%.5f\n", result.best_loss);
    return true;
}

// =============================================================================
// V12 Deep KAN Training — 2-layer EfficientKAN + Linear output
// =============================================================================
// Architecture: h[90] → KAN1(90→256,G=8,k=3) → LN → KAN2(256→128,G=5,k=3) → LN → W_a[128×VA]
// One block, 256 threads. Samples processed one at a time.
// Threads cooperate on matmuls within each token step.
// Weight updates are online SGD (no mini-batching).

// Compile-time constants (L1_IN is now runtime via H parameter)
static constexpr int DK_L1_OUT = 256, DK_L1_G = 8, DK_L1_K = 3;
static constexpr int DK_L1_BS = DK_L1_G + DK_L1_K;         // 11
static constexpr int DK_L1_NK = DK_L1_G + 2*DK_L1_K + 1;   // 15

static constexpr int DK_L2_IN = 256, DK_L2_OUT = 128, DK_L2_G = 5, DK_L2_K = 3;
static constexpr int DK_L2_BS = DK_L2_G + DK_L2_K;         // 8
static constexpr int DK_L2_FLAT = DK_L2_IN * DK_L2_BS;     // 2048
static constexpr int DK_L2_NK = DK_L2_G + 2*DK_L2_K + 1;   // 12

static constexpr int DK_L3_IN = 128, DK_L3_OUT = 128, DK_L3_G = 5, DK_L3_K = 3;
static constexpr int DK_L3_BS = DK_L3_G + DK_L3_K;         // 8
static constexpr int DK_L3_FLAT = DK_L3_IN * DK_L3_BS;     // 1024
static constexpr int DK_L3_NK = DK_L3_G + 2*DK_L3_K + 1;   // 12

static constexpr int DK_FEAT = DK_L3_OUT;                   // 128
static constexpr int DK_NT = 256;                            // threads per block

// ── B-spline basis computation (Cox-de Boor iterative) ──
__device__ void dk_bspline(double x, const double* knots,
                           int grid, int order, int bs,
                           double* basis) {
    int n0 = bs + order;
    double B[4][24];  // max: order=3, max n0=14+3=17
    for (int i = 0; i < n0 && i < 24; i++)
        B[0][i] = (x >= knots[i] && x < knots[i+1]) ? 1.0 : 0.0;
    if (n0 > 0 && x >= knots[n0]) B[0][n0-1] = 1.0;

    for (int p = 1; p <= order; p++) {
        int np = n0 - p;
        for (int i = 0; i < np && i < 24; i++) {
            B[p][i] = 0.0;
            double d1 = knots[i+p] - knots[i];
            double d2 = knots[i+p+1] - knots[i+1];
            if (d1 > 1e-10) B[p][i] += (x - knots[i]) / d1 * B[p-1][i];
            if (d2 > 1e-10) B[p][i] += (knots[i+p+1] - x) / d2 * B[p-1][i+1];
        }
    }
    for (int i = 0; i < bs; i++) basis[i] = B[order][i];
}

// ── Parallel sum reduction over 256 threads ──
// Uses volatile to prevent compiler from caching shared memory reads
__device__ double dk_reduce_sum(double val, volatile double* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = val;
    __syncthreads();
    for (int s = DK_NT/2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    return scratch[0];
}

// ── Parallel max reduction ──
__device__ double dk_reduce_max(double val, volatile double* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = val;
    __syncthreads();
    for (int s = DK_NT/2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] = fmax(scratch[tid], scratch[tid + s]);
        __syncthreads();
    }
    return scratch[0];
}

// ── Mini-batch Deep KAN kernel: 1 sample per block, gradient accumulation ──
__global__ void deep_kan_epoch(
    // KAN Layer 1 weights (read-only during mini-batch)
    const double* __restrict__ k1_w,
    const double* __restrict__ k1_res,
    const double* __restrict__ k1_g,
    const double* __restrict__ k1_b,
    const double* __restrict__ k1_kn,
    // KAN Layer 2 weights (read-only)
    const double* __restrict__ k2_w,
    const double* __restrict__ k2_res,
    const double* __restrict__ k2_g,
    const double* __restrict__ k2_b,
    const double* __restrict__ k2_kn,
    // KAN Layer 3 weights (read-only)
    const double* __restrict__ k3_w,
    const double* __restrict__ k3_res,
    const double* __restrict__ k3_g,
    const double* __restrict__ k3_b,
    const double* __restrict__ k3_kn,
    // Output weights (read-only)
    const double* __restrict__ W_a,
    // Gradient accumulators (atomicAdd targets, zeroed before each batch)
    double* __restrict__ g_k1_w, double* __restrict__ g_k1_res,
    double* __restrict__ g_k1_g, double* __restrict__ g_k1_b,
    double* __restrict__ g_k2_w, double* __restrict__ g_k2_res,
    double* __restrict__ g_k2_g, double* __restrict__ g_k2_b,
    double* __restrict__ g_k3_w, double* __restrict__ g_k3_res,
    double* __restrict__ g_k3_g, double* __restrict__ g_k3_b,
    double* __restrict__ g_Wa,
    // Training data (read-only)
    const uint16_t* __restrict__ tokens,
    const size_t* __restrict__ offsets,
    const size_t* __restrict__ lengths,
    const double* __restrict__ embeddings,
    const size_t* __restrict__ compress,
    const double* __restrict__ emb_table,
    const double* __restrict__ flex_table,
    const double* __restrict__ lr_scale,   // [H] per-input LR scale
    const double* __restrict__ conv_table, // [V * conv_dim]
    // Config
    size_t batch_offset, size_t V, size_t VA,
    size_t H_in, size_t FUSED_BASE, size_t flex_dim, size_t conv_dim,
    int train_kan,
    // Per-block workspace (global memory)
    double* __restrict__ ws, size_t ws_stride,
    // Output
    double* __restrict__ loss_out,
    unsigned long long* __restrict__ tok_out
) {
    const int tid = threadIdx.x;
    const size_t s = batch_offset + blockIdx.x;  // 1 sample per block
    const int H = (int)H_in;
    const int H_PAD = ((H + 31) / 32) * 32;
    const int L1_FLAT = H * DK_L1_BS;

    // ── Shared memory layout ──
    extern __shared__ double smem[];
    double* s_h     = smem;
    double* s_l1    = s_h + H_PAD;
    double* s_l2    = s_l1 + 256;
    double* s_feat  = s_l2 + 128;
    double* s_buf   = s_feat + 128;
    double* s_red   = s_buf + 256;
    double* s_dpn   = s_red + 256;

    // ── Per-block workspace (indexed by blockIdx.x) ──
    double* my_ws   = ws + blockIdx.x * ws_stride;
    double* w_b1    = my_ws;
    double* w_ko1   = w_b1 + L1_FLAT;
    double* w_zr1   = w_ko1 + DK_L1_OUT;
    double* w_xh1   = w_zr1 + DK_L1_OUT;
    double* w_b2    = w_xh1 + DK_L1_OUT;
    double* w_ko2   = w_b2 + DK_L2_FLAT;
    double* w_zr2   = w_ko2 + DK_L2_OUT;
    double* w_xh2   = w_zr2 + DK_L2_OUT;
    double* w_b3    = w_xh2 + DK_L2_OUT;
    double* w_ko3   = w_b3 + DK_L3_FLAT;
    double* w_zr3   = w_ko3 + DK_L3_OUT;
    double* w_xh3   = w_zr3 + DK_L3_OUT;
    double* w_is    = w_xh3 + DK_L3_OUT;
    double* w_mean  = w_is + 3;

    // No outer sample loop — blockIdx.x handles one sample
    {
        // Load hidden state
        for (int i = tid; i < H; i += DK_NT)
            s_h[i] = embeddings[s * H + i];
        for (int i = tid + H; i < H_PAD; i += DK_NT) s_h[i] = 0.0;
        __syncthreads();

        size_t off = offsets[s];
        size_t len = lengths[s];

        for (size_t t = 0; t < len; t++) {
            uint16_t tgt = tokens[off + t];
            if (tgt >= V) { __syncthreads(); continue; }
            size_t ca = compress[tgt];
            if (ca >= VA) { __syncthreads(); continue; }

            // ═══════════ LAYER 1 FORWARD ═══════════

            // B-spline basis (90 dims, 11 basis per dim)
            for (int i = tid; i < H; i += DK_NT) {
                double x = fmax(k1_kn[0], fmin(s_h[i], k1_kn[DK_L1_NK-1] - 1e-10));
                dk_bspline(x, k1_kn, DK_L1_G, DK_L1_K, DK_L1_BS, &w_b1[i * DK_L1_BS]);
            }
            __syncthreads();

            // KAN matmul: kan1_out[o] = W1[o,:] · basis1
            for (int o = tid; o < DK_L1_OUT; o += DK_NT) {
                double sum = 0.0;
                for (int j = 0; j < L1_FLAT; j++)
                    sum += k1_w[o * L1_FLAT + j] * w_b1[j];
                w_ko1[o] = sum;
            }
            __syncthreads();

            // Residual: z_res1[o] = h · W_res1[:,o], then SiLU+add
            for (int o = tid; o < DK_L1_OUT; o += DK_NT) {
                double z = 0.0;
                for (int i = 0; i < H; i++)
                    z += s_h[i] * k1_res[i * DK_L1_OUT + o];
                w_zr1[o] = z;
                double sig = 1.0 / (1.0 + exp(-z));
                s_l1[o] = w_ko1[o] + z * sig;  // pre_norm in s_l1
            }
            __syncthreads();

            // LayerNorm 1
            {
                double val = (tid < DK_L1_OUT) ? s_l1[tid] : 0.0;
                double mean = dk_reduce_sum(val, s_red) / DK_L1_OUT;
                if (tid == 0) { volatile double* vm = w_mean; vm[0] = mean; }
                __syncthreads();
                { volatile double* vm = w_mean; mean = vm[0]; }
                double diff = (tid < DK_L1_OUT) ? (s_l1[tid] - mean) : 0.0;
                double var = dk_reduce_sum(diff * diff, s_red) / DK_L1_OUT;
                double inv = rsqrt(var + 1e-5);
                if (tid == 0) { volatile double* vi = w_is; vi[0] = inv; }
                __syncthreads();
                { volatile double* vi = w_is; inv = vi[0]; }
                for (int o = tid; o < DK_L1_OUT; o += DK_NT) {
                    double xh = (s_l1[o] - mean) * inv;
                    w_xh1[o] = xh;
                    s_l1[o] = k1_g[o] * xh + k1_b[o];
                }
            }
            __syncthreads();

            // ═══════════ LAYER 2 FORWARD ═══════════

            // B-spline basis (256 dims, 8 basis per dim)
            for (int i = tid; i < DK_L2_IN; i += DK_NT)  {
                double x = fmax(k2_kn[0], fmin(s_l1[i], k2_kn[DK_L2_NK-1] - 1e-10));
                dk_bspline(x, k2_kn, DK_L2_G, DK_L2_K, DK_L2_BS, &w_b2[i * DK_L2_BS]);
            }
            __syncthreads();

            // KAN matmul: kan2_out[o] = W2[o,:] · basis2
            for (int o = tid; o < DK_L2_OUT; o += DK_NT) {
                double sum = 0.0;
                for (int j = 0; j < DK_L2_FLAT; j++)
                    sum += k2_w[o * DK_L2_FLAT + j] * w_b2[j];
                w_ko2[o] = sum;
            }
            __syncthreads();

            // Residual: z_res2[o] = l1 · W_res2[:,o], then SiLU+add
            for (int o = tid; o < DK_L2_OUT; o += DK_NT) {
                double z = 0.0;
                for (int i = 0; i < DK_L2_IN; i++)
                    z += s_l1[i] * k2_res[i * DK_L2_OUT + o];
                w_zr2[o] = z;
                double sig = 1.0 / (1.0 + exp(-z));
                s_l2[o] = w_ko2[o] + z * sig;
            }
            __syncthreads();

            // LayerNorm 2
            {
                double val = (tid < DK_L2_OUT) ? s_l2[tid] : 0.0;
                double mean = dk_reduce_sum(val, s_red) / DK_L2_OUT;
                if (tid == 0) { volatile double* vm = w_mean; vm[1] = mean; }
                __syncthreads();
                { volatile double* vm = w_mean; mean = vm[1]; }
                double diff = (tid < DK_L2_OUT) ? (s_l2[tid] - mean) : 0.0;
                double var = dk_reduce_sum(diff * diff, s_red) / DK_L2_OUT;
                double inv = rsqrt(var + 1e-5);
                if (tid == 0) { volatile double* vi = w_is; vi[1] = inv; }
                __syncthreads();
                { volatile double* vi = w_is; inv = vi[1]; }
                for (int o = tid; o < DK_L2_OUT; o += DK_NT) {
                    double xh = (s_l2[o] - mean) * inv;
                    w_xh2[o] = xh;
                    s_l2[o] = k2_g[o] * xh + k2_b[o];
                }
            }
            __syncthreads();

            // ═══════════ LAYER 3 FORWARD ═══════════

            // B-spline basis (128 dims, 8 basis per dim)
            for (int i = tid; i < DK_L3_IN; i += DK_NT) {
                double x = fmax(k3_kn[0], fmin(s_l2[i], k3_kn[DK_L3_NK-1] - 1e-10));
                dk_bspline(x, k3_kn, DK_L3_G, DK_L3_K, DK_L3_BS, &w_b3[i * DK_L3_BS]);
            }
            __syncthreads();

            // KAN matmul: kan3_out[o] = W3[o,:] · basis3
            for (int o = tid; o < DK_L3_OUT; o += DK_NT) {
                double sum = 0.0;
                for (int j = 0; j < DK_L3_FLAT; j++)
                    sum += k3_w[o * DK_L3_FLAT + j] * w_b3[j];
                w_ko3[o] = sum;
            }
            __syncthreads();

            // Residual: z_res3[o] = l2 · W_res3[:,o], then SiLU+add
            for (int o = tid; o < DK_L3_OUT; o += DK_NT) {
                double z = 0.0;
                for (int i = 0; i < DK_L3_IN; i++)
                    z += s_l2[i] * k3_res[i * DK_L3_OUT + o];
                w_zr3[o] = z;
                double sig = 1.0 / (1.0 + exp(-z));
                s_feat[o] = w_ko3[o] + z * sig;
            }
            __syncthreads();

            // LayerNorm 3
            {
                double val = (tid < DK_L3_OUT) ? s_feat[tid] : 0.0;
                double mean = dk_reduce_sum(val, s_red) / DK_L3_OUT;
                if (tid == 0) { volatile double* vm = w_mean; vm[2] = mean; }
                __syncthreads();
                { volatile double* vm = w_mean; mean = vm[2]; }
                double diff = (tid < DK_L3_OUT) ? (s_feat[tid] - mean) : 0.0;
                double var = dk_reduce_sum(diff * diff, s_red) / DK_L3_OUT;
                double inv = rsqrt(var + 1e-5);
                if (tid == 0) { volatile double* vi = w_is; vi[2] = inv; }
                __syncthreads();
                { volatile double* vi = w_is; inv = vi[2]; }
                for (int o = tid; o < DK_L3_OUT; o += DK_NT) {
                    double xh = (s_feat[o] - mean) * inv;
                    w_xh3[o] = xh;
                    s_feat[o] = k3_g[o] * xh + k3_b[o];
                }
            }
            __syncthreads();

            // ═══════════ OUTPUT: logits = feat · W_a ═══════════
            for (int a = tid; a < (int)VA; a += DK_NT) {
                double sum = 0.0;
                for (int i = 0; i < DK_FEAT; i++)
                    sum += s_feat[i] * W_a[i * VA + a];
                s_buf[a] = sum;
            }
            for (int i = tid + (int)VA; i < DK_NT; i += DK_NT)
                s_buf[i] = -1e30;
            __syncthreads();

            // Softmax
            double mx = dk_reduce_max((tid < (int)VA) ? s_buf[tid] : -1e30, s_red);
            __syncthreads();
            double ev = (tid < (int)VA) ? exp(fmin(s_buf[tid] - mx, 80.0)) : 0.0;
            double es = dk_reduce_sum(ev, s_red);
            __syncthreads();
            double inv_es = (es > 1e-12) ? 1.0 / es : 0.0;
            if (tid < (int)VA) s_buf[tid] = ev * inv_es;
            __syncthreads();

            // CE loss
            if (tid == 0) {
                double token_loss = -log(fmax(s_buf[ca], 1e-12));
                atomicAdd(loss_out, token_loss);
                atomicAdd(tok_out, 1ULL);
            }

            // ═══════════ BACKWARD ═══════════

            // d_features (before W_a update)
            if (train_kan) {
                for (int i = tid; i < DK_FEAT; i += DK_NT) {
                    double d = 0.0;
                    for (int a = 0; a < (int)VA; a++) {
                        double dl = s_buf[a] - ((size_t)a == ca ? 1.0 : 0.0);
                        d += dl * W_a[i * VA + a];
                    }
                    s_dpn[i] = d;  // d_features stored in s_dpn[0..127]
                }
            }
            __syncthreads();

            // W_a gradient: accumulate fi * dl (lr applied in apply step)
            for (int i = tid; i < DK_FEAT; i += DK_NT) {
                double fi = s_feat[i];
                for (int a = 0; a < (int)VA; a++) {
                    double dl = s_buf[a] - ((size_t)a == ca ? 1.0 : 0.0);
                    atomicAdd(&g_Wa[i * VA + a], fi * dl);
                }
            }
            __syncthreads();

            if (train_kan) {
                // ── Layer 3 backward: LN → KAN weights → Residual ──

                // LN3 backward: d_features → d_pre_norm3
                {
                    double inv; { volatile double* vi = w_is; inv = vi[2]; }
                    double dg = (tid < DK_L3_OUT) ? s_dpn[tid] * k3_g[tid] : 0.0;
                    double c1 = dk_reduce_sum(dg, s_red) / DK_L3_OUT;
                    __syncthreads();
                    double dgx = (tid < DK_L3_OUT) ? dg * w_xh3[tid] : 0.0;
                    double c2 = dk_reduce_sum(dgx, s_red) / DK_L3_OUT;
                    __syncthreads();
                    for (int o = tid; o < DK_L3_OUT; o += DK_NT) {
                        double d_out = s_dpn[o];  // save d_output before overwriting
                        double dx = d_out * k3_g[o];
                        s_dpn[o] = inv * (dx - c1 - w_xh3[o] * c2);  // d_pre_norm
                        atomicAdd(&g_k3_g[o], d_out * w_xh3[o]);  // d_gamma = d_output * x_hat
                        atomicAdd(&g_k3_b[o], d_out);              // d_beta = d_output
                    }
                }
                __syncthreads();

                // KAN3 gradient
                for (int o = tid; o < DK_L3_OUT; o += DK_NT) {
                    double dp = s_dpn[o];
                    for (int j = 0; j < DK_L3_FLAT; j++)
                        atomicAdd(&g_k3_w[o * DK_L3_FLAT + j], dp * w_b3[j]);
                }
                __syncthreads();

                // Residual3 backward
                for (int o = tid; o < DK_L3_OUT; o += DK_NT) {
                    double z = w_zr3[o];
                    double sig = 1.0 / (1.0 + exp(-z));
                    s_dpn[o] = s_dpn[o] * sig * (1.0 + z * (1.0 - sig));
                }
                __syncthreads();

                // d_l2 via residual path + W_res3 gradient
                for (int i = tid; i < DK_L3_IN; i += DK_NT) {
                    double dl = 0.0;
                    double li = s_l2[i];
                    for (int o = 0; o < DK_L3_OUT; o++) {
                        int idx = i * DK_L3_OUT + o;
                        dl += s_dpn[o] * k3_res[idx];
                        atomicAdd(&g_k3_res[idx], li * s_dpn[o]);
                    }
                    s_l2[i] = dl;  // overwrite with d_l2
                }
                __syncthreads();

                // ── Layer 2 backward: LN → KAN weights → Residual ──

                // LN2 backward: d_l2 → d_pre_norm2
                {
                    double inv; { volatile double* vi = w_is; inv = vi[1]; }
                    double dg = (tid < DK_L2_OUT) ? s_l2[tid] * k2_g[tid] : 0.0;
                    double c1 = dk_reduce_sum(dg, s_red) / DK_L2_OUT;
                    __syncthreads();
                    double dgx = (tid < DK_L2_OUT) ? dg * w_xh2[tid] : 0.0;
                    double c2 = dk_reduce_sum(dgx, s_red) / DK_L2_OUT;
                    __syncthreads();
                    for (int o = tid; o < DK_L2_OUT; o += DK_NT) {
                        double d_out = s_l2[o];  // save d_output before overwriting
                        double dx = d_out * k2_g[o];
                        s_dpn[o] = inv * (dx - c1 - w_xh2[o] * c2);  // d_pre_norm
                        atomicAdd(&g_k2_g[o], d_out * w_xh2[o]);  // d_gamma = d_output * x_hat
                        atomicAdd(&g_k2_b[o], d_out);              // d_beta = d_output
                    }
                }
                __syncthreads();

                // KAN2 gradient
                for (int o = tid; o < DK_L2_OUT; o += DK_NT) {
                    double dp = s_dpn[o];
                    for (int j = 0; j < DK_L2_FLAT; j++)
                        atomicAdd(&g_k2_w[o * DK_L2_FLAT + j], dp * w_b2[j]);
                }
                __syncthreads();

                // Residual2 backward
                for (int o = tid; o < DK_L2_OUT; o += DK_NT) {
                    double z = w_zr2[o];
                    double sig = 1.0 / (1.0 + exp(-z));
                    s_dpn[o] = s_dpn[o] * sig * (1.0 + z * (1.0 - sig));
                }
                __syncthreads();

                // d_l1 via residual path + W_res2 gradient
                for (int i = tid; i < DK_L2_IN; i += DK_NT) {
                    double dl = 0.0;
                    double li = s_l1[i];
                    for (int o = 0; o < DK_L2_OUT; o++) {
                        int idx = i * DK_L2_OUT + o;
                        dl += s_dpn[o] * k2_res[idx];
                        atomicAdd(&g_k2_res[idx], li * s_dpn[o]);
                    }
                    s_l1[i] = dl;  // overwrite with d_l1
                }
                __syncthreads();

                // ── Layer 1 backward: LN → KAN weights → Residual ──

                // LN1 backward: d_l1 → d_pre_norm1
                {
                    double inv; { volatile double* vi = w_is; inv = vi[0]; }
                    double dg = (tid < DK_L1_OUT) ? s_l1[tid] * k1_g[tid] : 0.0;
                    double c1 = dk_reduce_sum(dg, s_red) / DK_L1_OUT;
                    __syncthreads();
                    double dgx = (tid < DK_L1_OUT) ? dg * w_xh1[tid] : 0.0;
                    double c2 = dk_reduce_sum(dgx, s_red) / DK_L1_OUT;
                    __syncthreads();
                    for (int o = tid; o < DK_L1_OUT; o += DK_NT) {
                        double d_out = s_l1[o];  // save d_output before overwriting
                        double dx = d_out * k1_g[o];
                        s_dpn[o] = inv * (dx - c1 - w_xh1[o] * c2);  // d_pre_norm
                        atomicAdd(&g_k1_g[o], d_out * w_xh1[o]);  // d_gamma = d_output * x_hat
                        atomicAdd(&g_k1_b[o], d_out);              // d_beta = d_output
                    }
                }
                __syncthreads();

                // KAN1 gradient (includes lr_scale baked in)
                for (int o = tid; o < DK_L1_OUT; o += DK_NT) {
                    double dp = s_dpn[o];
                    for (int j = 0; j < L1_FLAT; j++) {
                        double sc = lr_scale[j / DK_L1_BS];
                        atomicAdd(&g_k1_w[o * L1_FLAT + j], sc * dp * w_b1[j]);
                    }
                }
                __syncthreads();

                // Residual1 backward: d_silu, W_res1 update
                for (int o = tid; o < DK_L1_OUT; o += DK_NT) {
                    double z = w_zr1[o];
                    double sig = 1.0 / (1.0 + exp(-z));
                    s_dpn[o] = s_dpn[o] * sig * (1.0 + z * (1.0 - sig));
                }
                __syncthreads();

                // W_res1 gradient (includes lr_scale baked in)
                for (int i = tid; i < H; i += DK_NT) {
                    double hi = s_h[i];
                    double sc = lr_scale[i];
                    for (int o = 0; o < DK_L1_OUT; o++)
                        atomicAdd(&g_k1_res[i * DK_L1_OUT + o], sc * hi * s_dpn[o]);
                }
                __syncthreads();
            }

            // ═══════════ HIDDEN STATE EVOLUTION (4-block) ═══════════
            {
                int conv_start_idx = H - (int)conv_dim;
                // Block 1: Token fused
                for (int i = tid; i < (int)FUSED_BASE; i += DK_NT) {
                    if (tgt < V) {
                        double te = emb_table[tgt * FUSED_BASE + i];
                        s_h[i] = s_h[i] * 0.8 + te * 0.2;
                    }
                }
                // Block 2: FlexDetail
                if (flex_dim > 0) {
                    for (int i = tid; i < (int)flex_dim; i += DK_NT) {
                        int idx = (int)FUSED_BASE + i;
                        if (idx < H) {
                            double fv = flex_table[tgt * flex_dim + i];
                            s_h[idx] = s_h[idx] * 0.9 + fv * 0.1;
                        }
                    }
                }
                // Block 3: DimCtx (stop before conv)
                for (int i = tid + (int)(FUSED_BASE + flex_dim); i < conv_start_idx; i += DK_NT)
                    s_h[i] *= 0.95;
                // Block 4: Convergence
                if (conv_dim > 0) {
                    for (int d = tid; d < (int)conv_dim; d += DK_NT)
                        s_h[conv_start_idx + d] = s_h[conv_start_idx + d] * 0.9
                            + conv_table[tgt * conv_dim + d] * 0.1;
                }
            }
            __syncthreads();
        }
    }
}

// ── Apply averaged gradients to weights (with per-element clipping) ──
__global__ void dk_apply_grads(double* w, const double* g, double lr_over_B, double max_update, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double upd = g[i] * lr_over_B;
        upd = fmin(upd, max_update);
        upd = fmax(upd, -max_update);
        w[i] -= upd;
    }
}

static void dk_apply(double* d_w, double* d_g, double lr, size_t B, size_t N, double max_upd = 1e30, cudaStream_t stream = 0) {
    if (N == 0) return;
    double lr_over_B = lr / (double)B;
    dk_apply_grads<<<(N + 255) / 256, 256, 0, stream>>>(d_w, d_g, lr_over_B, max_upd, N);
}

// ── Host function: train_deep_kan_gpu (mini-batch SGD) ──
bool train_deep_kan_gpu(const TrainingData& data,
                        DeepKANWeights& weights,
                        const DeepKANConfig& config,
                        TrainingResult& result) {
    int dev;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "[CUDA-DK] GPU: %s, VRAM: %.1f GB\n",
            prop.name, prop.totalGlobalMem / 1073741824.0);

    const size_t NS = data.num_samples;
    const size_t V = data.V;
    const size_t VA = data.VA;
    const size_t H = data.H;
    const size_t FB = data.FUSED_BASE;
    const size_t FD = data.flex_dim;
    const size_t CD = data.conv_dim;
    const size_t L1_FLAT = H * DK_L1_BS;
    const size_t H_PAD = ((H + 31) / 32) * 32;
    const size_t BATCH_SIZE = 64;  // samples per mini-batch

    // ── Weight sizes (for gradient buffers) ──
    size_t sz_k1w = weights.k1_weights.size(), sz_k1r = weights.k1_residual.size();
    size_t sz_k2w = weights.k2_weights.size(), sz_k2r = weights.k2_residual.size();
    size_t sz_k3w = weights.k3_weights.size(), sz_k3r = weights.k3_residual.size();
    size_t sz_Wa  = weights.W_a.size();

    // ── Allocate + upload weights ──
    double *d_k1w, *d_k1r, *d_k1g, *d_k1b, *d_k1kn;
    CUDA_CHECK(cudaMalloc(&d_k1w, sz_k1w * 8)); CUDA_CHECK(cudaMalloc(&d_k1r, sz_k1r * 8));
    CUDA_CHECK(cudaMalloc(&d_k1g, DK_L1_OUT * 8)); CUDA_CHECK(cudaMalloc(&d_k1b, DK_L1_OUT * 8));
    CUDA_CHECK(cudaMalloc(&d_k1kn, weights.k1_knots.size() * 8));
    CUDA_CHECK(cudaMemcpy(d_k1w, weights.k1_weights.data(), sz_k1w * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k1r, weights.k1_residual.data(), sz_k1r * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k1g, weights.k1_gamma.data(), DK_L1_OUT * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k1b, weights.k1_beta.data(), DK_L1_OUT * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k1kn, weights.k1_knots.data(), weights.k1_knots.size() * 8, cudaMemcpyHostToDevice));

    double *d_k2w, *d_k2r, *d_k2g, *d_k2b, *d_k2kn;
    CUDA_CHECK(cudaMalloc(&d_k2w, sz_k2w * 8)); CUDA_CHECK(cudaMalloc(&d_k2r, sz_k2r * 8));
    CUDA_CHECK(cudaMalloc(&d_k2g, DK_L2_OUT * 8)); CUDA_CHECK(cudaMalloc(&d_k2b, DK_L2_OUT * 8));
    CUDA_CHECK(cudaMalloc(&d_k2kn, weights.k2_knots.size() * 8));
    CUDA_CHECK(cudaMemcpy(d_k2w, weights.k2_weights.data(), sz_k2w * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k2r, weights.k2_residual.data(), sz_k2r * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k2g, weights.k2_gamma.data(), DK_L2_OUT * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k2b, weights.k2_beta.data(), DK_L2_OUT * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k2kn, weights.k2_knots.data(), weights.k2_knots.size() * 8, cudaMemcpyHostToDevice));

    double *d_k3w, *d_k3r, *d_k3g, *d_k3b, *d_k3kn;
    CUDA_CHECK(cudaMalloc(&d_k3w, sz_k3w * 8)); CUDA_CHECK(cudaMalloc(&d_k3r, sz_k3r * 8));
    CUDA_CHECK(cudaMalloc(&d_k3g, DK_L3_OUT * 8)); CUDA_CHECK(cudaMalloc(&d_k3b, DK_L3_OUT * 8));
    CUDA_CHECK(cudaMalloc(&d_k3kn, weights.k3_knots.size() * 8));
    CUDA_CHECK(cudaMemcpy(d_k3w, weights.k3_weights.data(), sz_k3w * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k3r, weights.k3_residual.data(), sz_k3r * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k3g, weights.k3_gamma.data(), DK_L3_OUT * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k3b, weights.k3_beta.data(), DK_L3_OUT * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k3kn, weights.k3_knots.data(), weights.k3_knots.size() * 8, cudaMemcpyHostToDevice));

    double* d_Wa;
    CUDA_CHECK(cudaMalloc(&d_Wa, sz_Wa * 8));
    CUDA_CHECK(cudaMemcpy(d_Wa, weights.W_a.data(), sz_Wa * 8, cudaMemcpyHostToDevice));

    // ── Gradient accumulators (same sizes as weights) ──
    double *g_k1w, *g_k1r, *g_k1g, *g_k1b;
    double *g_k2w, *g_k2r, *g_k2g, *g_k2b;
    double *g_k3w, *g_k3r, *g_k3g, *g_k3b;
    double *g_Wa;
    CUDA_CHECK(cudaMalloc(&g_k1w, sz_k1w * 8)); CUDA_CHECK(cudaMalloc(&g_k1r, sz_k1r * 8));
    CUDA_CHECK(cudaMalloc(&g_k1g, DK_L1_OUT * 8)); CUDA_CHECK(cudaMalloc(&g_k1b, DK_L1_OUT * 8));
    CUDA_CHECK(cudaMalloc(&g_k2w, sz_k2w * 8)); CUDA_CHECK(cudaMalloc(&g_k2r, sz_k2r * 8));
    CUDA_CHECK(cudaMalloc(&g_k2g, DK_L2_OUT * 8)); CUDA_CHECK(cudaMalloc(&g_k2b, DK_L2_OUT * 8));
    CUDA_CHECK(cudaMalloc(&g_k3w, sz_k3w * 8)); CUDA_CHECK(cudaMalloc(&g_k3r, sz_k3r * 8));
    CUDA_CHECK(cudaMalloc(&g_k3g, DK_L3_OUT * 8)); CUDA_CHECK(cudaMalloc(&g_k3b, DK_L3_OUT * 8));
    CUDA_CHECK(cudaMalloc(&g_Wa, sz_Wa * 8));

    // Total gradient memory: ~6.5MB
    size_t grad_total = (sz_k1w + sz_k1r + DK_L1_OUT*2 + sz_k2w + sz_k2r + DK_L2_OUT*2 +
                         sz_k3w + sz_k3r + DK_L3_OUT*2 + sz_Wa) * 8;

    // ── Training data ──
    uint16_t* d_tokens;
    size_t *d_offsets, *d_lengths, *d_compress;
    double *d_embeddings, *d_emb_table, *d_flex_table, *d_conv_table, *d_lr_scale;
    CUDA_CHECK(cudaMalloc(&d_tokens, data.all_tokens.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_offsets, NS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_lengths, NS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_embeddings, data.embeddings.size() * 8));
    CUDA_CHECK(cudaMalloc(&d_compress, V * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_emb_table, data.emb_table.size() * 8));
    CUDA_CHECK(cudaMalloc(&d_flex_table, std::max(data.flex_table.size(), (size_t)1) * 8));
    CUDA_CHECK(cudaMalloc(&d_conv_table, std::max(data.conv_table.size(), (size_t)1) * 8));
    CUDA_CHECK(cudaMalloc(&d_lr_scale, config.lr_scale.size() * 8));
    CUDA_CHECK(cudaMemcpy(d_tokens, data.all_tokens.data(), data.all_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, data.sample_offsets.data(), NS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, data.sample_lengths.data(), NS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_embeddings, data.embeddings.data(), data.embeddings.size() * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_compress, data.compress.data(), V * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emb_table, data.emb_table.data(), data.emb_table.size() * 8, cudaMemcpyHostToDevice));
    if (!data.flex_table.empty())
        CUDA_CHECK(cudaMemcpy(d_flex_table, data.flex_table.data(), data.flex_table.size() * 8, cudaMemcpyHostToDevice));
    if (!data.conv_table.empty())
        CUDA_CHECK(cudaMemcpy(d_conv_table, data.conv_table.data(), data.conv_table.size() * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lr_scale, config.lr_scale.data(), config.lr_scale.size() * 8, cudaMemcpyHostToDevice));

    // ── Per-block workspace (BATCH_SIZE blocks) ──
    size_t ws_per_block = L1_FLAT + DK_L1_OUT*3 + DK_L2_FLAT + DK_L2_OUT*3 + DK_L3_FLAT + DK_L3_OUT*3 + 6;
    double* d_ws;
    CUDA_CHECK(cudaMalloc(&d_ws, BATCH_SIZE * ws_per_block * 8));

    double* d_loss;
    unsigned long long* d_tok;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tok, sizeof(unsigned long long)));

    size_t smem_size = (H_PAD + 256 + 128 + 128 + 256 + 256 + 256) * 8;
    size_t num_batches = (NS + BATCH_SIZE - 1) / BATCH_SIZE;

    fprintf(stderr, "[CUDA-DK] H=%zu, smem=%zu, batch=%zu, %zu batches/epoch\n",
            H, smem_size, BATCH_SIZE, num_batches);
    fprintf(stderr, "[CUDA-DK] Grad buffers: %.1f MB, workspace: %.1f MB\n",
            grad_total / 1048576.0, BATCH_SIZE * ws_per_block * 8 / 1048576.0);
    fprintf(stderr, "[CUDA-DK] Uploaded: %zu samples, VA=%zu, H=%zu\n", NS, VA, H);

    // ── Training loop ──
    auto t_start = std::chrono::steady_clock::now();
    result.best_loss = 1e9;

    for (size_t epoch = 0; epoch < config.num_epochs; epoch++) {
        bool do_kan = (epoch >= config.warmup_epochs);
        double progress = (double)epoch / fmax((double)(config.num_epochs - 1), 1.0);
        double cos_mult = 0.5 * (1.0 + cos(progress * 3.14159265358979));
        // Scale LR by batch size to match per-sample SGD total update per epoch
        double lr_out_ep = config.lr_output * BATCH_SIZE * (0.1 + 0.9 * cos_mult);
        double lr_kan_ep = config.lr_kan * BATCH_SIZE * (0.1 + 0.9 * cos_mult);

        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_tok, 0, sizeof(unsigned long long)));

        // ── Mini-batch loop ──
        for (size_t batch_start = 0; batch_start < NS; batch_start += BATCH_SIZE) {
            size_t B = std::min(BATCH_SIZE, NS - batch_start);

            // Zero gradient accumulators
            CUDA_CHECK(cudaMemset(g_k1w, 0, sz_k1w * 8)); CUDA_CHECK(cudaMemset(g_k1r, 0, sz_k1r * 8));
            CUDA_CHECK(cudaMemset(g_k1g, 0, DK_L1_OUT * 8)); CUDA_CHECK(cudaMemset(g_k1b, 0, DK_L1_OUT * 8));
            CUDA_CHECK(cudaMemset(g_k2w, 0, sz_k2w * 8)); CUDA_CHECK(cudaMemset(g_k2r, 0, sz_k2r * 8));
            CUDA_CHECK(cudaMemset(g_k2g, 0, DK_L2_OUT * 8)); CUDA_CHECK(cudaMemset(g_k2b, 0, DK_L2_OUT * 8));
            CUDA_CHECK(cudaMemset(g_k3w, 0, sz_k3w * 8)); CUDA_CHECK(cudaMemset(g_k3r, 0, sz_k3r * 8));
            CUDA_CHECK(cudaMemset(g_k3g, 0, DK_L3_OUT * 8)); CUDA_CHECK(cudaMemset(g_k3b, 0, DK_L3_OUT * 8));
            CUDA_CHECK(cudaMemset(g_Wa, 0, sz_Wa * 8));

            // Forward + gradient accumulation (B blocks, 1 sample per block)
            deep_kan_epoch<<<B, DK_NT, smem_size>>>(
                d_k1w, d_k1r, d_k1g, d_k1b, d_k1kn,
                d_k2w, d_k2r, d_k2g, d_k2b, d_k2kn,
                d_k3w, d_k3r, d_k3g, d_k3b, d_k3kn,
                d_Wa,
                g_k1w, g_k1r, g_k1g, g_k1b,
                g_k2w, g_k2r, g_k2g, g_k2b,
                g_k3w, g_k3r, g_k3g, g_k3b,
                g_Wa,
                d_tokens, d_offsets, d_lengths, d_embeddings,
                d_compress, d_emb_table, d_flex_table, d_lr_scale, d_conv_table,
                batch_start, V, VA, H, FB, FD, CD,
                do_kan ? 1 : 0,
                d_ws, ws_per_block,
                d_loss, d_tok
            );

            // Apply averaged gradients: w -= (lr / B) * g
            dk_apply(d_Wa,  g_Wa,  lr_out_ep, B, sz_Wa);
            if (do_kan) {
                dk_apply(d_k1w, g_k1w, lr_kan_ep, B, sz_k1w);
                dk_apply(d_k1r, g_k1r, lr_kan_ep, B, sz_k1r);
                dk_apply(d_k1g, g_k1g, lr_kan_ep, B, DK_L1_OUT);
                dk_apply(d_k1b, g_k1b, lr_kan_ep, B, DK_L1_OUT);
                dk_apply(d_k2w, g_k2w, lr_kan_ep, B, sz_k2w);
                dk_apply(d_k2r, g_k2r, lr_kan_ep, B, sz_k2r);
                dk_apply(d_k2g, g_k2g, lr_kan_ep, B, DK_L2_OUT);
                dk_apply(d_k2b, g_k2b, lr_kan_ep, B, DK_L2_OUT);
                dk_apply(d_k3w, g_k3w, lr_kan_ep, B, sz_k3w);
                dk_apply(d_k3r, g_k3r, lr_kan_ep, B, sz_k3r);
                dk_apply(d_k3g, g_k3g, lr_kan_ep, B, DK_L3_OUT);
                dk_apply(d_k3b, g_k3b, lr_kan_ep, B, DK_L3_OUT);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        double ep_loss; unsigned long long ep_tok;
        CUDA_CHECK(cudaMemcpy(&ep_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&ep_tok, d_tok, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double loss = (ep_tok > 0) ? ep_loss / ep_tok : 1e9;
        if (loss < result.best_loss) result.best_loss = loss;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            fprintf(stderr, "[CUDA-DK]   Epoch %zu/%zu loss=%.5f lr_out=%.3f lr_kan=%.5f [%zu batches] (%zums)\n",
                    epoch + 1, config.num_epochs, loss, lr_out_ep, lr_kan_ep, (NS + BATCH_SIZE - 1) / BATCH_SIZE, elapsed);
        }
        if (loss < 0.5) break;
    }

    // ── Copy weights back ──
    CUDA_CHECK(cudaMemcpy(weights.k1_weights.data(), d_k1w, sz_k1w * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k1_residual.data(), d_k1r, sz_k1r * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k1_gamma.data(), d_k1g, DK_L1_OUT * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k1_beta.data(), d_k1b, DK_L1_OUT * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k2_weights.data(), d_k2w, sz_k2w * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k2_residual.data(), d_k2r, sz_k2r * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k2_gamma.data(), d_k2g, DK_L2_OUT * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k2_beta.data(), d_k2b, DK_L2_OUT * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k3_weights.data(), d_k3w, sz_k3w * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k3_residual.data(), d_k3r, sz_k3r * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k3_gamma.data(), d_k3g, DK_L3_OUT * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.k3_beta.data(), d_k3b, DK_L3_OUT * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W_a.data(), d_Wa, sz_Wa * 8, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_k1w); cudaFree(d_k1r); cudaFree(d_k1g); cudaFree(d_k1b); cudaFree(d_k1kn);
    cudaFree(d_k2w); cudaFree(d_k2r); cudaFree(d_k2g); cudaFree(d_k2b); cudaFree(d_k2kn);
    cudaFree(d_k3w); cudaFree(d_k3r); cudaFree(d_k3g); cudaFree(d_k3b); cudaFree(d_k3kn);
    cudaFree(d_Wa);
    cudaFree(g_k1w); cudaFree(g_k1r); cudaFree(g_k1g); cudaFree(g_k1b);
    cudaFree(g_k2w); cudaFree(g_k2r); cudaFree(g_k2g); cudaFree(g_k2b);
    cudaFree(g_k3w); cudaFree(g_k3r); cudaFree(g_k3g); cudaFree(g_k3b);
    cudaFree(g_Wa);
    cudaFree(d_tokens); cudaFree(d_offsets); cudaFree(d_lengths);
    cudaFree(d_embeddings); cudaFree(d_compress); cudaFree(d_emb_table);
    cudaFree(d_flex_table); cudaFree(d_conv_table); cudaFree(d_lr_scale);
    cudaFree(d_ws); cudaFree(d_loss); cudaFree(d_tok);

    fprintf(stderr, "[CUDA-DK] Complete, best loss=%.5f\n", result.best_loss);
    return true;
}

} // namespace cuda
} // namespace brain19

#endif // USE_CUDA

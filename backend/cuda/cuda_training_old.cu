// cuda/cuda_training.cu — Full SGD training loop on GPU
// Per-sample W copies for online SGD (matches CPU behavior exactly).
// All data stays in VRAM for all epochs.
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
        fprintf(stderr, "[CUDA-Train] %s failed: %s\n", #call, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

namespace brain19 {
namespace cuda {

// ─── Kernel: Copy shared weights into per-sample buffers ────────────────────
__global__ void kernel_broadcast_weights(
    const double* W_a,      // [H_EXT * VA]
    const double* W1,       // [H * K]
    const double* b1,       // [K]
    const double* W2,       // [K * H]
    const double* b2,       // [H]
    double* local_W_a,      // [batch_size * H_EXT * VA]
    double* local_W1,       // [batch_size * H * K]
    double* local_b1,       // [batch_size * K]
    double* local_W2,       // [batch_size * K * H]
    double* local_b2,       // [batch_size * H]
    size_t batch_size,
    size_t wa_size, size_t w1_size, size_t b1_size, size_t w2_size, size_t b2_size
) {
    size_t sid = blockIdx.x;
    if (sid >= batch_size) return;

    // Each thread copies a portion
    for (size_t i = threadIdx.x; i < wa_size; i += blockDim.x)
        local_W_a[sid * wa_size + i] = W_a[i];
    for (size_t i = threadIdx.x; i < w1_size; i += blockDim.x)
        local_W1[sid * w1_size + i] = W1[i];
    for (size_t i = threadIdx.x; i < b1_size; i += blockDim.x)
        local_b1[sid * b1_size + i] = b1[i];
    for (size_t i = threadIdx.x; i < w2_size; i += blockDim.x)
        local_W2[sid * w2_size + i] = W2[i];
    for (size_t i = threadIdx.x; i < b2_size; i += blockDim.x)
        local_b2[sid * b2_size + i] = b2[i];
}

// ─── Kernel: Online SGD per sample (per-token W updates, own W copy) ────────
__global__ void kernel_online_sgd(
    const uint16_t* all_tokens,
    const size_t* sample_offsets,
    const size_t* sample_lengths,
    const double* embeddings,
    const size_t* compress,
    const double* emb_table,
    // Per-sample weight copies (read-write)
    double* local_W_a,
    double* local_W1,
    double* local_b1,
    double* local_W2,
    double* local_b2,
    // Dimensions
    size_t batch_start, size_t batch_size, size_t total_samples,
    size_t V, size_t VA,
    size_t H, size_t H_EXT,
    size_t K, size_t FUSED_BASE,
    double lr_epoch, double lr_transform_epoch,
    bool train_transform,
    double* d_loss, size_t* d_tokens
) {
    size_t local_id = blockIdx.x;
    size_t sid = batch_start + local_id;
    if (sid >= total_samples || local_id >= batch_size) return;
    if (threadIdx.x != 0) return;

    size_t tok_start = sample_offsets[sid];
    size_t tok_len = sample_lengths[sid];

    // Pointers to this sample's weight copies
    size_t wa_size = H_EXT * VA;
    size_t w1_size = H * K;
    size_t w2_size = K * H;
    double* my_W_a = local_W_a + local_id * wa_size;
    double* my_W1  = local_W1  + local_id * w1_size;
    double* my_b1  = local_b1  + local_id * K;
    double* my_W2  = local_W2  + local_id * w2_size;
    double* my_b2  = local_b2  + local_id * H;

    // Stack arrays
    double h[128], h_out[128], h_ext[256];
    double z1[64], a1[64];
    double logits[512], probs[512];
    double d_h_ext[256], d_h_out[128], d_a1[64], d_z1[64];

    for (size_t i = 0; i < H; ++i) h[i] = embeddings[sid * H + i];

    double sample_loss = 0.0;
    size_t sample_tokens = 0;

    for (size_t t = 0; t < tok_len; ++t) {
        uint16_t target_tok = all_tokens[tok_start + t];
        if (target_tok >= V) continue;
        size_t ca = compress[target_tok];
        if (ca >= VA) continue;

        // Forward transform
        for (size_t k = 0; k < K; ++k) {
            double sum = my_b1[k];
            for (size_t i = 0; i < H; ++i) sum += h[i] * my_W1[i * K + k];
            z1[k] = sum;
            a1[k] = tanh(sum);
        }
        for (size_t j = 0; j < H; ++j) {
            double sum = h[j] + my_b2[j];
            for (size_t k = 0; k < K; ++k) sum += a1[k] * my_W2[k * H + j];
            h_out[j] = sum;
        }

        for (size_t i = 0; i < H; ++i) {
            h_ext[i] = h_out[i];
            h_ext[H + i] = h_out[i] * h_out[i];
        }

        for (size_t a = 0; a < VA; ++a) {
            double sum = 0.0;
            for (size_t i = 0; i < H_EXT; ++i) sum += h_ext[i] * my_W_a[i * VA + a];
            logits[a] = sum;
        }

        double max_val = logits[0];
        for (size_t a = 1; a < VA; ++a) if (logits[a] > max_val) max_val = logits[a];
        double exp_sum = 0.0;
        for (size_t a = 0; a < VA; ++a) {
            double v = logits[a] - max_val;
            probs[a] = exp(v < 80.0 ? v : 80.0);
            exp_sum += probs[a];
        }
        if (exp_sum > 1e-12) {
            double inv = 1.0 / exp_sum;
            for (size_t a = 0; a < VA; ++a) probs[a] *= inv;
        }

        double p = probs[ca] > 1e-12 ? probs[ca] : 1e-12;
        sample_loss += -log(p);
        sample_tokens++;

        // d_h_ext BEFORE W update
        if (train_transform) {
            for (size_t i = 0; i < H_EXT; ++i) {
                double d = 0.0;
                for (size_t a = 0; a < VA; ++a) d += probs[a] * my_W_a[i * VA + a];
                d -= my_W_a[i * VA + ca];
                d_h_ext[i] = d;
            }
        }

        // Per-token W update (own copy, no race condition)
        for (size_t i = 0; i < H_EXT; ++i) {
            double hi = h_ext[i];
            for (size_t a = 0; a < VA; ++a) {
                my_W_a[i * VA + a] -= lr_epoch * hi * probs[a];
            }
            my_W_a[i * VA + ca] += lr_epoch * hi;
        }

        // Backprop transform
        if (train_transform) {
            for (size_t i = 0; i < H; ++i)
                d_h_out[i] = d_h_ext[i] + 2.0 * h_out[i] * d_h_ext[H + i];
            for (size_t k = 0; k < K; ++k) {
                double d = 0.0;
                for (size_t j = 0; j < H; ++j) d += d_h_out[j] * my_W2[k * H + j];
                d_a1[k] = d;
            }
            for (size_t k = 0; k < K; ++k)
                for (size_t j = 0; j < H; ++j)
                    my_W2[k * H + j] -= lr_transform_epoch * a1[k] * d_h_out[j];
            for (size_t k = 0; k < K; ++k)
                d_z1[k] = d_a1[k] * (1.0 - a1[k] * a1[k]);
            for (size_t i = 0; i < H; ++i)
                for (size_t k = 0; k < K; ++k)
                    my_W1[i * K + k] -= lr_transform_epoch * h[i] * d_z1[k];
            for (size_t k = 0; k < K; ++k)
                my_b1[k] -= lr_transform_epoch * d_z1[k];
        }

        // Evolve hidden state
        if (target_tok < V) {
            for (size_t i = 0; i < FUSED_BASE; ++i)
                h[i] = h[i] * 0.8 + emb_table[target_tok * FUSED_BASE + i] * 0.2;
        }
        for (size_t i = FUSED_BASE; i < H; ++i) h[i] *= 0.95;
    }

    d_loss[local_id] = sample_loss;
    d_tokens[local_id] = sample_tokens;
}

// ─── Kernel: Average per-sample weights back into shared weights ────────────
__global__ void kernel_average_weights(
    double* W_out,              // shared weight [size]
    const double* local_W,      // [batch_size * size]
    size_t size,
    size_t batch_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    double sum = 0.0;
    for (size_t b = 0; b < batch_size; ++b) {
        sum += local_W[b * size + idx];
    }
    W_out[idx] = sum / batch_size;
}

// ─── Host: full training loop ───────────────────────────────────────────────

bool train_sgd_gpu(const TrainingData& data,
                   TrainingWeights& weights,
                   const TrainingConfig& config,
                   TrainingResult& result) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return false;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "[CUDA-Train] VRAM: %.1fMB free / %.1fMB total\n",
            free_mem/(1024.0*1024.0), total_mem/(1024.0*1024.0));

    const size_t NS = data.num_samples;
    const size_t H = data.H;
    const size_t H_EXT = data.H_EXT;
    const size_t K = data.K;
    const size_t VA = data.VA;
    const size_t V = data.V;
    const size_t FUSED_BASE = data.FUSED_BASE;

    if (H > 128 || H_EXT > 256 || K > 64 || VA > 512) {
        fprintf(stderr, "[CUDA-Train] Dims too large\n");
        return false;
    }

    // Size of weight arrays
    size_t wa_size = H_EXT * VA;
    size_t w1_size = H * K;
    size_t b1_size = K;
    size_t w2_size = K * H;
    size_t b2_size = H;
    size_t weights_per_sample = (wa_size + w1_size + b1_size + w2_size + b2_size) * sizeof(double);

    // Determine batch size: fit local weight copies in ~4GB
    // Fixed batch size: small enough for good convergence (more sync points),
    // large enough for GPU parallelism. 128 = 136 batches/epoch.
    size_t BATCH_SIZE = std::min(NS, (size_t)1024);
    // Verify VRAM fits
    size_t needed = BATCH_SIZE * weights_per_sample;
    if (needed > free_mem * 3 / 4) {
        BATCH_SIZE = std::min(NS, (free_mem * 3 / 4) / weights_per_sample);
        BATCH_SIZE = std::max(BATCH_SIZE, (size_t)1);
    }

    fprintf(stderr, "[CUDA-Train] Batch: %zu samples (%.1fKB weights/sample, %.1fMB total)\n",
            BATCH_SIZE, weights_per_sample/1024.0, (BATCH_SIZE * weights_per_sample)/(1024.0*1024.0));

    // ── Allocate device memory ──
    uint16_t* d_all_tokens; size_t* d_sample_offsets; size_t* d_sample_lengths;
    double* d_embeddings; size_t* d_compress; double* d_emb_table;
    double* d_W_a; double* d_W1; double* d_b1; double* d_W2; double* d_b2;
    double* d_local_W_a; double* d_local_W1; double* d_local_b1; double* d_local_W2; double* d_local_b2;
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
    CUDA_CHECK(cudaMalloc(&d_b2, b2_size * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_local_W_a, BATCH_SIZE * wa_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_W1, BATCH_SIZE * w1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_b1, BATCH_SIZE * b1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_W2, BATCH_SIZE * w2_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_b2, BATCH_SIZE * b2_size * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tokens, BATCH_SIZE * sizeof(size_t)));

    // ── Copy data (once) ──
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
    CUDA_CHECK(cudaMemcpy(d_b2, weights.b2.data(), b2_size * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> h_loss(BATCH_SIZE);
    std::vector<size_t> h_tokens(BATCH_SIZE);
    auto t_start = std::chrono::steady_clock::now();
    result.best_loss = 1e9;
    const size_t THREADS = 256;

    fprintf(stderr, "[CUDA-Train] Starting %zu epochs, %zu samples, batch=%zu\n",
            config.num_epochs, NS, BATCH_SIZE);

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        bool train_transform = (epoch >= config.transform_warmup);
        double progress = static_cast<double>(epoch) / std::max(config.num_epochs - 1, size_t(1));
        double cos_mult = 0.5 * (1.0 + cos(progress * 3.14159265358979));
        double lr_epoch = config.base_lr * (0.1 + 0.9 * cos_mult);
        double lr_transform_epoch = config.lr_transform_base * (0.1 + 0.9 * cos_mult);

        // No LR scaling — keep same LR as CPU, rely on smaller batch for convergence

        double epoch_loss = 0.0;
        size_t epoch_tokens = 0;

        for (size_t batch_start = 0; batch_start < NS; batch_start += BATCH_SIZE) {
            size_t batch_end = std::min(batch_start + BATCH_SIZE, NS);
            size_t cur_batch = batch_end - batch_start;

            // 1. Broadcast shared weights to per-sample copies
            kernel_broadcast_weights<<<cur_batch, THREADS>>>(
                d_W_a, d_W1, d_b1, d_W2, d_b2,
                d_local_W_a, d_local_W1, d_local_b1, d_local_W2, d_local_b2,
                cur_batch, wa_size, w1_size, b1_size, w2_size, b2_size
            );

            // 2. Online SGD per sample (each has own W copy)
            kernel_online_sgd<<<cur_batch, 1>>>(
                d_all_tokens, d_sample_offsets, d_sample_lengths,
                d_embeddings, d_compress, d_emb_table,
                d_local_W_a, d_local_W1, d_local_b1, d_local_W2, d_local_b2,
                batch_start, cur_batch, NS,
                V, VA, H, H_EXT, K, FUSED_BASE,
                lr_epoch, lr_transform_epoch, train_transform,
                d_loss, d_tokens
            );

            // 3. Average per-sample weights back to shared
            kernel_average_weights<<<(wa_size + THREADS - 1) / THREADS, THREADS>>>(
                d_W_a, d_local_W_a, wa_size, cur_batch);
            if (train_transform) {
                kernel_average_weights<<<(w1_size + THREADS - 1) / THREADS, THREADS>>>(
                    d_W1, d_local_W1, w1_size, cur_batch);
                kernel_average_weights<<<(b1_size + THREADS - 1) / THREADS, THREADS>>>(
                    d_b1, d_local_b1, b1_size, cur_batch);
                kernel_average_weights<<<(w2_size + THREADS - 1) / THREADS, THREADS>>>(
                    d_W2, d_local_W2, w2_size, cur_batch);
            }

            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_loss.data(), d_loss, cur_batch * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_tokens.data(), d_tokens, cur_batch * sizeof(size_t), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < cur_batch; ++i) {
                epoch_loss += h_loss[i];
                epoch_tokens += h_tokens[i];
            }
        }

        double loss = (epoch_tokens > 0) ? epoch_loss / epoch_tokens : 1e9;
        if (loss < result.best_loss) result.best_loss = loss;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            fprintf(stderr, "[CUDA-Train]   Epoch %zu/%zu loss=%.5f lr=%.4f (%zums)\n",
                    epoch + 1, config.num_epochs, loss, lr_epoch, elapsed);
        }
        if (loss < 0.5) break;
    }

    CUDA_CHECK(cudaMemcpy(weights.W_a.data(), d_W_a, wa_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W1.data(), d_W1, w1_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.b1.data(), d_b1, b1_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.W2.data(), d_W2, w2_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.b2.data(), d_b2, b2_size * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_all_tokens); cudaFree(d_sample_offsets); cudaFree(d_sample_lengths);
    cudaFree(d_embeddings); cudaFree(d_compress); cudaFree(d_emb_table);
    cudaFree(d_W_a); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_local_W_a); cudaFree(d_local_W1); cudaFree(d_local_b1);
    cudaFree(d_local_W2); cudaFree(d_local_b2);
    cudaFree(d_loss); cudaFree(d_tokens);

    fprintf(stderr, "[CUDA-Train] Complete, best loss=%.5f\n", result.best_loss);
    return true;
}

} // namespace cuda
} // namespace brain19

#endif // USE_CUDA

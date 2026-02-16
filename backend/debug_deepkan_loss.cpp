// Quick diagnostic: compute initial Deep KAN forward pass loss on CPU
// to compare with CUDA kernel loss of ~22
#include "language/deep_kan.hpp"
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>

using namespace brain19;

int main() {
    const size_t H = 122;      // extended fused dim
    const size_t FEAT = 128;   // Deep KAN output
    const size_t VA = 104;     // active vocab

    // Create same DeepKAN as train_v12
    DeepKAN kan({H, 256, 128, FEAT}, {8, 5, 5}, 3);
    printf("Deep KAN: %zu -> 256 -> 128 -> %zu, %zu params\n",
           H, FEAT, kan.num_params());

    // Random W_a with same seed as train_v12
    std::vector<double> W_a(FEAT * VA);
    {
        std::mt19937 rng(42);
        double scale = std::sqrt(6.0 / (double)(FEAT + VA));
        std::uniform_real_distribution<double> dist(-scale, scale);
        for (auto& w : W_a) w = dist(rng);
    }

    // Create some synthetic embeddings (values similar to real data: ~0.1-0.2 magnitude)
    std::mt19937 rng(123);
    std::normal_distribution<double> emb_dist(0.0, 0.15);

    double total_loss = 0.0;
    size_t total_tokens = 0;
    const int N_SAMPLES = 100;
    const int TOKENS_PER_SAMPLE = 5;

    for (int s = 0; s < N_SAMPLES; s++) {
        // Random embedding
        std::vector<double> h(H);
        for (auto& v : h) v = emb_dist(rng);

        for (int t = 0; t < TOKENS_PER_SAMPLE; t++) {
            // Forward through DeepKAN
            auto features = kan.forward(h);

            // Print feature stats for first sample
            if (s == 0 && t == 0) {
                double fmin = 1e30, fmax = -1e30, fsum = 0, f2sum = 0;
                for (auto v : features) {
                    if (v < fmin) fmin = v;
                    if (v > fmax) fmax = v;
                    fsum += v;
                    f2sum += v * v;
                }
                printf("Features[0]: min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                       fmin, fmax, fsum / features.size(),
                       std::sqrt(f2sum / features.size()));
            }

            // Compute logits = features * W_a
            std::vector<double> logits(VA);
            for (size_t a = 0; a < VA; a++) {
                double sum = 0;
                for (size_t i = 0; i < FEAT; i++)
                    sum += features[i] * W_a[i * VA + a];
                logits[a] = sum;
            }

            // Print logit stats for first sample
            if (s == 0 && t == 0) {
                double lmin = 1e30, lmax = -1e30, lsum = 0;
                for (auto v : logits) {
                    if (v < lmin) lmin = v;
                    if (v > lmax) lmax = v;
                    lsum += v;
                }
                printf("Logits[0]: min=%.6f max=%.6f mean=%.6f range=%.6f\n",
                       lmin, lmax, lsum / VA, lmax - lmin);
            }

            // Softmax + CE loss
            double mx = *std::max_element(logits.begin(), logits.end());
            double esum = 0;
            for (auto& v : logits) {
                v = std::exp(std::min(v - mx, 80.0));
                esum += v;
            }
            for (auto& v : logits) v /= esum;

            // Pick random target
            size_t target = rng() % VA;
            double loss = -std::log(std::max(logits[target], 1e-12));
            total_loss += loss;
            total_tokens++;

            if (s == 0 && t == 0) {
                printf("P(target=%zu) = %.8f, loss = %.6f\n", target, logits[target], loss);
            }

            // Simple h evolution (mimic kernel)
            for (size_t i = 0; i < H; i++) h[i] *= 0.9;
        }
    }

    printf("\nAverage loss over %zu tokens: %.6f\n", total_tokens, total_loss / total_tokens);
    printf("Expected random: %.6f (ln(%zu))\n", std::log(VA), VA);
    return 0;
}

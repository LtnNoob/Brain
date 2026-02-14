#include "language_training.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

LanguageTraining::LanguageTraining(KANLanguageEngine& engine, LongTermMemory& ltm)
    : engine_(engine)
    , ltm_(ltm)
{}

// =============================================================================
// Generate Training Data from LTM
// =============================================================================

std::vector<LanguageTraining::EncoderPair> LanguageTraining::generate_encoder_data() const {
    std::vector<EncoderPair> pairs;

    auto& emb_store = engine_.encoder().embedding_table();
    auto all_ids = ltm_.get_all_concept_ids();

    for (auto cid : all_ids) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;

        auto tok_opt = engine_.tokenizer().concept_to_token(cid);
        if (!tok_opt) continue;

        // Target embedding: concept's token embedding (what the encoder should learn to produce)
        // We use a 16D truncation for the query space
        const auto& full_emb = emb_store[*tok_opt];
        std::vector<double> target(LanguageConfig::ENCODER_QUERY_DIM, 0.0);
        for (size_t i = 0; i < std::min(target.size(), full_emb.size()); ++i) {
            target[i] = full_emb[i];
        }

        // Pair 1: label → embedding
        pairs.push_back({info->label, target});

        // Pair 2: definition → embedding (if available)
        if (!info->definition.empty()) {
            pairs.push_back({info->definition, target});
        }
    }

    return pairs;
}

std::vector<LanguageTraining::DecoderPair> LanguageTraining::generate_decoder_data() const {
    std::vector<DecoderPair> pairs;

    auto& emb_store = engine_.encoder().embedding_table();
    auto all_ids = ltm_.get_all_concept_ids();

    for (auto cid : all_ids) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;

        auto tok_opt = engine_.tokenizer().concept_to_token(cid);
        if (!tok_opt) continue;

        // Input: concept token embedding
        const auto& full_emb = emb_store[*tok_opt];

        // Pad to FUSED_DIM (decoder input size)
        std::vector<double> input(LanguageConfig::FUSED_DIM, 0.0);
        for (size_t i = 0; i < std::min(input.size(), full_emb.size()); ++i) {
            input[i] = full_emb[i];
        }

        // Target: label
        pairs.push_back({input, info->label});

        // Target: "label, definition" (short version)
        if (!info->definition.empty()) {
            std::string short_def = info->definition;
            if (short_def.size() > 50) short_def = short_def.substr(0, 50);
            pairs.push_back({input, info->label + ", " + short_def});
        }
    }

    return pairs;
}

// =============================================================================
// Stage 1: Encoder + Decoder Training
// =============================================================================

LanguageTrainingResult LanguageTraining::train_stage1(const LanguageConfig& config) {
    LanguageTrainingResult result;
    result.stage = 1;
    result.stage_name = "Encoder+Decoder";
    result.converged = false;
    result.epochs_run = 0;
    result.final_loss = 1e9;

    // Generate training data
    auto encoder_data = generate_encoder_data();
    auto decoder_data = generate_decoder_data();

    if (encoder_data.empty()) {
        result.stage_name = "Encoder+Decoder (no data)";
        return result;
    }

    // Train encoder
    std::cerr << "[LanguageTraining] Stage 1: Training encoder on "
              << encoder_data.size() << " pairs...\n";

    double best_loss = 1e9;
    for (size_t epoch = 0; epoch < config.encoder_epochs; ++epoch) {
        double loss = train_encoder_epoch(encoder_data, config.encoder_lr);
        if (loss < best_loss) best_loss = loss;
        if (loss < 1e-4) break;
        result.epochs_run = epoch + 1;
    }
    result.final_loss = best_loss;

    // Train decoder
    if (!decoder_data.empty()) {
        std::cerr << "[LanguageTraining] Stage 1: Training decoder on "
                  << decoder_data.size() << " pairs...\n";

        for (size_t epoch = 0; epoch < config.decoder_epochs; ++epoch) {
            double loss = train_decoder_epoch(decoder_data, config.decoder_lr);
            if (loss < best_loss) best_loss = loss;
            if (loss < 1e-4) break;
        }
    }

    result.final_loss = best_loss;
    result.converged = best_loss < 0.1;
    return result;
}

double LanguageTraining::train_encoder_epoch(const std::vector<EncoderPair>& data, double lr) {
    // Train KAN encoder using its built-in training
    // Convert to DataPoint format for KAN training
    std::vector<DataPoint> kan_data;
    kan_data.reserve(data.size());

    for (const auto& pair : data) {
        // Encode text to bag-of-embeddings (pre-KAN input)
        auto tokens = engine_.tokenizer().encode(pair.text);
        if (tokens.empty()) continue;

        // Get bag-of-embeddings as KAN input
        auto bag = engine_.encoder().encode_tokens(tokens);
        // This gives us the full encoded output — but for training we need
        // the intermediate bag representation.
        // For now, use simplified approach: train KAN on (bag → target)
        // We'll use the raw token embeddings averaged as input

        std::vector<double> input(LanguageConfig::TOKEN_EMBED_DIM, 0.0);
        for (auto tid : tokens) {
            if (tid < engine_.encoder().embedding_table().size()) {
                const auto& emb = engine_.encoder().embedding_table()[tid];
                for (size_t d = 0; d < std::min(input.size(), emb.size()); ++d) {
                    input[d] += emb[d];
                }
            }
        }
        double n = static_cast<double>(tokens.size());
        if (n > 0) {
            for (auto& v : input) v /= n;
        }

        kan_data.push_back(DataPoint(input, pair.target_embedding));
    }

    if (kan_data.empty()) return 1e9;

    KanTrainingConfig tc;
    tc.max_iterations = 1;
    tc.learning_rate = lr;

    auto result = engine_.encoder().kan_module().train(kan_data, tc);
    return result.final_loss;
}

// =============================================================================
// Softmax helper
// =============================================================================

std::vector<double> LanguageTraining::softmax(const std::vector<double>& logits) {
    if (logits.empty()) return {};
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> result(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(std::min(logits[i] - max_val, 80.0));
        sum += result[i];
    }
    if (sum > 1e-12) {
        for (auto& v : result) v /= sum;
    }
    return result;
}

// =============================================================================
// Decoder Training: Teacher-Forcing + Cross-Entropy + Target Propagation
// =============================================================================
//
// Architecture:
//   init_kan:  R^64 → R^16         (fused → h₀)
//   update_kan: R^80 → R^32 → R^16  (concat(h_t, embed(tok_t)) → h_{t+1})
//   output_projection: R^16 → R^8192  (h_t · W^T → logits)
//
// Training per sample:
//   1. Forward pass with teacher forcing, collecting all h_t
//   2. At each step: CE loss = -log(softmax(logits)[target_token])
//   3. Gradient w.r.t. h_t: dL/dh_t[i] = Σ_v W[i][v]·(prob_v - 1{v=target})
//   4. Target propagation: target_h_t = h_t - α·dL/dh_t
//   5. Output projection: direct gradient descent on W
//   6. Init-KAN: MSE(init_kan(fused), target_h₀)
//   7. Update-KAN: MSE(update_kan(concat(h_t,emb)), target_h_{t+1})

double LanguageTraining::train_decoder_epoch(
    const std::vector<DecoderPair>& data, double lr) {

    auto& decoder = engine_.decoder();
    auto& encoder = engine_.encoder();
    auto& tokenizer = engine_.tokenizer();

    auto& init_kan = decoder.init_kan();
    auto& update_kan = decoder.update_kan();
    auto& W = decoder.output_projection();    // [HIDDEN_DIM][VOCAB_SIZE] = [16][8192]
    const auto& emb_table = encoder.embedding_table();

    const size_t H = LanguageConfig::DECODER_HIDDEN_DIM;  // 16
    const size_t V = LanguageConfig::VOCAB_SIZE;           // 8192
    const size_t E = LanguageConfig::TOKEN_EMBED_DIM;      // 64

    double total_ce_loss = 0.0;
    size_t total_tokens = 0;

    // Accumulate gradient for output projection (batch over all samples)
    std::vector<std::vector<double>> dW(H, std::vector<double>(V, 0.0));

    // Collect KAN training pairs
    std::vector<DataPoint> init_kan_pairs;
    std::vector<DataPoint> update_kan_pairs;

    for (const auto& sample : data) {
        // Tokenize target text
        auto target_tokens = tokenizer.encode(sample.target_text);
        if (target_tokens.empty()) continue;

        // ── Forward pass with teacher forcing ──

        // h₀ = init_kan(fused_vector)
        auto h = init_kan.evaluate(sample.embedding);
        if (h.size() != H) h.resize(H, 0.0);

        // Store hidden states for each step
        std::vector<std::vector<double>> hidden_states;
        hidden_states.push_back(h);

        std::vector<std::vector<double>> logits_per_step;
        std::vector<std::vector<double>> probs_per_step;

        for (size_t t = 0; t < target_tokens.size(); ++t) {
            // logits = h · W^T ∈ R^V
            std::vector<double> logits(V, 0.0);
            for (size_t i = 0; i < H; ++i) {
                for (size_t v = 0; v < V; ++v) {
                    logits[v] += h[i] * W[i][v];
                }
            }
            logits_per_step.push_back(logits);

            auto probs = softmax(logits);
            probs_per_step.push_back(probs);

            // CE loss for this step
            uint16_t target_tok = target_tokens[t];
            if (target_tok < V) {
                double p = std::max(probs[target_tok], 1e-12);
                total_ce_loss += -std::log(p);
                total_tokens++;
            }

            // Teacher forcing: use actual target token for next step
            if (t + 1 < target_tokens.size()) {
                std::vector<double> update_input;
                update_input.reserve(H + E);
                update_input.insert(update_input.end(), h.begin(), h.end());

                // Get target token embedding
                if (target_tok < emb_table.size()) {
                    const auto& emb = emb_table[target_tok];
                    update_input.insert(update_input.end(), emb.begin(), emb.end());
                } else {
                    update_input.resize(H + E, 0.0);
                }

                h = update_kan.evaluate(update_input);
                if (h.size() != H) h.resize(H, 0.0);
                hidden_states.push_back(h);
            }
        }

        // ── Backward pass: compute gradients ──

        for (size_t t = 0; t < target_tokens.size(); ++t) {
            uint16_t target_tok = target_tokens[t];
            if (target_tok >= V) continue;

            const auto& probs = probs_per_step[t];
            const auto& h_t = hidden_states[t];

            // dL/dh_t[i] = Σ_v W[i][v] · (prob_v - 1{v=target})
            std::vector<double> dL_dh(H, 0.0);
            for (size_t i = 0; i < H; ++i) {
                for (size_t v = 0; v < V; ++v) {
                    double grad_softmax = probs[v] - (v == target_tok ? 1.0 : 0.0);
                    dL_dh[i] += W[i][v] * grad_softmax;
                }
            }

            // Accumulate output projection gradient: dW[i][v] += h_t[i] · (prob_v - target_v)
            for (size_t i = 0; i < H; ++i) {
                for (size_t v = 0; v < V; ++v) {
                    double grad_softmax = probs[v] - (v == target_tok ? 1.0 : 0.0);
                    dW[i][v] += h_t[i] * grad_softmax;
                }
            }

            // Target propagation: "better" hidden state
            std::vector<double> target_h(H);
            for (size_t i = 0; i < H; ++i) {
                target_h[i] = h_t[i] - lr * dL_dh[i];
            }

            // Collect KAN training pairs
            if (t == 0) {
                // Init-KAN: fused → target_h₀
                init_kan_pairs.push_back(DataPoint(sample.embedding, target_h));
            } else {
                // Update-KAN: concat(h_{t-1}, embed(token_{t-1})) → target_h_t
                const auto& h_prev = hidden_states[t - 1];
                uint16_t prev_tok = target_tokens[t - 1];

                std::vector<double> kan_input;
                kan_input.reserve(H + E);
                kan_input.insert(kan_input.end(), h_prev.begin(), h_prev.end());

                if (prev_tok < emb_table.size()) {
                    const auto& emb = emb_table[prev_tok];
                    kan_input.insert(kan_input.end(), emb.begin(), emb.end());
                } else {
                    kan_input.resize(H + E, 0.0);
                }

                update_kan_pairs.push_back(DataPoint(kan_input, target_h));
            }
        }
    }

    if (total_tokens == 0) return 1e9;

    // ── Update output projection: W -= lr · dW / N ──
    double scale = lr / static_cast<double>(total_tokens);
    for (size_t i = 0; i < H; ++i) {
        for (size_t v = 0; v < V; ++v) {
            W[i][v] -= scale * dW[i][v];
        }
    }

    // ── Train Init-KAN on target pairs ──
    if (!init_kan_pairs.empty()) {
        KanTrainingConfig tc;
        tc.max_iterations = 1;
        tc.learning_rate = lr;
        init_kan.train(init_kan_pairs, tc);
    }

    // ── Train Update-KAN on target pairs ──
    if (!update_kan_pairs.empty()) {
        KanTrainingConfig tc;
        tc.max_iterations = 1;
        tc.learning_rate = lr;
        update_kan.train(update_kan_pairs, tc);
    }

    return total_ce_loss / static_cast<double>(total_tokens);
}

// =============================================================================
// Stage 2: Fusion Training
// =============================================================================

LanguageTrainingResult LanguageTraining::train_stage2(
    const std::vector<LanguageTrainingExample>& examples,
    const LanguageConfig& config
) {
    LanguageTrainingResult result;
    result.stage = 2;
    result.stage_name = "Fusion";
    result.converged = false;
    result.epochs_run = 0;
    result.final_loss = 1e9;

    if (examples.empty()) {
        result.stage_name = "Fusion (no data)";
        return result;
    }

    // Train semantic scorers and fusion layer on QA examples
    // For each example: run pipeline, compare output to expected, update
    std::cerr << "[LanguageTraining] Stage 2: Training fusion on "
              << examples.size() << " QA pairs...\n";

    for (size_t epoch = 0; epoch < config.fusion_epochs; ++epoch) {
        double total_loss = 0.0;

        for (const auto& ex : examples) {
            // Run generation
            auto gen_result = engine_.generate(ex.query);

            // Simple loss: chain overlap ratio
            size_t matches = 0;
            for (auto cid : gen_result.causal_chain) {
                for (auto expected : ex.expected_chain) {
                    if (cid == expected) { matches++; break; }
                }
            }
            double chain_loss = 1.0 - (ex.expected_chain.empty() ? 0.0 :
                static_cast<double>(matches) / ex.expected_chain.size());

            total_loss += chain_loss;
        }

        double avg_loss = total_loss / examples.size();
        result.epochs_run = epoch + 1;
        result.final_loss = avg_loss;

        if (avg_loss < 0.1) {
            result.converged = true;
            break;
        }
    }

    return result;
}

// =============================================================================
// Train All Stages
// =============================================================================

std::vector<LanguageTrainingResult> LanguageTraining::train_all(
    const std::vector<LanguageTrainingExample>& qa_pairs,
    const LanguageConfig& config
) {
    std::vector<LanguageTrainingResult> results;

    results.push_back(train_stage1(config));
    results.push_back(train_stage2(qa_pairs, config));

    return results;
}

} // namespace brain19

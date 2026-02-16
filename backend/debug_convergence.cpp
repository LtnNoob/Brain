// =============================================================================
// Brain19 Convergence Port Diagnostic
// =============================================================================
//
// Diagnoses why V12 loss is stuck at 21+ instead of ~1.8 (V11 baseline).
// Checks the ConvergencePort integration: 122D input -> 32D output appended
// to the fused embedding, making it 122D total instead of 90D.
//
// Diagnostic checks:
//   1. ConvergencePort forward pass: input/output values and statistics
//   2. conv_table construction: non-zero entries, value statistics
//   3. Training pair generation: fused embedding dimension and convergence block
//

#include "ltm/long_term_memory.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "micromodel/concept_embedding_store.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "cmodel/concept_model.hpp"
#include "language/kan_language_engine.hpp"
#include "language/language_training.hpp"
#include "language/language_config.hpp"
#include "convergence/convergence_config.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace brain19;

static void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(72, '=') << "\n";
}

static void print_stats(const std::string& label, const double* data, size_t n) {
    double mn = *std::min_element(data, data + n);
    double mx = *std::max_element(data, data + n);
    double sum = std::accumulate(data, data + n, 0.0);
    double mean = sum / static_cast<double>(n);
    double sq_sum = 0.0;
    for (size_t i = 0; i < n; ++i) sq_sum += data[i] * data[i];
    double rms = std::sqrt(sq_sum / static_cast<double>(n));
    size_t nz = 0;
    for (size_t i = 0; i < n; ++i) if (std::abs(data[i]) > 1e-15) nz++;

    std::cout << "  " << label << ": min=" << std::fixed << std::setprecision(6) << mn
              << " max=" << mx << " mean=" << mean
              << " rms=" << rms << " nonzero=" << nz << "/" << n << "\n";
}

int main() {
    std::cout << "Brain19 Convergence Port Diagnostic\n";
    std::cout << "====================================\n\n";

    // Print dimension constants
    std::cout << "Constants:\n";
    std::cout << "  FUSED_DIM          = " << LanguageConfig::FUSED_DIM << "\n";
    std::cout << "  ENCODER_QUERY_DIM  = " << LanguageConfig::ENCODER_QUERY_DIM << "\n";
    std::cout << "  CONVERGENCE_DIM    = " << LanguageConfig::CONVERGENCE_DIM << "\n";
    std::cout << "  ConvergencePort::INPUT_DIM  = " << ConvergencePort::INPUT_DIM << "\n";
    std::cout << "  ConvergencePort::OUTPUT_DIM = " << ConvergencePort::OUTPUT_DIM << "\n";
    std::cout << "  convergence::QUERY_DIM      = " << convergence::QUERY_DIM << "\n";
    std::cout << "  convergence::CM_INPUT_DIM   = " << convergence::CM_INPUT_DIM << "\n";

    // ── Step 1: Load Foundation KB ──
    print_separator("Step 1: Loading Foundation KB");
    LongTermMemory ltm;

    bool loaded = false;
    for (const auto& path : {"../data/foundation_full.json", "data/foundation_full.json",
                              "../data/foundation.json", "data/foundation.json"}) {
        if (FoundationConcepts::seed_from_file(ltm, path)) {
            std::cout << "  Loaded from: " << path << "\n";
            loaded = true;
            break;
        }
    }
    if (!loaded) {
        std::cout << "  FALLBACK: using hardcoded seeds\n";
        FoundationConcepts::seed_all(ltm);
    }

    auto all_ids = ltm.get_all_concept_ids();
    std::cout << "  Concepts: " << all_ids.size() << "\n";
    std::cout << "  Relations: " << ltm.total_relation_count() << "\n";

    // ── Step 2: Property Inheritance ──
    print_separator("Step 2: Property Inheritance");
    {
        PropertyInheritance prop(ltm);
        PropertyInheritance::Config cfg;
        cfg.decay_per_hop = 0.9;
        cfg.trust_floor = 0.3;
        cfg.max_iterations = 50;
        cfg.max_hop_depth = 20;
        cfg.propagate_requires = true;
        cfg.propagate_uses = true;
        cfg.propagate_produces = true;
        auto pi_result = prop.propagate(cfg);
        std::cout << "  Iterations: " << pi_result.iterations_run
                  << (pi_result.converged ? " (converged)" : " (max reached)") << "\n";
        std::cout << "  Properties inherited: " << pi_result.properties_inherited << "\n";
        std::cout << "  Relations now: " << ltm.total_relation_count() << "\n";
    }

    // ── Step 3: Train Embeddings ──
    print_separator("Step 3: Training Embeddings");
    EmbeddingManager embeddings;
    {
        auto emb_result = embeddings.train_embeddings(ltm, 0.05, 10);
        std::cout << "  Embedding iterations: " << emb_result.iterations << "\n";
    }

    // ── Step 4: Train ConceptModels ──
    print_separator("Step 4: Training ConceptModels");
    ConceptModelRegistry registry;
    {
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto stats = trainer.train_all(registry, embeddings, ltm);
        std::cout << "  Models trained: " << stats.models_trained
                  << " (" << stats.models_converged << " converged)\n";
    }

    // ── Step 5: ConvergencePort Forward Pass for 10 Sample Concepts ──
    print_separator("Step 5: ConvergencePort Forward Pass (10 samples)");
    {
        all_ids = ltm.get_all_concept_ids();
        auto& concept_emb_store = embeddings.concept_embeddings();

        size_t count = 0;
        for (auto cid : all_ids) {
            if (count >= 10) break;
            if (!registry.has_model(cid)) continue;

            auto info = ltm.retrieve_concept(cid);
            if (!info) continue;

            auto* cm = registry.get_model(cid);

            // Build input: same as generate_decoder_data builds conv_input
            // It copies fused[0..QUERY_DIM-1] into conv_input
            // But first let's try the simpler path used by conv_table:
            // just FlexEmbedding core[0..15] into conv_input[0..15], rest zeros
            std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
            auto flex_emb = concept_emb_store.get_or_default(cid);
            for (size_t d = 0; d < std::min(size_t(16), flex_emb.core.size()); ++d)
                conv_input[d] = flex_emb.core[d];

            double conv_out[ConvergencePort::OUTPUT_DIM];
            cm->forward_convergence(conv_input.data(), conv_out);

            std::cout << "\n  Concept " << cid << " (" << info->label << "):\n";

            // Print input (first 10 dims)
            std::cout << "    Input[0..9]: ";
            for (size_t i = 0; i < 10; ++i)
                std::cout << std::fixed << std::setprecision(4) << conv_input[i] << " ";
            std::cout << "\n";

            // Print all 32 output dims
            std::cout << "    Output[0..31]: ";
            for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i)
                std::cout << std::fixed << std::setprecision(4) << conv_out[i] << " ";
            std::cout << "\n";

            print_stats("Output", conv_out, ConvergencePort::OUTPUT_DIM);

            // Also print the ConvergencePort weight statistics
            const auto& port = cm->convergence_port();
            print_stats("Port.W", port.W.data(), ConvergencePort::W_SIZE);
            print_stats("Port.b", port.b.data(), ConvergencePort::OUTPUT_DIM);

            count++;
        }

        if (count == 0)
            std::cout << "  WARNING: No concepts with ConceptModels found!\n";
    }

    // ── Step 6: Initialize KAN Language Engine ──
    print_separator("Step 6: KAN Language Engine Initialization");
    LanguageConfig lang_config;
    KANLanguageEngine engine(lang_config, ltm, registry, embeddings);
    engine.initialize();

    if (!engine.is_ready()) {
        std::cerr << "[FATAL] Language engine not ready\n";
        return 1;
    }
    std::cout << "  Tokenizer vocab: " << engine.tokenizer().vocab_size() << "\n";
    engine.rebuild_dimensional_context();

    auto& decoder = engine.decoder();
    size_t H = decoder.extended_fused_dim();
    size_t flex_dim = decoder.flex_dim();
    std::cout << "  extended_fused_dim (H) = " << H << "\n";
    std::cout << "  flex_dim = " << flex_dim << "\n";
    std::cout << "  dim_context decoder_dim = " << engine.dim_context().decoder_dim() << "\n";
    std::cout << "  Expected: FUSED(64) + flex(" << flex_dim
              << ") + dim_ctx(" << engine.dim_context().decoder_dim()
              << ") + conv(" << LanguageConfig::CONVERGENCE_DIM << ") = "
              << (LanguageConfig::FUSED_DIM + flex_dim
                  + engine.dim_context().decoder_dim()
                  + LanguageConfig::CONVERGENCE_DIM) << "\n";

    // ── Step 7: Build conv_table and analyze ──
    print_separator("Step 7: conv_table Analysis");
    {
        const size_t V = LanguageConfig::VOCAB_SIZE;
        const size_t CONV_DIM = LanguageConfig::CONVERGENCE_DIM;

        std::vector<double> conv_table(V * CONV_DIM, 0.0);
        size_t tokens_with_conv = 0;

        for (size_t v = 0; v < V; ++v) {
            auto cpt = engine.tokenizer().token_to_concept(static_cast<uint16_t>(v));
            if (cpt && registry.has_model(*cpt)) {
                auto* cm = registry.get_model(*cpt);
                std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
                auto flex_emb = embeddings.concept_embeddings().get_or_default(*cpt);
                for (size_t d = 0; d < std::min(size_t(16), flex_emb.core.size()); ++d)
                    conv_input[d] = flex_emb.core[d];
                double conv_out[ConvergencePort::OUTPUT_DIM];
                cm->forward_convergence(conv_input.data(), conv_out);
                for (size_t d = 0; d < CONV_DIM; ++d)
                    conv_table[v * CONV_DIM + d] = conv_out[d];
                tokens_with_conv++;
            }
        }

        // Count non-zero entries
        size_t nonzero = 0;
        double total_min = 1e30, total_max = -1e30, total_sum = 0.0;
        double total_abs_sum = 0.0;
        size_t total_entries = V * CONV_DIM;

        for (size_t i = 0; i < total_entries; ++i) {
            if (std::abs(conv_table[i]) > 1e-15) nonzero++;
            if (conv_table[i] < total_min) total_min = conv_table[i];
            if (conv_table[i] > total_max) total_max = conv_table[i];
            total_sum += conv_table[i];
            total_abs_sum += std::abs(conv_table[i]);
        }

        std::cout << "  Tokens with convergence data: " << tokens_with_conv << " / " << V << "\n";
        std::cout << "  Non-zero entries: " << nonzero << " / " << total_entries << "\n";
        std::cout << "  Min: " << std::fixed << std::setprecision(8) << total_min << "\n";
        std::cout << "  Max: " << total_max << "\n";
        std::cout << "  Mean: " << (total_sum / static_cast<double>(total_entries)) << "\n";
        std::cout << "  Mean(abs): " << (total_abs_sum / static_cast<double>(total_entries)) << "\n";

        // Only compute stats over non-zero rows
        if (tokens_with_conv > 0) {
            double nz_min = 1e30, nz_max = -1e30, nz_sum = 0.0, nz_sq_sum = 0.0;
            size_t nz_count = tokens_with_conv * CONV_DIM;
            for (size_t v = 0; v < V; ++v) {
                auto cpt = engine.tokenizer().token_to_concept(static_cast<uint16_t>(v));
                if (!cpt || !registry.has_model(*cpt)) continue;
                for (size_t d = 0; d < CONV_DIM; ++d) {
                    double val = conv_table[v * CONV_DIM + d];
                    if (val < nz_min) nz_min = val;
                    if (val > nz_max) nz_max = val;
                    nz_sum += val;
                    nz_sq_sum += val * val;
                }
            }
            std::cout << "\n  Stats over active tokens only (" << tokens_with_conv << " tokens):\n";
            std::cout << "    Min: " << nz_min << "\n";
            std::cout << "    Max: " << nz_max << "\n";
            std::cout << "    Mean: " << (nz_sum / static_cast<double>(nz_count)) << "\n";
            std::cout << "    RMS: " << std::sqrt(nz_sq_sum / static_cast<double>(nz_count)) << "\n";
        }

        // Print 5 sample entries
        std::cout << "\n  Sample conv_table entries:\n";
        size_t sample_count = 0;
        for (size_t v = 0; v < V && sample_count < 5; ++v) {
            auto cpt = engine.tokenizer().token_to_concept(static_cast<uint16_t>(v));
            if (!cpt || !registry.has_model(*cpt)) continue;

            auto info = ltm.retrieve_concept(*cpt);
            std::string label = info ? info->label : "?";

            std::cout << "    token=" << v << " concept=" << *cpt
                      << " (" << label << "): [";
            for (size_t d = 0; d < CONV_DIM; ++d) {
                if (d > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4)
                          << conv_table[v * CONV_DIM + d];
            }
            std::cout << "]\n";
            sample_count++;
        }
    }

    // ── Step 8: Generate decoder data and analyze fused embeddings ──
    print_separator("Step 8: Decoder Training Data Analysis");
    {
        LanguageTraining lang_trainer(engine, ltm, registry);
        auto decoder_data = lang_trainer.generate_decoder_data();

        std::cout << "  Total decoder pairs: " << decoder_data.size() << "\n";

        if (!decoder_data.empty()) {
            // Dimension analysis
            size_t first_dim = decoder_data[0].embedding.size();
            size_t min_dim = first_dim, max_dim = first_dim;
            for (const auto& pair : decoder_data) {
                if (pair.embedding.size() < min_dim) min_dim = pair.embedding.size();
                if (pair.embedding.size() > max_dim) max_dim = pair.embedding.size();
            }

            std::cout << "  Fused embedding dimension: min=" << min_dim
                      << " max=" << max_dim << "\n";
            std::cout << "  Expected: 64(fused) + 16(flex) + "
                      << engine.dim_context().decoder_dim() << "(dim_ctx) + 32(conv) = "
                      << (64 + 16 + engine.dim_context().decoder_dim() + 32) << "\n";

            // Analyze convergence block (last 32 dims) in training pairs
            const size_t CONV_DIM = LanguageConfig::CONVERGENCE_DIM;
            size_t all_zero_count = 0;
            size_t has_nonzero_count = 0;
            double conv_block_min = 1e30, conv_block_max = -1e30;
            double conv_block_sum = 0.0, conv_block_sq = 0.0;
            size_t conv_total_entries = 0;

            size_t pairs_to_check = std::min(decoder_data.size(), size_t(50));
            for (size_t i = 0; i < pairs_to_check; ++i) {
                const auto& emb = decoder_data[i].embedding;
                if (emb.size() < CONV_DIM) continue;

                size_t conv_start = emb.size() - CONV_DIM;
                bool all_zero = true;
                for (size_t d = conv_start; d < emb.size(); ++d) {
                    if (std::abs(emb[d]) > 1e-15) all_zero = false;
                    if (emb[d] < conv_block_min) conv_block_min = emb[d];
                    if (emb[d] > conv_block_max) conv_block_max = emb[d];
                    conv_block_sum += emb[d];
                    conv_block_sq += emb[d] * emb[d];
                    conv_total_entries++;
                }
                if (all_zero) all_zero_count++;
                else has_nonzero_count++;
            }

            std::cout << "\n  Convergence block (last " << CONV_DIM << " dims) in first "
                      << pairs_to_check << " pairs:\n";
            std::cout << "    All-zero pairs: " << all_zero_count << "\n";
            std::cout << "    Non-zero pairs: " << has_nonzero_count << "\n";

            if (conv_total_entries > 0) {
                double conv_mean = conv_block_sum / static_cast<double>(conv_total_entries);
                double conv_rms = std::sqrt(conv_block_sq / static_cast<double>(conv_total_entries));
                std::cout << "    Min: " << std::fixed << std::setprecision(8) << conv_block_min << "\n";
                std::cout << "    Max: " << conv_block_max << "\n";
                std::cout << "    Mean: " << conv_mean << "\n";
                std::cout << "    RMS: " << conv_rms << "\n";
            }

            // Print detailed breakdown for first 5 pairs
            std::cout << "\n  Detailed first 5 pairs:\n";
            for (size_t i = 0; i < std::min(decoder_data.size(), size_t(5)); ++i) {
                const auto& emb = decoder_data[i].embedding;
                std::cout << "    Pair " << i << ": dim=" << emb.size()
                          << " target=\"" << decoder_data[i].target_text.substr(0, 60) << "...\"\n";

                // Print fused block stats (first 64 dims)
                if (emb.size() >= 64) {
                    print_stats("Fused[0..63]", emb.data(), 64);
                }

                // Print flex block (64..79)
                if (emb.size() >= 80) {
                    print_stats("Flex[64..79]", emb.data() + 64, 16);
                }

                // Print dim_ctx block
                size_t dim_ctx_size = engine.dim_context().decoder_dim();
                if (emb.size() >= 80 + dim_ctx_size) {
                    print_stats("DimCtx[80.." + std::to_string(79 + dim_ctx_size) + "]",
                                emb.data() + 80, dim_ctx_size);
                }

                // Print convergence block (last 32 dims)
                if (emb.size() >= CONV_DIM) {
                    size_t conv_start = emb.size() - CONV_DIM;
                    std::cout << "    Conv[" << conv_start << ".." << (emb.size()-1) << "]: ";
                    for (size_t d = conv_start; d < emb.size(); ++d)
                        std::cout << std::fixed << std::setprecision(4) << emb[d] << " ";
                    std::cout << "\n";
                    print_stats("Conv block", emb.data() + conv_start, CONV_DIM);
                }
            }

            // ── Key diagnostic: compare magnitude of convergence vs other blocks ──
            std::cout << "\n  Block magnitude comparison (mean |x| over all pairs):\n";
            double fused_abs = 0, flex_abs = 0, dimctx_abs = 0, conv_abs = 0;
            size_t n_pairs = decoder_data.size();
            size_t dim_ctx_size = engine.dim_context().decoder_dim();

            for (const auto& pair : decoder_data) {
                const auto& emb = pair.embedding;
                for (size_t d = 0; d < std::min(size_t(64), emb.size()); ++d)
                    fused_abs += std::abs(emb[d]);
                for (size_t d = 64; d < std::min(size_t(80), emb.size()); ++d)
                    flex_abs += std::abs(emb[d]);
                for (size_t d = 80; d < std::min(size_t(80 + dim_ctx_size), emb.size()); ++d)
                    dimctx_abs += std::abs(emb[d]);
                if (emb.size() >= CONV_DIM) {
                    size_t conv_start = emb.size() - CONV_DIM;
                    for (size_t d = conv_start; d < emb.size(); ++d)
                        conv_abs += std::abs(emb[d]);
                }
            }

            std::cout << "    Fused (64D):   mean|x| = " << std::fixed << std::setprecision(6)
                      << (fused_abs / (n_pairs * 64.0)) << "\n";
            std::cout << "    Flex  (16D):   mean|x| = "
                      << (flex_abs / (n_pairs * 16.0)) << "\n";
            std::cout << "    DimCtx(" << dim_ctx_size << "D): mean|x| = "
                      << (dimctx_abs / (n_pairs * std::max(dim_ctx_size, size_t(1)))) << "\n";
            std::cout << "    Conv  (32D):   mean|x| = "
                      << (conv_abs / (n_pairs * 32.0)) << "\n";
        }
    }

    // ── Summary ──
    print_separator("DIAGNOSTIC SUMMARY");
    std::cout << "\n  Key questions to answer:\n";
    std::cout << "  1. Are ConvergencePort weights initialized (non-zero W)?\n";
    std::cout << "  2. Are ConvergencePort outputs saturated (all near +/-1 from tanh)?\n";
    std::cout << "  3. Is the conv_table sparsely populated (few tokens with data)?\n";
    std::cout << "  4. Is the convergence block magnitude comparable to other blocks?\n";
    std::cout << "  5. Does the fused dimension match expectations (122 = 64+16+10+32)?\n";
    std::cout << "\n  If convergence outputs are large/saturated, they dominate the\n";
    std::cout << "  output projection and break the signal from the other 90 dims.\n";
    std::cout << "  V11 baseline uses 90D without convergence -> loss ~1.8.\n";
    std::cout << "\n";

    return 0;
}

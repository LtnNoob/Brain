// =============================================================================
// Brain19 V12 CPU Training Run — Deep KAN Decoder (per-sample online SGD)
// =============================================================================
//
// Same pipeline as V12 GPU but uses the CPU DeepKAN forward/backward.
// Per-sample online SGD (no mini-batching) — this is the "gold standard"
// training path. Useful for:
//   1. Verifying GPU backward correctness
//   2. CPU performance baseline
//   3. Understanding convergence behavior without batch effects
//
// Architecture: DeepKAN(122→256→128→128) + Linear(128→VA)
// Parameters: ~818K KAN + ~13K output = ~831K total
//

#include "ltm/long_term_memory.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "micromodel/concept_embedding_store.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_model.hpp"
#include "cmodel/concept_trainer.hpp"
#include "language/kan_language_engine.hpp"
#include "language/language_training.hpp"
#include "language/language_config.hpp"
#include "language/deep_kan.hpp"
#include "convergence/convergence_config.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <atomic>
#include <numeric>

using namespace brain19;

static std::ofstream g_log;

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::string line = std::string("[") + ts + "] " + msg;
    std::cout << line << "\n";
    if (g_log.is_open()) {
        g_log << line << "\n";
        g_log.flush();
    }
}

int main() {
    g_log.open("/tmp/brain19_v12_cpu.log");
    log("=== Brain19 V12 CPU Training Run (Deep KAN, per-sample SGD) ===");
    log("");

    auto t0 = std::chrono::steady_clock::now();

    // ── Step 1: Load Foundation KB ──
    log("[1/6] Loading foundation KB...");
    LongTermMemory ltm;

    bool loaded = false;
    for (const auto& path : {"../data/foundation_full.json", "data/foundation_full.json",
                              "../data/foundation.json", "data/foundation.json"}) {
        if (FoundationConcepts::seed_from_file(ltm, path)) {
            log("  Loaded from: " + std::string(path));
            loaded = true;
            break;
        }
    }
    if (!loaded) {
        log("  FALLBACK: using hardcoded seeds");
        FoundationConcepts::seed_all(ltm);
    }

    size_t initial_concepts = ltm.get_all_concept_ids().size();
    size_t initial_relations = ltm.total_relation_count();
    log("  Concepts: " + std::to_string(initial_concepts));
    log("  Relations: " + std::to_string(initial_relations));

    // ── Step 2: Property Inheritance ──
    log("");
    log("[2/6] Running PropertyInheritance::propagate()...");
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

        log("  Iterations: " + std::to_string(pi_result.iterations_run)
            + (pi_result.converged ? " (converged)" : " (max reached)"));
        log("  Properties inherited: " + std::to_string(pi_result.properties_inherited));
    }

    // ── Step 3: Train Embeddings ──
    log("");
    log("[3/6] Training embeddings from graph...");
    EmbeddingManager embeddings;
    {
        auto t_emb = std::chrono::steady_clock::now();
        auto emb_result = embeddings.train_embeddings(ltm, 0.05, 10);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_emb).count();
        log("  Iterations: " + std::to_string(emb_result.iterations));
        log("  Time: " + std::to_string(elapsed) + "ms");
    }

    // ── Step 4: Train ConceptModels ──
    log("");
    log("[4/6] Training ConceptModels...");
    ConceptModelRegistry registry;
    {
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto t_cm = std::chrono::steady_clock::now();
        auto stats = trainer.train_all(registry, embeddings, ltm);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_cm).count();
        log("  Trained: " + std::to_string(stats.models_trained)
            + " models (" + std::to_string(stats.models_converged) + " converged)");
        log("  Time: " + std::to_string(elapsed) + "ms");
    }

    // ── Step 5: Init KAN Language Engine ──
    log("");
    log("[5/6] Initializing KAN Language Engine...");
    LanguageConfig lang_config;
    KANLanguageEngine engine(lang_config, ltm, registry, embeddings);
    engine.initialize();
    if (!engine.is_ready()) {
        std::cerr << "[FATAL] Language engine not ready after initialize()\n";
        return 1;
    }
    log("  Tokenizer: " + std::to_string(engine.tokenizer().vocab_size()) + " tokens");
    engine.rebuild_dimensional_context();
    log("  Dimensional context rebuilt");

    // ── Step 6: CPU Deep KAN Training ──
    log("");
    log("[6/6] CPU Deep KAN Training (per-sample online SGD)...");

    // Use the same data generation as the GPU path
    LanguageTraining lang_trainer(engine, ltm, registry);

    // Generate training data
    log("  Generating training data...");
    auto t_data = std::chrono::steady_clock::now();

    // Use the LanguageTraining class's data generation (via a minimal config)
    lang_config.encoder_epochs = 0;
    lang_config.decoder_epochs = 300;
    lang_config.decoder_lr = 2.0;          // Standard online SGD lr
    lang_config.deep_kan_lr = 0.01;        // Standard KAN lr
    lang_config.use_deep_kan = true;

    // We need to generate decoder data ourselves since train_stage1_deep_kan is private.
    // Re-implement the data generation and training loop here.

    auto& tok = engine.tokenizer();
    auto& decoder = engine.decoder();
    auto& emb_table = engine.encoder().embedding_table();
    const size_t V = LanguageConfig::VOCAB_SIZE;
    const size_t H = decoder.extended_fused_dim();  // 122
    const size_t FUSED_BASE = LanguageConfig::FUSED_DIM;  // 64
    const size_t CONV_DIM = LanguageConfig::CONVERGENCE_DIM;  // 32
    const size_t conv_start = H - CONV_DIM;  // 90
    const size_t flex_start = FUSED_BASE;  // 64
    const size_t flex_end = flex_start + decoder.flex_dim();  // 64+16=80
    const size_t dimctx_start = flex_end;  // 80
    const size_t FEAT_DIM = 128;
    const double lr_output = lang_config.decoder_lr;
    const double lr_kan = lang_config.deep_kan_lr;
    const size_t warmup_epochs = 10;

    log("  H=" + std::to_string(H) + " FUSED=" + std::to_string(FUSED_BASE)
        + " FLEX=" + std::to_string(decoder.flex_dim())
        + " CONV=" + std::to_string(CONV_DIM)
        + " FEAT=" + std::to_string(FEAT_DIM));

    // Collect quality concept descriptions (same logic as generate_decoder_data)
    struct DecoderSample {
        std::vector<double> embedding;  // H dims
        std::vector<uint16_t> tokens;
    };
    std::vector<DecoderSample> samples;
    std::vector<bool> seen(V, false);

    auto all_ids = ltm.get_all_concept_ids();
    for (auto cid : all_ids) {
        auto cinfo = ltm.retrieve_concept(cid);
        if (!cinfo || cinfo->definition.empty()) continue;
        if (cinfo->epistemic.trust < 0.3) continue;

        // Build embedding (same as train_stage1_deep_kan)
        std::vector<double> emb(H, 0.0);

        // Fused embedding from encoder
        auto enc_emb = engine.encoder().encode(cinfo->definition, tok);
        for (size_t i = 0; i < std::min(FUSED_BASE, enc_emb.size()); ++i)
            emb[i] = enc_emb[i];

        // Flex detail
        auto flex_emb = embeddings.concept_embeddings().get_or_default(cid);
        for (size_t i = 0; i < decoder.flex_dim() && i < flex_emb.detail.size(); ++i)
            emb[flex_start + i] = flex_emb.detail[i];

        // DimCtx (use dimensional context from engine)
        auto& dim_ctx = engine.dim_context();
        auto dim_vals = dim_ctx.to_decoder_vec(cid);
        for (size_t i = 0; i < dim_vals.size() && (dimctx_start + i) < conv_start; ++i)
            emb[dimctx_start + i] = dim_vals[i];

        // Convergence features
        if (registry.has_model(cid)) {
            auto* cm = registry.get_model(cid);
            std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
            for (size_t d = 0; d < std::min(size_t(16), flex_emb.core.size()); ++d)
                conv_input[d] = flex_emb.core[d];
            double conv_out[ConvergencePort::OUTPUT_DIM];
            cm->forward_convergence(conv_input.data(), conv_out);
            for (size_t d = 0; d < CONV_DIM; ++d)
                emb[conv_start + d] = conv_out[d];
        }

        // Tokenize description
        auto tokens = tok.encode(cinfo->definition);
        if (tokens.empty()) continue;
        for (auto t : tokens) if (t < V) seen[t] = true;

        samples.push_back({std::move(emb), std::move(tokens)});
    }

    // Also add relation descriptions (simplified)
    for (auto src_id : all_ids) {
        auto rels = ltm.get_outgoing_relations(src_id);
        for (auto& rel : rels) {
            auto tgt = ltm.retrieve_concept(rel.target);
            auto src = ltm.retrieve_concept(src_id);
            if (!tgt || !src || tgt->label.empty() || src->label.empty()) continue;

            std::string desc = src->label + " " + relation_type_to_string(rel.type) + " " + tgt->label;
            auto tokens = tok.encode(desc);
            if (tokens.empty() || tokens.size() < 2) continue;

            // Use source concept embedding
            std::vector<double> emb(H, 0.0);
            auto enc_emb = engine.encoder().encode(desc, tok);
            for (size_t i = 0; i < std::min(FUSED_BASE, enc_emb.size()); ++i)
                emb[i] = enc_emb[i];
            auto flex_emb = embeddings.concept_embeddings().get_or_default(src_id);
            for (size_t i = 0; i < decoder.flex_dim() && i < flex_emb.detail.size(); ++i)
                emb[flex_start + i] = flex_emb.detail[i];
            if (registry.has_model(src_id)) {
                auto* cm = registry.get_model(src_id);
                std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
                for (size_t d = 0; d < std::min(size_t(16), flex_emb.core.size()); ++d)
                    conv_input[d] = flex_emb.core[d];
                double conv_out[ConvergencePort::OUTPUT_DIM];
                cm->forward_convergence(conv_input.data(), conv_out);
                for (size_t d = 0; d < CONV_DIM; ++d)
                    emb[conv_start + d] = conv_out[d];
            }

            for (auto t : tokens) if (t < V) seen[t] = true;
            samples.push_back({std::move(emb), std::move(tokens)});
        }
    }

    // Build compressed active vocab
    std::vector<uint16_t> active_tokens;
    std::vector<size_t> compress(V, 0);
    for (size_t v = 0; v < V; ++v) {
        if (seen[v]) {
            compress[v] = active_tokens.size();
            active_tokens.push_back(static_cast<uint16_t>(v));
        }
    }
    const size_t VA = active_tokens.size();

    // Shuffle samples
    { std::mt19937 rng(12345); std::shuffle(samples.begin(), samples.end(), rng); }

    auto data_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t_data).count();
    log("  Samples: " + std::to_string(samples.size())
        + ", Active vocab: " + std::to_string(VA)
        + " (data gen: " + std::to_string(data_elapsed) + "s)");

    // Build lookup tables for hidden state evolution
    // Embedding table: [V * FUSED_BASE]
    std::vector<double> emb_table_flat(V * FUSED_BASE, 0.0);
    for (size_t v = 0; v < std::min(emb_table.size(), V); ++v)
        for (size_t j = 0; j < std::min(emb_table[v].size(), FUSED_BASE); ++j)
            emb_table_flat[v * FUSED_BASE + j] = emb_table[v][j];

    // Flex table: [V * flex_dim]
    size_t fd = decoder.flex_dim();
    std::vector<double> flex_table(V * fd, 0.0);
    for (size_t v = 0; v < V; ++v) {
        auto cpt = tok.token_to_concept(static_cast<uint16_t>(v));
        if (cpt) {
            auto fe = embeddings.concept_embeddings().get_or_default(*cpt);
            for (size_t j = 0; j < std::min(fd, fe.detail.size()); ++j)
                flex_table[v * fd + j] = fe.detail[j];
        }
    }

    // Conv table: [V * CONV_DIM]
    std::vector<double> conv_table(V * CONV_DIM, 0.0);
    for (size_t v = 0; v < V; ++v) {
        auto cpt = tok.token_to_concept(static_cast<uint16_t>(v));
        if (cpt && registry.has_model(*cpt)) {
            auto* cm = registry.get_model(*cpt);
            std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
            auto fe = embeddings.concept_embeddings().get_or_default(*cpt);
            for (size_t d = 0; d < std::min(size_t(16), fe.core.size()); ++d)
                conv_input[d] = fe.core[d];
            double conv_out[ConvergencePort::OUTPUT_DIM];
            cm->forward_convergence(conv_input.data(), conv_out);
            for (size_t d = 0; d < CONV_DIM; ++d)
                conv_table[v * CONV_DIM + d] = conv_out[d];
        }
    }

    // ── Init Deep KAN ──
    DeepKAN deep_kan({H, 256, 128, FEAT_DIM}, {8, 5, 5}, 3);
    log("  DeepKAN: " + std::to_string(H) + "→256→128→" + std::to_string(FEAT_DIM)
        + ", " + std::to_string(deep_kan.num_params()) + " params");

    // Block-aware LR scale for first KAN layer
    std::vector<double> lr_input_scale(H, 1.0);
    for (size_t i = flex_start; i < flex_end; ++i) lr_input_scale[i] = 0.3;
    for (size_t i = dimctx_start; i < conv_start; ++i) lr_input_scale[i] = 0.1;
    for (size_t i = conv_start; i < H; ++i) lr_input_scale[i] = 0.3;

    // Init W_a: [FEAT_DIM × VA]
    std::vector<double> W_a(FEAT_DIM * VA);
    {
        std::mt19937 rng(42);
        double scale = std::sqrt(6.0 / (double)(FEAT_DIM + VA));
        std::uniform_real_distribution<double> dist(-scale, scale);
        for (auto& w : W_a) w = dist(rng);
    }

    log("  W_a: " + std::to_string(FEAT_DIM) + "×" + std::to_string(VA)
        + " (" + std::to_string(FEAT_DIM * VA) + " params)");
    log("  LR: output=" + std::to_string(lr_output) + " kan=" + std::to_string(lr_kan));
    log("  Warmup: " + std::to_string(warmup_epochs) + " epochs (W_a only)");
    log("  Threads: " + std::to_string(std::thread::hardware_concurrency()) + " available");
    log("");

    // ── Training loop (per-sample online SGD) ──
    auto t_train = std::chrono::steady_clock::now();
    double best_loss = 1e9;
    const size_t NUM_EPOCHS = 300;

    // Pre-allocate working buffers
    std::vector<double> h(H);
    std::vector<double> logits(VA);
    std::vector<double> probs(VA);
    std::vector<double> d_features(FEAT_DIM);

    for (size_t epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        bool train_kan = (epoch >= warmup_epochs);

        // Cosine LR decay
        double progress = static_cast<double>(epoch)
            / std::max(NUM_EPOCHS - 1, size_t(1));
        double cos_mult = 0.5 * (1.0 + std::cos(progress * 3.14159265358979));
        double lr_out_ep = lr_output * (0.1 + 0.9 * cos_mult);
        double lr_kan_ep = lr_kan * (0.1 + 0.9 * cos_mult);

        double total_loss = 0.0;
        size_t total_tokens = 0;

        for (const auto& sample : samples) {
            // Reset h to sample embedding
            h = sample.embedding;

            for (size_t t = 0; t < sample.tokens.size(); ++t) {
                uint16_t tgt = sample.tokens[t];
                if (tgt >= V || !seen[tgt]) continue;
                size_t ca = compress[tgt];
                if (ca >= VA) continue;

                // ── Forward through DeepKAN ──
                auto features = deep_kan.forward(h);

                // ── Logits: features · W_a ──
                for (size_t a = 0; a < VA; ++a) {
                    double sum = 0.0;
                    for (size_t i = 0; i < FEAT_DIM; ++i)
                        sum += features[i] * W_a[i * VA + a];
                    logits[a] = sum;
                }

                // ── Softmax ──
                double mx = *std::max_element(logits.begin(), logits.begin() + VA);
                double esum = 0.0;
                for (size_t a = 0; a < VA; ++a) {
                    probs[a] = std::exp(std::min(logits[a] - mx, 80.0));
                    esum += probs[a];
                }
                double inv_es = (esum > 1e-12) ? 1.0 / esum : 0.0;
                for (size_t a = 0; a < VA; ++a) probs[a] *= inv_es;

                // ── CE loss ──
                double p = std::max(probs[ca], 1e-12);
                total_loss += -std::log(p);
                total_tokens++;

                // ── Backward: d_features ──
                if (train_kan) {
                    for (size_t i = 0; i < FEAT_DIM; ++i) {
                        double d = 0.0;
                        for (size_t a = 0; a < VA; ++a) {
                            double dl = probs[a] - (a == ca ? 1.0 : 0.0);
                            d += dl * W_a[i * VA + a];
                        }
                        d_features[i] = d;
                    }
                }

                // ── W_a update (per-sample SGD) ──
                for (size_t i = 0; i < FEAT_DIM; ++i) {
                    double fi = features[i];
                    for (size_t a = 0; a < VA; ++a) {
                        double dl = probs[a] - (a == ca ? 1.0 : 0.0);
                        W_a[i * VA + a] -= lr_out_ep * fi * dl;
                    }
                }

                // ── DeepKAN backward (updates KAN weights in-place) ──
                if (train_kan) {
                    deep_kan.backward(d_features, lr_kan_ep, lr_input_scale.data());
                }

                // ── Hidden state evolution (4-block) ──
                // Block 1: Token fused
                if (tgt < emb_table.size()) {
                    for (size_t i = 0; i < FUSED_BASE; ++i)
                        h[i] = h[i] * 0.8 + emb_table_flat[tgt * FUSED_BASE + i] * 0.2;
                }
                // Block 2: FlexDetail
                for (size_t i = 0; i < fd; ++i) {
                    size_t idx = flex_start + i;
                    if (idx < H)
                        h[idx] = h[idx] * 0.9 + flex_table[tgt * fd + i] * 0.1;
                }
                // Block 3: DimCtx
                for (size_t i = dimctx_start; i < conv_start; ++i)
                    h[i] *= 0.95;
                // Block 4: Convergence
                for (size_t d = 0; d < CONV_DIM; ++d)
                    h[conv_start + d] = h[conv_start + d] * 0.9
                        + conv_table[tgt * CONV_DIM + d] * 0.1;
            }
        }

        double avg_loss = (total_tokens > 0) ? total_loss / total_tokens : 1e9;
        if (avg_loss < best_loss) best_loss = avg_loss;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t_train).count();
            log("  Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(NUM_EPOCHS)
                + " loss=" + std::to_string(avg_loss)
                + " best=" + std::to_string(best_loss)
                + " lr_out=" + std::to_string(lr_out_ep)
                + " lr_kan=" + std::to_string(lr_kan_ep)
                + (train_kan ? " [KAN+W_a]" : " [W_a only]")
                + " (" + std::to_string(elapsed) + "s)");
        }

        if (avg_loss < 0.5) break;
    }

    auto total_train = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t_train).count();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0).count();

    log("");
    log("=== V12 CPU Training Complete ===");
    log("  Best loss: " + std::to_string(best_loss));
    log("  Training time: " + std::to_string(total_train) + "s");
    log("  Total wall time: " + std::to_string(total_elapsed) + "s");
    log("  Target: < 1.5 (beat V11 plateau of 1.848)");
    log("  Result: " + std::string(best_loss < 1.5 ? "TARGET HIT!" : "training in progress"));
    log("Log saved to: /tmp/brain19_v12_cpu.log");

    return 0;
}

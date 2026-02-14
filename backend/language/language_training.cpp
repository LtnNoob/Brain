#include "language_training.hpp"
#include "../memory/relation_type_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>

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

    auto& concept_emb_store = engine_.embeddings().concept_embeddings();
    auto& projection = engine_.fusion().projection();
    auto all_ids = ltm_.get_all_concept_ids();

    const size_t ACT_DIM = LanguageConfig::ENCODER_QUERY_DIM;  // 16
    const size_t FUSED = LanguageConfig::FUSED_DIM;             // 64

    for (auto cid : all_ids) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;

        // Get concept embedding (same hash-init R^16 used at inference)
        auto cemb = concept_emb_store.get_or_default(cid);

        // Build 57D raw vector matching FusionLayer format:
        // [slot1(16D) | slot2(16D) | slot3(16D) | gates(5D) | template_probs(4D)]
        // Single concept in slot 1, zeros in slots 2-3, DEFINITIONAL template
        std::vector<double> raw(3 * ACT_DIM + 5 + LanguageConfig::NUM_TEMPLATE_TYPES, 0.0);

        // Slot 1: concept embedding × gate weight
        for (size_t d = 0; d < ACT_DIM; ++d) {
            raw[d] = cemb.core[d] * 0.8;
        }
        // Slots 2-3: zeros (single concept)

        // Gates: [0.8, 0.0, 0.0, 0.0, 0.0]
        raw[3 * ACT_DIM] = 0.8;

        // Template probs: DEFINITIONAL dominant (index 1)
        size_t tpl_offset = 3 * ACT_DIM + 5;
        raw[tpl_offset + 1] = 0.8;  // DEFINITIONAL
        raw[tpl_offset + 0] = 0.2;  // small GENERAL

        // Project: raw × projection → R^64 (same as FusionLayer does)
        std::vector<double> fused(FUSED, 0.0);
        size_t raw_dim = std::min(raw.size(), projection.size());
        for (size_t i = 0; i < raw_dim; ++i) {
            if (std::abs(raw[i]) < 1e-12) continue;  // skip zeros
            for (size_t j = 0; j < FUSED; ++j) {
                fused[j] += raw[i] * projection[i][j];
            }
        }

        // Target: "Label is defined as definition."
        if (!info->definition.empty()) {
            std::string short_def = info->definition;
            if (short_def.size() > 80) short_def = short_def.substr(0, 80);
            pairs.push_back({fused, info->label + " is defined as " + short_def + "."});
        } else {
            pairs.push_back({fused, info->label + "."});
        }
    }

    // Subsample if over budget
    if (pairs.size() > LanguageConfig::MAX_DEFINITION_DECODER_PAIRS) {
        std::mt19937 rng(99);
        std::shuffle(pairs.begin(), pairs.end(), rng);
        pairs.resize(LanguageConfig::MAX_DEFINITION_DECODER_PAIRS);
    }

    return pairs;
}

// =============================================================================
// Helper: join a list of strings with Oxford comma
// =============================================================================

namespace {

std::string oxford_join(const std::vector<std::string>& items) {
    if (items.empty()) return "";
    if (items.size() == 1) return items[0];
    if (items.size() == 2) return items[0] + " and " + items[1];
    std::string result;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            result += (i == items.size() - 1) ? ", and " : ", ";
        }
        result += items[i];
    }
    return result;
}

// Ordered relation types for paragraph generation (most informative first)
static const std::vector<RelationType> PARAGRAPH_ORDER = {
    RelationType::IS_A,
    RelationType::INSTANCE_OF,
    RelationType::DERIVED_FROM,
    RelationType::HAS_PROPERTY,
    RelationType::PART_OF,
    RelationType::HAS_PART,
    RelationType::REQUIRES,
    RelationType::USES,
    RelationType::CAUSES,
    RelationType::ENABLES,
    RelationType::PRODUCES,
    RelationType::IMPLIES,
    RelationType::SUPPORTS,
    RelationType::CONTRADICTS,
    RelationType::SIMILAR_TO,
    RelationType::ASSOCIATED_WITH,
    RelationType::SOURCE,
    RelationType::TEMPORAL_BEFORE,
    RelationType::TEMPORAL_AFTER,
    RelationType::CUSTOM,
};

} // anonymous namespace

// =============================================================================
// Generate Relation-Based Decoder Data (compound paragraphs per concept)
// =============================================================================

std::vector<LanguageTraining::DecoderPair> LanguageTraining::generate_relation_decoder_data() const {
    std::vector<DecoderPair> pairs;

    auto& concept_emb_store = engine_.embeddings().concept_embeddings();
    auto& rel_registry = RelationTypeRegistry::instance();
    auto& projection = engine_.fusion().projection();
    auto all_ids = ltm_.get_all_concept_ids();

    const size_t ACT_DIM = LanguageConfig::ENCODER_QUERY_DIM;  // 16
    const size_t FUSED = LanguageConfig::FUSED_DIM;             // 64

    for (auto cid : all_ids) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;

        auto rels = ltm_.get_outgoing_relations(cid);
        if (rels.empty()) continue;

        // ── Group relations by type ──
        std::unordered_map<uint16_t, std::vector<std::string>> grouped;
        std::vector<FlexEmbedding> target_embeddings;
        std::vector<FlexEmbedding> rel_type_embeddings;

        for (const auto& rel : rels) {
            auto tgt_info = ltm_.retrieve_concept(rel.target);
            if (!tgt_info) continue;

            uint16_t type_key = static_cast<uint16_t>(rel.type);
            grouped[type_key].push_back(tgt_info->label);

            // Collect target and relation embeddings for the fused vector
            if (target_embeddings.size() < 5) {
                target_embeddings.push_back(concept_emb_store.get_or_default(rel.target));
            }
            if (rel_type_embeddings.size() < 5) {
                rel_type_embeddings.push_back(
                    engine_.embeddings().get_relation_embedding(rel.type));
            }
        }

        // ── Build compound paragraph ──
        std::string paragraph = info->label;
        bool first_sentence = true;
        size_t causal_count = 0;
        size_t hierarchical_count = 0;

        for (auto type : PARAGRAPH_ORDER) {
            uint16_t type_key = static_cast<uint16_t>(type);
            auto it = grouped.find(type_key);
            if (it == grouped.end() || it->second.empty()) continue;

            const auto& targets = it->second;
            const std::string& verb = rel_registry.get_name_en(type);
            auto category = rel_registry.get_category(type);

            if (category == RelationCategory::CAUSAL) causal_count += targets.size();
            if (category == RelationCategory::HIERARCHICAL) hierarchical_count += targets.size();

            std::string subject = first_sentence ? "" : " It ";
            std::string joined = oxford_join(targets);

            if (first_sentence) {
                // First clause directly follows the label
                paragraph += " " + verb + " " + joined + ".";
                first_sentence = false;
            } else {
                paragraph += " It " + verb + " " + joined + ".";
            }
        }

        // Skip concepts with only a label and no relation text generated
        if (first_sentence) continue;

        // ── Build 57D raw vector (matching FusionLayer format) ──
        // [slot1(16D) | slot2(16D) | slot3(16D) | gates(5D) | template_probs(4D)]
        std::vector<double> raw(3 * ACT_DIM + 5 + LanguageConfig::NUM_TEMPLATE_TYPES, 0.0);

        // Slot 1: source concept embedding × 0.7
        auto src_emb = concept_emb_store.get_or_default(cid);
        for (size_t d = 0; d < ACT_DIM; ++d) {
            raw[d] = src_emb.core[d] * 0.7;
        }

        // Slot 2: mean of target embeddings × 0.5
        if (!target_embeddings.empty()) {
            double inv_n = 0.5 / static_cast<double>(target_embeddings.size());
            for (const auto& temb : target_embeddings) {
                for (size_t d = 0; d < ACT_DIM; ++d) {
                    raw[ACT_DIM + d] += temb.core[d] * inv_n;
                }
            }
        }

        // Slot 3: mean of relation type embeddings × 0.3
        if (!rel_type_embeddings.empty()) {
            double inv_n = 0.3 / static_cast<double>(rel_type_embeddings.size());
            for (const auto& remb : rel_type_embeddings) {
                for (size_t d = 0; d < ACT_DIM; ++d) {
                    raw[2 * ACT_DIM + d] += remb.core[d] * inv_n;
                }
            }
        }

        // Gates: [0.8, 0.5, 0.3, 0.1, 0.0]
        size_t gate_offset = 3 * ACT_DIM;
        raw[gate_offset + 0] = 0.8;
        raw[gate_offset + 1] = 0.5;
        raw[gate_offset + 2] = 0.3;
        raw[gate_offset + 3] = 0.1;

        // Template probs: choose based on dominant relation category
        size_t tpl_offset = gate_offset + 5;
        if (causal_count > hierarchical_count && causal_count > 0) {
            // CAUSAL template (index 2)
            raw[tpl_offset + 2] = 0.6;
            raw[tpl_offset + 3] = 0.3;
            raw[tpl_offset + 0] = 0.1;
        } else if (hierarchical_count > 0) {
            // DEFINITIONAL template (index 1)
            raw[tpl_offset + 1] = 0.6;
            raw[tpl_offset + 0] = 0.2;
            raw[tpl_offset + 3] = 0.2;
        } else {
            // RELATIONAL template (index 3)
            raw[tpl_offset + 3] = 0.5;
            raw[tpl_offset + 0] = 0.3;
            raw[tpl_offset + 1] = 0.2;
        }

        // ── Project: raw × projection → R^64 ──
        std::vector<double> fused(FUSED, 0.0);
        size_t raw_dim = std::min(raw.size(), projection.size());
        for (size_t i = 0; i < raw_dim; ++i) {
            if (std::abs(raw[i]) < 1e-12) continue;
            for (size_t j = 0; j < FUSED; ++j) {
                fused[j] += raw[i] * projection[i][j];
            }
        }

        pairs.push_back({std::move(fused), std::move(paragraph)});
    }

    // Subsample if over budget
    if (pairs.size() > LanguageConfig::MAX_RELATION_DECODER_PAIRS) {
        std::mt19937 rng(42);
        std::shuffle(pairs.begin(), pairs.end(), rng);
        pairs.resize(LanguageConfig::MAX_RELATION_DECODER_PAIRS);
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
    std::cerr << "[LanguageTraining] Generating encoder data...\n";
    auto encoder_data = generate_encoder_data();
    std::cerr << "[LanguageTraining] Generating decoder data...\n";
    auto decoder_data = generate_decoder_data();
    std::cerr << "[LanguageTraining] Generating relation data...\n";
    auto relation_data = generate_relation_decoder_data();
    std::cerr << "[LanguageTraining] Data generated: enc=" << encoder_data.size()
              << " dec=" << decoder_data.size() << " rel=" << relation_data.size() << "\n";

    double best_loss = 1e9;

    // Train encoder (skip if no token-mapped concepts)
    if (!encoder_data.empty()) {
        std::cerr << "[LanguageTraining] Stage 1: Training encoder on "
                  << encoder_data.size() << " pairs...\n";

        for (size_t epoch = 0; epoch < config.encoder_epochs; ++epoch) {
            double loss = train_encoder_epoch(encoder_data, config.encoder_lr);
            if (loss < best_loss) best_loss = loss;
            if (loss < 1e-4) break;
            result.epochs_run = epoch + 1;
        }
        result.final_loss = best_loss;
    } else {
        std::cerr << "[LanguageTraining] Stage 1: Skipping encoder (no token-mapped data)\n";
    }

    // Merge definition + relation decoder data
    std::vector<DecoderPair> all_decoder_data;
    all_decoder_data.reserve(decoder_data.size() + relation_data.size());
    all_decoder_data.insert(all_decoder_data.end(),
        std::make_move_iterator(decoder_data.begin()),
        std::make_move_iterator(decoder_data.end()));
    all_decoder_data.insert(all_decoder_data.end(),
        std::make_move_iterator(relation_data.begin()),
        std::make_move_iterator(relation_data.end()));

    // Shuffle merged data
    {
        std::mt19937 rng(12345);
        std::shuffle(all_decoder_data.begin(), all_decoder_data.end(), rng);
    }

    // Train decoder on merged data
    if (!all_decoder_data.empty()) {
        std::cerr << "[LanguageTraining] Stage 1: Training decoder on "
                  << all_decoder_data.size() << " pairs ("
                  << decoder_data.size() << " def + "
                  << relation_data.size() << " rel)...\n";

        auto t_start = std::chrono::steady_clock::now();

        // ── Pre-tokenize ALL data ONCE (not per epoch) ──
        const size_t V = LanguageConfig::VOCAB_SIZE;
        auto& tok = engine_.tokenizer();
        std::vector<bool> seen(V, false);

        struct PreTokenizedSample {
            std::vector<uint16_t> tokens;
            size_t pair_idx;
        };
        std::vector<PreTokenizedSample> pretok_samples;
        pretok_samples.reserve(all_decoder_data.size());

        for (size_t idx = 0; idx < all_decoder_data.size(); ++idx) {
            auto tokens = tok.encode(all_decoder_data[idx].target_text);
            if (tokens.empty()) continue;
            for (auto t : tokens) {
                if (t < V) seen[t] = true;
            }
            pretok_samples.push_back({std::move(tokens), idx});
        }

        // Build compressed active vocab
        std::vector<uint16_t> active_tokens;
        active_tokens.reserve(256);
        std::vector<size_t> compress(V, 0);
        for (size_t v = 0; v < V; ++v) {
            if (seen[v]) {
                compress[v] = active_tokens.size();
                active_tokens.push_back(static_cast<uint16_t>(v));
            }
        }
        const size_t VA = active_tokens.size();

        engine_.decoder().set_trained_tokens(active_tokens);

        auto t_pretok = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();
        std::cerr << "[LanguageTraining]   Pre-tokenized " << pretok_samples.size()
                  << " samples, " << VA << " active tokens (" << t_pretok << "ms)\n";

        // ── Train with per-token CE-SGD using pre-tokenized data ──
        auto& decoder = engine_.decoder();
        auto& W = decoder.output_projection();
        auto& emb_table = engine_.encoder().embedding_table();

        const size_t H = LanguageConfig::FUSED_DIM;
        const size_t H_EXT = 2 * H;
        const double lr = config.decoder_lr;

        // Extract active columns of W into dense W_a[H_EXT][VA]
        std::vector<std::vector<double>> W_a(H_EXT, std::vector<double>(VA));
        for (size_t i = 0; i < H_EXT; ++i) {
            for (size_t a = 0; a < VA; ++a) {
                W_a[i][a] = W[i][active_tokens[a]];
            }
        }

        // Pre-allocate ALL working buffers outside loops
        std::vector<double> logits(VA);
        std::vector<double> probs(VA);
        std::vector<double> h(H);
        std::vector<double> h_ext(H_EXT);

        double best_decoder_loss = 1e9;
        for (size_t epoch = 0; epoch < config.decoder_epochs; ++epoch) {
            double total_ce_loss = 0.0;
            size_t total_tokens = 0;

            for (const auto& sample : pretok_samples) {
                const auto& target_tokens = sample.tokens;
                const auto& embedding = all_decoder_data[sample.pair_idx].embedding;

                // Reset h to fused vector
                std::fill(h.begin(), h.end(), 0.0);
                for (size_t i = 0; i < std::min(H, embedding.size()); ++i) {
                    h[i] = embedding[i];
                }

                for (size_t t = 0; t < target_tokens.size(); ++t) {
                    uint16_t target_tok = target_tokens[t];
                    if (target_tok >= V || !seen[target_tok]) continue;

                    size_t ca = compress[target_tok];

                    // Build quadratic features: h_ext = [h, h²]
                    for (size_t i = 0; i < H; ++i) {
                        h_ext[i] = h[i];
                        h_ext[H + i] = h[i] * h[i];
                    }

                    // logits = h_ext · W_a^T ∈ R^VA
                    for (size_t a = 0; a < VA; ++a) {
                        double sum = 0.0;
                        for (size_t i = 0; i < H_EXT; ++i) sum += h_ext[i] * W_a[i][a];
                        logits[a] = sum;
                    }

                    // Softmax
                    double max_val = *std::max_element(logits.begin(), logits.begin() + VA);
                    double exp_sum = 0.0;
                    for (size_t a = 0; a < VA; ++a) {
                        probs[a] = std::exp(std::min(logits[a] - max_val, 80.0));
                        exp_sum += probs[a];
                    }
                    if (exp_sum > 1e-12) {
                        double inv_sum = 1.0 / exp_sum;
                        for (size_t a = 0; a < VA; ++a) probs[a] *= inv_sum;
                    }

                    // CE loss
                    double p = std::max(probs[ca], 1e-12);
                    total_ce_loss += -std::log(p);
                    total_tokens++;

                    // Per-token W update
                    for (size_t i = 0; i < H_EXT; ++i) {
                        double hi = h_ext[i];
                        for (size_t a = 0; a < VA; ++a) {
                            W_a[i][a] -= lr * hi * probs[a];
                        }
                        W_a[i][ca] += lr * hi;
                    }

                    // Evolve hidden state
                    if (target_tok < emb_table.size()) {
                        const auto& tok_emb = emb_table[target_tok];
                        for (size_t i = 0; i < H && i < tok_emb.size(); ++i) {
                            h[i] = h[i] * 0.8 + tok_emb[i] * 0.2;
                        }
                    }
                }
            }

            double loss = (total_tokens > 0) ? total_ce_loss / total_tokens : 1e9;
            if (loss < best_decoder_loss) best_decoder_loss = loss;

            if ((epoch + 1) % 5 == 0 || epoch == 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t_start).count();
                std::cerr << "[LanguageTraining]   Epoch " << (epoch + 1)
                          << "/" << config.decoder_epochs
                          << " loss=" << loss
                          << " (" << elapsed / 1000 << "s)\n";
            }
            if (loss < 0.5) break;
        }

        // Write W_a back to full W
        for (size_t i = 0; i < H_EXT; ++i) {
            for (size_t a = 0; a < VA; ++a) {
                W[i][active_tokens[a]] = W_a[i][a];
            }
        }

        if (best_decoder_loss < best_loss) best_loss = best_decoder_loss;
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
// Closed-form ridge regression for output projection
// =============================================================================
//
// Instead of iterative SGD (slow convergence on linear model), solve:
//   W = (H^T H + λI)^{-1} H^T Y
// where H is [N × H_EXT] (all training h_ext vectors) and Y is one-hot targets.
// This gives the MSE-optimal W in one shot.
//

void LanguageTraining::train_decoder_closedform(
    const std::vector<DecoderPair>& data, double lambda) {

    auto& decoder = engine_.decoder();
    auto& tokenizer = engine_.tokenizer();
    auto& W = decoder.output_projection();
    auto& emb_table = engine_.encoder().embedding_table();

    const size_t H = LanguageConfig::FUSED_DIM;      // 64
    const size_t H_EXT = 2 * H;                       // 128
    const size_t V = LanguageConfig::VOCAB_SIZE;

    // ── Pre-tokenize and build active vocab ──
    struct TokenizedSample {
        std::vector<uint16_t> tokens;
        size_t data_idx;
    };
    std::vector<TokenizedSample> samples;
    std::vector<bool> token_active(V, false);

    for (size_t idx = 0; idx < data.size(); ++idx) {
        auto tokens = tokenizer.encode(data[idx].target_text);
        if (tokens.empty()) continue;
        for (auto t : tokens) {
            if (t < V) token_active[t] = true;
        }
        samples.push_back({std::move(tokens), idx});
    }
    if (samples.empty()) return;

    std::vector<uint16_t> active_tokens;
    std::vector<size_t> compress(V, 0);
    for (size_t v = 0; v < V; ++v) {
        if (token_active[v]) {
            compress[v] = active_tokens.size();
            active_tokens.push_back(static_cast<uint16_t>(v));
        }
    }
    const size_t VA = active_tokens.size();
    if (VA == 0) return;

    // ── Collect all (h_ext, target) pairs with hidden state evolution ──
    std::vector<std::vector<double>> all_h;   // N × H_EXT
    std::vector<size_t> all_targets;           // N
    all_h.reserve(data.size() * 15);
    all_targets.reserve(data.size() * 15);

    for (const auto& sample : samples) {
        const auto& target_tokens = sample.tokens;
        const auto& embedding = data[sample.data_idx].embedding;

        std::vector<double> h(H, 0.0);
        for (size_t i = 0; i < std::min(H, embedding.size()); ++i) {
            h[i] = embedding[i];
        }

        for (size_t t = 0; t < target_tokens.size(); ++t) {
            uint16_t target_tok = target_tokens[t];
            if (target_tok >= V || !token_active[target_tok]) continue;

            // Build h_ext = [h, h²]
            std::vector<double> h_ext(H_EXT);
            for (size_t i = 0; i < H; ++i) {
                h_ext[i] = h[i];
                h_ext[H + i] = h[i] * h[i];
            }
            all_h.push_back(std::move(h_ext));
            all_targets.push_back(compress[target_tok]);

            // Evolve hidden state (matches inference dynamics)
            if (target_tok < emb_table.size()) {
                const auto& tok_emb = emb_table[target_tok];
                for (size_t i = 0; i < H && i < tok_emb.size(); ++i) {
                    h[i] = h[i] * 0.8 + tok_emb[i] * 0.2;
                }
            }
        }
    }

    const size_t N = all_h.size();
    if (N == 0) return;

    std::cerr << "[LanguageTraining]   Ridge regression: " << N
              << " samples, " << H_EXT << "D features, " << VA << " active tokens\n";

    // ── Build C = H^T H + λI  [H_EXT × H_EXT] ──
    std::vector<std::vector<double>> C(H_EXT, std::vector<double>(H_EXT, 0.0));
    for (size_t n = 0; n < N; ++n) {
        const auto& hn = all_h[n];
        for (size_t i = 0; i < H_EXT; ++i) {
            double hi = hn[i];
            for (size_t j = i; j < H_EXT; ++j) {
                C[i][j] += hi * hn[j];
            }
        }
    }
    for (size_t i = 0; i < H_EXT; ++i) {
        for (size_t j = 0; j < i; ++j) C[i][j] = C[j][i];  // symmetrize
        C[i][i] += lambda;  // regularization
    }

    // ── Build B = H^T Y  [H_EXT × VA] ──
    // B[i][v] = Σ_{n: target[n]=v} h[n][i]
    // Scale targets so correct logit ≈ logit_scale instead of 1.0
    // MSE one-hot targets logit=1.0, but CE with VA tokens needs logit≈3-5
    // for reasonable probability: p = e^s / (VA-1+e^s) ≈ 30% for s=3, VA=63
    const double logit_scale = 4.0;
    std::vector<std::vector<double>> B(H_EXT, std::vector<double>(VA, 0.0));
    for (size_t n = 0; n < N; ++n) {
        size_t v = all_targets[n];
        const auto& hn = all_h[n];
        for (size_t i = 0; i < H_EXT; ++i) {
            B[i][v] += logit_scale * hn[i];
        }
    }

    // ── Invert C via Gauss-Jordan elimination ──
    // Augmented matrix [C | I] → [I | C^{-1}]
    std::vector<std::vector<double>> aug(H_EXT, std::vector<double>(2 * H_EXT, 0.0));
    for (size_t i = 0; i < H_EXT; ++i) {
        for (size_t j = 0; j < H_EXT; ++j) aug[i][j] = C[i][j];
        aug[i][H_EXT + i] = 1.0;
    }

    for (size_t k = 0; k < H_EXT; ++k) {
        // Partial pivot
        double max_val = std::abs(aug[k][k]);
        size_t max_row = k;
        for (size_t i = k + 1; i < H_EXT; ++i) {
            if (std::abs(aug[i][k]) > max_val) {
                max_val = std::abs(aug[i][k]);
                max_row = i;
            }
        }
        if (max_val < 1e-12) continue;
        if (max_row != k) std::swap(aug[k], aug[max_row]);

        double pivot = aug[k][k];
        for (size_t j = 0; j < 2 * H_EXT; ++j) aug[k][j] /= pivot;

        for (size_t i = 0; i < H_EXT; ++i) {
            if (i == k) continue;
            double factor = aug[i][k];
            for (size_t j = 0; j < 2 * H_EXT; ++j) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // ── W = C^{-1} · B  and write to output_projection ──
    for (size_t i = 0; i < H_EXT; ++i) {
        for (size_t v = 0; v < V; ++v) W[i][v] = 0.0;  // zero non-active

        for (size_t a = 0; a < VA; ++a) {
            double sum = 0.0;
            for (size_t j = 0; j < H_EXT; ++j) {
                sum += aug[i][H_EXT + j] * B[j][a];
            }
            W[i][active_tokens[a]] = sum;
        }
    }
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
    auto& tokenizer = engine_.tokenizer();

    auto& W = decoder.output_projection();    // [2*FUSED_DIM][VOCAB_SIZE] = [128][8192]
    auto& emb_table = engine_.encoder().embedding_table();  // token embeddings [V][64]

    const size_t H = LanguageConfig::FUSED_DIM;             // 64 (fused vector)
    const size_t H_EXT = 2 * H;                            // 128 (quadratic features [h, h²])
    const size_t V = LanguageConfig::VOCAB_SIZE;           // 8192

    // ── Pre-tokenize all samples and build active vocab ──
    // Active-vocab optimization: only compute logits/gradients for tokens
    // that appear in training data (~500 tokens vs 8192 → ~16x speedup)
    struct TokenizedSample {
        std::vector<uint16_t> tokens;
        size_t data_idx;
    };
    std::vector<TokenizedSample> samples;
    std::vector<bool> token_active(V, false);

    for (size_t idx = 0; idx < data.size(); ++idx) {
        auto tokens = tokenizer.encode(data[idx].target_text);
        if (tokens.empty()) continue;
        for (auto t : tokens) {
            if (t < V) token_active[t] = true;
        }
        samples.push_back({std::move(tokens), idx});
    }

    if (samples.empty()) return 1e9;

    // Build compressed vocab: full token ID → compressed index
    std::vector<uint16_t> active_tokens;
    active_tokens.reserve(1024);
    std::vector<size_t> compress(V, 0);  // compress[full_id] = compressed_idx
    for (size_t v = 0; v < V; ++v) {
        if (token_active[v]) {
            compress[v] = active_tokens.size();
            active_tokens.push_back(static_cast<uint16_t>(v));
        }
    }
    const size_t VA = active_tokens.size();
    if (VA == 0) return 1e9;

    // Extract active columns of W into dense W_a[H_EXT][VA]
    std::vector<std::vector<double>> W_a(H_EXT, std::vector<double>(VA));
    for (size_t i = 0; i < H_EXT; ++i) {
        for (size_t a = 0; a < VA; ++a) {
            W_a[i][a] = W[i][active_tokens[a]];
        }
    }

    double total_ce_loss = 0.0;
    size_t total_tokens = 0;

    // Pre-allocate working buffers
    std::vector<double> logits(VA);
    std::vector<double> probs(VA);

    // ── Per-token SGD: update W after EACH token position ──
    // Per-sample accumulation causes gradient cancellation (similar h at adjacent
    // positions push toward different targets). Per-token updates eliminate this.
    //
    // Hidden state evolves: h₀ = fused[0:H], h_{t+1} = h_t * 0.8 + embed(tok)[0:H] * 0.2
    // Stronger mixing (0.8/0.2) gives more position diversity than 0.9/0.1.
    for (const auto& sample : samples) {
        const auto& target_tokens = sample.tokens;
        const auto& embedding = data[sample.data_idx].embedding;

        // h = full fused vector (64D, unique per concept)
        std::vector<double> h(H, 0.0);
        for (size_t i = 0; i < std::min(H, embedding.size()); ++i) {
            h[i] = embedding[i];
        }

        for (size_t t = 0; t < target_tokens.size(); ++t) {
            uint16_t target_tok = target_tokens[t];
            if (target_tok >= V || !token_active[target_tok]) continue;

            size_t ca = compress[target_tok];

            // Build quadratic features: h_ext = [h, h²]
            std::vector<double> h_ext(H_EXT);
            for (size_t i = 0; i < H; ++i) {
                h_ext[i] = h[i];
                h_ext[H + i] = h[i] * h[i];
            }

            // logits = h_ext · W_a^T ∈ R^VA
            for (size_t a = 0; a < VA; ++a) {
                double sum = 0.0;
                for (size_t i = 0; i < H_EXT; ++i) sum += h_ext[i] * W_a[i][a];
                logits[a] = sum;
            }

            // Softmax
            double max_val = *std::max_element(logits.begin(), logits.begin() + VA);
            double exp_sum = 0.0;
            for (size_t a = 0; a < VA; ++a) {
                probs[a] = std::exp(std::min(logits[a] - max_val, 80.0));
                exp_sum += probs[a];
            }
            if (exp_sum > 1e-12) {
                double inv_sum = 1.0 / exp_sum;
                for (size_t a = 0; a < VA; ++a) probs[a] *= inv_sum;
            }

            // CE loss
            double p = std::max(probs[ca], 1e-12);
            total_ce_loss += -std::log(p);
            total_tokens++;

            // Per-token W update: W_a -= lr * h_ext * (probs - one_hot_target)
            for (size_t i = 0; i < H_EXT; ++i) {
                double hi = h_ext[i];
                for (size_t a = 0; a < VA; ++a) {
                    W_a[i][a] -= lr * hi * probs[a];
                }
                W_a[i][ca] += lr * hi;
            }

            // Evolve hidden state for next position (matches inference dynamics)
            // h_{t+1} = h_t * 0.8 + embed(token_t)[0:H] * 0.2
            if (target_tok < emb_table.size()) {
                const auto& tok_emb = emb_table[target_tok];
                for (size_t i = 0; i < H && i < tok_emb.size(); ++i) {
                    h[i] = h[i] * 0.8 + tok_emb[i] * 0.2;
                }
            }
        }
    }

    if (total_tokens == 0) return 1e9;

    // Write compressed W_a back to full W
    for (size_t i = 0; i < H_EXT; ++i) {
        for (size_t a = 0; a < VA; ++a) {
            W[i][active_tokens[a]] = W_a[i][a];
        }
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

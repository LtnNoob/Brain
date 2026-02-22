#include "language_training.hpp"
#include "deep_kan.hpp"
#include "../memory/relation_type_registry.hpp"
#include "../cmodel/concept_model.hpp"
#include "../convergence/convergence_config.hpp"
#include "../cuda/cuda_ops.h"
#include "../cuda/cuda_training.h"
#ifdef USE_LIBTORCH
#include "../libtorch/torch_training.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

LanguageTraining::LanguageTraining(KANLanguageEngine& engine, LongTermMemory& ltm,
                                   ConceptModelRegistry& registry)
    : engine_(engine)
    , ltm_(ltm)
    , registry_(registry)
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

// GAT: attention-weighted aggregation of target embeddings using source as query.
// Replaces simple mean with dot-product attention: relevant neighbors get higher weight.
// Degrades gracefully to uniform weights when all dot products are similar.
static void gat_aggregate_targets(
    const FlexEmbedding& src_emb,
    const std::vector<FlexEmbedding>& target_embeddings,
    double* raw_slot,   // output: raw[ACT_DIM..2*ACT_DIM]
    size_t ACT_DIM,
    double scale = 0.5)
{
    if (target_embeddings.empty()) return;

    // Compute dot-product attention weights
    std::vector<double> attn(target_embeddings.size());
    double attn_max = -1e9;
    for (size_t j = 0; j < target_embeddings.size(); ++j) {
        double dot = 0;
        for (size_t d = 0; d < ACT_DIM && d < src_emb.core.size(); ++d)
            dot += src_emb.core[d] * target_embeddings[j].core[d];
        attn[j] = dot * 4.0;  // temperature-scaled
        if (attn[j] > attn_max) attn_max = attn[j];
    }
    // Stable softmax
    double attn_sum = 0;
    for (auto& a : attn) {
        a = std::exp(std::min(a - attn_max, 80.0));
        attn_sum += a;
    }
    if (attn_sum > 1e-12)
        for (auto& a : attn) a /= attn_sum;

    // Attention-weighted aggregation
    for (size_t j = 0; j < target_embeddings.size(); ++j)
        for (size_t d = 0; d < ACT_DIM; ++d)
            raw_slot[d] += target_embeddings[j].core[d] * attn[j] * scale;
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
    // SIMILAR_TO and ASSOCIATED_WITH excluded — too vague, adds noise to concept prediction
    RelationType::SOURCE,
    RelationType::TEMPORAL_BEFORE,
    RelationType::TEMPORAL_AFTER,
    RelationType::CUSTOM,
};

} // anonymous namespace

std::vector<LanguageTraining::DecoderPair> LanguageTraining::generate_decoder_data() const {
    // Unified concept descriptions: definition + ALL relations as one paragraph.
    // Each concept becomes a rich object description:
    // "Dog is a mammal and omnivore. It is a loyal pet. It eats meat and bones."
    std::vector<DecoderPair> pairs;

    auto& concept_emb_store = engine_.embeddings().concept_embeddings();
    auto& rel_registry = RelationTypeRegistry::instance();
    auto& projection = engine_.fusion().projection();
    auto& dim_ctx = engine_.dim_context();
    auto all_ids = ltm_.get_all_concept_ids();

    const size_t ACT_DIM = LanguageConfig::ENCODER_QUERY_DIM;  // 16
    const size_t FUSED = LanguageConfig::FUSED_DIM;             // 64

    // Quality filter: count relations per concept
    std::unordered_map<uint32_t, size_t> rel_count;
    for (auto cid : all_ids)
        rel_count[cid] = ltm_.get_outgoing_relations(cid).size()
                       + ltm_.get_incoming_relations(cid).size();

    size_t filtered = 0;
    for (auto cid : all_ids) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;

        // Quality gate: require non-empty definition AND 1+ relations
        if (info->definition.empty() || rel_count[cid] < 1) {
            filtered++;
            continue;
        }

        auto rels = ltm_.get_outgoing_relations(cid);

        // ── Group relations by type ──
        std::unordered_map<uint16_t, std::vector<std::string>> grouped;
        std::vector<FlexEmbedding> target_embeddings;
        std::vector<FlexEmbedding> rel_type_embeddings;

        for (const auto& rel : rels) {
            auto tgt_info = ltm_.retrieve_concept(rel.target);
            if (!tgt_info) continue;
            grouped[static_cast<uint16_t>(rel.type)].push_back(tgt_info->label);
            if (target_embeddings.size() < 5)
                target_embeddings.push_back(concept_emb_store.get_or_default(rel.target));
            if (rel_type_embeddings.size() < 5)
                rel_type_embeddings.push_back(
                    engine_.embeddings().get_relation_embedding(rel.type));
        }

        // ── Build unified paragraph: definition + all relations ──
        std::string paragraph = info->label;
        bool first_sentence = true;
        size_t causal_count = 0, hierarchical_count = 0;

        // Start with hierarchy relations (IS_A, INSTANCE_OF first)
        for (auto type : PARAGRAPH_ORDER) {
            uint16_t type_key = static_cast<uint16_t>(type);
            auto it = grouped.find(type_key);
            if (it == grouped.end() || it->second.empty()) continue;

            const auto& targets = it->second;
            const std::string& verb = rel_registry.get_name_en(type);
            auto category = rel_registry.get_category(type);
            if (category == RelationCategory::CAUSAL) causal_count += targets.size();
            if (category == RelationCategory::HIERARCHICAL) hierarchical_count += targets.size();

            std::string joined = oxford_join(targets);
            if (first_sentence) {
                paragraph += " " + verb + " " + joined + ".";
                first_sentence = false;

                // Insert definition after first relation sentence
                std::string short_def = info->definition;
                if (short_def.size() > 80) short_def = short_def.substr(0, 80);
                paragraph += " It is " + short_def + ".";
            } else {
                paragraph += " It " + verb + " " + joined + ".";
            }
        }

        // If no relations were added, just use definition
        if (first_sentence) {
            std::string short_def = info->definition;
            if (short_def.size() > 80) short_def = short_def.substr(0, 80);
            paragraph += " is " + short_def + ".";
        }

        // ── Build fused vector (multi-slot: source + targets + relation types) ──
        std::vector<double> raw(3 * ACT_DIM + 5 + LanguageConfig::NUM_TEMPLATE_TYPES, 0.0);

        auto src_emb = concept_emb_store.get_or_default(cid);
        for (size_t d = 0; d < ACT_DIM; ++d)
            raw[d] = src_emb.core[d] * 0.7;

        if (!target_embeddings.empty()) {
            if (use_gat_) {
                gat_aggregate_targets(src_emb, target_embeddings, &raw[ACT_DIM], ACT_DIM, 0.5);
            } else {
                double inv_n = 0.5 / static_cast<double>(target_embeddings.size());
                for (const auto& temb : target_embeddings)
                    for (size_t d = 0; d < ACT_DIM; ++d)
                        raw[ACT_DIM + d] += temb.core[d] * inv_n;
            }
        }

        if (!rel_type_embeddings.empty()) {
            double inv_n = 0.3 / static_cast<double>(rel_type_embeddings.size());
            for (const auto& remb : rel_type_embeddings)
                for (size_t d = 0; d < ACT_DIM; ++d)
                    raw[2 * ACT_DIM + d] += remb.core[d] * inv_n;
        }

        size_t gate_offset = 3 * ACT_DIM;
        raw[gate_offset + 0] = 0.8;
        raw[gate_offset + 1] = 0.5;
        raw[gate_offset + 2] = 0.3;

        size_t tpl_offset = gate_offset + 5;
        if (causal_count > hierarchical_count && causal_count > 0) {
            raw[tpl_offset + 2] = 0.5; raw[tpl_offset + 1] = 0.3; raw[tpl_offset + 0] = 0.2;
        } else if (hierarchical_count > 0) {
            raw[tpl_offset + 1] = 0.6; raw[tpl_offset + 0] = 0.2; raw[tpl_offset + 3] = 0.2;
        } else {
            raw[tpl_offset + 3] = 0.4; raw[tpl_offset + 0] = 0.3; raw[tpl_offset + 1] = 0.3;
        }

        // Project: raw x projection -> R^64
        std::vector<double> fused(FUSED, 0.0);
        size_t raw_dim = std::min(raw.size(), projection.size());
        for (size_t i = 0; i < raw_dim; ++i) {
            if (std::abs(raw[i]) < 1e-12) continue;
            for (size_t j = 0; j < FUSED; ++j)
                fused[j] += raw[i] * projection[i][j];
        }

        // v11: Insert 16D FlexEmbedding detail vector between fused(64D) and dim_context
        {
            auto flex_emb = concept_emb_store.get_or_default(cid);
            size_t detail_dims = std::min(flex_emb.detail.size(), size_t(16));
            for (size_t d = 0; d < 16; ++d) {
                fused.push_back(d < detail_dims ? flex_emb.detail[d] : 0.0);
            }
        }

        if (dim_ctx.is_built()) {
            auto dim_vec = dim_ctx.to_decoder_vec(cid);
            fused.insert(fused.end(), dim_vec.begin(), dim_vec.end());
        }

        // Append ConvergencePort output (32D) for source concept
        if (registry_.has_model(cid)) {
            auto* cm = registry_.get_model(cid);
            std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
            for (size_t i = 0; i < std::min(fused.size(), size_t(convergence::QUERY_DIM)); ++i)
                conv_input[i] = fused[i];
            double conv_out[ConvergencePort::OUTPUT_DIM];
            cm->forward_convergence(conv_input.data(), conv_out);
            for (size_t d = 0; d < ConvergencePort::OUTPUT_DIM; ++d)
                fused.push_back(conv_out[d]);
        } else {
            fused.resize(fused.size() + ConvergencePort::OUTPUT_DIM, 0.0);
        }

        pairs.push_back({std::move(fused), std::move(paragraph)});
    }

    std::cerr << "[LanguageTraining] Decoder: " << pairs.size()
              << " quality concepts (" << filtered << " filtered)\n";

    // Subsample if over budget
    if (pairs.size() > LanguageConfig::MAX_DEFINITION_DECODER_PAIRS) {
        std::mt19937 rng(99);
        std::shuffle(pairs.begin(), pairs.end(), rng);
        pairs.resize(LanguageConfig::MAX_DEFINITION_DECODER_PAIRS);
    }

    return pairs;
}

// =============================================================================
// Generate Supplemental Relation-Based Data (concepts not covered by unified)
// =============================================================================

std::vector<LanguageTraining::DecoderPair> LanguageTraining::generate_relation_decoder_data() const {
    std::vector<DecoderPair> pairs;

    auto& concept_emb_store = engine_.embeddings().concept_embeddings();
    auto& rel_registry = RelationTypeRegistry::instance();
    auto& projection = engine_.fusion().projection();
    auto& dim_ctx = engine_.dim_context();
    auto all_ids = ltm_.get_all_concept_ids();

    const size_t ACT_DIM = LanguageConfig::ENCODER_QUERY_DIM;  // 16
    const size_t FUSED = LanguageConfig::FUSED_DIM;             // 64

    // Quality filter: count total relations per concept
    std::unordered_map<uint32_t, size_t> total_rel_count;
    for (auto cid : all_ids)
        total_rel_count[cid] = ltm_.get_outgoing_relations(cid).size()
                             + ltm_.get_incoming_relations(cid).size();

    size_t filtered_rel = 0;
    for (auto cid : all_ids) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;

        // Quality gate: require non-empty definition AND 1+ relations
        if (info->definition.empty() || total_rel_count[cid] < 1) {
            filtered_rel++;
            continue;
        }

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

        // Slot 1: source concept embedding x 0.7
        auto src_emb = concept_emb_store.get_or_default(cid);
        for (size_t d = 0; d < ACT_DIM; ++d) {
            raw[d] = src_emb.core[d] * 0.7;
        }

        // Slot 2: target embedding aggregation (GAT attention or mean) x 0.5
        if (!target_embeddings.empty()) {
            if (use_gat_) {
                gat_aggregate_targets(src_emb, target_embeddings, &raw[ACT_DIM], ACT_DIM, 0.5);
            } else {
                double inv_n = 0.5 / static_cast<double>(target_embeddings.size());
                for (const auto& temb : target_embeddings) {
                    for (size_t d = 0; d < ACT_DIM; ++d) {
                        raw[ACT_DIM + d] += temb.core[d] * inv_n;
                    }
                }
            }
        }

        // Slot 3: mean of relation type embeddings x 0.3
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

        // ── Project: raw x projection -> R^64 ──
        std::vector<double> fused(FUSED, 0.0);
        size_t raw_dim = std::min(raw.size(), projection.size());
        for (size_t i = 0; i < raw_dim; ++i) {
            if (std::abs(raw[i]) < 1e-12) continue;
            for (size_t j = 0; j < FUSED; ++j) {
                fused[j] += raw[i] * projection[i][j];
            }
        }

        // v11: Insert 16D FlexEmbedding detail vector between fused(64D) and dim_context
        {
            auto flex_emb = concept_emb_store.get_or_default(cid);
            size_t detail_dims = std::min(flex_emb.detail.size(), size_t(16));
            for (size_t d = 0; d < 16; ++d) {
                fused.push_back(d < detail_dims ? flex_emb.detail[d] : 0.0);
            }
        }

        // Append dimensional context for source concept (variable-length)
        if (dim_ctx.is_built()) {
            auto dim_vec = dim_ctx.to_decoder_vec(cid);
            fused.insert(fused.end(), dim_vec.begin(), dim_vec.end());
        }

        // Append ConvergencePort output (32D) for source concept
        if (registry_.has_model(cid)) {
            auto* cm = registry_.get_model(cid);
            std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
            for (size_t i = 0; i < std::min(fused.size(), size_t(convergence::QUERY_DIM)); ++i)
                conv_input[i] = fused[i];
            double conv_out[ConvergencePort::OUTPUT_DIM];
            cm->forward_convergence(conv_input.data(), conv_out);
            for (size_t d = 0; d < ConvergencePort::OUTPUT_DIM; ++d)
                fused.push_back(conv_out[d]);
        } else {
            fused.resize(fused.size() + ConvergencePort::OUTPUT_DIM, 0.0);
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
// Generate Concept Decoder Training Data
// =============================================================================

// Helper: build fused embedding vector for a source concept with specific targets/relations
std::vector<double> LanguageTraining::build_concept_fused_vector(
    ConceptId source,
    const std::vector<FlexEmbedding>& target_embeddings,
    const std::vector<FlexEmbedding>& rel_type_embeddings,
    const std::vector<RelationType>& rel_types) const
{
    auto& concept_emb_store = engine_.embeddings().concept_embeddings();
    auto& projection = engine_.fusion().projection();
    auto& dim_ctx = engine_.dim_context();
    const size_t ACT_DIM = LanguageConfig::ENCODER_QUERY_DIM;  // 16
    const size_t FUSED = LanguageConfig::FUSED_DIM;             // 64

    std::vector<double> raw(3 * ACT_DIM + 5 + LanguageConfig::NUM_TEMPLATE_TYPES, 0.0);

    auto src_emb = concept_emb_store.get_or_default(source);
    for (size_t d = 0; d < ACT_DIM; ++d)
        raw[d] = src_emb.core[d] * 0.7;

    if (!target_embeddings.empty()) {
        if (use_gat_) {
            gat_aggregate_targets(src_emb, target_embeddings, &raw[ACT_DIM], ACT_DIM, 0.5);
        } else {
            double inv_n = 0.5 / static_cast<double>(target_embeddings.size());
            for (const auto& temb : target_embeddings)
                for (size_t d = 0; d < ACT_DIM; ++d)
                    raw[ACT_DIM + d] += temb.core[d] * inv_n;
        }
    }

    if (!rel_type_embeddings.empty()) {
        double inv_n = 0.3 / static_cast<double>(rel_type_embeddings.size());
        for (const auto& remb : rel_type_embeddings)
            for (size_t d = 0; d < ACT_DIM; ++d)
                raw[2 * ACT_DIM + d] += remb.core[d] * inv_n;
    }

    size_t gate_offset = 3 * ACT_DIM;
    raw[gate_offset + 0] = 0.8;
    raw[gate_offset + 1] = 0.5;
    raw[gate_offset + 2] = 0.3;

    // Template slots: encode relation type distribution as graph structure signal.
    // [0] hierarchical (IS_A, INSTANCE_OF, DERIVED_FROM)
    // [1] causal (CAUSES, ENABLES, PRODUCES, IMPLIES)
    // [2] compositional (HAS_PROPERTY, PART_OF, HAS_PART, REQUIRES, USES)
    // [3] other
    size_t tpl_offset = gate_offset + 5;
    if (!rel_types.empty()) {
        double inv_n = 1.0 / static_cast<double>(rel_types.size());
        for (auto rt : rel_types) {
            switch (rt) {
                case RelationType::IS_A:
                case RelationType::INSTANCE_OF:
                case RelationType::DERIVED_FROM:
                    raw[tpl_offset + 0] += inv_n;
                    break;
                case RelationType::CAUSES:
                case RelationType::ENABLES:
                case RelationType::PRODUCES:
                case RelationType::IMPLIES:
                    raw[tpl_offset + 1] += inv_n;
                    break;
                case RelationType::HAS_PROPERTY:
                case RelationType::PART_OF:
                case RelationType::HAS_PART:
                case RelationType::REQUIRES:
                case RelationType::USES:
                    raw[tpl_offset + 2] += inv_n;
                    break;
                default:
                    raw[tpl_offset + 3] += inv_n;
                    break;
            }
        }
    } else {
        // Fallback for callers without relation types
        raw[tpl_offset + 0] = 0.3;
        raw[tpl_offset + 1] = 0.5;
    }

    // Project: raw x projection -> R^64
    std::vector<double> fused(FUSED, 0.0);
    size_t raw_dim = std::min(raw.size(), projection.size());
    for (size_t i = 0; i < raw_dim; ++i) {
        if (std::abs(raw[i]) < 1e-12) continue;
        for (size_t j = 0; j < FUSED; ++j)
            fused[j] += raw[i] * projection[i][j];
    }

    // v11: FlexEmbedding detail
    {
        size_t detail_dims = std::min(src_emb.detail.size(), size_t(16));
        for (size_t d = 0; d < 16; ++d)
            fused.push_back(d < detail_dims ? src_emb.detail[d] : 0.0);
    }

    // Dimensional context
    if (dim_ctx.is_built()) {
        auto dim_vec = dim_ctx.to_decoder_vec(source);
        fused.insert(fused.end(), dim_vec.begin(), dim_vec.end());
    }

    // ConvergencePort output
    if (registry_.has_model(source)) {
        auto* cm = registry_.get_model(source);
        std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
        for (size_t i = 0; i < std::min(fused.size(), size_t(convergence::QUERY_DIM)); ++i)
            conv_input[i] = fused[i];
        double conv_out[ConvergencePort::OUTPUT_DIM];
        cm->forward_convergence(conv_input.data(), conv_out);
        for (size_t d = 0; d < ConvergencePort::OUTPUT_DIM; ++d)
            fused.push_back(conv_out[d]);
    } else {
        fused.resize(fused.size() + ConvergencePort::OUTPUT_DIM, 0.0);
    }

    return fused;
}

// Helper: check if concept is valid for training
static bool is_valid_concept(const ConceptInfo& info) {
    return !info.is_anti_knowledge &&
           info.epistemic.status != EpistemicStatus::INVALIDATED;
}

std::vector<LanguageTraining::ConceptDecoderPair>
LanguageTraining::generate_concept_decoder_data() const {
    std::vector<ConceptDecoderPair> pairs;

    auto& concept_emb_store = engine_.embeddings().concept_embeddings();
    auto all_ids = ltm_.get_all_concept_ids();

    size_t filtered = 0, anti_knowledge = 0, invalidated = 0;
    size_t forward_pairs = 0, inherited_pairs = 0, noise_pairs = 0;

    // Inheritable relation types for IS_A chain walking
    static const std::vector<RelationType> INHERITABLE_TYPES = {
        RelationType::HAS_PROPERTY,
        RelationType::REQUIRES,
        RelationType::USES,
        RelationType::CAUSES,
        RelationType::ENABLES,
        RelationType::PRODUCES,
    };
    static constexpr size_t MAX_ISA_HOPS = 3;
    static constexpr double ISA_TRUST_DECAY = 0.7;  // per hop

    // ── Forward full-sequence pairs (one per concept with outgoing relations) ──
    for (auto cid : all_ids) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;
        if (info->is_anti_knowledge) { anti_knowledge++; continue; }
        if (info->epistemic.status == EpistemicStatus::INVALIDATED) { invalidated++; continue; }

        auto rels = ltm_.get_outgoing_relations(cid);
        if (rels.empty()) { filtered++; continue; }

        // Build target concept sequence ordered by PARAGRAPH_ORDER
        // Skip SIMILAR_TO and ASSOCIATED_WITH (noise — these are too vague for concept prediction)
        std::vector<ConceptId> full_targets;
        for (auto type : PARAGRAPH_ORDER) {
            if (type == RelationType::SIMILAR_TO || type == RelationType::ASSOCIATED_WITH) continue;
            for (const auto& rel : rels) {
                if (rel.type != type) continue;
                auto tgt_info = ltm_.retrieve_concept(rel.target);
                if (!tgt_info || !is_valid_concept(*tgt_info)) continue;
                bool dup = false;
                for (auto tc : full_targets)
                    if (tc == rel.target) { dup = true; break; }
                if (!dup) {
                    full_targets.push_back(rel.target);
                    if (full_targets.size() >= LanguageConfig::MAX_CONCEPT_SEQUENCE) break;
                }
            }
            if (full_targets.size() >= LanguageConfig::MAX_CONCEPT_SEQUENCE) break;
        }

        if (full_targets.empty()) { filtered++; continue; }

        // Collect embeddings and relation types for the fused vector
        std::vector<FlexEmbedding> tgt_embs, rel_embs;
        std::vector<RelationType> rel_types;
        for (const auto& rel : rels) {
            auto ti = ltm_.retrieve_concept(rel.target);
            if (!ti) continue;
            if (tgt_embs.size() < 5)
                tgt_embs.push_back(concept_emb_store.get_or_default(rel.target));
            if (rel_embs.size() < 5)
                rel_embs.push_back(engine_.embeddings().get_relation_embedding(rel.type));
            rel_types.push_back(rel.type);
        }

        auto fused = build_concept_fused_vector(cid, tgt_embs, rel_embs, rel_types);

        ConceptDecoderPair p;
        p.embedding = std::move(fused);
        p.target_concepts = std::move(full_targets);
        p.trust_weight = info->epistemic.trust;
        p.source_concept = cid;
        pairs.push_back(std::move(p));
        forward_pairs++;
    }

    // ── IS_A inheritance pairs: walk up IS_A chain, inherit ancestor properties ──
    // For each concept C with IS_A parent P, create a training pair:
    //   source=C, targets=P's inheritable targets, trust decayed by hop distance.
    // The fused vector uses P's targets as context, teaching C to predict
    // inherited properties (e.g., Water inherits Liquid's HAS_PROPERTY targets).
    for (auto cid : all_ids) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info || !is_valid_concept(*info)) continue;

        auto rels = ltm_.get_outgoing_relations(cid);

        // Walk IS_A chain
        ConceptId current = cid;
        double trust_decay = 1.0;
        for (size_t hop = 0; hop < MAX_ISA_HOPS; ++hop) {
            // Find first IS_A parent
            ConceptId parent = ConceptId{0};
            for (const auto& rel : ltm_.get_outgoing_relations(current)) {
                if (rel.type == RelationType::IS_A) {
                    parent = rel.target;
                    break;
                }
            }
            if (parent == ConceptId{0}) break;

            auto parent_info = ltm_.retrieve_concept(parent);
            if (!parent_info || !is_valid_concept(*parent_info)) break;

            trust_decay *= ISA_TRUST_DECAY;

            // Collect parent's inheritable targets (skip if C already has them)
            auto parent_rels = ltm_.get_outgoing_relations(parent);
            std::vector<ConceptId> inherited_targets;
            std::vector<FlexEmbedding> inh_tgt_embs, inh_rel_embs;
            std::vector<RelationType> inh_rel_types;

            for (const auto& prel : parent_rels) {
                // Check if relation type is inheritable
                bool inheritable = false;
                for (auto itype : INHERITABLE_TYPES)
                    if (prel.type == itype) { inheritable = true; break; }
                if (!inheritable) continue;

                auto tgt_info = ltm_.retrieve_concept(prel.target);
                if (!tgt_info || !is_valid_concept(*tgt_info)) continue;

                // Skip if C already has this target directly
                bool already_direct = false;
                for (const auto& crel : rels)
                    if (crel.target == prel.target) { already_direct = true; break; }
                if (already_direct) continue;

                // Avoid duplicates
                bool dup = false;
                for (auto tc : inherited_targets)
                    if (tc == prel.target) { dup = true; break; }
                if (dup) continue;

                inherited_targets.push_back(prel.target);
                if (inh_tgt_embs.size() < 5)
                    inh_tgt_embs.push_back(concept_emb_store.get_or_default(prel.target));
                if (inh_rel_embs.size() < 5)
                    inh_rel_embs.push_back(engine_.embeddings().get_relation_embedding(prel.type));
                inh_rel_types.push_back(prel.type);

                if (inherited_targets.size() >= LanguageConfig::MAX_CONCEPT_SEQUENCE) break;
            }

            if (!inherited_targets.empty()) {
                // Fused vector: source=C, context=parent's targets
                auto fused = build_concept_fused_vector(cid, inh_tgt_embs, inh_rel_embs, inh_rel_types);

                ConceptDecoderPair p;
                p.embedding = std::move(fused);
                p.target_concepts = std::move(inherited_targets);
                p.trust_weight = info->epistemic.trust * trust_decay;
                p.source_concept = cid;
                pairs.push_back(std::move(p));
                inherited_pairs++;
            }

            current = parent;
        }
    }

    // ── Noise-augmented copies (safe: same targets, slightly perturbed input) ──
    {
        std::mt19937 noise_rng(42424);
        std::normal_distribution<double> noise_dist(0.0, 0.02);
        size_t base_size = pairs.size();

        // Create 1 noisy copy per pair (2x data total)
        for (int copy = 0; copy < 1; ++copy) {
            for (size_t i = 0; i < base_size; ++i) {
                if (pairs[i].trust_weight < 0.3) continue;  // skip very low trust

                ConceptDecoderPair aug;
                aug.embedding.resize(pairs[i].embedding.size());
                for (size_t d = 0; d < pairs[i].embedding.size(); ++d)
                    aug.embedding[d] = pairs[i].embedding[d] + noise_dist(noise_rng);
                aug.target_concepts = pairs[i].target_concepts;
                aug.trust_weight = pairs[i].trust_weight * 0.7;  // moderate downweight
                aug.source_concept = pairs[i].source_concept;
                pairs.push_back(std::move(aug));
                noise_pairs++;
            }
        }
    }

    std::cerr << "[LanguageTraining] Concept decoder data: " << pairs.size()
              << " total pairs (forward=" << forward_pairs
              << ", inherited=" << inherited_pairs
              << ", noise=" << noise_pairs
              << ", filtered=" << filtered
              << ", anti_knowledge=" << anti_knowledge
              << ", invalidated=" << invalidated << ")\n";

    return pairs;
}

// =============================================================================
// Concept Decoder: Closed-Form Ridge Regression
// =============================================================================
//
// Solve: W = (H^T H + λI)^{-1} H^T E
// where H is [N × H_dim] hidden vectors, E is [N × 16] target embeddings.
// This gives MSE-optimal W_concept in one shot.
//

void LanguageTraining::train_concept_decoder_closedform(
    const std::vector<ConceptDecoderPair>& data, double lambda) {

    auto& decoder = engine_.decoder();
    auto& concept_emb_store = engine_.embeddings().concept_embeddings();

    const size_t H = decoder.extended_fused_dim();
    const size_t D = LanguageConfig::CONCEPT_EMBED_DIM;  // 16

    // Collect all (h_transformed, target_embedding) pairs
    // For each sample, we take h_transformed from the initial hidden state
    // and use the first target concept's embedding as the regression target.
    std::vector<std::vector<double>> all_h;
    std::vector<std::vector<double>> all_e;
    all_h.reserve(data.size());
    all_e.reserve(data.size());

    for (const auto& pair : data) {
        if (pair.target_concepts.empty()) continue;

        // Hidden state = the fused vector itself (before transform)
        std::vector<double> h(H, 0.0);
        for (size_t i = 0; i < std::min(H, pair.embedding.size()); ++i) {
            h[i] = pair.embedding[i];
        }

        // Transform
        auto h_transformed = decoder.transform(h);

        // For each target concept in the sequence, add a regression pair
        for (auto target_cid : pair.target_concepts) {
            auto flex = concept_emb_store.get_or_default(target_cid);
            std::vector<double> target_emb(D);
            for (size_t d = 0; d < D; ++d) {
                target_emb[d] = flex.core[d];
            }
            all_h.push_back(h_transformed);
            all_e.push_back(std::move(target_emb));
        }
    }

    const size_t N = all_h.size();
    if (N == 0) return;

    std::cerr << "[LanguageTraining]   Concept ridge regression: " << N
              << " samples, " << H << "D -> " << D << "D\n";

    // Build C = H^T H + λI  [H × H]
    std::vector<std::vector<double>> C(H, std::vector<double>(H, 0.0));
    for (size_t n = 0; n < N; ++n) {
        const auto& hn = all_h[n];
        for (size_t i = 0; i < H; ++i) {
            double hi = hn[i];
            for (size_t j = i; j < H; ++j) {
                C[i][j] += hi * hn[j];
            }
        }
    }
    for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j < i; ++j) C[i][j] = C[j][i];
        C[i][i] += lambda;
    }

    // Build B = H^T E  [H × D]
    std::vector<std::vector<double>> B(H, std::vector<double>(D, 0.0));
    for (size_t n = 0; n < N; ++n) {
        const auto& hn = all_h[n];
        const auto& en = all_e[n];
        for (size_t i = 0; i < H; ++i) {
            for (size_t d = 0; d < D; ++d) {
                B[i][d] += hn[i] * en[d];
            }
        }
    }

    // Invert C via Gauss-Jordan elimination
    std::vector<std::vector<double>> aug(H, std::vector<double>(2 * H, 0.0));
    for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j < H; ++j) aug[i][j] = C[i][j];
        aug[i][H + i] = 1.0;
    }

    for (size_t k = 0; k < H; ++k) {
        double max_val = std::abs(aug[k][k]);
        size_t max_row = k;
        for (size_t i = k + 1; i < H; ++i) {
            if (std::abs(aug[i][k]) > max_val) {
                max_val = std::abs(aug[i][k]);
                max_row = i;
            }
        }
        if (max_val < 1e-12) continue;
        if (max_row != k) std::swap(aug[k], aug[max_row]);

        double pivot = aug[k][k];
        for (size_t j = 0; j < 2 * H; ++j) aug[k][j] /= pivot;

        for (size_t i = 0; i < H; ++i) {
            if (i == k) continue;
            double factor = aug[i][k];
            for (size_t j = 0; j < 2 * H; ++j) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // Extract C^{-1} and compute W = C^{-1} B  [H × D]
    auto& W = decoder.concept_projection();
    W.resize(H);
    for (size_t i = 0; i < H; ++i) {
        W[i].resize(D, 0.0);
        for (size_t d = 0; d < D; ++d) {
            double sum = 0.0;
            for (size_t j = 0; j < H; ++j) {
                sum += aug[i][H + j] * B[j][d];
            }
            W[i][d] = sum;
        }
    }
}

// =============================================================================
// Concept Decoder: SGD Epoch with Trust-Weighted Cross-Entropy
// =============================================================================

double LanguageTraining::train_concept_decoder_epoch(
    const std::vector<ConceptDecoderPair>& data, double lr) {

    auto& decoder = engine_.decoder();
    auto& concept_emb_store = engine_.embeddings().concept_embeddings();
    auto& W = decoder.concept_projection();

    const size_t H = decoder.extended_fused_dim();
    const size_t D = LanguageConfig::CONCEPT_EMBED_DIM;  // 16
    const double temperature = LanguageConfig().concept_train_temperature;

    // Build concept index: concept_id -> index in concept_matrix
    const auto& concept_ids = decoder.concept_projection().empty()
        ? std::vector<std::vector<double>>() : decoder.concept_projection();
    (void)concept_ids;

    // Use the cached concept matrix from the decoder
    const auto& matrix = decoder.concept_projection();
    (void)matrix;

    // We need to build a mapping from concept IDs to indices in a flat embedding array
    // Collect all concept IDs that appear in target sequences
    std::unordered_map<ConceptId, size_t> cid_to_idx;
    std::vector<ConceptId> active_cids;
    std::vector<std::vector<double>> active_embeddings;  // [N_active × D]

    for (const auto& pair : data) {
        for (auto cid : pair.target_concepts) {
            if (cid_to_idx.find(cid) == cid_to_idx.end()) {
                size_t idx = active_cids.size();
                cid_to_idx[cid] = idx;
                active_cids.push_back(cid);
                auto flex = concept_emb_store.get_or_default(cid);
                std::vector<double> emb(D);
                for (size_t d = 0; d < D; ++d) emb[d] = flex.core[d];
                // L2-normalize
                double norm = 0.0;
                for (double v : emb) norm += v * v;
                norm = std::sqrt(norm);
                if (norm > 1e-12) for (auto& v : emb) v /= norm;
                active_embeddings.push_back(std::move(emb));
            }
        }
    }

    const size_t N_active = active_cids.size();
    if (N_active == 0) return 1e9;

    // Transform weights refs
    auto& W1 = decoder.transform_W1();
    auto& b1 = decoder.transform_b1();
    auto& W2 = decoder.transform_W2();
    auto& b2 = decoder.transform_b2();
    const size_t K = KANDecoder::TRANSFORM_K;
    const size_t FUSED_BASE = LanguageConfig::FUSED_DIM;
    const size_t CONV_DIM = LanguageConfig::CONVERGENCE_DIM;
    const size_t conv_start = H - CONV_DIM;

    double total_loss = 0.0;
    size_t total_steps = 0;

    // Pre-allocated buffers
    std::vector<double> h(H);
    std::vector<double> h_out(H);
    std::vector<double> z1(K), a1(K);
    std::vector<double> concept_emb(D);
    std::vector<double> logits(N_active);
    std::vector<double> probs(N_active);

    for (const auto& pair : data) {
        if (pair.target_concepts.empty()) continue;

        // Initialize hidden from fused vector
        std::fill(h.begin(), h.end(), 0.0);
        for (size_t i = 0; i < std::min(H, pair.embedding.size()); ++i)
            h[i] = pair.embedding[i];

        for (auto target_cid : pair.target_concepts) {
            auto tgt_it = cid_to_idx.find(target_cid);
            if (tgt_it == cid_to_idx.end()) continue;
            size_t target_idx = tgt_it->second;

            // Forward: transform
            for (size_t k = 0; k < K; ++k) {
                double sum = b1[k];
                for (size_t i = 0; i < H; ++i) sum += h[i] * W1[i][k];
                z1[k] = sum;
                a1[k] = std::tanh(sum);
            }
            for (size_t j = 0; j < H; ++j) {
                double sum = h[j] + b2[j];
                for (size_t k = 0; k < K; ++k) sum += a1[k] * W2[k][j];
                h_out[j] = sum;
            }

            // Project to concept space: concept_emb = W^T · h_out  [D]
            std::fill(concept_emb.begin(), concept_emb.end(), 0.0);
            for (size_t i = 0; i < H; ++i) {
                for (size_t d = 0; d < D; ++d) {
                    concept_emb[d] += h_out[i] * W[i][d];
                }
            }

            // L2-normalize
            double norm_emb = 0.0;
            for (double v : concept_emb) norm_emb += v * v;
            norm_emb = std::sqrt(norm_emb);
            if (norm_emb > 1e-12) {
                for (auto& v : concept_emb) v /= norm_emb;
            }

            // Cosine similarity logits
            for (size_t c = 0; c < N_active; ++c) {
                double dot = 0.0;
                for (size_t d = 0; d < D; ++d)
                    dot += concept_emb[d] * active_embeddings[c][d];
                logits[c] = dot / temperature;
            }

            // Softmax
            double max_val = *std::max_element(logits.begin(), logits.begin() + N_active);
            double exp_sum = 0.0;
            for (size_t c = 0; c < N_active; ++c) {
                probs[c] = std::exp(std::min(logits[c] - max_val, 80.0));
                exp_sum += probs[c];
            }
            if (exp_sum > 1e-12) {
                double inv = 1.0 / exp_sum;
                for (size_t c = 0; c < N_active; ++c) probs[c] *= inv;
            }

            // Trust-weighted CE loss
            double p_target = std::max(probs[target_idx], 1e-12);
            total_loss += -pair.trust_weight * std::log(p_target);
            total_steps++;

            // Gradient of CE w.r.t. concept_emb (simplified: through logits)
            // d_logit[c] = probs[c] - 1{c==target}
            // d_concept_emb[d] = Σ_c d_logit[c] * active_embeddings[c][d] / temperature
            // (Since cosine sim to pre-normalized embeddings, gradient is direct)
            std::vector<double> d_concept_emb(D, 0.0);
            for (size_t c = 0; c < N_active; ++c) {
                double grad = (probs[c] - (c == target_idx ? 1.0 : 0.0)) / temperature;
                grad *= pair.trust_weight;
                for (size_t d = 0; d < D; ++d) {
                    d_concept_emb[d] += grad * active_embeddings[c][d];
                }
            }

            // Update W_concept: W[i][d] -= lr * h_out[i] * d_concept_emb[d]
            for (size_t i = 0; i < H; ++i) {
                for (size_t d = 0; d < D; ++d) {
                    W[i][d] -= lr * h_out[i] * d_concept_emb[d];
                }
            }

            // Hidden evolution with concept embedding
            auto pred_flex = concept_emb_store.get_or_default(target_cid);

            // Block 1: core embedding feedback
            for (size_t i = 0; i < FUSED_BASE && i < CORE_DIM; ++i) {
                h[i] = h[i] * 0.8 + pred_flex.core[i] * 0.2;
            }
            // Block 2: flex detail
            size_t flex_end = std::min(FUSED_BASE + decoder.flex_dim(), H);
            for (size_t i = FUSED_BASE; i < flex_end; ++i) {
                size_t detail_idx = i - FUSED_BASE;
                double flex_val = (detail_idx < pred_flex.detail.size()) ? pred_flex.detail[detail_idx] : 0.0;
                h[i] = h[i] * 0.9 + flex_val * 0.1;
            }
            // Block 3: DimCtx decay
            for (size_t i = FUSED_BASE + decoder.flex_dim(); i < conv_start; ++i) {
                h[i] *= 0.95;
            }
            // Block 4: Convergence decay
            for (size_t i = conv_start; i < H; ++i) {
                h[i] *= 0.9;
            }
        }
    }

    return (total_steps > 0) ? total_loss / total_steps : 1e9;
}

// =============================================================================
// Stage 1: Encoder + Decoder Training
// =============================================================================

LanguageTrainingResult LanguageTraining::train_stage1(const LanguageConfig& config) {
    if (config.use_deep_kan) {
        return train_stage1_deep_kan(config);
    }

    LanguageTrainingResult result;
    result.stage = 1;
    result.stage_name = "Encoder+Decoder";
    result.converged = false;
    result.epochs_run = 0;
    result.final_loss = 1e9;

    // Generate training data
    std::cerr << "[LanguageTraining] Generating encoder data...\n";
    auto encoder_data = generate_encoder_data();
    std::cerr << "[LanguageTraining] Generating unified concept descriptions...\n";
    auto decoder_data = generate_decoder_data();
    std::cerr << "[LanguageTraining] Generating supplemental relation data...\n";
    auto relation_data = generate_relation_decoder_data();
    std::cerr << "[LanguageTraining] Data generated: enc=" << encoder_data.size()
              << " unified=" << decoder_data.size() << " rel=" << relation_data.size() << "\n";

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
                  << decoder_data.size() << " unified + "
                  << relation_data.size() << " rel)...\n";

        auto t_start = std::chrono::steady_clock::now();

        // ── Phase A: Closed-form ridge regression for output projection ──
        // Gives MSE-optimal W in one shot (~50ms), then SGD fine-tunes with CE.
        std::cerr << "[LanguageTraining]   Phase A: Closed-form ridge init...\n";
        train_decoder_closedform(all_decoder_data, 1.0);
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            std::cerr << "[LanguageTraining]   Ridge init done (" << elapsed << "ms)\n";
        }

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

        auto t_pretok = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();
        std::cerr << "[LanguageTraining]   Pre-tokenized " << pretok_samples.size()
                  << " samples, " << VA << " active tokens (" << t_pretok << "ms)\n";

        // ── Train with per-token CE-SGD using pre-tokenized data ──
        auto& decoder = engine_.decoder();
        auto& W = decoder.output_projection();
        auto& emb_table = engine_.encoder().embedding_table();

        // H = extended fused dim (FUSED_DIM + dim context, runtime)
        const size_t H = decoder.extended_fused_dim();
        const size_t H_EXT = 2 * H;
        const size_t FUSED_BASE = LanguageConfig::FUSED_DIM;  // 64 (token embedding dims)
        const size_t flex_start = FUSED_BASE;                   // Dynamic block boundary (Audit #15)
        const size_t dimctx_start = FUSED_BASE + decoder.flex_dim();
        const double lr = config.decoder_lr;

        const size_t CONV_DIM = LanguageConfig::CONVERGENCE_DIM;
        const size_t conv_start = H - CONV_DIM;

        std::cerr << "[LanguageTraining]   Hidden dim H=" << H
                  << " (base=" << FUSED_BASE
                  << " + flex=" << decoder.flex_dim()
                  << " + dim_ctx=" << (conv_start - dimctx_start)
                  << " + conv=" << CONV_DIM << ")\n";

        // ── Try GPU training loop (v11 3-block SM-parallel) ──
        if (cuda::gpu_available()) {
            std::cerr << "[LanguageTraining]   Trying CUDA V11 3-block SGD...\n";

            // Pack data into flat arrays
            cuda::TrainingData td;
            td.num_samples = pretok_samples.size();
            td.V = V;
            td.VA = VA;
            td.H = H;
            td.H_EXT = H_EXT;
            td.K = KANDecoder::TRANSFORM_K;
            td.FUSED_BASE = FUSED_BASE;
            td.flex_dim = decoder.flex_dim();

            // Flatten tokens
            td.sample_offsets.resize(td.num_samples);
            td.sample_lengths.resize(td.num_samples);
            for (size_t s = 0; s < td.num_samples; ++s) {
                td.sample_offsets[s] = td.all_tokens.size();
                td.sample_lengths[s] = pretok_samples[s].tokens.size();
                for (auto tok_id : pretok_samples[s].tokens) {
                    td.all_tokens.push_back(tok_id);
                }
            }

            // Flatten embeddings
            td.embeddings.resize(td.num_samples * H, 0.0);
            for (size_t s = 0; s < td.num_samples; ++s) {
                const auto& emb = all_decoder_data[pretok_samples[s].pair_idx].embedding;
                for (size_t i = 0; i < std::min(H, emb.size()); ++i) {
                    td.embeddings[s * H + i] = emb[i];
                }
            }

            // Compress map
            td.compress.resize(V, VA);  // default to VA (invalid) for unseen tokens
            for (size_t v = 0; v < V; ++v) {
                if (seen[v]) td.compress[v] = compress[v];
            }
            td.active_tokens.assign(active_tokens.begin(), active_tokens.end());

            // Flatten embedding table
            td.emb_table.resize(V * FUSED_BASE, 0.0);
            for (size_t v = 0; v < std::min(V, emb_table.size()); ++v) {
                for (size_t i = 0; i < std::min(FUSED_BASE, emb_table[v].size()); ++i) {
                    td.emb_table[v * FUSED_BASE + i] = emb_table[v][i];
                }
            }

            // Precompute flex_table: [V * flex_dim] — FlexDetail per token
            if (td.flex_dim > 0) {
                td.flex_table.resize(V * td.flex_dim, 0.0);
                auto& concept_emb_store = engine_.embeddings().concept_embeddings();
                for (size_t v = 0; v < V; ++v) {
                    auto tok_concept = engine_.tokenizer().token_to_concept(static_cast<uint16_t>(v));
                    if (tok_concept) {
                        auto flex_emb = concept_emb_store.get_or_default(*tok_concept);
                        for (size_t d = 0; d < td.flex_dim; ++d) {
                            td.flex_table[v * td.flex_dim + d] =
                                (d < flex_emb.detail.size()) ? flex_emb.detail[d] : 0.0;
                        }
                    }
                }
            }

            // Precompute conv_table: [V * CONV_DIM] — ConvergencePort output per token
            td.conv_dim = CONV_DIM;
            td.conv_table.resize(V * CONV_DIM, 0.0);
            for (size_t v = 0; v < V; ++v) {
                auto cpt = engine_.tokenizer().token_to_concept(static_cast<uint16_t>(v));
                if (cpt && registry_.has_model(*cpt)) {
                    auto* cm = registry_.get_model(*cpt);
                    std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
                    auto flex_emb = engine_.embeddings().concept_embeddings().get_or_default(*cpt);
                    for (size_t d = 0; d < std::min(size_t(16), flex_emb.core.size()); ++d)
                        conv_input[d] = flex_emb.core[d];
                    double conv_out[ConvergencePort::OUTPUT_DIM];
                    cm->forward_convergence(conv_input.data(), conv_out);
                    for (size_t d = 0; d < CONV_DIM; ++d)
                        td.conv_table[v * CONV_DIM + d] = conv_out[d];
                }
            }

            // Pack weights
            cuda::TrainingWeights tw;
            tw.W_a.resize(H_EXT * VA);
            for (size_t i = 0; i < H_EXT; ++i) {
                for (size_t a = 0; a < VA; ++a) {
                    tw.W_a[i * VA + a] = W[i][active_tokens[a]];
                }
            }

            auto& W1 = decoder.transform_W1();
            auto& b1 = decoder.transform_b1();
            auto& W2 = decoder.transform_W2();
            auto& b2 = decoder.transform_b2();

            tw.W1.resize(H * td.K);
            for (size_t i = 0; i < H; ++i)
                for (size_t k = 0; k < td.K; ++k)
                    tw.W1[i * td.K + k] = W1[i][k];

            tw.b1.assign(b1.begin(), b1.end());

            tw.W2.resize(td.K * H);
            for (size_t k = 0; k < td.K; ++k)
                for (size_t j = 0; j < H; ++j)
                    tw.W2[k * H + j] = W2[k][j];

            tw.b2.assign(b2.begin(), b2.end());

            cuda::TrainingConfig tc;
            tc.num_epochs = config.decoder_epochs;
            tc.base_lr = config.decoder_lr;
            tc.lr_transform_base = 0.001;
            tc.transform_warmup = 10;

            cuda::TrainingResult tr;

            bool used_gpu = cuda::train_sgd_v11_gpu(td, tw, tc, tr);
            if (used_gpu) {
                std::cerr << "[LanguageTraining]   CUDA V11 done! best_loss=" << tr.best_loss << "\n";

                // Unpack weights back
                for (size_t i = 0; i < H_EXT; ++i) {
                    for (size_t v = 0; v < V; ++v) W[i][v] = 0.0;
                    for (size_t a = 0; a < VA; ++a) {
                        W[i][active_tokens[a]] = tw.W_a[i * VA + a];
                    }
                }

                for (size_t i = 0; i < H; ++i)
                    for (size_t k = 0; k < td.K; ++k)
                        W1[i][k] = tw.W1[i * td.K + k];

                b1.assign(tw.b1.begin(), tw.b1.end());

                for (size_t k = 0; k < td.K; ++k)
                    for (size_t j = 0; j < H; ++j)
                        W2[k][j] = tw.W2[k * H + j];

                b2.assign(tw.b2.begin(), tw.b2.end());

                if (tr.best_loss < best_loss) best_loss = tr.best_loss;
                result.final_loss = best_loss;
                result.converged = best_loss < 0.1;
                return result;
            }
            std::cerr << "[LanguageTraining]   CUDA V11 fallback to CPU\n";
        }

        // ── CPU training loop (fallback) ──

        // Build conv_table for CPU path: [V * CONV_DIM]
        std::vector<double> conv_table(V * CONV_DIM, 0.0);
        for (size_t v = 0; v < V; ++v) {
            auto cpt = engine_.tokenizer().token_to_concept(static_cast<uint16_t>(v));
            if (cpt && registry_.has_model(*cpt)) {
                auto* cm = registry_.get_model(*cpt);
                std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
                auto flex_emb = engine_.embeddings().concept_embeddings().get_or_default(*cpt);
                for (size_t d = 0; d < std::min(size_t(16), flex_emb.core.size()); ++d)
                    conv_input[d] = flex_emb.core[d];
                double conv_out[ConvergencePort::OUTPUT_DIM];
                cm->forward_convergence(conv_input.data(), conv_out);
                for (size_t d = 0; d < CONV_DIM; ++d)
                    conv_table[v * CONV_DIM + d] = conv_out[d];
            }
        }

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

        // Transform buffers + references
        const size_t K = KANDecoder::TRANSFORM_K;
        auto& W1 = decoder.transform_W1();
        auto& b1 = decoder.transform_b1();
        auto& W2 = decoder.transform_W2();
        auto& b2 = decoder.transform_b2();
        std::vector<double> h_out(H);
        std::vector<double> z1(K);
        std::vector<double> a1(K);
        std::vector<double> d_h_ext(H_EXT);
        std::vector<double> d_h_out(H);
        std::vector<double> d_a1(K);
        std::vector<double> d_z1(K);
        const double lr_transform_base = 0.001;
        const size_t transform_warmup = 10;  // freeze transform for first N epochs

        double best_decoder_loss = 1e9;
        for (size_t epoch = 0; epoch < config.decoder_epochs; ++epoch) {
            bool train_transform = (epoch >= transform_warmup);

            // Cosine LR decay: lr starts at base, decays to base/10
            double progress = static_cast<double>(epoch) / std::max(config.decoder_epochs - 1, size_t(1));
            double cos_mult = 0.5 * (1.0 + std::cos(progress * 3.14159265358979));
            double lr_epoch = lr * (0.1 + 0.9 * cos_mult);  // decays from lr to 0.1*lr
            double lr_transform_epoch = lr_transform_base * (0.1 + 0.9 * cos_mult);

            double total_ce_loss = 0.0;
            size_t total_tokens = 0;

            for (const auto& sample : pretok_samples) {
                const auto& target_tokens = sample.tokens;
                const auto& embedding = all_decoder_data[sample.pair_idx].embedding;

                // Reset h to extended fused vector (64D base + dim context)
                std::fill(h.begin(), h.end(), 0.0);
                for (size_t i = 0; i < std::min(H, embedding.size()); ++i) {
                    h[i] = embedding[i];
                }

                for (size_t t = 0; t < target_tokens.size(); ++t) {
                    uint16_t target_tok = target_tokens[t];
                    if (target_tok >= V || !seen[target_tok]) continue;

                    size_t ca = compress[target_tok];

                    // Forward through transform: h' = h + tanh(h·W1+b1)·W2+b2
                    for (size_t k = 0; k < K; ++k) {
                        double sum = b1[k];
                        for (size_t i = 0; i < H; ++i) sum += h[i] * W1[i][k];
                        z1[k] = sum;
                        a1[k] = std::tanh(sum);
                    }
                    for (size_t j = 0; j < H; ++j) {
                        double sum = h[j] + b2[j];
                        for (size_t k = 0; k < K; ++k) sum += a1[k] * W2[k][j];
                        h_out[j] = sum;
                    }

                    // Build quadratic features: h_ext = [h_out, h_out^2]
                    for (size_t i = 0; i < H; ++i) {
                        h_ext[i] = h_out[i];
                        h_ext[H + i] = h_out[i] * h_out[i];
                    }

                    // logits = h_ext . W_a^T in R^VA
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

                    // Compute d_h_ext BEFORE W update (pre-update W_a for correct gradient)
                    if (train_transform) {
                        for (size_t i = 0; i < H_EXT; ++i) {
                            double d = 0.0;
                            for (size_t a = 0; a < VA; ++a) d += probs[a] * W_a[i][a];
                            d -= W_a[i][ca];
                            d_h_ext[i] = d;
                        }
                    }

                    // Per-token W update with 4-block-aware LR
                    // Block 1 (Token): lr × 1.0
                    // Block 2 (Flex): lr × 0.3
                    // Block 3 (DimCtx): lr × 0.1
                    // Block 4 (Conv): lr × 0.3
                    // (applies to both linear and quadratic halves of h_ext)
                    for (size_t i = 0; i < H_EXT; ++i) {
                        size_t base_dim = i % H;  // map quadratic half back to base dim
                        double block_lr = lr_epoch;
                        if (base_dim >= flex_start && base_dim < dimctx_start) {
                            block_lr = lr_epoch * 0.3;
                        } else if (base_dim >= conv_start) {
                            block_lr = lr_epoch * 0.3;  // Block 4: convergence
                        } else if (base_dim >= dimctx_start) {
                            block_lr = lr_epoch * 0.1;  // Block 3: dimctx
                        }
                        double hi = h_ext[i];
                        for (size_t a = 0; a < VA; ++a) {
                            W_a[i][a] -= block_lr * hi * probs[a];
                        }
                        W_a[i][ca] += block_lr * hi;
                    }

                    // Backprop through transform (only after warmup epochs)
                    if (train_transform) {
                        // Through quadratic: d_h_out[i] = d_h_ext[i] + 2*h_out[i]*d_h_ext[H+i]
                        for (size_t i = 0; i < H; ++i) {
                            d_h_out[i] = d_h_ext[i] + 2.0 * h_out[i] * d_h_ext[H + i];
                        }

                        // Through W2: d_a1[k] = Σ_j d_h_out[j] * W2[k][j]
                        for (size_t k = 0; k < K; ++k) {
                            double d = 0.0;
                            for (size_t j = 0; j < H; ++j) d += d_h_out[j] * W2[k][j];
                            d_a1[k] = d;
                        }

                        // W2 update
                        for (size_t k = 0; k < K; ++k) {
                            for (size_t j = 0; j < H; ++j) {
                                W2[k][j] -= lr_transform_epoch * a1[k] * d_h_out[j];
                            }
                        }

                        // b2: skip update (bias accumulates too much across tokens)

                        // Through tanh: d_z1[k] = d_a1[k] * (1 - a1[k]²)
                        for (size_t k = 0; k < K; ++k) {
                            d_z1[k] = d_a1[k] * (1.0 - a1[k] * a1[k]);
                        }

                        // W1 update
                        for (size_t i = 0; i < H; ++i) {
                            for (size_t k = 0; k < K; ++k) {
                                W1[i][k] -= lr_transform_epoch * h[i] * d_z1[k];
                            }
                        }

                        // b1 update
                        for (size_t k = 0; k < K; ++k) {
                            b1[k] -= lr_transform_epoch * d_z1[k];
                        }
                    }

                    // v11: 3-block hidden state evolution
                    // Block 1: Dims 0-63 (Token): h = 0.8*h + 0.2*tok_emb
                    if (target_tok < emb_table.size()) {
                        const auto& tok_emb = emb_table[target_tok];
                        for (size_t i = 0; i < FUSED_BASE && i < tok_emb.size(); ++i) {
                            h[i] = h[i] * 0.8 + tok_emb[i] * 0.2;
                        }
                    }
                    // Block 2: Flex dims: h = 0.9*h + 0.1*flex_detail
                    {
                        auto tok_concept = engine_.tokenizer().token_to_concept(target_tok);
                        if (tok_concept) {
                            auto flex_emb = engine_.embeddings().concept_embeddings().get_or_default(*tok_concept);
                            size_t flex_end = std::min(dimctx_start, H);
                            for (size_t i = flex_start; i < flex_end; ++i) {
                                size_t detail_idx = i - flex_start;
                                double flex_val = (detail_idx < flex_emb.detail.size()) ? flex_emb.detail[detail_idx] : 0.0;
                                h[i] = h[i] * 0.9 + flex_val * 0.1;
                            }
                        }
                    }
                    // Block 3: DimCtx dims: slow decay (stop before conv)
                    for (size_t i = dimctx_start; i < conv_start; ++i) {
                        h[i] *= 0.95;
                    }
                    // Block 4: Convergence dims — inject CM signal per token
                    for (size_t d = 0; d < CONV_DIM; ++d) {
                        h[conv_start + d] = h[conv_start + d] * 0.9
                            + conv_table[target_tok * CONV_DIM + d] * 0.1;
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
                          << " lr=" << lr_epoch
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

    // NOTE: CPU-based concept decoder training removed.
    // Concept prediction is now integrated into the LibTorch pipeline
    // via train_stage1_deep_kan_v2() → train_concept_deep_kan_v2().

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
        // For now, use simplified approach: train KAN on (bag -> target)
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
//   W = (H^T H + lambda I)^{-1} H^T Y
// where H is [N x H_EXT] (all training h_ext vectors) and Y is one-hot targets.
// This gives the MSE-optimal W in one shot.
//

void LanguageTraining::train_decoder_closedform(
    const std::vector<DecoderPair>& data, double lambda) {

    auto& decoder = engine_.decoder();
    auto& tokenizer = engine_.tokenizer();
    auto& W = decoder.output_projection();
    auto& emb_table = engine_.encoder().embedding_table();

    // H = extended fused dim (runtime, includes dim context + convergence)
    const size_t H = decoder.extended_fused_dim();
    const size_t H_EXT = 2 * H;
    const size_t FUSED_BASE = LanguageConfig::FUSED_DIM;
    const size_t flex_start = FUSED_BASE;
    const size_t dimctx_start = FUSED_BASE + decoder.flex_dim();
    const size_t CONV_DIM = LanguageConfig::CONVERGENCE_DIM;
    const size_t conv_start = H - CONV_DIM;
    const size_t V = LanguageConfig::VOCAB_SIZE;

    // Build conv_table for hidden state evolution
    std::vector<double> conv_table(V * CONV_DIM, 0.0);
    for (size_t v = 0; v < V; ++v) {
        auto cpt = engine_.tokenizer().token_to_concept(static_cast<uint16_t>(v));
        if (cpt && registry_.has_model(*cpt)) {
            auto* cm = registry_.get_model(*cpt);
            std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
            auto flex_emb = engine_.embeddings().concept_embeddings().get_or_default(*cpt);
            for (size_t d = 0; d < std::min(size_t(16), flex_emb.core.size()); ++d)
                conv_input[d] = flex_emb.core[d];
            double conv_out[ConvergencePort::OUTPUT_DIM];
            cm->forward_convergence(conv_input.data(), conv_out);
            for (size_t d = 0; d < CONV_DIM; ++d)
                conv_table[v * CONV_DIM + d] = conv_out[d];
        }
    }

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
    std::vector<std::vector<double>> all_h;   // N x H_EXT
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

            // Build h_ext = [h, h^2]
            std::vector<double> h_ext(H_EXT);
            for (size_t i = 0; i < H; ++i) {
                h_ext[i] = h[i];
                h_ext[H + i] = h[i] * h[i];
            }
            all_h.push_back(std::move(h_ext));
            all_targets.push_back(compress[target_tok]);

            // v11: 3-block hidden state evolution
            if (target_tok < emb_table.size()) {
                const auto& tok_emb = emb_table[target_tok];
                for (size_t i = 0; i < FUSED_BASE && i < tok_emb.size(); ++i) {
                    h[i] = h[i] * 0.8 + tok_emb[i] * 0.2;
                }
            }
            // Block 2: Flex detail (dynamic boundaries, Audit #15)
            {
                auto tok_concept = engine_.tokenizer().token_to_concept(target_tok);
                if (tok_concept) {
                    auto flex_emb = engine_.embeddings().concept_embeddings().get_or_default(*tok_concept);
                    size_t flex_end = std::min(dimctx_start, H);
                    for (size_t i = flex_start; i < flex_end; ++i) {
                        size_t detail_idx = i - flex_start;
                        double flex_val = (detail_idx < flex_emb.detail.size()) ? flex_emb.detail[detail_idx] : 0.0;
                        h[i] = h[i] * 0.9 + flex_val * 0.1;
                    }
                }
            }
            // Block 3: DimCtx (stop before conv)
            for (size_t i = dimctx_start; i < conv_start; ++i) {
                h[i] *= 0.95;
            }
            // Block 4: Convergence dims
            for (size_t d = 0; d < CONV_DIM; ++d) {
                h[conv_start + d] = h[conv_start + d] * 0.9
                    + conv_table[target_tok * CONV_DIM + d] * 0.1;
            }
        }
    }

    const size_t N = all_h.size();
    if (N == 0) return;

    std::cerr << "[LanguageTraining]   Ridge regression: " << N
              << " samples, " << H_EXT << "D features, " << VA << " active tokens\n";

    const double logit_scale = 4.0;

    // ── Try GPU path first ──
    if (cuda::gpu_available()) {
        std::cerr << "[LanguageTraining]   Trying CUDA ridge...\n";

        // Flatten all_h to row-major
        std::vector<double> H_flat(N * H_EXT);
        for (size_t n = 0; n < N; ++n) {
            std::copy(all_h[n].begin(), all_h[n].end(), H_flat.data() + n * H_EXT);
        }

        cuda::RidgeParams params;
        params.H = H_flat.data();
        params.N = N;
        params.D = H_EXT;
        params.targets = all_targets.data();
        params.VA = VA;
        params.lambda = lambda;
        params.logit_scale = logit_scale;

        // D×VA output
        std::vector<double> w_flat(H_EXT * VA, 0.0);
        bool used_gpu = cuda::ridge_solve(params, w_flat.data());

        if (used_gpu) {
            std::cerr << "[LanguageTraining]   CUDA ridge done!\n";
            // Write to output_projection
            for (size_t i = 0; i < H_EXT; ++i) {
                for (size_t v = 0; v < V; ++v) W[i][v] = 0.0;
                for (size_t a = 0; a < VA; ++a) {
                    W[i][active_tokens[a]] = w_flat[i * VA + a];
                }
            }
            return;
        }
        std::cerr << "[LanguageTraining]   CUDA fallback to CPU\n";
    }

    // ── CPU fallback ──
    // ── Build C = H^T H + lambda I  [H_EXT x H_EXT] ──
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

    // ── Build B = H^T Y  [H_EXT x VA] ──
    std::vector<std::vector<double>> B(H_EXT, std::vector<double>(VA, 0.0));
    for (size_t n = 0; n < N; ++n) {
        size_t v = all_targets[n];
        const auto& hn = all_h[n];
        for (size_t i = 0; i < H_EXT; ++i) {
            B[i][v] += logit_scale * hn[i];
        }
    }

    // ── Invert C via Gauss-Jordan elimination ──
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

    // ── W = C^{-1} . B  and write to output_projection ──
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

// =============================================================================
// Deep KAN Training Path (V12)
// =============================================================================
//
// 2-layer EfficientKAN feature extractor: H → 256 → 128
// Linear output: 128 → VA (ridge-initialized, SGD fine-tuned)
// ~585K parameters vs ~20K in V11
//

LanguageTrainingResult LanguageTraining::train_stage1_deep_kan(const LanguageConfig& config) {
    LanguageTrainingResult result;
    result.stage = 1;
    result.stage_name = "DeepKAN Decoder";
    result.converged = false;
    result.epochs_run = 0;
    result.final_loss = 1e9;

    // ── Generate training data (same as V11) ──
    std::cerr << "[LanguageTraining] Generating encoder data...\n";
    auto encoder_data = generate_encoder_data();
    std::cerr << "[LanguageTraining] Generating unified concept descriptions...\n";
    auto decoder_data = generate_decoder_data();
    std::cerr << "[LanguageTraining] Generating supplemental relation data...\n";
    auto relation_data = generate_relation_decoder_data();
    std::cerr << "[LanguageTraining] Data generated: enc=" << encoder_data.size()
              << " unified=" << decoder_data.size() << " rel=" << relation_data.size() << "\n";

    // Train encoder (brief)
    double best_loss = 1e9;
    if (!encoder_data.empty()) {
        std::cerr << "[LanguageTraining] Stage 1: Training encoder on "
                  << encoder_data.size() << " pairs...\n";
        for (size_t epoch = 0; epoch < config.encoder_epochs; ++epoch) {
            double loss = train_encoder_epoch(encoder_data, config.encoder_lr);
            if (loss < best_loss) best_loss = loss;
            if (loss < 1e-4) break;
        }
    }

    // Merge decoder data
    std::vector<DecoderPair> all_decoder_data;
    all_decoder_data.reserve(decoder_data.size() + relation_data.size());
    all_decoder_data.insert(all_decoder_data.end(),
        std::make_move_iterator(decoder_data.begin()),
        std::make_move_iterator(decoder_data.end()));
    all_decoder_data.insert(all_decoder_data.end(),
        std::make_move_iterator(relation_data.begin()),
        std::make_move_iterator(relation_data.end()));
    { std::mt19937 rng(12345); std::shuffle(all_decoder_data.begin(), all_decoder_data.end(), rng); }

    if (all_decoder_data.empty()) return result;

    std::cerr << "[LanguageTraining] Stage 1: Training Deep KAN decoder on "
              << all_decoder_data.size() << " pairs...\n";

    auto t_start = std::chrono::steady_clock::now();

    // ── Pre-tokenize ──
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
    std::vector<size_t> compress(V, 0);
    for (size_t v = 0; v < V; ++v) {
        if (seen[v]) {
            compress[v] = active_tokens.size();
            active_tokens.push_back(static_cast<uint16_t>(v));
        }
    }
    const size_t VA = active_tokens.size();

    auto t_pretok = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[LanguageTraining]   Pre-tokenized " << pretok_samples.size()
              << " samples, " << VA << " active tokens (" << t_pretok << "ms)\n";

    // ── Dimensions ──
    auto& decoder = engine_.decoder();
    auto& emb_table = engine_.encoder().embedding_table();
    const size_t H = decoder.extended_fused_dim();  // 122 (64+16+10+32)
    const size_t FUSED_BASE = LanguageConfig::FUSED_DIM;  // 64
    const size_t CONV_DIM = LanguageConfig::CONVERGENCE_DIM;  // 32
    const size_t conv_start = H - CONV_DIM;
    const size_t FEAT_DIM = 128;  // Deep KAN output dimension
    const double lr_output = config.decoder_lr;       // W_a LR (high: 2.0)
    const double lr_kan = config.deep_kan_lr;          // KAN LR (low: 0.01)

    // ── Construct Deep KAN: H → 256 → 128 → FEAT_DIM ──
    DeepKAN deep_kan({H, 256, 128, FEAT_DIM}, {8, 5, 5}, 3);
    std::cerr << "[LanguageTraining]   Deep KAN: " << H << "→256→128→" << FEAT_DIM
              << ", " << deep_kan.num_params() << " params (+ " << FEAT_DIM * VA << " output)\n";

    // Block-aware LR scale for first KAN layer (input dim grouping)
    std::vector<double> lr_input_scale(H, 1.0);
    for (size_t i = FUSED_BASE; i < FUSED_BASE + decoder.flex_dim() && i < H; ++i)
        lr_input_scale[i] = 0.3;
    for (size_t i = FUSED_BASE + decoder.flex_dim(); i < conv_start; ++i)
        lr_input_scale[i] = 0.1;  // dimctx
    for (size_t i = conv_start; i < H; ++i)
        lr_input_scale[i] = 0.3;  // convergence — medium LR (strong signal)

    // ── Pack training data for GPU ──
    std::cerr << "[LanguageTraining]   Packing data for GPU...\n";

    cuda::TrainingData td;
    td.num_samples = pretok_samples.size();
    td.V = V;
    td.VA = VA;
    td.H = H;
    td.FUSED_BASE = FUSED_BASE;
    td.flex_dim = decoder.flex_dim();

    // Flatten sample tokens
    size_t total_toks = 0;
    for (const auto& s : pretok_samples) total_toks += s.tokens.size();
    td.all_tokens.reserve(total_toks);
    td.sample_offsets.resize(pretok_samples.size());
    td.sample_lengths.resize(pretok_samples.size());
    for (size_t i = 0; i < pretok_samples.size(); ++i) {
        td.sample_offsets[i] = td.all_tokens.size();
        td.sample_lengths[i] = pretok_samples[i].tokens.size();
        td.all_tokens.insert(td.all_tokens.end(),
            pretok_samples[i].tokens.begin(), pretok_samples[i].tokens.end());
    }

    // Embeddings: [num_samples * H]
    td.embeddings.resize(pretok_samples.size() * H, 0.0);
    for (size_t i = 0; i < pretok_samples.size(); ++i) {
        const auto& emb = all_decoder_data[pretok_samples[i].pair_idx].embedding;
        for (size_t j = 0; j < std::min(H, emb.size()); ++j)
            td.embeddings[i * H + j] = emb[j];
    }

    // Compression map
    td.compress.resize(V);
    for (size_t v = 0; v < V; ++v) td.compress[v] = compress[v];
    td.active_tokens = active_tokens;

    // Embedding table: [V * FUSED_BASE]
    td.emb_table.resize(V * FUSED_BASE, 0.0);
    for (size_t v = 0; v < std::min(emb_table.size(), V); ++v) {
        for (size_t j = 0; j < std::min(emb_table[v].size(), FUSED_BASE); ++j)
            td.emb_table[v * FUSED_BASE + j] = emb_table[v][j];
    }

    // FlexDetail table: precompute per-token
    size_t fd = decoder.flex_dim();
    td.flex_table.resize(V * fd, 0.0);
    for (size_t v = 0; v < V; ++v) {
        auto cpt = engine_.tokenizer().token_to_concept(static_cast<uint16_t>(v));
        if (cpt) {
            auto flex_emb = engine_.embeddings().concept_embeddings().get_or_default(*cpt);
            for (size_t j = 0; j < std::min(fd, flex_emb.detail.size()); ++j)
                td.flex_table[v * fd + j] = flex_emb.detail[j];
        }
    }

    // ConvergencePort table: precompute per-token
    td.conv_dim = CONV_DIM;
    td.conv_table.resize(V * CONV_DIM, 0.0);
    for (size_t v = 0; v < V; ++v) {
        auto cpt = engine_.tokenizer().token_to_concept(static_cast<uint16_t>(v));
        if (cpt && registry_.has_model(*cpt)) {
            auto* cm = registry_.get_model(*cpt);
            std::vector<double> conv_input(ConvergencePort::INPUT_DIM, 0.0);
            auto flex_emb = engine_.embeddings().concept_embeddings().get_or_default(*cpt);
            for (size_t d = 0; d < std::min(size_t(16), flex_emb.core.size()); ++d)
                conv_input[d] = flex_emb.core[d];
            double conv_out[ConvergencePort::OUTPUT_DIM];
            cm->forward_convergence(conv_input.data(), conv_out);
            for (size_t d = 0; d < CONV_DIM; ++d)
                td.conv_table[v * CONV_DIM + d] = conv_out[d];
        }
    }

    // ── Extract DeepKAN weights ──
    cuda::DeepKANWeights dkw;
    auto& l1 = deep_kan.layer(0);
    auto& l2 = deep_kan.layer(1);
    auto& l3 = deep_kan.layer(2);

    dkw.k1_weights  = l1.spline_weights();
    dkw.k1_residual = l1.residual_W();
    dkw.k1_gamma    = l1.ln_gamma();
    dkw.k1_beta     = l1.ln_beta();
    dkw.k1_knots    = l1.knots();

    dkw.k2_weights  = l2.spline_weights();
    dkw.k2_residual = l2.residual_W();
    dkw.k2_gamma    = l2.ln_gamma();
    dkw.k2_beta     = l2.ln_beta();
    dkw.k2_knots    = l2.knots();

    dkw.k3_weights  = l3.spline_weights();
    dkw.k3_residual = l3.residual_W();
    dkw.k3_gamma    = l3.ln_gamma();
    dkw.k3_beta     = l3.ln_beta();
    dkw.k3_knots    = l3.knots();

    // Random init W_a [FEAT_DIM × VA] (GPU warmup replaces ridge)
    dkw.W_a.resize(FEAT_DIM * VA);
    {
        std::mt19937 rng(42);
        double scale = std::sqrt(6.0 / (double)(FEAT_DIM + VA));
        std::uniform_real_distribution<double> dist(-scale, scale);
        for (auto& w : dkw.W_a) w = dist(rng);
    }

    // Config
    cuda::DeepKANConfig dkc;
    dkc.num_epochs = config.decoder_epochs;
    dkc.lr_output = lr_output;
    dkc.lr_kan = lr_kan;
    dkc.warmup_epochs = 10;
    dkc.lr_scale = lr_input_scale;

    cuda::TrainingResult tr;
    tr.best_loss = 1e9;

    std::cerr << "[LanguageTraining]   Trying CUDA Deep KAN training...\n";
    bool gpu_ok = cuda::train_deep_kan_gpu(td, dkw, dkc, tr);

    if (!gpu_ok) {
        std::cerr << "[LanguageTraining]   CUDA not available, cannot train Deep KAN on CPU (too slow)\n";
        result.final_loss = 1e9;
        return result;
    }

    double best_decoder_loss = tr.best_loss;
    result.epochs_run = config.decoder_epochs;

    // ── Write weights back to DeepKAN object ──
    l1.spline_weights() = dkw.k1_weights;
    l1.residual_W()     = dkw.k1_residual;
    l1.ln_gamma()       = dkw.k1_gamma;
    l1.ln_beta()        = dkw.k1_beta;
    l2.spline_weights() = dkw.k2_weights;
    l2.residual_W()     = dkw.k2_residual;
    l2.ln_gamma()       = dkw.k2_gamma;
    l2.ln_beta()        = dkw.k2_beta;
    l3.spline_weights() = dkw.k3_weights;
    l3.residual_W()     = dkw.k3_residual;
    l3.ln_gamma()       = dkw.k3_gamma;
    l3.ln_beta()        = dkw.k3_beta;

    // ── Write W_a back to decoder output projection ──
    auto& W = decoder.output_projection();
    W.resize(FEAT_DIM);
    for (size_t i = 0; i < FEAT_DIM; ++i) {
        W[i].assign(V, 0.0);
        for (size_t a = 0; a < VA; ++a)
            W[i][active_tokens[a]] = dkw.W_a[i * VA + a];
    }

    if (best_decoder_loss < best_loss) best_loss = best_decoder_loss;
    result.final_loss = best_loss;
    result.converged = best_loss < 0.1;

    std::cerr << "[LanguageTraining]   Deep KAN training complete, best loss=" << best_decoder_loss
              << ", KAN params=" << deep_kan.num_params() << "\n";

    return result;
}

// =============================================================================
// Deep KAN v2: LibTorch training with CM-Feedback-Port
// =============================================================================

#ifdef USE_LIBTORCH
struct LanguageTraining::V2ConvergenceState {
    libtorch::ConvergencePortData cpd;
};

struct LanguageTraining::V2ConceptState {
    libtorch::ConceptWeights cw;
};
#endif

LanguageTrainingResult LanguageTraining::train_stage1_deep_kan_v2(const LanguageConfig& config) {
    LanguageTrainingResult result;
    result.stage = 1;
    result.stage_name = "DeepKAN v2 (LibTorch)";
    result.converged = false;
    result.epochs_run = 0;
    result.final_loss = 1e9;

#ifndef USE_LIBTORCH
    std::cerr << "[LanguageTraining] DeepKAN v2 requires USE_LIBTORCH. Build with TORCH=1.\n";
    return result;
#else
    // ── Set feature flags before data generation ──
    use_gat_ = config.use_gat;
    use_lstm_gates_ = config.use_lstm_gates;
    if (use_gat_)
        std::cerr << "[LanguageTraining] GAT attention enabled for target aggregation\n";
    if (use_lstm_gates_)
        std::cerr << "[LanguageTraining] LSTM-style gates enabled for hidden state evolution\n";

    // ── Generate training data (same pipeline as V12) ──
    std::cerr << "[LanguageTraining] Generating encoder data...\n";
    auto encoder_data = generate_encoder_data();
    std::cerr << "[LanguageTraining] Generating unified concept descriptions...\n";
    auto decoder_data = generate_decoder_data();
    std::cerr << "[LanguageTraining] Generating supplemental relation data...\n";
    auto relation_data = generate_relation_decoder_data();
    std::cerr << "[LanguageTraining] Data generated: enc=" << encoder_data.size()
              << " unified=" << decoder_data.size() << " rel=" << relation_data.size() << "\n";

    // Brief encoder training
    double best_loss = 1e9;
    if (!encoder_data.empty()) {
        std::cerr << "[LanguageTraining] Stage 1: Training encoder on "
                  << encoder_data.size() << " pairs...\n";
        for (size_t epoch = 0; epoch < config.encoder_epochs; ++epoch) {
            double loss = train_encoder_epoch(encoder_data, config.encoder_lr);
            if (loss < best_loss) best_loss = loss;
            if (loss < 1e-4) break;
        }
    }

    // Merge decoder data
    std::vector<DecoderPair> all_decoder_data;
    all_decoder_data.reserve(decoder_data.size() + relation_data.size());
    all_decoder_data.insert(all_decoder_data.end(),
        std::make_move_iterator(decoder_data.begin()),
        std::make_move_iterator(decoder_data.end()));
    all_decoder_data.insert(all_decoder_data.end(),
        std::make_move_iterator(relation_data.begin()),
        std::make_move_iterator(relation_data.end()));
    { std::mt19937 rng(12345); std::shuffle(all_decoder_data.begin(), all_decoder_data.end(), rng); }

    if (all_decoder_data.empty()) return result;

    std::cerr << "[LanguageTraining] Stage 1: Training Deep KAN v2 (LibTorch) on "
              << all_decoder_data.size() << " pairs...\n";

    // ── Pre-tokenize ──
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

    std::vector<uint16_t> active_tokens;
    std::vector<size_t> compress(V, 0);
    for (size_t v = 0; v < V; ++v) {
        if (seen[v]) {
            compress[v] = active_tokens.size();
            active_tokens.push_back(static_cast<uint16_t>(v));
        }
    }
    const size_t VA = active_tokens.size();
    std::cerr << "[LanguageTraining]   " << pretok_samples.size()
              << " samples, " << VA << " active tokens\n";

    // ── Dimensions (v2: 90D hidden, no convergence block in h) ──
    auto& decoder = engine_.decoder();
    auto& emb_table = engine_.encoder().embedding_table();
    const size_t H_OLD = decoder.extended_fused_dim();  // 122 (still used for old DeepKAN dim)
    const size_t FUSED_BASE = LanguageConfig::FUSED_DIM;  // 64
    const size_t fd = decoder.flex_dim();                 // 16
    const size_t H_90 = 90;  // v2: only Blocks 1-3
    const size_t FEAT_DIM = 128;

    // ── Pack TrainingData (reusing cuda struct for h precomputation) ──
    cuda::TrainingData td;
    td.num_samples = pretok_samples.size();
    td.V = V;
    td.VA = VA;
    td.H = H_OLD;
    td.FUSED_BASE = FUSED_BASE;
    td.flex_dim = fd;
    td.use_lstm_gates = config.use_lstm_gates;

    // Flatten tokens
    size_t total_toks = 0;
    for (const auto& s : pretok_samples) total_toks += s.tokens.size();
    td.all_tokens.reserve(total_toks);
    td.sample_offsets.resize(pretok_samples.size());
    td.sample_lengths.resize(pretok_samples.size());
    for (size_t i = 0; i < pretok_samples.size(); ++i) {
        td.sample_offsets[i] = td.all_tokens.size();
        td.sample_lengths[i] = pretok_samples[i].tokens.size();
        td.all_tokens.insert(td.all_tokens.end(),
            pretok_samples[i].tokens.begin(), pretok_samples[i].tokens.end());
    }

    // Embeddings [num_samples * H_OLD]
    td.embeddings.resize(pretok_samples.size() * H_OLD, 0.0);
    for (size_t i = 0; i < pretok_samples.size(); ++i) {
        const auto& emb = all_decoder_data[pretok_samples[i].pair_idx].embedding;
        for (size_t j = 0; j < std::min(H_OLD, emb.size()); ++j)
            td.embeddings[i * H_OLD + j] = emb[j];
    }

    td.compress.resize(V);
    for (size_t v = 0; v < V; ++v) td.compress[v] = compress[v];
    td.active_tokens = active_tokens;

    // Embedding table
    td.emb_table.resize(V * FUSED_BASE, 0.0);
    for (size_t v = 0; v < std::min(emb_table.size(), V); ++v) {
        for (size_t j = 0; j < std::min(emb_table[v].size(), FUSED_BASE); ++j)
            td.emb_table[v * FUSED_BASE + j] = emb_table[v][j];
    }

    // FlexDetail table
    td.flex_table.resize(V * fd, 0.0);
    for (size_t v = 0; v < V; ++v) {
        auto cpt = tok.token_to_concept(static_cast<uint16_t>(v));
        if (cpt) {
            auto flex_emb = engine_.embeddings().concept_embeddings().get_or_default(*cpt);
            for (size_t j = 0; j < std::min(fd, flex_emb.detail.size()); ++j)
                td.flex_table[v * fd + j] = flex_emb.detail[j];
        }
    }

    // conv_table not used in v2 (CM computed dynamically), but set conv_dim=0
    td.conv_dim = 0;

    // ── Construct C++ DeepKAN for weight initialization ──
    DeepKAN deep_kan({H_90, 256, 128, FEAT_DIM}, {8, 5, 5}, 3);

    // ── Extract initial weights ──
    cuda::DeepKANWeights dkw;
    auto& l1 = deep_kan.layer(0);
    // l2 unused: v2 L2 input is 288 (not 256), initialized fresh below
    auto& l3 = deep_kan.layer(2);

    dkw.k1_weights  = l1.spline_weights();
    dkw.k1_residual = l1.residual_W();
    dkw.k1_gamma    = l1.ln_gamma();
    dkw.k1_beta     = l1.ln_beta();
    dkw.k1_knots    = l1.knots();

    // v2: L2 input is 288 (256+32), but DeepKAN created with 256 input
    // We need fresh weights for the new L2 (288→128)
    {
        EfficientKANLayer l2_v2(288, 128, 5, 3);
        dkw.k2_weights  = l2_v2.spline_weights();
        dkw.k2_residual = l2_v2.residual_W();
        dkw.k2_gamma    = l2_v2.ln_gamma();
        dkw.k2_beta     = l2_v2.ln_beta();
        dkw.k2_knots    = l2_v2.knots();
    }

    dkw.k3_weights  = l3.spline_weights();
    dkw.k3_residual = l3.residual_W();
    dkw.k3_gamma    = l3.ln_gamma();
    dkw.k3_beta     = l3.ln_beta();
    dkw.k3_knots    = l3.knots();

    // Random init W_a [FEAT_DIM * VA]
    dkw.W_a.resize(FEAT_DIM * VA);
    {
        std::mt19937 rng(42);
        double scale = std::sqrt(6.0 / (double)(FEAT_DIM + VA));
        std::uniform_real_distribution<double> dist(-scale, scale);
        for (auto& w : dkw.W_a) w = dist(rng);
    }

    // ── ConvergencePort v2: shared Linear + Token Embedding ──
    // Empty init → LibTorch uses its own random init for conv_emb and conv_linear
    libtorch::ConvergencePortData cpd;

    // ══════════════════════════════════════════════════════════════════════════
    // Unified Training: Token + Concept in shared forward pass
    // Both heads trained simultaneously through shared KAN backbone.
    // Combined loss prevents catastrophic interference.
    // ══════════════════════════════════════════════════════════════════════════

    auto v2_cs = std::make_shared<V2ConceptState>();
    auto& cpt_emb_store = engine_.embeddings().concept_embeddings();

    {
        auto concept_data = generate_concept_decoder_data();

        if (!concept_data.empty()) {
            // ── Build concept vocabulary + ConceptTrainingData ──
            std::unordered_map<ConceptId, int64_t> concept_to_idx;
            std::vector<ConceptId> idx_to_concept;

            for (const auto& pair : concept_data) {
                if (concept_to_idx.find(pair.source_concept) == concept_to_idx.end()) {
                    concept_to_idx[pair.source_concept] = (int64_t)idx_to_concept.size();
                    idx_to_concept.push_back(pair.source_concept);
                }
                for (auto cid : pair.target_concepts) {
                    if (concept_to_idx.find(cid) == concept_to_idx.end()) {
                        concept_to_idx[cid] = (int64_t)idx_to_concept.size();
                        idx_to_concept.push_back(cid);
                    }
                }
            }

            size_t NC = idx_to_concept.size();
            std::cerr << "[Unified]   " << concept_data.size()
                      << " concept samples, " << NC << " unique concepts\n";

            libtorch::ConceptTrainingData ctd;
            ctd.num_samples = concept_data.size();
            ctd.num_concepts = NC;
            ctd.use_lstm_gates = config.use_lstm_gates;

            static constexpr size_t CPD = 64;  // CONCEPT_PROJ_DIM
            ctd.concept_matrix.resize(NC * CPD, 0.0);
            ctd.concept_emb_64d.resize(NC * FUSED_BASE, 0.0);
            ctd.concept_flex_16d.resize(NC * fd, 0.0);

            for (size_t i = 0; i < NC; ++i) {
                auto flex = cpt_emb_store.get_or_default(idx_to_concept[i]);
                ConceptId cid = idx_to_concept[i];

                // Build full 64D fused vector for this concept
                // (graph-aware: source emb + GAT target aggregation + relation types)
                auto rels = ltm_.get_outgoing_relations(cid);
                std::vector<FlexEmbedding> tgt_embs, rel_embs;
                std::vector<RelationType> rel_types;
                for (const auto& rel : rels) {
                    if (tgt_embs.size() >= 5) break;
                    tgt_embs.push_back(cpt_emb_store.get_or_default(rel.target));
                    rel_embs.push_back(engine_.embeddings().get_relation_embedding(rel.type));
                    rel_types.push_back(rel.type);
                }
                auto fused = build_concept_fused_vector(cid, tgt_embs, rel_embs, rel_types);

                // concept_matrix: first 64D of fused, L2-normalized
                for (size_t d = 0; d < CPD && d < fused.size(); ++d)
                    ctd.concept_matrix[i * CPD + d] = fused[d];
                double norm = 0.0;
                for (size_t d = 0; d < CPD; ++d)
                    norm += ctd.concept_matrix[i * CPD + d] * ctd.concept_matrix[i * CPD + d];
                norm = std::sqrt(norm);
                if (norm > 1e-8) {
                    for (size_t d = 0; d < CPD; ++d)
                        ctd.concept_matrix[i * CPD + d] /= norm;
                }

                // concept_emb_64d: full 64D fused projection for hidden state evolution
                for (size_t d = 0; d < FUSED_BASE && d < fused.size(); ++d)
                    ctd.concept_emb_64d[i * FUSED_BASE + d] = fused[d];

                // concept_flex_16d: detail embedding for Block 2
                for (size_t d = 0; d < fd && d < flex.detail.size(); ++d)
                    ctd.concept_flex_16d[i * fd + d] = flex.detail[d];
            }

            ctd.initial_h.resize(concept_data.size() * H_90, 0.0);
            ctd.seq_offsets.resize(concept_data.size());
            ctd.seq_lengths.resize(concept_data.size());
            ctd.trust_weights.resize(concept_data.size());

            for (size_t s = 0; s < concept_data.size(); ++s) {
                const auto& pair = concept_data[s];

                for (size_t i = 0; i < H_90 && i < pair.embedding.size(); ++i)
                    ctd.initial_h[s * H_90 + i] = pair.embedding[i];

                ctd.seq_offsets[s] = ctd.concept_seqs.size();
                ctd.concept_seqs.push_back(concept_to_idx[pair.source_concept]);
                for (auto tgt : pair.target_concepts)
                    ctd.concept_seqs.push_back(concept_to_idx[tgt]);
                ctd.seq_lengths[s] = 1 + pair.target_concepts.size();

                ctd.trust_weights[s] = pair.trust_weight;
            }

            // ── Unified training: token + concept through shared KAN backbone ──
            libtorch::UnifiedTrainingConfig ucfg;
            ucfg.num_epochs = config.unified_epochs;
            ucfg.lr_token_head = config.decoder_lr;
            ucfg.lr_concept_head = 0.002;
            ucfg.lr_kan = config.deep_kan_lr;
            ucfg.lr_conv = 0.0005;
            ucfg.warmup_epochs = 10;
            ucfg.batch_size = 2048;
            ucfg.dropout_p = 0.08;
            ucfg.weight_decay = 0.01;
            ucfg.patience = 25;
            ucfg.max_val_gap = 0.15;
            ucfg.concept_loss_weight = 1.0;
            ucfg.concept_temperature = (float)config.concept_train_temperature;
            ucfg.lr_scale.resize(H_90, 1.0);
            for (size_t i = FUSED_BASE; i < FUSED_BASE + fd && i < H_90; ++i)
                ucfg.lr_scale[i] = 0.3;
            for (size_t i = FUSED_BASE + fd; i < H_90; ++i)
                ucfg.lr_scale[i] = 0.1;

            libtorch::UnifiedTrainingResult uresult;
            std::cerr << "[Unified]   Starting unified training...\n";
            bool ok = libtorch::train_unified_deep_kan_v2(
                td, ctd, dkw, cpd, v2_cs->cw, ucfg, uresult);

            if (ok) {
                // Store concept state for inference
                v2_concept_state_ = v2_cs;
                v2_concept_matrix_ = ctd.concept_matrix;
                v2_concept_emb_64d_ = ctd.concept_emb_64d;
                v2_concept_flex_16d_ = ctd.concept_flex_16d;
                v2_num_concepts_ = NC;
                v2_idx_to_concept_ = idx_to_concept;
                v2_concept_valid_ = true;

                double unified_best = std::min(uresult.best_token_val, uresult.best_concept_val);
                if (unified_best < best_loss) best_loss = unified_best;

                std::cerr << "[Unified] Training complete: tok_val="
                          << uresult.best_token_val << " con_val="
                          << uresult.best_concept_val << "\n";
            }
        } else {
            std::cerr << "[Unified]   No concept data, falling back to token-only\n";
            libtorch::DeepKANv2Config dkc;
            dkc.num_epochs = config.decoder_epochs;
            dkc.lr_output = config.decoder_lr;
            dkc.lr_kan = config.deep_kan_lr;
            dkc.lr_conv = 0.0005;
            dkc.warmup_epochs = 10;
            dkc.batch_size = 2048;
            dkc.dropout_p = 0.05;
            dkc.weight_decay = 0.0;
            dkc.patience = 15;
            dkc.lr_scale.resize(H_90, 1.0);
            for (size_t i = FUSED_BASE; i < FUSED_BASE + fd && i < H_90; ++i)
                dkc.lr_scale[i] = 0.3;
            for (size_t i = FUSED_BASE + fd; i < H_90; ++i)
                dkc.lr_scale[i] = 0.1;
            cuda::TrainingResult tr;
            tr.best_loss = 1e9;
            libtorch::train_deep_kan_v2(td, dkw, cpd, dkc, tr);
            if (tr.best_loss < best_loss) best_loss = tr.best_loss;
        }
    }

    result.epochs_run = config.unified_epochs;

    // ── Writeback: W_a → decoder output projection ──
    auto& W = decoder.output_projection();
    W.resize(FEAT_DIM);
    for (size_t i = 0; i < FEAT_DIM; ++i) {
        W[i].assign(V, 0.0);
        for (size_t a = 0; a < VA; ++a)
            W[i][active_tokens[a]] = dkw.W_a[i * VA + a];
    }

    result.final_loss = best_loss;
    result.converged = best_loss < 0.1;

    // ── Store trained state for inference ──
    v2_dkw_ = dkw;
    v2_cpd_ = std::make_shared<V2ConvergenceState>();
    v2_cpd_->cpd = cpd;
    v2_emb_table_ = td.emb_table;
    v2_flex_table_ = td.flex_table;
    v2_active_tokens_ = active_tokens;
    v2_VA_ = VA;
    v2_V_ = V;
    v2_FUSED_BASE_ = FUSED_BASE;
    v2_flex_dim_ = fd;
    v2_valid_ = true;

    return result;
#endif
}

// =============================================================================
// Save/Load trained v2 state — binary format for fast checkpoint reload
// =============================================================================

// Helper: write a size-prefixed vector of T
namespace {
template<typename T>
bool write_vec(std::ofstream& out, const std::vector<T>& v) {
    uint64_t n = v.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (!v.empty())
        out.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
    return out.good();
}
template<typename T>
bool read_vec(std::ifstream& in, std::vector<T>& v) {
    uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in.good() || n > 100000000) return false;  // sanity limit: 100M elements
    v.resize(n);
    if (n > 0)
        in.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    return in.good();
}
template<typename T>
bool write_scalar(std::ofstream& out, T val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    return out.good();
}
template<typename T>
bool read_scalar(std::ifstream& in, T& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(val));
    return in.good();
}
} // anonymous namespace

bool LanguageTraining::save_v2_state(const std::string& path) const {
    if (!v2_valid_) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    // Magic + version
    const uint32_t magic = 0x4B414E32;  // "KAN2"
    const uint32_t version = 1;
    write_scalar(out, magic);
    write_scalar(out, version);

    // Token prediction state
    write_scalar(out, v2_VA_);
    write_scalar(out, v2_V_);
    write_scalar(out, v2_FUSED_BASE_);
    write_scalar(out, v2_flex_dim_);

    write_vec(out, v2_emb_table_);
    write_vec(out, v2_flex_table_);
    write_vec(out, v2_active_tokens_);

    // DeepKANWeights
    write_vec(out, v2_dkw_.k1_weights);
    write_vec(out, v2_dkw_.k1_residual);
    write_vec(out, v2_dkw_.k1_gamma);
    write_vec(out, v2_dkw_.k1_beta);
    write_vec(out, v2_dkw_.k1_knots);
    write_vec(out, v2_dkw_.k2_weights);
    write_vec(out, v2_dkw_.k2_residual);
    write_vec(out, v2_dkw_.k2_gamma);
    write_vec(out, v2_dkw_.k2_beta);
    write_vec(out, v2_dkw_.k2_knots);
    write_vec(out, v2_dkw_.k3_weights);
    write_vec(out, v2_dkw_.k3_residual);
    write_vec(out, v2_dkw_.k3_gamma);
    write_vec(out, v2_dkw_.k3_beta);
    write_vec(out, v2_dkw_.k3_knots);
    write_vec(out, v2_dkw_.W_a);

    // ConvergencePortData
#ifdef USE_LIBTORCH
    if (v2_cpd_) {
        write_scalar(out, (uint8_t)1);
        write_vec(out, v2_cpd_->cpd.conv_emb_weights);
        write_vec(out, v2_cpd_->cpd.conv_linear_W);
        write_vec(out, v2_cpd_->cpd.conv_linear_b);
    } else {
        write_scalar(out, (uint8_t)0);
    }

    // Concept state
    write_scalar(out, v2_concept_valid_);
    if (v2_concept_valid_ && v2_concept_state_) {
        write_scalar(out, (uint8_t)1);
        write_vec(out, v2_concept_state_->cw.concept_proj_W);
        write_vec(out, v2_concept_state_->cw.k1_proj_W);
        write_vec(out, v2_concept_state_->cw.k1_proj_b);
    } else {
        write_scalar(out, (uint8_t)0);
    }
#else
    write_scalar(out, (uint8_t)0);  // no cpd
    write_scalar(out, false);       // no concept state
    write_scalar(out, (uint8_t)0);
#endif

    // Concept prediction tables
    write_scalar(out, v2_num_concepts_);
    write_vec(out, v2_concept_matrix_);
    write_vec(out, v2_concept_emb_64d_);
    write_vec(out, v2_concept_flex_16d_);
    write_vec(out, v2_idx_to_concept_);

    std::cerr << "[LanguageTraining] Saved v2 state to " << path
              << " (" << out.tellp() << " bytes)\n";
    return out.good();
}

bool LanguageTraining::load_v2_state(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint32_t magic = 0, version = 0;
    read_scalar(in, magic);
    read_scalar(in, version);
    if (magic != 0x4B414E32 || version != 1) {
        std::cerr << "[LanguageTraining] Invalid v2 state file (magic="
                  << std::hex << magic << " version=" << std::dec << version << ")\n";
        return false;
    }

    // Token prediction state
    read_scalar(in, v2_VA_);
    read_scalar(in, v2_V_);
    read_scalar(in, v2_FUSED_BASE_);
    read_scalar(in, v2_flex_dim_);

    if (!read_vec(in, v2_emb_table_)) return false;
    if (!read_vec(in, v2_flex_table_)) return false;
    if (!read_vec(in, v2_active_tokens_)) return false;

    // DeepKANWeights
    if (!read_vec(in, v2_dkw_.k1_weights)) return false;
    if (!read_vec(in, v2_dkw_.k1_residual)) return false;
    if (!read_vec(in, v2_dkw_.k1_gamma)) return false;
    if (!read_vec(in, v2_dkw_.k1_beta)) return false;
    if (!read_vec(in, v2_dkw_.k1_knots)) return false;
    if (!read_vec(in, v2_dkw_.k2_weights)) return false;
    if (!read_vec(in, v2_dkw_.k2_residual)) return false;
    if (!read_vec(in, v2_dkw_.k2_gamma)) return false;
    if (!read_vec(in, v2_dkw_.k2_beta)) return false;
    if (!read_vec(in, v2_dkw_.k2_knots)) return false;
    if (!read_vec(in, v2_dkw_.k3_weights)) return false;
    if (!read_vec(in, v2_dkw_.k3_residual)) return false;
    if (!read_vec(in, v2_dkw_.k3_gamma)) return false;
    if (!read_vec(in, v2_dkw_.k3_beta)) return false;
    if (!read_vec(in, v2_dkw_.k3_knots)) return false;
    if (!read_vec(in, v2_dkw_.W_a)) return false;

    // ConvergencePortData
#ifdef USE_LIBTORCH
    uint8_t has_cpd = 0;
    read_scalar(in, has_cpd);
    if (has_cpd) {
        v2_cpd_ = std::make_shared<V2ConvergenceState>();
        if (!read_vec(in, v2_cpd_->cpd.conv_emb_weights)) return false;
        if (!read_vec(in, v2_cpd_->cpd.conv_linear_W)) return false;
        if (!read_vec(in, v2_cpd_->cpd.conv_linear_b)) return false;
    }

    // Concept state
    bool concept_valid_flag = false;
    read_scalar(in, concept_valid_flag);
    uint8_t has_concept = 0;
    read_scalar(in, has_concept);
    if (has_concept) {
        v2_concept_state_ = std::make_shared<V2ConceptState>();
        if (!read_vec(in, v2_concept_state_->cw.concept_proj_W)) return false;
        if (!read_vec(in, v2_concept_state_->cw.k1_proj_W)) return false;
        if (!read_vec(in, v2_concept_state_->cw.k1_proj_b)) return false;
    }
    v2_concept_valid_ = concept_valid_flag && has_concept;
#else
    // Skip cpd and concept sections
    uint8_t skip8 = 0;
    read_scalar(in, skip8); // has_cpd
    if (skip8) {
        std::vector<double> tmp;
        read_vec(in, tmp); read_vec(in, tmp); read_vec(in, tmp);
    }
    bool skip_bool = false;
    read_scalar(in, skip_bool);
    read_scalar(in, skip8); // has_concept
    if (skip8) {
        std::vector<double> tmp;
        read_vec(in, tmp); read_vec(in, tmp); read_vec(in, tmp);
    }
    v2_concept_valid_ = false;
#endif

    // Concept prediction tables
    read_scalar(in, v2_num_concepts_);
    if (!read_vec(in, v2_concept_matrix_)) return false;
    if (!read_vec(in, v2_concept_emb_64d_)) return false;
    if (!read_vec(in, v2_concept_flex_16d_)) return false;
    if (!read_vec(in, v2_idx_to_concept_)) return false;

    v2_valid_ = true;

    // Also writeback W_a to decoder output projection
    auto& decoder = engine_.decoder();
    auto& W = decoder.output_projection();
    const size_t FEAT_DIM = 128;
    W.resize(FEAT_DIM);
    for (size_t i = 0; i < FEAT_DIM; ++i) {
        W[i].assign(v2_V_, 0.0);
        for (size_t a = 0; a < v2_VA_ && a < v2_active_tokens_.size(); ++a)
            W[i][v2_active_tokens_[a]] = v2_dkw_.W_a[i * v2_VA_ + a];
    }

    std::cerr << "[LanguageTraining] Loaded v2 state from " << path
              << " (VA=" << v2_VA_ << ", concepts=" << v2_num_concepts_ << ")\n";
    return true;
}

#ifdef USE_LIBTORCH
// =============================================================================
// KAN → Graph Feedback: propose new relations from KAN inference
// =============================================================================

size_t LanguageTraining::apply_kan_graph_feedback(
    const LanguageConfig& config,
    const std::vector<ConceptId>& idx_to_concept,
    size_t num_concepts)
{
    if (!v2_concept_valid_ || !v2_concept_state_ || !v2_cpd_) return 0;

    size_t sample_size = std::min(config.kan_feedback_sample_size, num_concepts);
    if (sample_size == 0) return 0;

    // Sample random concept indices
    std::vector<size_t> sampled(num_concepts);
    std::iota(sampled.begin(), sampled.end(), 0);
    {
        std::mt19937 rng(std::random_device{}());
        std::shuffle(sampled.begin(), sampled.end(), rng);
    }
    sampled.resize(sample_size);

    // Track proposed pairs to avoid duplicates within this round
    std::unordered_set<uint64_t> proposed_pairs;
    auto pair_key = [](ConceptId a, ConceptId b) -> uint64_t {
        return ((uint64_t)a << 32) | (uint64_t)b;
    };

    size_t added = 0;

    for (size_t si = 0; si < sample_size && added < config.kan_feedback_max_relations; ++si) {
        size_t concept_idx = sampled[si];
        ConceptId source = idx_to_concept[concept_idx];

        // Build initial_h from fused embedding (same as generate_v2 lines 2870-2873)
        std::vector<float> h(90, 0.0f);
        for (size_t d = 0; d < 64 && d < v2_concept_emb_64d_.size() - concept_idx * 64; ++d)
            h[d] = (float)v2_concept_emb_64d_[concept_idx * 64 + d];
        for (size_t d = 0; d < 16 && d < v2_concept_flex_16d_.size() - concept_idx * 16; ++d)
            h[64 + d] = (float)v2_concept_flex_16d_[concept_idx * 16 + d];

        auto cgr = libtorch::generate_concept_deep_kan_v2(
            v2_dkw_, v2_cpd_->cpd, v2_concept_state_->cw,
            v2_concept_matrix_, v2_concept_emb_64d_, v2_concept_flex_16d_,
            v2_num_concepts_, h, (int64_t)concept_idx, 5,
            (float)config.concept_inference_temperature,
            use_lstm_gates_);

        for (size_t p = 0; p < cgr.concept_indices.size(); ++p) {
            if (cgr.confidences[p] < config.kan_feedback_min_confidence) continue;

            auto target_idx = cgr.concept_indices[p];
            if (target_idx < 0 || (size_t)target_idx >= num_concepts) continue;
            ConceptId target = idx_to_concept[(size_t)target_idx];
            if (target == source) continue;

            uint64_t pk = pair_key(source, target);
            if (proposed_pairs.count(pk)) continue;

            // Check if relation already exists
            auto existing = ltm_.get_relations_between(source, target);
            if (!existing.empty()) continue;

            // Propose new ASSOCIATED_WITH relation
            ltm_.add_relation(source, target, RelationType::ASSOCIATED_WITH,
                              config.kan_feedback_relation_weight);
            proposed_pairs.insert(pk);
            ++added;

            if (added >= config.kan_feedback_max_relations) break;
        }
    }

    return added;
}

// =============================================================================
// CM → Graph Feedback: adjust epistemic trust from KAN learning signal
// =============================================================================

size_t LanguageTraining::apply_cm_trust_feedback(
    const LanguageConfig& config,
    const std::vector<ConceptId>& idx_to_concept,
    size_t num_concepts)
{
    if (!v2_concept_valid_ || !v2_concept_state_ || !v2_cpd_) return 0;

    size_t sample_size = std::min(config.kan_feedback_sample_size, num_concepts);
    if (sample_size == 0) return 0;

    // Sample random concept indices (same seed strategy as KAN feedback)
    std::vector<size_t> sampled(num_concepts);
    std::iota(sampled.begin(), sampled.end(), 0);
    {
        std::mt19937 rng(std::random_device{}());
        std::shuffle(sampled.begin(), sampled.end(), rng);
    }
    sampled.resize(sample_size);

    size_t adjustments = 0;

    for (size_t si = 0; si < sample_size; ++si) {
        size_t concept_idx = sampled[si];
        ConceptId source = idx_to_concept[concept_idx];

        // Get actual outgoing relations
        auto actual_rels = ltm_.get_outgoing_relations(source);
        if (actual_rels.size() < config.cm_min_samples_for_adjust) continue;

        // Run inference to see what KAN predicts
        std::vector<float> h(90, 0.0f);
        for (size_t d = 0; d < 64 && d < v2_concept_emb_64d_.size() - concept_idx * 64; ++d)
            h[d] = (float)v2_concept_emb_64d_[concept_idx * 64 + d];
        for (size_t d = 0; d < 16 && d < v2_concept_flex_16d_.size() - concept_idx * 16; ++d)
            h[64 + d] = (float)v2_concept_flex_16d_[concept_idx * 16 + d];

        auto cgr = libtorch::generate_concept_deep_kan_v2(
            v2_dkw_, v2_cpd_->cpd, v2_concept_state_->cw,
            v2_concept_matrix_, v2_concept_emb_64d_, v2_concept_flex_16d_,
            v2_num_concepts_, h, (int64_t)concept_idx,
            std::min((size_t)10, actual_rels.size() * 2),
            (float)config.concept_inference_temperature,
            use_lstm_gates_);

        // Build set of predicted target concept IDs
        std::unordered_set<ConceptId> predicted_targets;
        for (size_t p = 0; p < cgr.concept_indices.size(); ++p) {
            auto tidx = cgr.concept_indices[p];
            if (tidx >= 0 && (size_t)tidx < num_concepts)
                predicted_targets.insert(idx_to_concept[(size_t)tidx]);
        }

        // Count how many actual relations were confirmed by KAN
        size_t confirmed = 0;
        for (const auto& rel : actual_rels) {
            if (predicted_targets.count(rel.target))
                ++confirmed;
        }
        double confirmed_ratio = (double)confirmed / (double)actual_rels.size();

        // Get concept info for trust adjustment
        auto cinfo = ltm_.retrieve_concept(source);
        if (!cinfo) continue;

        double trust = cinfo->epistemic.trust;
        double new_trust = trust;

        // High-trust concept but KAN couldn't learn its relations → suspicious
        if (trust >= 0.6 && confirmed_ratio < 0.2) {
            double delta = std::min(config.cm_trust_adjustment_max, trust * 0.1);
            new_trust = trust - delta;
        }
        // Low-trust concept but KAN learned its relations well → deserves more trust
        else if (trust <= 0.4 && confirmed_ratio > 0.8) {
            double delta = std::min(config.cm_trust_adjustment_max, (1.0 - trust) * 0.1);
            new_trust = trust + delta;
        }

        if (std::abs(new_trust - trust) > 1e-6) {
            EpistemicMetadata updated_meta(cinfo->epistemic.type,
                                           cinfo->epistemic.status,
                                           new_trust);
            ltm_.update_epistemic_metadata(source, updated_meta);
            ++adjustments;
        }
    }

    return adjustments;
}
#endif  // USE_LIBTORCH

// =============================================================================
// DeepKAN v2 inference: generate text for a query
// =============================================================================

std::string LanguageTraining::generate_v2(const std::string& query, size_t max_tokens) const {
#ifndef USE_LIBTORCH
    (void)query; (void)max_tokens;
    return "[DeepKAN v2 requires USE_LIBTORCH]";
#else
    if (!v2_valid_) return "[No trained DeepKAN v2 model]";

    auto& tok = engine_.tokenizer();
    auto& concept_emb_store = engine_.embeddings().concept_embeddings();
    auto& dim_ctx = engine_.dim_context();
    auto& projection = engine_.fusion().projection();
    const size_t ACT_DIM = LanguageConfig::ENCODER_QUERY_DIM;  // 16
    const size_t FUSED = LanguageConfig::FUSED_DIM;             // 64

    // Find seed concepts matching query
    auto seeds = engine_.find_seeds(query);
    if (seeds.empty()) return "[No concepts found for: " + query + "]";

    ConceptId primary = seeds[0];

    // ── Build fused vector EXACTLY matching generate_decoder_data() ──
    // Raw activation: [src_emb(16) | target_emb(16) | rel_type_emb(16) | gates(5) | templates(4+)]
    std::vector<double> raw(3 * ACT_DIM + 5 + LanguageConfig::NUM_TEMPLATE_TYPES, 0.0);

    auto src_emb = concept_emb_store.get_or_default(primary);
    for (size_t d = 0; d < ACT_DIM; ++d)
        raw[d] = src_emb.core[d] * 0.7;

    // Gather relation targets + types for richer fused vector
    auto rels = ltm_.get_outgoing_relations(primary);
    std::vector<FlexEmbedding> target_embs, rel_type_embs;
    for (const auto& rel : rels) {
        if (target_embs.size() < 5)
            target_embs.push_back(concept_emb_store.get_or_default(rel.target));
        if (rel_type_embs.size() < 5)
            rel_type_embs.push_back(engine_.embeddings().get_relation_embedding(rel.type));
    }
    if (!target_embs.empty()) {
        double inv_n = 0.5 / (double)target_embs.size();
        for (const auto& te : target_embs)
            for (size_t d = 0; d < ACT_DIM; ++d)
                raw[ACT_DIM + d] += te.core[d] * inv_n;
    }
    if (!rel_type_embs.empty()) {
        double inv_n = 0.3 / (double)rel_type_embs.size();
        for (const auto& re : rel_type_embs)
            for (size_t d = 0; d < ACT_DIM; ++d)
                raw[2 * ACT_DIM + d] += re.core[d] * inv_n;
    }

    // Gates
    size_t gate_offset = 3 * ACT_DIM;
    raw[gate_offset + 0] = 0.8;
    raw[gate_offset + 1] = 0.5;
    raw[gate_offset + 2] = 0.3;
    // Template slots: relation type distribution (must match build_concept_fused_vector)
    size_t tpl_offset = gate_offset + 5;
    if (!rels.empty()) {
        double inv_n = 1.0 / (double)rels.size();
        for (const auto& rel : rels) {
            switch (rel.type) {
                case RelationType::IS_A:
                case RelationType::INSTANCE_OF:
                case RelationType::DERIVED_FROM:
                    raw[tpl_offset + 0] += inv_n; break;
                case RelationType::CAUSES:
                case RelationType::ENABLES:
                case RelationType::PRODUCES:
                case RelationType::IMPLIES:
                    raw[tpl_offset + 1] += inv_n; break;
                case RelationType::HAS_PROPERTY:
                case RelationType::PART_OF:
                case RelationType::HAS_PART:
                case RelationType::REQUIRES:
                case RelationType::USES:
                    raw[tpl_offset + 2] += inv_n; break;
                default:
                    raw[tpl_offset + 3] += inv_n; break;
            }
        }
    } else {
        raw[tpl_offset + 0] = 0.3;
        raw[tpl_offset + 1] = 0.5;
    }

    // Project: raw × projection → R^64
    std::vector<double> fused(FUSED, 0.0);
    size_t raw_dim = std::min(raw.size(), projection.size());
    for (size_t i = 0; i < raw_dim; ++i) {
        if (std::abs(raw[i]) < 1e-12) continue;
        for (size_t j = 0; j < FUSED; ++j)
            fused[j] += raw[i] * projection[i][j];
    }

    // Flex detail (16D)
    auto flex_emb = concept_emb_store.get_or_default(primary);
    for (size_t d = 0; d < 16; ++d)
        fused.push_back(d < flex_emb.detail.size() ? flex_emb.detail[d] : 0.0);

    // Dimensional context
    if (dim_ctx.is_built()) {
        auto dim_vec = dim_ctx.to_decoder_vec(primary);
        fused.insert(fused.end(), dim_vec.begin(), dim_vec.end());
    }

    // Take first 90D as initial hidden state (no convergence block — model computes that)
    std::vector<float> h(90, 0.0f);
    for (size_t i = 0; i < std::min((size_t)90, fused.size()); ++i)
        h[i] = (float)fused[i];

    // ── Concept prediction path (preferred when available) ──
    if (v2_concept_valid_ && v2_concept_state_) {
        // Find primary concept's index in concept vocabulary
        int64_t start_idx = -1;
        for (size_t i = 0; i < v2_idx_to_concept_.size(); ++i) {
            if (v2_idx_to_concept_[i] == primary) {
                start_idx = (int64_t)i;
                break;
            }
        }

        if (start_idx >= 0) {
            auto cgr = libtorch::generate_concept_deep_kan_v2(
                v2_dkw_, v2_cpd_->cpd, v2_concept_state_->cw,
                v2_concept_matrix_, v2_concept_emb_64d_, v2_concept_flex_16d_,
                v2_num_concepts_, h, start_idx, max_tokens,
                (float)LanguageConfig().concept_inference_temperature,
                use_lstm_gates_);

            if (!cgr.concept_indices.empty()) {
                // Build text from predicted concept sequence
                std::string result;
                std::string debug;
                for (size_t i = 0; i < cgr.concept_indices.size(); ++i) {
                    auto idx = cgr.concept_indices[i];
                    if (idx < 0 || (size_t)idx >= v2_idx_to_concept_.size()) continue;
                    ConceptId cid = v2_idx_to_concept_[(size_t)idx];
                    auto cinfo = ltm_.retrieve_concept(cid);
                    std::string label = cinfo ? cinfo->label : "?";

                    if (!result.empty()) result += ", ";
                    result += label;

                    debug += "[" + label + " p="
                        + std::to_string(cgr.confidences[i]).substr(0, 4) + "] ";
                }
                std::cerr << "[Inference/Concept] " << query << ": " << debug << "\n";
                return result;
            }
        }
    }

    // ── Token prediction fallback ──
    uint16_t start_tok = 0;
    auto tok_opt = tok.concept_to_token(primary);
    if (tok_opt) start_tok = *tok_opt;

    auto gr = libtorch::generate_deep_kan_v2(
        v2_dkw_, v2_cpd_->cpd,
        v2_VA_, v2_V_,
        v2_emb_table_, v2_flex_table_,
        v2_FUSED_BASE_, v2_flex_dim_,
        v2_active_tokens_,
        h, start_tok, max_tokens,
        use_lstm_gates_);

    if (gr.tokens.empty()) return "[No tokens generated]";

    std::string debug = "";
    for (size_t i = 0; i < gr.tokens.size(); ++i) {
        std::string piece = tok.decode({gr.tokens[i]});
        debug += "[" + piece + " p=" + std::to_string(gr.probs[i]).substr(0, 4) + "] ";
    }
    std::cerr << "[Inference/Token] " << query << ": " << debug << "\n";

    return tok.decode(gr.tokens);
#endif
}

} // namespace brain19

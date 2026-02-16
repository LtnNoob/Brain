#include "kan_language_engine.hpp"
#include "../memory/relation_type_registry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <unordered_set>

namespace brain19 {

// =============================================================================
// Construction & Initialization
// =============================================================================

KANLanguageEngine::KANLanguageEngine(
    const LanguageConfig& config,
    LongTermMemory& ltm,
    ConceptModelRegistry& registry,
    EmbeddingManager& embeddings
)
    : config_(config)
    , encoder_(config)
    , decoder_(config)
    , scorer_(config)
    , fusion_(config)
    , ltm_(ltm)
    , registry_(registry)
    , embeddings_(embeddings)
{}

void KANLanguageEngine::initialize() {
    std::cerr << "[KANLanguageEngine] Building tokenizer...\n";

    // Build BPE corpus from concept definitions + relations for richer vocabulary
    std::vector<std::string> bpe_corpus;
    {
        auto all_ids = ltm_.get_all_concept_ids();
        bpe_corpus.reserve(all_ids.size() * 2);
        for (auto cid : all_ids) {
            auto info = ltm_.retrieve_concept(cid);
            if (!info) continue;
            if (!info->definition.empty())
                bpe_corpus.push_back(info->label + " is " + info->definition);
            for (const auto& rel : ltm_.get_outgoing_relations(cid)) {
                auto tgt = ltm_.retrieve_concept(rel.target);
                if (tgt)
                    bpe_corpus.push_back(info->label + " " + tgt->label);
            }
        }
    }
    // Train BPE with enough merges for 2000+ active tokens
    // Target: 4000 total vocab (1261 base + ~2739 BPE merges)
    tokenizer_.train(bpe_corpus, ltm_, 4000);
    std::cerr << "[KANLanguageEngine] Tokenizer built: " << tokenizer_.vocab_size()
              << " tokens (" << (tokenizer_.vocab_size() - 1261) << " BPE merges)\n";

    // Build dimensional context from graph structure.
    // Discovers dimensions from actual relation categories — variable per concept.
    std::cerr << "[KANLanguageEngine] Building dimensional context...\n";
    dim_context_.build(ltm_);
    std::cerr << "[KANLanguageEngine] Dimensional context built: "
              << dim_context_.observed_dimensions() << " observed dims, "
              << "max concept dimensionality=" << dim_context_.max_dimensionality()
              << ", decoder_dim=" << dim_context_.decoder_dim() << "\n";

    // v11: Set flex_dim=16 for FlexDetail injection between fused and dim_context
    decoder_.set_flex_dim(16);

    // Re-initialize decoder projection for the extended dimension:
    // FUSED_DIM (64) + flex_dim (v11: 16) + dim_context_.decoder_dim() (variable) + CONVERGENCE_DIM (32)
    size_t ext_dim = LanguageConfig::FUSED_DIM + decoder_.flex_dim()
                   + dim_context_.decoder_dim() + LanguageConfig::CONVERGENCE_DIM;
    std::cerr << "[KANLanguageEngine] Reinitializing decoder for extended_fused_dim=" << ext_dim
              << " (flex_dim=" << decoder_.flex_dim()
              << ", conv_dim=" << LanguageConfig::CONVERGENCE_DIM << ")\n";
    decoder_.reinitialize_for_extended_dim(ext_dim);
    std::cerr << "[KANLanguageEngine] Decoder reinitialized OK\n";

    initialized_ = true;
}

void KANLanguageEngine::rebuild_dimensional_context() {
    dim_context_.build(ltm_);

    size_t ext_dim = LanguageConfig::FUSED_DIM + decoder_.flex_dim()
                   + dim_context_.decoder_dim() + LanguageConfig::CONVERGENCE_DIM;
    decoder_.reinitialize_for_extended_dim(ext_dim);

    std::cerr << "[KANLanguageEngine] Rebuilt dim context: "
              << dim_context_.observed_dimensions() << " observed dims, "
              << "max_dim=" << dim_context_.max_dimensionality()
              << ", decoder_dim=" << dim_context_.decoder_dim()
              << ", extended_fused=" << ext_dim << "\n";
}

// =============================================================================
// Main Generation Pipeline
// =============================================================================

LanguageResult KANLanguageEngine::generate(const std::string& query, size_t max_tokens) const {
    LanguageResult result;
    result.confidence = 0.0;
    result.used_template = false;
    result.template_type = 1;  // DEFINITIONAL
    result.tokens_generated = 0;

    if (!initialized_) {
        result.text = "[Language engine not initialized]";
        result.used_template = true;
        return result;
    }

    // ── Step 1: Encode query → query embedding ──
    auto query_embedding = encode(query);

    // ── Step 2: Find seed concepts in LTM ──
    auto seeds = find_seeds(query);
    if (seeds.empty()) {
        result.text = "I don't have knowledge about that topic.";
        result.used_template = true;
        return result;
    }
    result.activated_concepts = seeds;

    // ── Step 3: Build concept activations (1-hop subgraph) ──
    auto activations = build_activations(seeds, query_embedding);

    // ── Step 4: Extract causal chain ──
    auto causal_chain = extract_causal_chain(seeds);
    result.causal_chain = causal_chain;

    // ── Step 5: Build causal pairs for scoring ──
    std::vector<std::pair<ConceptId, ConceptId>> causal_pairs;
    for (size_t i = 0; i + 1 < causal_chain.size(); ++i) {
        causal_pairs.push_back({causal_chain[i], causal_chain[i + 1]});
    }
    // Also add direct seed-to-neighbor pairs
    for (auto sid : seeds) {
        auto rels = ltm_.get_outgoing_relations(sid);
        for (const auto& r : rels) {
            if (activations.count(r.target)) {
                causal_pairs.push_back({r.source, r.target});
            }
        }
    }

    // ── Step 6: Semantic scoring ──
    auto rel_embeddings = build_relation_embeddings(causal_pairs);
    auto scores = scorer_.score(activations, query_embedding, causal_pairs, rel_embeddings);
    result.template_type = scores.best_template();

    // ── Step 7: Fusion ──
    auto fused = fusion_.fuse(activations, scores, causal_chain);

    // ── Step 7b: Attach dimensional context (variable-length, emergent) ──
    if (dim_context_.is_built()) {
        fused.dimensional_context = dim_context_.average_decoder_vec(seeds);
    }

    // ── Step 7c: Compute average FlexDetail for seed concepts (16D, zero-padded) ──
    {
        const auto& emb_store = embeddings_.concept_embeddings();
        const size_t FLEX_DIM = 16;
        fused.flex_detail.assign(FLEX_DIM, 0.0);
        size_t count = 0;
        for (auto sid : seeds) {
            auto flex_emb = emb_store.get_or_default(sid);
            if (!flex_emb.detail.empty()) {
                for (size_t d = 0; d < FLEX_DIM; ++d) {
                    if (d < flex_emb.detail.size()) {
                        fused.flex_detail[d] += flex_emb.detail[d];
                    }
                }
                ++count;
            }
        }
        if (count > 0) {
            double inv = 1.0 / static_cast<double>(count);
            for (auto& v : fused.flex_detail) v *= inv;
        }
    }

    // ── Step 8: Try concept prediction first ──
    auto ext_fused = fused.extended_fused_vector();
    auto concept_output = decoder_.decode_concepts(
        ext_fused, embeddings_.concept_embeddings());

    if (!concept_output.concept_ids.empty() &&
        concept_output.confidence >= config_.decoder_confidence_threshold) {
        // Concept prediction succeeded — build response from concept graph
        result.tokens_generated = concept_output.concept_ids.size();
        result.confidence = concept_output.confidence;
        result.activated_concepts = concept_output.concept_ids;

        // Build rich text from predicted concepts using template_generate
        result.text = template_generate(concept_output.concept_ids, result.template_type);
        result.used_template = false;
        return result;
    }

    // ── Step 8b: Fallback to token-based decode ──
    auto decoder_output = decoder_.decode(
        fused, tokenizer_, encoder_.embedding_table(),
        embeddings_.concept_embeddings(), max_tokens);

    result.tokens_generated = decoder_output.token_ids.size();
    result.confidence = decoder_output.confidence;

    // ── Step 9: Template fallback if decoder confidence is too low ──
    if (decoder_output.used_template || decoder_output.confidence < config_.decoder_confidence_threshold) {
        result.text = template_generate(
            fused.ordered_concepts.empty() ? seeds : fused.ordered_concepts,
            result.template_type);
        result.used_template = true;
    } else {
        result.text = decoder_output.text;
    }

    return result;
}

// =============================================================================
// Concept-Based Generation
// =============================================================================

LanguageResult KANLanguageEngine::generate_concept_response(const std::string& query) const {
    // Delegates to generate() which now tries concept prediction first
    return generate(query);
}

// =============================================================================
// Encode
// =============================================================================

std::vector<double> KANLanguageEngine::encode(const std::string& text) const {
    return encoder_.encode(text, tokenizer_);
}

// =============================================================================
// Seed Selection
// =============================================================================

std::vector<ConceptId> KANLanguageEngine::find_seeds(const std::string& text) const {
    return label_search(text);
}

std::vector<ConceptId> KANLanguageEngine::label_search(const std::string& text) const {
    // Lowercase query
    std::string lower_q = text;
    for (auto& c : lower_q) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    // Extract keywords (words >= 3 chars, no stop words)
    std::vector<std::string> keywords;
    std::string word;
    for (char c : lower_q) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            word += c;
        } else {
            if (word.size() >= 3) keywords.push_back(word);
            word.clear();
        }
    }
    if (word.size() >= 3) keywords.push_back(word);

    // Score concepts
    struct ScoredConcept {
        ConceptId id;
        double score;
    };
    std::vector<ScoredConcept> scored;

    for (auto cid : ltm_.get_all_concept_ids()) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;

        std::string lower_label = info->label;
        for (auto& c : lower_label) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        double score = 0.0;

        // Exact label match
        if (lower_q.find(lower_label) != std::string::npos && lower_label.size() >= 3) {
            score += 5.0 * (static_cast<double>(lower_label.size()) / lower_q.size());
        }

        // Keyword match
        for (const auto& kw : keywords) {
            if (lower_label.find(kw) != std::string::npos) {
                score += 3.0;
            }
        }

        // Epistemic trust boost
        score *= (0.8 + 0.2 * info->epistemic.trust);

        if (score > 0.0) {
            scored.push_back({cid, score});
        }
    }

    // Sort and take top seeds
    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.score > b.score; });

    std::vector<ConceptId> seeds;
    for (size_t i = 0; i < std::min(scored.size(), size_t(8)); ++i) {
        seeds.push_back(scored[i].id);
    }
    return seeds;
}

// =============================================================================
// Build Activations
// =============================================================================

std::unordered_map<ConceptId, std::vector<double>> KANLanguageEngine::build_activations(
    const std::vector<ConceptId>& seeds,
    const std::vector<double>& query_embedding
) const {
    std::unordered_map<ConceptId, std::vector<double>> activations;
    std::unordered_set<ConceptId> visited;

    // Expand seeds + 1-hop neighbors
    std::vector<ConceptId> subgraph;
    for (auto sid : seeds) {
        if (!visited.count(sid)) {
            subgraph.push_back(sid);
            visited.insert(sid);
        }
        auto rels = ltm_.get_outgoing_relations(sid);
        for (const auto& r : rels) {
            if (!visited.count(r.target) && visited.size() < 100) {
                subgraph.push_back(r.target);
                visited.insert(r.target);
            }
        }
    }

    // Build activations using ConceptModel predictions
    const size_t dim = LanguageConfig::ENCODER_QUERY_DIM;  // 16

    for (auto cid : subgraph) {
        auto* model = registry_.get_model(cid);
        if (model) {
            // Use concept embedding as activation basis
            const auto& emb_store = embeddings_.concept_embeddings();
            auto cid_emb = emb_store.get_or_default(cid);

            std::vector<double> activation(dim, 0.0);
            // Check if embedding has non-zero values
            bool has_embedding = false;
            for (size_t i = 0; i < CORE_DIM; ++i) {
                if (std::abs(cid_emb.core[i]) > 1e-12) {
                    has_embedding = true;
                    break;
                }
            }
            if (has_embedding) {
                // Use core embedding as activation
                for (size_t i = 0; i < std::min(dim, size_t(CORE_DIM)); ++i) {
                    activation[i] = cid_emb.core[i];
                }
            } else {
                // Generate activation from query embedding (scaled)
                for (size_t i = 0; i < std::min(dim, query_embedding.size()); ++i) {
                    activation[i] = query_embedding[i] * 0.5;
                }
            }

            // Modulate by query relevance (dot product)
            double relevance = 0.0;
            for (size_t i = 0; i < std::min(activation.size(), query_embedding.size()); ++i) {
                relevance += activation[i] * query_embedding[i];
            }
            // Boost seeds
            bool is_seed = false;
            for (auto sid : seeds) {
                if (sid == cid) { is_seed = true; break; }
            }
            double boost = is_seed ? 1.5 : 1.0;
            for (auto& v : activation) v *= boost;

            activations[cid] = std::move(activation);
        } else {
            // No model — use zero activation
            activations[cid] = std::vector<double>(dim, 0.0);
        }
    }

    return activations;
}

// =============================================================================
// Causal Chain Extraction
// =============================================================================

std::vector<ConceptId> KANLanguageEngine::extract_causal_chain(
    const std::vector<ConceptId>& seeds
) const {
    // Walk CAUSES/ENABLES/PRODUCES relations from seeds
    std::vector<ConceptId> chain;
    std::unordered_set<ConceptId> in_chain;

    for (auto sid : seeds) {
        if (in_chain.count(sid)) continue;
        chain.push_back(sid);
        in_chain.insert(sid);

        // Follow causal relations (up to 5 hops)
        ConceptId current = sid;
        for (size_t hop = 0; hop < 5; ++hop) {
            auto rels = ltm_.get_outgoing_relations(current);
            bool found = false;
            for (const auto& r : rels) {
                if ((r.type == RelationType::CAUSES ||
                     r.type == RelationType::ENABLES ||
                     r.type == RelationType::PRODUCES) &&
                    !in_chain.count(r.target)) {
                    chain.push_back(r.target);
                    in_chain.insert(r.target);
                    current = r.target;
                    found = true;
                    break;
                }
            }
            if (!found) break;
        }
    }
    return chain;
}

// =============================================================================
// Relation Embeddings
// =============================================================================

std::unordered_map<std::string, std::vector<double>> KANLanguageEngine::build_relation_embeddings(
    const std::vector<std::pair<ConceptId, ConceptId>>& pairs
) const {
    std::unordered_map<std::string, std::vector<double>> result;
    auto& reg = RelationTypeRegistry::instance();

    for (const auto& [src, tgt] : pairs) {
        std::string key = std::to_string(src) + ":" + std::to_string(tgt);
        if (result.count(key)) continue;

        // Find the relation type between these concepts
        auto rels = ltm_.get_relations_between(src, tgt);
        if (!rels.empty()) {
            const auto& rel_info = reg.get(rels[0].type);
            // Convert FlexEmbedding core to vector<double>
            std::vector<double> emb(16, 0.0);
            for (size_t i = 0; i < CORE_DIM; ++i) {
                emb[i] = rel_info.embedding.core[i];
            }
            result[key] = std::move(emb);
        } else {
            result[key] = std::vector<double>(16, 0.0);
        }
    }
    return result;
}

// =============================================================================
// Template Fallback
// =============================================================================

std::string KANLanguageEngine::template_generate(
    const std::vector<ConceptId>& chain,
    size_t /*template_type*/
) const {
    if (chain.empty()) return "I don't have enough information to answer.";

    auto& reg = RelationTypeRegistry::instance();
    std::string output;

    // Primary concept definition
    auto primary = ltm_.retrieve_concept(chain[0]);
    if (primary) {
        output += "**" + primary->label + "**: " + primary->definition;
    }

    // Group relations by (source, verb) → targets with category ordering
    if (chain.size() > 1 && primary) {
        struct RelGroup {
            RelationCategory category;
            std::string source_label;
            std::string verb;
            std::vector<std::string> targets;
        };
        std::vector<RelGroup> groups;
        std::unordered_set<ConceptId> chain_set(chain.begin(), chain.end());

        for (size_t i = 0; i < chain.size(); ++i) {
            auto rels = ltm_.get_outgoing_relations(chain[i]);
            for (const auto& r : rels) {
                if (!chain_set.count(r.target)) continue;

                auto src_info = ltm_.retrieve_concept(r.source);
                auto tgt_info = ltm_.retrieve_concept(r.target);
                if (!src_info || !tgt_info) continue;

                auto& type_info = reg.get(r.type);
                RelationCategory cat = type_info.category;

                // Try to merge into existing group
                bool merged = false;
                for (auto& g : groups) {
                    if (g.source_label == src_info->label && g.verb == type_info.name_en) {
                        bool dup = false;
                        for (const auto& t : g.targets) {
                            if (t == tgt_info->label) { dup = true; break; }
                        }
                        if (!dup) g.targets.push_back(tgt_info->label);
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    groups.push_back({cat, src_info->label, type_info.name_en, {tgt_info->label}});
                }
            }
        }

        // Sort by category: HIERARCHICAL, COMPOSITIONAL, FUNCTIONAL, CAUSAL, rest
        std::stable_sort(groups.begin(), groups.end(),
            [](const RelGroup& a, const RelGroup& b) {
                auto prio = [](RelationCategory c) -> int {
                    switch (c) {
                        case RelationCategory::HIERARCHICAL:  return 0;
                        case RelationCategory::COMPOSITIONAL: return 1;
                        case RelationCategory::FUNCTIONAL:    return 2;
                        case RelationCategory::CAUSAL:        return 3;
                        default: return 4;
                    }
                };
                return prio(a.category) < prio(b.category);
            });

        if (!groups.empty()) {
            output += "\n\n";
            bool first_sentence = true;
            for (const auto& g : groups) {
                // Pronoun substitution
                std::string subject;
                if (first_sentence || g.source_label != primary->label) {
                    subject = g.source_label;
                } else {
                    subject = "It";
                }
                first_sentence = false;

                output += subject + " " + g.verb + " ";
                // Oxford comma list
                for (size_t j = 0; j < g.targets.size(); ++j) {
                    if (g.targets.size() > 2 && j > 0 && j == g.targets.size() - 1) {
                        output += ", and ";
                    } else if (g.targets.size() == 2 && j == 1) {
                        output += " and ";
                    } else if (j > 0) {
                        output += ", ";
                    }
                    output += g.targets[j];
                }
                output += ". ";
            }
        }
    }

    // Related concept definitions (max 2)
    size_t extra = 0;
    for (size_t i = 1; i < chain.size() && extra < 2; ++i) {
        auto info = ltm_.retrieve_concept(chain[i]);
        if (!info || info->definition.empty()) continue;
        if (extra == 0) output += "\n\n";
        output += "**" + info->label + "**: " + info->definition.substr(0, 100);
        if (info->definition.size() > 100) output += "...";
        output += "\n";
        ++extra;
    }

    return output;
}

// =============================================================================
// Persistence
// =============================================================================

void KANLanguageEngine::save(const std::string& dir) const {
    namespace fs = std::filesystem;
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    tokenizer_.save(dir + "/tokenizer.txt");
    // KAN modules and embeddings would need binary serialization
    // (deferred to training implementation)
}

bool KANLanguageEngine::load(const std::string& dir) {
    namespace fs = std::filesystem;
    if (!fs::exists(dir)) return false;

    if (tokenizer_.load(dir + "/tokenizer.txt")) {
        initialized_ = true;
        return true;
    }
    return false;
}

} // namespace brain19

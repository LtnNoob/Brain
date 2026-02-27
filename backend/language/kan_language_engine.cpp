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

    // Initialize ConceptReasoner for composition-guided causal chain extraction
    std::cerr << "[KANLanguageEngine] Initializing ConceptReasoner...\n";
    {
        ReasonerConfig rcfg;
        rcfg.max_steps = 8;                 // language needs focused, shorter chains
        rcfg.enable_composition = true;
        rcfg.chain_coherence_weight = 0.3;
        rcfg.chain_ctx_blend = 0.15;
        rcfg.seed_anchor_weight = 0.35;     // was 0.15 — stronger anchor keeps topic focus
        rcfg.seed_anchor_decay = 0.03;      // was 0.08 — stays > 25% influence at step 8
        rcfg.min_coherence_gate = 0.25;
        rcfg.enable_chain_validation = true;
        rcfg.min_seed_similarity = 0.25;
        rcfg.max_consecutive_drops = 2;
        reasoner_ = std::make_unique<ConceptReasoner>(ltm_, registry_, embeddings_, rcfg);
    }
    std::cerr << "[KANLanguageEngine] ConceptReasoner initialized (composition ON)\n";

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

LanguageResult KANLanguageEngine::generate(const std::string& query, size_t /*max_tokens*/) const {
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

    // ── Step 4: Extract causal chain (via ConceptReasoner composition) ──
    auto causal_chain = extract_causal_chain(seeds);
    result.causal_chain = causal_chain;
    result.reasoning_chain = last_reasoning_chain_;  // attach full chain with coherence/state

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

    // ── Step 6b: Modulate causality scores with coherence from ReasoningChain ──
    // Each chain transition has a coherence score from ChainKAN — blend it into causality.
    if (!last_reasoning_chain_.empty() && last_reasoning_chain_.steps.size() >= 2) {
        const auto& steps = last_reasoning_chain_.steps;
        for (size_t i = 1; i < steps.size(); ++i) {
            std::string key = std::to_string(steps[i - 1].concept_id) + ":"
                            + std::to_string(steps[i].concept_id);
            auto it = scores.causality.find(key);
            if (it != scores.causality.end()) {
                // Blend: 70% KAN causality + 30% ChainKAN coherence
                it->second = 0.7 * it->second + 0.3 * steps[i].coherence_score;
            } else {
                // Chain pair not in causal_pairs — add it with coherence as causality
                scores.causality[key] = steps[i].coherence_score;
            }
        }
    }

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

    // ── Step 7d: Fill convergence state from reasoning chain (32D) ──
    // The CONVERGENCE_DIM slot was reserved in the decoder but never filled — now
    // it carries the accumulated ConvergencePort chain state from ConceptReasoner.
    {
        const size_t CONV_DIM = LanguageConfig::CONVERGENCE_DIM;  // 32
        fused.convergence_state.assign(CONV_DIM, 0.0);
        if (!last_reasoning_chain_.empty()) {
            // Use the last step's chain state = full accumulated reasoning context
            const auto& last_state = last_reasoning_chain_.steps.back().chain_state;
            for (size_t d = 0; d < CONV_DIM; ++d) {
                fused.convergence_state[d] = last_state[d];
            }
        }
    }

    // ── Step 8: Generate fluent text ──
    // (concept decoder path removed — concept_matrix_ is never populated outside LibTorch)
    result.text = generate_fluent_text(query, seeds, fused.ordered_concepts);
    result.used_template = true;
    result.confidence = 0.0;

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

    // Stopwords: common function words that produce noise as seeds
    static const std::unordered_set<std::string> stopwords = {
        "how", "does", "work", "what", "the", "tell", "about",
        "why", "who", "when", "where", "which", "that", "this",
        "are", "was", "were", "been", "being", "have", "has",
        "had", "did", "can", "could", "would", "should", "will",
        "may", "might", "shall", "must", "need", "explain",
        "describe", "define", "many", "much", "some", "any",
        "all", "most", "more", "very", "also", "just", "not",
        "its", "with", "from", "into", "for", "and", "but",
    };

    // Extract keywords (words >= 3 chars, not stopwords)
    std::vector<std::string> keywords;
    std::string word;
    for (char c : lower_q) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            word += c;
        } else {
            if (word.size() >= 3 && !stopwords.count(word)) keywords.push_back(word);
            word.clear();
        }
    }
    if (word.size() >= 3 && !stopwords.count(word)) keywords.push_back(word);

    // Score concepts via indexed lookup (O(K) instead of O(N))
    struct ScoredConcept {
        ConceptId id;
        double score;
    };
    std::unordered_map<ConceptId, double> scores;

    auto add_hits = [&](const std::vector<ConceptId>& hits, double base_score) {
        for (auto cid : hits) {
            auto info = ltm_.retrieve_concept(cid);
            if (!info) continue;
            // Skip linguistic word/sentence concepts
            if (info->label.size() >= 5 &&
                (info->label.substr(0, 5) == "word:" || info->label.substr(0, 5) == "sent:"))
                continue;
            double trust_mult = 0.8 + 0.2 * info->epistemic.trust;
            scores[cid] += base_score * trust_mult;
        }
    };

    // 1. Try full query as a label (highest value)
    add_hits(ltm_.find_by_label(lower_q), 8.0);

    // 2. Try multi-word combinations (adjacent keywords, 3-word and 2-word)
    for (size_t i = 0; i + 2 < keywords.size(); ++i) {
        add_hits(ltm_.find_by_label(keywords[i] + " " + keywords[i+1] + " " + keywords[i+2]), 7.0);
    }
    for (size_t i = 0; i + 1 < keywords.size(); ++i) {
        add_hits(ltm_.find_by_label(keywords[i] + " " + keywords[i+1]), 6.0);
    }

    // 3. Try each keyword as an exact label
    for (const auto& kw : keywords) {
        add_hits(ltm_.find_by_label(kw), 4.0);
    }

    // Sort and take top seeds
    std::vector<ScoredConcept> scored;
    scored.reserve(scores.size());
    for (auto& [cid, score] : scores) {
        scored.push_back({cid, score});
    }

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
// Causal Chain Extraction — via ConceptReasoner (CM Composition + ChainKAN)
// =============================================================================

std::vector<ConceptId> KANLanguageEngine::extract_causal_chain(
    const std::vector<ConceptId>& seeds
) const {
    // Use ConceptReasoner: ConvergencePort composition, ChainKAN coherence,
    // focus-gated traversal, seed anchoring, coherence-gated termination.
    if (reasoner_ && !seeds.empty()) {
        auto chain = reasoner_->reason_from(seeds);
        if (!chain.empty()) {
            // Store the full reasoning chain in a thread-local for generate() to pick up
            // (We cache it so generate() can attach it to LanguageResult)
            last_reasoning_chain_ = std::move(chain);
            return last_reasoning_chain_.concept_sequence();
        }
    }

    // Fallback: simple greedy walk (only if reasoner not available)
    std::vector<ConceptId> chain;
    std::unordered_set<ConceptId> in_chain;
    for (auto sid : seeds) {
        if (in_chain.count(sid)) continue;
        chain.push_back(sid);
        in_chain.insert(sid);

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
// Query Classification
// =============================================================================

bool KANLanguageEngine::is_causal_query(const std::string& query) {
    std::string lower = query;
    for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower.find("how does") != std::string::npos ||
           lower.find("how do") != std::string::npos ||
           lower.find("what happens") != std::string::npos ||
           lower.find("what would happen") != std::string::npos ||
           lower.find("what causes") != std::string::npos ||
           lower.find("why") != std::string::npos ||
           lower.find("cause") != std::string::npos ||
           lower.find("effect") != std::string::npos ||
           lower.find("result") != std::string::npos ||
           lower.find("lead to") != std::string::npos ||
           lower.find("work") != std::string::npos;
}

bool KANLanguageEngine::is_definitional_query(const std::string& query) {
    std::string lower = query;
    for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower.find("what is") != std::string::npos ||
           lower.find("tell me about") != std::string::npos ||
           lower.find("explain") != std::string::npos ||
           lower.find("define") != std::string::npos ||
           lower.find("describe") != std::string::npos;
}

// =============================================================================
// Fluent Text Generation
// =============================================================================

std::string KANLanguageEngine::generate_fluent_text(
    const std::string& query,
    const std::vector<ConceptId>& seeds,
    const std::vector<ConceptId>& ordered_concepts
) const {
    if (ordered_concepts.empty() && seeds.empty())
        return "I don't have enough information to answer.";

    const auto& chain_concepts = ordered_concepts.empty() ? seeds : ordered_concepts;
    auto& reg = RelationTypeRegistry::instance();
    bool causal = is_causal_query(query);
    std::string output;

    // ── Section 1: Primary concept definition ──
    auto primary = ltm_.retrieve_concept(chain_concepts[0]);
    if (primary) {
        output += "**" + primary->label + "**: " + primary->definition;
    }

    // ── Section 2: Reasoning chain as prose via TemplateEngine ──
    if (!last_reasoning_chain_.empty() && last_reasoning_chain_.steps.size() >= 2) {
        auto concept_seq = last_reasoning_chain_.concept_sequence();
        auto relation_seq = last_reasoning_chain_.relation_sequence();

        if (!concept_seq.empty() && !relation_seq.empty()) {
            TemplateEngine te(ltm_);

            // Separate chain sentences into causal vs non-causal for ordering
            struct ChainSentence {
                std::string text;
                bool is_causal_rel;
                float trust;
                EpistemicType etype;
            };
            std::vector<ChainSentence> sentences;
            size_t num_edges = std::min(relation_seq.size(), concept_seq.size() - 1);

            for (size_t i = 0; i < num_edges; ++i) {
                auto cat = reg.get_category(relation_seq[i]);
                bool is_causal_rel = (cat == RelationCategory::CAUSAL ||
                                      cat == RelationCategory::FUNCTIONAL);

                auto src_info = ltm_.retrieve_concept(concept_seq[i]);
                auto tgt_info = ltm_.retrieve_concept(concept_seq[i + 1]);
                if (!src_info || !tgt_info) continue;

                // For incoming (reverse) edges, swap subject/object so template reads correctly
                // e.g., chain walks Mathematician <-IS_A<- Bolyai → "Bolyai is a Mathematician"
                bool outgoing = (i + 1 < last_reasoning_chain_.steps.size())
                                ? last_reasoning_chain_.steps[i + 1].is_outgoing : true;
                std::string sent;
                if (outgoing) {
                    sent = te.relation_sentence_en(src_info->label, tgt_info->label, relation_seq[i]);
                } else {
                    sent = te.relation_sentence_en(tgt_info->label, src_info->label, relation_seq[i]);
                }

                // Epistemic framing for low-trust concepts
                float trust = static_cast<float>(tgt_info->epistemic.trust);
                EpistemicType etype = tgt_info->epistemic.type;
                if (trust < 0.85f) {
                    sent = TemplateEngine::epistemic_frame(trust, etype, sent);
                }

                sentences.push_back({sent, is_causal_rel, trust, etype});
            }

            // Sort: for causal queries, causal sentences first; for definitional, non-causal first
            if (causal) {
                std::stable_sort(sentences.begin(), sentences.end(),
                    [](const ChainSentence& a, const ChainSentence& b) {
                        return a.is_causal_rel > b.is_causal_rel;  // true (causal) first
                    });
            } else {
                std::stable_sort(sentences.begin(), sentences.end(),
                    [](const ChainSentence& a, const ChainSentence& b) {
                        return a.is_causal_rel < b.is_causal_rel;  // false (non-causal) first
                    });
            }

            if (!sentences.empty()) {
                output += "\n\n";
                for (size_t i = 0; i < sentences.size(); ++i) {
                    if (i > 0) output += " ";
                    output += sentences[i].text;
                }
            }
        }
    }

    // ── Section 3: Grouped supplementary relations (1-hop from primary, not in chain) ──
    if (primary) {
        // Build set of chain concept pairs to exclude already-covered relations
        std::unordered_set<std::string> chain_pairs;
        if (!last_reasoning_chain_.empty()) {
            auto cseq = last_reasoning_chain_.concept_sequence();
            for (size_t i = 0; i + 1 < cseq.size(); ++i) {
                chain_pairs.insert(std::to_string(cseq[i]) + ":" + std::to_string(cseq[i + 1]));
            }
        }

        struct RelGroup {
            RelationCategory category;
            std::string source_label;
            std::string verb;
            std::vector<std::string> targets;
        };
        std::vector<RelGroup> groups;

        auto rels = ltm_.get_outgoing_relations(primary->id);
        for (const auto& r : rels) {
            if (r.weight <= 0.5) continue;  // Filter low-weight

            // Skip if already covered by reasoning chain
            std::string pair_key = std::to_string(r.source) + ":" + std::to_string(r.target);
            if (chain_pairs.count(pair_key)) continue;

            auto tgt_info = ltm_.retrieve_concept(r.target);
            if (!tgt_info) continue;

            // Skip linguistic concepts
            if (tgt_info->label.size() >= 5 &&
                (tgt_info->label.substr(0, 5) == "word:" || tgt_info->label.substr(0, 5) == "sent:"))
                continue;

            auto& type_info = reg.get(r.type);
            RelationCategory cat = type_info.category;

            // Skip generic noise relations
            if (cat == RelationCategory::SIMILARITY &&
                (r.type == RelationType::ASSOCIATED_WITH || r.type == RelationType::SIMILAR_TO))
                continue;
            // Skip linguistic category
            if (cat == RelationCategory::LINGUISTIC) continue;

            // Merge into existing group
            bool merged = false;
            for (auto& g : groups) {
                if (g.source_label == primary->label && g.verb == type_info.name_en) {
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
                groups.push_back({cat, primary->label, type_info.name_en, {tgt_info->label}});
            }
        }

        // Sort by category priority based on query type
        if (!groups.empty()) {
            auto cat_priority = [causal](RelationCategory c) -> int {
                if (causal) {
                    switch (c) {
                        case RelationCategory::CAUSAL:        return 0;
                        case RelationCategory::FUNCTIONAL:    return 1;
                        case RelationCategory::COMPOSITIONAL: return 2;
                        case RelationCategory::HIERARCHICAL:  return 3;
                        default: return 4;
                    }
                } else {
                    switch (c) {
                        case RelationCategory::HIERARCHICAL:  return 0;
                        case RelationCategory::COMPOSITIONAL: return 1;
                        case RelationCategory::FUNCTIONAL:    return 2;
                        case RelationCategory::CAUSAL:        return 3;
                        default: return 4;
                    }
                }
            };
            std::stable_sort(groups.begin(), groups.end(),
                [&](const RelGroup& a, const RelGroup& b) {
                    return cat_priority(a.category) < cat_priority(b.category);
                });

            // Cap at 4 groups to avoid info-dump
            if (groups.size() > 4) groups.resize(4);

            output += "\n\n";
            bool first_sentence = true;
            for (const auto& g : groups) {
                // Pronoun substitution: use concept name first, "It" thereafter
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

    // ── Section 4: Related concept definitions (max 2 from chain) ──
    if (last_reasoning_chain_.steps.size() >= 2) {
        auto cseq = last_reasoning_chain_.concept_sequence();
        size_t extra = 0;
        for (size_t i = 1; i < cseq.size() && extra < 2; ++i) {
            if (cseq[i] == chain_concepts[0]) continue;  // skip primary
            auto info = ltm_.retrieve_concept(cseq[i]);
            if (!info || info->definition.empty()) continue;
            if (extra == 0) output += "\n\n";
            output += "**" + info->label + "**: " + info->definition.substr(0, 120);
            if (info->definition.size() > 120) output += "...";
            output += "\n";
            ++extra;
        }
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

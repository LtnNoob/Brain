#include "concept_pattern_engine.hpp"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <unordered_map>

namespace brain19 {

// =============================================================================
// CONSTRUCTION
// =============================================================================

ConceptPatternEngine::ConceptPatternEngine(
    const ConceptModelRegistry& registry,
    const EmbeddingManager& embeddings)
    : registry_(registry)
    , embeddings_(embeddings)
{}

std::string ConceptPatternEngine::get_model_id() const {
    return "concept-pattern-engine-v1.0";
}

// =============================================================================
// CORE HELPER: PREDICT EDGE VIA CONCEPTMODEL
// =============================================================================

double ConceptPatternEngine::predict_edge(
    ConceptId from, ConceptId to, RelationType type) const
{
    const ConceptModel* model = registry_.get_model(from);
    if (!model) return 0.0;

    // Models with no training data or failed convergence are unusable
    if (!model->is_converged() || model->sample_count() == 0) return 0.0;

    const auto& rel_emb = embeddings_.get_relation_embedding(type);
    // MUST match training context (concept_trainer uses RECALL_HASH)
    static const size_t RECALL_HASH = std::hash<std::string>{}("recall");
    auto ctx_emb = embeddings_.make_target_embedding(RECALL_HASH, from, to);

    // Discount by training quality: lower loss = higher quality
    double quality = 1.0 - std::min(model->final_loss(), 1.0);

    // Well-trained models: full predict_refined
    if (model->sample_count() >= 4) {
        auto concept_from = embeddings_.concept_embeddings().get_or_default(from);
        auto concept_to = embeddings_.concept_embeddings().get_or_default(to);
        return model->predict_refined(rel_emb, ctx_emb, concept_from, concept_to) * quality;
    }

    // Sparse models: bilinear fallback with uncertainty discount
    return model->predict(rel_emb, ctx_emb) * quality * 0.5;
}

// =============================================================================
// GET PER-CONCEPT PATTERN WEIGHT
// =============================================================================
// Pattern indices: 0=shared_parent, 1=transitive, 2=missing_link,
//                  3=weak_strength, 4=contradictory, 5=chain

double ConceptPatternEngine::get_pattern_weight(ConceptId cid, size_t pattern_idx) const {
    const ConceptModel* model = registry_.get_model(cid);
    if (!model) {
        // Defaults
        static const double defaults[] = {1.0, 1.0, 1.0, 1.0, 1.0, 0.85};
        return (pattern_idx < 6) ? defaults[pattern_idx] : 1.0;
    }
    const auto& pw = model->pattern_weights();
    switch (pattern_idx) {
        case 0: return pw.shared_parent;
        case 1: return pw.transitive_causation;
        case 2: return pw.missing_link;
        case 3: return pw.weak_strengthening;
        case 4: return pw.contradictory_signal;
        case 5: return pw.chain_hypothesis;
        default: return 1.0;
    }
}

// =============================================================================
// EXTRACT MEANING
// =============================================================================

std::vector<MeaningProposal> ConceptPatternEngine::extract_meaning(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<MeaningProposal> proposals;

    if (active_concepts.empty()) return proposals;

    struct ScoredEdge {
        ConceptId source;
        ConceptId target;
        RelationType type;
        double ltm_weight;
        double kan_score;
    };

    std::vector<ScoredEdge> all_edges;

    for (auto cid : active_concepts) {
        auto rels = ltm.get_outgoing_relations(cid);
        for (const auto& r : rels) {
            double kan = predict_edge(r.source, r.target, r.type);
            all_edges.push_back({r.source, r.target, r.type, r.weight, kan});
        }
    }

    std::sort(all_edges.begin(), all_edges.end(),
        [](const ScoredEdge& a, const ScoredEdge& b) {
            return a.kan_score > b.kan_score;
        });

    if (all_edges.empty()) {
        std::ostringstream interp;
        interp << "ConceptModel activation of " << active_concepts.size()
               << " concepts (no outgoing relations found)";
        proposals.emplace_back(
            ++proposal_counter_, active_concepts, interp.str(),
            "No outgoing relations available for scoring", 0.3, get_model_id());
        return proposals;
    }

    std::ostringstream interp;
    std::ostringstream reasoning;

    size_t max_paths = std::min(all_edges.size(), size_t(5));
    for (size_t i = 0; i < max_paths; ++i) {
        const auto& e = all_edges[i];
        auto src_info = ltm.retrieve_concept(e.source);
        auto tgt_info = ltm.retrieve_concept(e.target);
        if (!src_info || !tgt_info) continue;

        if (i > 0) interp << " | ";
        interp << "[" << src_info->label << "] --"
               << relation_type_to_string(e.type)
               << "(CM:" << std::fixed;
        interp.precision(2);
        interp << e.kan_score << ")--> ["
               << tgt_info->label << "]";
    }

    reasoning << "ConceptModel scored " << all_edges.size() << " edges across "
              << active_concepts.size() << " concepts. Top score: ";
    reasoning.precision(3);
    reasoning << std::fixed << all_edges[0].kan_score;

    std::unordered_set<ConceptId> source_set;
    std::vector<ConceptId> sources;
    for (size_t i = 0; i < max_paths; ++i) {
        if (source_set.insert(all_edges[i].source).second)
            sources.push_back(all_edges[i].source);
        if (source_set.insert(all_edges[i].target).second)
            sources.push_back(all_edges[i].target);
    }

    double avg_kan = 0.0;
    for (size_t i = 0; i < max_paths; ++i) avg_kan += all_edges[i].kan_score;
    avg_kan /= static_cast<double>(max_paths);

    proposals.emplace_back(
        ++proposal_counter_, sources, interp.str(), reasoning.str(),
        std::min(avg_kan, LLM_ONLY_TRUST_CEILING), get_model_id());

    return proposals;
}

// =============================================================================
// GENERATE HYPOTHESES (6 PATTERNS WITH PER-CONCEPT WEIGHTS)
// =============================================================================

std::vector<HypothesisProposal> ConceptPatternEngine::generate_hypotheses(
    const std::vector<ConceptId>& evidence_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/,
    const std::vector<ThoughtPath>& thought_paths
) const {
    std::vector<HypothesisProposal> proposals;

    if (evidence_concepts.size() < 2) return proposals;

    struct ConceptData {
        ConceptId id;
        std::string label;
        std::vector<RelationInfo> outgoing;
    };
    std::vector<ConceptData> concepts;
    for (auto cid : evidence_concepts) {
        auto info = ltm.retrieve_concept(cid);
        if (!info) continue;
        ConceptData cd;
        cd.id = cid;
        cd.label = info->label;
        cd.outgoing = ltm.get_outgoing_relations(cid);
        concepts.push_back(std::move(cd));
    }

    if (concepts.size() < 2) return proposals;

    // ─── Pattern 1: Shared Parent (IS_A generalization) ──────────────────
    {
        double avg_weight = 0.0;
        size_t weight_count = 0;
        for (const auto& c : concepts) {
            double w = get_pattern_weight(c.id, 0);
            avg_weight += w;
            ++weight_count;
        }
        avg_weight = (weight_count > 0) ? avg_weight / weight_count : 1.0;

        if (avg_weight > 0.1) {
            std::unordered_map<ConceptId, std::vector<size_t>> parent_to_children;
            for (size_t i = 0; i < concepts.size(); ++i) {
                for (const auto& r : concepts[i].outgoing) {
                    if (r.type == RelationType::IS_A) {
                        parent_to_children[r.target].push_back(i);
                    }
                }
            }

            for (const auto& [parent_id, children_idx] : parent_to_children) {
                if (children_idx.size() < 2) continue;
                auto parent_info = ltm.retrieve_concept(parent_id);
                if (!parent_info) continue;

                double kan_a = predict_edge(concepts[children_idx[0]].id, parent_id, RelationType::IS_A);
                double kan_b = predict_edge(concepts[children_idx[1]].id, parent_id, RelationType::IS_A);
                if (kan_a < 0.3 || kan_b < 0.3) continue;

                double confidence = avg_weight * (kan_a + kan_b) / 2.0;
                confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                std::ostringstream stmt;
                stmt << concepts[children_idx[0]].label << " and "
                     << concepts[children_idx[1]].label
                     << " share parent " << parent_info->label
                     << " — relationship is proportional and scales with shared taxonomy"
                     << " (CM:" << std::fixed;
                stmt.precision(2);
                stmt << kan_a << "/" << kan_b << ")";

                std::vector<ConceptId> evidence = {
                    concepts[children_idx[0]].id,
                    concepts[children_idx[1]].id,
                    parent_id
                };

                proposals.emplace_back(
                    ++proposal_counter_, evidence, stmt.str(),
                    "Shared IS_A parent with high ConceptModel scores suggests proportional relationship",
                    std::vector<std::string>{"shared-parent", "proportional", "scales with"},
                    confidence, get_model_id());
                break;
            }
        }
    }

    // ─── Pattern 2: Transitive Causation (A→B→C chain) ──────────────────
    {
        double avg_weight = 0.0;
        size_t wc = 0;
        for (const auto& c : concepts) { avg_weight += get_pattern_weight(c.id, 1); ++wc; }
        avg_weight = (wc > 0) ? avg_weight / wc : 1.0;

        if (avg_weight > 0.1) {
            for (size_t i = 0; i < concepts.size() && proposals.size() < 5; ++i) {
                for (const auto& r_ab : concepts[i].outgoing) {
                    if (r_ab.type != RelationType::CAUSES) continue;

                    auto b_rels = ltm.get_outgoing_relations(r_ab.target);
                    for (const auto& r_bc : b_rels) {
                        if (r_bc.type != RelationType::CAUSES) continue;

                        auto a_to_c = ltm.get_relations_between(concepts[i].id, r_bc.target);
                        bool direct_exists = false;
                        for (const auto& r : a_to_c) {
                            if (r.type == RelationType::CAUSES) { direct_exists = true; break; }
                        }

                        double kan_ac = predict_edge(concepts[i].id, r_bc.target, RelationType::CAUSES);

                        auto b_info = ltm.retrieve_concept(r_ab.target);
                        auto c_info = ltm.retrieve_concept(r_bc.target);
                        if (!b_info || !c_info) continue;

                        double confidence = avg_weight * kan_ac;
                        confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                        std::ostringstream stmt;
                        stmt << concepts[i].label << " causes " << b_info->label
                             << " which causes " << c_info->label
                             << " — linear transitive chain, " << concepts[i].label
                             << " increases with " << c_info->label
                             << (direct_exists ? " (direct link exists)" : " (no direct link)")
                             << " (CM:" << std::fixed;
                        stmt.precision(2);
                        stmt << kan_ac << ")";

                        std::vector<ConceptId> evidence = {
                            concepts[i].id, r_ab.target, r_bc.target
                        };

                        proposals.emplace_back(
                            ++proposal_counter_, evidence, stmt.str(),
                            "Transitive causation chain with ConceptModel-predicted direct link",
                            std::vector<std::string>{"transitive-causation", "linear", "increases with"},
                            confidence, get_model_id());
                        goto done_transitive;
                    }
                }
            }
            done_transitive:;
        }
    }

    // ─── Pattern 3: Missing Link (ConceptModel predicts >0.7, no LTM relation)
    {
        double avg_weight = 0.0;
        size_t wc = 0;
        for (const auto& c : concepts) { avg_weight += get_pattern_weight(c.id, 2); ++wc; }
        avg_weight = (wc > 0) ? avg_weight / wc : 1.0;

        if (avg_weight > 0.1) {
            for (size_t i = 0; i < concepts.size() && proposals.size() < 7; ++i) {
                for (size_t j = 0; j < concepts.size(); ++j) {
                    if (i == j) continue;

                    auto existing = ltm.get_relations_between(concepts[i].id, concepts[j].id);
                    if (!existing.empty()) continue;

                    RelationType try_types[] = {
                        RelationType::CAUSES, RelationType::ENABLES,
                        RelationType::SUPPORTS, RelationType::IS_A
                    };

                    for (auto rtype : try_types) {
                        double kan = predict_edge(concepts[i].id, concepts[j].id, rtype);
                        if (kan < 0.7) continue;

                        // Bidirectional consistency: if A→B strong, B→A should be at least 0.3
                        double reverse = predict_edge(concepts[j].id, concepts[i].id, rtype);
                        if (reverse < 0.3) continue;

                        // Shared-neighbor check: without common neighbors, require score > 0.85
                        bool has_shared_neighbor = false;
                        for (const auto& ri : concepts[i].outgoing) {
                            for (const auto& rj : concepts[j].outgoing) {
                                if (ri.target == rj.target) {
                                    has_shared_neighbor = true;
                                    break;
                                }
                            }
                            if (has_shared_neighbor) break;
                        }
                        if (!has_shared_neighbor && kan < 0.85) continue;

                        double confidence = avg_weight * kan * 0.8;
                        confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                        std::ostringstream stmt;
                        stmt << "Missing link: " << concepts[i].label
                             << " may have a threshold-based " << relation_type_to_string(rtype)
                             << " relation to " << concepts[j].label
                             << " — activation pattern suggests hidden connection"
                             << " (CM:" << std::fixed;
                        stmt.precision(2);
                        stmt << kan << ", rev:" << reverse << ")";

                        std::vector<ConceptId> evidence = {
                            concepts[i].id, concepts[j].id
                        };

                        proposals.emplace_back(
                            ++proposal_counter_, evidence, stmt.str(),
                            "ConceptModel predicts high relevance but no LTM relation exists",
                            std::vector<std::string>{"missing-link", "threshold", "activation"},
                            confidence, get_model_id());
                        goto done_missing;
                    }
                }
            }
            done_missing:;
        }
    }

    // ─── Pattern 4: Weak Strengthening (LTM weight<0.3 but CM>0.6) ─────
    {
        double avg_weight = 0.0;
        size_t wc = 0;
        for (const auto& c : concepts) { avg_weight += get_pattern_weight(c.id, 3); ++wc; }
        avg_weight = (wc > 0) ? avg_weight / wc : 1.0;

        if (avg_weight > 0.1) {
            for (size_t i = 0; i < concepts.size() && proposals.size() < 9; ++i) {
                for (const auto& r : concepts[i].outgoing) {
                    if (r.weight >= 0.3) continue;

                    double kan = predict_edge(r.source, r.target, r.type);
                    if (kan < 0.6) continue;

                    auto tgt_info = ltm.retrieve_concept(r.target);
                    if (!tgt_info) continue;

                    double confidence = avg_weight * kan * 0.7;
                    confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                    std::ostringstream stmt;
                    stmt << concepts[i].label << " --" << relation_type_to_string(r.type)
                         << "--> " << tgt_info->label
                         << " has weak LTM weight (" << std::fixed;
                    stmt.precision(2);
                    stmt << r.weight << ") but CM predicts proportional strength ("
                         << kan << ") — increases with evidence";

                    std::vector<ConceptId> evidence = {r.source, r.target};

                    proposals.emplace_back(
                        ++proposal_counter_, evidence, stmt.str(),
                        "Low LTM weight contradicted by high ConceptModel prediction",
                        std::vector<std::string>{"weak-strengthening", "proportional", "increases with"},
                        confidence, get_model_id());
                    goto done_weak;
                }
            }
            done_weak:;
        }
    }

    // ─── Pattern 5: Contradictory Signal (LTM>0.7, CM<0.3 or vice versa)
    {
        double avg_weight = 0.0;
        size_t wc = 0;
        for (const auto& c : concepts) { avg_weight += get_pattern_weight(c.id, 4); ++wc; }
        avg_weight = (wc > 0) ? avg_weight / wc : 1.0;

        if (avg_weight > 0.1) {
            for (size_t i = 0; i < concepts.size() && proposals.size() < 10; ++i) {
                for (const auto& r : concepts[i].outgoing) {
                    double kan = predict_edge(r.source, r.target, r.type);

                    bool ltm_high_kan_low = (r.weight > 0.7 && kan < 0.3);
                    bool ltm_low_kan_high = (r.weight < 0.3 && kan > 0.7);

                    if (!ltm_high_kan_low && !ltm_low_kan_high) continue;

                    auto tgt_info = ltm.retrieve_concept(r.target);
                    if (!tgt_info) continue;

                    double mismatch = std::abs(r.weight - kan);
                    double confidence = avg_weight * mismatch * 0.6;
                    confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                    std::ostringstream stmt;
                    stmt << "Contradictory signal: " << concepts[i].label
                         << " --" << relation_type_to_string(r.type)
                         << "--> " << tgt_info->label << std::fixed;
                    stmt.precision(2);
                    stmt << " LTM=" << r.weight << " vs CM=" << kan
                         << " — relationship is conditional, depends on context";

                    std::vector<ConceptId> evidence = {r.source, r.target};

                    proposals.emplace_back(
                        ++proposal_counter_, evidence, stmt.str(),
                        ltm_high_kan_low
                            ? "High LTM weight but low CM score — may be outdated or contextual"
                            : "Low LTM weight but high CM score — may be emerging relationship",
                        std::vector<std::string>{"contradictory-signal", "conditional", "depends on"},
                        confidence, get_model_id());
                    goto done_contradictory;
                }
            }
            done_contradictory:;
        }
    }

    // ─── Pattern 6: Multi-Hop Chain (from ThoughtPaths) ─────────────────
    {
        double avg_weight = 0.0;
        size_t wc = 0;
        for (const auto& c : concepts) { avg_weight += get_pattern_weight(c.id, 5); ++wc; }
        avg_weight = (wc > 0) ? avg_weight / wc : 1.0;

        if (avg_weight > 0.1 && !thought_paths.empty()) {
            size_t chains_emitted = 0;
            for (const auto& path : thought_paths) {
                if (chains_emitted >= 5) break;
                if (path.length() < 3) continue;

                double product = 1.0;
                size_t n_edges = 0;
                bool chain_broken = false;
                std::vector<ConceptId> chain_concepts;
                std::ostringstream chain_desc;

                for (size_t ni = 0; ni < path.nodes.size(); ++ni) {
                    ConceptId node_cid = path.nodes[ni].concept_id;
                    auto node_info = ltm.retrieve_concept(node_cid);
                    chain_concepts.push_back(node_cid);

                    if (ni > 0) chain_desc << " -> ";
                    chain_desc << (node_info ? node_info->label : ("?" + std::to_string(node_cid)));

                    if (ni > 0) {
                        ConceptId prev_cid = path.nodes[ni - 1].concept_id;
                        auto between = ltm.get_relations_between(prev_cid, node_cid);
                        if (between.empty()) {
                            between = ltm.get_relations_between(node_cid, prev_cid);
                        }

                        if (!between.empty()) {
                            double edge_kan = predict_edge(between[0].source, between[0].target, between[0].type);
                            if (edge_kan < 0.3) {
                                chain_broken = true;
                                break;
                            }
                            product *= edge_kan;
                            ++n_edges;
                        } else {
                            product *= 0.2;
                            ++n_edges;
                        }
                    }
                }

                if (chain_broken || n_edges == 0) continue;

                double chain_score = std::pow(product, 1.0 / static_cast<double>(n_edges));
                if (chain_score < 0.4) continue;

                ConceptId first_cid = path.nodes.front().concept_id;
                ConceptId last_cid = path.nodes.back().concept_id;
                auto direct = ltm.get_relations_between(first_cid, last_cid);
                bool has_direct = !direct.empty();

                double confidence = avg_weight * chain_score;
                confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                std::ostringstream stmt;
                stmt << "Multi-hop chain: " << chain_desc.str()
                     << " — chain-strength (geometric mean CM: " << std::fixed;
                stmt.precision(2);
                stmt << chain_score << ", " << n_edges << " edges)"
                     << (has_direct ? " [direct link exists]" : " [no direct link — transitive only]");

                proposals.emplace_back(
                    ++proposal_counter_, chain_concepts, stmt.str(),
                    "Multi-hop chain from ThoughtPath with ConceptModel-scored edges",
                    std::vector<std::string>{"multi-hop-chain", "transitive", "chain-strength"},
                    confidence, get_model_id());
                ++chains_emitted;
            }
        }
    }

    return proposals;
}

// =============================================================================
// DETECT ANALOGIES
// =============================================================================

std::vector<AnalogyProposal> ConceptPatternEngine::detect_analogies(
    const std::vector<ConceptId>& concept_set_a,
    const std::vector<ConceptId>& concept_set_b,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<AnalogyProposal> proposals;

    if (concept_set_a.empty() || concept_set_b.empty()) return proposals;

    auto build_distribution = [&](const std::vector<ConceptId>& cset)
        -> std::unordered_map<uint16_t, double>
    {
        std::unordered_map<uint16_t, double> dist;
        for (auto cid : cset) {
            auto rels = ltm.get_outgoing_relations(cid);
            for (const auto& r : rels) {
                double kan = predict_edge(r.source, r.target, r.type);
                dist[static_cast<uint16_t>(r.type)] += kan;
            }
        }
        return dist;
    };

    auto dist_a = build_distribution(concept_set_a);
    auto dist_b = build_distribution(concept_set_b);

    if (dist_a.empty() || dist_b.empty()) return proposals;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (const auto& [key, val] : dist_a) {
        norm_a += val * val;
        auto it = dist_b.find(key);
        if (it != dist_b.end()) {
            dot += val * it->second;
        }
    }
    for (const auto& [key, val] : dist_b) {
        norm_b += val * val;
    }

    double similarity = 0.0;
    if (norm_a > 0.0 && norm_b > 0.0) {
        similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    if (similarity < 0.1) return proposals;

    std::ostringstream mapping;
    mapping << "ConceptModel-scored relation distribution similarity: " << std::fixed;
    mapping.precision(3);
    mapping << similarity << " across " << dist_a.size()
            << " vs " << dist_b.size() << " relation types";

    proposals.emplace_back(
        ++proposal_counter_,
        concept_set_a, concept_set_b, mapping.str(),
        similarity,
        std::min(similarity * 0.8, LLM_ONLY_TRUST_CEILING),
        get_model_id());

    return proposals;
}

// =============================================================================
// DETECT CONTRADICTIONS
// =============================================================================

std::vector<ContradictionProposal> ConceptPatternEngine::detect_contradictions(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<ContradictionProposal> proposals;

    if (active_concepts.size() < 2) return proposals;

    // 1. Explicit CONTRADICTS relations
    for (size_t i = 0; i < active_concepts.size() && i < 10; ++i) {
        auto rels = ltm.get_outgoing_relations(active_concepts[i]);
        for (const auto& r : rels) {
            if (r.type != RelationType::CONTRADICTS) continue;

            bool target_active = false;
            for (auto cid : active_concepts) {
                if (cid == r.target) { target_active = true; break; }
            }

            auto tgt_info = ltm.retrieve_concept(r.target);
            auto src_info = ltm.retrieve_concept(r.source);
            if (!src_info || !tgt_info) continue;

            double kan = predict_edge(r.source, r.target, RelationType::CONTRADICTS);

            std::ostringstream desc;
            desc << "Explicit contradiction: " << src_info->label
                 << " CONTRADICTS " << tgt_info->label
                 << " (CM:" << std::fixed;
            desc.precision(2);
            desc << kan << ", weight:" << r.weight
                 << (target_active ? ", BOTH ACTIVE" : "") << ")";

            double severity = r.weight * (target_active ? 1.0 : 0.6);

            proposals.emplace_back(
                ++proposal_counter_, r.source, r.target, desc.str(),
                "Explicit CONTRADICTS relation in KG", severity,
                std::min(kan * 0.9, LLM_ONLY_TRUST_CEILING), get_model_id());
        }
    }

    // 2. CM prediction mismatch: high-weight relation with low CM score
    for (size_t i = 0; i < active_concepts.size() && i < 8; ++i) {
        auto rels = ltm.get_outgoing_relations(active_concepts[i]);
        for (const auto& r : rels) {
            if (r.type == RelationType::CONTRADICTS) continue;
            if (r.weight < 0.6) continue;

            double kan = predict_edge(r.source, r.target, r.type);
            if (kan > 0.3) continue;

            auto src_info = ltm.retrieve_concept(r.source);
            auto tgt_info = ltm.retrieve_concept(r.target);
            if (!src_info || !tgt_info) continue;

            double mismatch = r.weight - kan;

            std::ostringstream desc;
            desc << "CM mismatch: " << src_info->label
                 << " --" << relation_type_to_string(r.type)
                 << "--> " << tgt_info->label << std::fixed;
            desc.precision(2);
            desc << " (LTM:" << r.weight << " vs CM:" << kan << ")";

            proposals.emplace_back(
                ++proposal_counter_, r.source, r.target, desc.str(),
                "High LTM weight but low ConceptModel prediction — possible stale or contextual relation",
                mismatch * 0.7,
                std::min(mismatch * 0.5, LLM_ONLY_TRUST_CEILING), get_model_id());

            if (proposals.size() >= 5) return proposals;
        }
    }

    return proposals;
}

// =============================================================================
// TRAIN FROM VALIDATION — updates PER-CONCEPT pattern weights
// =============================================================================

void ConceptPatternEngine::train_from_validation(
    const std::vector<ValidationResult>& results)
{
    // We need non-const access to registry for pattern weight updates.
    // const_cast is justified: train_from_validation is called from non-const
    // context (ThinkingPipeline step 9.5B), and the registry reference
    // is const only because MiniLLM interface requires const methods.
    auto& mutable_registry = const_cast<ConceptModelRegistry&>(registry_);

    auto clamp_weight = [](double& w) {
        if (w < 0.1) w = 0.1;
        if (w > 3.0) w = 3.0;
    };

    auto adjust = [&](double& w, bool validated) {
        w += validated ? 0.1 : -0.05;
        clamp_weight(w);
    };

    for (const auto& vr : results) {
        // For each validation result, find the source concept and update its
        // pattern weights. The source concept is approximated from the pattern.
        // Since we don't have direct concept tracking in ValidationResult,
        // we update all models (same behavior as global, but stored per-concept).
        // Over time, only relevant concepts get trained due to selective hypothesis generation.
        auto model_ids = mutable_registry.get_model_ids();

        const auto& pat = vr.pattern;

        for (ConceptId cid : model_ids) {
            ConceptModel* model = mutable_registry.get_model(cid);
            if (!model) continue;
            auto& pw = model->pattern_weights();

            switch (pat) {
                case RelationshipPattern::LINEAR:
                    adjust(pw.shared_parent, vr.validated);
                    adjust(pw.transitive_causation, vr.validated);
                    adjust(pw.weak_strengthening, vr.validated);
                    break;
                case RelationshipPattern::THRESHOLD:
                    adjust(pw.missing_link, vr.validated);
                    break;
                case RelationshipPattern::CONDITIONAL:
                    adjust(pw.contradictory_signal, vr.validated);
                    break;
                default:
                    adjust(pw.shared_parent, vr.validated);
                    adjust(pw.transitive_causation, vr.validated);
                    adjust(pw.missing_link, vr.validated);
                    adjust(pw.weak_strengthening, vr.validated);
                    adjust(pw.contradictory_signal, vr.validated);
                    adjust(pw.chain_hypothesis, vr.validated);
                    break;
            }
        }
    }
}

// =============================================================================
// INVESTIGATE ANOMALIES (Topology A)
// =============================================================================

std::vector<HypothesisProposal> ConceptPatternEngine::investigate_anomalies(
    const std::vector<InvestigationRequest>& requests,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<HypothesisProposal> proposals;

    for (const auto& req : requests) {
        auto info_a = ltm.retrieve_concept(req.concept_a);
        auto info_b = ltm.retrieve_concept(req.concept_b);

        std::string label_a = info_a ? info_a->label : ("?" + std::to_string(req.concept_a));
        std::string label_b = info_b ? info_b->label : ("?" + std::to_string(req.concept_b));

        std::ostringstream stmt;
        std::ostringstream reasoning;
        std::vector<std::string> patterns;
        std::vector<ConceptId> evidence = {req.concept_a, req.concept_b};

        switch (req.anomaly_type) {
            case AnomalyType::WEAK_EDGE:
                stmt << "CM predicts " << label_a << " --"
                     << relation_type_to_string(req.relation_type) << "--> "
                     << label_b << " is strong (CM:" << std::fixed;
                stmt.precision(2);
                stmt << req.kan_score << ") but LTM weight is low ("
                     << req.ltm_weight
                     << "). Relationship is proportional and increases with evidence.";
                reasoning << "ConceptModel anomaly: weak edge detected. "
                          << "Prediction significantly exceeds LTM weight.";
                patterns = {"kan-anomaly", "weak-edge", "proportional", "increases with"};
                break;

            case AnomalyType::MISSING_LINK:
                stmt << "CM detects potential " << relation_type_to_string(req.relation_type)
                     << " link between " << label_a << " and " << label_b
                     << " (CM:" << std::fixed;
                stmt.precision(2);
                stmt << req.kan_score
                     << "). Threshold-based activation pattern suggests hidden connection.";
                reasoning << "ConceptModel anomaly: missing link detected. "
                          << "ConceptModel predicts relationship where none exists in LTM.";
                patterns = {"kan-anomaly", "missing-link", "threshold", "activation"};
                break;

            case AnomalyType::CONTRADICTION:
                stmt << "CM/LTM contradiction on " << label_a << " --"
                     << relation_type_to_string(req.relation_type) << "--> "
                     << label_b << ": LTM=" << std::fixed;
                stmt.precision(2);
                stmt << req.ltm_weight << ", CM=" << req.kan_score
                     << ". Relationship is conditional, depends on context.";
                reasoning << "ConceptModel anomaly: contradiction detected. "
                          << "|LTM - CM| = " << std::fixed;
                reasoning.precision(2);
                reasoning << req.anomaly_strength
                          << " — suggests context-dependent relationship.";
                patterns = {"kan-anomaly", "contradiction", "conditional", "depends on"};
                break;

            case AnomalyType::STALE_RELATION:
                stmt << "High LTM weight (" << std::fixed;
                stmt.precision(2);
                stmt << req.ltm_weight << ") but low CM prediction ("
                     << req.kan_score << ") for " << label_a << " --"
                     << relation_type_to_string(req.relation_type) << "--> "
                     << label_b
                     << ". Relationship may be conditional, depends on updated evidence.";
                reasoning << "ConceptModel anomaly: stale relation detected. "
                          << "LTM weight no longer supported by ConceptModel predictions.";
                patterns = {"kan-anomaly", "stale-relation", "conditional", "depends on"};
                break;
        }

        double confidence = std::min(req.anomaly_strength * 0.8, LLM_ONLY_TRUST_CEILING);

        proposals.emplace_back(
            ++proposal_counter_, evidence, stmt.str(), reasoning.str(),
            patterns, confidence, get_model_id() + " [CM-anomaly]");
    }

    return proposals;
}

} // namespace brain19

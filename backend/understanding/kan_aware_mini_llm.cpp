#include "kan_aware_mini_llm.hpp"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <unordered_map>

namespace brain19 {

// =============================================================================
// CONSTRUCTION
// =============================================================================

KanAwareMiniLLM::KanAwareMiniLLM(
    const MicroModelRegistry& registry,
    const EmbeddingManager& embeddings)
    : registry_(registry)
    , embeddings_(embeddings)
{}

std::string KanAwareMiniLLM::get_model_id() const {
    return "kan-aware-mini-llm-v1.0";
}

// =============================================================================
// CORE HELPER: PREDICT EDGE VIA MICROMODEL
// =============================================================================
//
// This is how MiniLLM "reads KAN logic":
// Gets the MicroModel for `from`, uses relation embedding for `type`
// and a query context embedding to predict edge relevance.
//

double KanAwareMiniLLM::predict_edge(
    ConceptId from, ConceptId /*to*/, RelationType type) const
{
    const MicroModel* model = registry_.get_model(from);
    if (!model) return 0.0;

    const auto& rel_emb = embeddings_.get_relation_embedding(type);
    auto ctx_emb = embeddings_.make_context_embedding("query");

    return model->predict(rel_emb, ctx_emb);
}

// =============================================================================
// EXTRACT MEANING (READ-ONLY)
// =============================================================================
//
// For each active concept:
// 1. Get outgoing relations from LTM (READ-ONLY)
// 2. Score each with predict_edge() to get KAN weight
// 3. Report strongest paths as structured meaning
//

std::vector<MeaningProposal> KanAwareMiniLLM::extract_meaning(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<MeaningProposal> proposals;

    if (active_concepts.empty()) return proposals;

    // Build KAN-scored relation paths for active concepts
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

    // Sort by KAN score descending, take top paths
    std::sort(all_edges.begin(), all_edges.end(),
        [](const ScoredEdge& a, const ScoredEdge& b) {
            return a.kan_score > b.kan_score;
        });

    if (all_edges.empty()) {
        // Fallback: basic activation pattern
        std::ostringstream interp;
        interp << "KAN-scored activation of " << active_concepts.size() << " concepts (no outgoing relations found)";

        proposals.emplace_back(
            ++proposal_counter_,
            active_concepts,
            interp.str(),
            "No outgoing relations available for KAN scoring",
            0.3,
            get_model_id()
        );
        return proposals;
    }

    // Build structured meaning from top KAN-scored paths
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
               << "(KAN:" << std::fixed;
        interp.precision(2);
        interp << e.kan_score << ")--> ["
               << tgt_info->label << "]";
    }

    reasoning << "KAN-scored " << all_edges.size() << " edges across "
              << active_concepts.size() << " concepts. Top edge KAN score: ";
    reasoning.precision(3);
    reasoning << std::fixed << all_edges[0].kan_score;

    // Collect source concepts from top edges
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
        ++proposal_counter_,
        sources,
        interp.str(),
        reasoning.str(),
        std::min(avg_kan, LLM_ONLY_TRUST_CEILING),  // Cap at LLM-only ceiling
        get_model_id()
    );

    return proposals;
}

// =============================================================================
// GENERATE HYPOTHESES (5 KAN-AWARE PATTERNS)
// =============================================================================

std::vector<HypothesisProposal> KanAwareMiniLLM::generate_hypotheses(
    const std::vector<ConceptId>& evidence_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<HypothesisProposal> proposals;

    if (evidence_concepts.size() < 2) return proposals;

    // Collect concept info for evidence
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
    // Find concepts sharing an IS_A parent, both KAN-scored high
    // Keywords: "proportional", "scales with" → LINEAR
    if (weights_.shared_parent > 0.1) {
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

            // KAN-score both children's IS_A edges
            double kan_a = predict_edge(concepts[children_idx[0]].id, parent_id, RelationType::IS_A);
            double kan_b = predict_edge(concepts[children_idx[1]].id, parent_id, RelationType::IS_A);

            if (kan_a < 0.3 || kan_b < 0.3) continue;

            double confidence = weights_.shared_parent * (kan_a + kan_b) / 2.0;
            confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

            std::ostringstream stmt;
            stmt << concepts[children_idx[0]].label << " and "
                 << concepts[children_idx[1]].label
                 << " share parent " << parent_info->label
                 << " — relationship is proportional and scales with shared taxonomy"
                 << " (KAN:" << std::fixed;
            stmt.precision(2);
            stmt << kan_a << "/" << kan_b << ")";

            std::vector<ConceptId> evidence = {
                concepts[children_idx[0]].id,
                concepts[children_idx[1]].id,
                parent_id
            };

            proposals.emplace_back(
                ++proposal_counter_,
                evidence,
                stmt.str(),
                "Shared IS_A parent with high KAN scores suggests proportional relationship",
                std::vector<std::string>{"shared-parent", "proportional", "scales with"},
                confidence,
                get_model_id()
            );
            break;  // One per pattern type
        }
    }

    // ─── Pattern 2: Transitive Causation (A→B→C chain) ──────────────────
    // Find A→B→C chains via CAUSES, predict direct A→C
    // Keywords: "linear", "increases with" → LINEAR
    if (weights_.transitive_causation > 0.1) {
        for (size_t i = 0; i < concepts.size() && proposals.size() < 5; ++i) {
            for (const auto& r_ab : concepts[i].outgoing) {
                if (r_ab.type != RelationType::CAUSES) continue;

                auto b_rels = ltm.get_outgoing_relations(r_ab.target);
                for (const auto& r_bc : b_rels) {
                    if (r_bc.type != RelationType::CAUSES) continue;

                    // Check if direct A→C exists
                    auto a_to_c = ltm.get_relations_between(concepts[i].id, r_bc.target);
                    bool direct_exists = false;
                    for (const auto& r : a_to_c) {
                        if (r.type == RelationType::CAUSES) { direct_exists = true; break; }
                    }

                    double kan_ac = predict_edge(concepts[i].id, r_bc.target, RelationType::CAUSES);

                    auto b_info = ltm.retrieve_concept(r_ab.target);
                    auto c_info = ltm.retrieve_concept(r_bc.target);
                    if (!b_info || !c_info) continue;

                    double confidence = weights_.transitive_causation * kan_ac;
                    confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                    std::ostringstream stmt;
                    stmt << concepts[i].label << " causes " << b_info->label
                         << " which causes " << c_info->label
                         << " — linear transitive chain, " << concepts[i].label
                         << " increases with " << c_info->label
                         << (direct_exists ? " (direct link exists)" : " (no direct link)")
                         << " (KAN:" << std::fixed;
                    stmt.precision(2);
                    stmt << kan_ac << ")";

                    std::vector<ConceptId> evidence = {
                        concepts[i].id, r_ab.target, r_bc.target
                    };

                    proposals.emplace_back(
                        ++proposal_counter_,
                        evidence,
                        stmt.str(),
                        "Transitive causation chain with KAN-predicted direct link",
                        std::vector<std::string>{"transitive-causation", "linear", "increases with"},
                        confidence,
                        get_model_id()
                    );
                    goto done_transitive;  // One per pattern type
                }
            }
        }
        done_transitive:;
    }

    // ─── Pattern 3: Missing Link (KAN predicts >0.5, no relation in LTM) ─
    // Keywords: "threshold", "activation" → THRESHOLD
    if (weights_.missing_link > 0.1) {
        for (size_t i = 0; i < concepts.size() && proposals.size() < 7; ++i) {
            for (size_t j = 0; j < concepts.size(); ++j) {
                if (i == j) continue;

                // Check if any relation exists between i and j
                auto existing = ltm.get_relations_between(concepts[i].id, concepts[j].id);
                if (!existing.empty()) continue;

                // Try multiple relation types
                RelationType try_types[] = {
                    RelationType::CAUSES, RelationType::ENABLES,
                    RelationType::SUPPORTS, RelationType::IS_A
                };

                for (auto rtype : try_types) {
                    double kan = predict_edge(concepts[i].id, concepts[j].id, rtype);
                    if (kan < 0.5) continue;

                    double confidence = weights_.missing_link * kan * 0.8;
                    confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                    std::ostringstream stmt;
                    stmt << "Missing link: " << concepts[i].label
                         << " may have a threshold-based " << relation_type_to_string(rtype)
                         << " relation to " << concepts[j].label
                         << " — activation pattern suggests hidden connection"
                         << " (KAN:" << std::fixed;
                    stmt.precision(2);
                    stmt << kan << ")";

                    std::vector<ConceptId> evidence = {
                        concepts[i].id, concepts[j].id
                    };

                    proposals.emplace_back(
                        ++proposal_counter_,
                        evidence,
                        stmt.str(),
                        "KAN predicts high relevance but no LTM relation exists",
                        std::vector<std::string>{"missing-link", "threshold", "activation"},
                        confidence,
                        get_model_id()
                    );
                    goto done_missing;
                }
            }
        }
        done_missing:;
    }

    // ─── Pattern 4: Weak Strengthening (LTM weight<0.3 but KAN>0.6) ─────
    // Keywords: "proportional", "increases with" → LINEAR
    if (weights_.weak_strengthening > 0.1) {
        for (size_t i = 0; i < concepts.size() && proposals.size() < 9; ++i) {
            for (const auto& r : concepts[i].outgoing) {
                if (r.weight >= 0.3) continue;

                double kan = predict_edge(r.source, r.target, r.type);
                if (kan < 0.6) continue;

                auto tgt_info = ltm.retrieve_concept(r.target);
                if (!tgt_info) continue;

                double confidence = weights_.weak_strengthening * kan * 0.7;
                confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                std::ostringstream stmt;
                stmt << concepts[i].label << " --" << relation_type_to_string(r.type)
                     << "--> " << tgt_info->label
                     << " has weak LTM weight (" << std::fixed;
                stmt.precision(2);
                stmt << r.weight << ") but KAN predicts proportional strength ("
                     << kan << ") — increases with evidence";

                std::vector<ConceptId> evidence = {r.source, r.target};

                proposals.emplace_back(
                    ++proposal_counter_,
                    evidence,
                    stmt.str(),
                    "Low LTM weight contradicted by high KAN prediction — suggests underweighted relation",
                    std::vector<std::string>{"weak-strengthening", "proportional", "increases with"},
                    confidence,
                    get_model_id()
                );
                goto done_weak;
            }
        }
        done_weak:;
    }

    // ─── Pattern 5: Contradictory Signal (LTM>0.7, KAN<0.3 or vice versa) ─
    // Keywords: "conditional", "depends on" → CONDITIONAL
    if (weights_.contradictory_signal > 0.1) {
        for (size_t i = 0; i < concepts.size() && proposals.size() < 10; ++i) {
            for (const auto& r : concepts[i].outgoing) {
                double kan = predict_edge(r.source, r.target, r.type);

                bool ltm_high_kan_low = (r.weight > 0.7 && kan < 0.3);
                bool ltm_low_kan_high = (r.weight < 0.3 && kan > 0.7);

                if (!ltm_high_kan_low && !ltm_low_kan_high) continue;

                auto tgt_info = ltm.retrieve_concept(r.target);
                if (!tgt_info) continue;

                double mismatch = std::abs(r.weight - kan);
                double confidence = weights_.contradictory_signal * mismatch * 0.6;
                confidence = std::min(confidence, LLM_ONLY_TRUST_CEILING);

                std::ostringstream stmt;
                stmt << "Contradictory signal: " << concepts[i].label
                     << " --" << relation_type_to_string(r.type)
                     << "--> " << tgt_info->label << std::fixed;
                stmt.precision(2);
                stmt << " LTM=" << r.weight << " vs KAN=" << kan
                     << " — relationship is conditional, depends on context";

                std::vector<ConceptId> evidence = {r.source, r.target};

                proposals.emplace_back(
                    ++proposal_counter_,
                    evidence,
                    stmt.str(),
                    ltm_high_kan_low
                        ? "High LTM weight but low KAN score — may be outdated or contextual"
                        : "Low LTM weight but high KAN score — may be emerging relationship",
                    std::vector<std::string>{"contradictory-signal", "conditional", "depends on"},
                    confidence,
                    get_model_id()
                );
                goto done_contradictory;
            }
        }
        done_contradictory:;
    }

    return proposals;
}

// =============================================================================
// DETECT ANALOGIES (KAN-SCORED RELATION DISTRIBUTIONS)
// =============================================================================
//
// Compute KAN-scored relation-type frequency distributions for two concept sets.
// Cosine similarity of distributions → structural analogy score.
//

std::vector<AnalogyProposal> KanAwareMiniLLM::detect_analogies(
    const std::vector<ConceptId>& concept_set_a,
    const std::vector<ConceptId>& concept_set_b,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<AnalogyProposal> proposals;

    if (concept_set_a.empty() || concept_set_b.empty()) return proposals;

    // Build relation-type frequency distribution weighted by KAN scores
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

    // Cosine similarity between distributions
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
    mapping << "KAN-scored relation distribution similarity: " << std::fixed;
    mapping.precision(3);
    mapping << similarity << " across " << dist_a.size()
            << " vs " << dist_b.size() << " relation types";

    proposals.emplace_back(
        ++proposal_counter_,
        concept_set_a,
        concept_set_b,
        mapping.str(),
        similarity,
        std::min(similarity * 0.8, LLM_ONLY_TRUST_CEILING),
        get_model_id()
    );

    return proposals;
}

// =============================================================================
// DETECT CONTRADICTIONS (EXPLICIT + KAN MISMATCH)
// =============================================================================

std::vector<ContradictionProposal> KanAwareMiniLLM::detect_contradictions(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<ContradictionProposal> proposals;

    if (active_concepts.size() < 2) return proposals;

    // 1. Find explicit CONTRADICTS relations
    for (size_t i = 0; i < active_concepts.size() && i < 10; ++i) {
        auto rels = ltm.get_outgoing_relations(active_concepts[i]);
        for (const auto& r : rels) {
            if (r.type != RelationType::CONTRADICTS) continue;

            // Check if target is also active
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
                 << " (KAN:" << std::fixed;
            desc.precision(2);
            desc << kan << ", weight:" << r.weight
                 << (target_active ? ", BOTH ACTIVE" : "") << ")";

            double severity = r.weight * (target_active ? 1.0 : 0.6);

            proposals.emplace_back(
                ++proposal_counter_,
                r.source,
                r.target,
                desc.str(),
                "Explicit CONTRADICTS relation in KG",
                severity,
                std::min(kan * 0.9, LLM_ONLY_TRUST_CEILING),
                get_model_id()
            );
        }
    }

    // 2. KAN prediction mismatch: high-weight relation with low KAN score
    for (size_t i = 0; i < active_concepts.size() && i < 8; ++i) {
        auto rels = ltm.get_outgoing_relations(active_concepts[i]);
        for (const auto& r : rels) {
            if (r.type == RelationType::CONTRADICTS) continue;
            if (r.weight < 0.6) continue;

            double kan = predict_edge(r.source, r.target, r.type);
            if (kan > 0.3) continue;  // No mismatch

            auto src_info = ltm.retrieve_concept(r.source);
            auto tgt_info = ltm.retrieve_concept(r.target);
            if (!src_info || !tgt_info) continue;

            double mismatch = r.weight - kan;

            std::ostringstream desc;
            desc << "KAN mismatch: " << src_info->label
                 << " --" << relation_type_to_string(r.type)
                 << "--> " << tgt_info->label << std::fixed;
            desc.precision(2);
            desc << " (LTM:" << r.weight << " vs KAN:" << kan << ")";

            proposals.emplace_back(
                ++proposal_counter_,
                r.source,
                r.target,
                desc.str(),
                "High LTM weight but low KAN prediction — possible stale or contextual relation",
                mismatch * 0.7,
                std::min(mismatch * 0.5, LLM_ONLY_TRUST_CEILING),
                get_model_id()
            );

            if (proposals.size() >= 5) return proposals;
        }
    }

    return proposals;
}

// =============================================================================
// TRAIN FROM VALIDATION (KAN FEEDBACK → PATTERN WEIGHTS)
// =============================================================================
//
// RL-style weight adjustment:
// +0.1 for validated patterns, -0.05 for refuted
// Clamp all weights to [0.1, 3.0]
//

void KanAwareMiniLLM::train_from_validation(
    const std::vector<ValidationResult>& results)
{
    auto clamp_weight = [](double& w) {
        if (w < 0.1) w = 0.1;
        if (w > 3.0) w = 3.0;
    };

    auto adjust = [&](double& w, bool validated) {
        w += validated ? 0.1 : -0.05;
        clamp_weight(w);
    };

    for (const auto& vr : results) {
        // Match detected_patterns from the original hypothesis to weight keys
        // The hypothesis proposals we generated include pattern tags like
        // "shared-parent", "transitive-causation", "missing-link", etc.
        const auto& pat = vr.pattern;

        // We can match by RelationshipPattern or by checking the explanation
        switch (pat) {
            case RelationshipPattern::LINEAR:
                // LINEAR maps to shared-parent, transitive-causation, weak-strengthening
                adjust(weights_.shared_parent, vr.validated);
                adjust(weights_.transitive_causation, vr.validated);
                adjust(weights_.weak_strengthening, vr.validated);
                break;
            case RelationshipPattern::THRESHOLD:
                // THRESHOLD maps to missing-link
                adjust(weights_.missing_link, vr.validated);
                break;
            case RelationshipPattern::CONDITIONAL:
                // CONDITIONAL maps to contradictory-signal
                adjust(weights_.contradictory_signal, vr.validated);
                break;
            default:
                // For other patterns, adjust all weights slightly
                adjust(weights_.shared_parent, vr.validated);
                adjust(weights_.transitive_causation, vr.validated);
                adjust(weights_.missing_link, vr.validated);
                adjust(weights_.weak_strengthening, vr.validated);
                adjust(weights_.contradictory_signal, vr.validated);
                break;
        }
    }
}

} // namespace brain19

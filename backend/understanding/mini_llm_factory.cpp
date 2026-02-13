#include "mini_llm_factory.hpp"
#include "../ltm/relation.hpp"  // relation_type_to_string
#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

namespace brain19 {

// =============================================================================
// MINI-LLM FACTORY IMPLEMENTATION
// =============================================================================

MiniLLMFactory::MiniLLMFactory()
    : created_count_(0)
{
}

std::string MiniLLMFactory::extract_knowledge_context(
    const std::vector<ConceptId>& concepts,
    const LongTermMemory& ltm
) const {
    std::ostringstream ctx;

    for (auto cid : concepts) {
        auto cinfo = ltm.retrieve_concept(cid);
        if (!cinfo) continue;

        ctx << "CONCEPT:" << cid << ":" << cinfo->label
            << ":" << cinfo->definition << "\n";

        // Outgoing relations
        auto out_rels = ltm.get_outgoing_relations(cid);
        for (const auto& r : out_rels) {
            ctx << "REL:" << r.source << ":"
                << relation_type_to_string(r.type) << ":"
                << r.target << ":" << r.weight << "\n";
        }

        // Incoming relations (capped at 10 per concept)
        auto in_rels = ltm.get_incoming_relations(cid);
        size_t in_count = 0;
        for (const auto& r : in_rels) {
            if (in_count >= 10) break;
            ctx << "REL:" << r.source << ":"
                << relation_type_to_string(r.type) << ":"
                << r.target << ":" << r.weight << "\n";
            ++in_count;
        }
    }

    return ctx.str();
}

std::string MiniLLMFactory::build_specialization_prompt(
    const std::vector<ConceptId>& concepts,
    const LongTermMemory& ltm
) const {
    // Encode focal concept IDs as parseable prefix
    std::ostringstream prompt;
    prompt << "FOCAL_CONCEPTS:";
    for (size_t i = 0; i < concepts.size(); ++i) {
        if (i > 0) prompt << ",";
        prompt << concepts[i];
    }
    prompt << "\n";

    // Append knowledge context
    prompt << extract_knowledge_context(concepts, ltm);

    return prompt.str();
}

std::unique_ptr<MiniLLM> MiniLLMFactory::create_specialized_mini_llm(
    const std::vector<ConceptId>& focal_concepts,
    const LongTermMemory& ltm,
    const std::string& specialization_name
) {
    auto context = build_specialization_prompt(focal_concepts, ltm);
    auto llm = std::make_unique<SpecializedMiniLLM>(specialization_name, context);
    ++created_count_;
    return llm;
}

std::unique_ptr<MiniLLM> MiniLLMFactory::create_relation_expert_mini_llm(
    RelationType relation_type,
    const std::vector<ConceptId>& domain_concepts,
    const LongTermMemory& ltm,
    const std::string& specialization_name
) {
    // Filter domain to concepts with outgoing relations of given type
    std::vector<ConceptId> filtered;
    for (auto cid : domain_concepts) {
        auto rels = ltm.get_outgoing_relations(cid);
        for (const auto& r : rels) {
            if (r.type == relation_type) {
                filtered.push_back(cid);
                break;
            }
        }
    }

    if (filtered.empty()) {
        filtered = domain_concepts;  // Fallback to full domain
    }

    return create_specialized_mini_llm(filtered, ltm, specialization_name);
}

std::unique_ptr<MiniLLM> MiniLLMFactory::create_epistemic_expert_mini_llm(
    EpistemicType type,
    const std::vector<ConceptId>& domain_concepts,
    const LongTermMemory& ltm,
    const std::string& specialization_name
) {
    // Filter to concepts matching EpistemicType
    std::vector<ConceptId> filtered;
    for (auto cid : domain_concepts) {
        auto cinfo = ltm.retrieve_concept(cid);
        if (cinfo && cinfo->epistemic.type == type) {
            filtered.push_back(cid);
        }
    }

    if (filtered.empty()) {
        filtered = domain_concepts;  // Fallback to full domain
    }

    return create_specialized_mini_llm(filtered, ltm, specialization_name);
}

// =============================================================================
// SPECIALIZED MINI-LLM IMPLEMENTATION
// =============================================================================

SpecializedMiniLLM::SpecializedMiniLLM(
    const std::string& name,
    const std::string& specialization_context
) : name_(name)
  , specialization_context_(specialization_context)
  , proposal_counter_(0)
{
    // Parse "FOCAL_CONCEPTS:1,2,3\n" prefix from context
    const std::string prefix = "FOCAL_CONCEPTS:";
    if (specialization_context_.rfind(prefix, 0) == 0) {
        auto newline_pos = specialization_context_.find('\n');
        if (newline_pos != std::string::npos) {
            std::string ids_str = specialization_context_.substr(
                prefix.size(), newline_pos - prefix.size());
            std::istringstream iss(ids_str);
            std::string token;
            while (std::getline(iss, token, ',')) {
                if (!token.empty()) {
                    try {
                        focal_concepts_.push_back(
                            static_cast<ConceptId>(std::stoull(token)));
                    } catch (...) {
                        // Skip malformed IDs
                    }
                }
            }
        }
    }
}

std::string SpecializedMiniLLM::get_model_id() const {
    return "specialized-mini-llm:" + name_;
}

bool SpecializedMiniLLM::is_relevant_for(
    const std::vector<ConceptId>& concepts,
    const LongTermMemory& ltm
) const {
    std::unordered_set<ConceptId> focal_set(
        focal_concepts_.begin(), focal_concepts_.end());

    for (auto cid : concepts) {
        // Direct match
        if (focal_set.count(cid)) return true;

        // 1-hop IS_A check: if concept IS_A a focal concept
        auto rels = ltm.get_outgoing_relations(cid);
        for (const auto& r : rels) {
            if (r.type == RelationType::IS_A && focal_set.count(r.target)) {
                return true;
            }
        }
    }
    return false;
}

// =============================================================================
// MEANING EXTRACTION — Graph-based
// =============================================================================

std::vector<MeaningProposal> SpecializedMiniLLM::extract_meaning(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<MeaningProposal> proposals;

    if (active_concepts.empty() || focal_concepts_.empty()) {
        return proposals;
    }

    // Find overlap between active and focal concepts
    std::unordered_set<ConceptId> focal_set(
        focal_concepts_.begin(), focal_concepts_.end());
    std::vector<ConceptId> overlap;
    for (auto cid : active_concepts) {
        if (focal_set.count(cid)) {
            overlap.push_back(cid);
        }
    }

    if (overlap.empty()) return proposals;

    // Ratio = how many of our focal concepts are active (domain coverage)
    // NOT: how many active concepts are focal (penalizes multi-domain queries)
    double overlap_ratio = static_cast<double>(overlap.size()) /
                           static_cast<double>(focal_concepts_.size());
    // Clamp: even 1 focal match → meaningful output (must clear 0.3 threshold)
    overlap_ratio = std::max(overlap_ratio, 0.5);

    // For each overlapping concept, build a sentence from its relations
    for (auto cid : overlap) {
        auto cinfo = ltm.retrieve_concept(cid);
        if (!cinfo) continue;

        std::ostringstream interpretation;
        interpretation << cinfo->label;

        auto rels = ltm.get_outgoing_relations(cid);

        // Group relations by type
        std::unordered_map<uint16_t, std::vector<const RelationInfo*>> by_type;
        for (const auto& r : rels) {
            by_type[static_cast<uint16_t>(r.type)].push_back(&r);
        }

        bool has_content = false;

        // IS_A
        auto it = by_type.find(static_cast<uint16_t>(RelationType::IS_A));
        if (it != by_type.end() && !it->second.empty()) {
            auto target = ltm.retrieve_concept(it->second[0]->target);
            if (target) {
                interpretation << " is a " << target->label;
                has_content = true;
            }
        }

        // HAS_PROPERTY
        it = by_type.find(static_cast<uint16_t>(RelationType::HAS_PROPERTY));
        if (it != by_type.end()) {
            for (size_t i = 0; i < it->second.size() && i < 3; ++i) {
                auto target = ltm.retrieve_concept(it->second[i]->target);
                if (target) {
                    interpretation << (has_content ? ", with property " : " with property ")
                                   << target->label;
                    has_content = true;
                }
            }
        }

        // CAUSES
        it = by_type.find(static_cast<uint16_t>(RelationType::CAUSES));
        if (it != by_type.end()) {
            for (size_t i = 0; i < it->second.size() && i < 2; ++i) {
                auto target = ltm.retrieve_concept(it->second[i]->target);
                if (target) {
                    interpretation << ", causing " << target->label;
                    has_content = true;
                }
            }
        }

        // PART_OF
        it = by_type.find(static_cast<uint16_t>(RelationType::PART_OF));
        if (it != by_type.end() && !it->second.empty()) {
            auto target = ltm.retrieve_concept(it->second[0]->target);
            if (target) {
                interpretation << ", part of " << target->label;
                has_content = true;
            }
        }

        if (!has_content) continue;

        double confidence = std::min(0.7, cinfo->epistemic.trust * overlap_ratio);

        proposals.emplace_back(
            ++proposal_counter_,
            std::vector<ConceptId>{cid},
            interpretation.str(),
            "Graph-based meaning extraction from " + name_ + " domain",
            confidence,
            get_model_id()
        );
    }

    return proposals;
}

// =============================================================================
// HYPOTHESIS GENERATION — 3 patterns
// =============================================================================

std::vector<HypothesisProposal> SpecializedMiniLLM::generate_hypotheses(
    const std::vector<ConceptId>& evidence_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<HypothesisProposal> proposals;

    if (evidence_concepts.size() < 2) return proposals;

    // Limit scope to focal concepts for efficiency
    std::unordered_set<ConceptId> focal_set(
        focal_concepts_.begin(), focal_concepts_.end());
    std::vector<ConceptId> relevant;
    for (auto cid : evidence_concepts) {
        if (focal_set.count(cid)) {
            relevant.push_back(cid);
        }
    }
    if (relevant.size() < 2) return proposals;

    // ── Pattern 1: Shared IS_A parent → generalization hypothesis ──
    {
        // Map parent → children
        std::unordered_map<ConceptId, std::vector<ConceptId>> parent_to_children;
        for (auto cid : relevant) {
            auto rels = ltm.get_outgoing_relations(cid);
            for (const auto& r : rels) {
                if (r.type == RelationType::IS_A) {
                    parent_to_children[r.target].push_back(cid);
                }
            }
        }

        for (const auto& [parent_id, children] : parent_to_children) {
            if (children.size() < 2) continue;
            auto parent_info = ltm.retrieve_concept(parent_id);
            if (!parent_info) continue;

            std::ostringstream stmt;
            stmt << "Generalization: ";
            for (size_t i = 0; i < children.size() && i < 4; ++i) {
                auto ci = ltm.retrieve_concept(children[i]);
                if (ci) {
                    if (i > 0) stmt << ", ";
                    stmt << ci->label;
                }
            }
            stmt << " share common category " << parent_info->label;

            proposals.emplace_back(
                ++proposal_counter_,
                children,
                stmt.str(),
                "Shared IS_A parent detected in " + name_ + " domain",
                std::vector<std::string>{"shared_parent", "generalization"},
                std::min(0.6, 0.4 + 0.05 * children.size()),
                get_model_id()
            );
        }
    }

    // ── Pattern 2: Missing property inheritance ──
    // A IS_A B, B HAS_PROPERTY P, A lacks P → "A might have P"
    {
        for (auto cid : relevant) {
            auto rels = ltm.get_outgoing_relations(cid);
            for (const auto& r : rels) {
                if (r.type != RelationType::IS_A) continue;

                // Get parent's properties
                auto parent_rels = ltm.get_outgoing_relations(r.target);
                // Get child's existing property targets
                std::unordered_set<ConceptId> child_props;
                for (const auto& cr : rels) {
                    if (cr.type == RelationType::HAS_PROPERTY) {
                        child_props.insert(cr.target);
                    }
                }

                for (const auto& pr : parent_rels) {
                    if (pr.type != RelationType::HAS_PROPERTY) continue;
                    if (child_props.count(pr.target)) continue;  // Already has it

                    auto child_info = ltm.retrieve_concept(cid);
                    auto parent_info = ltm.retrieve_concept(r.target);
                    auto prop_info = ltm.retrieve_concept(pr.target);
                    if (!child_info || !parent_info || !prop_info) continue;

                    std::ostringstream stmt;
                    stmt << child_info->label << " might have property "
                         << prop_info->label << " (inherited from "
                         << parent_info->label << ")";

                    proposals.emplace_back(
                        ++proposal_counter_,
                        std::vector<ConceptId>{cid, r.target, pr.target},
                        stmt.str(),
                        "Property inheritance gap in " + name_ + " domain",
                        std::vector<std::string>{"property_inheritance", "gap_filling"},
                        std::min(0.5, pr.weight * 0.45),
                        get_model_id()
                    );

                    // Limit to 3 inheritance hypotheses per concept
                    if (proposals.size() > 10) break;
                }
            }
        }
    }

    // ── Pattern 3: Transitive causation ──
    // A CAUSES B, B CAUSES C → "A might cause C"
    {
        for (auto cid : relevant) {
            auto rels_a = ltm.get_outgoing_relations(cid);
            for (const auto& r1 : rels_a) {
                if (r1.type != RelationType::CAUSES) continue;

                auto rels_b = ltm.get_outgoing_relations(r1.target);
                for (const auto& r2 : rels_b) {
                    if (r2.type != RelationType::CAUSES) continue;
                    if (r2.target == cid) continue;  // Skip cycles

                    auto a_info = ltm.retrieve_concept(cid);
                    auto b_info = ltm.retrieve_concept(r1.target);
                    auto c_info = ltm.retrieve_concept(r2.target);
                    if (!a_info || !b_info || !c_info) continue;

                    std::ostringstream stmt;
                    stmt << a_info->label << " might cause "
                         << c_info->label << " (via " << b_info->label << ")";

                    proposals.emplace_back(
                        ++proposal_counter_,
                        std::vector<ConceptId>{cid, r1.target, r2.target},
                        stmt.str(),
                        "Transitive causation in " + name_ + " domain",
                        std::vector<std::string>{"transitive_causation", "chain_reasoning"},
                        std::min(0.4, r1.weight * r2.weight * 0.35),
                        get_model_id()
                    );
                }
            }
        }
    }

    return proposals;
}

// =============================================================================
// ANALOGY DETECTION — Relation-type vector similarity
// =============================================================================

std::vector<AnalogyProposal> SpecializedMiniLLM::detect_analogies(
    const std::vector<ConceptId>& concept_set_a,
    const std::vector<ConceptId>& concept_set_b,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<AnalogyProposal> proposals;

    if (concept_set_a.empty() || concept_set_b.empty()) {
        return proposals;
    }

    // Build relation-type frequency signature per set
    auto build_signature = [&](const std::vector<ConceptId>& cset)
        -> std::unordered_map<uint16_t, size_t>
    {
        std::unordered_map<uint16_t, size_t> sig;
        for (auto cid : cset) {
            auto rels = ltm.get_outgoing_relations(cid);
            for (const auto& r : rels) {
                sig[static_cast<uint16_t>(r.type)]++;
            }
        }
        return sig;
    };

    auto sig_a = build_signature(concept_set_a);
    auto sig_b = build_signature(concept_set_b);

    if (sig_a.empty() || sig_b.empty()) return proposals;

    // Compute similarity: shared relation-type counts / max total
    size_t shared_counts = 0;
    size_t total_a = 0, total_b = 0;

    for (const auto& [type_id, count] : sig_a) {
        total_a += count;
        auto it = sig_b.find(type_id);
        if (it != sig_b.end()) {
            shared_counts += std::min(count, it->second);
        }
    }
    for (const auto& [type_id, count] : sig_b) {
        total_b += count;
    }

    size_t max_total = std::max(total_a, total_b);
    if (max_total == 0) return proposals;

    double similarity = static_cast<double>(shared_counts) /
                        static_cast<double>(max_total);

    if (similarity < 0.1) return proposals;

    double confidence = similarity * 0.6;

    std::ostringstream mapping;
    mapping << "Structural analogy between " << concept_set_a.size()
            << " concepts and " << concept_set_b.size()
            << " concepts in " << name_ << " domain"
            << " (relation pattern similarity: " << similarity << ")";

    proposals.emplace_back(
        ++proposal_counter_,
        concept_set_a,
        concept_set_b,
        mapping.str(),
        similarity,
        confidence,
        get_model_id()
    );

    return proposals;
}

// =============================================================================
// CONTRADICTION DETECTION — 3 checks
// =============================================================================

std::vector<ContradictionProposal> SpecializedMiniLLM::detect_contradictions(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    ContextId /*context*/
) const {
    std::vector<ContradictionProposal> proposals;

    if (active_concepts.size() < 2) return proposals;

    std::unordered_set<ConceptId> active_set(
        active_concepts.begin(), active_concepts.end());
    std::unordered_set<ConceptId> focal_set(
        focal_concepts_.begin(), focal_concepts_.end());

    // ── Check 1: Explicit CONTRADICTS relations between active concepts ──
    for (auto cid : active_concepts) {
        auto rels = ltm.get_outgoing_relations(cid);
        for (const auto& r : rels) {
            if (r.type != RelationType::CONTRADICTS) continue;
            if (!active_set.count(r.target)) continue;
            // Avoid duplicate (a,b) and (b,a)
            if (r.target < cid) continue;

            auto a_info = ltm.retrieve_concept(cid);
            auto b_info = ltm.retrieve_concept(r.target);
            if (!a_info || !b_info) continue;

            double avg_trust = (a_info->epistemic.trust + b_info->epistemic.trust) / 2.0;

            proposals.emplace_back(
                ++proposal_counter_,
                cid,
                r.target,
                a_info->label + " contradicts " + b_info->label,
                "Explicit CONTRADICTS relation found in " + name_ + " domain",
                avg_trust,   // severity
                0.8,         // confidence
                get_model_id()
            );
        }
    }

    // ── Check 2: Circular IS_A chains (walk up to 10 hops) ──
    for (auto cid : active_concepts) {
        if (!focal_set.count(cid)) continue;

        ConceptId current = cid;
        std::unordered_set<ConceptId> visited;
        bool cycle_found = false;

        for (size_t depth = 0; depth < 10; ++depth) {
            if (visited.count(current)) {
                cycle_found = true;
                break;
            }
            visited.insert(current);

            auto rels = ltm.get_outgoing_relations(current);
            bool found_parent = false;
            for (const auto& r : rels) {
                if (r.type == RelationType::IS_A) {
                    current = r.target;
                    found_parent = true;
                    break;
                }
            }
            if (!found_parent) break;
        }

        if (cycle_found) {
            auto cinfo = ltm.retrieve_concept(cid);
            std::string label = cinfo ? cinfo->label : std::to_string(cid);

            proposals.emplace_back(
                ++proposal_counter_,
                cid,
                current,  // The concept where cycle was detected
                "Circular IS_A chain detected starting from " + label,
                "IS_A hierarchy cycle in " + name_ + " domain",
                0.9,   // severity
                0.9,   // confidence
                get_model_id()
            );
        }
    }

    // ── Check 3: INVALIDATED + ACTIVE co-activation (scoped to focal) ──
    for (size_t i = 0; i < active_concepts.size(); ++i) {
        if (!focal_set.count(active_concepts[i])) continue;
        auto a_info = ltm.retrieve_concept(active_concepts[i]);
        if (!a_info) continue;

        for (size_t j = i + 1; j < active_concepts.size(); ++j) {
            if (!focal_set.count(active_concepts[j])) continue;
            auto b_info = ltm.retrieve_concept(active_concepts[j]);
            if (!b_info) continue;

            bool mismatch =
                (a_info->epistemic.status == EpistemicStatus::INVALIDATED &&
                 b_info->epistemic.status == EpistemicStatus::ACTIVE) ||
                (a_info->epistemic.status == EpistemicStatus::ACTIVE &&
                 b_info->epistemic.status == EpistemicStatus::INVALIDATED);

            if (!mismatch) continue;

            proposals.emplace_back(
                ++proposal_counter_,
                active_concepts[i],
                active_concepts[j],
                "Epistemic mismatch: " + a_info->label + " (" +
                    epistemic_status_to_string(a_info->epistemic.status) +
                    ") co-active with " + b_info->label + " (" +
                    epistemic_status_to_string(b_info->epistemic.status) + ")",
                "INVALIDATED/ACTIVE co-activation in " + name_ + " domain",
                0.6,   // severity
                0.5,   // confidence
                get_model_id()
            );
        }
    }

    return proposals;
}

} // namespace brain19

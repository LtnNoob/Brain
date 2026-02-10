#include "concept_proposal.hpp"
#include <algorithm>
#include <unordered_set>

namespace brain19 {

ConceptProposer::ConceptProposer(const LongTermMemory& ltm)
    : ltm_(ltm)
{
}

std::vector<ConceptProposal> ConceptProposer::from_curiosity(
    const std::vector<CuriosityTrigger>& triggers)
{
    std::vector<ConceptProposal> proposals;

    for (const auto& trigger : triggers) {
        std::string label;
        std::string description;
        std::string source_str;
        EpistemicType type = EpistemicType::SPECULATION;
        double trust = 0.2;

        switch (trigger.type) {
            case TriggerType::SHALLOW_RELATIONS:
                if (trigger.related_concept_ids.size() < 2) continue;
                label = "bridge_" + std::to_string(trigger.related_concept_ids[0]) +
                        "_" + std::to_string(trigger.related_concept_ids[1]);
                description = "Potential bridging concept for shallowly related concepts";
                source_str = "curiosity:shallow_relations";
                trust = 0.15;
                break;

            case TriggerType::MISSING_DEPTH:
                if (trigger.related_concept_ids.empty()) continue;
                label = "depth_" + std::to_string(trigger.related_concept_ids[0]);
                description = "Deeper structural concept underlying repeated pattern";
                source_str = "curiosity:missing_depth";
                trust = 0.2;
                break;

            case TriggerType::LOW_EXPLORATION:
                label = "exploration_ctx_" + std::to_string(trigger.context_id);
                description = "Unexplored concept area detected in stable context";
                source_str = "curiosity:low_exploration";
                trust = 0.1;
                break;

            case TriggerType::RECURRENT_WITHOUT_FUNCTION:
                if (trigger.related_concept_ids.empty()) continue;
                label = "function_" + std::to_string(trigger.related_concept_ids[0]);
                description = "Potential functional concept for recurrent activation pattern";
                source_str = "curiosity:recurrent_pattern";
                trust = 0.25;
                type = EpistemicType::HYPOTHESIS;
                break;

            default:
                continue;
        }

        if (!concept_exists_similar(label)) {
            proposals.emplace_back(
                label, description, type, trust,
                source_str, trigger.related_concept_ids,
                trigger.description
            );
        }
    }

    return proposals;
}

std::vector<ConceptProposal> ConceptProposer::from_relevance_anomalies(
    const RelevanceMap& map, double anomaly_threshold)
{
    std::vector<ConceptProposal> proposals;

    auto high_scores = map.above_threshold(anomaly_threshold);
    ConceptId source_id = map.source();

    for (const auto& [target_id, score] : high_scores) {
        // Check if there's already a direct relation
        auto relations = ltm_.get_relations_between(source_id, target_id);
        if (!relations.empty()) continue;  // Already connected, not anomalous

        std::string label = "anomaly_" + std::to_string(source_id) +
                            "_" + std::to_string(target_id);
        std::string description = "Unexpected high relevance between concepts " +
                                  std::to_string(source_id) + " and " +
                                  std::to_string(target_id) +
                                  " (score: " + std::to_string(score) + ")";
        std::string source_str = "relevance:anomaly:" +
                                 std::to_string(source_id) + "↔" +
                                 std::to_string(target_id);

        double trust = std::min(0.3, score * 0.3);

        if (!concept_exists_similar(label)) {
            proposals.emplace_back(
                label, description,
                EpistemicType::SPECULATION, trust,
                source_str,
                std::vector<ConceptId>{source_id, target_id},
                "High relevance score without direct relation suggests hidden connection"
            );
        }
    }

    return proposals;
}

std::vector<ConceptProposal> ConceptProposer::from_analogies(
    const std::vector<AnalogyProposal>& analogies)
{
    std::vector<ConceptProposal> proposals;

    for (const auto& analogy : analogies) {
        if (analogy.structural_similarity < 0.5) continue;

        // Combine source and target domains as evidence
        std::vector<ConceptId> evidence;
        evidence.insert(evidence.end(),
                        analogy.source_domain.begin(),
                        analogy.source_domain.end());
        evidence.insert(evidence.end(),
                        analogy.target_domain.begin(),
                        analogy.target_domain.end());

        std::string label = "analogy_" + std::to_string(analogy.proposal_id);
        std::string description = "Cross-domain analogy: " + analogy.mapping_description;
        std::string source_str = "analogy:" + analogy.source_model;

        double trust = std::min(0.4, analogy.structural_similarity * 0.5);
        EpistemicType type = (analogy.structural_similarity > 0.7)
                                 ? EpistemicType::HYPOTHESIS
                                 : EpistemicType::SPECULATION;

        proposals.emplace_back(
            label, description, type, trust,
            source_str, evidence,
            analogy.mapping_description
        );
    }

    return proposals;
}

std::vector<ConceptProposal> ConceptProposer::rank_proposals(
    std::vector<ConceptProposal>& proposals, size_t max_k)
{
    // Deduplicate by label
    std::unordered_set<std::string> seen_labels;
    std::vector<ConceptProposal> unique;
    unique.reserve(proposals.size());

    for (auto& p : proposals) {
        if (seen_labels.insert(p.label).second) {
            unique.push_back(std::move(p));
        }
    }

    // Sort by quality score (descending)
    std::sort(unique.begin(), unique.end(),
        [this](const ConceptProposal& a, const ConceptProposal& b) {
            return compute_quality_score(a) > compute_quality_score(b);
        });

    // Truncate to max_k
    if (unique.size() > max_k) {
        unique.erase(unique.begin() + static_cast<std::ptrdiff_t>(max_k), unique.end());
    }

    return unique;
}

bool ConceptProposer::concept_exists_similar(const std::string& label) const {
    // Check all existing concepts for label match
    auto all_ids = ltm_.get_all_concept_ids();
    for (auto id : all_ids) {
        auto cinfo = ltm_.retrieve_concept(id);
        if (cinfo && cinfo->label == label) {
            return true;
        }
    }
    return false;
}

double ConceptProposer::compute_quality_score(const ConceptProposal& proposal) const {
    double score = 0.0;

    // Evidence count contributes
    score += static_cast<double>(proposal.evidence.size()) * 0.15;

    // Trust level
    score += proposal.initial_trust;

    // HYPOTHESIS > SPECULATION
    if (proposal.initial_type == EpistemicType::HYPOTHESIS) {
        score += 0.2;
    }

    // Verified evidence (concepts that actually exist in LTM)
    size_t verified = 0;
    for (auto id : proposal.evidence) {
        if (ltm_.exists(id)) {
            auto c = ltm_.retrieve_concept(id);
            if (c && c->epistemic.is_active()) {
                ++verified;
            }
        }
    }
    score += static_cast<double>(verified) * 0.1;

    return score;
}

} // namespace brain19

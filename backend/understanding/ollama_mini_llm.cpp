#include "ollama_mini_llm.hpp"
#include <sstream>
#include <algorithm>

namespace brain19 {

// =============================================================================
// CONSTRUCTOR / DESTRUCTOR
// =============================================================================

OllamaMiniLLM::OllamaMiniLLM(const OllamaConfig& config)
    : config_(config)
    , proposal_counter_(0)
{
    ollama_.initialize(config_);
}

OllamaMiniLLM::~OllamaMiniLLM() = default;

std::string OllamaMiniLLM::get_model_id() const {
    return "ollama-" + config_.model;
}

bool OllamaMiniLLM::is_available() const {
    return ollama_.is_available();
}

// =============================================================================
// HELPER: BUILD CONCEPT DESCRIPTION (READ-ONLY)
// =============================================================================

std::string OllamaMiniLLM::build_concept_description(
    const std::vector<ConceptId>& concepts,
    const LongTermMemory& ltm
) const {
    std::ostringstream desc;

    desc << "Active concepts in working memory:\n";

    for (size_t i = 0; i < concepts.size() && i < 10; ++i) {
        auto concept_info = ltm.retrieve_concept(concepts[i]);  // READ-ONLY
        if (concept_info.has_value()) {
            desc << "- " << concept_info->label;

            if (!concept_info->definition.empty()) {
                desc << ": " << concept_info->definition;
            }

            // Include epistemic metadata for context (READ-ONLY)
            desc << " (epistemic: ";
            switch (concept_info->epistemic.type) {
                case EpistemicType::FACT: desc << "FACT"; break;
                case EpistemicType::THEORY: desc << "THEORY"; break;
                case EpistemicType::HYPOTHESIS: desc << "HYPOTHESIS"; break;
                case EpistemicType::SPECULATION: desc << "SPECULATION"; break;
                default: desc << "UNKNOWN"; break;
            }
            desc << ", trust=" << concept_info->epistemic.trust << ")\n";
        }
    }

    if (concepts.size() > 10) {
        desc << "... and " << (concepts.size() - 10) << " more concepts\n";
    }

    return desc.str();
}

// =============================================================================
// HELPER: PARSE LLM RESPONSE
// =============================================================================

std::string OllamaMiniLLM::extract_interpretation(const std::string& response) const {
    // Simple extraction: first sentence or first 200 chars
    size_t end = response.find('.');
    if (end != std::string::npos && end < 200) {
        return response.substr(0, end + 1);
    }
    return response.substr(0, std::min(size_t(200), response.length()));
}

std::string OllamaMiniLLM::extract_reasoning(const std::string& response) const {
    // Extract everything after first sentence
    size_t start = response.find('.');
    if (start != std::string::npos && start + 2 < response.length()) {
        return response.substr(start + 2);
    }
    return response;
}

double OllamaMiniLLM::estimate_confidence(const std::string& response) const {
    // Heuristic: presence of uncertainty words lowers confidence
    std::string lower_response = response;
    std::transform(lower_response.begin(), lower_response.end(),
                   lower_response.begin(), ::tolower);

    double confidence = 0.7;  // Base confidence

    // Reduce confidence for uncertainty markers
    if (lower_response.find("might") != std::string::npos) confidence -= 0.1;
    if (lower_response.find("maybe") != std::string::npos) confidence -= 0.1;
    if (lower_response.find("possibly") != std::string::npos) confidence -= 0.1;
    if (lower_response.find("uncertain") != std::string::npos) confidence -= 0.2;

    // Increase confidence for certainty markers
    if (lower_response.find("clearly") != std::string::npos) confidence += 0.1;
    if (lower_response.find("obviously") != std::string::npos) confidence += 0.1;

    return std::max(0.1, std::min(0.9, confidence));
}

// =============================================================================
// MEANING EXTRACTION
// =============================================================================

std::vector<MeaningProposal> OllamaMiniLLM::extract_meaning(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) const {
    std::vector<MeaningProposal> proposals;

    if (active_concepts.empty()) {
        return proposals;
    }

    if (!is_available()) {
        // Fallback: return empty if Ollama not available
        return proposals;
    }

    // Build prompt
    std::ostringstream prompt;
    prompt << "You are a semantic analyzer for a cognitive architecture system.\n\n";
    prompt << build_concept_description(active_concepts, ltm);
    prompt << "\nTask: Provide a brief semantic interpretation of what these ";
    prompt << "co-activated concepts might mean together. ";
    prompt << "Focus on relationships and patterns. Be concise (1-2 sentences).\n";
    prompt << "Note: Your interpretation is a HYPOTHESIS, not a fact.\n";

    // Call Ollama
    std::vector<OllamaMessage> messages = {
        {"system", "You are a helpful semantic analysis assistant. You provide brief, analytical interpretations."},
        {"user", prompt.str()}
    };

    auto response = ollama_.chat(messages);

    if (response.success && !response.content.empty()) {
        // Parse response into proposal
        std::string interpretation = extract_interpretation(response.content);
        std::string reasoning = extract_reasoning(response.content);
        double confidence = estimate_confidence(response.content);

        proposals.emplace_back(
            ++proposal_counter_,
            active_concepts,
            interpretation,
            reasoning,
            confidence,
            get_model_id()
        );

        // CRITICAL VERIFICATION: Ensure HYPOTHESIS
        if (proposals.back().epistemic_type != EpistemicType::HYPOTHESIS) {
            throw std::logic_error("EPISTEMIC VIOLATION: OllamaMiniLLM proposal not HYPOTHESIS");
        }
    }

    return proposals;
}

// =============================================================================
// HYPOTHESIS GENERATION
// =============================================================================

std::vector<HypothesisProposal> OllamaMiniLLM::generate_hypotheses(
    const std::vector<ConceptId>& evidence_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) const {
    std::vector<HypothesisProposal> proposals;

    if (evidence_concepts.size() < 2) {
        return proposals;
    }

    if (!is_available()) {
        return proposals;
    }

    // Build prompt
    std::ostringstream prompt;
    prompt << "You are a hypothesis generator for a cognitive architecture system.\n\n";
    prompt << "Evidence:\n";
    prompt << build_concept_description(evidence_concepts, ltm);
    prompt << "\nTask: Based on this evidence, suggest ONE testable hypothesis. ";
    prompt << "Format:\n";
    prompt << "HYPOTHESIS: [your hypothesis statement]\n";
    prompt << "REASONING: [why this hypothesis makes sense]\n";
    prompt << "PATTERNS: [detected patterns, comma-separated]\n";

    std::vector<OllamaMessage> messages = {
        {"system", "You are a scientific hypothesis generator. Be analytical and concise."},
        {"user", prompt.str()}
    };

    auto response = ollama_.chat(messages);

    if (response.success && !response.content.empty()) {
        // Simple parsing (in production, use more robust parsing)
        std::string statement = response.content;
        std::string reasoning = "Generated by " + get_model_id();
        std::vector<std::string> patterns = {"llm-detected-pattern"};

        // Try to extract structured parts
        size_t hyp_pos = response.content.find("HYPOTHESIS:");
        size_t reas_pos = response.content.find("REASONING:");
        size_t pat_pos = response.content.find("PATTERNS:");

        if (hyp_pos != std::string::npos && reas_pos != std::string::npos) {
            statement = response.content.substr(hyp_pos + 11, reas_pos - hyp_pos - 11);
            if (pat_pos != std::string::npos) {
                reasoning = response.content.substr(reas_pos + 10, pat_pos - reas_pos - 10);
            } else {
                reasoning = response.content.substr(reas_pos + 10);
            }
        }

        double confidence = estimate_confidence(response.content);

        proposals.emplace_back(
            ++proposal_counter_,
            evidence_concepts,
            statement,
            reasoning,
            patterns,
            confidence,
            get_model_id()
        );

        // CRITICAL VERIFICATION
        if (proposals.back().suggested_epistemic.suggested_type != EpistemicType::HYPOTHESIS) {
            throw std::logic_error("EPISTEMIC VIOLATION: Hypothesis proposal not HYPOTHESIS");
        }
    }

    return proposals;
}

// =============================================================================
// ANALOGY DETECTION
// =============================================================================

std::vector<AnalogyProposal> OllamaMiniLLM::detect_analogies(
    const std::vector<ConceptId>& concept_set_a,
    const std::vector<ConceptId>& concept_set_b,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) const {
    std::vector<AnalogyProposal> proposals;

    if (concept_set_a.empty() || concept_set_b.empty()) {
        return proposals;
    }

    if (!is_available()) {
        return proposals;
    }

    // Build prompt
    std::ostringstream prompt;
    prompt << "You are an analogy detector for a cognitive architecture system.\n\n";
    prompt << "Domain A:\n" << build_concept_description(concept_set_a, ltm) << "\n";
    prompt << "Domain B:\n" << build_concept_description(concept_set_b, ltm) << "\n";
    prompt << "Task: Identify structural similarities between these two domains. ";
    prompt << "What patterns or relationships are analogous? Be brief.\n";

    std::vector<OllamaMessage> messages = {
        {"system", "You are an analogy detection specialist. Focus on structural similarities."},
        {"user", prompt.str()}
    };

    auto response = ollama_.chat(messages);

    if (response.success && !response.content.empty()) {
        double confidence = estimate_confidence(response.content);

        // Simple structural similarity estimate (could be improved)
        double similarity = 0.5 * confidence;

        proposals.emplace_back(
            ++proposal_counter_,
            concept_set_a,
            concept_set_b,
            response.content,
            similarity,
            confidence,
            get_model_id()
        );
    }

    return proposals;
}

// =============================================================================
// CONTRADICTION DETECTION
// =============================================================================

std::vector<ContradictionProposal> OllamaMiniLLM::detect_contradictions(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) const {
    std::vector<ContradictionProposal> proposals;

    if (active_concepts.size() < 2) {
        return proposals;
    }

    if (!is_available()) {
        return proposals;
    }

    // Build prompt
    std::ostringstream prompt;
    prompt << "You are a contradiction detector for a cognitive architecture system.\n\n";
    prompt << build_concept_description(active_concepts, ltm);
    prompt << "\nTask: Identify any potential contradictions or inconsistencies ";
    prompt << "between these concepts. If you find any, describe them briefly. ";
    prompt << "If no contradictions, respond with 'NO_CONTRADICTION'.\n";

    std::vector<OllamaMessage> messages = {
        {"system", "You are a logical consistency checker. Be analytical and precise."},
        {"user", prompt.str()}
    };

    auto response = ollama_.chat(messages);

    if (response.success && !response.content.empty()) {
        // Check if contradiction detected
        if (response.content.find("NO_CONTRADICTION") == std::string::npos &&
            response.content.find("no contradiction") == std::string::npos) {

            // Parse which concepts are contradicting (simplified)
            // In production, would use more sophisticated parsing
            if (active_concepts.size() >= 2) {
                double confidence = estimate_confidence(response.content);
                double severity = confidence * 0.8;  // Severity correlates with confidence

                proposals.emplace_back(
                    ++proposal_counter_,
                    active_concepts[0],
                    active_concepts[1],
                    response.content,
                    "Detected by " + get_model_id(),
                    severity,
                    confidence,
                    get_model_id()
                );
            }
        }
    }

    return proposals;
}

} // namespace brain19

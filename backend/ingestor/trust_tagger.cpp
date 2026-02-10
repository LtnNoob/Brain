#include "trust_tagger.hpp"
#include <algorithm>
#include <cctype>

namespace brain19 {

TrustTagger::TrustTagger() {
}

TrustTagger::TrustRange TrustTagger::get_trust_range(TrustCategory category) const {
    switch (category) {
        case TrustCategory::FACTS:
            return {0.95, 0.99, 0.98};
        case TrustCategory::DEFINITIONS:
            return {0.90, 0.99, 0.95};
        case TrustCategory::THEORIES:
            return {0.85, 0.95, 0.90};
        case TrustCategory::HYPOTHESES:
            return {0.50, 0.80, 0.65};
        case TrustCategory::INFERENCES:
            return {0.40, 0.70, 0.55};
        case TrustCategory::SPECULATION:
            return {0.10, 0.40, 0.30};
        case TrustCategory::INVALIDATED:
            return {0.01, 0.10, 0.05};
    }
    return {0.10, 0.40, 0.30}; // Default to speculation range
}

TrustAssignment TrustTagger::assign_trust(TrustCategory category) const {
    auto range = get_trust_range(category);

    TrustAssignment assignment;
    assignment.category = category;
    assignment.trust_value = range.default_trust;
    assignment.epistemic_status = (category == TrustCategory::INVALIDATED)
        ? EpistemicStatus::INVALIDATED
        : EpistemicStatus::ACTIVE;

    switch (category) {
        case TrustCategory::FACTS:
            assignment.epistemic_type = EpistemicType::FACT;
            assignment.reasoning = "Classified as verified fact (trust " +
                std::to_string(range.default_trust) + ")";
            break;
        case TrustCategory::DEFINITIONS:
            assignment.epistemic_type = EpistemicType::DEFINITION;
            assignment.reasoning = "Classified as definition (trust " +
                std::to_string(range.default_trust) + ")";
            break;
        case TrustCategory::THEORIES:
            assignment.epistemic_type = EpistemicType::THEORY;
            assignment.reasoning = "Classified as well-supported theory (trust " +
                std::to_string(range.default_trust) + ")";
            break;
        case TrustCategory::HYPOTHESES:
            assignment.epistemic_type = EpistemicType::HYPOTHESIS;
            assignment.reasoning = "Classified as testable hypothesis (trust " +
                std::to_string(range.default_trust) + ")";
            break;
        case TrustCategory::INFERENCES:
            assignment.epistemic_type = EpistemicType::INFERENCE;
            assignment.reasoning = "Classified as derived inference (trust " +
                std::to_string(range.default_trust) + ")";
            break;
        case TrustCategory::SPECULATION:
            assignment.epistemic_type = EpistemicType::SPECULATION;
            assignment.reasoning = "Classified as speculation (trust " +
                std::to_string(range.default_trust) + ")";
            break;
        case TrustCategory::INVALIDATED:
            assignment.epistemic_type = EpistemicType::FACT; // Original type preserved
            assignment.reasoning = "Classified as invalidated (trust " +
                std::to_string(range.default_trust) + ")";
            break;
    }

    return assignment;
}

TrustAssignment TrustTagger::assign_trust_with_value(
    TrustCategory category, double trust) const
{
    auto range = get_trust_range(category);
    double clamped = std::clamp(trust, range.min_trust, range.max_trust);

    auto assignment = assign_trust(category);
    assignment.trust_value = clamped;
    assignment.reasoning += " (custom trust: " + std::to_string(clamped) + ")";

    return assignment;
}

TrustAssignment TrustTagger::suggest_from_source(SourceType source) const {
    switch (source) {
        case SourceType::WIKIPEDIA:
            // Wikipedia content is generally well-referenced
            return assign_trust(TrustCategory::THEORIES);

        case SourceType::GOOGLE_SCHOLAR:
            // Peer-reviewed academic content
            return assign_trust(TrustCategory::THEORIES);

        case SourceType::UNKNOWN:
        default:
            // Unknown source → speculation level
            return assign_trust(TrustCategory::SPECULATION);
    }
}

TrustAssignment TrustTagger::suggest_from_text(const std::string& text) const {
    double confidence = compute_text_confidence(text);

    // Map confidence to trust category
    if (has_definition_pattern(text)) {
        // Definitional text gets definition category
        auto assignment = assign_trust(TrustCategory::DEFINITIONS);
        assignment.reasoning = "Text contains definitional patterns";
        return assignment;
    }

    if (confidence >= 0.85 && has_certainty_language(text) && has_citation_markers(text)) {
        auto assignment = assign_trust(TrustCategory::FACTS);
        assignment.reasoning = "Text has high certainty language and citations";
        return assignment;
    }

    if (confidence >= 0.7 && has_citation_markers(text)) {
        auto assignment = assign_trust(TrustCategory::THEORIES);
        assignment.reasoning = "Text is well-supported with citations";
        return assignment;
    }

    if (confidence >= 0.5) {
        auto assignment = assign_trust(TrustCategory::HYPOTHESES);
        assignment.reasoning = "Text has moderate certainty";
        return assignment;
    }

    if (has_hedging_language(text)) {
        auto assignment = assign_trust(TrustCategory::SPECULATION);
        assignment.reasoning = "Text contains hedging language indicating uncertainty";
        return assignment;
    }

    // Default: hypothesis level
    auto assignment = assign_trust(TrustCategory::HYPOTHESES);
    assignment.reasoning = "Default classification for uncharacterized text";
    return assignment;
}

TrustAssignment TrustTagger::suggest_from_proposal(
    SuggestedEpistemicType suggested_type) const
{
    switch (suggested_type) {
        case SuggestedEpistemicType::FACT_CANDIDATE:
            return assign_trust(TrustCategory::THEORIES); // Conservative: fact candidate → theory
        case SuggestedEpistemicType::THEORY_CANDIDATE:
            return assign_trust(TrustCategory::HYPOTHESES); // Conservative: theory → hypothesis
        case SuggestedEpistemicType::HYPOTHESIS_CANDIDATE:
            return assign_trust(TrustCategory::HYPOTHESES);
        case SuggestedEpistemicType::DEFINITION_CANDIDATE:
            return assign_trust(TrustCategory::DEFINITIONS);
        case SuggestedEpistemicType::UNKNOWN_CANDIDATE:
        default:
            return assign_trust(TrustCategory::SPECULATION);
    }
}

bool TrustTagger::has_hedging_language(const std::string& text) const {
    static const std::vector<std::string> hedging_words = {
        "might", "may", "could", "possibly", "perhaps", "potentially",
        "suggests", "appears", "seems", "likely", "unlikely", "probable",
        "uncertain", "debatable", "controversial", "hypothetical",
        "speculative", "preliminary", "tentative", "approximate"
    };

    // Convert to lowercase for matching
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    for (const auto& word : hedging_words) {
        if (lower.find(word) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool TrustTagger::has_certainty_language(const std::string& text) const {
    static const std::vector<std::string> certainty_words = {
        "proven", "established", "confirmed", "demonstrated", "verified",
        "known", "definite", "certain", "conclusive", "undeniable",
        "evidence shows", "studies confirm", "research demonstrates"
    };

    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    for (const auto& word : certainty_words) {
        if (lower.find(word) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool TrustTagger::has_definition_pattern(const std::string& text) const {
    static const std::vector<std::string> def_patterns = {
        " is defined as ", " refers to ", " is the study of ",
        " is a branch of ", " describes the ", " is the process of ",
        " is a type of ", " is a form of "
    };

    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    for (const auto& pattern : def_patterns) {
        if (lower.find(pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool TrustTagger::has_citation_markers(const std::string& text) const {
    static const std::vector<std::string> citation_markers = {
        "[1]", "[2]", "[3]", "(et al.", "et al.,", "doi:", "DOI:",
        "(20", "(19", "pp.", "vol.", "journal", "proceedings"
    };

    for (const auto& marker : citation_markers) {
        if (text.find(marker) != std::string::npos) {
            return true;
        }
    }
    return false;
}

double TrustTagger::compute_text_confidence(const std::string& text) const {
    double score = 0.5; // Start at neutral

    if (has_certainty_language(text)) score += 0.2;
    if (has_citation_markers(text)) score += 0.15;
    if (has_definition_pattern(text)) score += 0.1;
    if (has_hedging_language(text)) score -= 0.25;

    // Length factor: very short text → less confidence
    if (text.size() < 50) score -= 0.1;
    if (text.size() > 200) score += 0.05;

    return std::clamp(score, 0.0, 1.0);
}

std::string TrustTagger::category_to_string(TrustCategory cat) {
    switch (cat) {
        case TrustCategory::FACTS: return "FACTS";
        case TrustCategory::DEFINITIONS: return "DEFINITIONS";
        case TrustCategory::THEORIES: return "THEORIES";
        case TrustCategory::HYPOTHESES: return "HYPOTHESES";
        case TrustCategory::INFERENCES: return "INFERENCES";
        case TrustCategory::SPECULATION: return "SPECULATION";
        case TrustCategory::INVALIDATED: return "INVALIDATED";
    }
    return "UNKNOWN";
}

} // namespace brain19

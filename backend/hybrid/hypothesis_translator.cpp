#include "hypothesis_translator.hpp"
#include <cmath>
#include <algorithm>
#include <cctype>

namespace brain19 {

HypothesisTranslator::HypothesisTranslator(Config config)
    : config_(std::move(config))
{}

std::string HypothesisTranslator::to_lower(const std::string& text) const {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

bool HypothesisTranslator::contains_any(
    const std::string& text,
    const std::vector<std::string>& keywords
) const {
    std::string lower = to_lower(text);
    for (const auto& kw : keywords) {
        if (lower.find(kw) != std::string::npos) return true;
    }
    return false;
}

RelationshipPattern HypothesisTranslator::detect_pattern(const std::string& hypothesis_text) const {
    std::string text = to_lower(hypothesis_text);

    // Check patterns in order of specificity (most specific first)

    // PERIODIC
    if (contains_any(text, {"periodic", "oscillat", "cycl", "wave", "sinusoid",
                            "rhythm", "fluctuat", "harmonic"})) {
        return RelationshipPattern::PERIODIC;
    }

    // EXPONENTIAL
    if (contains_any(text, {"exponential", "exp growth", "doubles", "halves",
                            "logarithm", "decay", "geometric"})) {
        return RelationshipPattern::EXPONENTIAL;
    }

    // THRESHOLD
    if (contains_any(text, {"threshold", "trigger", "activat", "switch",
                            "step function", "binary", "on/off", "exceeds"})) {
        return RelationshipPattern::THRESHOLD;
    }

    // POLYNOMIAL
    if (contains_any(text, {"quadratic", "polynomial", "squared", "cubic",
                            "parabol", "power law", "nonlinear"})) {
        return RelationshipPattern::POLYNOMIAL;
    }

    // LINEAR
    if (contains_any(text, {"linear", "proportional", "correlat", "increases with",
                            "decreases with", "ratio", "scales with", "directly"})) {
        return RelationshipPattern::LINEAR;
    }

    // If we found any numeric/quantitative indicators, default to LINEAR
    if (contains_any(text, {"increases", "decreases", "grows", "shrinks",
                            "more", "less", "higher", "lower", "rate"})) {
        return RelationshipPattern::LINEAR;
    }

    return RelationshipPattern::NOT_QUANTIFIABLE;
}

TranslationResult HypothesisTranslator::translate(const HypothesisProposal& proposal) const {
    // Combine hypothesis statement and patterns for analysis
    std::string combined_text = proposal.hypothesis_statement + " " + proposal.supporting_reasoning;
    for (const auto& p : proposal.detected_patterns) {
        combined_text += " " + p;
    }

    RelationshipPattern pattern = detect_pattern(combined_text);

    if (pattern == RelationshipPattern::NOT_QUANTIFIABLE) {
        return TranslationResult::not_quantifiable(
            "Hypothesis does not describe a quantifiable numeric relationship: '"
            + proposal.hypothesis_statement + "'"
        );
    }

    // Generate training data
    auto data = generate_training_data(
        pattern,
        config_.max_data_points,
        config_.data_range_min,
        config_.data_range_max
    );

    if (data.size() < config_.min_data_points) {
        return TranslationResult::not_quantifiable(
            "Insufficient training data generated for pattern: "
            + std::string(pattern_to_string(pattern))
        );
    }

    auto topology = suggest_topology(pattern);
    auto config = suggest_config(pattern);

    KanTrainingProblem problem(
        proposal.proposal_id,
        pattern,
        1,  // input_dim (univariate for now)
        1,  // output_dim
        std::move(data),
        std::move(topology),
        config,
        "Pattern: " + std::string(pattern_to_string(pattern))
        + " | Hypothesis: " + proposal.hypothesis_statement
    );

    TranslationResult result;
    result.translatable = true;
    result.detected_pattern = pattern;
    result.problem = std::move(problem);
    result.explanation = "Successfully translated to " + std::string(pattern_to_string(pattern))
                       + " KAN training problem";
    return result;
}

// =============================================================================
// SYNTHETIC DATA GENERATORS
// =============================================================================

std::vector<DataPoint> HypothesisTranslator::generate_training_data(
    RelationshipPattern pattern,
    size_t num_points,
    double range_min,
    double range_max
) const {
    switch (pattern) {
        case RelationshipPattern::LINEAR:
            return generate_linear_data(num_points, range_min, range_max);
        case RelationshipPattern::POLYNOMIAL:
            return generate_polynomial_data(num_points, range_min, range_max);
        case RelationshipPattern::EXPONENTIAL:
            return generate_exponential_data(num_points, range_min, range_max);
        case RelationshipPattern::PERIODIC:
            return generate_periodic_data(num_points, range_min, range_max);
        case RelationshipPattern::THRESHOLD:
            return generate_threshold_data(num_points, range_min, range_max);
        case RelationshipPattern::NOT_QUANTIFIABLE:
            return {};
    }
    return {};
}

std::vector<DataPoint> HypothesisTranslator::generate_linear_data(
    size_t n, double min, double max
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    double step = (max - min) / static_cast<double>(n - 1);
    // y = 0.7x + 0.1 (canonical linear)
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = 0.7 * x + 0.1;
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_polynomial_data(
    size_t n, double min, double max
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    double step = (max - min) / static_cast<double>(n - 1);
    // y = x^2 (canonical quadratic)
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = x * x;
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_exponential_data(
    size_t n, double min, double max
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    double step = (max - min) / static_cast<double>(n - 1);
    // y = e^(2x) / e^(2*max) — normalized to [0,1] range
    double norm = std::exp(2.0 * max);
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = std::exp(2.0 * x) / norm;
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_periodic_data(
    size_t n, double min, double max
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    double step = (max - min) / static_cast<double>(n - 1);
    // y = 0.5 * sin(2π * x) + 0.5 — normalized to [0,1]
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = 0.5 * std::sin(2.0 * M_PI * x) + 0.5;
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_threshold_data(
    size_t n, double min, double max
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    double step = (max - min) / static_cast<double>(n - 1);
    double midpoint = (min + max) / 2.0;
    // Sigmoid: y = 1 / (1 + e^(-20*(x - mid)))
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = 1.0 / (1.0 + std::exp(-20.0 * (x - midpoint)));
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

// =============================================================================
// TOPOLOGY & CONFIG SUGGESTIONS
// =============================================================================

std::vector<size_t> HypothesisTranslator::suggest_topology(RelationshipPattern pattern) const {
    size_t hidden = config_.default_hidden_dim;
    switch (pattern) {
        case RelationshipPattern::LINEAR:
            return {1, 1};  // Direct mapping, no hidden layer needed
        case RelationshipPattern::POLYNOMIAL:
            return {1, hidden, 1};
        case RelationshipPattern::EXPONENTIAL:
            return {1, hidden, 1};
        case RelationshipPattern::PERIODIC:
            return {1, hidden * 2, hidden, 1};  // More capacity for oscillation
        case RelationshipPattern::THRESHOLD:
            return {1, hidden, 1};
        default:
            return {1, hidden, 1};
    }
}

KanTrainingConfig HypothesisTranslator::suggest_config(RelationshipPattern pattern) const {
    KanTrainingConfig config;
    config.learning_rate = 0.01;
    config.convergence_threshold = 1e-6;

    switch (pattern) {
        case RelationshipPattern::LINEAR:
            config.max_iterations = 500;
            break;
        case RelationshipPattern::POLYNOMIAL:
            config.max_iterations = 1000;
            break;
        case RelationshipPattern::EXPONENTIAL:
            config.max_iterations = 1500;
            config.learning_rate = 0.005;
            break;
        case RelationshipPattern::PERIODIC:
            config.max_iterations = 2000;
            config.learning_rate = 0.005;
            break;
        case RelationshipPattern::THRESHOLD:
            config.max_iterations = 1000;
            break;
        default:
            config.max_iterations = 1000;
            break;
    }
    return config;
}

} // namespace brain19

#include "hypothesis_translator.hpp"
#include <cmath>
#include <numbers>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <regex>
#include <random>
#include <set>

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

// =============================================================================
// C1: NLP-LITE HELPERS
// =============================================================================

std::vector<std::string> HypothesisTranslator::split_sentences(const std::string& text) const {
    std::vector<std::string> sentences;
    std::string current;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        current += c;
        // Split on sentence terminators or conjunctions that separate clauses
        if (c == '.' || c == ';' || c == '!' || c == '?') {
            auto trimmed = current;
            // trim whitespace
            while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front()))) trimmed.erase(trimmed.begin());
            while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) trimmed.pop_back();
            if (!trimmed.empty()) sentences.push_back(std::move(trimmed));
            current.clear();
        }
    }
    // Remaining text
    while (!current.empty() && std::isspace(static_cast<unsigned char>(current.front()))) current.erase(current.begin());
    while (!current.empty() && std::isspace(static_cast<unsigned char>(current.back()))) current.pop_back();
    if (!current.empty()) sentences.push_back(std::move(current));

    // Also split on " but ", " however ", " and " when they separate independent clauses
    std::vector<std::string> result;
    for (const auto& sent : sentences) {
        std::string lower = to_lower(sent);
        // Split on clause-separating conjunctions
        bool split = false;
        for (const auto& conj : {", but ", ", however ", " whereas "}) {
            auto pos = lower.find(conj);
            if (pos != std::string::npos) {
                result.push_back(sent.substr(0, pos));
                result.push_back(sent.substr(pos + std::string(conj).size()));
                split = true;
                break;
            }
        }
        if (!split) result.push_back(sent);
    }
    return result;
}

bool HypothesisTranslator::is_negated(const std::string& text, const std::string& keyword) const {
    std::string lower = to_lower(text);
    auto pos = lower.find(keyword);
    if (pos == std::string::npos) return false;

    // Check for negation words within 3 words before the keyword
    std::vector<std::string> negators = {"not ", "no ", "non-", "non ", "never ", "without ", "isn't ", "isn't ", "not_"};
    // Look back up to 30 chars for a negation
    size_t look_start = (pos > 30) ? pos - 30 : 0;
    std::string prefix = lower.substr(look_start, pos - look_start);

    for (const auto& neg : negators) {
        if (prefix.find(neg) != std::string::npos) return true;
    }
    return false;
}

size_t HypothesisTranslator::count_variables(const std::string& text) const {
    std::string lower = to_lower(text);
    // Look for "X and Y", "X, Y, and Z", "relates to ... and ..."
    // Count variable-like references
    std::regex var_pattern(R"(\b([a-z])\b)");  // single letters as variables
    std::set<char> vars;
    auto begin = std::sregex_iterator(lower.begin(), lower.end(), var_pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        char v = (*it)[1].str()[0];
        // FIX NEW-2: Skip common non-variable single letters
        if (v != 'a' && v != 'i' && v != 's' && v != 't' && v != 'o') {
            vars.insert(v);
        }
    }
    if (vars.size() >= 2) return vars.size();

    // Also check for multi-word patterns: "X depends on Y and Z"
    if (lower.find(" and ") != std::string::npos &&
        (lower.find("depends on") != std::string::npos ||
         lower.find("relates to") != std::string::npos ||
         lower.find("function of") != std::string::npos)) {
        return 2; // At minimum multivariate
    }

    return 1; // Default univariate
}

double HypothesisTranslator::detect_quantifier_modifier(const std::string& text) const {
    std::string lower = to_lower(text);

    // Strong hedging → lower confidence
    if (contains_any(lower, {"rarely", "seldom", "unlikely", "barely"})) return 0.3;
    if (contains_any(lower, {"sometimes", "occasionally", "might", "could", "possibly"})) return 0.5;
    if (contains_any(lower, {"often", "usually", "typically", "generally", "tends to"})) return 0.8;
    if (contains_any(lower, {"always", "certainly", "definitely", "must", "invariably"})) return 1.0;

    return 0.9; // Default: fairly confident (no hedge)
}

// =============================================================================
// C1: ROBUST PATTERN DETECTION WITH CONFIDENCE
// =============================================================================

PatternDetectionResult HypothesisTranslator::detect_pattern_detailed(const std::string& hypothesis_text) const {
    auto sentences = split_sentences(hypothesis_text);
    if (sentences.empty()) sentences.push_back(hypothesis_text);

    // Accumulate pattern scores across all clauses
    struct PatternScore {
        double score = 0.0;
        bool negated = false;
    };
    std::unordered_map<int, PatternScore> scores; // int = RelationshipPattern

    auto add_score = [&](RelationshipPattern p, double s, bool neg) {
        auto& ps = scores[static_cast<int>(p)];
        if (neg) {
            ps.negated = true;
            ps.score -= s * 0.5; // Negation reduces score
        } else {
            ps.score += s;
        }
    };

    for (const auto& clause : sentences) {
        std::string text = to_lower(clause);

        // --- PERIODIC ---
        struct KeywordGroup { std::string keyword; double weight; };
        std::vector<KeywordGroup> periodic_kw = {
            {"periodic", 0.9}, {"oscillat", 0.9}, {"cycl", 0.8}, {"wave", 0.8},
            {"sinusoid", 0.95}, {"rhythm", 0.7}, {"fluctuat", 0.6}, {"harmonic", 0.9}
        };
        for (const auto& [kw, w] : periodic_kw) {
            if (text.find(kw) != std::string::npos) {
                bool neg = is_negated(text, kw);
                add_score(RelationshipPattern::PERIODIC, w, neg);
            }
        }

        // --- EXPONENTIAL ---
        std::vector<KeywordGroup> exp_kw = {
            {"exponential", 0.95}, {"exp growth", 0.9}, {"doubles", 0.7}, {"halves", 0.7},
            {"logarithm", 0.8}, {"decay", 0.7}, {"geometric", 0.6}, {"exp(", 0.95}
        };
        for (const auto& [kw, w] : exp_kw) {
            if (text.find(kw) != std::string::npos) {
                bool neg = is_negated(text, kw);
                add_score(RelationshipPattern::EXPONENTIAL, w, neg);
            }
        }

        // --- THRESHOLD ---
        std::vector<KeywordGroup> thresh_kw = {
            {"threshold", 0.9}, {"trigger", 0.8}, {"activat", 0.8}, {"switch", 0.7},
            {"step function", 0.95}, {"binary", 0.6}, {"on/off", 0.8}, {"exceeds", 0.7}
        };
        for (const auto& [kw, w] : thresh_kw) {
            if (text.find(kw) != std::string::npos) {
                bool neg = is_negated(text, kw);
                add_score(RelationshipPattern::THRESHOLD, w, neg);
            }
        }

        // --- POLYNOMIAL ---
        std::vector<KeywordGroup> poly_kw = {
            {"quadratic", 0.9}, {"polynomial", 0.95}, {"squared", 0.85}, {"cubic", 0.85},
            {"parabol", 0.85}, {"power law", 0.8}, {"nonlinear", 0.5},
            {"inverted-u", 0.85}, {"inverted u", 0.85}, {"u-shaped", 0.85}, {"u shaped", 0.85},
            {"bell curve", 0.8}, {"bell-curve", 0.8}, {"diminishing returns", 0.7},
            {"peaks at", 0.7}, {"optimal at", 0.7}
        };
        for (const auto& [kw, w] : poly_kw) {
            if (text.find(kw) != std::string::npos) {
                bool neg = is_negated(text, kw);
                add_score(RelationshipPattern::POLYNOMIAL, w, neg);
            }
        }

        // --- CONDITIONAL (new) ---
        // "X increases when Y", "X depends on whether Y"
        std::vector<KeywordGroup> cond_kw = {
            {"when", 0.5}, {"if ", 0.6}, {"depends on whether", 0.8},
            {"only if", 0.8}, {"given that", 0.7}, {"provided", 0.5},
            {"conditional", 0.9}, {"contingent", 0.7}
        };
        bool has_conditional = false;
        for (const auto& [kw, w] : cond_kw) {
            if (text.find(kw) != std::string::npos) {
                // "when" alone is weak; needs a quantitative word too
                if (kw == "when" || kw == "if ") {
                    if (contains_any(text, {"increases", "decreases", "grows", "changes", "rises", "drops"})) {
                        add_score(RelationshipPattern::CONDITIONAL, w + 0.2, false);
                        has_conditional = true;
                    }
                } else {
                    add_score(RelationshipPattern::CONDITIONAL, w, false);
                    has_conditional = true;
                }
            }
        }

        // --- LINEAR ---
        std::vector<KeywordGroup> lin_kw = {
            {"linear", 0.9}, {"proportional", 0.85}, {"correlat", 0.6},
            {"increases with", 0.7}, {"decreases with", 0.7}, {"ratio", 0.5},
            {"scales with", 0.8}, {"directly", 0.4}
        };
        for (const auto& [kw, w] : lin_kw) {
            if (text.find(kw) != std::string::npos) {
                bool neg = is_negated(text, kw);
                add_score(RelationshipPattern::LINEAR, w, neg);
            }
        }

        // Weak quantitative indicators → low-confidence LINEAR
        std::vector<KeywordGroup> weak_kw = {
            {"increases", 0.3}, {"decreases", 0.3}, {"grows", 0.3}, {"shrinks", 0.3},
            {"more", 0.2}, {"less", 0.2}, {"higher", 0.25}, {"lower", 0.25}, {"rate", 0.3}
        };
        for (const auto& [kw, w] : weak_kw) {
            if (text.find(kw) != std::string::npos && !has_conditional) {
                add_score(RelationshipPattern::LINEAR, w, false);
            }
        }
    }

    // Find best pattern
    RelationshipPattern best_pattern = RelationshipPattern::NOT_QUANTIFIABLE;
    double best_score = 0.0;
    bool best_negated = false;

    for (const auto& [pat_int, ps] : scores) {
        if (ps.score > best_score) {
            best_score = ps.score;
            best_pattern = static_cast<RelationshipPattern>(pat_int);
            best_negated = ps.negated;
        }
    }

    // Apply quantifier modifier
    double quant_mod = detect_quantifier_modifier(hypothesis_text);

    // Compute final confidence
    double confidence = std::min(1.0, best_score) * quant_mod;

    // If negated, the pattern was explicitly denied
    if (best_negated) {
        confidence = 0.0;
        best_pattern = RelationshipPattern::NOT_QUANTIFIABLE;
    }

    // Below threshold → NOT_QUANTIFIABLE
    if (confidence < config_.confidence_threshold) {
        best_pattern = RelationshipPattern::NOT_QUANTIFIABLE;
    }

    size_t num_vars = count_variables(hypothesis_text);

    std::string expl = "Detected " + std::string(pattern_to_string(best_pattern))
                     + " (confidence=" + std::to_string(confidence)
                     + ", variables=" + std::to_string(num_vars)
                     + ", quantifier_mod=" + std::to_string(quant_mod) + ")";

    return PatternDetectionResult{
        best_pattern,
        confidence,
        best_negated,
        num_vars,
        quant_mod,
        std::move(expl)
    };
}

RelationshipPattern HypothesisTranslator::detect_pattern(const std::string& hypothesis_text) const {
    auto result = detect_pattern_detailed(hypothesis_text);
    return result.pattern;
}

// =============================================================================
// H1: NUMERIC HINT EXTRACTION
// =============================================================================

NumericHints HypothesisTranslator::extract_numeric_hints(const std::string& text) const {
    NumericHints hints;

    // Extract all numbers (integers and decimals) from text
    std::regex num_pattern(R"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), num_pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        try {
            double val = std::stod(it->str());
            hints.numbers.push_back(val);
        } catch (...) {}
    }

    std::string lower = to_lower(text);

    // Try to extract slope hints: "slope of 2.5", "rate of 3", "factor 0.5"
    std::regex slope_pattern(R"((?:slope|rate|factor|coefficient)\s*(?:of|=|:)?\s*([-+]?\d+\.?\d*))");
    std::smatch slope_match;
    if (std::regex_search(lower, slope_match, slope_pattern)) {
        try { hints.slope = std::stod(slope_match[1].str()); } catch (...) {}
    }

    // Try to extract range: "between X and Y", "from X to Y", "range X-Y"
    std::regex range_pattern(R"((?:between|from|range)\s*([-+]?\d+\.?\d*)\s*(?:and|to|-)\s*([-+]?\d+\.?\d*))");
    std::smatch range_match;
    if (std::regex_search(lower, range_match, range_pattern)) {
        try {
            hints.range_min = std::stod(range_match[1].str());
            hints.range_max = std::stod(range_match[2].str());
        } catch (...) {}
    }

    // Scale hints: "around 1000", "magnitude of 100"
    std::regex scale_pattern(R"((?:around|approximately|magnitude|scale)\s*(?:of)?\s*([-+]?\d+\.?\d*))");
    std::smatch scale_match;
    if (std::regex_search(lower, scale_match, scale_pattern)) {
        try { hints.scale_factor = std::stod(scale_match[1].str()); } catch (...) {}
    }

    return hints;
}

// =============================================================================
// TRANSLATE
// =============================================================================

TranslationResult HypothesisTranslator::translate(const HypothesisProposal& proposal) const {
    std::string combined_text = proposal.hypothesis_statement + " " + proposal.supporting_reasoning;
    for (const auto& p : proposal.detected_patterns) {
        combined_text += " " + p;
    }

    auto detection = detect_pattern_detailed(combined_text);

    if (detection.pattern == RelationshipPattern::NOT_QUANTIFIABLE) {
        return TranslationResult::not_quantifiable(
            "Hypothesis does not describe a quantifiable numeric relationship (confidence="
            + std::to_string(detection.confidence) + "): '"
            + proposal.hypothesis_statement + "'"
        );
    }

    // H1: Extract numeric hints for hypothesis-specific data
    NumericHints hints = extract_numeric_hints(combined_text);

    // Determine data quality
    DataQuality quality = hints.has_hints()
        ? DataQuality::SYNTHETIC_SPECIFIC
        : DataQuality::SYNTHETIC_CANONICAL;

    // Use hint-based range if available
    double range_min = hints.range_min.value_or(config_.data_range_min);
    double range_max = hints.range_max.value_or(config_.data_range_max);

    auto data = generate_training_data(
        detection.pattern,
        config_.max_data_points,
        range_min,
        range_max,
        hints
    );

    if (data.size() < config_.min_data_points) {
        return TranslationResult::not_quantifiable(
            "Insufficient training data generated for pattern: "
            + std::string(pattern_to_string(detection.pattern))
        );
    }

    // C1: Determine input_dim from detected variables
    size_t input_dim = std::min(detection.detected_variables, static_cast<size_t>(1));
    // Note: multivariate support (>1) requires KAN topology changes;
    // for now cap at 1 but record detected_variables for future use
    size_t output_dim = 1;

    auto topology = suggest_topology(detection.pattern, input_dim);
    auto config = suggest_config(detection.pattern);

    KanTrainingProblem problem(
        proposal.proposal_id,
        detection.pattern,
        input_dim,
        output_dim,
        std::move(data),
        std::move(topology),
        config,
        "Pattern: " + std::string(pattern_to_string(detection.pattern))
        + " (confidence=" + std::to_string(detection.confidence) + ")"
        + " | Hypothesis: " + proposal.hypothesis_statement,
        quality,
        detection.confidence
    );

    TranslationResult result;
    result.translatable = true;
    result.detected_pattern = detection.pattern;
    result.pattern_confidence = detection.confidence;
    result.problem = std::move(problem);
    result.explanation = "Successfully translated to " + std::string(pattern_to_string(detection.pattern))
                       + " (confidence=" + std::to_string(detection.confidence) + ")";
    return result;
}

// =============================================================================
// H1: HYPOTHESIS-SPECIFIC DATA GENERATORS
// =============================================================================

std::vector<DataPoint> HypothesisTranslator::generate_training_data(
    RelationshipPattern pattern,
    size_t num_points,
    double range_min,
    double range_max,
    const NumericHints& hints
) const {
    switch (pattern) {
        case RelationshipPattern::LINEAR:
            return generate_linear_data(num_points, range_min, range_max, hints);
        case RelationshipPattern::POLYNOMIAL:
            return generate_polynomial_data(num_points, range_min, range_max, hints);
        case RelationshipPattern::EXPONENTIAL:
            return generate_exponential_data(num_points, range_min, range_max, hints);
        case RelationshipPattern::PERIODIC:
            return generate_periodic_data(num_points, range_min, range_max, hints);
        case RelationshipPattern::THRESHOLD:
            return generate_threshold_data(num_points, range_min, range_max, hints);
        case RelationshipPattern::CONDITIONAL:
            return generate_conditional_data(num_points, range_min, range_max, hints);
        case RelationshipPattern::NOT_QUANTIFIABLE:
            return {};
    }
    return {};
}

std::vector<DataPoint> HypothesisTranslator::generate_linear_data(
    size_t n, double min, double max, const NumericHints& hints
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    if (n < 2) return data;  // Guard: n=1 → division by zero
    double step = (max - min) / static_cast<double>(n - 1);

    if (!hints.has_hints()) {
        // NEW-5 FIX: Use a single linear function with slight noise instead of
        // piecewise-linear blocks that create discontinuities at block boundaries.
        // This is semantically "linear" and learnable by a KAN.
        double s = 0.7, b = 0.1;
        std::mt19937 rng(42);  // deterministic seed for reproducibility
        std::normal_distribution<double> noise(0.0, 0.02);
        for (size_t i = 0; i < n; ++i) {
            double x = min + step * static_cast<double>(i);
            double y = s * x + b + noise(rng);
            data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
        }
    } else {
        // H1: Hypothesis-specific
        double s = hints.slope.value_or(1.0);
        double b = hints.intercept.value_or(0.0);
        if (hints.scale_factor.has_value()) s *= hints.scale_factor.value();
        for (size_t i = 0; i < n; ++i) {
            double x = min + step * static_cast<double>(i);
            double y = s * x + b;
            data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
        }
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_polynomial_data(
    size_t n, double min, double max, const NumericHints& hints
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    if (n < 2) return data;
    double step = (max - min) / static_cast<double>(n - 1);

    double scale = hints.scale_factor.value_or(1.0);
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = scale * x * x;
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_exponential_data(
    size_t n, double min, double max, const NumericHints& hints
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    if (n < 2) return data;
    double step = (max - min) / static_cast<double>(n - 1);

    double rate = hints.slope.value_or(2.0);
    double norm = std::exp(rate * max);
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = std::exp(rate * x) / norm;
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_periodic_data(
    size_t n, double min, double max, const NumericHints& hints
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    if (n < 2) return data;
    double step = (max - min) / static_cast<double>(n - 1);

    double freq = hints.scale_factor.value_or(1.0);
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = 0.5 * std::sin(2.0 * std::numbers::pi * freq * x) + 0.5;
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_threshold_data(
    size_t n, double min, double max, const NumericHints& hints
) const {
    std::vector<DataPoint> data;
    data.reserve(n);
    if (n < 2) return data;
    double step = (max - min) / static_cast<double>(n - 1);

    // Use extracted threshold if available, otherwise midpoint
    double threshold = (min + max) / 2.0;
    if (!hints.numbers.empty()) {
        // Use the first number as threshold if it's within range
        for (double num : hints.numbers) {
            if (num >= min && num <= max) {
                threshold = num;
                break;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = 1.0 / (1.0 + std::exp(-20.0 * (x - threshold)));
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

std::vector<DataPoint> HypothesisTranslator::generate_conditional_data(
    size_t n, double min, double max, const NumericHints& hints
) const {
    // Conditional: piecewise — flat below midpoint, linear above
    std::vector<DataPoint> data;
    data.reserve(n);
    if (n < 2) return data;
    double step = (max - min) / static_cast<double>(n - 1);
    double midpoint = (min + max) / 2.0;

    if (!hints.numbers.empty()) {
        for (double num : hints.numbers) {
            if (num >= min && num <= max) { midpoint = num; break; }
        }
    }

    double slope = hints.slope.value_or(1.0);
    for (size_t i = 0; i < n; ++i) {
        double x = min + step * static_cast<double>(i);
        double y = (x < midpoint) ? 0.0 : slope * (x - midpoint);
        data.emplace_back(std::vector<double>{x}, std::vector<double>{y});
    }
    return data;
}

// =============================================================================
// TOPOLOGY & CONFIG SUGGESTIONS
// =============================================================================

std::vector<size_t> HypothesisTranslator::suggest_topology(RelationshipPattern pattern, size_t input_dim) const {
    size_t hidden = config_.default_hidden_dim;
    size_t in = std::max(input_dim, static_cast<size_t>(1));
    switch (pattern) {
        case RelationshipPattern::LINEAR:
            return {in, 1};
        case RelationshipPattern::POLYNOMIAL:
            return {in, hidden, 1};
        case RelationshipPattern::EXPONENTIAL:
            return {in, hidden, 1};
        case RelationshipPattern::PERIODIC:
            return {in, hidden * 2, hidden, 1};
        case RelationshipPattern::THRESHOLD:
            return {in, hidden, 1};
        case RelationshipPattern::CONDITIONAL:
            return {in, hidden, hidden, 1};
        default:
            return {in, hidden, 1};
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
        case RelationshipPattern::CONDITIONAL:
            config.max_iterations = 1500;
            config.learning_rate = 0.005;
            break;
        default:
            config.max_iterations = 1000;
            break;
    }
    return config;
}

} // namespace brain19

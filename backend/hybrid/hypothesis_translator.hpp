#pragma once

#include "../understanding/understanding_proposals.hpp"
#include "../kan/kan_module.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <regex>

namespace brain19 {

// =============================================================================
// RELATIONSHIP PATTERN
// =============================================================================

enum class RelationshipPattern {
    LINEAR,          // y = ax + b
    POLYNOMIAL,      // y = ax^2 + bx + c (or higher)
    EXPONENTIAL,     // y = a * e^(bx)
    PERIODIC,        // y = a * sin(bx + c)
    THRESHOLD,       // y = 0 if x < t, else 1 (sigmoid approx)
    CONDITIONAL,     // X increases when Y (conditional dependency)
    NOT_QUANTIFIABLE // Cannot be translated to numeric relationship
};

inline const char* pattern_to_string(RelationshipPattern p) {
    switch (p) {
        case RelationshipPattern::LINEAR: return "LINEAR";
        case RelationshipPattern::POLYNOMIAL: return "POLYNOMIAL";
        case RelationshipPattern::EXPONENTIAL: return "EXPONENTIAL";
        case RelationshipPattern::PERIODIC: return "PERIODIC";
        case RelationshipPattern::THRESHOLD: return "THRESHOLD";
        case RelationshipPattern::CONDITIONAL: return "CONDITIONAL";
        case RelationshipPattern::NOT_QUANTIFIABLE: return "NOT_QUANTIFIABLE";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// DATA QUALITY — tracks provenance of training data
// =============================================================================

enum class DataQuality {
    SYNTHETIC_CANONICAL,  // Generic canonical function (y=0.7x+0.1)
    SYNTHETIC_SPECIFIC,   // Hypothesis-specific parameters extracted
    EXTRACTED             // Real data extracted from LTM/external source
};

inline const char* data_quality_to_string(DataQuality q) {
    switch (q) {
        case DataQuality::SYNTHETIC_CANONICAL: return "SYNTHETIC_CANONICAL";
        case DataQuality::SYNTHETIC_SPECIFIC: return "SYNTHETIC_SPECIFIC";
        case DataQuality::EXTRACTED: return "EXTRACTED";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// PATTERN DETECTION RESULT (C1: confidence-scored detection)
// =============================================================================

struct PatternDetectionResult {
    RelationshipPattern pattern;
    double confidence;           // [0.0, 1.0] — how sure are we about this pattern
    bool negated;                // "not exponential" → true
    size_t detected_variables;   // number of variables detected (1=univariate, 2+=multivariate)
    double quantifier_modifier;  // 1.0 = certain, <1.0 = hedged ("sometimes", "usually")
    std::string explanation;

    bool is_reliable(double threshold = 0.3) const {
        return confidence >= threshold && !negated;
    }
};

// =============================================================================
// EXTRACTED NUMERIC HINTS (H1: hypothesis-specific data)
// =============================================================================

struct NumericHints {
    std::vector<double> numbers;          // All numbers found in text
    std::optional<double> slope;          // Extracted slope hint
    std::optional<double> intercept;      // Extracted intercept hint
    std::optional<double> range_min;      // Extracted range minimum
    std::optional<double> range_max;      // Extracted range maximum
    std::optional<double> scale_factor;   // Extracted scale/magnitude
    bool has_hints() const { return !numbers.empty(); }
};

// =============================================================================
// KAN TRAINING PROBLEM
// =============================================================================

struct KanTrainingProblem {
    uint64_t hypothesis_id;
    RelationshipPattern pattern;
    size_t input_dim;
    size_t output_dim;
    std::vector<DataPoint> training_data;
    std::vector<size_t> suggested_topology;
    KanTrainingConfig suggested_config;
    std::string relationship_description;
    DataQuality data_quality;
    double pattern_confidence;

    KanTrainingProblem() = delete;

    KanTrainingProblem(
        uint64_t hyp_id,
        RelationshipPattern pat,
        size_t in_dim,
        size_t out_dim,
        std::vector<DataPoint> data,
        std::vector<size_t> topology,
        KanTrainingConfig config,
        std::string desc,
        DataQuality quality = DataQuality::SYNTHETIC_CANONICAL,
        double confidence = 1.0
    ) : hypothesis_id(hyp_id)
      , pattern(pat)
      , input_dim(in_dim)
      , output_dim(out_dim)
      , training_data(std::move(data))
      , suggested_topology(std::move(topology))
      , suggested_config(config)
      , relationship_description(std::move(desc))
      , data_quality(quality)
      , pattern_confidence(confidence)
    {}
};

// =============================================================================
// TRANSLATION RESULT
// =============================================================================

struct TranslationResult {
    bool translatable;
    std::optional<KanTrainingProblem> problem;
    RelationshipPattern detected_pattern;
    double pattern_confidence;
    std::string explanation;

    static TranslationResult not_quantifiable(const std::string& reason) {
        TranslationResult r;
        r.translatable = false;
        r.problem = std::nullopt;
        r.detected_pattern = RelationshipPattern::NOT_QUANTIFIABLE;
        r.pattern_confidence = 0.0;
        r.explanation = reason;
        return r;
    }

private:
    TranslationResult() : translatable(false), detected_pattern(RelationshipPattern::NOT_QUANTIFIABLE), pattern_confidence(0.0) {}
    friend class HypothesisTranslator;
};

// =============================================================================
// HYPOTHESIS TRANSLATOR
// =============================================================================

class HypothesisTranslator {
public:
    struct Config {
        size_t min_data_points = 20;
        size_t max_data_points = 100;
        size_t default_num_knots = 10;
        double data_range_min = 0.0;
        double data_range_max = 1.0;
        size_t default_hidden_dim = 5;
        double confidence_threshold = 0.3;  // Below this → NOT_QUANTIFIABLE
    };

    HypothesisTranslator() : HypothesisTranslator(Config{}) {}
    explicit HypothesisTranslator(Config config);

    TranslationResult translate(const HypothesisProposal& proposal) const;

    // C1: Returns simple pattern (backward compat) — uses detect_pattern_detailed internally
    RelationshipPattern detect_pattern(const std::string& hypothesis_text) const;

    // C1: Full detailed pattern detection with confidence
    PatternDetectionResult detect_pattern_detailed(const std::string& hypothesis_text) const;

    // H1: Generate hypothesis-specific training data
    std::vector<DataPoint> generate_training_data(
        RelationshipPattern pattern,
        size_t num_points,
        double range_min = 0.0,
        double range_max = 1.0,
        const NumericHints& hints = {}
    ) const;

    // H1: Extract numeric hints from hypothesis text
    NumericHints extract_numeric_hints(const std::string& text) const;

    const Config& get_config() const { return config_; }

private:
    Config config_;

    // Pattern-specific data generators (updated for H1)
    std::vector<DataPoint> generate_linear_data(size_t n, double min, double max, const NumericHints& hints) const;
    std::vector<DataPoint> generate_polynomial_data(size_t n, double min, double max, const NumericHints& hints) const;
    std::vector<DataPoint> generate_exponential_data(size_t n, double min, double max, const NumericHints& hints) const;
    std::vector<DataPoint> generate_periodic_data(size_t n, double min, double max, const NumericHints& hints) const;
    std::vector<DataPoint> generate_threshold_data(size_t n, double min, double max, const NumericHints& hints) const;
    std::vector<DataPoint> generate_conditional_data(size_t n, double min, double max, const NumericHints& hints) const;

    std::vector<size_t> suggest_topology(RelationshipPattern pattern, size_t input_dim = 1) const;
    KanTrainingConfig suggest_config(RelationshipPattern pattern) const;

    // C1: NLP-lite helpers
    bool contains_any(const std::string& text, const std::vector<std::string>& keywords) const;
    std::string to_lower(const std::string& text) const;
    std::vector<std::string> split_sentences(const std::string& text) const;
    bool is_negated(const std::string& text, const std::string& keyword) const;
    size_t count_variables(const std::string& text) const;
    double detect_quantifier_modifier(const std::string& text) const;
};

} // namespace brain19

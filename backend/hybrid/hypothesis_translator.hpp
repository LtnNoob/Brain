#pragma once

#include "../understanding/understanding_proposals.hpp"
#include "../kan/kan_module.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

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
    NOT_QUANTIFIABLE // Cannot be translated to numeric relationship
};

inline const char* pattern_to_string(RelationshipPattern p) {
    switch (p) {
        case RelationshipPattern::LINEAR: return "LINEAR";
        case RelationshipPattern::POLYNOMIAL: return "POLYNOMIAL";
        case RelationshipPattern::EXPONENTIAL: return "EXPONENTIAL";
        case RelationshipPattern::PERIODIC: return "PERIODIC";
        case RelationshipPattern::THRESHOLD: return "THRESHOLD";
        case RelationshipPattern::NOT_QUANTIFIABLE: return "NOT_QUANTIFIABLE";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// KAN TRAINING PROBLEM
// =============================================================================
//
// Represents a translated hypothesis as a KAN-trainable problem.
// Generated from HypothesisProposal by HypothesisTranslator.
//
struct KanTrainingProblem {
    // Original hypothesis reference
    uint64_t hypothesis_id;

    // Detected pattern
    RelationshipPattern pattern;

    // KAN dimensions
    size_t input_dim;
    size_t output_dim;

    // Synthetic training data generated from the hypothesis
    std::vector<DataPoint> training_data;

    // Suggested KAN topology (e.g., [input_dim, hidden, output_dim])
    std::vector<size_t> suggested_topology;

    // Suggested training config
    KanTrainingConfig suggested_config;

    // Human-readable description of the extracted relationship
    std::string relationship_description;

    // NO default constructor — must be explicitly constructed
    KanTrainingProblem() = delete;

    KanTrainingProblem(
        uint64_t hyp_id,
        RelationshipPattern pat,
        size_t in_dim,
        size_t out_dim,
        std::vector<DataPoint> data,
        std::vector<size_t> topology,
        KanTrainingConfig config,
        std::string desc
    ) : hypothesis_id(hyp_id)
      , pattern(pat)
      , input_dim(in_dim)
      , output_dim(out_dim)
      , training_data(std::move(data))
      , suggested_topology(std::move(topology))
      , suggested_config(config)
      , relationship_description(std::move(desc))
    {}
};

// =============================================================================
// TRANSLATION RESULT
// =============================================================================

struct TranslationResult {
    bool translatable;
    std::optional<KanTrainingProblem> problem;
    RelationshipPattern detected_pattern;
    std::string explanation;

    // For non-quantifiable results
    static TranslationResult not_quantifiable(const std::string& reason) {
        TranslationResult r;
        r.translatable = false;
        r.problem = std::nullopt;
        r.detected_pattern = RelationshipPattern::NOT_QUANTIFIABLE;
        r.explanation = reason;
        return r;
    }

private:
    TranslationResult() : translatable(false), detected_pattern(RelationshipPattern::NOT_QUANTIFIABLE) {}
    friend class HypothesisTranslator;
};

// =============================================================================
// HYPOTHESIS TRANSLATOR
// =============================================================================
//
// Translates LLM-generated HypothesisProposals into KAN-trainable problems.
//
// ARCHITECTURE:
// - Analyzes hypothesis text for numeric relationship patterns
// - Generates synthetic training data based on detected pattern
// - Configures KAN topology appropriate for the pattern
//
// SUPPORTED PATTERNS:
// - LINEAR: "X increases proportionally with Y"
// - POLYNOMIAL: "X grows quadratically with Y"  
// - EXPONENTIAL: "X grows exponentially with Y"
// - PERIODIC: "X cycles/oscillates with Y"
// - THRESHOLD: "X triggers when Y exceeds Z"
//
class HypothesisTranslator {
public:
    struct Config {
        size_t min_data_points = 20;
        size_t max_data_points = 100;
        size_t default_num_knots = 10;
        double data_range_min = 0.0;
        double data_range_max = 1.0;
        size_t default_hidden_dim = 5;
    };

    HypothesisTranslator() : HypothesisTranslator(Config{}) {}
    explicit HypothesisTranslator(Config config);

    // Translate a HypothesisProposal into a KAN training problem
    TranslationResult translate(const HypothesisProposal& proposal) const;

    // Detect relationship pattern from hypothesis text
    RelationshipPattern detect_pattern(const std::string& hypothesis_text) const;

    // Generate synthetic training data for a given pattern
    std::vector<DataPoint> generate_training_data(
        RelationshipPattern pattern,
        size_t num_points,
        double range_min = 0.0,
        double range_max = 1.0
    ) const;

    const Config& get_config() const { return config_; }

private:
    Config config_;

    // Pattern-specific data generators
    std::vector<DataPoint> generate_linear_data(size_t n, double min, double max) const;
    std::vector<DataPoint> generate_polynomial_data(size_t n, double min, double max) const;
    std::vector<DataPoint> generate_exponential_data(size_t n, double min, double max) const;
    std::vector<DataPoint> generate_periodic_data(size_t n, double min, double max) const;
    std::vector<DataPoint> generate_threshold_data(size_t n, double min, double max) const;

    // Pattern-specific topology suggestions
    std::vector<size_t> suggest_topology(RelationshipPattern pattern) const;

    // Pattern-specific training config
    KanTrainingConfig suggest_config(RelationshipPattern pattern) const;

    // Text analysis helpers
    bool contains_any(const std::string& text, const std::vector<std::string>& keywords) const;
    std::string to_lower(const std::string& text) const;
};

} // namespace brain19

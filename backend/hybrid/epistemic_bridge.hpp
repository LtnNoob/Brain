#pragma once

#include "../epistemic/epistemic_metadata.hpp"
#include "../kan/function_hypothesis.hpp"
#include "../kan/kan_module.hpp"
#include "hypothesis_translator.hpp"
#include <string>
#include <optional>

namespace brain19 {

// =============================================================================
// EPISTEMIC ASSESSMENT
// =============================================================================
//
// Result of evaluating a KAN training outcome epistemically.
// Maps mathematical fit quality to epistemic trust and type.
//
struct EpistemicAssessment {
    EpistemicMetadata metadata;
    double mse;
    bool converged;
    size_t iterations_used;
    double convergence_speed;  // iterations / max_iterations (lower = faster)
    std::string explanation;
    bool is_interpretable;     // KAN approximated a known function form

    // NO default constructor
    EpistemicAssessment() = delete;

    EpistemicAssessment(
        EpistemicMetadata meta,
        double mse_val,
        bool conv,
        size_t iters,
        double conv_speed,
        std::string expl,
        bool interpretable
    ) : metadata(meta)
      , mse(mse_val)
      , converged(conv)
      , iterations_used(iters)
      , convergence_speed(conv_speed)
      , explanation(std::move(expl))
      , is_interpretable(interpretable)
    {}
};

// =============================================================================
// EPISTEMIC BRIDGE
// =============================================================================
//
// Bridges KAN training results to epistemic metadata.
// Maps mathematical fit quality to trust scores and epistemic types.
//
// MAPPING:
//   MSE < 0.01        → THEORY candidate  (Trust 0.7-0.9)
//   MSE < 0.1         → HYPOTHESIS        (Trust 0.4-0.6)
//   MSE >= 0.1        → SPECULATION       (Trust 0.1-0.3)
//   Not converged     → INVALIDATED       (Trust 0.05)
//
// MODIFIERS:
//   Fast convergence   → +0.1 trust bonus
//   Interpretable form → +0.05 trust bonus
//
class EpistemicBridge {
public:
    struct Config {
        double theory_mse_threshold = 0.01;
        double hypothesis_mse_threshold = 0.1;
        double fast_convergence_ratio = 0.3;   // < 30% of max_iterations = fast
        double convergence_trust_bonus = 0.1;
        double interpretability_trust_bonus = 0.05;
    };

    EpistemicBridge() : EpistemicBridge(Config{}) {}
    explicit EpistemicBridge(Config config);

    // Assess a KAN training result epistemically
    EpistemicAssessment assess(
        const FunctionHypothesis& hypothesis,
        const KanTrainingResult& training_result,
        const KanTrainingConfig& training_config
    ) const;

    // Check if a trained KAN approximates a known function form
    // (linear, quadratic, etc.) by analyzing B-spline coefficients
    bool check_interpretability(const KANModule& module) const;

    const Config& get_config() const { return config_; }

private:
    Config config_;

    // Compute trust score from MSE and modifiers
    double compute_trust(
        double mse,
        bool converged,
        double convergence_speed,
        bool interpretable
    ) const;

    // Determine epistemic type from MSE
    EpistemicType determine_type(double mse, bool converged) const;

    // Determine epistemic status
    EpistemicStatus determine_status(bool converged) const;

    // Build explanation string
    std::string build_explanation(
        double mse,
        bool converged,
        double convergence_speed,
        bool interpretable,
        EpistemicType type,
        double trust
    ) const;
};

} // namespace brain19

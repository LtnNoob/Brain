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

struct EpistemicAssessment {
    EpistemicMetadata metadata;
    double mse;
    bool converged;
    size_t iterations_used;
    double convergence_speed;
    std::string explanation;
    bool is_interpretable;
    DataQuality data_quality;  // H2: track data source

    EpistemicAssessment() = delete;

    EpistemicAssessment(
        EpistemicMetadata meta,
        double mse_val,
        bool conv,
        size_t iters,
        double conv_speed,
        std::string expl,
        bool interpretable,
        DataQuality quality = DataQuality::SYNTHETIC_CANONICAL
    ) : metadata(meta)
      , mse(mse_val)
      , converged(conv)
      , iterations_used(iters)
      , convergence_speed(conv_speed)
      , explanation(std::move(expl))
      , is_interpretable(interpretable)
      , data_quality(quality)
    {}
};

// =============================================================================
// EPISTEMIC BRIDGE
// =============================================================================
//
// MAPPING:
//   MSE < 0.01        → THEORY candidate  (Trust 0.7-0.9)
//   MSE < 0.1         → HYPOTHESIS        (Trust 0.4-0.6)
//   MSE >= 0.1        → SPECULATION       (Trust 0.1-0.3)
//   Not converged     → INVALIDATED       (Trust 0.05)
//
// H2 MODIFIERS:
//   Synthetic data     → max trust 0.6 (hard cap)
//   EXTRACTED data     → 1.0x multiplier
//   SYNTHETIC          → 0.6x multiplier
//   Trivial convergence (<10 epochs) → novelty penalty
//   Trust > 0.5 requires ≥ 50 data points
//
class EpistemicBridge {
public:
    struct Config {
        double theory_mse_threshold = 0.01;
        double hypothesis_mse_threshold = 0.1;
        double fast_convergence_ratio = 0.3;
        double convergence_trust_bonus = 0.1;
        double interpretability_trust_bonus = 0.05;
        // H2: Trust-inflation caps
        double synthetic_trust_cap = 0.6;
        double synthetic_multiplier = 0.6;
        double extracted_multiplier = 1.0;
        size_t trivial_convergence_epochs = 10;
        double trivial_convergence_penalty = 0.15;
        size_t min_data_points_for_high_trust = 50;
    };

    EpistemicBridge() : EpistemicBridge(Config{}) {}
    explicit EpistemicBridge(Config config);

    // H2: Updated assess with data quality and data point count
    EpistemicAssessment assess(
        const FunctionHypothesis& hypothesis,
        const KanTrainingResult& training_result,
        const KanTrainingConfig& training_config,
        DataQuality data_quality = DataQuality::SYNTHETIC_CANONICAL,
        size_t num_data_points = 0
    ) const;

    bool check_interpretability(const KANModule& module) const;
    const Config& get_config() const { return config_; }

private:
    Config config_;

    double compute_trust(
        double mse,
        bool converged,
        double convergence_speed,
        bool interpretable,
        DataQuality data_quality,
        size_t iterations_used,
        size_t num_data_points
    ) const;

    EpistemicType determine_type(double mse, bool converged) const;
    EpistemicStatus determine_status(bool converged) const;

    std::string build_explanation(
        double mse,
        bool converged,
        double convergence_speed,
        bool interpretable,
        EpistemicType type,
        double trust,
        DataQuality data_quality
    ) const;
};

} // namespace brain19

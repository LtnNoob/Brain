#include "epistemic_bridge.hpp"
#include <sstream>
#include <cmath>
#include <algorithm>

namespace brain19 {

EpistemicBridge::EpistemicBridge(Config config)
    : config_(std::move(config))
{}

EpistemicAssessment EpistemicBridge::assess(
    const FunctionHypothesis& hypothesis,
    const KanTrainingResult& training_result,
    const KanTrainingConfig& training_config,
    DataQuality data_quality,
    size_t num_data_points
) const {
    double mse = training_result.final_loss;
    bool converged = training_result.converged;
    size_t iters = training_result.iterations_run;

    double convergence_speed = (training_config.max_iterations > 0)
        ? static_cast<double>(iters) / static_cast<double>(training_config.max_iterations)
        : 1.0;

    bool interpretable = false;
    if (hypothesis.module) {
        interpretable = check_interpretability(*hypothesis.module);
    }

    EpistemicType type = determine_type(mse, converged);
    EpistemicStatus status = determine_status(converged);
    double trust = compute_trust(mse, converged, convergence_speed, interpretable,
                                  data_quality, iters, num_data_points);

    std::string explanation = build_explanation(
        mse, converged, convergence_speed, interpretable, type, trust, data_quality
    );

    EpistemicMetadata metadata(type, status, trust);

    return EpistemicAssessment(
        metadata, mse, converged, iters, convergence_speed,
        std::move(explanation), interpretable, data_quality
    );
}

double EpistemicBridge::compute_trust(
    double mse,
    bool converged,
    double convergence_speed,
    bool interpretable,
    DataQuality data_quality,
    size_t iterations_used,
    size_t num_data_points
) const {
    if (!converged) {
        return 0.05;
    }

    double base_trust;
    if (mse < config_.theory_mse_threshold) {
        double mse_ratio = (config_.theory_mse_threshold > 1e-12) ? mse / config_.theory_mse_threshold : 0.0;
        base_trust = 0.9 - 0.2 * mse_ratio;
    } else if (mse < config_.hypothesis_mse_threshold) {
        double denom = config_.hypothesis_mse_threshold - config_.theory_mse_threshold;
        double mse_ratio = (denom > 1e-12) ? (mse - config_.theory_mse_threshold) / denom : 0.5;
        base_trust = 0.6 - 0.2 * mse_ratio;
    } else {
        double mse_capped = std::min(mse, 1.0);
        double denom = 1.0 - config_.hypothesis_mse_threshold;
        double mse_ratio = (denom > 1e-12) ? (mse_capped - config_.hypothesis_mse_threshold) / denom : 0.5;
        base_trust = 0.3 - 0.2 * mse_ratio;
    }

    // Apply bonuses
    double trust = base_trust;
    if (convergence_speed < config_.fast_convergence_ratio) {
        trust += config_.convergence_trust_bonus;
    }
    if (interpretable) {
        trust += config_.interpretability_trust_bonus;
    }

    // ==========================================================
    // H2: TRUST-INFLATION CAPS
    // ==========================================================

    // H2: Data source multiplier
    double source_multiplier = (data_quality == DataQuality::EXTRACTED)
        ? config_.extracted_multiplier
        : config_.synthetic_multiplier;
    trust *= source_multiplier;

    // H2: Hard cap for synthetic data
    if (data_quality != DataQuality::EXTRACTED) {
        trust = std::min(trust, config_.synthetic_trust_cap);
    }

    // H2: Novelty penalty — trivial convergence means the problem was too easy
    if (iterations_used < config_.trivial_convergence_epochs) {
        trust -= config_.trivial_convergence_penalty;
    }

    // H2: Minimum data points for high trust
    // FIX NEW-3: num_data_points=0 means "unknown" → treat as insufficient data
    if (num_data_points < config_.min_data_points_for_high_trust) {
        trust = std::min(trust, 0.5);
    }

    // Clamp to [0.0, 1.0]
    return std::max(0.0, std::min(1.0, trust));
}

EpistemicType EpistemicBridge::determine_type(double mse, bool converged) const {
    if (!converged) {
        return EpistemicType::SPECULATION;
    }
    if (mse < config_.theory_mse_threshold) {
        return EpistemicType::THEORY;
    }
    if (mse < config_.hypothesis_mse_threshold) {
        return EpistemicType::HYPOTHESIS;
    }
    return EpistemicType::SPECULATION;
}

EpistemicStatus EpistemicBridge::determine_status(bool converged) const {
    if (!converged) {
        return EpistemicStatus::INVALIDATED;
    }
    return EpistemicStatus::ACTIVE;
}

bool EpistemicBridge::check_interpretability(const KANModule& module) const {
    // M5 FIX: Extended interpretability check with B-spline monotonicity & linearity
    if (module.num_layers() != 1) return false;
    if (module.input_dim() != 1 || module.output_dim() != 1) return false;

    const auto& layer = module.layer(0);
    const auto& node = layer.node(0, 0);
    const auto& coefs = node.get_coefficients();

    if (coefs.size() < 3) return false;

    // Compute first differences (approximate first derivative of B-spline)
    std::vector<double> diffs;
    diffs.reserve(coefs.size() - 1);
    for (size_t i = 1; i < coefs.size(); ++i) {
        diffs.push_back(coefs[i] - coefs[i - 1]);
    }

    // Check 1: Monotonicity — all diffs same sign (or near-zero)
    bool monotone_increasing = true;
    bool monotone_decreasing = true;
    constexpr double mono_eps = 1e-4;
    for (double d : diffs) {
        if (d < -mono_eps) monotone_increasing = false;
        if (d > mono_eps)  monotone_decreasing = false;
    }
    bool is_monotone = monotone_increasing || monotone_decreasing;

    // Check 2: Linearity — low variance in first differences (constant slope)
    double mean = 0.0;
    for (double d : diffs) mean += d;
    mean /= static_cast<double>(diffs.size());

    double variance = 0.0;
    for (double d : diffs) {
        double diff = d - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(diffs.size());

    bool is_nearly_linear = (variance < 0.01);

    // Interpretable if monotone OR nearly linear
    return is_monotone || is_nearly_linear;
}

std::string EpistemicBridge::build_explanation(
    double mse,
    bool converged,
    double convergence_speed,
    bool interpretable,
    EpistemicType type,
    double trust,
    DataQuality data_quality
) const {
    std::ostringstream oss;
    oss << "KAN Validation: ";

    if (!converged) {
        oss << "FAILED (did not converge). MSE=" << mse;
        return oss.str();
    }

    oss << epistemic_type_to_string(type) << " (Trust=" << trust << "). ";
    oss << "MSE=" << mse << ", ";
    oss << "Convergence speed=" << (convergence_speed * 100.0) << "% of max iterations";

    if (convergence_speed < config_.fast_convergence_ratio) {
        oss << " [FAST]";
    }
    if (interpretable) {
        oss << " [INTERPRETABLE]";
    }

    // H2: Show data quality impact
    oss << " [Data: " << data_quality_to_string(data_quality) << "]";
    if (data_quality != DataQuality::EXTRACTED) {
        oss << " [TRUST CAPPED at " << config_.synthetic_trust_cap << "]";
    }

    return oss.str();
}

} // namespace brain19

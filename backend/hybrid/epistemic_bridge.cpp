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
    const KanTrainingConfig& training_config
) const {
    double mse = training_result.final_loss;
    bool converged = training_result.converged;
    size_t iters = training_result.iterations_run;

    // Convergence speed: fraction of max iterations used (lower = faster)
    double convergence_speed = (training_config.max_iterations > 0)
        ? static_cast<double>(iters) / static_cast<double>(training_config.max_iterations)
        : 1.0;

    // Check interpretability if we have a valid module
    bool interpretable = false;
    if (hypothesis.module) {
        interpretable = check_interpretability(*hypothesis.module);
    }

    EpistemicType type = determine_type(mse, converged);
    EpistemicStatus status = determine_status(converged);
    double trust = compute_trust(mse, converged, convergence_speed, interpretable);

    std::string explanation = build_explanation(
        mse, converged, convergence_speed, interpretable, type, trust
    );

    EpistemicMetadata metadata(type, status, trust);

    return EpistemicAssessment(
        metadata, mse, converged, iters, convergence_speed,
        std::move(explanation), interpretable
    );
}

double EpistemicBridge::compute_trust(
    double mse,
    bool converged,
    double convergence_speed,
    bool interpretable
) const {
    if (!converged) {
        return 0.05;  // INVALIDATED — minimal trust
    }

    double base_trust;
    if (mse < config_.theory_mse_threshold) {
        // THEORY range: 0.7 - 0.9
        // Better MSE → higher trust within range
        double mse_ratio = mse / config_.theory_mse_threshold;
        base_trust = 0.9 - 0.2 * mse_ratio;
    } else if (mse < config_.hypothesis_mse_threshold) {
        // HYPOTHESIS range: 0.4 - 0.6
        double mse_ratio = (mse - config_.theory_mse_threshold) 
                         / (config_.hypothesis_mse_threshold - config_.theory_mse_threshold);
        base_trust = 0.6 - 0.2 * mse_ratio;
    } else {
        // SPECULATION range: 0.1 - 0.3
        double mse_capped = std::min(mse, 1.0);
        double mse_ratio = (mse_capped - config_.hypothesis_mse_threshold)
                         / (1.0 - config_.hypothesis_mse_threshold);
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

    // Clamp to [0.0, 1.0]
    return std::max(0.0, std::min(1.0, trust));
}

EpistemicType EpistemicBridge::determine_type(double mse, bool converged) const {
    if (!converged) {
        return EpistemicType::SPECULATION;  // Will be INVALIDATED by status
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
    // Simple heuristic: If the KAN has a single layer with 1→1 topology,
    // check if the B-spline coefficients form a near-linear or near-quadratic pattern.
    if (module.num_layers() != 1) return false;
    if (module.input_dim() != 1 || module.output_dim() != 1) return false;

    const auto& layer = module.layer(0);
    const auto& node = layer.node(0, 0);
    const auto& coefs = node.get_coefficients();

    if (coefs.size() < 3) return false;

    // Check linearity: differences between consecutive coefficients should be ~constant
    std::vector<double> diffs;
    for (size_t i = 1; i < coefs.size(); ++i) {
        diffs.push_back(coefs[i] - coefs[i - 1]);
    }

    // Compute variance of diffs
    double mean = 0.0;
    for (double d : diffs) mean += d;
    mean /= static_cast<double>(diffs.size());

    double variance = 0.0;
    for (double d : diffs) {
        double diff = d - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(diffs.size());

    // Low variance in diffs → approximately linear
    return variance < 0.01;
}

std::string EpistemicBridge::build_explanation(
    double mse,
    bool converged,
    double convergence_speed,
    bool interpretable,
    EpistemicType type,
    double trust
) const {
    std::ostringstream oss;
    oss << "KAN Validation: ";

    if (!converged) {
        oss << "FAILED (did not converge). MSE=" << mse;
        return oss.str();
    }

    oss << to_string(type) << " (Trust=" << trust << "). ";
    oss << "MSE=" << mse << ", ";
    oss << "Convergence speed=" << (convergence_speed * 100.0) << "% of max iterations";

    if (convergence_speed < config_.fast_convergence_ratio) {
        oss << " [FAST]";
    }
    if (interpretable) {
        oss << " [INTERPRETABLE]";
    }

    return oss.str();
}

} // namespace brain19

#include "refinement_loop.hpp"
#include <sstream>
#include <cmath>

namespace brain19 {

RefinementLoop::RefinementLoop(
    KanValidator validator,
    Config config
) : validator_(std::move(validator))
  , config_(std::move(config))
{}

RefinementResult RefinementLoop::run(
    const HypothesisProposal& initial_hypothesis,
    HypothesisRefinerFn refiner
) const {
    std::vector<RefinementIteration> chain;
    HypothesisProposal current_hypothesis = initial_hypothesis;
    double prev_mse = std::numeric_limits<double>::max();
    ValidationResult last_validation = validator_.validate(current_hypothesis);

    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        auto validation = validator_.validate(current_hypothesis);
        double current_mse = validation.assessment.mse;

        std::string feedback = build_residual_feedback(validation, iter);

        chain.emplace_back(
            iter,
            current_hypothesis,
            validation,
            feedback,
            current_mse
        );

        last_validation = std::move(validation);

        // Check convergence: MSE below threshold
        if (current_mse < config_.mse_threshold) {
            return RefinementResult(
                true,
                iter + 1,
                std::move(chain),
                std::move(last_validation)
            );
        }

        // Check improvement stall
        double improvement = prev_mse - current_mse;
        if (iter > 0 && improvement < config_.improvement_threshold) {
            return RefinementResult(
                false,
                iter + 1,
                std::move(chain),
                std::move(last_validation)
            );
        }

        prev_mse = current_mse;

        // Don't refine after last iteration
        if (iter + 1 < config_.max_iterations) {
            current_hypothesis = refiner(current_hypothesis, feedback);
        }
    }

    // Max iterations reached
    return RefinementResult(
        false,
        config_.max_iterations,
        std::move(chain),
        std::move(last_validation)
    );
}

std::string RefinementLoop::build_residual_feedback(
    const ValidationResult& result,
    size_t iteration
) const {
    std::ostringstream oss;
    oss << "Iteration " << (iteration + 1) << ": ";

    if (!result.validated) {
        oss << "Hypothesis could not be validated. ";
        oss << "Pattern: " << pattern_to_string(result.pattern) << ". ";
    }

    oss << "MSE=" << result.assessment.mse << ". ";

    if (result.assessment.converged) {
        oss << "KAN converged in " << result.assessment.iterations_used << " iterations. ";
    } else {
        oss << "KAN did NOT converge. ";
    }

    oss << "Trust=" << result.assessment.metadata.trust << ". ";

    if (result.assessment.mse > config_.mse_threshold) {
        oss << "Your hypothesis deviates significantly. ";
        oss << "Please refine the relationship to better match the mathematical pattern.";
    } else {
        oss << "Good fit achieved.";
    }

    return oss.str();
}

} // namespace brain19

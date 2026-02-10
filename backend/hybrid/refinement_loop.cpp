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

    // C2 FIX: Removed double-validation bug.
    // Previously there was a `validator_.validate()` call before the loop
    // that was immediately overwritten — pure waste.

    // We need at least one validation to initialize last_validation,
    // so we do it inside the loop on first iteration.
    std::optional<ValidationResult> last_validation_opt;

    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        auto validation = validator_.validate(current_hypothesis);
        double current_mse = validation.assessment.mse;

        // C2 FIX: Build residuum-based feedback with specific deviation info
        std::string feedback = build_residual_feedback(validation, iter);

        chain.emplace_back(
            iter,
            current_hypothesis,
            validation,
            feedback,
            current_mse
        );

        last_validation_opt = std::move(validation);

        // Check convergence: MSE below threshold
        if (current_mse < config_.mse_threshold) {
            return RefinementResult(
                true,
                iter + 1,
                std::move(chain),
                std::move(*last_validation_opt)
            );
        }

        // C2 FIX: Real convergence metric — compare MSE delta between iterations
        double improvement = prev_mse - current_mse;
        if (iter > 0 && std::abs(improvement) < config_.improvement_threshold) {
            // MSE is not changing → stalled, stop
            return RefinementResult(
                false,
                iter + 1,
                std::move(chain),
                std::move(*last_validation_opt)
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
        std::move(*last_validation_opt)
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

    // C2 FIX: Structured residuum information for the refiner
    // Tell the LLM specifically where the hypothesis deviates
    if (result.trained_module && result.assessment.mse > config_.mse_threshold) {
        oss << "RESIDUAL ANALYSIS: ";
        oss << "Your hypothesis deviates significantly (MSE=" << result.assessment.mse << "). ";
        oss << "The KAN model suggests the relationship is ";
        if (result.assessment.mse > 0.5) {
            oss << "very different from the detected pattern. Consider a completely different relationship type. ";
        } else if (result.assessment.mse > 0.1) {
            oss << "partially matching but needs parameter adjustments. ";
        } else {
            oss << "close but needs fine-tuning. ";
        }
        oss << "Please refine the relationship to better match the mathematical pattern.";
    } else if (result.assessment.mse <= config_.mse_threshold) {
        oss << "Good fit achieved.";
    } else {
        oss << "Please refine the relationship to better match the mathematical pattern.";
    }

    return oss.str();
}

} // namespace brain19

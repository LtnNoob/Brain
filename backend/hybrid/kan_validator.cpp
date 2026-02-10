#include "kan_validator.hpp"
#include <memory>

namespace brain19 {

KanValidator::KanValidator(Config config)
    : config_(std::move(config))
    , translator_(config_.translator_config)
    , bridge_(config_.bridge_config)
{}

ValidationResult KanValidator::validate(const HypothesisProposal& proposal) const {
    // Step 1: Translate hypothesis to KAN problem
    auto translation = translator_.translate(proposal);

    if (!translation.translatable || !translation.problem.has_value()) {
        // Not quantifiable — return low-trust SPECULATION
        EpistemicAssessment not_quant_assessment(
            EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.1),
            1.0,    // high MSE (no fit)
            false,  // not converged
            0,      // no iterations
            1.0,    // worst convergence speed
            "Not quantifiable: " + translation.explanation,
            false   // not interpretable
        );

        return ValidationResult(
            false,
            std::move(not_quant_assessment),
            RelationshipPattern::NOT_QUANTIFIABLE,
            nullptr,
            translation.explanation
        );
    }

    const auto& problem = translation.problem.value();

    // Step 2: Check minimum data points
    if (problem.training_data.size() < config_.min_data_points) {
        EpistemicAssessment insufficient_assessment(
            EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.1),
            1.0, false, 0, 1.0,
            "Insufficient training data: " + std::to_string(problem.training_data.size())
            + " < " + std::to_string(config_.min_data_points),
            false
        );

        return ValidationResult(
            false,
            std::move(insufficient_assessment),
            problem.pattern,
            nullptr,
            "Insufficient training data"
        );
    }

    // Step 3: Create and train KAN
    auto kan = std::make_shared<KANModule>(
        problem.suggested_topology,
        config_.default_num_knots
    );

    KanTrainingConfig train_config = problem.suggested_config;
    train_config.max_iterations = config_.max_epochs;
    train_config.convergence_threshold = config_.convergence_threshold;

    auto training_result = kan->train(problem.training_data, train_config);

    // Step 4: Create FunctionHypothesis for epistemic assessment
    FunctionHypothesis func_hyp(
        problem.input_dim,
        problem.output_dim,
        kan,
        training_result.iterations_run,
        training_result.final_loss
    );

    // Step 5: Epistemic assessment
    auto assessment = bridge_.assess(func_hyp, training_result, train_config);

    bool validated = assessment.converged && 
                     assessment.metadata.type != EpistemicType::SPECULATION;

    return ValidationResult(
        validated,
        std::move(assessment),
        problem.pattern,
        kan,
        assessment.explanation
    );
}

} // namespace brain19

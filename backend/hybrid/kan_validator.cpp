#include "kan_validator.hpp"
#include <memory>
#include <cmath>
#include <sstream>

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
        EpistemicAssessment not_quant_assessment(
            EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.1),
            1.0, false, 0, 1.0,
            "Not quantifiable: " + translation.explanation,
            false
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

    // Step 3: Create KAN (warm-start from cache if available)
    std::shared_ptr<KANModule> kan;
    if (config_.enable_model_cache) {
        auto cached = model_cache_.get(problem.pattern);
        if (cached) {
            kan = cached->clone();
        }
    }
    if (!kan) {
        kan = std::make_shared<KANModule>(
            problem.suggested_topology,
            config_.default_num_knots
        );
    }

    KanTrainingConfig train_config = problem.suggested_config;
    // M2 FIX: Only override translator suggestions when explicitly configured
    if (config_.max_epochs_override.has_value()) {
        train_config.max_iterations = config_.max_epochs_override.value();
    }
    // else: keep problem.suggested_config.max_iterations from translator
    if (config_.convergence_threshold_override.has_value()) {
        train_config.convergence_threshold = config_.convergence_threshold_override.value();
    }

    auto training_result = kan->train(problem.training_data, train_config);

    // Step 4: Create FunctionHypothesis for epistemic assessment
    FunctionHypothesis func_hyp(
        problem.input_dim,
        problem.output_dim,
        kan,
        training_result.iterations_run,
        training_result.final_loss
    );

    // Step 5: Epistemic assessment — H2: pass data quality and data point count
    auto assessment = bridge_.assess(
        func_hyp, training_result, train_config,
        problem.data_quality,
        problem.training_data.size()
    );

    bool validated = assessment.converged &&
                     assessment.metadata.type != EpistemicType::SPECULATION;

    // Cache the trained model if it performed well
    if (config_.enable_model_cache && validated) {
        model_cache_.store(problem.pattern, kan, training_result.final_loss);
    }

    return ValidationResult(
        validated,
        std::move(assessment),
        problem.pattern,
        kan,
        assessment.explanation
    );
}

// =============================================================================
// CHAIN VALIDATION
// =============================================================================

ChainValidationResult KanValidator::validate_chain(
    const HypothesisProposal& proposal,
    const LongTermMemory& ltm) const
{
    const auto& evidence = proposal.evidence_concepts;
    if (evidence.size() < 3) {
        return ChainValidationResult(false, 0.0, 0.0, "Chain too short");
    }

    std::vector<ValidationResult> edge_results;
    double product = 1.0;
    double weakest = 1.0;
    size_t n_edges = 0;
    std::ostringstream summary;

    // Walk evidence_concepts as consecutive pairs
    for (size_t i = 0; i + 1 < evidence.size(); ++i) {
        ConceptId src = evidence[i];
        ConceptId tgt = evidence[i + 1];

        auto rels = ltm.get_relations_between(src, tgt);
        if (rels.empty()) {
            rels = ltm.get_relations_between(tgt, src);
        }

        if (rels.empty()) {
            // No relation found — weak link
            product *= 0.1;
            weakest = std::min(weakest, 0.1);
            ++n_edges;
            continue;
        }

        // Build a single-edge hypothesis for validation
        HypothesisProposal edge_hyp(
            0,
            {src, tgt},
            "Edge: " + std::to_string(src) + " -> " + std::to_string(tgt),
            "Chain edge validation",
            std::vector<std::string>{"chain-edge", "proportional", "scales with"},
            proposal.model_confidence,
            proposal.source_model
        );

        try {
            auto vr = validate(edge_hyp);
            double edge_trust = vr.assessment.metadata.trust;
            product *= edge_trust;
            weakest = std::min(weakest, edge_trust);
            edge_results.push_back(std::move(vr));
        } catch (...) {
            product *= 0.1;
            weakest = std::min(weakest, 0.1);
        }
        ++n_edges;
    }

    if (n_edges == 0) {
        return ChainValidationResult(false, 0.0, 0.0, "No edges in chain");
    }

    double geo_mean = std::pow(product, 1.0 / static_cast<double>(n_edges));
    bool all_strong = weakest >= config_.min_chain_edge_confidence;

    summary << n_edges << " edges, geo_mean=" << std::fixed;
    summary.precision(3);
    summary << geo_mean << ", weakest=" << weakest;

    ChainValidationResult result(all_strong, geo_mean, weakest, summary.str());
    result.edge_results = std::move(edge_results);
    return result;
}

} // namespace brain19

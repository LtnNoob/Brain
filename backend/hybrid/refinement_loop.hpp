#pragma once

#include "kan_validator.hpp"
#include "../understanding/mini_llm.hpp"
#include "../understanding/understanding_proposals.hpp"
#include <vector>
#include <string>
#include <functional>

namespace brain19 {

// =============================================================================
// REFINEMENT ITERATION
// =============================================================================

struct RefinementIteration {
    size_t iteration_number;
    HypothesisProposal hypothesis;
    ValidationResult validation;
    std::string residual_feedback;  // Feedback sent back to LLM
    double mse;

    RefinementIteration() = delete;

    RefinementIteration(
        size_t iter,
        HypothesisProposal hyp,
        ValidationResult val,
        std::string feedback,
        double mse_val
    ) : iteration_number(iter)
      , hypothesis(std::move(hyp))
      , validation(std::move(val))
      , residual_feedback(std::move(feedback))
      , mse(mse_val)
    {}
};

// =============================================================================
// REFINEMENT RESULT
// =============================================================================

struct RefinementResult {
    bool converged;
    size_t iterations_performed;
    std::vector<RefinementIteration> provenance_chain;
    ValidationResult final_validation;

    RefinementResult() = delete;

    RefinementResult(
        bool conv,
        size_t iters,
        std::vector<RefinementIteration> chain,
        ValidationResult final_val
    ) : converged(conv)
      , iterations_performed(iters)
      , provenance_chain(std::move(chain))
      , final_validation(std::move(final_val))
    {}
};

// =============================================================================
// HYPOTHESIS REFINER (callback for LLM interaction)
// =============================================================================

// Callback type: Given residual feedback, produce a refined hypothesis
using HypothesisRefinerFn = std::function<HypothesisProposal(
    const HypothesisProposal& previous,
    const std::string& residual_feedback
)>;

// =============================================================================
// REFINEMENT LOOP
// =============================================================================
//
// Bidirectional dialog between LLM and KAN:
// 1. LLM generates hypothesis
// 2. KAN validates → computes residuum
// 3. Residuum fed back to LLM
// 4. LLM generates refined hypothesis
// 5. Repeat until convergence or max iterations
//
class RefinementLoop {
public:
    struct Config {
        size_t max_iterations = 5;
        double mse_threshold = 0.01;         // Converged when MSE < this
        double improvement_threshold = 0.001; // Stop if improvement < this
    };

    explicit RefinementLoop(KanValidator validator)
        : RefinementLoop(std::move(validator), Config{}) {}
    RefinementLoop(KanValidator validator, Config config);

    // Run the refinement loop
    // initial_hypothesis: First hypothesis from LLM
    // refiner: Callback to get refined hypothesis from LLM given feedback
    RefinementResult run(
        const HypothesisProposal& initial_hypothesis,
        HypothesisRefinerFn refiner
    ) const;

    const Config& get_config() const { return config_; }

private:
    KanValidator validator_;
    Config config_;

    // Build residual feedback string from validation result
    std::string build_residual_feedback(
        const ValidationResult& result,
        size_t iteration
    ) const;
};

} // namespace brain19

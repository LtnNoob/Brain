#pragma once

#include "hypothesis_translator.hpp"
#include "epistemic_bridge.hpp"
#include "../kan/kan_module.hpp"
#include "../adapter/kan_adapter.hpp"
#include <memory>
#include <string>

namespace brain19 {

// =============================================================================
// VALIDATION RESULT
// =============================================================================

struct ValidationResult {
    bool validated;              // true if KAN could model the hypothesis
    EpistemicAssessment assessment;
    RelationshipPattern pattern;
    std::shared_ptr<KANModule> trained_module;  // The trained KAN (if successful)
    std::string explanation;

    // NO default constructor
    ValidationResult() = delete;

    ValidationResult(
        bool valid,
        EpistemicAssessment assess,
        RelationshipPattern pat,
        std::shared_ptr<KANModule> module,
        std::string expl
    ) : validated(valid)
      , assessment(std::move(assess))
      , pattern(pat)
      , trained_module(std::move(module))
      , explanation(std::move(expl))
    {}
};

// =============================================================================
// KAN VALIDATOR
// =============================================================================
//
// Orchestrates the full LLM → KAN validation flow:
// 1. Receive HypothesisProposal
// 2. HypothesisTranslator → KAN training problem
// 3. KANModule::train() 
// 4. EpistemicBridge → Trust score
// 5. Return ValidationResult
//
class KanValidator {
public:
    struct Config {
        HypothesisTranslator::Config translator_config{};
        EpistemicBridge::Config bridge_config{};
        size_t max_epochs = 1000;
        double convergence_threshold = 1e-6;
        size_t min_data_points = 10;
        size_t default_num_knots = 10;
    };

    KanValidator() : KanValidator(Config{}) {}
    explicit KanValidator(Config config);

    // Validate a hypothesis proposal end-to-end
    ValidationResult validate(const HypothesisProposal& proposal) const;

    // Access sub-components
    const HypothesisTranslator& translator() const { return translator_; }
    const EpistemicBridge& bridge() const { return bridge_; }
    const Config& get_config() const { return config_; }

private:
    Config config_;
    HypothesisTranslator translator_;
    EpistemicBridge bridge_;
};

} // namespace brain19

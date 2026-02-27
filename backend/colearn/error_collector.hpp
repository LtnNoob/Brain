#pragma once

#include "../common/types.hpp"
#include "../memory/active_relation.hpp"
#include "../graph_net/epistemic_trace.hpp"
#include "../graph_net/types.hpp"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// ERROR COLLECTOR — Prediction Error → Corrective Training Samples
// =============================================================================
//
// During wake phase, compares what each model predicted (composite_score,
// nn_quality, kan_quality) with what actually happened (chain terminated,
// quality dropped). Converts prediction errors into corrective training
// samples that teach the model "this edge performed worse/better than
// you expected."
//
// Three error types:
//   1. Terminal Error — step before chain death predicted high quality
//   2. Quality Drop  — step where composite_score dropped significantly
//   3. Success       — steps in high-quality chains (positive reinforcement)
//

enum class CorrectionType { TERMINAL, QUALITY_DROP, SUCCESS };

struct CorrectionSample {
    ConceptId source_concept = 0;
    ConceptId target_concept = 0;
    RelationType relation = RelationType::CUSTOM;
    double predicted_score = 0.0;    // What model produced during reasoning
    double corrected_target = 0.0;   // What target should be
    double sample_weight = 1.0;      // Training importance
    CorrectionType type = CorrectionType::TERMINAL;
};

struct ErrorCorrectionConfig {
    double correction_weight = 2.0;           // Base weight for correction samples
    double quality_drop_threshold = 0.1;      // Min quality drop to trigger correction
    double success_quality_threshold = 0.7;   // Chain quality above this -> success reinforcement
    double success_weight = 0.5;              // Weight for success samples (lower)
    double terminal_penalty_floor = 0.05;     // Minimum target for terminal errors
};

class ErrorCollector {
public:
    explicit ErrorCollector(const ErrorCorrectionConfig& cfg = {});

    // Collect correction samples from a reasoning chain and its signal
    void collect_from_chain(const GraphChain& chain, const ChainSignal& signal);

    // Get corrections for a specific concept (model to retrain)
    const std::vector<CorrectionSample>& get_corrections(ConceptId cid) const;

    // Clear all collected corrections (call after train phase)
    void clear();

    // Stats
    size_t total_corrections() const;
    size_t terminal_count() const { return terminal_count_; }
    size_t quality_drop_count() const { return drop_count_; }
    size_t success_count() const { return success_count_; }

private:
    ErrorCorrectionConfig config_;
    std::unordered_map<ConceptId, std::vector<CorrectionSample>> corrections_;
    size_t terminal_count_ = 0;
    size_t drop_count_ = 0;
    size_t success_count_ = 0;
};

} // namespace brain19

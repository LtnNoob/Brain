#include "error_collector.hpp"

#include <algorithm>
#include <cmath>

namespace brain19 {

ErrorCollector::ErrorCollector(const ErrorCorrectionConfig& cfg)
    : config_(cfg)
{
}

// =============================================================================
// collect_from_chain — Extract corrective training samples from a chain
// =============================================================================
//
// Three error types:
//
// 1. Terminal Error: last step's source predicted high quality, chain died.
//    Penalty target depends on termination reason (dead end = harsh).
//
// 2. Quality Drop: composite_score dropped significantly vs previous step.
//    Correction target = actual composite_score at that step.
//
// 3. Success Reinforcement: steps in high-quality chains that ran to max steps.
//    Reinforce what worked (lower weight to avoid overwhelming error corrections).
//

void ErrorCollector::collect_from_chain(const GraphChain& chain,
                                         const ChainSignal& signal) {
    if (chain.steps.size() < 2) return;

    // --- 1. Terminal Error ---
    // Chain died with a failure reason AND has >= 2 steps
    bool is_failure =
        chain.termination == TerminationReason::NO_VIABLE_CANDIDATES ||
        chain.termination == TerminationReason::TRUST_TOO_LOW ||
        chain.termination == TerminationReason::ACTIVATION_DECAY ||
        chain.termination == TerminationReason::COHERENCE_GATE ||
        chain.termination == TerminationReason::SEED_DRIFT;

    if (is_failure) {
        // Penalty target depends on how bad the termination was
        double penalty_target = 0.25; // default
        switch (chain.termination) {
            case TerminationReason::NO_VIABLE_CANDIDATES:
                penalty_target = config_.terminal_penalty_floor; // 0.05
                break;
            case TerminationReason::TRUST_TOO_LOW:
                penalty_target = 0.10;
                break;
            case TerminationReason::ACTIVATION_DECAY:
                penalty_target = 0.15;
                break;
            case TerminationReason::COHERENCE_GATE:
                penalty_target = 0.20;
                break;
            case TerminationReason::SEED_DRIFT:
                penalty_target = 0.25;
                break;
            default:
                break;
        }

        // Last step in chain — the one that led to termination
        const auto& last_step = chain.steps.back();
        double predicted = last_step.composite_score;

        // Only correct if prediction was significantly above the penalty
        if (predicted > penalty_target + 0.1) {
            double error_magnitude = predicted - penalty_target;

            CorrectionSample sample;
            sample.source_concept = last_step.source_id;
            sample.target_concept = last_step.target_id;
            sample.relation = last_step.relation;
            sample.predicted_score = predicted;
            sample.corrected_target = penalty_target;
            sample.sample_weight = config_.correction_weight * error_magnitude;
            sample.type = CorrectionType::TERMINAL;

            corrections_[sample.source_concept].push_back(sample);
            ++terminal_count_;
        }
    }

    // --- 2. Quality Drop ---
    // Step where composite_score dropped significantly vs previous step
    for (size_t i = 2; i < chain.steps.size(); ++i) {
        double curr_score = chain.steps[i].composite_score;
        double prev_score = chain.steps[i - 1].composite_score;
        double drop = prev_score - curr_score;

        if (drop >= config_.quality_drop_threshold) {
            CorrectionSample sample;
            sample.source_concept = chain.steps[i].source_id;
            sample.target_concept = chain.steps[i].target_id;
            sample.relation = chain.steps[i].relation;
            sample.predicted_score = prev_score; // model expected continuation quality
            sample.corrected_target = curr_score; // actual quality at this step
            sample.sample_weight = config_.correction_weight * drop;
            sample.type = CorrectionType::QUALITY_DROP;

            corrections_[sample.source_concept].push_back(sample);
            ++drop_count_;
        }
    }

    // --- 3. Success Reinforcement ---
    // High-quality chain that reached max steps — reinforce what worked
    if (signal.chain_quality >= config_.success_quality_threshold &&
        chain.termination == TerminationReason::MAX_STEPS_REACHED) {

        for (size_t i = 1; i < chain.steps.size(); ++i) {
            const auto& step = chain.steps[i];

            CorrectionSample sample;
            sample.source_concept = step.source_id;
            sample.target_concept = step.target_id;
            sample.relation = step.relation;
            sample.predicted_score = step.composite_score;
            sample.corrected_target = step.composite_score; // reinforce actual
            sample.sample_weight = config_.success_weight;
            sample.type = CorrectionType::SUCCESS;

            corrections_[sample.source_concept].push_back(sample);
            ++success_count_;
        }
    }
}

const std::vector<CorrectionSample>& ErrorCollector::get_corrections(ConceptId cid) const {
    static const std::vector<CorrectionSample> empty;
    auto it = corrections_.find(cid);
    return it != corrections_.end() ? it->second : empty;
}

void ErrorCollector::clear() {
    corrections_.clear();
    terminal_count_ = 0;
    drop_count_ = 0;
    success_count_ = 0;
}

size_t ErrorCollector::total_corrections() const {
    size_t total = 0;
    for (const auto& [cid, samples] : corrections_) {
        total += samples.size();
    }
    return total;
}

} // namespace brain19

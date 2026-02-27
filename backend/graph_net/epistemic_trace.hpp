#pragma once

#include "types.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include "../convergence/convergence_config.hpp"
#include "../ltm/long_term_memory.hpp"

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace brain19 {

// =============================================================================
// EPISTEMIC TRACE --- Full Audit Trail for Graph Reasoning
// =============================================================================
//
// Every step is fully documented: input/output activations, epistemic metadata,
// why this path was chosen, which alternatives were rejected, and how trust
// changes along the chain.
//
// OOP Design:
//   TraceAlternative: rejected candidate with rejection reason
//   TraceStep: full record of one reasoning step
//   GraphChain: complete chain with audit trail and explain()
//
// Extension point for future Memory system:
//   TraceStep stores source/target epistemic metadata --- when memories are added,
//   a MemoryTraceStep subclass can add memory references and memory-based trust.
//

// =============================================================================
// TraceAlternative --- A rejected candidate with explanation
// =============================================================================

struct TraceAlternative {
    ConceptId target_id = 0;
    RelationType relation = RelationType::CUSTOM;
    double composite_score = 0.0;
    std::string rejection_reason;  // "INVALIDATED target", "below focus gate", etc.
};

// =============================================================================
// TraceStep --- Complete record of one reasoning step
// =============================================================================

class TraceStep {
public:
    // --- Identity ---
    ConceptId source_id = 0;
    ConceptId target_id = 0;
    RelationType relation = RelationType::CUSTOM;
    bool is_outgoing = true;
    bool focus_shifted = false;
    size_t step_index = 0;

    // --- Activations (full vectors, NOT scalars) ---
    Activation input_activation;
    Activation output_activation;
    CoreVec dimensional_contribution{};  // v = W*c + b

    // --- Epistemic metadata: snapshots of source AND target ---
    EpistemicType source_epistemic_type = EpistemicType::HYPOTHESIS;
    EpistemicType target_epistemic_type = EpistemicType::HYPOTHESIS;
    EpistemicStatus source_epistemic_status = EpistemicStatus::ACTIVE;
    EpistemicStatus target_epistemic_status = EpistemicStatus::ACTIVE;
    double source_trust = 0.5;
    double target_trust = 0.5;

    // --- Edge quality ---
    double edge_confidence = 0.0;       // From ConceptModel convergence quality
    double transform_quality = 0.0;     // Magnitude preservation
    double coherence = 0.0;             // Activation -> target alignment
    double epistemic_alignment = 0.0;   // Trust compatibility source -> target

    // --- Dual neuron quality ---
    double nn_quality = 0.0;            // NN path: W*x+b transform quality
    double kan_quality = 0.0;           // KAN path: FlexKAN gate value (reasoning approval)
    double kan_gate = 1.0;              // Effective gating factor applied

    // --- Step trust (weighted combination) ---
    double step_trust = 0.0;

    // --- ChainKAN state (from ConceptReasoner composition) ---
    std::array<double, convergence::OUTPUT_DIM> chain_state{};
    double chain_coherence = 0.0;
    double seed_similarity = 0.0;

    // --- Top contributing dimensions ---
    std::vector<size_t> top_dims;
    std::vector<double> top_dim_values;

    // --- Alternatives with rejection reasons ---
    std::vector<TraceAlternative> alternatives;

    // --- Composite score for this step ---
    double composite_score = 0.0;

    // Compute step trust from component values
    void compute_step_trust(double src_w, double edge_w,
                            double tgt_w, double transform_w) {
        step_trust = src_w * source_trust
                   + edge_w * edge_confidence
                   + tgt_w * target_trust
                   + transform_w * transform_quality;
    }
};

// =============================================================================
// GraphChain --- Kette mit vollstaendiger Audit-Trail
// =============================================================================
//
// Stores the complete reasoning chain with all trace steps, chain-level metrics,
// and human-readable explain() output.
//
// Interface compatible with ReasoningChain: provides concept_sequence() and
// relation_sequence() so KANLanguageEngine can use it as a drop-in replacement.
//

class GraphChain {
public:
    // --- Steps ---
    std::vector<TraceStep> steps;

    // --- Chain-level metrics ---
    double chain_trust = 0.0;             // Geometric mean of step_trusts
    EpistemicType chain_epistemic_type = EpistemicType::FACT; // Weakest link
    TerminationReason termination = TerminationReason::STILL_RUNNING;

    // --- Activation trace ---
    Activation initial_activation;
    Activation final_activation;
    double magnitude_ratio = 0.0;        // |final| / |initial|

    // --- Compatibility with ReasoningChain ---
    double avg_confidence = 0.0;

    // Concept sequence (for template engine)
    std::vector<ConceptId> concept_sequence() const {
        std::vector<ConceptId> seq;
        seq.reserve(steps.size());
        for (const auto& s : steps)
            seq.push_back(s.step_index == 0 ? s.source_id : s.target_id);
        return seq;
    }

    // Relation sequence (for template engine)
    std::vector<RelationType> relation_sequence() const {
        std::vector<RelationType> seq;
        if (steps.size() <= 1) return seq;
        seq.reserve(steps.size() - 1);
        for (size_t i = 1; i < steps.size(); ++i)
            seq.push_back(steps[i].relation);
        return seq;
    }

    bool empty() const { return steps.empty(); }
    size_t length() const { return steps.empty() ? 0 : steps.size() - 1; }

    // Compute chain-level metrics from steps
    void compute_chain_metrics() {
        if (steps.empty()) return;

        // Chain trust: geometric mean of step_trusts (exclude seed)
        if (steps.size() > 1) {
            double log_sum = 0.0;
            size_t count = 0;
            for (size_t i = 1; i < steps.size(); ++i) {
                double t = std::max(1e-10, steps[i].step_trust);
                log_sum += std::log(t);
                ++count;
            }
            chain_trust = count > 0 ? std::exp(log_sum / static_cast<double>(count)) : 0.0;
        }

        // Chain epistemic type: weakest link (highest enum value = weakest)
        chain_epistemic_type = EpistemicType::FACT;
        for (const auto& s : steps) {
            auto types = {s.source_epistemic_type, s.target_epistemic_type};
            for (auto t : types) {
                if (static_cast<int>(t) > static_cast<int>(chain_epistemic_type))
                    chain_epistemic_type = t;
            }
        }

        // Magnitude ratio
        if (!steps.empty()) {
            initial_activation = steps[0].input_activation;
            final_activation = steps.back().output_activation;
            double init_mag = initial_activation.core_magnitude();
            double final_mag = final_activation.core_magnitude();
            magnitude_ratio = init_mag > 1e-12 ? final_mag / init_mag : 0.0;
        }

        // Average confidence (compat with ReasoningChain)
        if (steps.size() > 1) {
            double sum = 0.0;
            for (size_t i = 1; i < steps.size(); ++i)
                sum += steps[i].composite_score;
            avg_confidence = sum / static_cast<double>(steps.size() - 1);
        }
    }

    // Human-readable audit trail
    std::string explain(const LongTermMemory& ltm) const {
        std::ostringstream os;

        os << "=== Graph Reasoning Chain ===\n";
        os << "Length: " << length() << " steps\n";
        os << "Chain trust: " << chain_trust << "\n";
        os << "Chain epistemic type: "
           << epistemic_type_to_string(chain_epistemic_type) << "\n";
        os << "Termination: " << termination_reason_to_string(termination) << "\n";
        os << "Magnitude ratio: " << magnitude_ratio << "\n";

        if (steps.empty()) {
            os << "(empty chain)\n";
            return os.str();
        }

        // Seed info
        auto seed_info = ltm.retrieve_concept(steps[0].source_id);
        os << "\nSEED: " << (seed_info ? seed_info->label : "?")
           << " [" << steps[0].source_id << "]\n";
        os << "  Trust: " << steps[0].source_trust
           << " (" << epistemic_type_to_string(steps[0].source_epistemic_type) << ")\n";
        os << "  Activation magnitude: " << steps[0].input_activation.core_magnitude() << "\n";

        // Each step
        for (size_t i = 1; i < steps.size(); ++i) {
            const auto& step = steps[i];

            auto src_info = ltm.retrieve_concept(step.source_id);
            auto tgt_info = ltm.retrieve_concept(step.target_id);
            std::string src_label = src_info ? src_info->label : "?";
            std::string tgt_label = tgt_info ? tgt_info->label : "?";

            os << "\nStep " << i << ": " << src_label << " [" << step.source_id << "]"
               << " --" << relation_type_to_string(step.relation) << "--> "
               << tgt_label << " [" << step.target_id << "]\n";

            os << "  Transform quality: " << step.transform_quality
               << " | Coherence: " << step.coherence
               << " | Score: " << step.composite_score << "\n";

            os << "  Source trust: " << step.source_trust
               << " (" << epistemic_type_to_string(step.source_epistemic_type) << ")"
               << " -> Target trust: " << step.target_trust
               << " (" << epistemic_type_to_string(step.target_epistemic_type) << ")\n";

            os << "  Step trust: " << step.step_trust
               << " | Edge confidence: " << step.edge_confidence
               << " | Epistemic alignment: " << step.epistemic_alignment << "\n";

            os << "  Dual neurons: NN=" << step.nn_quality
               << " KAN=" << step.kan_quality
               << " gate=" << step.kan_gate << "\n";

            double in_mag = step.input_activation.core_magnitude();
            double out_mag = step.output_activation.core_magnitude();
            os << "  Activation: |in|=" << in_mag
               << " -> |out|=" << out_mag << "\n";

            // Top dimensions
            if (!step.top_dims.empty()) {
                os << "  Top dims: [";
                for (size_t j = 0; j < step.top_dims.size(); ++j) {
                    if (j > 0) os << ", ";
                    os << "d" << step.top_dims[j] << "="
                       << step.top_dim_values[j];
                }
                os << "]\n";
            }

            // Alternatives
            if (!step.alternatives.empty()) {
                os << "  Alternatives considered (" << step.alternatives.size() << " total):\n";
                size_t show = std::min(step.alternatives.size(), size_t(3));
                for (size_t j = 0; j < show; ++j) {
                    const auto& alt = step.alternatives[j];
                    auto alt_info = ltm.retrieve_concept(alt.target_id);
                    std::string alt_label = alt_info ? alt_info->label : "?";
                    os << "    - " << alt_label << " [" << alt.target_id << "]"
                       << " via " << relation_type_to_string(alt.relation)
                       << " (score=" << alt.composite_score << ")"
                       << " reason: " << alt.rejection_reason << "\n";
                }
                if (step.alternatives.size() > show) {
                    os << "    ... and " << (step.alternatives.size() - show) << " more\n";
                }
            }
        }

        return os.str();
    }
};

} // namespace brain19

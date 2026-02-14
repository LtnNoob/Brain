#pragma once

#include "investigation_request.hpp"
#include "kan_validator.hpp"
#include <vector>
#include <cstddef>
#include <algorithm>
#include <numeric>

namespace brain19 {

// =============================================================================
// TOPOLOGY MODE
// =============================================================================

enum class TopologyMode {
    B_ONLY,    // Standard: LLM generates, KAN validates
    A_AND_B,   // + KAN anomaly detection → LLM investigation
    B_AND_C,   // + Refinement loop for partial validations
    ALL        // All three topologies active
};

inline const char* topology_mode_to_string(TopologyMode m) {
    switch (m) {
        case TopologyMode::B_ONLY:  return "B_ONLY";
        case TopologyMode::A_AND_B: return "A_AND_B";
        case TopologyMode::B_AND_C: return "B_AND_C";
        case TopologyMode::ALL:     return "ALL";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// ROUTING DECISION
// =============================================================================

struct RoutingDecision {
    TopologyMode mode;
    std::vector<InvestigationRequest> investigations;  // For Topology A
    std::vector<size_t> refine_indices;                // For Topology C (indices into validated_hypotheses)
};

// =============================================================================
// TOPOLOGY ROUTER
// =============================================================================
//
// Stateless decision maker — examines anomalies + validation results
// and decides which topologies to activate.
//
// "Tool" in Brain19 philosophy: no learning, no state, pure function.
//
class TopologyRouter {
public:
    struct Config {
        size_t min_anomalies_for_a = 2;
        double min_avg_anomaly_strength = 0.3;
        double max_mse_for_c = 0.5;   // Only refine if initial MSE <= this
        size_t max_investigations = 5;
        size_t max_refinements = 3;
    };

    TopologyRouter() : config_() {}
    explicit TopologyRouter(Config config) : config_(config) {}

    RoutingDecision route(
        const std::vector<InvestigationRequest>& anomalies,
        const std::vector<ValidationResult>& validations) const
    {
        RoutingDecision decision;
        decision.mode = TopologyMode::B_ONLY;

        bool activate_a = should_activate_a(anomalies);
        bool activate_c = should_activate_c(validations, decision.refine_indices);

        if (activate_a && activate_c) {
            decision.mode = TopologyMode::ALL;
        } else if (activate_a) {
            decision.mode = TopologyMode::A_AND_B;
        } else if (activate_c) {
            decision.mode = TopologyMode::B_AND_C;
        }

        // Select top investigations for Topology A
        if (activate_a) {
            size_t count = std::min(anomalies.size(), config_.max_investigations);
            decision.investigations.assign(anomalies.begin(), anomalies.begin() + count);
        }

        return decision;
    }

    const Config& get_config() const { return config_; }

private:
    Config config_;

    // A activates when enough anomalies with sufficient average strength
    bool should_activate_a(const std::vector<InvestigationRequest>& anomalies) const {
        if (anomalies.size() < config_.min_anomalies_for_a) return false;

        double avg_strength = 0.0;
        for (const auto& a : anomalies) {
            avg_strength += a.anomaly_strength;
        }
        avg_strength /= static_cast<double>(anomalies.size());

        return avg_strength >= config_.min_avg_anomaly_strength;
    }

    // C activates when any validation has hope for refinement:
    // - Not validated (failed KAN check)
    // - But MSE <= threshold (close enough to refine)
    // - And pattern != NOT_QUANTIFIABLE (there's a translatable pattern)
    bool should_activate_c(
        const std::vector<ValidationResult>& validations,
        std::vector<size_t>& refine_indices) const
    {
        refine_indices.clear();

        for (size_t i = 0; i < validations.size(); ++i) {
            const auto& vr = validations[i];

            // Only refine failed validations that have hope
            if (vr.validated) continue;
            if (vr.pattern == RelationshipPattern::NOT_QUANTIFIABLE) continue;

            // Check if MSE is close enough to refine
            // MSE is encoded in the assessment
            double mse = vr.assessment.mse;
            if (mse <= config_.max_mse_for_c) {
                refine_indices.push_back(i);
                if (refine_indices.size() >= config_.max_refinements) break;
            }
        }

        return !refine_indices.empty();
    }
};

} // namespace brain19

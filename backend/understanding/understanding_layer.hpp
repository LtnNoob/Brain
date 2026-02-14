#pragma once

#include "mini_llm.hpp"
#include "understanding_proposals.hpp"
#include "../cognitive/cognitive_dynamics.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/stm.hpp"
#include <memory>
#include <vector>
#include <unordered_map>

namespace brain19 {

// =============================================================================
// UNDERSTANDING LAYER CONFIGURATION
// =============================================================================

struct UnderstandingLayerConfig {
    // Mini-LLM parallelism
    bool enable_parallel_llms = false;  // Conservative default

    // Proposal thresholds
    double min_meaning_confidence = 0.3;
    double min_hypothesis_confidence = 0.2;
    double min_analogy_confidence = 0.4;
    double min_contradiction_severity = 0.5;

    // Rate limiting
    size_t max_proposals_per_cycle = 10;

    // Logging
    bool verbose_logging = false;
};

// =============================================================================
// UNDERSTANDING LAYER
// =============================================================================
//
// UnderstandingLayer: Semantische Analyse-Schicht über Cognitive Dynamics
//
// ARCHITEKTUR-VERTRAG:
// ✅ Verwendet Cognitive Dynamics (Spreading Activation, Salience, etc.)
// ✅ Generiert semantische Vorschläge via Mini-LLMs
// ✅ Alle Outputs sind HYPOTHESIS
// ✅ READ-ONLY Zugriff auf LTM
// ✅ Kein epistemischer Schreibzugriff
//
// ❌ DARF NICHT:
// - Knowledge Graph modifizieren
// - Trust-Werte setzen
// - Epistemische Entscheidungen treffen
// - FACT-Promotion durchführen
// - Regeln generieren
// - Autonome Aktionen
//
// INTEGRATION:
// - Wird vom BrainController aufgerufen
// - Nutzt Cognitive Dynamics für Fokus/Salience
// - Gibt Proposals an BrainController zurück
// - BrainController → Epistemic Core entscheidet über Akzeptanz
//
// ENFORCEMENT:
// - Alle Mini-LLMs haben READ-ONLY LTM-Zugriff
// - Alle Proposals sind HYPOTHESIS
// - Vollständiges Logging
// - Determinismus (optional parallele LLMs, aber deterministische Aggregation)
//
class UnderstandingLayer {
public:
    explicit UnderstandingLayer(
        UnderstandingLayerConfig config = UnderstandingLayerConfig()
    );
    ~UnderstandingLayer();

    // No copy (owns Mini-LLMs)
    UnderstandingLayer(const UnderstandingLayer&) = delete;
    UnderstandingLayer& operator=(const UnderstandingLayer&) = delete;

    // Move allowed
    UnderstandingLayer(UnderstandingLayer&&) = default;
    UnderstandingLayer& operator=(UnderstandingLayer&&) = default;

    // =========================================================================
    // MINI-LLM MANAGEMENT
    // =========================================================================

    // Register a Mini-LLM (takes ownership)
    void register_mini_llm(std::unique_ptr<MiniLLM> mini_llm);

    // Get number of registered Mini-LLMs
    size_t get_mini_llm_count() const { return mini_llms_.size(); }

    // Visit all registered MiniLLMs (for KAN feedback)
    template<typename Fn>
    void for_each_mini_llm(Fn&& fn) {
        for (auto& llm : mini_llms_) { fn(*llm); }
    }

    // =========================================================================
    // SEMANTIC ANALYSIS (READ-ONLY)
    // =========================================================================

    // Analyze activated concepts and generate meaning proposals
    // INPUT: Active concepts from STM (via Cognitive Dynamics)
    // OUTPUT: Meaning proposals (ALL HYPOTHESIS)
    std::vector<MeaningProposal> analyze_meaning(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    );

    // Generate hypothesis proposals based on evidence
    // INPUT: Evidence concepts (READ-ONLY), optional ThoughtPaths for multi-hop reasoning
    // OUTPUT: Hypothesis proposals (NOT accepted hypotheses!)
    std::vector<HypothesisProposal> propose_hypotheses(
        const std::vector<ConceptId>& evidence_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context,
        const std::vector<ThoughtPath>& thought_paths = {}
    );

    // Detect structural analogies
    // INPUT: Concept sets (READ-ONLY)
    // OUTPUT: Analogy proposals
    std::vector<AnalogyProposal> find_analogies(
        const std::vector<ConceptId>& concept_set_a,
        const std::vector<ConceptId>& concept_set_b,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    );

    // Detect potential contradictions
    // INPUT: Active concepts (READ-ONLY)
    // OUTPUT: Contradiction proposals (NOT resolutions!)
    std::vector<ContradictionProposal> check_contradictions(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    );

    // =========================================================================
    // INTEGRATED ANALYSIS (using Cognitive Dynamics)
    // =========================================================================

    // Perform full understanding cycle:
    // 1. Use Cognitive Dynamics for salience/focus
    // 2. Apply Mini-LLMs to salient concepts
    // 3. Generate proposals
    // OUTPUT: All proposal types, filtered by confidence thresholds
    struct UnderstandingResult {
        std::vector<MeaningProposal> meaning_proposals;
        std::vector<HypothesisProposal> hypothesis_proposals;
        std::vector<AnalogyProposal> analogy_proposals;
        std::vector<ContradictionProposal> contradiction_proposals;

        // Statistics
        size_t total_proposals_generated = 0;
        size_t proposals_filtered_by_threshold = 0;
    };

    UnderstandingResult perform_understanding_cycle(
        ConceptId seed_concept,
        CognitiveDynamics& cognitive_dynamics,
        const LongTermMemory& ltm,  // READ-ONLY!
        ShortTermMemory& stm,  // For Cognitive Dynamics activation
        ContextId context
    );

    // =========================================================================
    // STATISTICS & INTROSPECTION
    // =========================================================================

    struct Statistics {
        size_t total_meaning_proposals = 0;
        size_t total_hypothesis_proposals = 0;
        size_t total_analogy_proposals = 0;
        size_t total_contradiction_proposals = 0;
        size_t total_cycles_performed = 0;
    };

    const Statistics& get_statistics() const { return stats_; }
    void reset_statistics() { stats_ = Statistics(); }

private:
    UnderstandingLayerConfig config_;
    std::vector<std::unique_ptr<MiniLLM>> mini_llms_;
    Statistics stats_;

    // Internal helpers
    template<typename ProposalType>
    std::vector<ProposalType> filter_proposals_by_confidence(
        const std::vector<ProposalType>& proposals,
        double min_confidence
    ) const;

    void log_message(const std::string& message) const;
};

} // namespace brain19

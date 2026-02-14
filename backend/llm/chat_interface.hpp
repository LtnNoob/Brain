#pragma once

#include "../ltm/long_term_memory.hpp"
#include "../memory/brain_controller.hpp"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace brain19 {

// Intent classification for user queries
enum class QueryIntent {
    GREETING,    // "hey", "hi", "hello"
    QUESTION,    // "What is X?", "How does Y work?"
    COMMAND,     // "list", "show", "explain"
    STATEMENT,   // Declarative input
    UNKNOWN
};

// ChatResponse: Response from Brain19
struct ChatResponse {
    std::string answer;
    std::vector<ConceptId> referenced_concepts;
    bool contains_speculation = false;
    std::string epistemic_note;
    bool used_llm = false;
    double llm_time_ms = 0.0;
    QueryIntent intent = QueryIntent::UNKNOWN;
};

// ThinkingContext: Full cognitive pipeline output for response generation
//
// Populated by SystemOrchestrator from ThinkingResult.
// Contains pre-processed insights from ALL pipeline stages —
// no raw proposal types needed (decoupled from understanding_proposals.hpp).
struct ThinkingContext {
    // Salient concepts from cognitive pipeline
    std::vector<ConceptId> salient_concepts;
    std::vector<std::string> thought_path_summaries;

    // Domain detection: which knowledge domains were activated
    struct DomainInsight {
        std::string domain_name;
        std::vector<ConceptId> concepts;
        double relevance;  // [0,1] — based on seed scores
    };
    std::vector<DomainInsight> detected_domains;

    // Meaning insights from ConceptModels (Understanding Layer Step 8)
    struct MeaningInsight {
        std::string interpretation;
        double confidence;
        std::string source_model;
        std::vector<ConceptId> source_concepts;
    };
    std::vector<MeaningInsight> meaning_insights;

    // Hypothesis insights from ConceptModels + KAN validation (Steps 8-9)
    struct HypothesisInsight {
        std::string statement;
        double confidence;
        std::string source_model;
        bool kan_validated = false;
        std::string validation_status;  // "validated", "refuted", "inconclusive", ""
    };
    std::vector<HypothesisInsight> hypothesis_insights;

    // KAN-Relations between salient concepts (graph structure)
    struct RelationLink {
        ConceptId source;
        ConceptId target;
        std::string relation_name;  // e.g., "IS_A", "CAUSES"
        double weight;
        std::string source_label;
        std::string target_label;
    };
    std::vector<RelationLink> relation_links;

    // Contradiction alerts from ConceptModels
    struct ContradictionNote {
        ConceptId concept_a;
        ConceptId concept_b;
        std::string description;
        double severity;
    };
    std::vector<ContradictionNote> contradiction_notes;

    // Autonomous thinking insights (from GDO background thinking)
    struct AutonomousInsight {
        std::vector<ConceptId> seed_concepts;
        std::vector<std::string> discovered_labels;
        size_t proposals_generated = 0;
        double duration_ms = 0.0;
    };
    std::vector<AutonomousInsight> autonomous_insights;

    // Embedding-similar concepts discovered (Strategy 7)
    struct EmbeddingSeed {
        ConceptId concept_id;
        ConceptId similar_to;  // which text-matched seed it's similar to
        double similarity;
        std::string label;
    };
    std::vector<EmbeddingSeed> embedding_discoveries;

    // Pipeline statistics
    size_t steps_completed = 0;
    double thinking_duration_ms = 0.0;
    size_t total_proposals = 0;
};

// ChatInterface: Knowledge-based verbalization of Brain19 knowledge
//
// Uses LTM knowledge directly — no external LLM.
// Template-Engine (cursor/template_engine.hpp) handles sentence generation.
class ChatInterface {
public:
    ChatInterface();
    ~ChatInterface();

    // LLM is not available — knowledge-only mode
    bool is_llm_available() const;

    // Ask a question — intent-aware formatting
    ChatResponse ask(
        const std::string& question,
        const LongTermMemory& ltm
    );

    // Ask with thinking context (salient concepts from ThinkingPipeline)
    ChatResponse ask_with_context(
        const std::string& question,
        const LongTermMemory& ltm,
        const std::vector<ConceptId>& salient_concepts,
        const std::vector<std::string>& thought_paths_summary = {},
        QueryIntent intent = QueryIntent::UNKNOWN
    );

    // Ask with full cognitive pipeline output (ThinkingContext)
    // Routes: Input → KAN-Relations + Pattern Matching → Topic Detection →
    //         Generative Thinking → Multi-ConceptModel Orchestration → Output Fusion
    ChatResponse ask_with_thinking(
        const std::string& question,
        const LongTermMemory& ltm,
        const ThinkingContext& thinking,
        QueryIntent intent = QueryIntent::UNKNOWN
    );

    // List all knowledge of specific type
    std::string list_knowledge(
        const LongTermMemory& ltm,
        EpistemicType type
    );

    // Explain a concept
    std::string explain_concept(
        ConceptId id,
        const LongTermMemory& ltm
    );

    // Compare two concepts
    std::string compare(
        ConceptId id1,
        ConceptId id2,
        const LongTermMemory& ltm
    );

    // Get knowledge summary
    std::string get_summary(const LongTermMemory& ltm);

    // Intent classification
    static QueryIntent classify_intent(const std::string& question);

    // Set total counts for greeting
    void set_totals(size_t concepts, size_t relations) {
        total_concepts_ = concepts;
        total_relations_ = relations;
    }

private:
    size_t total_concepts_ = 0;
    size_t total_relations_ = 0;

    // Build epistemic context
    std::string build_epistemic_context(
        const std::vector<ConceptInfo>& concepts
    );

    // Search for relevant concepts
    std::vector<ConceptInfo> find_relevant_concepts(
        const std::string& question,
        const LongTermMemory& ltm
    );

    // Intent-aware response formatting
    std::string format_greeting(const std::vector<ConceptInfo>& top_concepts);
    std::string format_question(const std::vector<ConceptInfo>& top_concepts,
                                const std::vector<std::string>& thought_paths);
    std::string format_statement(const std::vector<ConceptInfo>& top_concepts);

    // Full-pipeline response formatting (multi-domain, fusion)
    std::string format_thinking_response(
        const std::vector<ConceptInfo>& top_concepts,
        const ThinkingContext& thinking,
        const LongTermMemory& ltm
    );
};

} // namespace brain19

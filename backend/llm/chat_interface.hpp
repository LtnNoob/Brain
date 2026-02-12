#pragma once

#include "../ltm/long_term_memory.hpp"
#include "../memory/brain_controller.hpp"
#include <string>
#include <vector>
#include <memory>

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

private:
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
};

} // namespace brain19

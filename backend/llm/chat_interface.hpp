#pragma once

#include "../ltm/long_term_memory.hpp"
#include "../memory/brain_controller.hpp"
#include <string>
#include <vector>
#include <memory>

namespace brain19 {

// ChatResponse: Response from Brain19
struct ChatResponse {
    std::string answer;
    std::vector<ConceptId> referenced_concepts;
    bool contains_speculation = false;
    std::string epistemic_note;
    bool used_llm = false;
    double llm_time_ms = 0.0;
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

    // Ask a question
    ChatResponse ask(
        const std::string& question,
        const LongTermMemory& ltm
    );

    // Ask with thinking context (salient concepts from ThinkingPipeline)
    ChatResponse ask_with_context(
        const std::string& question,
        const LongTermMemory& ltm,
        const std::vector<ConceptId>& salient_concepts,
        const std::vector<std::string>& thought_paths_summary = {}
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
};

} // namespace brain19
